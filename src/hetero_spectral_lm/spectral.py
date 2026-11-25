"""Static-shape per-sequence intra-document causal Spectral-Reservoir mixer in JAX/Flax (0.00e+00 leak)."""

from __future__ import annotations

import functools
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .analog import AnalogLinear, AnalogNoiseConfig, adc_quantize_ste, deterministic_chip_key
from .packing import StaticDocPlan

TOEPLITZ_MAX_SEG_LEN = 256


@functools.lru_cache(maxsize=16)
def compute_hankel_spectral_filters(seq_len: int, num_filters: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact positive & alternating Hankel spectral filters in FP64 -> FP32 (cached across layers).

    Matrix entry: Z[i, j] = 2 / ((i + j + 2)^3 - (i + j + 2)) for 0 <= i, j < seq_len.
    Eigenvectors are scaled by sigma_k^(1/4); alternating branch multiplies by (-1)^t.
    """
    idx = np.arange(1, seq_len + 1, dtype=np.float64)
    s = idx[:, None] + idx[None, :]
    z = 2.0 / (s**3 - s)
    eigvals, eigvecs = np.linalg.eigh(z)
    top_vals = np.maximum(eigvals[-num_filters:][::-1], 1e-12)
    top_vecs = eigvecs[:, -num_filters:][:, ::-1].T  # [K, L]
    # Canonicalize eigenvector signs for LAPACK/NumPy/PyTorch portability
    max_idx = np.argmax(np.abs(top_vecs), axis=1)
    signs = np.where(top_vecs[np.arange(num_filters), max_idx] < 0.0, -1.0, 1.0)
    top_vecs = top_vecs * signs[:, None]
    phi_pos = (top_vecs * (top_vals**0.25)[:, None]).astype(np.float32)
    alt = np.where((np.arange(seq_len) % 2) == 0, 1.0, -1.0).astype(np.float32)
    phi_neg = phi_pos * alt[None, :]
    phi_pos.setflags(write=False)
    phi_neg.setflags(write=False)
    return phi_pos, phi_neg


def _next_pow2(n: int) -> int:
    return 1 << (2 * n - 1).bit_length()


@functools.lru_cache(maxsize=32)
def compute_hankel_filter_ffts(
    max_seq_len: int,
    seg_len: int,
    num_filters: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute static Hankel filter rffts in complex64 once on host (zero per-layer rfft(phi) in XLA)."""
    phi_pos, phi_neg = compute_hankel_spectral_filters(max_seq_len, num_filters=num_filters)
    n_fft = _next_pow2(seg_len)
    fp = np.fft.rfft(phi_pos[:, :seg_len].astype(np.float32), n=n_fft, axis=1).astype(np.complex64)
    fn = np.fft.rfft(phi_neg[:, :seg_len].astype(np.float32), n=n_fft, axis=1).astype(np.complex64)
    fp.setflags(write=False)
    fn.setflags(write=False)
    return fp, fn


@functools.lru_cache(maxsize=16)
def compute_hankel_causal_toeplitz(
    max_seq_len: int,
    seg_len: int,
    num_filters: int = 16,
) -> np.ndarray:
    """Precompute exact lower-triangular causal Toeplitz tensor [K, S, S] for MXU/Tensor-Core short buckets."""
    phi_pos, _ = compute_hankel_spectral_filters(max_seq_len, num_filters=num_filters)
    diff = np.arange(seg_len)[:, None] - np.arange(seg_len)[None, :]
    toep = np.where((diff >= 0)[None, :, :], phi_pos[:, np.clip(diff, 0, seg_len - 1)], 0.0).astype(np.float32)
    toep.setflags(write=False)
    return toep


def _contracted_fft_conv_1d(
    packed: jax.Array,
    phi_pos: jax.Array,
    phi_neg: jax.Array,
    w_pos: jax.Array,
    w_neg: jax.Array,
    *,
    fp_precomp: jax.Array | None = None,
    fn_precomp: jax.Array | None = None,
) -> jax.Array:
    """Exact frequency-domain linear contraction in FP32: sum_k w_k * (x * phi_k)."""
    seg_len = packed.shape[1]
    n_fft = _next_pow2(seg_len)
    x_fft = jnp.fft.rfft(packed.astype(jnp.float32), n=n_fft, axis=1)  # [M, F, C]
    fp = fp_precomp if fp_precomp is not None else jnp.fft.rfft(phi_pos[:, :seg_len].astype(jnp.float32), n=n_fft, axis=1)
    fn = fn_precomp if fn_precomp is not None else jnp.fft.rfft(phi_neg[:, :seg_len].astype(jnp.float32), n=n_fft, axis=1)
    wp = w_pos.astype(jnp.float32)
    wn = w_neg.astype(jnp.float32)
    h_real = jnp.matmul(fp.real.T, wp) + jnp.matmul(fn.real.T, wn)  # [F, C]
    h_imag = jnp.matmul(fp.imag.T, wp) + jnp.matmul(fn.imag.T, wn)  # [F, C]
    h_fft = jax.lax.complex(h_real, h_imag)
    y_fft = x_fft * h_fft[None, :, :]
    return jnp.fft.irfft(y_fft, n=n_fft, axis=1)[:, :seg_len, :]


def _causal_multi_conv_1d(
    packed: jax.Array,
    filters: jax.Array,
    *,
    f_fft_precomp: jax.Array | None = None,
    toeplitz_precomp: jax.Array | None = None,
) -> jax.Array:
    """Exact causal 1D convolution of [M, S, C] with [K, L_f] -> [M, S, K, C] in FP32.

    Uses exact causal Toeplitz GEMM when `toeplitz_precomp` is supplied (S <= 256, MXU/Tensor-Core accelerated)
    and precomputed-filter FFT otherwise.
    """
    if toeplitz_precomp is not None:
        return jnp.einsum("kts,msc->mtkc", toeplitz_precomp, packed.astype(jnp.float32))
    seg_len = packed.shape[1]
    n_fft = _next_pow2(seg_len)
    x_fft = jnp.fft.rfft(packed.astype(jnp.float32), n=n_fft, axis=1)  # [M, F, C]
    f_fft = f_fft_precomp if f_fft_precomp is not None else jnp.fft.rfft(filters[:, :seg_len].astype(jnp.float32), n=n_fft, axis=1)
    y_fft = x_fft[:, None, :, :] * f_fft[None, :, :, None]  # [M, K, F, C]
    y = jnp.fft.irfft(y_fft, n=n_fft, axis=2)[:, :, :seg_len, :]  # [M, K, S, C]
    return jnp.transpose(y, (0, 2, 1, 3))


def static_doc_isolated_contracted_fft(
    x: jax.Array,
    phi_pos: jax.Array,
    phi_neg: jax.Array,
    w_pos: jax.Array,
    w_neg: jax.Array,
    doc_plan: StaticDocPlan | None = None,
    *,
    max_seq_len_cache: int | None = None,
) -> jax.Array:
    """100% static-shape per-sequence intra-document contracted FFT [B, L, C] -> [B, L, C] (zero cross-chip collectives)."""
    _, L, C = x.shape
    K = phi_pos.shape[0]
    xf = x.astype(jnp.float32)

    def _precomp(seg_len: int) -> tuple[jax.Array | None, jax.Array | None]:
        if max_seq_len_cache is None:
            return None, None
        fp_np, fn_np = compute_hankel_filter_ffts(max_seq_len_cache, seg_len, num_filters=K)
        return jnp.asarray(fp_np), jnp.asarray(fn_np)

    if doc_plan is None:
        fp_c, fn_c = _precomp(L)
        return _contracted_fft_conv_1d(xf, phi_pos, phi_neg, w_pos, w_neg, fp_precomp=fp_c, fn_precomp=fn_c)

    bucket_precomps = [_precomp(int(bt.shape[2])) for bt in doc_plan.bucket_t]

    def _per_seq(x_row: jax.Array, bt_tuple: tuple[jax.Array, ...], bv_tuple: tuple[jax.Array, ...]) -> jax.Array:
        x_pad = jnp.concatenate([x_row, jnp.zeros((1, C), dtype=jnp.float32)], axis=0)
        out = jnp.zeros((L + 1, C), dtype=jnp.float32)
        for t_idx, valid, (fp_c, fn_c) in zip(bt_tuple, bv_tuple, bucket_precomps):
            mask = valid[..., None].astype(jnp.float32)
            packed = x_pad[t_idx] * mask  # [M_seq, S, C]
            y_seg = _contracted_fft_conv_1d(
                packed, phi_pos, phi_neg, w_pos, w_neg, fp_precomp=fp_c, fn_precomp=fn_c
            ) * mask
            out = out.at[t_idx].add(y_seg)
        return out[:L, :]

    return jax.vmap(_per_seq)(xf, doc_plan.bucket_t, doc_plan.bucket_valid)


def static_doc_isolated_causal_fft(
    x: jax.Array,
    filters: jax.Array,
    doc_plan: StaticDocPlan | None = None,
    *,
    max_seq_len_cache: int | None = None,
    toeplitz_max_len: int = TOEPLITZ_MAX_SEG_LEN,
) -> jax.Array:
    """100% static-shape per-sequence intra-document multi-filter causal conv [B, L, C] -> [B, L, K, C]."""
    _, L, C = x.shape
    K = filters.shape[0]
    xf = x.astype(jnp.float32)

    def _precomp(seg_len: int) -> tuple[jax.Array | None, jax.Array | None]:
        if max_seq_len_cache is None:
            return None, None
        if seg_len <= toeplitz_max_len:
            return None, jnp.asarray(compute_hankel_causal_toeplitz(max_seq_len_cache, seg_len, num_filters=K))
        fp_np, _ = compute_hankel_filter_ffts(max_seq_len_cache, seg_len, num_filters=K)
        return jnp.asarray(fp_np), None

    if doc_plan is None:
        f_c, t_c = _precomp(L)
        return _causal_multi_conv_1d(xf, filters, f_fft_precomp=f_c, toeplitz_precomp=t_c)

    bucket_precomps = [_precomp(int(bt.shape[2])) for bt in doc_plan.bucket_t]

    def _per_seq(x_row: jax.Array, bt_tuple: tuple[jax.Array, ...], bv_tuple: tuple[jax.Array, ...]) -> jax.Array:
        x_pad = jnp.concatenate([x_row, jnp.zeros((1, C), dtype=jnp.float32)], axis=0)
        out = jnp.zeros((L + 1, K, C), dtype=jnp.float32)
        for t_idx, valid, (f_c, t_c) in zip(bt_tuple, bv_tuple, bucket_precomps):
            mask_in = valid[..., None].astype(jnp.float32)
            mask_out = valid[..., None, None].astype(jnp.float32)
            packed = x_pad[t_idx] * mask_in  # [M_seq, S, C]
            y_seg = _causal_multi_conv_1d(packed, filters, f_fft_precomp=f_c, toeplitz_precomp=t_c) * mask_out
            out = out.at[t_idx].add(y_seg)
        return out[:L, :, :]

    return jax.vmap(_per_seq)(xf, doc_plan.bucket_t, doc_plan.bucket_valid)


def make_orthogonal_reservoir_bank(hidden_size: int, res_channels: int, layer_idx: int, chip_seed: int) -> np.ndarray:
    """Deterministic frozen orthogonal random feature bank W_res [hidden_size, res_channels]."""
    seed = (int(chip_seed) * 9973 + int(layer_idx) * 131 + 17) % (2**31 - 1)
    rng = np.random.default_rng(seed)
    w_raw = rng.standard_normal((hidden_size, res_channels), dtype=np.float32)
    q_mat, _ = np.linalg.qr(w_raw)
    return (q_mat[:, :res_channels] * math.sqrt(float(hidden_size) / float(res_channels))).astype(np.float32)


class SpectralReservoirMixer(nn.Module):
    """Frozen Hankel spectral filter bank + fixed random orthogonal tanh feature map + learned read-outs."""

    hidden_size: int
    max_seq_len: int
    analog_cfg: AnalogNoiseConfig
    layer_idx: int
    num_filters: int = 16
    inner_dim: int = 512
    res_channels: int = 64
    reservoir_gain: float = 1.25
    init_std: float = 0.02
    out_init_std: float = 0.02
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self) -> None:
        phi_pos, phi_neg = compute_hankel_spectral_filters(self.max_seq_len, num_filters=self.num_filters)
        self.phi_pos = jnp.asarray(phi_pos)
        self.phi_neg = jnp.asarray(phi_neg)
        self.w_res_fixed = jnp.asarray(
            make_orthogonal_reservoir_bank(
                self.hidden_size, self.res_channels, self.layer_idx, self.analog_cfg.chip_seed
            )
        )

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        doc_plan: StaticDocPlan | None = None,
        training: bool = True,
        eval_chip_seed: int | None = None,
    ) -> jax.Array:
        B, L, _ = x.shape
        cfg = self.analog_cfg

        # Path A: Frequency-contracted Hankel spectral filtering (precomputed filter spectra)
        u = AnalogLinear(
            self.inner_dim, cfg, layer_tag=f"L{self.layer_idx}.res_in", init_std=self.init_std, dtype=self.dtype, name="in_proj"
        )(x, training=training, eval_chip_seed=eval_chip_seed)

        init_val = 1.0 / math.sqrt(self.num_filters)
        mode_w_pos = self.param(
            "mode_weight_pos", lambda _k, s: jnp.full(s, init_val, dtype=jnp.float32), (self.num_filters, self.inner_dim)
        )
        mode_w_neg = self.param(
            "mode_weight_neg", lambda _k, s: jnp.full(s, init_val, dtype=jnp.float32), (self.num_filters, self.inner_dim)
        )
        y_lin = static_doc_isolated_contracted_fft(
            u,
            self.phi_pos,
            self.phi_neg,
            mode_w_pos,
            mode_w_neg,
            doc_plan=doc_plan,
            max_seq_len_cache=self.max_seq_len,
        )
        y_lin = AnalogLinear(
            self.inner_dim, cfg, layer_tag=f"L{self.layer_idx}.spec_out", init_std=self.init_std, dtype=self.dtype, name="spec_readout"
        )(y_lin.astype(self.dtype), training=training, eval_chip_seed=eval_chip_seed)

        # Path B: Frozen orthogonal nonlinear feature map + multi-timescale spectral readout (Toeplitz GEMM <=256, FFT >256)
        w_res = self.w_res_fixed
        if cfg.dev_mismatch_sigma > 0.0:
            if training:
                delta = jax.random.normal(self.make_rng("analog_noise"), w_res.shape, dtype=jnp.float32) * cfg.dev_mismatch_sigma
            else:
                seed = cfg.chip_seed if eval_chip_seed is None else int(eval_chip_seed)
                ck = deterministic_chip_key(f"L{self.layer_idx}.w_res", w_res.shape, seed)
                delta = jax.random.normal(ck, w_res.shape, dtype=jnp.float32) * cfg.dev_mismatch_sigma
            w_res = w_res * (1.0 + delta)

        pre = jnp.matmul(x.astype(jnp.float32), w_res) * self.reservoir_gain
        if cfg.therm_noise_sigma > 0.0:
            if training or self.has_rng("analog_noise"):
                therm_key = self.make_rng("analog_noise")
            else:
                seed = cfg.chip_seed if eval_chip_seed is None else int(eval_chip_seed)
                therm_key = deterministic_chip_key(f"L{self.layer_idx}.w_res_therm", pre.shape, seed)
            rms = jax.lax.stop_gradient(jnp.sqrt(jnp.mean(jnp.square(pre), axis=-1, keepdims=True) + 1e-6))
            pre = pre + jax.random.normal(therm_key, pre.shape, dtype=jnp.float32) * (
                rms * cfg.therm_noise_sigma
            )
        r = jnp.tanh(pre)
        if cfg.adc_bits > 0:
            r = adc_quantize_ste(r, cfg.adc_bits, cfg.adc_clip_sigma)

        r_spec = static_doc_isolated_causal_fft(
            r,
            self.phi_pos,
            doc_plan=doc_plan,
            max_seq_len_cache=self.max_seq_len,
        )
        r_flat = jnp.reshape(r_spec, (B, L, self.num_filters * self.res_channels)).astype(self.dtype)
        y_res = AnalogLinear(
            self.inner_dim, cfg, layer_tag=f"L{self.layer_idx}.res_readout", init_std=self.init_std, dtype=self.dtype, name="res_readout"
        )(r_flat, training=training, eval_chip_seed=eval_chip_seed)

        gate = nn.silu(
            AnalogLinear(
                self.inner_dim, cfg, layer_tag=f"L{self.layer_idx}.res_gate", init_std=self.init_std, dtype=self.dtype, name="gate_proj"
            )(x, training=training, eval_chip_seed=eval_chip_seed)
        )
        return AnalogLinear(
            self.hidden_size, cfg, layer_tag=f"L{self.layer_idx}.o_proj", init_std=self.out_init_std, dtype=self.dtype, name="o_proj"
        )((y_lin + y_res) * gate, training=training, eval_chip_seed=eval_chip_seed)
