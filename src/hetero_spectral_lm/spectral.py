"""Static-shape intra-document causal Spectral-Reservoir mixer in JAX/Flax (0.00e+00 cross-doc leakage)."""

from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .analog import AnalogLinear, AnalogNoiseConfig, adc_quantize_ste, deterministic_chip_key
from .packing import StaticDocPlan


def compute_hankel_spectral_filters(seq_len: int, num_filters: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact positive & alternating Hankel spectral filters in FP64 -> FP32 (Hazan et al. STU).

    Matrix entry: Z[i, j] = 2 / ((i + j + 2)^3 - (i + j + 2)) for 0 <= i, j < seq_len.
    Eigenvectors are scaled by sigma_k^(1/4); alternating branch multiplies by (-1)^t.
    """
    idx = np.arange(1, seq_len + 1, dtype=np.float64)
    s = idx[:, None] + idx[None, :]
    z = 2.0 / (s**3 - s)
    eigvals, eigvecs = np.linalg.eigh(z)
    top_vals = np.maximum(eigvals[-num_filters:][::-1], 1e-12)
    top_vecs = eigvecs[:, -num_filters:][:, ::-1].T  # [K, L]
    phi_pos = (top_vecs * (top_vals**0.25)[:, None]).astype(np.float32)
    alt = np.where((np.arange(seq_len) % 2) == 0, 1.0, -1.0).astype(np.float32)
    phi_neg = phi_pos * alt[None, :]
    return phi_pos, phi_neg


def _next_pow2(n: int) -> int:
    return 1 << (2 * n - 1).bit_length()


def _contracted_fft_conv_1d(
    packed: jax.Array,
    phi_pos: jax.Array,
    phi_neg: jax.Array,
    w_pos: jax.Array,
    w_neg: jax.Array,
) -> jax.Array:
    """Exact frequency-domain linear contraction in FP32: sum_k w_k * (x * phi_k)."""
    seg_len = packed.shape[1]
    n_fft = _next_pow2(seg_len)
    x_fft = jnp.fft.rfft(packed.astype(jnp.float32), n=n_fft, axis=1)  # [M, F, C]
    fp = jnp.fft.rfft(phi_pos[:, :seg_len].astype(jnp.float32), n=n_fft, axis=1)  # [K, F]
    fn = jnp.fft.rfft(phi_neg[:, :seg_len].astype(jnp.float32), n=n_fft, axis=1)  # [K, F]
    wp = w_pos.astype(jnp.float32)
    wn = w_neg.astype(jnp.float32)
    h_real = jnp.matmul(fp.real.T, wp) + jnp.matmul(fn.real.T, wn)  # [F, C]
    h_imag = jnp.matmul(fp.imag.T, wp) + jnp.matmul(fn.imag.T, wn)  # [F, C]
    h_fft = jax.lax.complex(h_real, h_imag)
    y_fft = x_fft * h_fft[None, :, :]
    return jnp.fft.irfft(y_fft, n=n_fft, axis=1)[:, :seg_len, :]


def _causal_multi_fft_conv_1d(packed: jax.Array, filters: jax.Array) -> jax.Array:
    """Exact causal 1D FFT convolution of [M, S, C] with [K, L_f] -> [M, S, K, C] in FP32."""
    seg_len = packed.shape[1]
    n_fft = _next_pow2(seg_len)
    x_fft = jnp.fft.rfft(packed.astype(jnp.float32), n=n_fft, axis=1)  # [M, F, C]
    f_fft = jnp.fft.rfft(filters[:, :seg_len].astype(jnp.float32), n=n_fft, axis=1)  # [K, F]
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
) -> jax.Array:
    """100% static-shape intra-document causal contracted FFT convolution [B, L, C] -> [B, L, C]."""
    B, L, C = x.shape
    xf = x.astype(jnp.float32)
    if doc_plan is None:
        return _contracted_fft_conv_1d(xf, phi_pos, phi_neg, w_pos, w_neg)

    # Pad one trash slot at index L so invalid slots never contend on t=0 (conflict-free 1-to-1 scatter)
    xf_pad = jnp.concatenate([xf, jnp.zeros((B, 1, C), dtype=jnp.float32)], axis=1)
    out = jnp.zeros((B, L + 1, C), dtype=jnp.float32)
    for b_idx, t_idx, valid in (
        (doc_plan.short_b, doc_plan.short_t, doc_plan.short_valid),
        (doc_plan.long_b, doc_plan.long_t, doc_plan.long_valid),
    ):
        mask = valid[..., None].astype(jnp.float32)
        packed = xf_pad[b_idx, t_idx] * mask  # [M, S, C]; zero-padded beyond each doc
        y_seg = _contracted_fft_conv_1d(packed, phi_pos, phi_neg, w_pos, w_neg) * mask
        out = out.at[b_idx, t_idx].add(y_seg)
    return out[:, :L, :]


def static_doc_isolated_causal_fft(
    x: jax.Array,
    filters: jax.Array,
    doc_plan: StaticDocPlan | None = None,
) -> jax.Array:
    """100% static-shape intra-document multi-filter causal FFT [B, L, C] -> [B, L, K, C]."""
    B, L, C = x.shape
    K = filters.shape[0]
    xf = x.astype(jnp.float32)
    if doc_plan is None:
        return _causal_multi_fft_conv_1d(xf, filters)

    xf_pad = jnp.concatenate([xf, jnp.zeros((B, 1, C), dtype=jnp.float32)], axis=1)
    out = jnp.zeros((B, L + 1, K, C), dtype=jnp.float32)
    for b_idx, t_idx, valid in (
        (doc_plan.short_b, doc_plan.short_t, doc_plan.short_valid),
        (doc_plan.long_b, doc_plan.long_t, doc_plan.long_valid),
    ):
        mask_in = valid[..., None].astype(jnp.float32)
        mask_out = valid[..., None, None].astype(jnp.float32)
        packed = xf_pad[b_idx, t_idx] * mask_in  # [M, S, C]
        y_seg = _causal_multi_fft_conv_1d(packed, filters) * mask_out  # [M, S, K, C]
        out = out.at[b_idx, t_idx].add(y_seg)
    return out[:, :L, :, :]


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

        # Path A: Frequency-contracted Hankel spectral filtering
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
        y_lin = static_doc_isolated_contracted_fft(u, self.phi_pos, self.phi_neg, mode_w_pos, mode_w_neg, doc_plan=doc_plan)
        y_lin = AnalogLinear(
            self.inner_dim, cfg, layer_tag=f"L{self.layer_idx}.spec_out", init_std=self.init_std, dtype=self.dtype, name="spec_readout"
        )(y_lin.astype(self.dtype), training=training, eval_chip_seed=eval_chip_seed)

        # Path B: Frozen orthogonal nonlinear feature map + multi-timescale spectral readout
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
            rms = jax.lax.stop_gradient(jnp.sqrt(jnp.mean(jnp.square(pre), axis=-1, keepdims=True) + 1e-6))
            pre = pre + jax.random.normal(self.make_rng("analog_noise"), pre.shape, dtype=jnp.float32) * (
                rms * cfg.therm_noise_sigma
            )
        r = jnp.tanh(pre)
        if cfg.adc_bits > 0:
            r = adc_quantize_ste(r, cfg.adc_bits, cfg.adc_clip_sigma)

        r_spec = static_doc_isolated_causal_fft(r, self.phi_pos, doc_plan=doc_plan)
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
            self.hidden_size, cfg, layer_tag=f"L{self.layer_idx}.o_proj", init_std=self.init_std, dtype=self.dtype, name="o_proj"
        )((y_lin + y_res) * gate, training=training, eval_chip_seed=eval_chip_seed)
