"""Physically-faithful W4 INT4 / 1.58-bit ternary + Analog CIM non-ideality simulator in JAX/Flax."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class AnalogNoiseConfig:
    """Hardware non-ideality parameters for Hardware-Aware Training (HAT) and unseen-chip evaluation.

    - `quant_mode`: `"none"`, `"int4_sym"` (per-output-channel symmetric signed INT4 {-7..+7}), or `"ternary_158"` (BitNet b1.58 {-1, 0, +1}).
    - `dev_mismatch_sigma`: Static per-weight conductance mismatch N(1, sigma^2), frozen per chip seed at eval.
    - `therm_noise_sigma`: Dynamic kT/C thermal read noise scaled to output RMS on every forward pass.
    - `adc_bits`: Symmetric signed uniform ADC bit-width (0 disables ADC quantization; typical range 3..8).
    - `adc_clip_sigma`: Full-scale rail clip in units of activation RMS before uniform ADC quantization.
    - `chip_seed`: Deterministic silicon die seed for static device mismatch.
    """

    quant_mode: str = "none"
    dev_mismatch_sigma: float = 0.0
    therm_noise_sigma: float = 0.0
    adc_bits: int = 0
    adc_clip_sigma: float = 4.0
    chip_seed: int = 1001

    @property
    def is_active(self) -> bool:
        return (
            self.quant_mode != "none"
            or self.dev_mismatch_sigma > 0.0
            or self.therm_noise_sigma > 0.0
            or self.adc_bits > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deterministic_chip_key(layer_tag: str, shape: tuple[int, ...], chip_seed: int) -> jax.Array:
    """Platform-independent deterministic PRNG key for frozen silicon mismatch."""
    payload = f"{layer_tag}:{shape}:{int(chip_seed)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    seed_u32 = int.from_bytes(digest, "little")
    return jax.random.PRNGKey(seed_u32)


def ternary_absmean_ste(w: jax.Array, eps: float = 1e-6) -> jax.Array:
    """BitNet b1.58 exact absmean ternary quantization {-1, 0, +1} with Straight-Through Estimator (STE)."""
    wf = w.astype(jnp.float32)
    scale = jnp.maximum(jnp.mean(jnp.abs(wf)), eps)
    q = jnp.clip(jnp.round(wf / scale), -1.0, 1.0)
    w_hat = (q * scale).astype(w.dtype)
    return w + jax.lax.stop_gradient(w_hat - w)


def int4_sym_per_row_ste(w: jax.Array, eps: float = 1e-6, *, channel_axis: int = -1) -> jax.Array:
    """Per-output-channel symmetric signed INT4 {-7..+7} quantization with Straight-Through Estimator (STE).

    Defaults to `channel_axis=-1` for Flax `[in_features, out_features]` kernels (reducing over `in_features`);
    pass `channel_axis=0` for PyTorch-layout `[out_features, in_features]` weight matrices.
    """
    wf = w.astype(jnp.float32)
    ch_ax = channel_axis % wf.ndim
    reduce_axes = tuple(i for i in range(wf.ndim) if i != ch_ax)
    scale = jnp.maximum(jnp.max(jnp.abs(wf), axis=reduce_axes, keepdims=True) / 7.0, eps)
    q = jnp.clip(jnp.round(wf / scale), -7.0, 7.0)
    w_hat = (q * scale).astype(w.dtype)
    return w + jax.lax.stop_gradient(w_hat - w)


def adc_quantize_ste(
    x: jax.Array,
    bits: int,
    clip_sigma: float = 4.0,
    eps: float = 1e-6,
) -> jax.Array:
    """Symmetric signed uniform ADC quantization (2^(b-1)-1 levels) with RMS-calibrated full-scale rails + STE."""
    if bits <= 0:
        return x
    xf = x.astype(jnp.float32)
    qmax = float((1 << (bits - 1)) - 1)
    rms = jnp.sqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + eps)
    fs = jnp.maximum(rms * clip_sigma, eps)
    step = fs / qmax
    xq = (jnp.clip(jnp.round(xf / step), -qmax, qmax) * step).astype(x.dtype)
    return x + jax.lax.stop_gradient(xq - x)


class AnalogLinear(nn.Module):
    """Linear macro supporting clean BF16/FP32 or 1.58-bit ternary + Analog CIM mismatch/thermal/ADC noise."""

    features: int
    analog_cfg: AnalogNoiseConfig
    layer_tag: str = ""
    init_std: float = 0.02
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        training: bool = True,
        eval_chip_seed: int | None = None,
    ) -> jax.Array:
        in_features = x.shape[-1]
        kernel = self.param(
            "kernel",
            nn.initializers.normal(stddev=self.init_std),
            (in_features, self.features),
            jnp.float32,
        )
        cfg = self.analog_cfg

        # Clean fast path
        if not cfg.is_active:
            return jnp.matmul(x.astype(self.dtype), kernel.astype(self.dtype))

        x_in = adc_quantize_ste(x, cfg.adc_bits, cfg.adc_clip_sigma) if cfg.adc_bits > 0 else x
        w = kernel.astype(self.dtype)
        if cfg.quant_mode == "ternary_158":
            w = ternary_absmean_ste(w)
        elif cfg.quant_mode == "int4_sym":
            w = int4_sym_per_row_ste(w, channel_axis=-1)
        elif cfg.quant_mode != "none":
            raise ValueError(f"Unsupported quant_mode: {cfg.quant_mode!r}")

        if cfg.dev_mismatch_sigma > 0.0:
            if training:
                noise_key = self.make_rng("analog_noise")
                delta = jax.random.normal(noise_key, w.shape, dtype=jnp.float32) * cfg.dev_mismatch_sigma
            else:
                seed = cfg.chip_seed if eval_chip_seed is None else int(eval_chip_seed)
                chip_key = deterministic_chip_key(self.layer_tag, w.shape, seed)
                delta = jax.random.normal(chip_key, w.shape, dtype=jnp.float32) * cfg.dev_mismatch_sigma
            mult = jnp.maximum(1.0 + delta, 0.25).astype(w.dtype)
            w = w * mult

        y = jnp.matmul(x_in.astype(self.dtype), w)

        if cfg.therm_noise_sigma > 0.0:
            therm_key = self.make_rng("analog_noise")
            rms = jax.lax.stop_gradient(
                jnp.sqrt(jnp.mean(jnp.square(y.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6)
            )
            noise = jax.random.normal(therm_key, y.shape, dtype=jnp.float32) * (rms * cfg.therm_noise_sigma)
            y = y + noise.astype(y.dtype)

        if cfg.adc_bits > 0:
            y = adc_quantize_ste(y, cfg.adc_bits, cfg.adc_clip_sigma)
        return y
