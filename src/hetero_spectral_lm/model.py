"""Hetero-Physical 3:1 Spectral-Reservoir / Transformer LM in JAX/Flax (Pre-RMSNorm, SwiGLU, GQA, QK-Norm, RoPE)."""

from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp

from .analog import AnalogLinear, AnalogNoiseConfig
from .config import ModelConfig
from .packing import StaticDocPlan
from .spectral import SpectralReservoirMixer


def _resolve_dtype(name: str) -> jnp.dtype:
    return jnp.bfloat16 if name == "bfloat16" else jnp.float32


class RMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        dim = x.shape[-1]
        weight = self.param("weight", nn.initializers.ones, (dim,), jnp.float32)
        xf = x.astype(jnp.float32)
        inv = jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + self.eps)
        return (xf * inv * weight).astype(x.dtype)


def rope_cos_sin(
    position_ids: jax.Array,
    head_dim: int,
    theta: float,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / float(head_dim)))
    freqs = position_ids.astype(jnp.float32)[..., None] * inv_freq[None, None, :]  # [B, L, D/2]
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # [B, L, D]
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Apply rotary position embeddings to x [B, H, L, D] with cos/sin [B, 1, L, D]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = jnp.concatenate([-x2, x1], axis=-1)
    return (x * cos) + (rot * sin)


def build_doc_causal_mask(doc_ids: jax.Array | None, seq_len: int) -> jax.Array:
    """Exact intra-document causal boolean mask [B, 1, L, L] (or [1, 1, L, L] if doc_ids is None)."""
    pos = jnp.arange(seq_len)
    causal = pos[:, None] >= pos[None, :]  # [L, L]
    if doc_ids is None:
        return causal[None, None, :, :]
    same_doc = (doc_ids[:, :, None] == doc_ids[:, None, :]) & (doc_ids[:, :, None] >= 0)
    is_pad = doc_ids < 0
    eye = jnp.eye(seq_len, dtype=jnp.bool_)[None, :, :]
    allow = (same_doc & causal[None, :, :]) | (is_pad[:, :, None] & eye)
    return allow[:, None, :, :]


class GQAWithQKNorm(nn.Module):
    config: ModelConfig
    layer_idx: int
    analog_cfg: AnalogNoiseConfig

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        attn_mask: jax.Array,
        *,
        training: bool = True,
        eval_chip_seed: int | None = None,
    ) -> jax.Array:
        c = self.config
        dtype = _resolve_dtype(c.dtype)
        B, L, _ = x.shape
        nh = c.num_attention_heads
        nkv = c.num_key_value_heads
        nrep = nh // nkv
        hd = c.head_dim

        q = AnalogLinear(nh * hd, self.analog_cfg, f"L{self.layer_idx}.q", c.init_std, dtype, name="q_proj")(
            x, training=training, eval_chip_seed=eval_chip_seed
        )
        k = AnalogLinear(nkv * hd, self.analog_cfg, f"L{self.layer_idx}.k", c.init_std, dtype, name="k_proj")(
            x, training=training, eval_chip_seed=eval_chip_seed
        )
        v = AnalogLinear(nkv * hd, self.analog_cfg, f"L{self.layer_idx}.v", c.init_std, dtype, name="v_proj")(
            x, training=training, eval_chip_seed=eval_chip_seed
        )

        q = RMSNorm(c.rms_norm_eps, name="q_norm")(q.reshape(B, L, nh, hd))
        k = RMSNorm(c.rms_norm_eps, name="k_norm")(k.reshape(B, L, nkv, hd))
        v = v.reshape(B, L, nkv, hd)

        q = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, L, D]
        k = jnp.transpose(k, (0, 2, 1, 3))  # [B, KV, L, D]
        v = jnp.transpose(v, (0, 2, 1, 3))

        q = apply_rope(q, cos, sin).astype(v.dtype)
        k = apply_rope(k, cos, sin).astype(v.dtype)

        if nrep > 1:
            k = jnp.repeat(k, nrep, axis=1)
            v = jnp.repeat(v, nrep, axis=1)

        scale = 1.0 / math.sqrt(float(hd))
        scores = jnp.matmul(q.astype(jnp.float32), jnp.swapaxes(k.astype(jnp.float32), -2, -1)) * scale
        scores = jnp.where(attn_mask, scores, jnp.asarray(-1e9, dtype=jnp.float32))
        probs = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        out = jnp.matmul(probs, v)  # [B, H, L, D]
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, L, nh * hd)
        return AnalogLinear(c.hidden_size, self.analog_cfg, f"L{self.layer_idx}.o", c.init_std, dtype, name="o_proj")(
            out, training=training, eval_chip_seed=eval_chip_seed
        )


class SwiGLUMLP(nn.Module):
    config: ModelConfig
    layer_idx: int
    analog_cfg: AnalogNoiseConfig

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        training: bool = True,
        eval_chip_seed: int | None = None,
    ) -> jax.Array:
        c = self.config
        dtype = _resolve_dtype(c.dtype)
        g = AnalogLinear(c.intermediate_size, self.analog_cfg, f"L{self.layer_idx}.mlp_g", c.init_std, dtype, name="gate_proj")(
            x, training=training, eval_chip_seed=eval_chip_seed
        )
        u = AnalogLinear(c.intermediate_size, self.analog_cfg, f"L{self.layer_idx}.mlp_u", c.init_std, dtype, name="up_proj")(
            x, training=training, eval_chip_seed=eval_chip_seed
        )
        return AnalogLinear(c.hidden_size, self.analog_cfg, f"L{self.layer_idx}.mlp_d", c.init_std, dtype, name="down_proj")(
            nn.silu(g) * u, training=training, eval_chip_seed=eval_chip_seed
        )


class DecoderLayer(nn.Module):
    config: ModelConfig
    layer_idx: int
    training: bool = True
    eval_chip_seed: int | None = None

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        attn_mask: jax.Array,
        doc_plan: StaticDocPlan | None = None,
    ) -> jax.Array:
        c = self.config
        kind = c.layer_mixer_kind(self.layer_idx)
        mix_cfg = c.analog_config_for(self.layer_idx, "mixer")
        mlp_cfg = c.analog_config_for(self.layer_idx, "mlp")

        h = RMSNorm(c.rms_norm_eps, name="input_layernorm")(x)
        if kind == "gqa":
            mix_out = GQAWithQKNorm(c, self.layer_idx, mix_cfg, name="self_attn")(
                h, cos, sin, attn_mask, training=self.training, eval_chip_seed=self.eval_chip_seed
            )
        else:
            mix_out = SpectralReservoirMixer(
                hidden_size=c.hidden_size,
                max_seq_len=c.max_position_embeddings,
                analog_cfg=mix_cfg,
                layer_idx=self.layer_idx,
                num_filters=c.num_spectral_filters,
                inner_dim=c.non_attn_inner_dim,
                res_channels=c.res_channels,
                reservoir_gain=c.reservoir_gain,
                init_std=c.init_std,
                dtype=_resolve_dtype(c.dtype),
                name="self_attn",
            )(h, doc_plan=doc_plan, training=self.training, eval_chip_seed=self.eval_chip_seed)

        x = x + mix_out
        x = x + SwiGLUMLP(c, self.layer_idx, mlp_cfg, name="mlp")(
            RMSNorm(c.rms_norm_eps, name="post_attention_layernorm")(x),
            training=self.training,
            eval_chip_seed=self.eval_chip_seed,
        )
        return x


class HeteroSpectralLM(nn.Module):
    """Full causal language model supporting Transformer, 3:1 Spectral-Reservoir, and Hetero-Physical HAT."""

    config: ModelConfig
    remat_layers: bool = False

    @nn.compact
    def __call__(
        self,
        input_ids: jax.Array,
        position_ids: jax.Array | None = None,
        doc_ids: jax.Array | None = None,
        doc_plan: StaticDocPlan | None = None,
        *,
        training: bool = True,
        eval_chip_seed: int | None = None,
    ) -> jax.Array:
        c = self.config
        dtype = _resolve_dtype(c.dtype)
        B, L = input_ids.shape
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(L, dtype=jnp.int32)[None, :], (B, L))

        embed_table = self.param(
            "embed_tokens",
            nn.initializers.normal(stddev=c.init_std),
            (c.vocab_size, c.hidden_size),
            jnp.float32,
        )
        x = embed_table[input_ids].astype(dtype)

        cos, sin = rope_cos_sin(position_ids, c.head_dim, c.rope_theta, dtype)
        cos = cos[:, None, :, :]  # [B, 1, L, D]
        sin = sin[:, None, :, :]
        attn_mask = build_doc_causal_mask(doc_ids, L)

        layer_cls = nn.remat(DecoderLayer) if self.remat_layers else DecoderLayer
        for idx in range(c.num_hidden_layers):
            x = layer_cls(c, idx, training=training, eval_chip_seed=eval_chip_seed, name=f"layer_{idx}")(
                x, cos, sin, attn_mask, doc_plan
            )

        x = RMSNorm(c.rms_norm_eps, name="final_norm")(x)
        if c.tie_word_embeddings:
            lm_w = embed_table.T
        else:
            lm_w = self.param(
                "lm_head",
                nn.initializers.normal(stddev=c.init_std),
                (c.hidden_size, c.vocab_size),
                jnp.float32,
            )
        return jnp.matmul(x.astype(jnp.float32), lm_w.astype(jnp.float32))


def compute_ce_and_zloss(
    logits: jax.Array,
    input_ids: jax.Array,
    doc_ids: jax.Array | None = None,
    z_loss_coef: float = 1e-4,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Next-token cross-entropy + z-loss, strictly masking out cross-document boundary predictions & pads."""
    shift_logits = logits[:, :-1, :].astype(jnp.float32)  # [B, L-1, V]
    targets = input_ids[:, 1:]  # [B, L-1]

    if doc_ids is not None:
        valid = (doc_ids[:, :-1] == doc_ids[:, 1:]) & (doc_ids[:, 1:] >= 0)
    else:
        valid = jnp.ones_like(targets, dtype=jnp.bool_)

    log_z = jax.nn.logsumexp(shift_logits, axis=-1)  # [B, L-1]
    target_logits = jnp.take_along_axis(shift_logits, targets[..., None], axis=-1)[..., 0]
    ce_per_tok = log_z - target_logits
    z_per_tok = jnp.square(log_z)

    mask = valid.astype(jnp.float32)
    n_valid = jnp.maximum(jnp.sum(mask), 1.0)
    ce_loss = jnp.sum(ce_per_tok * mask) / n_valid
    z_loss = jnp.sum(z_per_tok * mask) / n_valid
    total_loss = ce_loss + z_loss_coef * z_loss
    return total_loss, {
        "loss": total_loss,
        "ce_loss": ce_loss,
        "z_loss": z_loss,
        "valid_tokens": jnp.sum(mask),
    }
