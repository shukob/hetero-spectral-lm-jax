"""Strictly causal 12.5 Hz streaming audio / multimodal turn-taking adapter on 3:1 Spectral-Reservoir (JAX/Flax)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from .analog import AnalogLinear
from .config import ModelConfig
from .model import DecoderLayer, RMSNorm, build_doc_causal_mask, rope_cos_sin


@dataclass(frozen=True)
class StreamingAudioConfig:
    """Configuration for causal 12.5 Hz (80 ms/frame) streaming audio / multimodal turn-taking heads."""

    in_feat_dim: int = 524  # e.g., kyutai/mimi 512d + 12d causal prosody (or +384d streaming text)
    hidden_size: int = 192
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    intermediate_size: int = 384
    max_frames: int = 50  # 4.0s causal window @ 12.5 Hz
    num_classes: int = 3  # 0=HOLD_PAUSE, 1=BACKCHANNEL_CUE, 2=TURN_SHIFT
    proj_dim: int = 128
    arch_mode: str = "hybrid_reservoir"  # "hybrid_reservoir" | "transformer"
    hybrid_attn_stride: int = 4
    num_spectral_filters: int = 8
    quant_mode: str = "none"  # "none" | "int4_sym" | "ternary_158"
    analog_dev_sigma: float = 0.0
    analog_therm_sigma: float = 0.0
    analog_adc_bits: int = 0
    analog_scope: str = "except_gqa_attn"
    dtype: str = "float32"

    def to_backbone_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=256,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_frames,
            use_mup=False,
            arch_mode=self.arch_mode,
            hybrid_attn_stride=self.hybrid_attn_stride,
            num_spectral_filters=self.num_spectral_filters,
            quant_mode=self.quant_mode,
            analog_dev_sigma=self.analog_dev_sigma,
            analog_therm_sigma=self.analog_therm_sigma,
            analog_adc_bits=self.analog_adc_bits,
            analog_scope=self.analog_scope,
            dtype=self.dtype,
        )

    @property
    def kv_cache_ratio(self) -> float:
        bb = self.to_backbone_config()
        n_gqa = sum(1 for i in range(self.num_hidden_layers) if bb.layer_mixer_kind(i) == "gqa")
        return float(n_gqa) / float(max(self.num_hidden_layers, 1))


class AudioTurnPointerHead(nn.Module):
    """Calibrated scaled dot-product pointer over turn actions (zero autoregressive token decode)."""

    num_classes: int = 3
    proj_dim: int = 128

    @nn.compact
    def __call__(self, h: jax.Array) -> jax.Array:
        h_n = RMSNorm(name="norm")(h).astype(jnp.float32)
        q = nn.Dense(self.proj_dim, dtype=jnp.float32, name="q_proj")(h_n)
        keys = self.param(
            "action_keys",
            nn.initializers.normal(stddev=0.05),
            (self.num_classes, self.proj_dim),
            jnp.float32,
        )
        bias = self.param("action_bias", nn.initializers.zeros, (self.num_classes,), jnp.float32)
        log_temp = self.param("log_temperature", nn.initializers.zeros, (1,), jnp.float32)
        temp = jnp.clip(jnp.exp(log_temp), 0.25, 4.0)[0]
        return jnp.einsum("btd,cd->btc", q, keys) / (math.sqrt(float(self.proj_dim)) * temp) + bias


class StreamingAudioTurnModel(nn.Module):
    """Strictly causal 12.5 Hz 3:1 Spectral-Reservoir turn-taking model (reuses core DecoderLayer)."""

    config: StreamingAudioConfig

    @nn.compact
    def __call__(
        self,
        frame_feats: jax.Array,
        *,
        training: bool = True,
        eval_chip_seed: int | None = None,
        return_all_frames: bool = False,
    ) -> jax.Array:
        c = self.config
        bb = c.to_backbone_config()
        dtype = jnp.bfloat16 if c.dtype == "bfloat16" else jnp.float32
        B, T, _ = frame_feats.shape

        x_norm = RMSNorm(name="in_norm")(frame_feats)
        x = nn.silu(
            AnalogLinear(
                c.hidden_size,
                bb.analog_config(),
                layer_tag="audio.in_proj",
                init_std=0.02,
                dtype=dtype,
                name="in_proj",
            )(x_norm, training=training, eval_chip_seed=eval_chip_seed)
        )

        pos_ids = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32)[None, :], (B, T))
        cos, sin = rope_cos_sin(pos_ids, bb.head_dim, bb.rope_theta, dtype)
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
        attn_mask = build_doc_causal_mask(None, T)

        for idx in range(c.num_hidden_layers):
            x = DecoderLayer(bb, idx, training=training, eval_chip_seed=eval_chip_seed, name=f"layer_{idx}")(
                x, cos, sin, attn_mask, None
            )

        x = RMSNorm(bb.rms_norm_eps, name="final_norm")(x)
        all_logits = AudioTurnPointerHead(c.num_classes, c.proj_dim, name="turn_head")(x)
        return all_logits if return_all_frames else all_logits[:, -1, :]
