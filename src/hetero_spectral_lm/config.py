"""Model and hetero-physical noise configuration presets (0.10B -> 0.26B -> 0.65B -> 1.3B)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .analog import AnalogNoiseConfig


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for Transformer and 3:1 Spectral-Reservoir Hybrid LMs with Hetero-Physical partitioning."""

    vocab_size: int = 32768
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    rope_theta: float = 500_000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    z_loss_coef: float = 1e-4
    init_std: float = 0.02
    # Architecture and hetero-physical noise settings
    arch_mode: str = "hybrid_reservoir"  # "transformer" | "hybrid_reservoir" | "pure_reservoir"
    hybrid_attn_stride: int = 4  # Every 4th layer (idx 3, 7, 11, ...) uses GQA
    num_spectral_filters: int = 16
    reservoir_gain: float = 1.25
    quant_mode: str = "none"  # "none" | "ternary_158"
    analog_dev_sigma: float = 0.0
    analog_therm_sigma: float = 0.0
    analog_adc_bits: int = 0
    analog_adc_clip_sigma: float = 4.0
    analog_chip_seed: int = 1001
    analog_scope: str = "except_gqa_attn"  # "all" | "non_attn_layers" | "except_gqa_attn"
    dtype: str = "bfloat16"  # "float32" | "bfloat16"

    @property
    def head_dim(self) -> int:
        assert self.hidden_size % self.num_attention_heads == 0
        return self.hidden_size // self.num_attention_heads

    @property
    def non_attn_inner_dim(self) -> int:
        """Width-proportional inner dimension for spectral-reservoir layers (multiple of head_dim)."""
        hd = self.head_dim
        raw = max(hd * 2, (self.hidden_size * 2) // 3)
        return max(hd * 2, (raw // hd) * hd)

    @property
    def res_channels(self) -> int:
        return max(16, self.non_attn_inner_dim // 8)

    def layer_mixer_kind(self, layer_idx: int) -> str:
        if self.arch_mode == "transformer":
            return "gqa"
        if self.arch_mode == "pure_reservoir":
            return "reservoir"
        if self.arch_mode == "hybrid_reservoir":
            is_attn = ((layer_idx + 1) % max(self.hybrid_attn_stride, 1)) == 0
            return "gqa" if is_attn else "reservoir"
        raise ValueError(f"Unsupported arch_mode: {self.arch_mode}")

    def analog_config(self) -> AnalogNoiseConfig:
        return AnalogNoiseConfig(
            quant_mode=self.quant_mode,
            dev_mismatch_sigma=self.analog_dev_sigma,
            therm_noise_sigma=self.analog_therm_sigma,
            adc_bits=self.analog_adc_bits,
            adc_clip_sigma=self.analog_adc_clip_sigma,
            chip_seed=self.analog_chip_seed,
        )

    def analog_config_for(self, layer_idx: int, part: str) -> AnalogNoiseConfig:
        """Hetero-physical partition: `except_gqa_attn` keeps ~94% analog and ~6% sparse GQA digital."""
        base = self.analog_config()
        if not base.is_active or self.analog_scope == "all":
            return base
        kind = self.layer_mixer_kind(layer_idx)
        if self.analog_scope == "non_attn_layers":
            return AnalogNoiseConfig() if kind == "gqa" else base
        if self.analog_scope == "except_gqa_attn":
            return AnalogNoiseConfig() if (kind == "gqa" and part == "mixer") else base
        raise ValueError(f"Unknown analog_scope: {self.analog_scope}")

    def estimate_params(self) -> dict[str, Any]:
        """Exact analytical trainable parameter count and non-embedding analog weight fraction."""
        d = self.hidden_size
        hd = self.head_dim
        kv_d = self.num_key_value_heads * hd
        inner = self.non_attn_inner_dim
        rc = self.res_channels
        kf = self.num_spectral_filters
        mlp_w = 3 * d * self.intermediate_size
        norms = 2 * d

        non_emb = d  # final RMSNorm
        analog_non_emb = 0
        for i in range(self.num_hidden_layers):
            kind = self.layer_mixer_kind(i)
            mix_cfg = self.analog_config_for(i, "mixer")
            mlp_cfg = self.analog_config_for(i, "mlp")
            if kind == "gqa":
                mix_lin = d * d + 2 * d * kv_d + d * d
                mix_total = mix_lin + 2 * hd  # QK-Norm
            else:
                mix_lin = 3 * d * inner + inner * inner + (rc * kf) * inner
                mix_total = mix_lin + 2 * (kf * inner)  # mode weights
            non_emb += mix_total + mlp_w + norms
            if mix_cfg.is_active:
                analog_non_emb += mix_lin
            if mlp_cfg.is_active:
                analog_non_emb += mlp_w

        emb = self.vocab_size * d
        lm_head = 0 if self.tie_word_embeddings else self.vocab_size * d
        return {
            "non_embedding": non_emb,
            "embedding": emb + lm_head,
            "total": non_emb + emb + lm_head,
            "analog_non_emb": analog_non_emb,
            "analog_non_emb_fraction": analog_non_emb / max(non_emb, 1),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESET_CONFIGS: dict[str, ModelConfig] = {
    "0.10b": ModelConfig(
        vocab_size=32768,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=2048,
    ),
    "0.26b": ModelConfig(
        vocab_size=32768,
        hidden_size=1024,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=2816,
    ),
    "0.65b": ModelConfig(
        vocab_size=32768,
        hidden_size=1536,
        num_hidden_layers=24,
        num_attention_heads=24,
        num_key_value_heads=6,
        intermediate_size=4096,
    ),
    "1.3b": ModelConfig(
        vocab_size=32768,
        hidden_size=2048,
        num_hidden_layers=28,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=5632,
    ),
}
