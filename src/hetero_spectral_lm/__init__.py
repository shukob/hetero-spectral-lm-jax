"""JAX/Flax implementation of Hetero-Physical Spectral-Reservoir Language & Causal Audio Models."""

from .analog import (
    AnalogLinear,
    AnalogNoiseConfig,
    adc_quantize_ste,
    int4_sym_per_row_ste,
    ternary_absmean_ste,
)
from .config import PRESET_CONFIGS, ModelConfig
from .model import HeteroSpectralLM, compute_ce_and_zloss
from .optim import (
    build_hybrid_muon_adamw,
    build_wsd_schedule,
    classify_param_for_muon,
    zeropower_via_newtonschulz5,
)
from .packing import StaticDocPlan, pack_documents_static
from .spectral import SpectralReservoirMixer, compute_hankel_spectral_filters
from .streaming_audio import AudioTurnPointerHead, StreamingAudioConfig, StreamingAudioTurnModel

__all__ = [
    "AnalogLinear",
    "AnalogNoiseConfig",
    "AudioTurnPointerHead",
    "HeteroSpectralLM",
    "ModelConfig",
    "PRESET_CONFIGS",
    "SpectralReservoirMixer",
    "StaticDocPlan",
    "StreamingAudioConfig",
    "StreamingAudioTurnModel",
    "adc_quantize_ste",
    "build_hybrid_muon_adamw",
    "build_wsd_schedule",
    "classify_param_for_muon",
    "compute_ce_and_zloss",
    "compute_hankel_spectral_filters",
    "int4_sym_per_row_ste",
    "pack_documents_static",
    "ternary_absmean_ste",
    "zeropower_via_newtonschulz5",
]
