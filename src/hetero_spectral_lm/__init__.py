"""JAX/Flax implementation of Hetero-Physical 1.58-Bit Spectral-Reservoir Language Models."""

from .analog import AnalogLinear, AnalogNoiseConfig, adc_quantize_ste, ternary_absmean_ste
from .config import PRESET_CONFIGS, ModelConfig
from .model import HeteroSpectralLM, compute_ce_and_zloss
from .optim import build_hybrid_muon_adamw, classify_param_for_muon, zeropower_via_newtonschulz5
from .packing import StaticDocPlan, pack_documents_static
from .spectral import SpectralReservoirMixer, compute_hankel_spectral_filters

__all__ = [
    "AnalogLinear",
    "AnalogNoiseConfig",
    "HeteroSpectralLM",
    "ModelConfig",
    "PRESET_CONFIGS",
    "SpectralReservoirMixer",
    "StaticDocPlan",
    "adc_quantize_ste",
    "build_hybrid_muon_adamw",
    "classify_param_for_muon",
    "compute_ce_and_zloss",
    "compute_hankel_spectral_filters",
    "pack_documents_static",
    "ternary_absmean_ste",
    "zeropower_via_newtonschulz5",
]
