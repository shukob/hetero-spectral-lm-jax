# hetero-spectral-lm-jax

Open-science **JAX / Flax / Optax / Orbax** reference implementation for the scaling study:

> **Do Attention-Light, Analog-Noise-Tolerant Language Models Keep Up with Transformers at Scale? (0.10B–1.3B Parameters)**

This repository provides a clean, static-shape XLA/TPU-ready implementation of:
1. **3:1 Spectral-Reservoir Hybrid Language Model** ([model.py](src/hetero_spectral_lm/model.py), [spectral.py](src/hetero_spectral_lm/spectral.py)):
   - 3 of every 4 layers replace self-attention with frozen Hankel Spectral Transform Unit (STU) filters + a fixed random orthogonal `tanh` feature map with learned read-outs.
   - 1 of every 4 layers retains Grouped-Query Attention (GQA) with QK-Norm and RoPE, reducing attention KV-cache memory by **75% by construction**.
2. **Hetero-Physical 1.58-Bit + Analog CIM Simulator** ([analog.py](src/hetero_spectral_lm/analog.py)):
   - Multiplier-free `1.58-bit` absmean ternary weights (`{-1, 0, +1}` with STE), static per-chip fabrication conductance mismatch (`chip_seed`-deterministic at evaluation), dynamic thermal read noise, and symmetric RMS-calibrated finite-bit ADC quantization (`3–8 bit`).
   - `analog_scope="except_gqa_attn"` assigns **~94% of non-embedding weights** to noisy `1.58-bit` analog hardware while keeping only the sparse GQA projections (~6%) in digital BF16 precision.
3. **Zero-Recompilation Static-Shape Document Isolation for XLA** ([packing.py](src/hetero_spectral_lm/packing.py)):
   - Offline two-bucket fixed-shape packing (`StaticDocPlan`) guarantees **strictly `0.00e+00` cross-document leakage** in both FFT convolutions and causal attention while keeping all tensor shapes constant so `jax.jit` compiles **once**.
4. **Hybrid Muon + AdamW Optimizer in Optax** ([optim.py](src/hetero_spectral_lm/optim.py)):
   - Routes 2D hidden linear matrices to 5-step Newton-Schulz Muon with Moonlight RMS scaling (`0.2 * sqrt(max(M, N))`), while embeddings and 1D/norm/mode parameters use AdamW.
5. **TPU / Multi-GPU SPMD Harness + Orbax Preemption Recovery** ([train.py](src/hetero_spectral_lm/train.py)):
   - Data-parallel `jax.sharding.Mesh` loop with full Orbax checkpointing (parameters, optimizer state, RNG key, and data cursor) verified for exact resume across Spot/preemptible interruptions.

## Architecture Presets (32K Vocab, Tied Embeddings)

| Preset | Hidden | Layers (Res : GQA) | Total Params (TF / Hybrid) | Non-Emb Analog Fraction (`except_gqa_attn`) |
| :--- | ---: | ---: | ---: | ---: |
| `0.10b` | 768 | 12 (9 : 3) | 100.7M / 104.4M | 93.83% |
| `0.26b` | 1024 | 20 (15 : 5) | 259.0M / 267.9M | 94.26% |
| `0.65b` | 1536 | 24 (18 : 6) | 645.0M / 671.5M | 94.37% |
| `1.3b` | 2048 | 28 (21 : 7) | 1.33B / 1.39B | 94.42% |

## Quickstart

```bash
# Install (CPU / TPU)
pip install -e ".[dev]"

# Or on CUDA 12 GPU
pip install "jax[cuda12]" && pip install -e ".[dev]"

# Run the full verification suite
pytest -q

# Run a 0.10B static-shape training smoke benchmark
python -m hetero_spectral_lm.train \
  --preset 0.10b \
  --arch-mode hybrid_reservoir \
  --analog-hat \
  --steps 10 \
  --batch-size 4 \
  --seq-len 512
```

## Verified Results (JAX 0.10.2 / Flax 0.12.8 on 2× RTX 2080 Ti, FP32, `remat=True`)

### 1. Cross-Framework Numerical Parity vs. PyTorch Reference (`max |JAX - PyTorch|`)

| Operator | Max Abs Error |
| :--- | ---: |
| Hankel Spectral Filters (`phi_pos`, `phi_neg`) | `0.000e+00` |
| Static-Shape Document-Isolated Contracted FFT | `2.384e-07` |
| 1.58-Bit Absmean Ternary STE | `2.384e-07` |
| 6-Bit RMS-Calibrated ADC STE | `7.153e-07` |
| 5-Step Newton-Schulz Orthogonalization (Muon) | `5.960e-07` |

### 2. `0.10B` Multi-GPU SPMD Smoke Benchmark (`batch_size=4`, `seq_len=512`, 6 steps)

| Configuration | Params (Non-Emb) | Analog Fraction | XLA Compiles | Steady Non-Pad tok/s |
| :--- | ---: | ---: | ---: | ---: |
| Model A: Transformer | 100.7M (75.5M) | 0.0% | **1** | 7,457.9 |
| Model B: 3:1 Spectral-Reservoir Hybrid | 104.4M (79.2M) | 0.0% | **1** | 5,316.6 |
| Model C: Hetero-Physical 1.58b Analog HAT | 104.4M (79.2M) | **93.83%** | **1** | 5,060.1 (-4.8% vs B) |

> Under static-shape XLA compilation, adding full `1.58-bit` STE + device mismatch + thermal noise + 6-bit ADC across `93.83%` of non-embedding weights adds only **4.8%** steady-state step overhead relative to the clean hybrid baseline.

## Verification Suite (`tests/test_equivalence_and_xla.py`)

- **Exact FFT Contraction Parity**: Frequency-domain contraction matches naive O(L²K) causal time-domain convolution to `<1e-4`.
- **Zero Cross-Document Leakage (`0.00e+00`)**: Scrambling Document 0 inside a packed sequence produces `0.0` logit difference on Document 1 across all spectral and GQA layers.
- **Single-Compile Static-Shape XLA Proof**: 5 consecutive batches with varying document counts and lengths trigger **strictly 1** `jax.jit` trace.
- **Hardware-Aware Noise & Unseen-Chip Determinism**: Verifies `{-1, 0, +1}` ternary STE, 2-bit vs 8-bit ADC sensitivity, deterministic per-chip mismatch, and exact `93.83%` analog weight fraction at `0.10B`.
- **Exact Orbax Preemption Resume**: 5 continuous steps match `3 steps -> Orbax save -> restore -> 2 steps` to `<1e-6` across all parameters and loss.

## License

Apache-2.0. This is independent, non-commercial open academic research.
