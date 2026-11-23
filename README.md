# hetero-spectral-lm-jax

Open-science **JAX / Flax / Optax / Orbax** reference implementation for the scaling study:

> **Do Attention-Light, Analog-Noise-Tolerant Language Models Keep Up with Transformers at Scale? (0.10B–1.4B Parameters)**
>
> **Research Overview Deck (10 slides, Keynote style PDF)**: [docs/research_overview.pdf](docs/research_overview.pdf) (reproducible source: [docs/research_overview.tex](docs/research_overview.tex), figures: [scripts/build_overview_figs.py](scripts/build_overview_figs.py))

## Architecture & Key Modules

1. **3:1 Spectral-Reservoir Hybrid Language Model** ([model.py](src/hetero_spectral_lm/model.py), [spectral.py](src/hetero_spectral_lm/spectral.py)):
   - 3 of every 4 layers replace self-attention with cached precomputed-FFT Hankel Spectral Transform Unit (STU) filters + a fixed random orthogonal `tanh` feature map with exact causal Toeplitz GEMM for short buckets (`S <= 256`, MXU/Tensor-Core accelerated) and FFT for longer buckets.
   - 1 of every 4 layers retains Grouped-Query Attention (GQA) with QK-Norm, RoPE, and `jax.nn.dot_product_attention`, reducing **attention KV-cache entries per decoded token by 75% by construction** at matched depth and head dimensions.
   - Implements full μP initialization and `logits / width_mult` scaling (`mup_base_width=256`).
2. **Hetero-Physical 1.58-Bit + Analog CIM Simulator** ([analog.py](src/hetero_spectral_lm/analog.py)):
   - Multiplier-free `1.58-bit` absmean ternary weights (`{-1, 0, +1}` with STE), static per-chip conductance mismatch (resampled during training, fixed per `chip_seed` at evaluation), dynamic thermal read noise, and symmetric RMS-calibrated finite-bit ADC quantization (`3–8 bit`).
   - `analog_scope="except_gqa_attn"` assigns **~94% of trainable non-embedding weights** to noisy `1.58-bit` analog hardware while keeping only the sparse GQA projections (~6%) in digital precision.
3. **Per-Sequence Static-Shape Dyadic Document Isolation for Zero-Collective XLA SPMD** ([packing.py](src/hetero_spectral_lm/packing.py)):
   - Offline dyadic per-sequence packing (`StaticDocPlan`, sharded along `P("data")`, Hall's-theorem upward-spill capacities such as `(256×16, 512×4, 1024×2, 2048×1)` at `L=2048`) guarantees **strictly `0.00e+00` cross-document leakage** across both spectral and attention layers, triggers **1 XLA JIT trace** across variable document layouts, cuts `L=2048` padded FFT work by `2.4×` (`12× -> 5×`) and `n_fft=4096` slots by `8×`, and executes per-sequence convolutions via `jax.vmap` with **zero cross-chip collectives**.
   - Explicitly rejects non-contiguous `doc_id` reuse and missing `doc_plan` on packed hybrid calls.
4. **Hybrid Muon + AdamW with WSD Schedule in Optax** ([optim.py](src/hetero_spectral_lm/optim.py)):
   - Routes 2D hidden linear matrices to 5-step Newton-Schulz Muon with tracked schedule step count and Moonlight RMS scaling (`0.2 * sqrt(max(M, N))`), while embeddings and 1D/norm/mode parameters use AdamW.
5. **SPMD / Lightweight 1D FSDP Harness + Async Orbax GCS Checkpointing** ([train.py](src/hetero_spectral_lm/train.py)):
   - Supports `--fsdp` (sharding 2D+ matrices across `P("data", None)` for 16 GiB HBM devices), `--save-every`, `--resume-from`, and `etils.epath.Path` for both local paths and regional `gs://` buckets via `ocp.CheckpointManager(enable_async_checkpointing=True)`.

## Architecture Presets (32,768 Vocab, Tied Embeddings, Verified Against `estimate_params()`)

| Preset | Hidden | Layers (Res : GQA) | Total Params (A: TF / B,C: Hybrid) | Non-Emb Diff (B vs A) | Analog Non-Emb Fraction (C) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `0.10b` | 768 | 12 (9 : 3) | 100.68M / 104.37M | +4.88% | 93.83% |
| `0.26b` | 1024 | 20 (15 : 5) | 259.04M / 267.95M | +3.95% | 94.26% |
| `0.65b` | 1536 | 24 (18 : 6) | 644.95M / 680.93M | +6.05% | 94.28% |
| `1.3b` | 2048 | 28 (21 : 7) | 1,329.71M / 1,397.62M | +5.38% | 94.41% |

## Quickstart

```bash
# Install on Cloud TPU (see https://docs.jax.dev/en/latest/installation.html)
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -e ".[dev]"

# Or on CUDA 12 GPU
pip install "jax[cuda12]" && pip install -e ".[dev]"

# Run the full verification suite
pytest -v

# Reproduce the 0.10B FP32 512-token smoke benchmark (single-host SPMD/FSDP)
python -m hetero_spectral_lm.train \
  --preset 0.10b \
  --arch-mode hybrid_reservoir \
  --analog-hat \
  --dtype float32 \
  --steps 6 \
  --batch-size 4 \
  --seq-len 512
```

## Verification Suite (`tests/test_equivalence_and_xla.py`)

- **Exact FFT Contraction Parity**: Frequency-domain contraction matches naive O(L²K) NumPy causal time-domain convolution to `<1e-4`.
- **Zero Cross-Document Leakage (`0.00e+00`) & API Guards**: Scrambling Document 0 inside a packed sequence produces `0.0` logit difference on Document 1; missing `doc_plan` with `doc_ids` or non-contiguous `doc_id` reuse raises `ValueError`.
- **Single-Trace Static-Shape XLA (FP32 & BF16)**: 5 batches with varying document counts and lengths trigger **strictly 1** `jax.jit` trace in both `float32` and `bfloat16` with `remat_layers=True`.
- **L=2048 Packing Stress & Exact Preset Counts**: Validates complete token coverage with zero bucket overflow at `L=2048` and exact parameter counts across all 4 presets.
- **Muon WSD Schedule & Async Orbax Resume**: Verifies zero-start callable schedules advance per step, `gs://` URI preservation without local `.resolve()`, bitwise `0.0` restore of parameters, all optimizer-state leaves (`treedef` + tensors), and RNG keys, and `<1e-5` continuation parity across checkpoint boundaries.
- **PyTorch Operator Parity (`<1e-6` FP32, Checked-In Reference Fixture)**: Unconditionally verifies `<1e-6` max abs error against `tests/fixtures/pytorch_operator_ref_fp32.npz` (generated via [scripts/generate_pytorch_parity_fixture.py](scripts/generate_pytorch_parity_fixture.py)) across Hankel filters, contracted causal FFTs, `1.58-bit` STE, `6-bit` ADC STE, and 5-step Newton-Schulz—even in JAX-only CPU/TPU environments—plus a live PyTorch test when `torch` is installed.

## Local Smoke Benchmark (`2× RTX 2080 Ti`, FP32, 6 steps)

*Measured on synthetic packed batches with `JAX 0.10.2 / Flax 0.12.8`, `remat=True`, per-sequence `P("data")` sharding (`Steady Valid Targets/s` over steps 1–5 after step-0 compile). These are local FP32 GPU smoke checks, not TPU BF16 throughput estimates.*

| Configuration | Params (Non-Emb) | Analog Fraction | XLA Traces | `L=512, B=4` (DP) | `L=2048, B=2` (FSDP) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Model A: Transformer (FP32) | 100.68M (75.52M) | 0.0% | **1** | **7,404.1** (1.00×) | **8,980.7** (1.00×) |
| Model B: 3:1 Spectral-Reservoir (FP32) | 104.37M (79.20M) | 0.0% | **1** | **6,252.0** (0.84×) | **7,365.3** (0.82×) |
| Model C: Hetero-Physical 1.58b HAT (FP32) | 104.37M (79.20M) | **93.83%** | **1** | **6,017.4** (0.81×) | **7,197.9** (0.80×) |

## License

Apache-2.0 (see [LICENSE](LICENSE)). This is independent, non-commercial open academic research.
