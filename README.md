# hetero-spectral-lm-jax

Open-science **JAX / Flax / Optax / Orbax** reference implementation (`v0.4.1`) for the scaling study:

> **Do Attention-Light, Analog-Noise-Tolerant Language Models Keep Up with Transformers at Scale? (0.10B–1.4B Parameters)**
>
> **Research Overview Deck (14 slides, Keynote style PDF)**: [docs/research_overview.pdf](docs/research_overview.pdf) (reproducible source: [docs/research_overview.tex](docs/research_overview.tex), figures: [scripts/build_overview_figs.py](scripts/build_overview_figs.py))

## Architecture & Key Modules (`v0.4.1`)

1. **3:1 Spectral-Reservoir Hybrid Language Model** ([model.py](src/hetero_spectral_lm/model.py), [spectral.py](src/hetero_spectral_lm/spectral.py)):
   - 3 of every 4 layers replace self-attention with cached precomputed-FFT Hankel Spectral Transform Unit (STU) filters + a fixed random orthogonal `tanh` feature map with exact causal Toeplitz GEMM for short buckets (`S <= 256`, MXU/Tensor-Core accelerated) and FFT for longer buckets.
   - **Why "Spectral-Reservoir" on discrete text**: Once tokens are embedded (`x_1..x_L`), text forms a multi-channel discrete-time signal along the token-position axis. Eigenvectors of the Hilbert-Hankel memory matrix (`Z[i,j] = 2 / ((i+j+2)^3 - (i+j+2))`) provide frozen multi-horizon decay bases (from adjacent syntax to 2K-token discourse via `O(L log L)` FFT), while the frozen orthogonal `tanh` reservoir mixes features passively (BIBO-stable) without recurrent noise amplification.
   - 1 of every 4 layers retains Grouped-Query Attention (GQA) with QK-Norm, RoPE, and `jax.nn.dot_product_attention`, reducing **attention KV-cache entries per decoded token by 75% by construction** at matched depth and head dimensions.
   - Implements full μP initialization and `logits / width_mult` scaling (`mup_base_width=256`).
2. **Hetero-Physical W4 INT4 / 1.58-Bit + Analog CIM Simulator** ([analog.py](src/hetero_spectral_lm/analog.py)):
   - Supports per-output-channel symmetric signed **W4 INT4** (`int4_sym`, `{-7..+7}` with STE — mainline) and multiplier-free **1.58-bit** absmean ternary (`ternary_158`, `{-1, 0, +1}` with STE), static per-chip conductance mismatch (resampled during training, fixed per `chip_seed` at evaluation), dynamic thermal read noise (deterministically keyed from `eval_chip_seed` when `rngs` is omitted at evaluation), and symmetric RMS-calibrated finite-bit ADC quantization (`3–8 bit`).
   - `analog_scope="except_gqa_attn"` assigns **~94% of trainable non-embedding weights** to noisy low-bit analog hardware while keeping only the sparse GQA projections (~6%) in digital precision; `analog_scope="except_stride_gqa"` provides the exact equal-scope (~93.7% analog) Transformer control.
3. **Auxiliary Causal 12.5 Hz Streaming Turn-Taking Head Prototype** ([streaming_audio.py](src/hetero_spectral_lm/streaming_audio.py)):
   - Standalone lightweight 4-layer causal head prototype reusing `DecoderLayer` (KV-cache ratio `0.25`, `int4_sym`+CIM6 `except_gqa_attn`) with a calibrated `AudioTurnPointerHead` over causal 12.5 Hz (80 ms/frame) audio codec latents, verifying `<1e-5` future-leak causality and 1 XLA JIT trace (auxiliary downstream prototype outside the core 0.10B–1.4B pretraining sweep).
4. **Per-Sequence Static-Shape Dyadic Document Isolation for Zero-Collective XLA SPMD** ([packing.py](src/hetero_spectral_lm/packing.py)):
   - Offline dyadic per-sequence packing (`StaticDocPlan`, sharded along `P("data")`, Hall's-theorem upward-spill capacities such as `(256×16, 512×4, 1024×2, 2048×1)` at `L=2048`) guarantees **strictly `0.00e+00` cross-document leakage** across both spectral and attention layers, triggers **1 XLA JIT trace** across variable document layouts, cuts `L=2048` padded FFT work by `2.4×` (`12× -> 5×`) and `n_fft=4096` slots by `8×`, and executes per-sequence convolutions via `jax.vmap` with **zero cross-chip collectives**.
   - Explicitly rejects non-contiguous `doc_id` reuse and missing `doc_plan` on packed hybrid calls.
5. **Hybrid Muon + AdamW with WSD Schedule, Multi-Host TPU / 1D FSDP & Async Orbax GCS Checkpointing** ([optim.py](src/hetero_spectral_lm/optim.py), [train.py](src/hetero_spectral_lm/train.py)):
   - Routes 2D hidden linear matrices to 5-step Newton-Schulz Muon with tracked schedule step count and Moonlight RMS scaling (`0.2 * sqrt(max(M, N))`), while embeddings and 1D/norm/mode parameters use AdamW.
   - Supports `--multihost` (`jax.distributed.initialize()` + `jax.make_array_from_process_local_data` for multi-host TPU pods), `--fsdp` (`P("data", None)` for 16 GiB HBM devices), `--quant-mode {none,int4_sym,ternary_158}`, `--analog-scope`, `--save-every`, `--resume-from`, and `etils.epath.Path` for both local paths and regional `gs://` buckets via `ocp.CheckpointManager(enable_async_checkpointing=True)`.

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

# Run the full verification suite (10 automated tests pass in JAX-only environments; +1 live PyTorch test if torch installed)
pytest -v

# Reproduce the 0.10B FP32 512-token smoke benchmark (W4 int4_sym + 6-bit Analog CIM HAT)
python -m hetero_spectral_lm.train \
  --preset 0.10b \
  --arch-mode hybrid_reservoir \
  --analog-hat \
  --quant-mode int4_sym \
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
- **W4 `int4_sym` STE, Equal-Scope `except_stride_gqa` & Unseen-Chip Determinism**: Verifies signed `{-7..+7}` INT4 and `{-1,0,+1}` ternary STE values/gradients, exact `93.72%` equal-scope Transformer analog fraction at `0.10b`, 2-bit vs 8-bit ADC fault detection, and bitwise per-chip determinism (including `training=False` with `therm_noise_sigma > 0` without explicit `rngs`).
- **Causal 12.5 Hz Audio Turn-Taking Adapter**: Verifies `kv_cache_ratio == 0.25`, `<1e-5` future-leak causality under perturbation at `t >= 2.400s`, and strictly `1` XLA trace on `StreamingAudioTurnModel`.
- **Muon WSD Schedule & Async Orbax Resume**: Verifies zero-start callable schedules advance per step, `gs://` URI preservation without local `.resolve()`, bitwise `0.0` restore of parameters/optimizer/RNG state, and `<1e-5` continuation parity across checkpoint boundaries.
- **PyTorch Operator Parity (`<1e-6` FP32, Checked-In Reference Fixture)**: Unconditionally verifies `<1e-6` max abs error against `tests/fixtures/pytorch_operator_ref_fp32.npz` (generated via [scripts/generate_pytorch_parity_fixture.py](scripts/generate_pytorch_parity_fixture.py)) across Hankel filters, contracted causal FFTs, **W4 `int4_sym` STE (Flax & PyTorch layouts)**, `1.58-bit` STE, `6-bit` ADC STE, and 5-step Newton-Schulz—even in JAX-only CPU/TPU environments.

## Local Smoke Benchmark (`2× RTX 2080 Ti`, FP32, 6 steps)

*Measured on synthetic packed batches with `JAX 0.10.2 / Flax 0.12.8`, `remat=True`, per-sequence `P("data")` sharding (`Steady Valid Targets/s` over steps 1–5 after step-0 compile). These are local FP32 GPU smoke checks, not TPU BF16 throughput estimates.*

| Configuration | Params (Non-Emb) | Analog Fraction | XLA Traces | `L=512, B=4` (DP) | `L=2048, B=2` (FSDP) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Model A: Transformer (FP32) | 100.68M (75.52M) | 0.0% | **1** | **7,404.1** (1.00×) | **8,980.7** (1.00×) |
| Model B: 3:1 Spectral-Reservoir (FP32) | 104.37M (79.20M) | 0.0% | **1** | **6,252.0** (0.84×) | **7,365.3** (0.82×) |
| Model C (Mainline): W4 `int4_sym` + 6b CIM HAT | 104.37M (79.20M) | **93.83%** | **1** | **6,038.9** (0.82×) | **6,592.7** (0.73×) |
| Model C (Ablation): 1.58b Ternary + 6b CIM HAT | 104.37M (79.20M) | **93.83%** | **1** | **6,017.4** (0.81×) | **7,197.9** (0.80×) |

## License

Apache-2.0 (see [LICENSE](LICENSE)). This is independent, non-commercial open academic research.
