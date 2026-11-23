"""Verification suite: numerical invariants, 0.00e+00 doc isolation, single-compile XLA, analog faults, Orbax resume."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from hetero_spectral_lm import (
    PRESET_CONFIGS,
    AnalogNoiseConfig,
    HeteroSpectralLM,
    ModelConfig,
    adc_quantize_ste,
    build_hybrid_muon_adamw,
    compute_ce_and_zloss,
    compute_hankel_spectral_filters,
    pack_documents_static,
    ternary_absmean_ste,
    zeropower_via_newtonschulz5,
)
from hetero_spectral_lm.packing import make_synthetic_packed_batch
from hetero_spectral_lm.spectral import static_doc_isolated_contracted_fft
from hetero_spectral_lm.train import restore_checkpoint, save_checkpoint


def _tiny_cfg(**overrides) -> ModelConfig:
    base = ModelConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        arch_mode="hybrid_reservoir",
        hybrid_attn_stride=4,
        num_spectral_filters=8,
        dtype="float32",
    )
    return replace(base, **overrides)


def test_hankel_and_contracted_fft_exact_equivalence() -> None:
    """Verify frequency-domain contraction matches naive per-filter time-domain causal convolution (<1e-4)."""
    L, K, C = 64, 8, 16
    phi_pos, phi_neg = compute_hankel_spectral_filters(L, num_filters=K)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, L, C)).astype(np.float32)
    wp = rng.standard_normal((K, C)).astype(np.float32) * 0.2
    wn = rng.standard_normal((K, C)).astype(np.float32) * 0.2

    y_fft = np.asarray(
        static_doc_isolated_contracted_fft(
            jnp.asarray(x), jnp.asarray(phi_pos), jnp.asarray(phi_neg), jnp.asarray(wp), jnp.asarray(wn), doc_plan=None
        )
    )

    # Naive exact O(L^2 * K) reference
    y_ref = np.zeros_like(x)
    for b in range(2):
        for c in range(C):
            for k in range(K):
                conv_p = np.convolve(x[b, :, c], phi_pos[k, :L], mode="full")[:L]
                conv_n = np.convolve(x[b, :, c], phi_neg[k, :L], mode="full")[:L]
                y_ref[b, :, c] += wp[k, c] * conv_p + wn[k, c] * conv_n

    max_err = float(np.max(np.abs(y_fft - y_ref)))
    assert max_err < 1e-4, f"Contracted FFT mismatch vs exact reference: {max_err}"


def test_zero_cross_document_leakage_in_full_model() -> None:
    """Known-fault probe: mutating doc 0 tokens must produce strictly 0.00e+00 logit diff on doc 1."""
    cfg = _tiny_cfg(quant_mode="none")
    model = HeteroSpectralLM(cfg)
    B, L = 2, 128
    rng = np.random.default_rng(123)
    tok_a = rng.integers(1, cfg.vocab_size, size=(B, L), dtype=np.int32)
    tok_b = tok_a.copy()
    # Completely scramble doc 0 (positions 0..49) in batch item 0; keep doc 1 (50..127) identical
    tok_b[0, :50] = (tok_a[0, :50] * 7 + 13) % (cfg.vocab_size - 1) + 1

    doc_ids = np.zeros((B, L), dtype=np.int32)
    doc_ids[:, 50:] = 1
    pos_ids = np.zeros((B, L), dtype=np.int32)
    pos_ids[:, :50] = np.arange(50)
    pos_ids[:, 50:] = np.arange(L - 50)
    plan = pack_documents_static(doc_ids, short_len=64)

    params = model.init(
        {"params": jax.random.PRNGKey(0), "analog_noise": jax.random.PRNGKey(1)},
        jnp.asarray(tok_a),
        jnp.asarray(pos_ids),
        jnp.asarray(doc_ids),
        plan,
        training=False,
    )["params"]

    logits_a = model.apply({"params": params}, jnp.asarray(tok_a), jnp.asarray(pos_ids), jnp.asarray(doc_ids), plan, training=False)
    logits_b = model.apply({"params": params}, jnp.asarray(tok_b), jnp.asarray(pos_ids), jnp.asarray(doc_ids), plan, training=False)

    doc0_diff = float(jnp.max(jnp.abs(logits_a[0, :50] - logits_b[0, :50])))
    doc1_diff = float(jnp.max(jnp.abs(logits_a[0, 50:] - logits_b[0, 50:])))
    assert doc0_diff > 1e-2, "Doc 0 perturbation had no effect (broken sensitivity check)"
    assert doc1_diff == 0.0, f"Cross-document leakage detected on doc 1: {doc1_diff:.3e}"


def test_xla_single_compile_across_variable_doc_layouts() -> None:
    """5 batches with different doc counts & lengths must trigger strictly ONE XLA jit compilation."""
    cfg = _tiny_cfg(quant_mode="ternary_158", analog_dev_sigma=0.05, analog_therm_sigma=0.03, analog_adc_bits=6)
    model = HeteroSpectralLM(cfg)
    B, L = 2, 128

    t0, p0, d0, plan0 = make_synthetic_packed_batch(B, L, cfg.vocab_size, seed=1, min_doc_len=16, max_doc_len=96, short_len=64, max_short_segs=16, max_long_segs=8)
    params = model.init({"params": jax.random.PRNGKey(0), "analog_noise": jax.random.PRNGKey(1)}, t0, p0, d0, plan0, training=True)["params"]
    opt = build_hybrid_muon_adamw(params, lr=1e-3, muon_lr=1e-3, use_bf16_ns=False)
    opt_state = opt.init(params)

    traces = {"n": 0}

    @jax.jit
    def step_fn(p, st, key, tok, pos, doc, plan):
        traces["n"] += 1

        def loss_fn(w):
            logits = model.apply({"params": w}, tok, pos, doc, plan, training=True, rngs={"analog_noise": key})
            return compute_ce_and_zloss(logits, tok, doc_ids=doc)

        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        upd, new_st = opt.update(grads, st, p)
        return optax.apply_updates(p, upd), new_st, loss

    for i in range(5):
        tok, pos, doc, plan = make_synthetic_packed_batch(
            B, L, cfg.vocab_size, seed=10 + i, min_doc_len=16 + i * 4, max_doc_len=96, short_len=64, max_short_segs=16, max_long_segs=8
        )
        params, opt_state, loss = step_fn(params, opt_state, jax.random.PRNGKey(i), tok, pos, doc, plan)
        assert jnp.isfinite(loss), f"Non-finite loss at step {i}: {loss}"

    assert traces["n"] == 1, f"Expected strictly 1 XLA trace across variable doc layouts, got {traces['n']}"


def test_analog_simulator_quantization_unseen_chip_and_hetero_fraction() -> None:
    """Verify 1.58-bit STE values, ADC fault sensitivity, deterministic per-chip seeds, and 93.8% analog ratio."""
    w = jnp.asarray([[0.01, -0.8, 0.9], [-0.02, 0.7, -0.6]], dtype=jnp.float32)
    w_q = ternary_absmean_ste(w)
    scale = jnp.mean(jnp.abs(w))
    q_int = jnp.round(w_q / scale)
    assert set(np.asarray(q_int).ravel().tolist()).issubset({-1.0, 0.0, 1.0})

    # Verify STE gradient is strictly identity (non-zero!)
    grad_w = jax.grad(lambda x: jnp.sum(ternary_absmean_ste(x)))(w)
    assert np.allclose(np.asarray(grad_w), 1.0, atol=1e-6)

    # Known-fault check: 2-bit ADC must perturb continuous activations far more than 8-bit ADC
    x = jax.random.normal(jax.random.PRNGKey(9), (4, 64), dtype=jnp.float32)
    err_8b = float(jnp.mean(jnp.abs(adc_quantize_ste(x, 8) - x)))
    err_2b = float(jnp.mean(jnp.abs(adc_quantize_ste(x, 2) - x)))
    assert err_2b > 10.0 * err_8b, f"Expected 2-bit ADC error >> 8-bit error, got {err_2b} vs {err_8b}"

    # Unseen chip seed determinism vs distinctness
    cfg = _tiny_cfg(quant_mode="ternary_158", analog_dev_sigma=0.2, analog_therm_sigma=0.0, analog_adc_bits=6)
    model = HeteroSpectralLM(cfg)
    tok, pos, doc, plan = make_synthetic_packed_batch(1, 128, cfg.vocab_size, seed=2, short_len=64)
    params = model.init({"params": jax.random.PRNGKey(0), "analog_noise": jax.random.PRNGKey(1)}, tok, pos, doc, plan, training=False)["params"]
    y_c1a = model.apply({"params": params}, tok, pos, doc, plan, training=False, eval_chip_seed=2001)
    y_c1b = model.apply({"params": params}, tok, pos, doc, plan, training=False, eval_chip_seed=2001)
    y_c2 = model.apply({"params": params}, tok, pos, doc, plan, training=False, eval_chip_seed=2002)
    assert float(jnp.max(jnp.abs(y_c1a - y_c1b))) == 0.0
    assert float(jnp.max(jnp.abs(y_c1a - y_c2))) > 1e-2

    # Check exact 0.10B hetero-physical non-embedding analog fraction (~93.83%)
    c100 = replace(PRESET_CONFIGS["0.10b"], arch_mode="hybrid_reservoir", quant_mode="ternary_158", analog_scope="except_gqa_attn")
    frac = c100.estimate_params()["analog_non_emb_fraction"]
    assert abs(frac - 0.9383) < 1e-3, f"Unexpected 0.10B analog_non_emb_fraction: {frac:.4f}"


def test_newton_schulz_orthogonality_and_orbax_preemption_resume(tmp_path: Path) -> None:
    """Verify Muon Newton-Schulz singular values near 1 and exact Orbax checkpoint preemption resume."""
    g = jax.random.normal(jax.random.PRNGKey(3), (32, 64), dtype=jnp.float32)
    q = zeropower_via_newtonschulz5(g, steps=5, use_bf16=False)
    sv = np.linalg.svd(np.asarray(q), compute_uv=False)
    assert 0.5 < float(sv.min()) and float(sv.max()) < 1.3

    cfg = _tiny_cfg(quant_mode="ternary_158", analog_dev_sigma=0.05, analog_therm_sigma=0.03, analog_adc_bits=6)
    model = HeteroSpectralLM(cfg)
    batches = [make_synthetic_packed_batch(2, 128, cfg.vocab_size, seed=50 + i, short_len=64, max_short_segs=16, max_long_segs=8) for i in range(5)]
    init_p = model.init({"params": jax.random.PRNGKey(0), "analog_noise": jax.random.PRNGKey(1)}, *batches[0], training=True)["params"]
    opt = build_hybrid_muon_adamw(init_p, lr=1e-3, muon_lr=1e-3, use_bf16_ns=False)
    init_st = opt.init(init_p)

    @jax.jit
    def step_fn(p, st, key, tok, pos, doc, plan):
        def loss_fn(w):
            logits = model.apply({"params": w}, tok, pos, doc, plan, training=True, rngs={"analog_noise": key})
            return compute_ce_and_zloss(logits, tok, doc_ids=doc)

        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        upd, new_st = opt.update(grads, st, p)
        return optax.apply_updates(p, upd), new_st, loss

    # Continuous 5-step run
    p_cont, st_cont, rng_cont = init_p, init_st, jax.random.PRNGKey(77)
    for b in batches:
        rng_cont, sk = jax.random.split(rng_cont)
        p_cont, st_cont, loss_cont = step_fn(p_cont, st_cont, sk, *b)

    # Interrupted run: 3 steps -> Orbax save -> fresh restore -> 2 steps
    p_int, st_int, rng_int = init_p, init_st, jax.random.PRNGKey(77)
    for b in batches[:3]:
        rng_int, sk = jax.random.split(rng_int)
        p_int, st_int, _ = step_fn(p_int, st_int, sk, *b)

    ckpt_path = save_checkpoint(tmp_path / "ckpts", step=3, params=p_int, opt_state=st_int, rng_key=rng_int, data_cursor=3)
    restored = restore_checkpoint(ckpt_path, abstract_params=init_p, abstract_opt_state=init_st, abstract_rng_key=rng_int)
    p_res, st_res, rng_res = restored["params"], restored["opt_state"], restored["rng_key"]
    assert int(restored["step"]) == 3 and int(restored["data_cursor"]) == 3
    # Verify exact bitwise state preservation at checkpoint boundary
    ckpt_diffs = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda a, b: jnp.max(jnp.abs(a - b)), p_int, p_res))
    assert max(float(d) for d in ckpt_diffs) == 0.0
    assert np.array_equal(np.asarray(rng_int), np.asarray(rng_res))

    for b in batches[3:]:
        rng_res, sk = jax.random.split(rng_res)
        p_res, st_res, loss_res = step_fn(p_res, st_res, sk, *b)

    assert abs(float(loss_cont) - float(loss_res)) < 1e-5
    diffs = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda a, b: jnp.max(jnp.abs(a - b)), p_cont, p_res))
    assert max(float(d) for d in diffs) < 1e-5
