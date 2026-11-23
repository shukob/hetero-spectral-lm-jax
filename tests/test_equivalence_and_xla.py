"""Verification suite: numerical invariants, 0.00e+00 doc isolation, API guards, BF16/FP32 XLA trace=1, L=2048 packing, WSD Muon + async Orbax resume."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from hetero_spectral_lm import (
    PRESET_CONFIGS,
    HeteroSpectralLM,
    ModelConfig,
    adc_quantize_ste,
    build_hybrid_muon_adamw,
    build_wsd_schedule,
    compute_ce_and_zloss,
    compute_hankel_spectral_filters,
    pack_documents_static,
    ternary_absmean_ste,
    zeropower_via_newtonschulz5,
)
from hetero_spectral_lm.packing import make_synthetic_packed_batch
from hetero_spectral_lm.spectral import static_doc_isolated_contracted_fft
from hetero_spectral_lm.train import _to_epath, make_checkpoint_manager, restore_checkpoint, save_checkpoint


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
    """Frequency-domain contraction matches naive per-filter time-domain causal convolution (<1e-4)."""
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

    y_ref = np.zeros_like(x)
    for b in range(2):
        for c in range(C):
            for k in range(K):
                conv_p = np.convolve(x[b, :, c], phi_pos[k, :L], mode="full")[:L]
                conv_n = np.convolve(x[b, :, c], phi_neg[k, :L], mode="full")[:L]
                y_ref[b, :, c] += wp[k, c] * conv_p + wn[k, c] * conv_n

    max_err = float(np.max(np.abs(y_fft - y_ref)))
    assert max_err < 1e-4, f"Contracted FFT mismatch vs exact reference: {max_err}"


def test_zero_cross_document_leakage_and_api_guards() -> None:
    """Mutating doc 0 yields 0.00e+00 diff on doc 1; missing doc_plan or reused doc_ids raise ValueError."""
    cfg = _tiny_cfg(quant_mode="none")
    model = HeteroSpectralLM(cfg)
    B, L = 2, 128
    rng = np.random.default_rng(123)
    tok_a = rng.integers(1, cfg.vocab_size, size=(B, L), dtype=np.int32)
    tok_b = tok_a.copy()
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

    assert float(jnp.max(jnp.abs(logits_a[0, :50] - logits_b[0, :50]))) > 1e-2
    assert float(jnp.max(jnp.abs(logits_a[0, 50:] - logits_b[0, 50:]))) == 0.0

    # Guard 1: passing doc_ids without doc_plan to hybrid model must refuse silent leakage
    with pytest.raises(ValueError, match="without a matching `doc_plan`"):
        model.apply({"params": params}, jnp.asarray(tok_a), jnp.asarray(pos_ids), jnp.asarray(doc_ids), None, training=False)

    # Guard 2: reusing a doc_id across non-contiguous segments must raise ValueError
    bad_doc_ids = np.array([[0, 0, 1, 1, 0, 0, -1, -1]], dtype=np.int32)
    with pytest.raises(ValueError, match="Non-contiguous reuse"):
        pack_documents_static(bad_doc_ids, short_len=4)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_xla_single_trace_across_variable_doc_layouts_fp32_and_bf16(dtype: str) -> None:
    """5 batches with varying doc counts & lengths trigger strictly ONE XLA jit trace in both FP32 and BF16."""
    cfg = _tiny_cfg(
        quant_mode="ternary_158",
        analog_dev_sigma=0.05,
        analog_therm_sigma=0.03,
        analog_adc_bits=6,
        dtype=dtype,
    )
    model = HeteroSpectralLM(cfg, remat_layers=True)
    B, L = 2, 128

    t0, p0, d0, plan0 = make_synthetic_packed_batch(B, L, cfg.vocab_size, seed=1, min_doc_len=16, max_doc_len=96, short_len=64)
    params = model.init({"params": jax.random.PRNGKey(0), "analog_noise": jax.random.PRNGKey(1)}, t0, p0, d0, plan0, training=True)["params"]
    sched = build_wsd_schedule(1e-3, total_steps=5, warmup_steps=2, decay_steps=1)
    opt = build_hybrid_muon_adamw(params, lr=sched, muon_lr=sched, use_bf16_ns=(dtype == "bfloat16"))
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
            B, L, cfg.vocab_size, seed=10 + i, min_doc_len=16 + i * 4, max_doc_len=96, short_len=64
        )
        params, opt_state, loss = step_fn(params, opt_state, jax.random.PRNGKey(i), tok, pos, doc, plan)
        assert jnp.isfinite(loss), f"Non-finite loss ({dtype}) at step {i}: {loss}"

    assert traces["n"] == 1, f"Expected strictly 1 XLA trace ({dtype}), got {traces['n']}"


def test_packing_l2048_stress_and_preset_param_counts() -> None:
    """Verify L=2048 per-sequence dyadic packing handles worst-case long docs with exact coverage & <=5x padding."""
    for seed in (140, 143, 145, 200):
        tok, pos, doc, plan = make_synthetic_packed_batch(
            batch_size=8, seq_len=2048, vocab_size=32768, seed=seed, min_doc_len=32, max_doc_len=512, short_len=256
        )
        assert tok.shape == (8, 2048)
        assert plan.long_t.shape == (8, 1, 2048)
        padded_tok_per_seq = sum(int(bt.shape[1] * bt.shape[2]) for bt in plan.bucket_t)
        assert padded_tok_per_seq <= 5 * 2048
        # Verify every valid token (doc_id >= 0) appears exactly once across ALL dyadic buckets
        covered = np.zeros((8, 2048), dtype=np.int32)
        for b in range(8):
            for bt, bv in zip(plan.bucket_t, plan.bucket_valid):
                np.add.at(covered[b], np.asarray(bt[b])[np.asarray(bv[b])], 1)
        expected = (np.asarray(doc) >= 0).astype(np.int32)
        assert np.array_equal(covered, expected)

    # Also verify exact Toeplitz GEMM vs FFT parity inside static_doc_isolated_causal_fft (S <= 256)
    from hetero_spectral_lm.spectral import static_doc_isolated_causal_fft

    phi_pos, _ = compute_hankel_spectral_filters(256, num_filters=8)
    _, _, doc_small, plan_small = make_synthetic_packed_batch(2, 256, 256, seed=11, short_len=128)
    r_in = jax.random.normal(jax.random.PRNGKey(5), (2, 256, 16), dtype=jnp.float32)
    y_toep = static_doc_isolated_causal_fft(r_in, jnp.asarray(phi_pos), plan_small, max_seq_len_cache=256, toeplitz_max_len=256)
    y_fft = static_doc_isolated_causal_fft(r_in, jnp.asarray(phi_pos), plan_small, max_seq_len_cache=256, toeplitz_max_len=0)
    assert float(jnp.max(jnp.abs(y_toep - y_fft))) < 1e-5

    # Check exact analytical counts across all 4 presets match real Flax param tree on 0.10b
    expected_totals = {
        "0.10b": (100_684_032, 104_369_280, 0.9383),
        "0.26b": (259_042_816, 267_949_696, 0.9426),
        "0.65b": (644_952_576, 680_929_536, 0.9428),
        "1.3b": (1_329_714_688, 1_397_621_632, 0.9441),
    }
    for name, (tf_tot, hyb_tot, frac) in expected_totals.items():
        c_tf = replace(PRESET_CONFIGS[name], arch_mode="transformer")
        c_hyb = replace(PRESET_CONFIGS[name], arch_mode="hybrid_reservoir", quant_mode="ternary_158", analog_scope="except_gqa_attn")
        assert c_tf.estimate_params()["total"] == tf_tot
        st = c_hyb.estimate_params()
        assert st["total"] == hyb_tot
        assert abs(st["analog_non_emb_fraction"] - frac) < 1e-4

    # Verify Flax tree leaf count on tiny model equals estimate_params()
    tiny = _tiny_cfg(quant_mode="ternary_158", analog_scope="except_gqa_attn")
    m = HeteroSpectralLM(tiny)
    b = make_synthetic_packed_batch(1, 128, tiny.vocab_size, seed=0, short_len=64)
    p = m.init({"params": jax.random.PRNGKey(0), "analog_noise": jax.random.PRNGKey(1)}, *b, training=False)["params"]
    tree_total = sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(p))
    assert tree_total == tiny.estimate_params()["total"]


def test_analog_simulator_quantization_and_unseen_chip_determinism() -> None:
    """Verify 1.58-bit STE values/grad, 2-bit vs 8-bit ADC fault detection, and per-chip determinism."""
    w = jnp.asarray([[0.01, -0.8, 0.9], [-0.02, 0.7, -0.6]], dtype=jnp.float32)
    w_q = ternary_absmean_ste(w)
    scale = jnp.mean(jnp.abs(w))
    assert set(np.asarray(jnp.round(w_q / scale)).ravel().tolist()).issubset({-1.0, 0.0, 1.0})
    assert np.allclose(np.asarray(jax.grad(lambda x: jnp.sum(ternary_absmean_ste(x)))(w)), 1.0, atol=1e-6)

    x = jax.random.normal(jax.random.PRNGKey(9), (4, 64), dtype=jnp.float32)
    assert float(jnp.mean(jnp.abs(adc_quantize_ste(x, 2) - x))) > 10.0 * float(jnp.mean(jnp.abs(adc_quantize_ste(x, 8) - x)))

    cfg = _tiny_cfg(quant_mode="ternary_158", analog_dev_sigma=0.2, analog_therm_sigma=0.0, analog_adc_bits=6)
    model = HeteroSpectralLM(cfg)
    b = make_synthetic_packed_batch(1, 128, cfg.vocab_size, seed=2, short_len=64)
    params = model.init({"params": jax.random.PRNGKey(0), "analog_noise": jax.random.PRNGKey(1)}, *b, training=False)["params"]
    y_c1a = model.apply({"params": params}, *b, training=False, eval_chip_seed=2001)
    y_c1b = model.apply({"params": params}, *b, training=False, eval_chip_seed=2001)
    y_c2 = model.apply({"params": params}, *b, training=False, eval_chip_seed=2002)
    assert float(jnp.max(jnp.abs(y_c1a - y_c1b))) == 0.0
    assert float(jnp.max(jnp.abs(y_c1a - y_c2))) > 1e-2


def test_muon_wsd_schedule_and_orbax_async_preemption_resume(tmp_path: Path) -> None:
    """Verify Muon advances a zero-start callable schedule AND matches continuous run across async Orbax resume."""
    g = jax.random.normal(jax.random.PRNGKey(3), (32, 64), dtype=jnp.float32)
    q = zeropower_via_newtonschulz5(g, steps=5, use_bf16=False)
    sv = np.linalg.svd(np.asarray(q), compute_uv=False)
    assert 0.5 < float(sv.min()) and float(sv.max()) < 1.3

    # Regression test: zero-at-step-0 schedule MUST produce non-zero Muon update on step 1+
    p_test = {"layer_0": {"q_proj": {"kernel": jnp.ones((16, 16), dtype=jnp.float32)}}}
    zero_start_sched = lambda s: jnp.asarray(s, dtype=jnp.float32) * 1e-3  # noqa: E731
    opt_sched = build_hybrid_muon_adamw(p_test, lr=zero_start_sched, muon_lr=zero_start_sched, weight_decay=0.0, use_bf16_ns=False)
    st_s = opt_sched.init(p_test)
    g_test = {"layer_0": {"q_proj": {"kernel": jax.random.normal(jax.random.PRNGKey(1), (16, 16))}}}
    u0, st_s = opt_sched.update(g_test, st_s, p_test)
    u1, st_s = opt_sched.update(g_test, st_s, p_test)
    assert float(jnp.max(jnp.abs(u0["layer_0"]["q_proj"]["kernel"]))) == 0.0
    assert float(jnp.max(jnp.abs(u1["layer_0"]["q_proj"]["kernel"]))) > 1e-4

    # Full model WSD + async Orbax CheckpointManager preemption resume test
    cfg = _tiny_cfg(quant_mode="ternary_158", analog_dev_sigma=0.05, analog_therm_sigma=0.03, analog_adc_bits=6)
    model = HeteroSpectralLM(cfg)
    batches = [make_synthetic_packed_batch(2, 128, cfg.vocab_size, seed=50 + i, short_len=64) for i in range(5)]
    init_p = model.init({"params": jax.random.PRNGKey(0), "analog_noise": jax.random.PRNGKey(1)}, *batches[0], training=True)["params"]
    sched = build_wsd_schedule(2e-3, total_steps=5, warmup_steps=2, decay_steps=2)
    opt = build_hybrid_muon_adamw(init_p, lr=sched, muon_lr=sched, use_bf16_ns=False)
    init_st = opt.init(init_p)

    @jax.jit
    def step_fn(p, st, key, tok, pos, doc, plan):
        def loss_fn(w):
            logits = model.apply({"params": w}, tok, pos, doc, plan, training=True, rngs={"analog_noise": key})
            return compute_ce_and_zloss(logits, tok, doc_ids=doc)

        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        upd, new_st = opt.update(grads, st, p)
        return optax.apply_updates(p, upd), new_st, loss

    # Verify GCS URI preservation without local Path.resolve() corruption
    assert str(_to_epath("gs://trc-bucket/runs/exp01")) == "gs://trc-bucket/runs/exp01"

    p_int, st_int, rng_int = init_p, init_st, jax.random.PRNGKey(77)
    for b in batches[:3]:
        rng_int, sk = jax.random.split(rng_int)
        p_int, st_int, _ = step_fn(p_int, st_int, sk, *b)

    with make_checkpoint_manager(tmp_path / "ckpts") as mgr:
        save_checkpoint(mgr, step=3, params=p_int, opt_state=st_int, rng_key=rng_int, data_cursor=3)
        mgr.wait_until_finished()
        restored = restore_checkpoint(mgr, abstract_params=init_p, abstract_opt_state=init_st, abstract_rng_key=rng_int)

    p_res, st_res, rng_res = restored["params"], restored["opt_state"], restored["rng_key"]
    assert int(restored["step"]) == 3 and int(restored["data_cursor"]) == 3
    ckpt_diffs = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda a, b: jnp.max(jnp.abs(a - b)), p_int, p_res))
    assert max(float(d) for d in ckpt_diffs) == 0.0
    st_int_leaves, st_int_td = jax.tree_util.tree_flatten(st_int)
    st_res_leaves, st_res_td = jax.tree_util.tree_flatten(st_res)
    assert st_int_td == st_res_td and len(st_int_leaves) == len(st_res_leaves)
    assert max(float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))) for a, b in zip(st_int_leaves, st_res_leaves)) == 0.0
    assert np.array_equal(np.asarray(rng_int), np.asarray(rng_res))

    # Branch both continuous in-memory state and restored checkpoint state through remaining steps 3..4
    p_cont, st_cont, rng_cont = p_int, st_int, rng_int
    for b in batches[3:]:
        rng_cont, sk_c = jax.random.split(rng_cont)
        p_cont, st_cont, loss_cont = step_fn(p_cont, st_cont, sk_c, *b)
        rng_res, sk_r = jax.random.split(rng_res)
        p_res, st_res, loss_res = step_fn(p_res, st_res, sk_r, *b)

    assert abs(float(loss_cont) - float(loss_res)) < 1e-5
    diffs = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda a, b: jnp.max(jnp.abs(a - b)), p_cont, p_res))
    assert max(float(d) for d in diffs) < 1e-5


def test_pytorch_operator_parity_fixture() -> None:
    """Verify <1e-6 FP32 parity against checked-in PyTorch reference outputs (runs in JAX-only CI too)."""
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "pytorch_operator_ref_fp32.npz"
    assert fixture_path.exists(), f"Missing PyTorch reference fixture: {fixture_path}"
    ref = np.load(fixture_path)

    x_in = ref["x_in"]
    w_in = ref["w_in"]
    w_pos = ref["w_pos"]
    w_neg = ref["w_neg"]
    L, K = ref["phi_pos_ref"].shape[1], ref["phi_pos_ref"].shape[0]

    # 1. Hankel spectral filters
    phi_pos_j, phi_neg_j = compute_hankel_spectral_filters(L, num_filters=K)
    assert float(np.max(np.abs(phi_pos_j - ref["phi_pos_ref"]))) < 1e-6
    assert float(np.max(np.abs(phi_neg_j - ref["phi_neg_ref"]))) < 1e-6

    # 2. Contracted causal 1D FFT convolution
    y_fft_j = np.asarray(
        static_doc_isolated_contracted_fft(
            jnp.asarray(x_in),
            jnp.asarray(phi_pos_j),
            jnp.asarray(phi_neg_j),
            jnp.asarray(w_pos),
            jnp.asarray(w_neg),
            doc_plan=None,
        )
    )
    assert float(np.max(np.abs(y_fft_j - ref["y_fft_ref"]))) < 1e-6

    # 3. 1.58-bit ternary absmean STE
    wq_j = np.asarray(ternary_absmean_ste(jnp.asarray(w_in)))
    assert float(np.max(np.abs(wq_j - ref["w_ternary_ref"]))) < 1e-6

    # 4. 6-bit RMS-calibrated ADC STE
    xq_j = np.asarray(adc_quantize_ste(jnp.asarray(x_in), bits=6, clip_sigma=4.0))
    assert float(np.max(np.abs(xq_j - ref["x_adc6_ref"]))) < 1e-6

    # 5. 5-step quintic Newton-Schulz (FP32)
    ns_j = np.asarray(zeropower_via_newtonschulz5(jnp.asarray(w_in), steps=5, use_bf16=False))
    assert float(np.max(np.abs(ns_j - ref["w_ns5_ref"]))) < 1e-6


def test_optional_pytorch_operator_parity() -> None:
    """When PyTorch is installed, directly verify <1e-6 FP32 parity for core operators against PyTorch formulas."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2026)
    w_np = rng.standard_normal((16, 32)).astype(np.float32)
    x_np = rng.standard_normal((2, 64, 16)).astype(np.float32)

    # 1. Ternary absmean STE
    wt = torch.from_numpy(w_np)
    s_t = wt.abs().mean().clamp(min=1e-6)
    wq_t = (torch.clamp(torch.round(wt / s_t), -1.0, 1.0) * s_t).numpy()
    wq_j = np.asarray(ternary_absmean_ste(jnp.asarray(w_np)))
    assert float(np.max(np.abs(wq_j - wq_t))) < 1e-6

    # 2. 6-bit RMS-calibrated ADC STE
    xt = torch.from_numpy(x_np)
    qmax = float((1 << 5) - 1)
    rms_t = torch.sqrt(xt.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    step_t = (rms_t * 4.0).clamp(min=1e-6) / qmax
    xq_t = (torch.clamp(torch.round(xt / step_t), -qmax, qmax) * step_t).numpy()
    xq_j = np.asarray(adc_quantize_ste(jnp.asarray(x_np), bits=6, clip_sigma=4.0))
    assert float(np.max(np.abs(xq_j - xq_t))) < 1e-6

    # 3. 5-step Newton-Schulz
    a, b, c = 3.4445, -4.7750, 2.0315
    X = wt.clone()
    X = X / (X.norm() + 1e-7)
    for _ in range(5):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    ns_j = np.asarray(zeropower_via_newtonschulz5(jnp.asarray(w_np), steps=5, use_bf16=False))
    assert float(np.max(np.abs(ns_j - X.numpy()))) < 1e-6
