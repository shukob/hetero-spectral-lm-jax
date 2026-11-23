"""TPU/Multi-GPU SPMD & FSDP training harness with per-sequence static XLA steps and async Orbax GCS checkpointing."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from etils import epath
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .config import PRESET_CONFIGS, ModelConfig
from .model import HeteroSpectralLM, compute_ce_and_zloss
from .optim import build_hybrid_muon_adamw, build_wsd_schedule
from .packing import StaticDocPlan, make_synthetic_packed_batch


def _to_epath(path_str: str | Path | epath.Path) -> epath.Path:
    """Convert local or `gs://` URI to an `etils.epath.Path` without breaking GCS URIs via `.resolve()`."""
    s = str(path_str)
    if s.startswith("gs://") or "://" in s:
        return epath.Path(s)
    return epath.Path(Path(s).expanduser().resolve())


def _key_to_raw(rng_key: jax.Array) -> jax.Array:
    if hasattr(jax.random, "key_data") and jnp.issubdtype(rng_key.dtype, jax.dtypes.prng_key):
        return jax.random.key_data(rng_key)
    return rng_key


def make_checkpoint_manager(ckpt_dir: str | Path | epath.Path, *, max_to_keep: int = 2) -> ocp.CheckpointManager:
    """Create an async Orbax CheckpointManager compatible with both local paths and regional `gs://` buckets."""
    root = _to_epath(ckpt_dir)
    if not str(root).startswith("gs://"):
        root.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        enable_async_checkpointing=True,
    )
    return ocp.CheckpointManager(root, options=options)


def save_checkpoint(
    manager: ocp.CheckpointManager,
    *,
    step: int,
    params: Any,
    opt_state: Any,
    rng_key: jax.Array,
    data_cursor: int,
) -> None:
    """Asynchronously enqueue a full training-state checkpoint (params, optimizer, RNG key, data cursor)."""
    payload = {
        "step": jnp.asarray(step, dtype=jnp.int32),
        "data_cursor": jnp.asarray(data_cursor, dtype=jnp.int32),
        "rng_key": _key_to_raw(rng_key),
        "params": params,
        "opt_state": opt_state,
    }
    manager.save(int(step), args=ocp.args.StandardSave(payload), force=True)


def restore_checkpoint(
    manager: ocp.CheckpointManager,
    *,
    abstract_params: Any,
    abstract_opt_state: Any,
    abstract_rng_key: jax.Array,
    step: int | None = None,
) -> dict[str, Any]:
    """Restore full training state from the latest (or requested) Orbax checkpoint."""
    target_step = manager.latest_step() if step is None else int(step)
    if target_step is None:
        raise FileNotFoundError(f"No checkpoints found under {manager.directory}")
    abstract_payload = {
        "step": jnp.asarray(0, dtype=jnp.int32),
        "data_cursor": jnp.asarray(0, dtype=jnp.int32),
        "rng_key": _key_to_raw(abstract_rng_key),
        "params": abstract_params,
        "opt_state": abstract_opt_state,
    }
    return manager.restore(int(target_step), args=ocp.args.StandardRestore(abstract_payload))


def _make_param_sharding(tree: Any, mesh: Mesh, *, fsdp: bool) -> Any:
    """Data-parallel replicated (`P()`) or lightweight 1D FSDP (`P('data', None)` on divisible 2D+ matrices)."""
    n_dev = mesh.devices.size

    def _leaf_sharding(leaf: Any) -> NamedSharding:
        if fsdp and hasattr(leaf, "ndim") and leaf.ndim >= 2 and leaf.shape[0] % n_dev == 0:
            return NamedSharding(mesh, P("data", *((None,) * (leaf.ndim - 1))))
        return NamedSharding(mesh, P())

    return jax.tree_util.tree_map(_leaf_sharding, tree)


def run_benchmark(
    *,
    preset: str = "0.10b",
    arch_mode: str = "hybrid_reservoir",
    analog_hat: bool = False,
    steps: int = 6,
    batch_size: int = 4,
    seq_len: int = 512,
    lr: float = 3.6e-3,
    warmup_steps: int = 2,
    seed: int = 42,
    remat: bool = True,
    fsdp: bool = False,
    dtype: str = "bfloat16",
    ckpt_dir: str | None = None,
    save_every: int = 0,
    resume_from: str | None = None,
) -> dict[str, Any]:
    """Run a single-host SPMD/FSDP static-shape benchmark measuring steady valid targets/s and XLA trace count."""
    base_cfg = PRESET_CONFIGS[preset]
    cfg: ModelConfig = replace(
        base_cfg,
        arch_mode=arch_mode,
        max_position_embeddings=seq_len,
        quant_mode="ternary_158" if analog_hat else "none",
        analog_dev_sigma=0.05 if analog_hat else 0.0,
        analog_therm_sigma=0.03 if analog_hat else 0.0,
        analog_adc_bits=6 if analog_hat else 0,
        analog_scope="except_gqa_attn",
        dtype=dtype,
    )

    devices = jax.devices()
    n_dev = len(devices)
    if batch_size % n_dev != 0:
        batch_size = max(n_dev, ((batch_size + n_dev - 1) // n_dev) * n_dev)
    mesh = Mesh(devices, ("data",))
    data_sharding = NamedSharding(mesh, P("data"))
    replicated = NamedSharding(mesh, P())

    model = HeteroSpectralLM(cfg, remat_layers=remat)
    rng = jax.random.PRNGKey(seed)
    rng, init_key, noise_key = jax.random.split(rng, 3)
    rng = jax.device_put(rng, replicated)

    init_tok, init_pos, init_doc, init_plan = make_synthetic_packed_batch(
        batch_size, seq_len, cfg.vocab_size, seed=seed
    )
    variables = model.init(
        {"params": init_key, "analog_noise": noise_key},
        init_tok,
        init_pos,
        init_doc,
        init_plan,
        training=True,
    )
    params = variables["params"]
    param_sharding = _make_param_sharding(params, mesh, fsdp=fsdp)
    params = jax.device_put(params, param_sharding)

    schedule = build_wsd_schedule(
        peak_lr=lr,
        total_steps=max(steps, 1),
        warmup_steps=min(warmup_steps, max(steps // 4, 0)),
        decay_steps=max(steps // 5, 0),
    )
    optimizer = build_hybrid_muon_adamw(
        params,
        lr=schedule,
        muon_lr=schedule,
        use_bf16_ns=(dtype == "bfloat16"),
    )
    opt_state = optimizer.init(params)
    opt_sharding = _make_param_sharding(opt_state, mesh, fsdp=fsdp)
    opt_state = jax.device_put(opt_state, opt_sharding)

    start_step = 0
    if resume_from is not None:
        with make_checkpoint_manager(resume_from) as rmgr:
            restored = restore_checkpoint(
                rmgr,
                abstract_params=params,
                abstract_opt_state=opt_state,
                abstract_rng_key=rng,
            )
        params = jax.device_put(restored["params"], param_sharding)
        opt_state = jax.device_put(restored["opt_state"], opt_sharding)
        rng = jax.device_put(restored["rng_key"], replicated)
        start_step = int(restored["step"])

    trace_counter = {"count": 0}

    @jax.jit
    def train_step(
        params_in: Any,
        opt_state_in: Any,
        step_rng: jax.Array,
        tokens: jax.Array,
        pos_ids: jax.Array,
        doc_ids: jax.Array,
        doc_plan: StaticDocPlan,
    ) -> tuple[Any, Any, dict[str, jax.Array]]:
        trace_counter["count"] += 1

        def loss_fn(p: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
            logits = model.apply(
                {"params": p},
                tokens,
                pos_ids,
                doc_ids,
                doc_plan,
                training=True,
                rngs={"analog_noise": step_rng},
            )
            return compute_ce_and_zloss(logits, tokens, doc_ids=doc_ids, z_loss_coef=cfg.z_loss_coef)

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_in)
        updates, new_opt_state = optimizer.update(grads, opt_state_in, params_in)
        new_params = optax.apply_updates(params_in, updates)
        return new_params, new_opt_state, metrics

    ckpt_mgr = make_checkpoint_manager(ckpt_dir) if ckpt_dir is not None else None
    step_times: list[float] = []
    valid_toks: list[float] = []
    losses: list[float] = []
    t_compile = 0.0

    try:
        for s in range(start_step, steps):
            # Stream batches on demand (avoids pre-materializing all steps in host RAM)
            tok, pos, doc, plan = make_synthetic_packed_batch(
                batch_size,
                seq_len,
                cfg.vocab_size,
                seed=seed + s + 100,
                min_doc_len=32,
                max_doc_len=min(512, seq_len),
            )
            tok = jax.device_put(tok, data_sharding)
            pos = jax.device_put(pos, data_sharding)
            doc = jax.device_put(doc, data_sharding)
            # Per-sequence StaticDocPlan shards cleanly on P("data") -> zero cross-chip FFT collectives!
            plan = jax.device_put(plan, data_sharding)
            rng, step_rng = jax.random.split(rng)
            step_rng = jax.device_put(step_rng, replicated)

            t0 = time.perf_counter()
            params, opt_state, metrics = train_step(params, opt_state, step_rng, tok, pos, doc, plan)
            ce_val = float(jax.block_until_ready(metrics["ce_loss"]))
            dt = time.perf_counter() - t0
            if s == start_step:
                t_compile = dt
            else:
                step_times.append(dt)
                valid_toks.append(float(metrics["valid_tokens"]))
            losses.append(ce_val)

            if ckpt_mgr is not None and save_every > 0 and ((s + 1) % save_every == 0):
                save_checkpoint(
                    ckpt_mgr,
                    step=s + 1,
                    params=params,
                    opt_state=opt_state,
                    rng_key=rng,
                    data_cursor=s + 1,
                )

        if ckpt_mgr is not None and (save_every <= 0 or (steps % save_every != 0)):
            save_checkpoint(
                ckpt_mgr,
                step=steps,
                params=params,
                opt_state=opt_state,
                rng_key=rng,
                data_cursor=steps,
            )
    finally:
        if ckpt_mgr is not None:
            ckpt_mgr.wait_until_finished()
            ckpt_mgr.close()

    steady_s = sum(step_times)
    steady_tok_s = sum(valid_toks) / max(steady_s, 1e-9) if step_times else 0.0
    pstats = cfg.estimate_params()
    return {
        "backend": jax.default_backend(),
        "devices": [str(d) for d in devices],
        "preset": preset,
        "arch_mode": arch_mode,
        "analog_hat": analog_hat,
        "dtype": dtype,
        "fsdp": fsdp,
        "params_total": pstats["total"],
        "params_non_emb": pstats["non_embedding"],
        "analog_non_emb_fraction": round(pstats["analog_non_emb_fraction"], 4),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "steps": steps,
        "xla_jit_trace_count": trace_counter["count"],
        "step0_compile_and_run_s": round(t_compile, 3),
        "steady_valid_targets_per_s": round(steady_tok_s, 1),
        "first_ce": round(losses[0], 4) if losses else None,
        "last_ce": round(losses[-1], 4) if losses else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="JAX/Flax Hetero-Physical Spectral-Reservoir LM benchmark")
    ap.add_argument("--preset", choices=list(PRESET_CONFIGS.keys()), default="0.10b")
    ap.add_argument("--arch-mode", choices=["transformer", "hybrid_reservoir", "pure_reservoir"], default="hybrid_reservoir")
    ap.add_argument("--analog-hat", action="store_true")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3.6e-3)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-remat", action="store_true")
    ap.add_argument("--fsdp", action="store_true")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--ckpt-dir", type=str, default=None)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--resume-from", type=str, default=None)
    args = ap.parse_args()

    report = run_benchmark(
        preset=args.preset,
        arch_mode=args.arch_mode,
        analog_hat=args.analog_hat,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        remat=not args.no_remat,
        fsdp=args.fsdp,
        dtype=args.dtype,
        ckpt_dir=args.ckpt_dir,
        save_every=args.save_every,
        resume_from=args.resume_from,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
