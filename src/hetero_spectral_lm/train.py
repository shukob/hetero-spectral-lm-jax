"""TPU/Multi-GPU SPMD training harness with static-shape XLA steps and Orbax preemption recovery."""

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
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .config import PRESET_CONFIGS, ModelConfig
from .model import HeteroSpectralLM, compute_ce_and_zloss
from .optim import build_hybrid_muon_adamw
from .packing import StaticDocPlan, make_synthetic_packed_batch


def save_checkpoint(
    ckpt_dir: Path,
    *,
    step: int,
    params: Any,
    opt_state: Any,
    rng_key: jax.Array,
    data_cursor: int,
) -> Path:
    """Save full training state (params, optimizer, RNG key, data cursor) with Orbax."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target = (ckpt_dir / f"step_{step:07d}").resolve()
    payload = {
        "step": jnp.asarray(step, dtype=jnp.int32),
        "data_cursor": jnp.asarray(data_cursor, dtype=jnp.int32),
        "rng_key": jax.random.key_data(rng_key) if hasattr(jax.random, "key_data") and jnp.issubdtype(rng_key.dtype, jax.dtypes.prng_key) else rng_key,
        "params": params,
        "opt_state": opt_state,
    }
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(target, payload, force=True)
    return target


def restore_checkpoint(
    ckpt_path: Path,
    *,
    abstract_params: Any,
    abstract_opt_state: Any,
    abstract_rng_key: jax.Array,
) -> dict[str, Any]:
    """Restore full training state from an Orbax checkpoint for exact preemption recovery."""
    rng_ref = (
        jax.random.key_data(abstract_rng_key)
        if hasattr(jax.random, "key_data") and jnp.issubdtype(abstract_rng_key.dtype, jax.dtypes.prng_key)
        else abstract_rng_key
    )
    abstract_payload = {
        "step": jnp.asarray(0, dtype=jnp.int32),
        "data_cursor": jnp.asarray(0, dtype=jnp.int32),
        "rng_key": rng_ref,
        "params": abstract_params,
        "opt_state": abstract_opt_state,
    }
    with ocp.StandardCheckpointer() as checkpointer:
        return checkpointer.restore(ckpt_path.resolve(), target=abstract_payload)


def run_benchmark(
    *,
    preset: str = "0.10b",
    arch_mode: str = "hybrid_reservoir",
    analog_hat: bool = False,
    steps: int = 10,
    batch_size: int = 4,
    seq_len: int = 512,
    lr: float = 3.6e-3,
    seed: int = 42,
    remat: bool = True,
    dtype: str = "bfloat16",
    ckpt_dir: str | None = None,
) -> dict[str, Any]:
    """Run a static-shape SPMD training smoke benchmark measuring steady non-padding tokens/s."""
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
    replicated = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P("data"))

    model = HeteroSpectralLM(cfg, remat_layers=remat)
    rng = jax.random.PRNGKey(seed)
    rng, init_key, noise_key = jax.random.split(rng, 3)

    init_tok, init_pos, init_doc, init_plan = make_synthetic_packed_batch(
        batch_size, seq_len, cfg.vocab_size, seed=seed, short_len=min(256, seq_len)
    )
    variables = model.init(
        {"params": init_key, "analog_noise": noise_key},
        init_tok,
        init_pos,
        init_doc,
        init_plan,
        training=True,
    )
    params = jax.device_put(variables["params"], replicated)
    optimizer = build_hybrid_muon_adamw(params, lr=lr, muon_lr=lr)
    opt_state = jax.device_put(optimizer.init(params), replicated)

    compile_counter = {"count": 0}

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
        compile_counter["count"] += 1

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

    # Pre-generate batches with varying doc counts/lengths to verify zero XLA recompiles
    batches = [
        make_synthetic_packed_batch(
            batch_size,
            seq_len,
            cfg.vocab_size,
            seed=seed + s + 100,
            min_doc_len=32,
            max_doc_len=min(512, seq_len),
            short_len=min(256, seq_len),
        )
        for s in range(steps)
    ]

    step_times: list[float] = []
    valid_toks: list[float] = []
    losses: list[float] = []
    t_compile = 0.0

    for s, (tok, pos, doc, plan) in enumerate(batches):
        tok = jax.device_put(tok, data_sharding)
        pos = jax.device_put(pos, data_sharding)
        doc = jax.device_put(doc, data_sharding)
        plan = jax.device_put(plan, replicated)
        rng, step_rng = jax.random.split(rng)

        t0 = time.perf_counter()
        params, opt_state, metrics = train_step(params, opt_state, step_rng, tok, pos, doc, plan)
        ce_val = float(jax.block_until_ready(metrics["ce_loss"]))
        dt = time.perf_counter() - t0
        if s == 0:
            t_compile = dt
        else:
            step_times.append(dt)
            valid_toks.append(float(metrics["valid_tokens"]))
        losses.append(ce_val)

    if ckpt_dir is not None:
        save_checkpoint(Path(ckpt_dir), step=steps, params=params, opt_state=opt_state, rng_key=rng, data_cursor=steps)

    steady_s = sum(step_times)
    steady_tok_s = sum(valid_toks) / max(steady_s, 1e-9) if step_times else 0.0
    pstats = cfg.estimate_params()
    return {
        "backend": jax.default_backend(),
        "devices": [str(d) for d in devices],
        "preset": preset,
        "arch_mode": arch_mode,
        "analog_hat": analog_hat,
        "params_total": pstats["total"],
        "params_non_emb": pstats["non_embedding"],
        "analog_non_emb_fraction": round(pstats["analog_non_emb_fraction"], 4),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "steps": steps,
        "xla_compile_count": compile_counter["count"],
        "step0_compile_and_run_s": round(t_compile, 3),
        "steady_non_pad_tok_per_s": round(steady_tok_s, 1),
        "first_ce": round(losses[0], 4),
        "last_ce": round(losses[-1], 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="JAX/Flax Hetero-Physical Spectral-Reservoir LM benchmark")
    ap.add_argument("--preset", choices=list(PRESET_CONFIGS.keys()), default="0.10b")
    ap.add_argument("--arch-mode", choices=["transformer", "hybrid_reservoir", "pure_reservoir"], default="hybrid_reservoir")
    ap.add_argument("--analog-hat", action="store_true")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3.6e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-remat", action="store_true")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--ckpt-dir", type=str, default=None)
    args = ap.parse_args()

    report = run_benchmark(
        preset=args.preset,
        arch_mode=args.arch_mode,
        analog_hat=args.analog_hat,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        seed=args.seed,
        remat=not args.no_remat,
        dtype=args.dtype,
        ckpt_dir=args.ckpt_dir,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
