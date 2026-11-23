"""Hybrid Muon (5-step Newton-Schulz + Moonlight RMS scale) + AdamW optimizer in Optax/JAX."""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


def classify_param_for_muon(path: str, ndim: int) -> str:
    """Deterministic parameter routing: hidden 2D Linear matrices -> Muon, embed -> AdamW decay, 1D/norm/mode -> wd=0."""
    low = path.lower()
    if (
        ndim < 2
        or "norm" in low
        or low.endswith("bias")
        or "mode_weight" in low
        or "conv_weight" in low
    ):
        return "adamw_nodecay"
    if "embed" in low or "lm_head" in low:
        return "adamw_decay"
    if ndim == 2:
        return "muon"
    return "adamw_decay"


def zeropower_via_newtonschulz5(
    g: jax.Array,
    *,
    steps: int = 5,
    eps: float = 1e-7,
    use_bf16: bool = True,
) -> jax.Array:
    """5-step quintic Newton-Schulz matrix orthogonalization (Keller Jordan / Moonlight coefficients)."""
    if g.ndim != 2:
        raise ValueError(f"Muon requires a 2D matrix, got shape={g.shape}")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(jnp.bfloat16 if use_bf16 else jnp.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (jnp.linalg.norm(x) + eps)
    for _ in range(steps):
        mat_a = jnp.matmul(x, x.T)
        mat_b = b * mat_a + c * jnp.matmul(mat_a, mat_a)
        x = a * x + jnp.matmul(mat_b, x)
    if transposed:
        x = x.T
    return x.astype(g.dtype)


class _MuonState(NamedTuple):
    momentum_buffer: Any


def _muon_transform(
    *,
    lr: float | optax.Schedule,
    momentum: float = 0.95,
    weight_decay: float = 0.05,
    ns_steps: int = 5,
    use_bf16_ns: bool = True,
) -> optax.GradientTransformation:
    """Functional Optax GradientTransformation for Muon with Nesterov lookahead + Moonlight RMS scaling."""

    def init_fn(params: Any) -> _MuonState:
        return _MuonState(momentum_buffer=jax.tree_util.tree_map(jnp.zeros_like, params))

    def update_fn(updates: Any, state: _MuonState, params: Any = None) -> tuple[Any, _MuonState]:
        if params is None:
            raise ValueError("Muon requires `params` for decoupled weight decay.")

        def _update_leaf(g: jax.Array, buf: jax.Array, p: jax.Array) -> tuple[jax.Array, jax.Array]:
            gf = g.astype(jnp.float32)
            new_buf = buf * momentum + gf
            nesterov = gf + momentum * new_buf
            ortho = zeropower_via_newtonschulz5(nesterov, steps=ns_steps, use_bf16=use_bf16_ns).astype(jnp.float32)
            m, n = p.shape[0], p.shape[1]
            rms_scale = 0.2 * math.sqrt(float(max(m, n)))
            # Combined update delta u such that p_new = p + u
            step_lr = lr if isinstance(lr, (float, int)) else lr(0)
            delta = -step_lr * (rms_scale * ortho + weight_decay * p.astype(jnp.float32))
            return delta.astype(p.dtype), new_buf

        leaves_u, treedef = jax.tree_util.tree_flatten(updates)
        leaves_b = jax.tree_util.tree_leaves(state.momentum_buffer)
        leaves_p = jax.tree_util.tree_leaves(params)
        out_u, out_b = [], []
        for g, b, p in zip(leaves_u, leaves_b, leaves_p):
            du, nb = _update_leaf(g, b, p)
            out_u.append(du)
            out_b.append(nb)
        return jax.tree_util.tree_unflatten(treedef, out_u), _MuonState(
            momentum_buffer=jax.tree_util.tree_unflatten(treedef, out_b)
        )

    return optax.GradientTransformation(init_fn, update_fn)


def _path_labels(params: Any) -> Any:
    def _label_leaf(path: tuple[Any, ...], leaf: jax.Array) -> str:
        parts = []
        for k in path:
            parts.append(str(getattr(k, "key", k)))
        return classify_param_for_muon(".".join(parts), leaf.ndim)

    return jax.tree_util.tree_map_with_path(_label_leaf, params)


def build_hybrid_muon_adamw(
    params: Any,
    *,
    lr: float = 3.6e-3,
    muon_lr: float = 3.6e-3,
    weight_decay: float = 0.05,
    muon_momentum: float = 0.95,
    ns_steps: int = 5,
    grad_clip_norm: float = 1.0,
    use_bf16_ns: bool = True,
) -> optax.GradientTransformation:
    """Build Hybrid Muon + AdamW Optax chain with global norm clipping (exact XLA-compilable parity)."""
    labels = _path_labels(params)
    transforms = {
        "muon": _muon_transform(
            lr=muon_lr,
            momentum=muon_momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            use_bf16_ns=use_bf16_ns,
        ),
        "adamw_decay": optax.adamw(learning_rate=lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=weight_decay),
        "adamw_nodecay": optax.adamw(learning_rate=lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.0),
    }
    return optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.multi_transform(transforms, labels),
    )
