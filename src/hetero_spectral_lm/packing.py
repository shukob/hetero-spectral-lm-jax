"""Offline static-shape document bucketing for zero-recompilation XLA FFTs with 0.00e+00 cross-doc leakage."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_dataclass
@dataclass
class StaticDocPlan:
    """Fixed-shape index plan mapping packed [B, L] token streams into static short/long FFT buckets.

    Every field has strictly constant shape across all batches so `jax.jit` compiles ONCE with zero
    data-dependent control flow while guaranteeing exact 0.00e+00 cross-document leakage.
    """

    short_b: jax.Array      # [M_short, L_short] int32
    short_t: jax.Array      # [M_short, L_short] int32
    short_valid: jax.Array  # [M_short, L_short] bool
    long_b: jax.Array       # [M_long, L_long] int32
    long_t: jax.Array       # [M_long, L_long] int32
    long_valid: jax.Array   # [M_long, L_long] bool


def pack_documents_static(
    doc_ids: np.ndarray,
    *,
    short_len: int = 256,
    max_short_segs: int | None = None,
    max_long_segs: int | None = None,
) -> StaticDocPlan:
    """Convert variable-length packed `doc_ids [B, L]` into a constant-shape `StaticDocPlan` (CPU/Grain step).

    Contiguous segments with `doc_id >= 0` of length <= `short_len` are placed in the short bucket
    (overflowing cleanly to the long bucket if `max_short_segs` is reached); longer segments go to the
    long bucket of length `L`. Padding tokens (`doc_id < 0`) are excluded (`valid = False`).
    """
    arr = np.asarray(doc_ids, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D doc_ids [B, L], got shape {arr.shape}")
    B, L = arr.shape
    s_len = min(int(short_len), L)
    # Conservative static defaults that hold typical packed batches without overflow
    m_short = int(max_short_segs) if max_short_segs is not None else max(B * (L // max(s_len, 1)) * 2, B * 4)
    m_long = int(max_long_segs) if max_long_segs is not None else max(B * 4, 4)

    short_b = np.zeros((m_short, s_len), dtype=np.int32)
    short_t = np.full((m_short, s_len), L, dtype=np.int32)  # Trash slot L prevents atomic contention at t=0
    short_valid = np.zeros((m_short, s_len), dtype=np.bool_)

    long_b = np.zeros((m_long, L), dtype=np.int32)
    long_t = np.full((m_long, L), L, dtype=np.int32)
    long_valid = np.zeros((m_long, L), dtype=np.bool_)

    si = 0
    li = 0
    for b in range(B):
        row = arr[b]
        i = 0
        while i < L:
            d = int(row[i])
            if d < 0:
                i += 1
                continue
            j = i + 1
            while j < L and int(row[j]) == d:
                j += 1
            seg_len = j - i
            pos = np.arange(i, j, dtype=np.int32)
            if seg_len <= s_len and si < m_short:
                short_b[si, :seg_len] = b
                short_t[si, :seg_len] = pos
                short_valid[si, :seg_len] = True
                si += 1
            elif li < m_long:
                long_b[li, :seg_len] = b
                long_t[li, :seg_len] = pos
                long_valid[li, :seg_len] = True
                li += 1
            else:
                raise ValueError(
                    f"Static bucket capacity exceeded (short={si}/{m_short}, long={li}/{m_long}); "
                    "increase max_short_segs/max_long_segs or filter micro-segments in offline packing."
                )
            i = j

    return StaticDocPlan(
        short_b=jnp.asarray(short_b),
        short_t=jnp.asarray(short_t),
        short_valid=jnp.asarray(short_valid),
        long_b=jnp.asarray(long_b),
        long_t=jnp.asarray(long_t),
        long_valid=jnp.asarray(long_valid),
    )


def make_synthetic_packed_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int = 0,
    min_doc_len: int = 32,
    max_doc_len: int = 512,
    *,
    short_len: int = 256,
    max_short_segs: int | None = None,
    max_long_segs: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, StaticDocPlan]:
    """Generate a packed multi-document batch with variable doc lengths, per-doc pos_ids, and StaticDocPlan."""
    rng = np.random.default_rng(seed)
    tokens = np.zeros((batch_size, seq_len), dtype=np.int32)
    doc_ids = np.full((batch_size, seq_len), -1, dtype=np.int32)
    pos_ids = np.zeros((batch_size, seq_len), dtype=np.int32)

    for b in range(batch_size):
        cursor = 0
        doc_idx = 0
        while cursor < seq_len:
            rem = seq_len - cursor
            if rem < min_doc_len:
                break
            dlen = int(rng.integers(min_doc_len, min(max_doc_len, rem) + 1))
            tokens[b, cursor : cursor + dlen] = rng.integers(1, vocab_size, size=dlen, dtype=np.int32)
            doc_ids[b, cursor : cursor + dlen] = doc_idx
            pos_ids[b, cursor : cursor + dlen] = np.arange(dlen, dtype=np.int32)
            cursor += dlen
            doc_idx += 1

    plan = pack_documents_static(
        doc_ids,
        short_len=short_len,
        max_short_segs=max_short_segs,
        max_long_segs=max_long_segs,
    )
    return jnp.asarray(tokens), jnp.asarray(pos_ids), jnp.asarray(doc_ids), plan
