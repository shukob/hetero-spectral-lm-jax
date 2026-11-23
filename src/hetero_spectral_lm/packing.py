"""Per-sequence offline static-shape document bucketing for zero-collective SPMD XLA FFTs (0.00e+00 leak)."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_dataclass
@dataclass
class StaticDocPlan:
    """Per-sequence constant-shape index plan sharded cleanly along batch axis `B` (`P("data")`).

    Because every sequence in a batch carries its own local indices `[B, M_*, L_*]`, `jax.vmap` over `B`
    executes document-isolated FFTs with ZERO cross-chip collectives under data-parallel / FSDP sharding,
    while keeping all tensor shapes constant so `jax.jit` compiles strictly once.
    """

    short_t: jax.Array      # [B, M_short_seq, L_short] int32 (padding uses trash slot L)
    short_valid: jax.Array  # [B, M_short_seq, L_short] bool
    long_t: jax.Array       # [B, M_long_seq, L_long] int32 (padding uses trash slot L)
    long_valid: jax.Array   # [B, M_long_seq, L_long] bool


def pack_documents_static(
    doc_ids: np.ndarray,
    *,
    short_len: int = 256,
    max_short_per_seq: int | None = None,
    max_long_per_seq: int | None = None,
) -> StaticDocPlan:
    """Convert variable-length packed `doc_ids [B, L]` into a per-sequence constant-shape `StaticDocPlan`.

    - Validates that non-negative `doc_id` values are never reused across non-contiguous segments in a row
      (which would otherwise couple disjoint spans in attention while separating them in FFT).
    - Default `max_long_per_seq = max(L // (short_len + 1), 1)` is a mathematical upper bound on the number
      of segments strictly longer than `short_len` in any sequence of length `L`, so the long bucket can
      never overflow for normal long segments.
    - Padding slots point to trash index `L` (with `valid = False`) so valid writes are conflict-free 1-to-1.
    """
    arr = np.asarray(doc_ids, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D doc_ids [B, L], got shape {arr.shape}")
    B, L = arr.shape
    s_len = min(int(short_len), L)
    # Mathematical upper bound for segments > s_len, plus 1 spill slot if short bucket overflows
    min_long_bound = max(L // (s_len + 1), 1)
    m_long = int(max_long_per_seq) if max_long_per_seq is not None else max(min_long_bound + 1, 2)
    # Tight short-bucket capacity (~2x-4x padding ceiling; excess short docs spill into extra long slots)
    m_short = int(max_short_per_seq) if max_short_per_seq is not None else max(L // 64, 8)

    short_t = np.full((B, m_short, s_len), L, dtype=np.int32)
    short_valid = np.zeros((B, m_short, s_len), dtype=np.bool_)
    long_t = np.full((B, m_long, L), L, dtype=np.int32)
    long_valid = np.zeros((B, m_long, L), dtype=np.bool_)

    for b in range(B):
        row = arr[b]
        seen_ids: set[int] = set()
        si = 0
        li = 0
        i = 0
        while i < L:
            d = int(row[i])
            if d < 0:
                i += 1
                continue
            if d in seen_ids:
                raise ValueError(
                    f"Non-contiguous reuse of doc_id={d} in batch row {b} at pos {i}; "
                    "assign distinct doc_ids to disjoint document segments."
                )
            seen_ids.add(d)
            j = i + 1
            while j < L and int(row[j]) == d:
                j += 1
            seg_len = j - i
            pos = np.arange(i, j, dtype=np.int32)
            if seg_len <= s_len and si < m_short:
                short_t[b, si, :seg_len] = pos
                short_valid[b, si, :seg_len] = True
                si += 1
            elif li < m_long:
                long_t[b, li, :seg_len] = pos
                long_valid[b, li, :seg_len] = True
                li += 1
            else:
                raise ValueError(
                    f"Per-sequence bucket capacity exceeded in row {b} (short={si}/{m_short}, long={li}/{m_long}); "
                    "increase max_short_per_seq/max_long_per_seq or filter micro-segments offline."
                )
            i = j

    return StaticDocPlan(
        short_t=jnp.asarray(short_t),
        short_valid=jnp.asarray(short_valid),
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
    max_short_per_seq: int | None = None,
    max_long_per_seq: int | None = None,
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
        max_short_per_seq=max_short_per_seq,
        max_long_per_seq=max_long_per_seq,
    )
    return jnp.asarray(tokens), jnp.asarray(pos_ids), jnp.asarray(doc_ids), plan
