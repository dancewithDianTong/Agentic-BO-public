"""Pareto-front and hypervolume utilities (maximization convention).

All functions take objective matrices ``Y`` of shape ``(n, m)`` in which **larger is
better** in every column (lenz negates minimized metrics before calling them).

* :func:`pareto_mask` — non-dominated rows.
* :func:`hypervolume` — exact for ``m <= 3`` (2-D sweep, 3-D slicing), Monte Carlo with a
  seeded generator for ``m > 3``.
* :func:`hvi_2d` — exact, vectorized hypervolume improvement of many candidate points
  over a fixed 2-D front (the inner loop of MC-EHVI).
* :func:`default_ref_point` — ``min(Y) - 0.1 * (max(Y) - min(Y))`` per objective.

Rows containing non-finite values are ignored everywhere: they are never Pareto-optimal,
never dominate another row and never add hypervolume.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils import LenzError, make_rng

__all__ = ["pareto_mask", "hypervolume", "hvi_2d", "default_ref_point"]

# Rows per broadcast block; keeps the (block, n, m) comparison tensors small.
_CHUNK_ELEMS = 1 << 21


def _as_matrix(Y: Any, name: str = "Y", m: int | None = None) -> np.ndarray:
    """``Y`` as a float ``(n, m)`` matrix; a 1-D input is read as ``n`` points with ``m = 1``
    (or as one point when ``m`` is given and matches its length)."""
    A = np.asarray(Y, dtype=float)
    if A.ndim == 1:
        A = A.reshape(1, -1) if (m is not None and A.size == m and m > 1) else A.reshape(-1, 1)
    if A.ndim != 2:
        raise LenzError(f"{name} must be a 2-D array (n, m), got shape {A.shape}")
    if m is not None and A.shape[0] and A.shape[1] != m:
        raise LenzError(f"{name} must have {m} columns, got {A.shape[1]}")
    return A


def _as_ref(ref_point: Any, m: int) -> np.ndarray:
    ref = np.asarray(ref_point, dtype=float).reshape(-1)
    if ref.size != m:
        raise LenzError(f"ref_point must have {m} entries (one per objective), got {ref.size}")
    if not np.all(np.isfinite(ref)):
        raise LenzError(f"ref_point must be finite, got {ref.tolist()}")
    return ref


def pareto_mask(Y: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated rows of ``Y`` (maximize every column).

    Row ``i`` is dominated if some row ``j`` is ``>=`` in every objective and ``>`` in at
    least one. Exact duplicates do not dominate each other, so duplicated points are
    treated consistently: all copies are kept when the point is non-dominated and all
    copies are dropped when it is dominated. Rows with NaN/inf are always ``False`` and
    dominate nothing. ``(n,)`` inputs are read as ``n`` single-objective values.
    """
    A = _as_matrix(Y)
    n, m = A.shape
    mask = np.zeros(n, dtype=bool)
    valid = np.all(np.isfinite(A), axis=1)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return mask
    V = A[idx]
    k = V.shape[0]
    block = max(1, _CHUNK_ELEMS // max(k * m, 1))
    for s in range(0, k, block):
        P = V[s:s + block, None, :]                      # candidates (b, 1, m)
        ge = np.all(V[None, :, :] >= P, axis=2)          # (b, k): V_j >= P_i everywhere
        gt = np.any(V[None, :, :] > P, axis=2)           # strictly better somewhere
        mask[idx[s:s + block]] = ~np.any(ge & gt, axis=1)
    return mask


def _clean_points(Y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Finite rows strictly better than ``ref`` in every objective (the only ones that add volume)."""
    if Y.shape[0] == 0:
        return Y
    keep = np.all(np.isfinite(Y), axis=1) & np.all(Y > ref, axis=1)
    return Y[keep]


def _hv2d(P: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2-D hypervolume of points ``P`` that all strictly dominate ``ref``.

    Works for any such set (dominated points included): sorting by the first objective,
    the staircase height over ``[x_{j-1}, x_j]`` is the max second objective of the
    points with ``x >= x_j``.
    """
    if P.shape[0] == 0:
        return 0.0
    order = np.argsort(P[:, 0], kind="stable")
    x, y = P[order, 0], P[order, 1]
    height = np.maximum.accumulate(y[::-1])[::-1]
    widths = np.diff(np.concatenate(([ref[0]], x)))
    return float(np.sum(widths * (height - ref[1])))


def _hv3d(P: np.ndarray, ref: np.ndarray) -> float:
    """Exact 3-D hypervolume by slicing along the third objective (``O(k^2 log k)``)."""
    order = np.argsort(-P[:, 2], kind="stable")
    P = P[order]
    z_next = np.concatenate((P[1:, 2], [ref[2]]))
    total = 0.0
    for j in range(P.shape[0]):
        depth = P[j, 2] - z_next[j]
        if depth > 0:
            total += _hv2d(P[: j + 1, :2], ref[:2]) * depth
    return float(total)


def _hv_mc(P: np.ndarray, ref: np.ndarray, n_samples: int, rng: Any) -> float:
    """Monte Carlo hypervolume: uniform samples in the box ``[ref, max(P)]``."""
    gen = make_rng(rng)
    hi = P.max(axis=0)
    span = hi - ref
    box = float(np.prod(span))
    if box <= 0:
        return 0.0
    k, m = P.shape
    n_samples = max(int(n_samples), 1)
    block = max(1, _CHUNK_ELEMS // max(k * m, 1))
    hits = 0
    for s in range(0, n_samples, block):
        b = min(block, n_samples - s)
        U = ref + gen.uniform(size=(b, m)) * span
        hits += int(np.count_nonzero(np.any(np.all(P[None, :, :] >= U[:, None, :], axis=2), axis=1)))
    return box * hits / n_samples


def hypervolume(Y: np.ndarray, ref_point: Any, *, n_samples: int = 200_000,
                rng: np.random.Generator | int | None = 0) -> float:
    """Hypervolume dominated by ``Y`` (maximize) and bounded below by ``ref_point``.

    Exact for ``m <= 3`` objectives; for ``m > 3`` a Monte Carlo estimate with
    ``n_samples`` uniform points drawn from ``rng`` (seeded by default, so repeated calls
    agree). Points that do not strictly dominate ``ref_point`` contribute nothing.
    """
    ref = np.asarray(ref_point, dtype=float).reshape(-1)
    A = _as_matrix(Y, m=ref.size if ref.size > 0 else None)
    m = A.shape[1] if A.shape[0] else ref.size
    ref = _as_ref(ref, m)
    P = _clean_points(A, ref)
    if P.shape[0] == 0:
        return 0.0
    if m == 1:
        return float(P.max() - ref[0])
    P = np.unique(P[pareto_mask(P)], axis=0)
    if m == 2:
        return _hv2d(P, ref)
    if m == 3:
        return _hv3d(P, ref)
    return float(_hv_mc(P, ref, n_samples, rng))


def hvi_2d(samples: np.ndarray, front: np.ndarray, ref: Any) -> np.ndarray:
    """Exact hypervolume improvement of each 2-D ``samples`` row over ``front`` (maximize).

    ``HVI(y) = HV(front + {y}) - HV(front)``, computed for all ``N`` rows at once from the
    staircase of ``front``: the strip ``[lo_j, hi_j)`` of the first objective is covered
    by the front up to height ``H_j``, so ``y`` adds
    ``sum_j max(0, min(y_1, hi_j) - lo_j) * max(0, y_2 - H_j)``.
    ``front`` may contain dominated or out-of-box points (they are filtered). Rows of
    ``samples`` that do not dominate ``ref`` get 0.
    """
    S = np.asarray(samples, dtype=float)
    if S.ndim == 1:
        S = S.reshape(1, -1) if S.size == 2 else S
    if S.ndim != 2 or S.shape[1] != 2:
        raise LenzError(f"hvi_2d expects samples of shape (N, 2), got {np.shape(samples)}")
    r = _as_ref(ref, 2)
    F = np.asarray(front, dtype=float)
    F = F.reshape(-1, 2) if F.size else np.zeros((0, 2))
    F = _clean_points(F, r)
    order = np.argsort(F[:, 0], kind="stable")
    fx, fy = F[order, 0], F[order, 1]
    height = np.maximum.accumulate(fy[::-1])[::-1] if fy.size else fy
    lo = np.concatenate(([r[0]], fx))
    hi = np.concatenate((fx, [np.inf]))
    H = np.concatenate((height, [r[1]]))
    N = S.shape[0]
    out = np.empty(N)
    block = max(1, _CHUNK_ELEMS // H.size)
    for s in range(0, N, block):
        y1 = S[s:s + block, 0:1]
        y2 = S[s:s + block, 1:2]
        width = np.clip(np.minimum(y1, hi) - lo, 0.0, None)
        tall = np.clip(y2 - H, 0.0, None)
        out[s:s + block] = np.sum(width * tall, axis=1)
    return out


def default_ref_point(Y: np.ndarray) -> np.ndarray:
    """``min(Y) - 0.1 * (max(Y) - min(Y))`` per objective (maximize convention).

    Non-finite rows are ignored. For an objective whose observed range is zero (e.g. a
    single observation) the offset uses ``max(|min|, 1)`` as the span so the reference
    point still lies strictly below the data.
    """
    A = _as_matrix(Y)
    A = A[np.all(np.isfinite(A), axis=1)]
    if A.shape[0] == 0:
        raise LenzError("default_ref_point needs at least one finite observation")
    lo, hi = A.min(axis=0), A.max(axis=0)
    span = hi - lo
    span = np.where(span > 0, span, np.maximum(np.abs(lo), 1.0))
    return lo - 0.1 * span
