"""Regret curves and run summaries (paper §6; DESIGN.md §8).

All curves are computed from **noise-free** values in the maximization convention
(``utility = -value`` for minimized objectives) and are indexed by evaluation: entry
``t - 1`` describes the run after ``t`` evaluations.

Conventions for infeasible points (constraints violated) and failed evaluations
(non-finite utility, treated as infeasible):

* simple regret: ``r_t = u* - max_{s<=t, feasible} u_s``; before the first feasible
  point, the regret of the worst point so far, ``max(max_{s<=t} (u* - u_s), 0)``;
* cumulative regret: feasible points are charged ``u* - u_s``; infeasible points are
  charged ``max(u* - u_s, max regret seen so far, 0)``, so violating constraints is never
  cheaper than the worst behaviour observed (and an infeasible point that beats the
  constrained optimum is never negative regret); failed evaluations before the first
  finite observation are charged that observation's worst-regret term;
* hypervolume regret: ``max_hv`` minus the hypervolume of the feasible points so far
  (``max_hv`` before any useful point, including when there is no evaluation at all).

Early stopping (``budget`` larger than the number of evaluations): simple and
hypervolume regret stay at their last value; cumulative regret charges every unused
evaluation the regret of the best feasible point observed (as if the agent had
exploited it), or the max regret seen if no feasible point exists, so stopping cannot
game ``R_T``. Entries that cannot be defined (no finite observation yet, unknown
optimum) are NaN.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from .utils import LenzError

__all__ = ["simple_regret_curve", "cumulative_regret_curve", "hypervolume_regret_curve",
           "best_so_far_curve", "dominated_hypervolume", "summarize_curves"]


# --------------------------------------------------------------------------- helpers


def _prepare(utilities: Sequence[float] | np.ndarray, feasible: Sequence[bool] | np.ndarray | None
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(u, finite, valid)``: utilities, finite mask, and feasible-and-finite mask."""
    u = np.asarray(utilities, dtype=float).reshape(-1)
    finite = np.isfinite(u)
    if feasible is None:
        feas = np.ones(u.shape, dtype=bool)
    else:
        feas = np.asarray(feasible, dtype=bool).reshape(-1)
        if feas.shape != u.shape:
            raise LenzError(f"feasible has {feas.size} entries but there are {u.size} utilities")
    return u, finite, feas & finite


def _check_budget(n: int, budget: int | None) -> int:
    if budget is None:
        return n
    budget = int(budget)
    if budget < n:
        raise LenzError(f"budget {budget} is smaller than the number of evaluations {n}")
    return budget


def _running_max(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Running max of ``values`` over entries where ``mask`` holds (-inf before the first one)."""
    if values.size == 0:
        return values.copy()
    return np.maximum.accumulate(np.where(mask, values, -np.inf))


def _pad_last(curve: np.ndarray, length: int) -> np.ndarray:
    if curve.size >= length:
        return curve
    fill = curve[-1] if curve.size else np.nan
    return np.concatenate([curve, np.full(length - curve.size, fill)])


# --------------------------------------------------------------------------- single objective


def best_so_far_curve(utilities: Sequence[float] | np.ndarray, feasible: Sequence[bool] | np.ndarray | None = None,
                      *, budget: int | None = None) -> np.ndarray:
    """Best feasible utility after each evaluation (NaN before the first feasible point).

    Useful when the optimum is unknown; padded with the last value up to ``budget``.
    """
    u, _, valid = _prepare(utilities, feasible)
    total = _check_budget(u.size, budget)
    best = _running_max(u, valid)
    best = np.where(np.isfinite(best), best, np.nan)
    return _pad_last(best, total)


def simple_regret_curve(utilities: Sequence[float] | np.ndarray, u_star: float | None,
                        feasible: Sequence[bool] | np.ndarray | None = None, *,
                        budget: int | None = None) -> np.ndarray:
    """Simple regret ``r_t = u* - max_{s<=t} u_s`` over feasible points (see module docstring).

    Args:
        utilities: per-evaluation utilities (maximization convention, noise-free).
        u_star: optimal utility; ``None`` gives an all-NaN curve.
        feasible: per-evaluation feasibility (default: all feasible).
        budget: pad an early-stopped run to this length (last value repeated).
    """
    u, finite, valid = _prepare(utilities, feasible)
    total = _check_budget(u.size, budget)
    if u_star is None or u.size == 0:
        return np.full(total, np.nan)
    u_star = float(u_star)
    best = _running_max(u, valid)
    # before the first feasible point; an infeasible point better than u* is not negative regret
    worst_regret = np.maximum(_running_max(u_star - u, finite), 0.0)
    r = np.where(np.isfinite(best), u_star - best, np.where(finite.cumsum() > 0, worst_regret, np.nan))
    return _pad_last(r, total)


def cumulative_regret_curve(utilities: Sequence[float] | np.ndarray, u_star: float | None,
                            feasible: Sequence[bool] | np.ndarray | None = None, *,
                            budget: int | None = None) -> np.ndarray:
    """Cumulative regret ``R_t = sum_{s<=t} rho_s`` with the infeasible / early-stop charges above.

    ``rho_s = u* - u_s`` for feasible points and ``max(u* - u_s, max_{s'<=s} (u* - u_s'), 0)``
    for infeasible / failed ones. Failed evaluations before the first finite observation
    ``k`` are charged ``rho_k``'s worst-regret term; ``R_t`` is NaN for ``t < k`` (nothing to
    measure yet) and defined from ``k`` on. With ``budget > len(utilities)`` every unused
    evaluation is charged the regret of the best feasible point observed.
    """
    u, finite, valid = _prepare(utilities, feasible)
    total = _check_budget(u.size, budget)
    first = int(np.argmax(finite)) if finite.any() else None
    if u_star is None or first is None:
        return np.full(total, np.nan)
    u_star = float(u_star)
    raw = u_star - u
    # worst regret seen so far (current finite point included), never below 0: violating
    # constraints is never cheaper than the worst behaviour observed, nor than the optimum
    charge = np.maximum(_running_max(raw, finite), 0.0)
    charge[:first] = charge[first]             # failures before any finite observation
    rho = np.where(valid, raw, charge)
    if total > u.size:
        best = _running_max(u, valid)
        pad = u_star - best[-1] if np.isfinite(best[-1]) else charge[-1]
        rho = np.concatenate([rho, np.full(total - u.size, pad)])
    out = np.cumsum(rho)
    out[:first] = np.nan
    return out


# --------------------------------------------------------------------------- hypervolume


def _hv_max(Y: np.ndarray, ref: np.ndarray) -> float:
    """Exact hypervolume of ``Y`` (maximization) above ``ref``; recursive slicing for m >= 3."""
    if Y.size == 0:
        return 0.0
    Y = Y[np.all(Y > ref, axis=1)]
    if Y.shape[0] == 0:
        return 0.0
    m = Y.shape[1]
    if m == 1:
        return float(Y[:, 0].max() - ref[0])
    if m == 2:
        order = np.argsort(-Y[:, 0], kind="stable")
        y0, y1 = Y[order, 0], Y[order, 1]
        top = np.maximum.accumulate(y1)
        prev = np.concatenate([[ref[1]], top[:-1]])
        return float(np.sum((y0 - ref[0]) * (top - prev)))
    order = np.argsort(-Y[:, -1], kind="stable")
    Y = Y[order]
    z = np.concatenate([Y[:, -1], [ref[-1]]])
    total = 0.0
    for i in range(Y.shape[0]):
        h = z[i] - z[i + 1]
        if h > 0:
            total += h * _hv_max(Y[: i + 1, :-1], ref[:-1])
    return total


def _signs(minimize_flags: Sequence[bool], m: int) -> np.ndarray:
    flags = np.asarray(minimize_flags, dtype=bool).reshape(-1)
    if flags.size != m:
        raise LenzError(f"minimize_flags needs {m} entries, got {flags.size}")
    return np.where(flags, -1.0, 1.0)


def dominated_hypervolume(Y: np.ndarray, ref_point: Sequence[float],
                          minimize_flags: Sequence[bool] | None = None) -> float:
    """Exact hypervolume dominated by the rows of ``Y`` (raw units) w.r.t. ``ref_point``.

    ``minimize_flags[j]`` says objective ``j`` is minimized (default: all maximized).
    Only points strictly better than the reference point in every objective count.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    ref = np.asarray(ref_point, dtype=float).reshape(-1)
    m = ref.size
    if Y.size and Y.shape[1] != m:
        raise LenzError(f"Y has {Y.shape[1]} objectives but ref_point has {m}")
    sign = _signs([False] * m if minimize_flags is None else minimize_flags, m)
    Y = Y.reshape(-1, m)
    Y = Y[np.all(np.isfinite(Y), axis=1)]
    return _hv_max(Y * sign, ref * sign)


def hypervolume_regret_curve(Y_true: np.ndarray, feasible: Sequence[bool] | np.ndarray | None,
                             ref_point: Sequence[float], max_hv: float | None,
                             minimize_flags: Sequence[bool], *, budget: int | None = None) -> np.ndarray:
    """``max_hv - HV_t`` where ``HV_t`` is the hypervolume of the feasible points among the first ``t``.

    Args:
        Y_true: ``(n, m)`` noise-free objective values in raw units.
        feasible: ``(n,)`` feasibility (``None`` = all feasible); rows with non-finite
            values never count.
        ref_point: reference point in raw units.
        max_hv: hypervolume of the true Pareto front (``None`` gives an all-NaN curve).
        minimize_flags: per-objective raw direction.
        budget: pad an early-stopped run to this length (last value repeated).
    """
    ref = np.asarray(ref_point, dtype=float).reshape(-1)
    m = ref.size
    Y = np.asarray(Y_true, dtype=float).reshape(-1, m)
    n = Y.shape[0]
    total = _check_budget(n, budget)
    if feasible is None:
        feas = np.ones(n, dtype=bool)
    else:
        feas = np.asarray(feasible, dtype=bool).reshape(-1)
        if feas.size != n:
            raise LenzError(f"feasible has {feas.size} entries but Y_true has {n} rows")
    if max_hv is None:
        return np.full(total, np.nan)
    if n == 0:      # nothing evaluated dominates nothing: HV = 0 (an early stop at t = 0)
        return np.full(total, float(max_hv))
    sign = _signs(minimize_flags, m)
    Ym, refm = Y * sign, ref * sign
    useful = feas & np.all(np.isfinite(Ym), axis=1) & np.all(Ym > refm, axis=1)

    hv = np.zeros(n)
    front = np.zeros((0, m))
    current = 0.0
    for t in range(n):
        if useful[t]:
            y = Ym[t]
            # recompute only when the new point is not weakly dominated by the current front
            if not np.any(np.all(front >= y, axis=1)):
                front = np.vstack([front[~np.all(y >= front, axis=1)], y])
                current = _hv_max(front, refm)
        hv[t] = current
    return _pad_last(float(max_hv) - hv, total)


# --------------------------------------------------------------------------- aggregation


def summarize_curves(curves: Sequence[np.ndarray], pad: str = "last") -> dict:
    """Pointwise statistics across runs: ``{"median", "q25", "q75", "mean", "sem"}`` (lists of floats).

    Curves of different lengths are padded to the longest one with their last value
    (``pad="last"``, the early-stop convention for simple / hypervolume regret) or NaN
    (``pad="nan"``: later points summarize only the runs that reached them). NaN entries
    are ignored; ``sem`` is 0 where fewer than two runs contribute.
    """
    keys = ("median", "q25", "q75", "mean", "sem")
    arrays = [np.asarray(c, dtype=float).reshape(-1) for c in curves]
    if not arrays:
        return {k: [] for k in keys}
    if pad not in ("last", "nan"):
        raise LenzError(f"pad must be 'last' or 'nan', got {pad!r}")
    length = max(a.size for a in arrays)
    rows = []
    for a in arrays:
        if pad == "last":
            rows.append(_pad_last(a, length))
        else:
            rows.append(np.concatenate([a, np.full(length - a.size, np.nan)]))
    M = np.vstack(rows) if length else np.zeros((len(rows), 0))
    count = np.sum(np.isfinite(M), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN columns -> NaN
        median = np.nanmedian(M, axis=0)
        q25, q75 = np.nanpercentile(M, [25, 75], axis=0)
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0, ddof=1) if M.shape[0] > 1 else np.zeros(length)
    sem = np.where(count >= 2, std / np.sqrt(np.maximum(count, 1)), 0.0)
    return {
        "median": median.tolist(), "q25": q25.tolist(), "q75": q75.tolist(),
        "mean": mean.tolist(), "sem": sem.tolist(),
    }
