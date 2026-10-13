"""Acquisition optimization in the encoded space.

* :func:`optimize_acqf` — BoTorch-style multi-start optimization: scrambled Sobol raw
  samples (plus Gaussian perturbations of the best baseline points) are scored in one
  vectorized call, the best ``num_restarts`` start L-BFGS-B (analytic gradients when the
  acquisition has them, batched finite differences otherwise), and the results are
  projected onto representable configurations, re-scored and filtered against
  ``exclude``. One-hot categorical blocks are optimized as a continuous relaxation, then
  projected; for mixed spaces the continuous parameters of the best projected points
  are polished with the discrete ones held fixed.
* :func:`thompson_sample` — argmax of one joint posterior sample over a candidate set.
* :func:`trust_region` — TuRBO-style box around a center point.

All boxes ``(lb, ub)`` are in encoded coordinates (``SearchSpace.encoded_bounds``);
``lb == ub`` marks a fixed dimension. Everything is deterministic given ``rng``.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

from ..space import CategoricalParam, IntParam, OrdinalParam, RealParam, SearchSpace, sobol_points
from ..utils import LenzError, make_rng

__all__ = ["optimize_acqf", "thompson_sample", "trust_region", "project_box"]

_FIXED_TOL = 1e-12        # dims with ub - lb below this are held fixed
_BOX_TOL = 1e-9           # slack when testing box membership of projected values
_PERTURB_SD = 0.05        # sd of baseline perturbations, as a fraction of the box width
_N_BEST_BASELINE = 5      # baseline points perturbed for raw samples
_START_SEPARATION = 1e-3  # min RMS distance between L-BFGS-B starts
_FD_STEP = 1e-6           # finite-difference step (encoded units)
_N_POLISH = 3             # projected points whose continuous dims are re-optimized (mixed spaces)


# --------------------------------------------------------------------------- helpers


def _check_box(space: SearchSpace, lb: Any, ub: Any) -> tuple[np.ndarray, np.ndarray]:
    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    if lb.size != space.D or ub.size != space.D:
        raise LenzError(f"bounds must have {space.D} encoded dimensions, got {lb.size} and {ub.size}")
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
        raise LenzError("encoded bounds must be finite")
    if np.any(lb > ub + _FIXED_TOL):
        raise LenzError(f"encoded bounds need lb <= ub (violated in dims {np.flatnonzero(lb > ub).tolist()})")
    lb = np.clip(lb, 0.0, 1.0)
    ub = np.clip(np.maximum(ub, lb), 0.0, 1.0)
    return lb, ub


def _safe_eval(f: Any, Z: np.ndarray) -> np.ndarray:
    """Vectorized evaluation with NaN mapped to ``-inf`` (never selected)."""
    if Z.shape[0] == 0:
        return np.zeros(0)
    v = np.asarray(f(Z), dtype=float).reshape(-1)
    return np.where(np.isnan(v), -np.inf, v)


def _rms_far(P: np.ndarray, E: np.ndarray | None, tol: float) -> np.ndarray:
    """``True`` for rows of ``P`` whose RMS distance to every row of ``E`` exceeds ``tol``."""
    far = np.ones(P.shape[0], dtype=bool)
    if E is None or E.shape[0] == 0 or P.shape[0] == 0:
        return far
    block = max(1, (1 << 20) // max(E.size, 1))
    for s in range(0, P.shape[0], block):
        d2 = np.mean((P[s:s + block, None, :] - E[None, :, :]) ** 2, axis=2)
        far[s:s + block] = np.min(d2, axis=1) > tol * tol
    return far


def _as_rows(X: Any, D: int, name: str) -> np.ndarray | None:
    """Finite rows of an optional ``(n, D)`` matrix (a single ``(D,)`` row is accepted)."""
    if X is None or np.size(X) == 0:
        return None
    A = np.asarray(X, dtype=float)
    if A.ndim == 1 and A.size == D:
        A = A.reshape(1, D)
    if A.ndim != 2 or A.shape[1] != D:
        raise LenzError(f"{name} must have shape (n, {D}), got {np.shape(X)}")
    A = A[np.all(np.isfinite(A), axis=1)]
    return A if A.shape[0] else None


def _to_unit(p: IntParam | RealParam, v: np.ndarray) -> np.ndarray:
    if p.log_scale:
        return (np.log(v) - math.log(p.lower)) / (math.log(p.upper) - math.log(p.lower))
    return (v - p.lower) / (p.upper - p.lower)


def _from_unit(p: IntParam | RealParam, u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    if p.log_scale:
        return np.exp(math.log(p.lower) + u * (math.log(p.upper) - math.log(p.lower)))
    return p.lower + u * (p.upper - p.lower)


def project_box(space: SearchSpace, Z: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Snap encoded points onto representable configurations inside the box ``[lb, ub]``.

    ``space.project`` rounds integers / ordinals and takes the categorical argmax, but it
    knows nothing about an arbitrary box (e.g. a trust region). This helper

    1. clips into the box and hides categorical choices that are not representable inside
       it (their one-hot vector would leave the box) from the argmax,
    2. applies ``space.project``,
    3. re-snaps integer / ordinal values that rounding pushed out of the box to the
       nearest representable value inside it,
    4. clips into the box. A dimension whose box contains no representable value (a
       box narrower than the integer grid) therefore stays clipped, not representable.
    """
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    single = np.ndim(Z) == 1
    Zc = np.clip(np.atleast_2d(np.asarray(Z, dtype=float)), lb, ub)
    if Zc.shape[0] == 0:
        return Zc
    if space.is_continuous:
        P = Zc
    else:
        masked = Zc.copy()
        for p in space.params:
            if isinstance(p, CategoricalParam):
                sl = space.slices[p.name]
                ok = _allowed_choices(lb[sl], ub[sl])
                if ok.any():
                    masked[:, sl] = np.where(ok, masked[:, sl], -1.0)
        P = space.project(masked)
        for p in space.params:
            if isinstance(p, (IntParam, OrdinalParam)):
                i = space.slices[p.name].start
                out = (P[:, i] < lb[i] - _BOX_TOL) | (P[:, i] > ub[i] + _BOX_TOL)
                if out.any():
                    P[out, i] = _snap_inside(p, Zc[out, i], lb[i], ub[i], P[out, i])
    P = np.clip(P, lb, ub)
    return P[0] if single else P


def _allowed_choices(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Choices ``k`` whose one-hot vector ``e_k`` lies in the box ``[lb, ub]`` of the block."""
    hi_ok = ub >= 1.0 - _BOX_TOL
    lo_bad = lb > _BOX_TOL
    n_lo_bad = int(lo_bad.sum())
    # e_k needs lb_j <= 0 for every j != k, i.e. no "lo_bad" outside position k
    return hi_ok & ((n_lo_bad == 0) | ((n_lo_bad == 1) & lo_bad))


def _snap_inside(p: IntParam | OrdinalParam, z: np.ndarray, lo: float, hi: float,
                 fallback: np.ndarray) -> np.ndarray:
    """Nearest representable encoded value to ``z`` within ``[lo, hi]`` (``fallback`` if none)."""
    if isinstance(p, OrdinalParam):
        grid = np.arange(len(p.values)) / (len(p.values) - 1)
        inside = grid[(grid >= lo - _BOX_TOL) & (grid <= hi + _BOX_TOL)]
        if inside.size == 0:
            return fallback
        return inside[np.argmin(np.abs(z[:, None] - inside[None, :]), axis=1)]
    vlo = math.ceil(float(_from_unit(p, np.array(lo))) - 1e-9)
    vhi = math.floor(float(_from_unit(p, np.array(hi))) + 1e-9)
    if vlo > vhi:
        return fallback
    v = np.clip(np.round(_from_unit(p, z)), vlo, vhi)
    return _to_unit(p, v)


def _select_starts(Z: np.ndarray, vals: np.ndarray, k: int) -> np.ndarray:
    """Top-``k`` rows by value, skipping near-duplicates of already chosen starts."""
    order = np.argsort(-vals, kind="stable")
    order = order[np.isfinite(vals[order])]
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= k:
            break
        if chosen:
            d2 = np.mean((Z[chosen] - Z[i]) ** 2, axis=1)
            if np.min(d2) <= _START_SEPARATION ** 2:
                continue
        chosen.append(int(i))
    return np.asarray(chosen, dtype=int)


def _fd_value_and_grad(acq: Any, x: np.ndarray, idx: np.ndarray,
                       ub: np.ndarray) -> tuple[float, np.ndarray]:
    """Forward differences (backward at the upper bound) in one batched evaluation."""
    k = idx.size
    X = np.repeat(x[None, :], k + 1, axis=0)
    step = np.where(x[idx] + _FD_STEP <= ub[idx], _FD_STEP, -_FD_STEP)
    X[np.arange(1, k + 1), idx] += step
    v = np.asarray(acq(X), dtype=float).reshape(-1)
    return float(v[0]), (v[1:] - v[0]) / step


def _value_scale(vals: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """``(center, scale)`` that put acquisition values on an O(1) scale for L-BFGS-B.

    L-BFGS-B's stopping rules are absolute (projected gradient ``<= 1e-5``) or relative
    to ``max(|f|, 1)``, so raw acquisition units (``ucb``/``pm``/``ei`` of a metric of
    order 1e-5, or of 1e6) would stop it at its start or let it stop early. The scale is
    the spread of ``vals`` (the starting values), falling back to the spread of ``ref``
    (all raw values) and then to the magnitude when the spread is at round-off level.
    """
    v = vals[np.isfinite(vals)]
    r = ref[np.isfinite(ref)]
    if r.size == 0:
        return 0.0, 1.0
    center = float(np.max(v)) if v.size else float(np.max(r))
    mag = float(np.max(np.abs(r)))
    floor = 1e-12 * mag
    for w in (v, r):
        if w.size >= 2:
            spread = float(np.max(w) - np.min(w))
            if math.isfinite(spread) and spread > floor:
                return center, spread
    return center, (mag if math.isfinite(mag) and mag > 0 else 1.0)


def _local_opt(acq: Any, x0: np.ndarray, v0: float, lb: np.ndarray, ub: np.ndarray,
               idx: np.ndarray, maxiter: int, center: float = 0.0,
               scale: float = 1.0) -> tuple[np.ndarray, float]:
    """L-BFGS-B over the dims ``idx`` from ``x0``; returns the best point visited.

    The minimized objective is ``-(acq - center) / scale`` (see :func:`_value_scale`);
    the returned value is in the acquisition's own units.
    """
    best_x, best_v = x0.copy(), float(v0)
    if idx.size == 0 or maxiter <= 0:
        return best_x, best_v
    base = x0.copy()
    use_grad = bool(getattr(acq, "has_gradient", False))

    def fun(u: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal best_x, best_v
        x = base.copy()
        x[idx] = np.clip(u, lb[idx], ub[idx])
        if use_grad:
            v, g = acq.value_and_grad(x[None, :])
            v, g = float(np.asarray(v).reshape(-1)[0]), np.asarray(g, dtype=float).reshape(-1)[idx]
        else:
            v, g = _fd_value_and_grad(acq, x, idx, ub)
        if not math.isfinite(v):
            # a huge finite loss makes the line search backtrack instead of failing
            return 1e100, np.zeros(idx.size)
        if v > best_v:
            best_x, best_v = x, v
        g = np.where(np.isfinite(g), g, 0.0)
        return -(v - center) / scale, -g / scale

    try:
        minimize(fun, x0[idx], jac=True, method="L-BFGS-B",
                 bounds=list(zip(lb[idx], ub[idx])), options={"maxiter": int(maxiter)})
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass  # keep the best point visited so far
    return best_x, best_v


def _raw_candidates(acq_or_score: Any, space: SearchSpace, lb: np.ndarray, ub: np.ndarray,
                    rng: np.random.Generator, n: int, X_baseline: Any,
                    around_frac: float) -> np.ndarray:
    """Sobol points in the box plus Gaussian perturbations of the best baseline points."""
    D = space.D
    Xb = _as_rows(X_baseline, D, "X_baseline")
    if Xb is not None:
        Xb = np.clip(Xb, lb, ub)
    n_around = 0
    if Xb is not None:
        frac = min(max(float(around_frac), 0.0), 1.0)
        n_around = min(int(round(frac * n)), n)
    U = sobol_points(n - n_around, D, rng)
    parts = [lb + U * (ub - lb)]
    if n_around:
        scores = np.asarray(acq_or_score(Xb), dtype=float).reshape(-1)
        scores = np.where(np.isnan(scores), -np.inf, scores)
        best = Xb[np.argsort(-scores, kind="stable")[:_N_BEST_BASELINE]]
        centers = best[np.arange(n_around) % best.shape[0]]
        noise = rng.normal(size=(n_around, D)) * (_PERTURB_SD * (ub - lb))
        parts += [best, np.clip(centers + noise, lb, ub)]
    return np.vstack(parts)


# --------------------------------------------------------------------------- public API


def optimize_acqf(acq: Any, space: SearchSpace, lb: np.ndarray, ub: np.ndarray,
                  rng: np.random.Generator | int | None, *, raw_samples: int = 1024,
                  num_restarts: int = 10, X_baseline: np.ndarray | None = None,
                  around_best_frac: float = 0.2, maxiter: int = 200,
                  exclude: np.ndarray | None = None, exclude_tol: float = 1e-6) -> tuple[np.ndarray, float]:
    """Maximize ``acq`` over the encoded box ``[lb, ub]``.

    Returns ``(z_best, value)``: ``z_best`` ``(D,)`` is projected onto a representable
    configuration (see :func:`project_box`) and lies in ``[lb, ub]``; ``value`` is
    ``acq(z_best)``. Among all projected restart results and raw samples, the best point
    whose RMS encoded distance to every row of ``exclude`` exceeds ``exclude_tol`` is
    returned; if every candidate is excluded the overall best is returned.
    """
    lb, ub = _check_box(space, lb, ub)
    gen = make_rng(rng)
    D = space.D
    E = _as_rows(exclude, D, "exclude")
    n_raw = max(int(raw_samples), 1)
    free = np.flatnonzero(ub - lb > _FIXED_TOL)

    raw = _raw_candidates(acq, space, lb, ub, gen, n_raw, X_baseline, around_best_frac)
    raw_vals = _safe_eval(acq, raw)

    starts = _select_starts(raw, raw_vals, max(int(num_restarts), 0))
    center, scale = _value_scale(raw_vals[starts], raw_vals)
    opt_Z, opt_v = [], []
    for i in starts:
        z, v = _local_opt(acq, raw[i], raw_vals[i], lb, ub, free, maxiter, center, scale)
        opt_Z.append(z)
        opt_v.append(v)

    pool = np.vstack([np.asarray(opt_Z).reshape(-1, D), raw])
    known = np.concatenate([np.asarray(opt_v, dtype=float), raw_vals])
    P = project_box(space, pool, lb, ub)
    changed = np.any(P != pool, axis=1)
    vals = known.copy()
    vals[changed] = _safe_eval(acq, P[changed])

    if not space.is_continuous:
        P, vals = _polish_continuous(acq, space, P, vals, lb, ub, E, exclude_tol, maxiter,
                                     center, scale)

    return _pick(P, vals, E, exclude_tol)


def _polish_continuous(acq: Any, space: SearchSpace, P: np.ndarray, vals: np.ndarray,
                       lb: np.ndarray, ub: np.ndarray, E: np.ndarray | None, tol: float,
                       maxiter: int, center: float = 0.0,
                       scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Re-optimize the real-valued dims of the best projected points, discrete dims fixed."""
    real = np.zeros(space.D, dtype=bool)
    for p in space.params:
        if isinstance(p, RealParam):
            real[space.slices[p.name]] = True
    idx = np.flatnonzero(real & (ub - lb > _FIXED_TOL))
    if idx.size == 0:
        return P, vals
    order = np.argsort(-vals, kind="stable")
    allowed = _rms_far(P[order], E, tol)
    order = np.concatenate([order[allowed], order[~allowed]])
    picked: list[int] = []
    for i in order:
        if len(picked) >= _N_POLISH or not np.isfinite(vals[i]):
            break
        if any(np.array_equal(P[i], P[j]) for j in picked):
            continue
        picked.append(int(i))
    new_Z, new_v = [], []
    for i in picked:
        z, v = _local_opt(acq, P[i], vals[i], lb, ub, idx, maxiter, center, scale)
        new_Z.append(np.clip(z, lb, ub))
        new_v.append(v)
    if not new_Z:
        return P, vals
    return np.vstack([np.asarray(new_Z), P]), np.concatenate([np.asarray(new_v), vals])


def _pick(P: np.ndarray, vals: np.ndarray, E: np.ndarray | None, tol: float) -> tuple[np.ndarray, float]:
    order = np.argsort(-vals, kind="stable")
    far = _rms_far(P[order], E, tol)
    i = order[int(np.argmax(far))] if far.any() else order[0]
    return P[i].copy(), float(vals[i])


def thompson_sample(model: Any, space: SearchSpace, lb: np.ndarray, ub: np.ndarray,
                    rng: np.random.Generator | int | None, *, n_candidates: int = 1024,
                    X_baseline: np.ndarray | None = None, exclude: np.ndarray | None = None,
                    exclude_tol: float = 1e-6,
                    constraint_models: Sequence | None = None) -> tuple[np.ndarray, float]:
    """Thompson sampling: argmax of one joint posterior sample over a candidate set.

    Candidates are scrambled Sobol points in ``[lb, ub]`` plus perturbations of the
    baseline points with the highest posterior mean, projected (:func:`project_box`),
    de-duplicated, and filtered against ``exclude`` (unless that removes all of them).
    Returns ``(z, sampled value)`` (maximize convention, the model's units). With
    ``constraint_models`` (``(model, lower, upper)`` triples) each constraint is sampled
    jointly too and the best sample among sampled-feasible candidates wins (minimum total
    violation if none is feasible), as in scalable constrained BO (Eriksson & Poloczek 2021).
    """
    lb, ub = _check_box(space, lb, ub)
    gen = make_rng(rng)
    n = max(int(n_candidates), 1)

    def post_mean(X: np.ndarray) -> np.ndarray:
        return np.asarray(model.posterior(X)[0], dtype=float).reshape(-1)

    C = project_box(space, _raw_candidates(post_mean, space, lb, ub, gen, n, X_baseline, 0.2), lb, ub)
    _, first = np.unique(C, axis=0, return_index=True)
    C = C[np.sort(first)]
    E = _as_rows(exclude, space.D, "exclude")
    far = _rms_far(C, E, exclude_tol)
    if far.any():
        C = C[far]
    f = np.asarray(model.sample(C, 1, gen), dtype=float).reshape(-1)
    f = np.where(np.isnan(f), -np.inf, f)
    if constraint_models:
        violation = np.zeros(C.shape[0])
        for item in constraint_models:
            cm, lo, hi = item
            c = np.asarray(cm.sample(C, 1, gen), dtype=float).reshape(-1)
            if lo is not None:
                violation += np.maximum(float(lo) - c, 0.0)
            if hi is not None:
                violation += np.maximum(c - float(hi), 0.0)
        violation = np.where(np.isnan(violation), np.inf, violation)
        feasible = violation <= 0
        i = int(np.argmax(np.where(feasible, f, -np.inf))) if feasible.any() else int(np.argmin(violation))
    else:
        i = int(np.argmax(f))
    return C[i].copy(), float(f[i])


def trust_region(center_z: np.ndarray, radius: float, lb: np.ndarray, ub: np.ndarray,
                 lengthscales: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Box of half-width ``radius * w_i`` around ``center_z``, intersected with ``[lb, ub]``.

    ``w_i = l_i / geomean(l)`` clipped to ``[0.25, 4]`` when ``lengthscales`` are given
    (TuRBO: longer lengthscale = less sensitive = wider), else ``w_i = 1``. The center is
    first clipped into ``[lb, ub]`` so the result is never empty and always contains it.
    """
    c = np.asarray(center_z, dtype=float).reshape(-1)
    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    if not (c.size == lb.size == ub.size):
        raise LenzError(f"trust_region: center ({c.size}) and bounds ({lb.size}, {ub.size}) must have the same length")
    try:
        r = float(radius)
    except (TypeError, ValueError):
        raise LenzError(f"trust region radius must be a number, got {radius!r}") from None
    if not (math.isfinite(r) and r >= 0):
        raise LenzError(f"trust region radius must be a finite number >= 0, got {radius!r}")
    if np.any(lb > ub):
        raise LenzError("trust_region: need lb <= ub")
    if lengthscales is None:
        half = np.full(c.size, r)
    else:
        ls = np.asarray(lengthscales, dtype=float).reshape(-1)
        if ls.size != c.size or not np.all(np.isfinite(ls)) or np.any(ls <= 0):
            raise LenzError(f"trust_region: need {c.size} positive finite lengthscales")
        w = ls / math.exp(float(np.mean(np.log(ls))))
        half = r * np.clip(w, 0.25, 4.0)
    c = np.clip(c, lb, ub)
    return np.maximum(lb, c - half), np.minimum(ub, c + half)
