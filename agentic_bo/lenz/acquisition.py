"""Acquisition functions on encoded points, with analytic gradients.

Every acquisition is built by :func:`build_acquisition` from **sign-adjusted** surrogate
models (larger objective = better) and evaluated on encoded points ``Z`` of shape
``(m, D)``. A model is any object exposing

* ``posterior(Xs, full_cov=False, return_grad=False, observation_noise=False)`` returning
  ``(mean, var)`` or, with ``return_grad=True``, ``(mean, var, dmean, dvar)`` with
  ``dmean, dvar`` of shape ``(m, D)``; values in the metric's original units,
* ``X_train`` ``(n, D)`` and ``y_train`` ``(n,)`` (used for defaults and scaling),
* ``sample(Xs, n_samples, rng)`` (only Thompson sampling, see ``optimize.py``).

Single-objective acquisitions (``logei`` and friends) are analytic with exact gradients;
``logei`` uses the numerically stable ``log_h`` of Ament et al. (2023, "Unexpected
Improvements to Expected Improvement"), so values *and* gradients stay finite and
informative far from the incumbent. Outcome constraints multiply in the probability of
feasibility (``PoF``) in the log domain. Multi-objective ``ehvi`` is a Monte Carlo
estimate with fixed quasi-random base samples (derivative-free); ``parego`` is ``logei``
on a GP fitted to augmented-Chebyshev scalarizations (:func:`parego_scalarize`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from scipy.special import erfcx, log_ndtr, ndtr, ndtri

from ..space import sobol_points
from ..utils import LenzError, make_rng
from .pareto import default_ref_point, hvi_2d, pareto_mask

__all__ = [
    "ACQF_NAMES", "SINGLE_OBJECTIVE_ACQFS", "MULTI_OBJECTIVE_ACQFS", "MODEL_FREE_ACQFS",
    "POINTWISE_ACQFS", "GRADIENT_ACQFS", "DEFAULT_UCB_BETA", "EHVI_N_SAMPLES",
    "Acquisition", "AnalyticAcquisition", "EHVIAcquisition", "SurrogateModel",
    "build_acquisition", "parego_scalarize", "parego_weights",
    "log_prob_feasible", "prob_feasible", "log_h", "log1mexp",
]

SINGLE_OBJECTIVE_ACQFS: tuple[str, ...] = ("logei", "noisy_logei", "ei", "logpi", "pi", "ucb", "pm", "std", "ts")
MULTI_OBJECTIVE_ACQFS: tuple[str, ...] = ("ehvi", "parego")
MODEL_FREE_ACQFS: tuple[str, ...] = ("sobol", "random")
ACQF_NAMES: tuple[str, ...] = SINGLE_OBJECTIVE_ACQFS + MULTI_OBJECTIVE_ACQFS + MODEL_FREE_ACQFS
#: names that :func:`build_acquisition` turns into a pointwise ``Acquisition`` (usable in ``score``)
POINTWISE_ACQFS: tuple[str, ...] = tuple(n for n in ACQF_NAMES if n not in ("ts",) + MODEL_FREE_ACQFS)
#: pointwise acquisitions with analytic gradients
GRADIENT_ACQFS: tuple[str, ...] = tuple(n for n in POINTWISE_ACQFS if n != "ehvi")

DEFAULT_UCB_BETA = 4.0
EHVI_N_SAMPLES = 128

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_LOG_SQRT_PI_DIV_2 = 0.5 * math.log(math.pi / 2.0)
_SQRT_2_DIV_PI = math.sqrt(2.0 / math.pi)
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_LN2 = math.log(2.0)
# Below -1/sqrt(eps) the log1mexp term of log_h only carries its leading order -2 log|z|.
_NEG_INV_SQRT_EPS = -1.0 / math.sqrt(np.finfo(float).eps)
# Below this z the gradient ratio Phi(z)/h(z) comes from its asymptotic series.
_SERIES_BELOW = -100.0
# Posterior variances are floored at this multiple of the model's output variance.
_REL_VAR_FLOOR = 1e-12
# Training targets whose std is below this fraction of max|y| count as constant (as in gp.py).
_REL_CONSTANT_TOL = 1e-12
_EI_FAMILY = ("logei", "ei", "logpi", "pi")


class SurrogateModel(Protocol):
    """Structural type of the GP-like models acquisitions are built from."""

    X_train: np.ndarray
    y_train: np.ndarray

    def posterior(self, Xs: np.ndarray, full_cov: bool = False, return_grad: bool = False,
                  observation_noise: bool = False) -> tuple: ...

    def sample(self, Xs: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray: ...


# --------------------------------------------------------------------------- numerics


def log_phi(z: np.ndarray) -> np.ndarray:
    """Log density of the standard normal."""
    return -0.5 * np.square(z) - _LOG_SQRT_2PI


def log1mexp(x: np.ndarray) -> np.ndarray:
    """``log(1 - exp(x))`` for ``x <= 0``, accurate near 0 and for very negative ``x``."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        near = x > -_LN2
        a = np.log(-np.expm1(np.where(near, x, -1.0)))
        b = np.log1p(-np.exp(np.where(near, -1.0, x)))
    return np.where(near, a, b)


def _inv_mills(z: np.ndarray) -> np.ndarray:
    """``phi(z) / Phi(z)`` (the d/dz of ``log Phi``), stable for all ``z``."""
    neg = z < 0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        left = _SQRT_2_DIV_PI / erfcx(-np.where(neg, z, -1.0) * _INV_SQRT2)
        zr = np.where(neg, 0.0, z)
        right = np.exp(log_phi(zr) - log_ndtr(zr))
    return np.where(neg, left, right)


def _log_h_parts(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``log h(z)`` and ``log(Phi(z) / h(z))`` (= log of ``d/dz log h``), ``h = phi + z Phi``.

    Two branches as in BoTorch's ``_log_ei_helper``: for ``z > -1`` the direct formula
    is accurate; for ``z <= -1`` write ``h = phi(z) (1 - exp(w))`` with
    ``w = log(|z| Phi(z) / phi(z)) = log(|z| erfcx(-z/sqrt2)) + log(sqrt(pi/2)) < 0`` and use
    ``log1mexp``; below ``-1/sqrt(eps)`` only the leading order ``-2 log|z|`` survives.
    The gradient ratio switches to its asymptotic series below ``z = -100``.
    """
    z = np.asarray(z, dtype=float)
    upper = z > -1.0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        zu = np.where(upper, z, 0.0)
        Phi_u = ndtr(zu)
        log_h_u = np.log(np.exp(log_phi(zu)) + zu * Phi_u)
        lr_u = np.log(Phi_u) - log_h_u

        zl = np.where(upper, -1.0, z)
        abs_zl = -zl
        log_abs = np.log(abs_zl)
        zc = np.maximum(zl, _NEG_INV_SQRT_EPS)
        w = np.log(erfcx(-zc * _INV_SQRT2) * (-zc)) + _LOG_SQRT_PI_DIV_2
        # w rounds to >= 0 only deep in the tail, where the asymptotic form is exact anyway
        mid = (zl > _NEG_INV_SQRT_EPS) & (w < 0)
        l1m = np.where(mid, log1mexp(np.minimum(w, -np.finfo(float).tiny)), -2.0 * log_abs)
        log_h_l = log_phi(zl) + l1m
        # log Phi = log phi + w - log|z|  =>  log(Phi/h) = w - log|z| - log1mexp(w).
        # That difference loses ~z^2 eps relative accuracy (w -> 0^-), so in the far tail
        # the gradient ratio uses the asymptotic series of Phi = phi/|z| (1 - 1/z^2 + ...)
        # and h = phi/z^2 (1 - 3/z^2 + ...) instead (truncation error < 1e-16 there).
        lr_l = np.where(mid, w - log_abs - l1m, log_abs)
        tail = zl < _SERIES_BELOW
        if tail.any():
            q = 1.0 / np.square(np.where(tail, zl, _SERIES_BELOW))
            series = (np.log1p(q * (-1.0 + q * (3.0 + q * (-15.0 + q * 105.0))))
                      - np.log1p(q * (-3.0 + q * (15.0 + q * (-105.0 + q * 945.0)))))
            lr_l = np.where(tail, log_abs + series, lr_l)
    return np.where(upper, log_h_u, log_h_l), np.where(upper, lr_u, lr_l)


def log_h(z: np.ndarray) -> np.ndarray:
    """``log(phi(z) + z Phi(z))``, the log of the standardized EI.

    Finite (and accurate to a few ulp) for ``|z|`` up to ~1e150, where ``z**2`` overflows.
    """
    return _log_h_parts(z)[0]


def _log_prob_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``log(Phi(b) - Phi(a))`` for ``a < b`` (``a`` may be ``-inf``, ``b`` may be ``+inf``)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        la, lb = log_ndtr(a), log_ndtr(b)
        lna, lnb = log_ndtr(-a), log_ndtr(-b)
        left = lb + log1mexp(la - lb)                 # both bounds in the lower tail
        right = lna + log1mexp(lnb - lna)             # both bounds in the upper tail
        middle = np.log1p(-(ndtr(a) + ndtr(-b)))      # interval straddles 0
    return np.where(b <= 0, left, np.where(a >= 0, right, middle))


# --------------------------------------------------------------------------- posterior helpers


def _as_points(Z: Any) -> np.ndarray:
    A = np.asarray(Z, dtype=float)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if A.ndim != 2:
        raise LenzError(f"acquisition inputs must have shape (m, D), got {np.shape(Z)}")
    return A


def _output_scale(model: Any) -> tuple[float, float]:
    """Mean and std of the model's training targets (``(0, 1)`` when unavailable).

    The std falls back to 1 only when the targets are constant *relative to their
    magnitude* (as in the GP's own standardization): an absolute floor would put the
    variance floor far above the posterior variance of a tiny-scale objective.
    """
    y = getattr(model, "y_train", None)
    if y is None:
        return 0.0, 1.0
    y = np.asarray(y, dtype=float).reshape(-1)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0, 1.0
    mag = float(np.max(np.abs(y)))
    if mag == 0.0:
        return 0.0, 1.0
    u = y / mag                      # scale-free statistics (no under/overflow)
    std = float(np.std(u))
    if not std > _REL_CONSTANT_TOL:
        return mag * float(np.mean(u)), 1.0
    return mag * float(np.mean(u)), mag * std


@dataclass
class _Post:
    mean: np.ndarray
    sigma: np.ndarray
    dmean: np.ndarray | None
    dsigma: np.ndarray | None


def _posterior(model: Any, Z: np.ndarray, grad: bool, var_floor: float) -> _Post:
    """Posterior mean/std (and their input gradients) with the variance floored."""
    if grad:
        mean, var, dmean, dvar = model.posterior(Z, return_grad=True)
        dmean = np.asarray(dmean, dtype=float).reshape(Z.shape)
        dvar = np.asarray(dvar, dtype=float).reshape(Z.shape)
    else:
        mean, var = model.posterior(Z)[:2]
    mean = np.asarray(mean, dtype=float).reshape(-1)
    var = np.asarray(var, dtype=float).reshape(-1)
    floored = var < var_floor
    sigma = np.sqrt(np.maximum(var, var_floor))
    if not grad:
        return _Post(mean, sigma, None, None)
    dsigma = np.where(floored[:, None], 0.0, dvar) / (2.0 * sigma[:, None])
    return _Post(mean, sigma, dmean, dsigma)


@dataclass(frozen=True)
class _Constraint:
    model: Any
    lower: float | None
    upper: float | None
    var_floor: float


def _check_constraints(constraint_models: Sequence | None) -> list[_Constraint]:
    out: list[_Constraint] = []
    for i, item in enumerate(constraint_models or ()):
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise LenzError(f"constraint {i}: expected a (model, lower, upper) triple, got {item!r}")
        model, lower, upper = item
        bounds = []
        for label, v in (("lower", lower), ("upper", upper)):
            if v is None:
                bounds.append(None)
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                raise LenzError(f"constraint {i}: {label} must be a number or None, got {v!r}") from None
            if not math.isfinite(fv):
                raise LenzError(f"constraint {i}: {label} must be finite, got {v!r}")
            bounds.append(fv)
        lo, hi = bounds
        if lo is None and hi is None:
            raise LenzError(f"constraint {i}: at least one of lower/upper must be given")
        if lo is not None and hi is not None and not lo < hi:
            raise LenzError(f"constraint {i}: need lower < upper, got [{lo}, {hi}]")
        out.append(_Constraint(model, lo, hi, _REL_VAR_FLOOR * _output_scale(model)[1] ** 2))
    return out


def _log_pof(constraints: Sequence[_Constraint], Z: np.ndarray,
             grad: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """``sum_j log P(lower_j <= c_j(Z) <= upper_j)`` and its gradient (``(m,)``, ``(m, D)``)."""
    m, D = Z.shape
    total = np.zeros(m)
    dtotal = np.zeros((m, D)) if grad else None
    for c in constraints:
        p = _posterior(c.model, Z, grad, c.var_floor)
        s = p.sigma
        if c.upper is not None and c.lower is None:            # P = Phi(b)
            b = (c.upper - p.mean) / s
            lp = log_ndtr(b)
            if grad:
                rb = _inv_mills(b)
                d_mu, d_s = -rb / s, -b * rb / s
        elif c.lower is not None and c.upper is None:          # P = Phi(-a)
            a = (c.lower - p.mean) / s
            lp = log_ndtr(-a)
            if grad:
                ra = _inv_mills(-a)
                d_mu, d_s = ra / s, a * ra / s
        else:
            a = (c.lower - p.mean) / s
            b = (c.upper - p.mean) / s
            lp = _log_prob_between(a, b)
            if grad:
                with np.errstate(over="ignore", invalid="ignore"):
                    ra = np.exp(log_phi(a) - lp)
                    rb = np.exp(log_phi(b) - lp)
                # dP/dmu = (phi(a) - phi(b)) / s,  dP/ds = (a phi(a) - b phi(b)) / s
                d_mu, d_s = (ra - rb) / s, (a * ra - b * rb) / s
        total += lp
        if grad:
            dtotal += d_mu[:, None] * p.dmean + d_s[:, None] * p.dsigma
    if grad:
        dtotal = np.where(np.isfinite(dtotal), dtotal, 0.0)
    return total, dtotal


def log_prob_feasible(constraint_models: Sequence | None, Z: np.ndarray,
                      return_grad: bool = False):
    """``log PoF(Z) = sum_j log P(lower_j <= c_j(Z) <= upper_j)`` for ``(model, lower, upper)`` triples.

    Returns ``(m,)`` (zeros when there are no constraints), plus the ``(m, D)`` gradient
    when ``return_grad`` is true.
    """
    Zp = _as_points(Z)
    lp, dlp = _log_pof(_check_constraints(constraint_models), Zp, return_grad)
    return (lp, dlp) if return_grad else lp


def prob_feasible(constraint_models: Sequence | None, Z: np.ndarray) -> np.ndarray:
    """Probability that every constraint holds at ``Z`` (independent constraint models)."""
    return np.exp(log_prob_feasible(constraint_models, Z))


# --------------------------------------------------------------------------- acquisition objects


class Acquisition:
    """Base class: ``acq(Z) -> (m,)`` values to be maximized over encoded points."""

    name: str = "acquisition"
    has_gradient: bool = False

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def value_and_grad(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Values ``(m,)`` and input gradients ``(m, D)``; only when ``has_gradient``."""
        raise NotImplementedError(f"acquisition {self.name!r} is derivative-free (has_gradient=False)")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class AnalyticAcquisition(Acquisition):
    """Closed-form single-objective acquisitions with exact gradients.

    ``kind`` is one of ``logei, ei, logpi, pi, ucb, pm, std`` (``noisy_logei`` and
    ``parego`` are ``logei`` with a different incumbent / model). With constraints:

    * EI family, log variants: ``value = log a(x) + sum_j log PoF_j``;
      ``ei`` / ``pi``: ``value = a(x) * prod_j PoF_j``.
    * ``feasibility_phase`` (no feasible observation yet): only the PoF term
      (``sum_j log PoF_j`` for log variants, ``prod_j PoF_j`` for ``ei`` / ``pi``).
    * ``ucb`` / ``pm`` / ``std``: value in standardized objective units
      (``(v - mean(y_train)) / std(y_train)``; ``std`` only scaled) plus ``sum_j log PoF_j``.
      Without constraints these are reported in raw objective units.
    """

    has_gradient = True

    def __init__(self, name: str, kind: str, model: Any, *, best_f: float | None = None,
                 beta: float = DEFAULT_UCB_BETA, constraints: Sequence[_Constraint] = (),
                 feasibility_phase: bool = False):
        if kind not in _EI_FAMILY + ("ucb", "pm", "std"):
            raise LenzError(f"unknown analytic acquisition kind {kind!r}")
        self.name = name
        self.kind = kind
        self.model = model
        self.best_f = None if best_f is None else float(best_f)
        self.beta = float(beta)
        self.constraints = list(constraints)
        self.feasibility_phase = bool(feasibility_phase)
        if kind in _EI_FAMILY and not self.feasibility_phase and self.best_f is None:
            raise LenzError(f"{name}: best_f is required outside the feasibility phase")
        y_mean, y_std = _output_scale(model)
        self._var_floor = _REL_VAR_FLOOR * y_std ** 2
        # ucb / pm / std are put on the scale of log PoF only when constraints are present
        self._center, self._scale = 0.0, 1.0
        if self.constraints and kind in ("ucb", "pm", "std"):
            self._center = 0.0 if kind == "std" else y_mean
            self._scale = y_std

    @property
    def multiplicative(self) -> bool:
        """``ei`` / ``pi`` combine with the PoF by multiplication instead of in the log domain."""
        return self.kind in ("ei", "pi")

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        return self._evaluate(_as_points(Z), grad=False)[0]

    def value_and_grad(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._evaluate(_as_points(Z), grad=True)

    def _objective(self, Z: np.ndarray, grad: bool) -> tuple[np.ndarray, np.ndarray | None]:
        p = _posterior(self.model, Z, grad, self._var_floor)
        mu, sigma = p.mean, p.sigma
        kind, g = self.kind, None
        if kind in _EI_FAMILY:
            z = (mu - self.best_f) / sigma
        if kind == "logei":
            lh, lr = _log_h_parts(z)
            v = np.log(sigma) + lh
            if grad:
                r = np.exp(lr)                                           # Phi(z) / h(z)
                with np.errstate(over="ignore", invalid="ignore"):
                    # phi(z)/h(z) = 1 - z Phi/h; the exp form avoids cancellation for z > 0
                    c_s = np.where(z < 0, 1.0 - z * r, np.exp(log_phi(z) - lh))
                g = (r[:, None] * p.dmean + c_s[:, None] * p.dsigma) / sigma[:, None]
        elif kind == "ei":
            v = sigma * np.exp(_log_h_parts(z)[0])
            if grad:
                g = ndtr(z)[:, None] * p.dmean + np.exp(log_phi(z))[:, None] * p.dsigma
        elif kind in ("logpi", "pi"):
            v = log_ndtr(z) if kind == "logpi" else ndtr(z)
            if grad:
                dz = (p.dmean - z[:, None] * p.dsigma) / sigma[:, None]
                dv_dz = _inv_mills(z) if kind == "logpi" else np.exp(log_phi(z))
                g = dv_dz[:, None] * dz
        elif kind == "ucb":
            sb = math.sqrt(self.beta)
            v = mu + sb * sigma
            if grad:
                g = p.dmean + sb * p.dsigma
        elif kind == "pm":
            v = mu
            if grad:
                g = p.dmean
        else:  # std
            v = sigma
            if grad:
                g = p.dsigma
        if self._scale != 1.0 or self._center != 0.0:
            v = (v - self._center) / self._scale
            if grad:
                g = g / self._scale
        return v, g

    def _evaluate(self, Z: np.ndarray, grad: bool) -> tuple[np.ndarray, np.ndarray | None]:
        m, D = Z.shape
        if self.constraints:
            lp, dlp = _log_pof(self.constraints, Z, grad)
        else:
            lp, dlp = np.zeros(m), (np.zeros((m, D)) if grad else None)
        if self.feasibility_phase:
            if self.multiplicative:
                v = np.exp(lp)
                return v, (v[:, None] * dlp if grad else None)
            return lp, dlp
        v, g = self._objective(Z, grad)
        if not self.constraints:
            return v, g
        if self.multiplicative:
            pof = np.exp(lp)
            return v * pof, (pof[:, None] * (g + v[:, None] * dlp) if grad else None)
        return v + lp, (g + dlp if grad else None)


class EHVIAcquisition(Acquisition):
    """Monte Carlo expected hypervolume improvement for two objectives.

    ``EHVI(x) = mean_s HVI(mu(x) + sigma(x) * eps_s)`` with fixed base samples
    ``eps`` (scrambled Sobol mapped through the normal quantile) and independent
    objective posteriors; ``HVI`` is exact (:func:`pareto.hvi_2d`). Constraints multiply
    by ``prod_j PoF_j``; in the feasibility phase the value is ``prod_j PoF_j``.
    """

    name = "ehvi"
    has_gradient = False

    def __init__(self, models: Sequence[Any], front: np.ndarray, ref_point: np.ndarray | None,
                 base_samples: np.ndarray, constraints: Sequence[_Constraint] = (),
                 feasibility_phase: bool = False):
        self.models = list(models)
        self.front = np.asarray(front, dtype=float).reshape(-1, 2)
        self.ref_point = None if ref_point is None else np.asarray(ref_point, dtype=float)
        self.base_samples = np.asarray(base_samples, dtype=float)
        self.constraints = list(constraints)
        self.feasibility_phase = bool(feasibility_phase)
        self._var_floors = [_REL_VAR_FLOOR * _output_scale(mdl)[1] ** 2 for mdl in self.models]

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        Z = _as_points(Z)
        m = Z.shape[0]
        lp = _log_pof(self.constraints, Z, False)[0] if self.constraints else np.zeros(m)
        if self.feasibility_phase:
            return np.exp(lp)
        posts = [_posterior(mdl, Z, False, fl) for mdl, fl in zip(self.models, self._var_floors)]
        mu = np.stack([p.mean for p in posts], axis=1)                     # (m, 2)
        sd = np.stack([p.sigma for p in posts], axis=1)
        Y = mu[:, None, :] + sd[:, None, :] * self.base_samples[None, :, :]  # (m, N, 2)
        hvi = hvi_2d(Y.reshape(-1, 2), self.front, self.ref_point).reshape(m, -1)
        v = hvi.mean(axis=1)
        return v * np.exp(lp) if self.constraints else v


# --------------------------------------------------------------------------- ParEGO


def parego_weights(m: int, rng: np.random.Generator | int | None) -> np.ndarray:
    """Random ParEGO weights ``~ Dirichlet(1, ..., 1)`` (uniform on the simplex)."""
    if int(m) < 1:
        raise LenzError("parego needs at least one objective")
    return make_rng(rng).dirichlet(np.ones(int(m)))


def parego_scalarize(Y: np.ndarray, weights: Sequence[float] | np.ndarray | None,
                     rho: float = 0.05, *, rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Augmented Chebyshev scalarization of observed objectives (maximize convention).

    Each objective is min-max normalized over the rows of ``Y`` (``yhat in [0, 1]``, 1 =
    best; a constant column maps to 1 so it does not dominate the min), then
    ``s = min_i w_i yhat_i + rho * sum_i w_i yhat_i``. ``weights`` are normalized to sum 1;
    ``None`` draws Dirichlet(1) weights from ``rng``. Rows with NaN give NaN.
    """
    A = np.asarray(Y, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise LenzError(f"parego_scalarize expects Y of shape (n, m), got {np.shape(Y)}")
    n, m = A.shape
    if weights is None:
        w = parego_weights(m, rng)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != m:
            raise LenzError(f"parego weights must have {m} entries (one per objective), got {w.size}")
        if not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
            raise LenzError(f"parego weights must be non-negative with a positive sum, got {w.tolist()}")
    w = w / w.sum()
    if not math.isfinite(float(rho)) or rho < 0:
        raise LenzError(f"parego rho must be a non-negative number, got {rho!r}")
    if n == 0:
        return np.zeros(0)
    finite = np.all(np.isfinite(A), axis=1)
    if finite.any():
        lo, hi = A[finite].min(axis=0), A[finite].max(axis=0)
    else:
        lo, hi = np.zeros(m), np.zeros(m)
    span = hi - lo
    Yhat = np.where(span > 0, (A - lo) / np.where(span > 0, span, 1.0), 1.0)
    wy = w * Yhat
    out = wy.min(axis=1) + float(rho) * wy.sum(axis=1)
    return np.where(finite, out, np.nan)


# --------------------------------------------------------------------------- factory


def _incumbent_from_data(model: Any, name: str) -> float:
    y = getattr(model, "y_train", None)
    vals = np.asarray(y, dtype=float).reshape(-1) if y is not None else np.zeros(0)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise LenzError(f"{name}: best_f is required (the model has no finite training targets)")
    return float(vals.max())


def _check_best_f(best_f: Any) -> float | None:
    if best_f is None:
        return None
    try:
        v = float(best_f)
    except (TypeError, ValueError):
        raise LenzError(f"best_f must be a number or None, got {best_f!r}") from None
    if not math.isfinite(v):
        raise LenzError(f"best_f must be finite, got {best_f!r}")
    return v


def build_acquisition(name: str,
                      objective_models: Sequence[Any],
                      constraint_models: Sequence | None = None,
                      best_f: float | None = None,
                      X_observed: np.ndarray | None = None,
                      params: Mapping[str, Any] | None = None,
                      Y_observed: np.ndarray | None = None,
                      rng: np.random.Generator | int | None = None) -> Acquisition:
    """Build an acquisition function (maximize convention) from fitted models.

    Args:
        name: one of :data:`POINTWISE_ACQFS` (case-insensitive). ``ts``, ``sobol`` and
            ``random`` are valid acquisition names but are handled by the optimizer / lenz
            and raise ``LenzError`` here.
        objective_models: sign-adjusted models; exactly 1 for single-objective names and
            ``parego`` (the GP on scalarized values), exactly 2 for ``ehvi``.
        constraint_models: ``(model, lower, upper)`` triples on raw metric scale.
        best_f: best observed feasible objective. With constraints, ``None`` means no
            feasible point exists yet (feasibility phase). Without constraints ``None``
            falls back to ``max(model.y_train)``. ``noisy_logei`` replaces it by the max
            posterior mean over ``X_observed`` (default ``model.X_train``; pass feasible
            observed points when constraints are active).
        params: ``beta`` (ucb, default 4), ``ref_point`` (ehvi, maximize convention,
            default :func:`pareto.default_ref_point`), ``n_samples`` (ehvi, default 128).
            Other keys are ignored.
        Y_observed: ``(n, 2)`` observed (feasible) objectives for ``ehvi``; default stacks
            the models' ``y_train`` when they align.
        rng: generator for the ``ehvi`` base samples (default: fixed seed 0).
    """
    if not isinstance(name, str):
        raise LenzError(f"acquisition name must be a string, got {name!r}")
    key = name.strip().lower()
    if key not in ACQF_NAMES:
        raise LenzError(f"unknown acquisition function {name!r}; choose one of: {', '.join(ACQF_NAMES)}")
    if key == "ts":
        raise LenzError("'ts' (Thompson sampling) has no pointwise value: it draws one joint posterior "
                        "sample over a candidate set (optimize.thompson_sample), so it cannot be scored")
    if key in MODEL_FREE_ACQFS:
        raise LenzError(f"{key!r} is model-free space filling handled by lenz, not a pointwise acquisition")
    params = dict(params or {})
    models = list(objective_models or [])
    constraints = _check_constraints(constraint_models)
    best = _check_best_f(best_f)

    if key == "ehvi":
        if len(models) != 2:
            raise LenzError(f"'ehvi' supports exactly 2 objectives, got {len(models)} objective models"
                            + ("; use 'parego' for more than 2 objectives" if len(models) > 2 else
                               "; use a single-objective acquisition such as 'logei'"))
        return _build_ehvi(models, constraints, params, Y_observed, rng)

    if len(models) != 1:
        if key == "parego":
            raise LenzError(f"'parego' expects the single GP fitted to the ParEGO-scalarized objective "
                            f"(see parego_scalarize), got {len(models)} objective models")
        raise LenzError(f"{key!r} is a single-objective acquisition and needs exactly 1 objective model, "
                        f"got {len(models)}" + ("; with several objectives use 'ehvi' (2) or 'parego'"
                                                if len(models) > 1 else ""))
    model = models[0]
    if key in ("ucb", "pm", "std"):
        beta = params.get("beta", DEFAULT_UCB_BETA)
        if beta is None:
            beta = DEFAULT_UCB_BETA
        try:
            beta = float(beta)
        except (TypeError, ValueError):
            raise LenzError(f"ucb beta must be a number, got {beta!r}") from None
        if not (math.isfinite(beta) and beta >= 0):
            raise LenzError(f"ucb beta must be a finite number >= 0, got {beta!r}")
        return AnalyticAcquisition(key, key, model, beta=beta, constraints=constraints)

    kind = "logei" if key in ("noisy_logei", "parego") else key
    feasibility = bool(constraints) and best is None
    if not feasibility:
        if key == "noisy_logei":
            best = _noisy_incumbent(model, X_observed, best)
        elif best is None:
            best = _incumbent_from_data(model, key)
    return AnalyticAcquisition(key, kind, model, best_f=best, constraints=constraints,
                               feasibility_phase=feasibility)


def _noisy_incumbent(model: Any, X_observed: np.ndarray | None, best_f: float | None) -> float:
    X = X_observed if X_observed is not None else getattr(model, "X_train", None)
    if X is not None and np.size(X):
        mean = np.asarray(model.posterior(np.atleast_2d(np.asarray(X, dtype=float)))[0], dtype=float)
        mean = mean.reshape(-1)[np.isfinite(mean.reshape(-1))]
        if mean.size:
            return float(mean.max())
    if best_f is not None:
        return best_f
    return _incumbent_from_data(model, "noisy_logei")


def _build_ehvi(models: list, constraints: list[_Constraint], params: dict,
                Y_observed: np.ndarray | None, rng: Any) -> EHVIAcquisition:
    if Y_observed is None:
        ys = [getattr(mdl, "y_train", None) for mdl in models]
        if all(y is not None for y in ys) and len({np.size(y) for y in ys}) == 1:
            Y = np.column_stack([np.asarray(y, dtype=float).reshape(-1) for y in ys])
        else:
            Y = np.zeros((0, 2))
    else:
        Y = np.asarray(Y_observed, dtype=float)
        if Y.size == 0:
            Y = np.zeros((0, 2))
        elif Y.ndim == 1 and Y.size == 2:
            Y = Y.reshape(1, 2)
        if Y.ndim != 2 or Y.shape[1] != 2:
            raise LenzError(f"ehvi: Y_observed must have shape (n, 2), got {np.shape(Y_observed)}")
    Y = Y[np.all(np.isfinite(Y), axis=1)]
    ref = params.get("ref_point")
    if ref is not None:
        ref = np.asarray(ref, dtype=float).reshape(-1)
        if ref.size != 2 or not np.all(np.isfinite(ref)):
            raise LenzError(f"ehvi: ref_point must be 2 finite numbers (maximize convention), got {params['ref_point']!r}")
    feasibility = bool(constraints) and Y.shape[0] == 0
    if ref is None and not feasibility:
        if Y.shape[0] == 0:
            raise LenzError("ehvi needs a ref_point (params['ref_point']) or observed objectives (Y_observed)")
        ref = default_ref_point(Y)
    n_samples = params.get("n_samples", EHVI_N_SAMPLES)
    try:
        n_samples = int(n_samples)
    except (TypeError, ValueError):
        raise LenzError(f"ehvi: n_samples must be a positive integer, got {n_samples!r}") from None
    if n_samples < 1:
        raise LenzError(f"ehvi: n_samples must be a positive integer, got {n_samples}")
    U = sobol_points(n_samples, 2, make_rng(0 if rng is None else rng))
    base = ndtri(np.clip(U, 1e-10, 1.0 - 1e-10))
    front = Y[pareto_mask(Y)] if Y.shape[0] else Y
    return EHVIAcquisition(models, front, ref, base, constraints=constraints, feasibility_phase=feasibility)
