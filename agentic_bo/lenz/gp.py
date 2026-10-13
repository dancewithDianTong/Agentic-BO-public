"""Exact Gaussian-process surrogate: Matern-5/2 / RBF ARD kernels, MAP hyperparameters.

This is the statistical core of the lenz backend (DESIGN.md §5). Inputs live in the
encoded space ``[0, 1]^D``; outputs are standardized internally and every value the
model returns is in the original ``y`` units.

Model (BoTorch-style defaults, paper §5.1)::

    y_std = (y - y_mean) / y_std_scale
    y_std ~ GP(0, s * k(r)) + N(0, noise),   r^2 = sum_i (x_i - x'_i)^2 / l_i^2

with ``k`` the Matern-5/2 (default) or squared-exponential correlation. The
hyperparameters ``theta = log([l_1..l_D, s, noise])`` are fitted by maximum a posteriori
estimation in log space: log marginal likelihood plus Normal log-priors on the log
parameters. Priors: lengthscale ``LogNormal(sqrt(2) + ln(D)/2, sqrt(3))`` (dimension-scaled,
Hvarfner et al. 2024), outputscale ``LogNormal(0, 1)``, noise variance ``LogNormal(-4, 1)``
on ``[1e-6, 1]``. Two prior forms are available (see :func:`hyperprior`): ``"botorch"``
(default) evaluates the lengthscale and noise LogNormal densities on the parameters
themselves, as GPyTorch/BoTorch do, which in log space is ``Normal(mu - sigma^2, sigma)``;
``"median"`` uses ``Normal(mu, sigma)`` on the log parameters (the LogNormal including the
log-Jacobian), whose mode is the prior median ``exp(mu)``. The negative log posterior and its analytic
gradient are exposed as :func:`neg_log_posterior`; kernel helpers (:func:`kernel_matrix`,
:func:`kernel_grad_x`, ...) are public so acquisition code can reuse them.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.linalg.lapack import dpotrf, dpotri
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist, squareform

from ..utils import LenzError, make_rng

__all__ = [
    "GP",
    "KERNELS",
    "hyperprior",
    "kernel_from_sqdist",
    "kernel_grad_x",
    "kernel_matrix",
    "neg_log_posterior", "PRIOR_FORMS", "DEFAULT_PRIOR",
    "scaled_sqdist",
]

KERNELS = ("matern52", "rbf")

LENGTHSCALE_BOUNDS = (1e-2, 1e3)
OUTPUTSCALE_BOUNDS = (1e-2, 1e2)
NOISE_BOUNDS = (1e-6, 1.0)
OUTPUTSCALE_PRIOR = (0.0, 1.0)      # (mu, sigma) of the Normal prior on log(outputscale)
NOISE_PRIOR = (-4.0, 1.0)           # (mu, sigma) of the Normal prior on log(noise variance)

_SQRT5 = math.sqrt(5.0)
_LOG_2PI = math.log(2.0 * math.pi)
_JITTERS = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
_VAR_FLOOR = 1e-12                  # smallest latent variance returned (standardized units)
_STD_REL_TOL = 1e-12                # y with std <= this * max|y| is treated as constant
_FAILED_NLP = 1e25                  # objective value reported when the kernel matrix cannot be factorized
_MAXITER = 200


# --------------------------------------------------------------------------- kernels


def _check_kind(kind: str) -> str:
    if kind not in KERNELS:
        raise LenzError(f"unknown kernel {kind!r}; expected one of {list(KERNELS)}")
    return kind


def scaled_sqdist(X1: np.ndarray, X2: np.ndarray | None, lengthscales: np.ndarray) -> np.ndarray:
    """Squared ARD distances ``r^2_ab = sum_i (X1_ai - X2_bi)^2 / l_i^2``.

    ``X2=None`` computes the symmetric matrix of ``X1`` with itself (exact zero diagonal).
    Differences are formed directly (no ``|a|^2 + |b|^2 - 2ab`` cancellation).
    """
    ls = np.asarray(lengthscales, dtype=float)
    A = np.asarray(X1, dtype=float) / ls
    if X2 is None:
        n = A.shape[0]
        if n < 2:
            return np.zeros((n, n))
        return squareform(pdist(A, "sqeuclidean"))
    B = np.asarray(X2, dtype=float) / ls
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros((A.shape[0], B.shape[0]))
    return cdist(A, B, "sqeuclidean")


def kernel_from_sqdist(r2: np.ndarray, kind: str = "matern52",
                       outputscale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Kernel values ``k`` and derivatives ``dk/d(r^2)`` from squared scaled distances.

    Matern-5/2: ``k = s (1 + sqrt5 r + 5/3 r^2) exp(-sqrt5 r)``,
    ``dk/dr^2 = -(5/6) s (1 + sqrt5 r) exp(-sqrt5 r)`` (finite at ``r = 0``).
    RBF: ``k = s exp(-r^2 / 2)``, ``dk/dr^2 = -k / 2``.
    """
    r2 = np.maximum(np.asarray(r2, dtype=float), 0.0)
    if _check_kind(kind) == "matern52":
        r = np.sqrt(r2)
        e = np.exp(-_SQRT5 * r)
        k = (1.0 + _SQRT5 * r + (5.0 / 3.0) * r2) * e
        dk = -(5.0 / 6.0) * (1.0 + _SQRT5 * r) * e
    else:
        k = np.exp(-0.5 * r2)
        dk = -0.5 * k
    return outputscale * k, outputscale * dk


def kernel_matrix(X1: np.ndarray, X2: np.ndarray | None, lengthscales: np.ndarray,
                  outputscale: float = 1.0, kind: str = "matern52") -> np.ndarray:
    """Covariance matrix ``outputscale * k(X1, X2)`` of shape ``(n1, n2)`` (``X2=None``: ``X1`` with itself)."""
    return kernel_from_sqdist(scaled_sqdist(X1, X2, lengthscales), kind, outputscale)[0]


def kernel_grad_x(X1: np.ndarray, X2: np.ndarray, lengthscales: np.ndarray,
                  outputscale: float = 1.0, kind: str = "matern52") -> np.ndarray:
    """Gradient of ``k(X1_a, X2_b)`` with respect to ``X1_a``: shape ``(n1, n2, D)``.

    ``dk/dx_i = 2 dk/d(r^2) (x_i - x'_i) / l_i^2``; allocates an ``n1 x n2 x D`` array, so
    for large problems prefer contracting ``dk/d(r^2)`` directly as :meth:`GP.posterior` does.
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    ls = np.asarray(lengthscales, dtype=float)
    _, dk = kernel_from_sqdist(scaled_sqdist(X1, X2, ls), kind, outputscale)
    diff = (X1[:, None, :] - X2[None, :, :]) / ls**2
    return 2.0 * dk[:, :, None] * diff


# --------------------------------------------------------------------------- priors / objective

#: ``botorch``: LogNormal densities evaluated on the parameters (default); ``median``: on their logs
PRIOR_FORMS = ("botorch", "median")
DEFAULT_PRIOR = "botorch"


def hyperprior(D: int, learn_noise: bool = True, form: str = DEFAULT_PRIOR,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(mu, sigma, lower, upper)`` over ``theta = log([l (D), outputscale, noise?])``.

    ``mu``/``sigma`` parameterize the independent Normal densities on the log parameters
    that enter the MAP objective; ``lower``/``upper`` are the L-BFGS-B box.

    ``form="median"``: ``Normal(mu0, sigma)`` with ``mu0`` the LogNormal location, i.e. the
    LogNormal including the log-Jacobian; its mode in ``theta`` is the prior median.
    ``form="botorch"`` (default): the LogNormal density evaluated on the lengthscales and the
    noise variance themselves (the GPyTorch/BoTorch convention). Over ``theta`` that density
    is proportional to ``Normal(mu0 - sigma^2, sigma)``, so it prefers shorter lengthscales
    and less noise. In a 288-run A/B (6 benchmarks x 2 policies x 8 seeds) it was better or
    tied everywhere, significantly so on ackley10 and levy10. The outputscale keeps the
    log-space form in both cases.
    """
    if D < 1:
        raise LenzError(f"input dimension must be >= 1, got {D}")
    form = _check_prior(form)
    p = D + 1 + int(learn_noise)
    mu, sigma = np.empty(p), np.empty(p)
    lo, hi = np.empty(p), np.empty(p)
    mu[:D], sigma[:D] = math.sqrt(2.0) + 0.5 * math.log(D), math.sqrt(3.0)
    lo[:D], hi[:D] = np.log(LENGTHSCALE_BOUNDS)
    mu[D], sigma[D] = OUTPUTSCALE_PRIOR
    lo[D], hi[D] = np.log(OUTPUTSCALE_BOUNDS)
    if learn_noise:
        mu[D + 1], sigma[D + 1] = NOISE_PRIOR
        lo[D + 1], hi[D + 1] = np.log(NOISE_BOUNDS)
    if form == "botorch":
        mu[:D] -= sigma[:D] ** 2
        if learn_noise:
            mu[D + 1] -= sigma[D + 1] ** 2
    return mu, sigma, lo, hi


def _check_prior(form: Any) -> str:
    f = form.strip().lower() if isinstance(form, str) else form
    if f not in PRIOR_FORMS:
        raise LenzError(f"prior must be one of {list(PRIOR_FORMS)}, got {form!r}")
    return f


def _add_diag(K: np.ndarray, v: float) -> np.ndarray:
    K = K.copy()
    K.flat[:: K.shape[0] + 1] += v
    return K


def _cholesky(K: np.ndarray) -> tuple[np.ndarray, float]:
    """Lower Cholesky factor of ``K + jitter I`` with jitter escalation 0, 1e-8, ..., 1e-4."""
    for jitter in _JITTERS:
        A = _add_diag(K, jitter) if jitter else K
        L, info = dpotrf(A, lower=1, clean=1, overwrite_a=0)
        if info == 0:
            return L, jitter
    raise np.linalg.LinAlgError("kernel matrix is not positive definite even with jitter 1e-4")


def _inverse_from_cholesky(L: np.ndarray) -> np.ndarray:
    inv, info = dpotri(L, lower=1)
    if info != 0:  # pragma: no cover - L comes from a successful dpotrf
        raise np.linalg.LinAlgError("dpotri failed")
    return np.tril(inv) + np.tril(inv, -1).T


def _split_theta(theta: np.ndarray, D: int, fixed_noise: float | None) -> tuple[np.ndarray, float, float]:
    theta = np.asarray(theta, dtype=float)
    ls = np.exp(theta[:D])
    s = float(np.exp(theta[D]))
    noise = float(np.exp(theta[D + 1])) if fixed_noise is None else float(fixed_noise)
    return ls, s, noise


def neg_log_posterior(theta: np.ndarray, X: np.ndarray, y: np.ndarray, kind: str = "matern52",
                      noise: float | None = None, prior: str = DEFAULT_PRIOR) -> tuple[float, np.ndarray]:
    """Negative log posterior of ``theta`` and its analytic gradient.

    ``theta = log([l (D), outputscale, noise])`` when ``noise is None`` (learned) and
    ``log([l (D), outputscale])`` when ``noise`` is a fixed variance. ``y`` must already be
    standardized. The value is ``-(log p(y | theta) + log N(theta; mu, sigma))``.
    Raises ``numpy.linalg.LinAlgError`` if the kernel matrix cannot be factorized.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    theta = np.asarray(theta, dtype=float)
    n, D = X.shape
    learn = noise is None
    mu, sigma, _, _ = hyperprior(D, learn, prior)
    if theta.shape != mu.shape:
        raise LenzError(f"theta must have {mu.size} entries, got {theta.size}")
    ls, s, nv = _split_theta(theta, D, noise)

    # centring keeps the quadratic-form expansion below free of cancellation
    U = (X - X.mean(axis=0)) / ls
    r2 = scaled_sqdist(U, None, np.ones(D))
    Kf, dKf = kernel_from_sqdist(r2, kind, s)
    L, _ = _cholesky(_add_diag(Kf, nv))
    alpha = cho_solve((L, True), y, check_finite=False)
    Kinv = _inverse_from_cholesky(L)
    lml = -0.5 * float(y @ alpha) - float(np.sum(np.log(np.diag(L)))) - 0.5 * n * _LOG_2PI

    # dLML/dtheta_j = 0.5 tr(W dK/dtheta_j) with W = alpha alpha^T - K^{-1}
    W = np.outer(alpha, alpha) - Kinv
    M = W * dKf
    # dK_ab/dlog l_d = dKf_ab * (-2 (u_ad - u_bd)^2) and
    # sum_ab M_ab (u_a - u_b)^2 = 2 sum_a u_a^2 rowsum(M)_a - 2 u^T M u   (M symmetric)
    quad = 2.0 * (M.sum(axis=1) @ U**2) - 2.0 * np.einsum("ad,ad->d", U, M @ U)
    grad = np.empty_like(theta)
    grad[:D] = -quad
    grad[D] = 0.5 * float(np.sum(W * Kf))
    if learn:
        grad[D + 1] = 0.5 * nv * float(np.trace(W))

    z = (theta - mu) / sigma
    log_prior = float(np.sum(-0.5 * z**2 - np.log(sigma) - 0.5 * _LOG_2PI))
    return -(lml + log_prior), -(grad - z / sigma)


def _safe_objective(theta: np.ndarray, X: np.ndarray, y: np.ndarray, kind: str,
                    noise: float | None, prior: str = DEFAULT_PRIOR) -> tuple[float, np.ndarray]:
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            val, grad = neg_log_posterior(theta, X, y, kind, noise, prior)
    except np.linalg.LinAlgError:
        return _FAILED_NLP, np.zeros_like(theta)
    if not (math.isfinite(val) and np.all(np.isfinite(grad))):
        return _FAILED_NLP, np.zeros_like(theta)
    return val, grad


# --------------------------------------------------------------------------- helpers


def _is_constant(y: np.ndarray) -> bool:
    if y.shape[0] < 2:
        return True
    scale = float(np.max(np.abs(y)))
    if scale == 0.0:
        return True
    return not float(np.std(y / scale, ddof=1)) > _STD_REL_TOL


def _standardization(y: np.ndarray) -> tuple[float, float]:
    """``(mean, std)`` used to standardize ``y``; ``std = 1`` for ``n == 1`` or constant ``y``.

    Statistics are computed on ``y / max|y|`` so that huge magnitudes cannot overflow.
    """
    scale = float(np.max(np.abs(y)))
    if scale == 0.0:
        return 0.0, 1.0
    mean = scale * float(np.mean(y / scale))
    if _is_constant(y):
        return mean, 1.0
    return mean, scale * float(np.std(y / scale, ddof=1))


def _psd_factor(C: np.ndarray) -> np.ndarray:
    """A matrix ``F`` with ``F F^T ~= C`` for a (numerically) PSD ``C``.

    Cholesky with relative jitter escalation; falls back to a clipped eigendecomposition.
    """
    m = C.shape[0]
    base = max(float(np.mean(np.diag(C))), 1e-300)
    for rel in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        A = _add_diag(C, rel * base) if rel else C
        L, info = dpotrf(A, lower=1, clean=1, overwrite_a=0)
        if info == 0:
            return L
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    return V * np.sqrt(np.clip(w, 0.0, None))[None, :] if m else np.zeros((0, 0))


# --------------------------------------------------------------------------- GP


class GP:
    """Exact GP regression with ARD Matern-5/2 or RBF kernel and MAP hyperparameters.

    Parameters
    ----------
    kernel:
        ``"matern52"`` (default) or ``"rbf"``.
    noise:
        ``"learn"`` (MAP-estimated, ``LogNormal(-4, 1)`` prior on ``[1e-6, 1]``) or a fixed
        noise variance ``>= 0`` in standardized units.
    n_restarts:
        Number of L-BFGS-B runs: the warm start (if given), the prior median, then random
        prior draws. ``n_restarts <= 0`` disables optimization and conditions on the warm
        start (or the prior median) as is.
    seed:
        Seed for the random restarts; fits are deterministic given the seed.
    prior:
        Hyperprior form, ``"botorch"`` (default) or ``"median"``; see :func:`hyperprior`.

    Attributes
    ----------
    X_train, y_train:
        Training data in encoded / original units (``None`` before :meth:`fit`).
    y_mean, y_std:
        Output standardization constants (``y_std = 1`` when ``n == 1`` or ``y`` is constant).
    fit_info:
        Details of the last fit (``n_starts``, ``neg_log_posterior``, ``converged``, ...).
    """

    def __init__(self, kernel: str = "matern52", noise: float | str = "learn",
                 n_restarts: int = 3, seed: int = 0, prior: str = DEFAULT_PRIOR):
        self.kernel = _check_kind(kernel)
        self.prior = _check_prior(prior)
        if isinstance(noise, str):
            if noise != "learn":
                raise LenzError(f"noise must be 'learn' or a non-negative float, got {noise!r}")
            self.noise: float | str = "learn"
        else:
            try:
                nv = float(noise)
            except (TypeError, ValueError):
                raise LenzError(f"noise must be 'learn' or a non-negative float, got {noise!r}") from None
            if not (math.isfinite(nv) and nv >= 0.0):
                raise LenzError(f"fixed noise variance must be finite and >= 0, got {noise!r}")
            self.noise = nv
        self.n_restarts = int(n_restarts)
        self.seed = int(seed)
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0
        self.fit_info: dict[str, Any] = {}
        self._theta: np.ndarray | None = None
        self._ls: np.ndarray | None = None
        self._s = 1.0
        self._noise = 0.0
        self._jitter = 0.0
        self._L: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._ys: np.ndarray | None = None
        self._Kinv: np.ndarray | None = None
        self._lml = float("nan")

    # ---- basic properties -------------------------------------------------------------
    @property
    def learn_noise(self) -> bool:
        return self.noise == "learn"

    @property
    def n_train(self) -> int:
        return 0 if self.X_train is None else int(self.X_train.shape[0])

    @property
    def D(self) -> int | None:
        return None if self.X_train is None else int(self.X_train.shape[1])

    @property
    def is_fitted(self) -> bool:
        return self._L is not None

    def __repr__(self) -> str:
        state = f"n={self.n_train}, D={self.D}" if self.is_fitted else "unfitted"
        return f"GP(kernel={self.kernel!r}, noise={self.noise!r}, {state})"

    # ---- fitting ----------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, warm_start: Mapping[str, Any] | None = None) -> "GP":
        """Standardize ``y``, fit MAP hyperparameters and factorize the kernel matrix.

        ``warm_start`` is a :meth:`hyperparameters` dict (missing / mismatched entries fall
        back to the prior median). Raises ``LenzError`` on empty or non-finite data.
        """
        X, y = self._validate_data(X, y)
        n, D = X.shape
        if warm_start is not None and not isinstance(warm_start, Mapping):
            raise LenzError(f"warm_start must be a hyperparameter dict, got {type(warm_start).__name__}")
        y_mean, y_std = _standardization(y)
        ys = (y - y_mean) / y_std
        fixed = None if self.learn_noise else float(self.noise)
        mu, sigma, lo, hi = hyperprior(D, self.learn_noise, self.prior)
        median = np.clip(mu, lo, hi)
        warm = self._theta_from_dict(warm_start, D, median, lo, hi) if warm_start is not None else None

        info: dict[str, Any] = {"n_starts": 0, "converged": None, "n_evals": 0}
        if n == 1:
            theta = median
            info["reason"] = "n == 1: prior medians"
        elif self.n_restarts <= 0:
            theta = warm if warm is not None else median
            info["reason"] = "n_restarts <= 0: no optimization"
        else:
            starts = [warm] if warm is not None else []
            starts.append(median)
            rng = make_rng(self.seed)
            while len(starts) < self.n_restarts:
                starts.append(np.clip(mu + sigma * rng.standard_normal(mu.size), lo, hi))
            best_val, theta = math.inf, median
            bounds = list(zip(lo, hi))
            for x0 in starts[: max(self.n_restarts, 1)]:
                res = minimize(_safe_objective, x0, args=(X, ys, self.kernel, fixed, self.prior), jac=True,
                               method="L-BFGS-B", bounds=bounds, options={"maxiter": _MAXITER})
                info["n_starts"] += 1
                info["n_evals"] += int(res.nfev)
                val = float(res.fun)
                if math.isfinite(val) and val < _FAILED_NLP and val < best_val:
                    best_val, theta = val, np.clip(res.x, lo, hi)
                    info["converged"] = bool(res.success)
            info["neg_log_posterior"] = best_val if math.isfinite(best_val) else None

        self.X_train, self.y_train = X, y
        self.y_mean, self.y_std = y_mean, y_std
        self._ys = ys
        try:
            self._set_theta(theta)
        except np.linalg.LinAlgError:
            try:
                self._set_theta(median)
                info["reason"] = "fitted hyperparameters not factorizable: prior medians"
            except np.linalg.LinAlgError:
                self._L = None
                raise LenzError("could not factorize the GP kernel matrix (check for non-finite inputs)") from None
        self.fit_info = info
        return self

    def _validate_data(self, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1) if X.shape[0] == y.shape[0] else X.reshape(1, -1)
        if X.ndim != 2:
            raise LenzError(f"X must be a 2-D array (n, D), got shape {X.shape}")
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise LenzError("cannot fit a GP to zero observations")
        if X.shape[0] != y.shape[0]:
            raise LenzError(f"X has {X.shape[0]} rows but y has {y.shape[0]} values")
        if X.shape[1] < 1:
            raise LenzError("X must have at least one column")
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            raise LenzError("GP training data must be finite")
        return X.copy(), y.copy()

    def _theta_from_dict(self, hp: Mapping[str, Any], D: int, median: np.ndarray,
                         lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        theta = median.copy()

        def _log_pos(v: Any) -> np.ndarray | None:
            try:
                a = np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                return None
            if a.size == 0 or not np.all(np.isfinite(a)) or np.any(a <= 0):
                return None
            return np.log(a)

        ls = _log_pos(hp.get("lengthscales"))
        if ls is not None and ls.size in (1, D):
            theta[:D] = ls.reshape(-1)
        s = _log_pos(hp.get("outputscale"))
        if s is not None and s.size == 1:
            theta[D] = float(s.reshape(-1)[0])
        if self.learn_noise:
            nz = _log_pos(hp.get("noise"))
            if nz is not None and nz.size == 1:
                theta[D + 1] = float(nz.reshape(-1)[0])
        return np.clip(theta, lo, hi)

    def _set_theta(self, theta: np.ndarray) -> None:
        X = self.X_train
        D = X.shape[1]
        fixed = None if self.learn_noise else float(self.noise)
        ls, s, nv = _split_theta(theta, D, fixed)
        K = kernel_matrix(X, None, ls, s, self.kernel)
        L, jitter = _cholesky(_add_diag(K, nv))
        self._theta = np.asarray(theta, dtype=float).copy()
        self._ls, self._s, self._noise, self._jitter = ls, s, nv, jitter
        self._L = L
        self._refresh_solve()

    def _refresh_solve(self) -> None:
        self._alpha = cho_solve((self._L, True), self._ys, check_finite=False)
        self._Kinv = None
        n = self._ys.shape[0]
        self._lml = (-0.5 * float(self._ys @ self._alpha)
                     - float(np.sum(np.log(np.diag(self._L)))) - 0.5 * n * _LOG_2PI)

    # ---- prediction -------------------------------------------------------------------
    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise LenzError("the GP has not been fitted yet")

    def _as_points(self, Xs: Any) -> np.ndarray:
        D = self.D
        Z = np.asarray(Xs, dtype=float)
        if Z.ndim == 1:
            if Z.shape[0] == D:
                Z = Z.reshape(1, D)
            elif D == 1:
                Z = Z.reshape(-1, 1)
        if Z.ndim != 2 or Z.shape[1] != D:
            raise LenzError(f"expected points of shape (m, {D}), got {np.shape(Xs)}")
        if not np.all(np.isfinite(Z)):
            raise LenzError("prediction inputs must be finite")
        return Z

    def posterior(self, Xs: np.ndarray, full_cov: bool = False, return_grad: bool = False,
                  observation_noise: bool = False):
        """Posterior at ``Xs`` (``(m, D)``, or a single ``(D,)`` point), in original ``y`` units.

        Returns ``(mean (m,), var (m,))``; ``(mean, cov (m, m))`` if ``full_cov``;
        ``(mean, var, dmean (m, D), dvar (m, D))`` if ``return_grad`` (analytic gradients
        with respect to the input points). ``var`` is the latent-``f`` variance, plus the
        noise variance if ``observation_noise``. Variances are clamped at a tiny positive
        floor (the variance gradient is zero where the clamp is active).
        """
        self._require_fitted()
        if full_cov and return_grad:
            raise LenzError("posterior: return_grad requires full_cov=False")
        Z = self._as_points(Xs)
        m = Z.shape[0]
        X, L, alpha = self.X_train, self._L, self._alpha
        ls, s = self._ls, self._s
        ys2 = self.y_std**2
        noise_var = self._noise * ys2 if observation_noise else 0.0

        r2 = scaled_sqdist(Z, X, ls)
        Ks, dKs = kernel_from_sqdist(r2, self.kernel, s)
        mean = self.y_mean + self.y_std * (Ks @ alpha)
        V = solve_triangular(L, Ks.T, lower=True, check_finite=False)            # (n, m)

        if full_cov:
            cov = kernel_matrix(Z, None, ls, s, self.kernel) - V.T @ V
            cov = 0.5 * (cov + cov.T)
            d = np.diag(cov).copy()
            np.fill_diagonal(cov, np.maximum(d, _VAR_FLOOR * s))
            cov *= ys2
            if observation_noise:
                cov.flat[:: m + 1] += noise_var
            return mean, cov

        var_s = s - np.einsum("ij,ij->j", V, V)
        clamped = var_s < _VAR_FLOOR * s
        var = np.where(clamped, _VAR_FLOOR * s, var_s) * ys2 + noise_var
        if not return_grad:
            return mean, var

        # d k(x, X_j) / dx_d = 2 dk/dr2 (x_d - X_jd) / l_d^2; centring avoids cancellation
        c = X.mean(axis=0)
        Zc, Xc = Z - c, X - c
        inv_l2 = 1.0 / ls**2
        Ga = dKs * alpha[None, :]
        dmean = 2.0 * inv_l2 * (Zc * Ga.sum(axis=1)[:, None] - Ga @ Xc)
        A = solve_triangular(L, V, lower=True, trans="T", check_finite=False)   # K^{-1} k(X, Z)
        H = dKs * A.T
        dvar = -4.0 * inv_l2 * (Zc * H.sum(axis=1)[:, None] - H @ Xc)
        dvar[clamped] = 0.0
        return mean, var, self.y_std * dmean, ys2 * dvar

    def sample(self, Xs: np.ndarray, n_samples: int, rng: np.random.Generator | int | None) -> np.ndarray:
        """Joint posterior samples of the latent ``f`` at ``Xs``: shape ``(n_samples, m)``, original units."""
        self._require_fitted()
        Z = self._as_points(Xs)
        n_samples = int(n_samples)
        if n_samples < 0:
            raise LenzError(f"n_samples must be >= 0, got {n_samples}")
        rng = make_rng(rng)
        m = Z.shape[0]
        if m == 0 or n_samples == 0:
            return np.zeros((n_samples, m))
        mean, cov = self.posterior(Z, full_cov=True)
        # factor in standardized units so the relative jitter is scale-free
        F = _psd_factor(cov / self.y_std**2)
        eps = rng.standard_normal((n_samples, F.shape[1]))
        return mean[None, :] + self.y_std * (eps @ F.T)

    # ---- model criticism ----------------------------------------------------------------
    def _kinv(self) -> np.ndarray:
        if self._Kinv is None:
            self._Kinv = _inverse_from_cholesky(self._L)
        return self._Kinv

    def loo(self) -> tuple[np.ndarray, np.ndarray]:
        """Closed-form leave-one-out predictive mean / variance (original units, variance incl. noise).

        Uses ``mu_i = y_i - [K^-1 y]_i / [K^-1]_ii`` and ``var_i = 1 / [K^-1]_ii``
        (Rasmussen & Williams eq. 5.12) with fixed hyperparameters and standardization.
        """
        self._require_fitted()
        d = np.diag(self._kinv())
        mean_s = self._ys - self._alpha / d
        return self.y_mean + self.y_std * mean_s, self.y_std**2 / d

    def cv_r2(self) -> float | None:
        """LOO cross-validated ``R^2 = 1 - SSE_loo / SST``; ``None`` if ``n < 3`` or ``y`` is constant."""
        self._require_fitted()
        y = self.y_train
        if y.shape[0] < 3 or _is_constant(y):
            return None
        sst = float(np.sum((y - np.mean(y)) ** 2))
        if not sst > 0.0:
            return None
        mean, _ = self.loo()
        return 1.0 - float(np.sum((y - mean) ** 2)) / sst

    def log_marginal_likelihood(self) -> float:
        """Log marginal likelihood of the *standardized* targets at the fitted hyperparameters (no prior)."""
        self._require_fitted()
        return float(self._lml)

    def hyperparameters(self) -> dict:
        """``{"lengthscales", "outputscale", "noise" (standardized), "noise_var" (original units)}``."""
        self._require_fitted()
        return {
            "lengthscales": [float(v) for v in self._ls],
            "outputscale": float(self._s),
            "noise": float(self._noise),
            "noise_var": float(self._noise * self.y_std**2),
        }

    def sensitivity(self) -> np.ndarray:
        """Normalized inverse lengthscales ``(1/l_i) / sum_j (1/l_j)`` (sums to 1)."""
        self._require_fitted()
        inv = 1.0 / self._ls
        return inv / inv.sum()

    # ---- fantasies ----------------------------------------------------------------------
    def fantasize(self, X_new: np.ndarray, y_new: np.ndarray) -> "GP":
        """A new GP conditioned on the extra data with the same hyperparameters and standardization.

        No refit: the Cholesky factor is extended by a block update (falling back to a full
        refactorization with jitter escalation if the Schur complement is not PD). The
        original model is left untouched.
        """
        self._require_fitted()
        Xn = np.asarray(X_new, dtype=float)
        yn = np.asarray(y_new, dtype=float).reshape(-1)
        if Xn.ndim == 1:
            Xn = Xn.reshape(1, -1) if Xn.shape[0] == self.D else Xn.reshape(-1, 1)
        if Xn.ndim != 2 or Xn.shape[1] != self.D:
            raise LenzError(f"fantasize: expected X_new of shape (k, {self.D}), got {np.shape(X_new)}")
        if Xn.shape[0] != yn.shape[0]:
            raise LenzError(f"fantasize: X_new has {Xn.shape[0]} rows but y_new has {yn.shape[0]} values")
        if not (np.all(np.isfinite(Xn)) and np.all(np.isfinite(yn))):
            raise LenzError("fantasize: data must be finite")

        new = copy.copy(self)
        new.fit_info = {**self.fit_info, "fantasized": int(Xn.shape[0])}
        if Xn.shape[0] == 0:
            return new
        new.X_train = np.vstack([self.X_train, Xn])
        new.y_train = np.concatenate([self.y_train, yn])
        new._ys = (new.y_train - self.y_mean) / self.y_std

        ls, s, nv = self._ls, self._s, self._noise
        K_on = kernel_matrix(self.X_train, Xn, ls, s, self.kernel)
        K_nn = _add_diag(kernel_matrix(Xn, None, ls, s, self.kernel), nv + self._jitter)
        B = solve_triangular(self._L, K_on, lower=True, check_finite=False)
        C, info = dpotrf(K_nn - B.T @ B, lower=1, clean=1, overwrite_a=0)
        if info == 0:
            n, k = self._L.shape[0], Xn.shape[0]
            L = np.zeros((n + k, n + k))
            L[:n, :n] = self._L
            L[n:, :n] = B.T
            L[n:, n:] = C
            new._L = L
            new._refresh_solve()
        else:
            try:
                new._set_theta(self._theta)
            except np.linalg.LinAlgError:
                raise LenzError("fantasize: could not factorize the extended kernel matrix") from None
        return new
