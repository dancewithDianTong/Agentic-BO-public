import time

import numpy as np
import pytest
from scipy.linalg import cho_solve, cholesky, solve_triangular

from agentic_bo.lenz.acquisition import build_acquisition
from agentic_bo.lenz.optimize import optimize_acqf, project_box, thompson_sample, trust_region
from agentic_bo.space import SearchSpace
from agentic_bo.utils import LenzError


# --------------------------------------------------------------------------- stubs


class Quadratic:
    """acq(z) = -sum(((z - c) / s)^2), maximized at c."""

    name = "quad"

    def __init__(self, c, s=None, has_gradient=True):
        self.c = np.asarray(c, float)
        self.s = np.ones_like(self.c) if s is None else np.asarray(s, float)
        self.has_gradient = has_gradient
        self.n_points = 0

    def __call__(self, Z):
        Z = np.atleast_2d(Z)
        self.n_points += Z.shape[0]
        return -np.sum(((Z - self.c) / self.s) ** 2, axis=1)

    def value_and_grad(self, Z):
        if not self.has_gradient:
            raise NotImplementedError
        Z = np.atleast_2d(Z)
        return self(Z), -2 * (Z - self.c) / self.s ** 2


class Bumps:
    """Sum of Gaussian bumps (multimodal), with gradient."""

    name = "bumps"
    has_gradient = True

    def __init__(self, centers, heights, width):
        self.C = np.asarray(centers, float)
        self.h = np.asarray(heights, float)
        self.w = float(width)

    def _parts(self, Z):
        Z = np.atleast_2d(Z)
        diff = Z[:, None, :] - self.C[None]
        e = self.h * np.exp(-0.5 * np.sum(diff ** 2, axis=2) / self.w ** 2)
        return Z, diff, e

    def __call__(self, Z):
        return self._parts(Z)[2].sum(axis=1)

    def value_and_grad(self, Z):
        Z, diff, e = self._parts(Z)
        return e.sum(axis=1), -np.einsum("mk,mkd->md", e, diff) / self.w ** 2


class RBFGP:
    """Exact GP with an RBF kernel and fixed hyperparameters (GP-like stub with gradients)."""

    def __init__(self, X, y, ls=0.3, noise=1e-4):
        self.X_train = np.asarray(X, float)
        self.y_train = np.asarray(y, float)
        self.ls = np.full(self.X_train.shape[1], ls) if np.ndim(ls) == 0 else np.asarray(ls, float)
        self.ym, self.ys = self.y_train.mean(), max(self.y_train.std(), 1e-9)
        K = self._k(self.X_train, self.X_train) + noise * np.eye(len(self.y_train))
        self.L = cholesky(K, lower=True)
        self.alpha = cho_solve((self.L, True), (self.y_train - self.ym) / self.ys)
        self.sample_calls = []

    def _k(self, A, B):
        r2 = np.sum(((A[:, None, :] - B[None, :, :]) / self.ls) ** 2, axis=2)
        return np.exp(-0.5 * r2)

    def posterior(self, Xs, full_cov=False, return_grad=False, observation_noise=False):
        X = np.atleast_2d(np.asarray(Xs, float))
        Ks = self._k(X, self.X_train)
        V = solve_triangular(self.L, Ks.T, lower=True)
        mean = self.ym + self.ys * (Ks @ self.alpha)
        if full_cov:
            cov = self._k(X, X) - V.T @ V
            return mean, self.ys ** 2 * cov
        var = self.ys ** 2 * np.maximum(1.0 - np.sum(V ** 2, axis=0), 1e-12)
        if not return_grad:
            return mean, var
        dK = -Ks[:, :, None] * (X[:, None, :] - self.X_train[None]) / self.ls ** 2
        W = solve_triangular(self.L.T, V, lower=False)
        dmean = self.ys * np.einsum("mnd,n->md", dK, self.alpha)
        dvar = -2 * self.ys ** 2 * np.einsum("mnd,nm->md", dK, W)
        return mean, var, dmean, dvar

    def sample(self, Xs, n_samples, rng):
        mean, cov = self.posterior(Xs, full_cov=True)
        Lc = np.linalg.cholesky(cov + 1e-8 * self.ys ** 2 * np.eye(len(mean)))
        out = mean + (Lc @ rng.standard_normal((len(mean), n_samples))).T
        self.sample_calls.append((np.array(Xs, copy=True), out.copy()))
        return out


def cont_space(D):
    return SearchSpace.from_spec([{"name": f"x{i}", "type": "real", "lower": 0, "upper": 1} for i in range(D)])


MIXED_SPEC = [
    {"name": "x", "type": "real", "lower": -2.0, "upper": 2.0},
    {"name": "k", "type": "int", "lower": 1, "upper": 9},
    {"name": "opt", "type": "categorical", "choices": ["a", "b", "c"]},
    {"name": "w", "type": "ordinal", "values": [8, 16, 32, 64]},
    {"name": "lr", "type": "real", "lower": 1e-3, "upper": 1.0, "log_scale": True},
]


def rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# --------------------------------------------------------------------------- optimize_acqf


@pytest.mark.parametrize("has_gradient", [True, False])
def test_finds_interior_maximizer(has_gradient):
    space = cont_space(4)
    c = np.array([0.2, 0.7, 0.45, 0.9])
    acq = Quadratic(c, s=[1.0, 0.5, 2.0, 1.0], has_gradient=has_gradient)
    z, v = optimize_acqf(acq, space, np.zeros(4), np.ones(4), rng=0, raw_samples=256, num_restarts=4)
    np.testing.assert_allclose(z, c, atol=1e-5 if has_gradient else 1e-4)
    assert v == pytest.approx(acq(z)[0])
    assert v > -1e-8


def test_boundary_and_subbox():
    space = cont_space(3)
    lb, ub = np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.35])
    c = np.array([0.0, 0.45, 1.0])
    z, v = optimize_acqf(Quadratic(c), space, lb, ub, rng=1, raw_samples=128, num_restarts=3)
    np.testing.assert_allclose(z, [0.1, 0.45, 0.35], atol=1e-6)
    assert np.all(z >= lb) and np.all(z <= ub)


def test_fixed_dims():
    space = cont_space(4)
    lb = np.array([0.0, 0.3, 0.0, 0.8])
    ub = np.array([1.0, 0.3, 1.0, 0.8])
    acq = Quadratic([0.6, 0.9, 0.25, 0.1])
    z, _ = optimize_acqf(acq, space, lb, ub, rng=2, raw_samples=64, num_restarts=2)
    np.testing.assert_allclose(z, [0.6, 0.3, 0.25, 0.8], atol=1e-6)
    # fully fixed box
    z, v = optimize_acqf(acq, space, lb + 0.0 * ub, lb, rng=2, raw_samples=8, num_restarts=2)
    np.testing.assert_allclose(z, lb)


def test_multimodal_global_maximum():
    space = cont_space(3)
    centers = [[0.15, 0.2, 0.8], [0.7, 0.75, 0.3], [0.5, 0.1, 0.5]]
    acq = Bumps(centers, heights=[1.0, 1.4, 1.2], width=0.08)
    z, v = optimize_acqf(acq, space, np.zeros(3), np.ones(3), rng=3)
    assert rms(z, centers[1]) < 1e-3
    assert v == pytest.approx(1.4, rel=1e-3)


def test_deterministic_given_rng():
    space = cont_space(5)
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(20, 5))
    gp = RBFGP(X, np.sin(4 * X).sum(axis=1))
    acq = build_acquisition("logei", [gp])
    a = optimize_acqf(acq, space, np.zeros(5), np.ones(5), rng=np.random.default_rng(7), X_baseline=X)
    b = optimize_acqf(acq, space, np.zeros(5), np.ones(5), rng=np.random.default_rng(7), X_baseline=X)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1]


def test_baseline_perturbations_find_narrow_peak():
    space = cont_space(6)
    c = np.full(6, 0.37)
    acq = Bumps([c], heights=[1.0], width=0.01)  # invisible to a handful of Sobol points
    Xb = np.vstack([np.random.default_rng(0).uniform(size=(9, 6)), c + 0.01])
    z, v = optimize_acqf(acq, space, np.zeros(6), np.ones(6), rng=4, raw_samples=32, num_restarts=3,
                         X_baseline=Xb)
    np.testing.assert_allclose(z, c, atol=1e-4)
    z0, v0 = optimize_acqf(acq, space, np.zeros(6), np.ones(6), rng=4, raw_samples=32, num_restarts=3)
    assert v0 < 1e-3  # without the baseline the peak is missed (sanity check of the test)


def test_exclusion_returns_alternative():
    space = cont_space(2)
    c = np.array([0.3, 0.6])
    acq = Quadratic(c)
    z, v = optimize_acqf(acq, space, np.zeros(2), np.ones(2), rng=5, exclude=c[None, :], exclude_tol=1e-3)
    assert rms(z, c) > 1e-3
    assert v > -1e-2  # still close to the maximum
    # default tolerance: an excluded row elsewhere does not matter
    z2, _ = optimize_acqf(acq, space, np.zeros(2), np.ones(2), rng=5, exclude=np.array([[0.9, 0.9]]))
    np.testing.assert_allclose(z2, c, atol=1e-5)


def test_exclusion_without_alternative_returns_best():
    space = SearchSpace.from_spec([{"name": "o", "type": "categorical", "choices": ["a", "b"]}])
    lb, ub = space.encoded_bounds({"o": ["b"]})
    acq = Quadratic([0.0, 1.0])
    z, v = optimize_acqf(acq, space, lb, ub, rng=0, exclude=np.array([[0.0, 1.0]]), raw_samples=16)
    np.testing.assert_allclose(z, [0.0, 1.0])
    # two representable points: the excluded optimum is skipped
    lb, ub = space.encoded_bounds()
    z, v = optimize_acqf(acq, space, lb, ub, rng=0, exclude=np.array([[0.0, 1.0]]), raw_samples=16)
    np.testing.assert_allclose(z, [1.0, 0.0])


def test_mixed_space_returns_representable_optimum():
    space = SearchSpace.from_spec(MIXED_SPEC)
    target = {"x": 0.5, "k": 4, "opt": "b", "w": 32, "lr": 0.05}
    zt = space.encode(target)
    acq = Quadratic(zt, s=np.full(space.D, 0.5))
    lb, ub = space.encoded_bounds()
    z, v = optimize_acqf(acq, space, lb, ub, rng=6, raw_samples=256, num_restarts=5)
    np.testing.assert_allclose(space.project(z), z, atol=1e-12)
    cfg = space.decode(z)
    assert cfg["k"] == 4 and cfg["opt"] == "b" and cfg["w"] == 32
    assert cfg["x"] == pytest.approx(0.5, abs=1e-4) and cfg["lr"] == pytest.approx(0.05, rel=1e-3)


def test_mixed_space_continuous_dims_polished_after_projection():
    # The relaxed optimum has a fractional one-hot block (p, q) = (0.6, 0.55) with x = 0.56;
    # projecting snaps to category p, whose best x is 0.2. The continuous dim must be
    # re-optimized for the chosen category.
    space = SearchSpace.from_spec([{"name": "x", "type": "real", "lower": 0, "upper": 1},
                                   {"name": "c", "type": "categorical", "choices": ["p", "q"]}])

    class Coupled:
        name, has_gradient = "coupled", True

        def __call__(self, Z):
            return self.value_and_grad(Z)[0]

        def value_and_grad(self, Z):
            Z = np.atleast_2d(Z)
            x, p, q = Z[:, 0], Z[:, 1], Z[:, 2]
            r = x - (0.2 * p + 0.8 * q)
            v = -r ** 2 - (p - 0.6) ** 2 - (q - 0.55) ** 2
            g = np.column_stack([-2 * r, 0.4 * r - 2 * (p - 0.6), 1.6 * r - 2 * (q - 0.55)])
            return v, g

    z, v = optimize_acqf(Coupled(), space, np.zeros(3), np.ones(3), rng=0, raw_samples=64, num_restarts=4)
    cfg = space.decode(z)
    assert cfg["c"] == "p" and cfg["x"] == pytest.approx(0.2, abs=1e-6)
    assert v == pytest.approx(-0.16 - 0.3025, abs=1e-10)


def test_disallowed_category_never_returned():
    space = SearchSpace.from_spec(MIXED_SPEC)
    zt = space.encode({"x": 0.5, "k": 4, "opt": "b", "w": 32, "lr": 0.05})
    acq = Quadratic(zt)
    lb, ub = space.encoded_bounds({"opt": ["a", "c"], "k": [6, 9]})
    z, _ = optimize_acqf(acq, space, lb, ub, rng=7, raw_samples=128, num_restarts=3)
    cfg = space.decode(z)
    assert cfg["opt"] in ("a", "c") and 6 <= cfg["k"] <= 9
    assert np.all(z >= lb - 1e-12) and np.all(z <= ub + 1e-12)
    np.testing.assert_allclose(space.project(z), z, atol=1e-12)


def test_trust_region_box_with_discrete_dims():
    space = SearchSpace.from_spec(MIXED_SPEC)
    center = space.encode({"x": 0.0, "k": 5, "opt": "c", "w": 16, "lr": 0.1})
    lb0, ub0 = space.encoded_bounds()
    lb, ub = trust_region(center, 0.05, lb0, ub0)
    acq = Quadratic(space.encode({"x": 2.0, "k": 9, "opt": "a", "w": 64, "lr": 1.0}))
    z, _ = optimize_acqf(acq, space, lb, ub, rng=8, raw_samples=128, num_restarts=3)
    assert np.all(z >= lb - 1e-12) and np.all(z <= ub + 1e-12)
    cfg = space.decode(z)
    assert cfg["opt"] == "c" and cfg["k"] == 5 and cfg["w"] == 16
    np.testing.assert_allclose(space.project(z), z, atol=1e-12)


def test_rejects_bad_inputs():
    space = cont_space(2)
    with pytest.raises(LenzError):
        optimize_acqf(Quadratic([0, 0]), space, np.zeros(3), np.ones(3), rng=0)
    with pytest.raises(LenzError):
        optimize_acqf(Quadratic([0, 0]), space, np.array([0.5, 0.0]), np.array([0.4, 1.0]), rng=0)
    with pytest.raises(LenzError, match="X_baseline"):
        optimize_acqf(Quadratic([0, 0]), space, np.zeros(2), np.ones(2), rng=0, X_baseline=np.ones((4, 3)))
    with pytest.raises(LenzError, match="exclude"):
        optimize_acqf(Quadratic([0, 0]), space, np.zeros(2), np.ones(2), rng=0, exclude=np.ones((2, 1)))


def test_ehvi_derivative_free_optimization():
    D = 3
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(15, D))
    y1, y2 = -np.sum((X - 0.2) ** 2, axis=1), -np.sum((X - 0.8) ** 2, axis=1)
    g1, g2 = RBFGP(X, y1, ls=0.4), RBFGP(X, y2, ls=0.4)
    acq = build_acquisition("ehvi", [g1, g2], Y_observed=np.column_stack([y1, y2]), rng=0)
    assert not acq.has_gradient
    lb, ub = np.full(D, 0.1), np.full(D, 0.9)
    z, v = optimize_acqf(acq, cont_space(D), lb, ub, rng=0, X_baseline=X, exclude=X)
    assert np.all((z >= lb) & (z <= ub))
    assert v == pytest.approx(acq(z)[0]) and v > 0
    raw = lb + rng.uniform(size=(2000, D)) * (ub - lb)
    assert v >= acq(raw).max() - 1e-9


def test_logei_on_gp_stub_fast_d20():
    D, n = 20, 200
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(n, D))
    y = -np.sum((X - 0.4) ** 2, axis=1) + 0.3 * np.sin(5 * X[:, 0])
    gp = RBFGP(X, y, ls=0.8)
    acq = build_acquisition("logei", [gp], best_f=float(y.max()))
    space = cont_space(D)
    optimize_acqf(acq, space, np.zeros(D), np.ones(D), rng=0, raw_samples=64, num_restarts=1, maxiter=5)  # warm-up
    t0 = time.perf_counter()
    z, v = optimize_acqf(acq, space, np.zeros(D), np.ones(D), rng=1, X_baseline=X, exclude=X)
    elapsed = time.perf_counter() - t0
    assert elapsed < 4.0, f"optimize_acqf took {elapsed:.2f}s"
    assert np.all((z >= 0) & (z <= 1))
    # at least as good as the best raw candidate and the best observed point
    raw = rng.uniform(size=(1024, D))
    assert v >= acq(raw).max() - 1e-9
    assert v >= acq(X).max() - 1e-9
    assert v == pytest.approx(acq(z)[0])


# --------------------------------------------------------------------------- project_box


def test_project_box_snaps_discrete_dims_inside_box():
    space = SearchSpace.from_spec(MIXED_SPEC)
    lb, ub = space.encoded_bounds()
    k = space.slices["k"].start
    # k in [3.2, 3.9] (raw): no integer inside -> clipped; [3.2, 4.4] -> snaps to 4 although 3.4 rounds to 3
    for (lo, hi), expect in (((3.2, 3.9), None), ((3.2, 4.4), 4)):
        lb2, ub2 = lb.copy(), ub.copy()
        lb2[k], ub2[k] = (lo - 1) / 8, (hi - 1) / 8
        z = np.full(space.D, 0.5)
        z[k] = (3.4 - 1) / 8
        P = project_box(space, z, lb2, ub2)
        assert lb2[k] <= P[k] <= ub2[k]
        if expect is not None:
            assert space.decode(P)["k"] == expect
            np.testing.assert_allclose(space.project(P), P, atol=1e-12)
    # ordinal w (encoded 0, 1/3, 2/3, 1) in box [0.4, 0.9] -> only 2/3 representable
    w = space.slices["w"].start
    lb2, ub2 = lb.copy(), ub.copy()
    lb2[w], ub2[w] = 0.4, 0.9
    z = np.full(space.D, 0.5)
    z[w] = 0.95
    assert project_box(space, z, lb2, ub2)[w] == pytest.approx(2 / 3)
    # categorical block inside a trust-region box around 'b' only allows 'b'
    sl = space.slices["opt"]
    lb2, ub2 = lb.copy(), ub.copy()
    lb2[sl], ub2[sl] = [0.0, 0.9, 0.0], [0.1, 1.0, 0.1]
    z = np.full(space.D, 0.5)
    z[sl] = [0.1, 0.9, 0.1]
    np.testing.assert_allclose(project_box(space, z, lb2, ub2)[sl], [0.0, 1.0, 0.0])


def test_project_box_batch_and_continuous():
    space = cont_space(3)
    Z = np.array([[-0.5, 0.5, 2.0], [0.2, 0.3, 0.4]])
    P = project_box(space, Z, np.full(3, 0.1), np.full(3, 0.9))
    np.testing.assert_allclose(P, [[0.1, 0.5, 0.9], [0.2, 0.3, 0.4]])


# --------------------------------------------------------------------------- thompson_sample


def test_thompson_sample_returns_argmax_of_sample():
    D = 3
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(12, D))
    gp = RBFGP(X, np.cos(3 * X).sum(axis=1))
    space = cont_space(D)
    lb, ub = np.array([0.1, 0.0, 0.2]), np.array([0.8, 0.5, 0.9])
    z, v = thompson_sample(gp, space, lb, ub, rng=np.random.default_rng(3), n_candidates=256, X_baseline=X)
    Xs, f = gp.sample_calls[-1]
    assert Xs.shape[1] == D and len(Xs) <= 256 + 5 and np.unique(Xs, axis=0).shape[0] == len(Xs)
    i = int(np.argmax(f[0]))
    np.testing.assert_array_equal(z, Xs[i])
    assert v == f[0, i]
    assert np.all(z >= lb) and np.all(z <= ub)
    assert np.all(Xs >= lb) and np.all(Xs <= ub)
    z2, v2 = thompson_sample(gp, space, lb, ub, rng=np.random.default_rng(3), n_candidates=256, X_baseline=X)
    np.testing.assert_array_equal(z, z2)


def test_thompson_sample_exclusion_and_discrete_space():
    space = SearchSpace.from_spec([{"name": "k", "type": "int", "lower": 0, "upper": 3},
                                   {"name": "o", "type": "categorical", "choices": ["a", "b"]}])
    X = space.encode_many([{"k": 0, "o": "a"}, {"k": 3, "o": "b"}, {"k": 1, "o": "b"}])
    gp = RBFGP(X, np.array([0.0, 1.0, 0.5]), ls=0.5)
    lb, ub = space.encoded_bounds()
    excl = space.encode_many([{"k": k, "o": o} for k in range(4) for o in "ab" if (k, o) != (2, "a")])
    z, v = thompson_sample(gp, space, lb, ub, rng=0, n_candidates=64, exclude=excl)
    assert space.decode(z) == {"k": 2, "o": "a"}
    Xs, _ = gp.sample_calls[-1]
    assert len(Xs) == 1  # de-duplicated representable candidates minus the excluded ones


def test_thompson_sample_with_constraints():
    D = 2
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(15, D))
    obj = RBFGP(X, X[:, 0] + X[:, 1], ls=0.5)
    con = RBFGP(X, X[:, 0], ls=0.5)  # feasible iff x0 <= 0.3
    space = cont_space(D)
    z, _ = thompson_sample(obj, space, np.zeros(D), np.ones(D), rng=2, n_candidates=256,
                           constraint_models=[(con, None, 0.3)])
    Xs, f = obj.sample_calls[-1]
    _, c = con.sample_calls[-1]
    feasible = c[0] <= 0.3
    assert feasible.any()
    i = np.flatnonzero(feasible)[np.argmax(f[0][feasible])]
    np.testing.assert_array_equal(z, Xs[i])


# --------------------------------------------------------------------------- trust_region


def test_trust_region_isotropic():
    lb, ub = np.zeros(3), np.ones(3)
    lo, hi = trust_region([0.5, 0.05, 0.98], 0.1, lb, ub)
    np.testing.assert_allclose(lo, [0.4, 0.0, 0.88])
    np.testing.assert_allclose(hi, [0.6, 0.15, 1.0])


def test_trust_region_lengthscale_weights():
    ls = np.array([1.0, 4.0, 0.25, 100.0])
    c = np.full(4, 0.5)
    lo, hi = trust_region(c, 0.01, np.zeros(4), np.ones(4), lengthscales=ls)
    w = ls / np.exp(np.mean(np.log(ls)))
    np.testing.assert_allclose((hi - lo) / 2, 0.01 * np.clip(w, 0.25, 4.0))
    assert np.all(lo <= c) and np.all(c <= hi)
    # unclipped weights have geometric mean 1
    ls2 = np.array([0.5, 1.0, 2.0])
    lo, hi = trust_region(np.full(3, 0.5), 0.02, np.zeros(3), np.ones(3), lengthscales=ls2)
    half = (hi - lo) / 2
    assert np.exp(np.mean(np.log(half / 0.02))) == pytest.approx(1.0)
    np.testing.assert_allclose(half, 0.02 * ls2)


def test_trust_region_intersects_active_region():
    lb, ub = np.array([0.2, 0.0]), np.array([0.6, 0.3])
    lo, hi = trust_region([0.9, 0.1], 0.2, lb, ub)  # center outside the region gets clipped
    assert np.all(lo >= lb) and np.all(hi <= ub) and np.all(lo <= hi)
    np.testing.assert_allclose(lo, [0.4, 0.0])
    np.testing.assert_allclose(hi, [0.6, 0.3])
    lo, hi = trust_region([0.3, 0.1], 0.0, lb, ub)
    np.testing.assert_allclose(lo, hi)


@pytest.mark.parametrize("kwargs", [{"radius": -0.1}, {"radius": float("nan")},
                                    {"lengthscales": [1.0, -1.0]}, {"lengthscales": [1.0]}])
def test_trust_region_errors(kwargs):
    args = {"center_z": [0.5, 0.5], "radius": 0.1, "lb": np.zeros(2), "ub": np.ones(2)}
    args.update(kwargs)
    with pytest.raises(LenzError):
        trust_region(**args)


class Scaled:
    """``scale * acq + offset`` (same maximizer), forwarding gradients."""

    def __init__(self, acq, scale, offset=0.0):
        self.acq, self.scale, self.offset = acq, scale, offset
        self.name = acq.name
        self.has_gradient = acq.has_gradient

    def __call__(self, Z):
        return self.scale * self.acq(Z) + self.offset

    def value_and_grad(self, Z):
        v, g = self.acq.value_and_grad(Z)
        return self.scale * v + self.offset, self.scale * g


@pytest.mark.parametrize("scale,offset,has_gradient", [
    (1e-6, 0.0, True), (1e-6, 0.0, False), (1e-9, 3.0, True), (1e6, -5e6, True), (1e6, 0.0, False)])
def test_local_optimization_is_invariant_to_acquisition_units(scale, offset, has_gradient):
    # Regression: L-BFGS-B ran on raw acquisition values, so its absolute tolerances
    # (pgtol 1e-5, ftol relative to max(|f|, 1)) stopped every restart at its start when
    # the acquisition lived in tiny raw units (ucb / pm / ei of a metric of order 1e-5).
    space = cont_space(3)
    c = np.array([0.31, 0.62, 0.47])
    base = Quadratic(c, s=[1.0, 0.5, 2.0], has_gradient=has_gradient)
    z1, _ = optimize_acqf(base, space, np.zeros(3), np.ones(3), rng=0, raw_samples=32, num_restarts=3)
    z2, v2 = optimize_acqf(Scaled(base, scale, offset), space, np.zeros(3), np.ones(3), rng=0,
                           raw_samples=32, num_restarts=3)
    tol = 1e-5 if has_gradient else 1e-4
    np.testing.assert_allclose(z1, c, atol=tol)
    np.testing.assert_allclose(z2, c, atol=tol)
    assert v2 == pytest.approx(scale * base(z2)[0] + offset, rel=1e-12, abs=1e-12 * abs(offset))


def test_gp_acquisitions_invariant_to_objective_scale():
    # the GP fit is scale-free (standardization); the optimizer must be too
    gp_mod = pytest.importorskip("agentic_bo.lenz.gp")
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(12, 2))
    y = -((X[:, 0] - 0.7) ** 2 + (X[:, 1] - 0.3) ** 2)
    space = cont_space(2)
    for name in ("pm", "ucb", "ei"):
        zs = []
        for scale in (1.0, 1e-6):
            gp = gp_mod.GP(seed=0).fit(X, scale * y)
            acq = build_acquisition(name, [gp])
            zs.append(optimize_acqf(acq, space, np.zeros(2), np.ones(2), rng=1, raw_samples=64)[0])
        assert rms(zs[0], zs[1]) < 1e-3, (name, zs)


# --------------------------------------------------------------------------- real GP (optional)


def test_real_gp_end_to_end_if_available():
    gp_mod = pytest.importorskip("agentic_bo.lenz.gp")
    space = SearchSpace.from_spec([{"name": "a", "type": "real", "lower": -5, "upper": 10},
                                   {"name": "b", "type": "real", "lower": 0, "upper": 15}])
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(12, 2))

    def branin(Z):
        a, b = -5 + 15 * Z[:, 0], 15 * Z[:, 1]
        return -((b - 5.1 / (4 * np.pi ** 2) * a ** 2 + 5 / np.pi * a - 6) ** 2
                 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(a) + 10)

    y = branin(X)
    gp = gp_mod.GP(seed=0).fit(X, y)
    lb, ub = space.encoded_bounds()
    for name in ("logei", "ucb", "ehvi"):
        if name == "ehvi":
            gp2 = gp_mod.GP(seed=1).fit(X, -X[:, 0])
            acq = build_acquisition("ehvi", [gp, gp2], Y_observed=np.column_stack([y, -X[:, 0]]))
        else:
            acq = build_acquisition(name, [gp], best_f=float(y.max()))
        z, v = optimize_acqf(acq, space, lb, ub, rng=0, X_baseline=X, exclude=X)
        assert np.all((z >= lb) & (z <= ub))
        assert v >= acq(X).max() - 1e-9
    z, v = thompson_sample(gp, space, lb, ub, rng=0, X_baseline=X, exclude=X)
    assert np.all((z >= lb) & (z <= ub)) and np.isfinite(v)
