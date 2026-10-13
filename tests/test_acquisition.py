import math

import numpy as np
import pytest
from scipy.special import log_ndtr, ndtr

from agentic_bo.lenz import acquisition as acqmod
from agentic_bo.lenz.acquisition import (
    ACQF_NAMES,
    GRADIENT_ACQFS,
    POINTWISE_ACQFS,
    build_acquisition,
    log1mexp,
    log_h,
    log_prob_feasible,
    parego_scalarize,
    parego_weights,
    prob_feasible,
)
from agentic_bo.lenz.pareto import default_ref_point, hypervolume, pareto_mask
from agentic_bo.utils import LenzError


# --------------------------------------------------------------------------- stub model


class StubModel:
    """Analytic surrogate with exact input gradients.

    mean(x) = shift + scale * (a . sin(b * x) + w . x)
    var(x)  = var_scale * (v0 + sum_i (d_i cos(e_i x_i))^2)
    """

    def __init__(self, D=3, seed=0, shift=0.0, scale=1.0, var_scale=1.0, n_train=8, v0=0.05):
        rng = np.random.default_rng(seed)
        self.a = rng.normal(size=D)
        self.b = rng.uniform(1.0, 3.0, size=D)
        self.w = rng.normal(size=D)
        self.d = rng.uniform(0.3, 1.0, size=D)
        self.e = rng.uniform(1.0, 4.0, size=D)
        self.v0, self.shift, self.scale, self.var_scale = v0, shift, scale, var_scale
        self.calls = 0
        self.X_train = rng.uniform(size=(n_train, D))
        self.y_train = self.posterior(self.X_train)[0] + 0.1 * scale * rng.normal(size=n_train)

    def posterior(self, Xs, full_cov=False, return_grad=False, observation_noise=False):
        self.calls += 1
        X = np.atleast_2d(np.asarray(Xs, float))
        mean = self.shift + self.scale * (np.sin(self.b * X) @ self.a + X @ self.w)
        ce, se = np.cos(self.e * X), np.sin(self.e * X)
        var = self.var_scale * (self.v0 + np.sum((self.d * ce) ** 2, axis=1))
        if full_cov:
            return mean, np.diag(var)
        if return_grad:
            dmean = self.scale * (self.a * self.b * np.cos(self.b * X) + self.w)
            dvar = self.var_scale * (-2.0 * self.d ** 2 * self.e * ce * se)
            return mean, var, dmean, dvar
        return mean, var

    def sample(self, Xs, n_samples, rng):
        mean, var = self.posterior(Xs)
        return mean + np.sqrt(var) * rng.standard_normal((n_samples, mean.size))


class NoDataModel(StubModel):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        del self.y_train


def fd_grad(f, Z, h=1e-6):
    G = np.zeros_like(Z)
    for j in range(Z.shape[1]):
        E = np.zeros_like(Z)
        E[:, j] = h
        G[:, j] = (f(Z + E) - f(Z - E)) / (2 * h)
    return G


def assert_grad_ok(acq, Z, rtol=2e-5, h=1e-6):
    v, g = acq.value_and_grad(Z)
    np.testing.assert_allclose(v, acq(Z), rtol=1e-12, atol=1e-12)
    G = fd_grad(acq, Z, h)
    scale = np.maximum(np.abs(G).max(axis=1, keepdims=True), 1e-8)
    assert np.all(np.isfinite(g))
    np.testing.assert_allclose(g / scale, G / scale, atol=rtol)


@pytest.fixture
def Z():
    return np.random.default_rng(42).uniform(0.05, 0.95, size=(7, 3))


# --------------------------------------------------------------------------- log_h numerics


def naive_log_h(z):
    return np.log(np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) + z * ndtr(z))


def asymptotic_log_h(z):
    iz2 = 1.0 / (z * z)
    return -0.5 * z * z - 0.5 * math.log(2 * math.pi) - 2 * np.log(-z) + np.log1p(-3 * iz2 + 15 * iz2 * iz2 - 105 * iz2 ** 3)


def test_log_h_matches_naive_where_naive_is_accurate():
    z = np.linspace(-8, 30, 20001)
    np.testing.assert_allclose(log_h(z), naive_log_h(z), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("z", [-30.0, -100.0, -1e3, -1e4, -1e6, -6.6e7, -6.8e7, -1e8, -1e12, -1e50])
def test_log_h_asymptotic_tail(z):
    val = log_h(np.array([z]))[0]
    assert np.isfinite(val)
    ref = asymptotic_log_h(np.array([z]))[0]
    tol = 1e-9 if z >= -30 else 1e-14
    assert abs(val - ref) <= tol * abs(ref) + 1e-9


def test_log_h_finite_monotone_extremes():
    z = -np.logspace(-3, 100, 400)[::-1]
    z = np.concatenate([z, np.linspace(0.0, 1e3, 200), [1e10, 1e100]])
    v = log_h(z)
    assert np.all(np.isfinite(v))
    assert np.all(np.diff(v) > 0)  # h is strictly increasing


def test_log_h_continuity_at_branch_points():
    for b in (-1.0, acqmod._NEG_INV_SQRT_EPS):
        eps = 1e-9 * abs(b)
        lo, hi = log_h(np.array([b - eps, b + eps]))
        slope = math.exp(acqmod._log_h_parts(np.array([b]))[1][0])
        assert abs(hi - lo - 2 * eps * slope) <= 1e-6 * max(1.0, abs(lo))


def test_log_h_gradient():
    z = np.concatenate([np.linspace(-40, 40, 801), [-0.999999, -1.000001]])
    lh, lr = acqmod._log_h_parts(z)
    h = 1e-6
    fd = (log_h(z + h) - log_h(z - h)) / (2 * h)
    np.testing.assert_allclose(np.exp(lr), fd, rtol=1e-6)
    # far tail: d/dz log h(z) = Phi/h -> |z| (1 + 2/z^2 - 6/z^4 + 42/z^6 - ...), accurate on both sides of
    # the series switch and of the -1/sqrt(eps) branch point
    zt = np.array([-50.0, -99.0, -101.0, -1e3, -1e4, -1e5, -1e7, -5e7, -1e8, -1e20])
    q = 1 / zt ** 2
    expected = -zt * (1 + q * (2 + q * (-6 + q * (42 - 414 * q))))
    np.testing.assert_allclose(np.exp(acqmod._log_h_parts(zt)[1]), expected, rtol=1e-11)


def test_log1mexp():
    x = -np.logspace(-12, 2.5, 300)
    near, far = x[x > -0.5], x[x < -1.0]
    np.testing.assert_allclose(log1mexp(near), np.log(-np.expm1(near)), rtol=1e-13)
    np.testing.assert_allclose(log1mexp(far), np.log1p(-np.exp(far)), rtol=1e-13)
    np.testing.assert_allclose(log1mexp(np.array([-1e-20])), [math.log(1e-20)], rtol=1e-12)
    np.testing.assert_allclose(log1mexp(np.array([-50.0])), [-math.exp(-50.0)], rtol=1e-12)
    assert log1mexp(np.array([-np.inf]))[0] == 0.0


def test_inv_mills():
    z = np.linspace(-30, 8, 500)
    np.testing.assert_allclose(acqmod._inv_mills(z), np.exp(acqmod.log_phi(z)) / ndtr(z), rtol=1e-10)
    zt = np.array([-1e4, -1e8])
    np.testing.assert_allclose(acqmod._inv_mills(zt), -zt, rtol=1e-7)


def test_log_prob_between():
    a = np.array([-np.inf, -1.0, 1.0, -3.0, 0.0, 2.0])
    b = np.array([0.5, 1.0, 2.0, np.inf, 0.1, 9.0])
    np.testing.assert_allclose(acqmod._log_prob_between(a, b), np.log(ndtr(b) - ndtr(a)), rtol=1e-12)
    # deep tails stay finite and accurate
    v = acqmod._log_prob_between(np.array([-41.0, 40.0]), np.array([-40.0, 41.0]))
    np.testing.assert_allclose(v, [log_ndtr(-40.0), log_ndtr(-40.0)], rtol=1e-12)


# --------------------------------------------------------------------------- analytic values


def test_values_match_closed_forms(Z):
    model = StubModel(seed=1)
    mu, var = model.posterior(Z)
    s = np.sqrt(var)
    best = float(np.max(model.y_train))
    z = (mu - best) / s
    ei = s * (z * ndtr(z) + np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi))
    np.testing.assert_allclose(build_acquisition("ei", [model], best_f=best)(Z), ei, rtol=1e-12)
    np.testing.assert_allclose(build_acquisition("logei", [model], best_f=best)(Z), np.log(ei), rtol=1e-12)
    np.testing.assert_allclose(build_acquisition("pi", [model], best_f=best)(Z), ndtr(z), rtol=1e-12)
    np.testing.assert_allclose(build_acquisition("logpi", [model], best_f=best)(Z), log_ndtr(z), rtol=1e-12)
    np.testing.assert_allclose(build_acquisition("ucb", [model])(Z), mu + 2.0 * s, rtol=1e-12)
    np.testing.assert_allclose(build_acquisition("ucb", [model], params={"beta": 0.25})(Z), mu + 0.5 * s)
    np.testing.assert_allclose(build_acquisition("pm", [model])(Z), mu)
    np.testing.assert_allclose(build_acquisition("std", [model])(Z), s)
    # a single 1-D point is accepted
    assert build_acquisition("pm", [model])(Z[0]).shape == (1,)


def test_best_f_defaults_to_max_observation(Z):
    model = StubModel(seed=2)
    a = build_acquisition("logei", [model])
    b = build_acquisition("logei", [model], best_f=float(np.max(model.y_train)))
    np.testing.assert_allclose(a(Z), b(Z))
    assert a.best_f == pytest.approx(np.max(model.y_train))
    with pytest.raises(LenzError, match="best_f"):
        build_acquisition("logei", [NoDataModel(seed=2)])


def test_noisy_logei_uses_max_posterior_mean(Z):
    model = StubModel(seed=3)
    Xo = np.random.default_rng(0).uniform(size=(5, 3))
    acq = build_acquisition("noisy_logei", [model], X_observed=Xo, best_f=123.0)
    inc = float(np.max(model.posterior(Xo)[0]))
    assert acq.best_f == pytest.approx(inc)
    np.testing.assert_allclose(acq(Z), build_acquisition("logei", [model], best_f=inc)(Z))
    default = build_acquisition("noisy_logei", [model])
    assert default.best_f == pytest.approx(float(np.max(model.posterior(model.X_train)[0])))


def test_logei_far_below_incumbent_is_finite_and_informative():
    model = StubModel(seed=4, var_scale=1e-4)
    Z = np.random.default_rng(1).uniform(0.1, 0.9, size=(6, 3))
    mu, var = model.posterior(Z)
    for zt in (1e3, 1e4):
        best = float(np.max(mu) + zt * np.sqrt(np.max(var)))
        logei = build_acquisition("logei", [model], best_f=best)
        v, g = logei.value_and_grad(Z)
        assert np.all(np.isfinite(v)) and np.all(np.isfinite(g))
        assert np.all(np.abs(g).sum(axis=1) > 0)
        assert np.all(build_acquisition("ei", [model], best_f=best)(Z) == 0.0)  # plain EI underflows
        # ranking is preserved: logEI orders points like z = (mu - best) / sigma in the tail
        z = (mu - best) / np.sqrt(var)
        assert np.array_equal(np.argsort(v), np.argsort(z))
        assert_grad_ok(logei, Z, rtol=1e-4)


# --------------------------------------------------------------------------- constraints


def _constraint_set(kind, seed=10):
    c1 = StubModel(seed=seed, shift=0.3, scale=0.8)
    c2 = StubModel(seed=seed + 1, shift=-0.2, scale=0.5, var_scale=0.3)
    return {
        "none": [],
        "upper": [(c1, None, 0.4)],
        "lower": [(c1, -0.1, None)],
        "both": [(c1, -0.3, 0.6)],
        "two": [(c1, None, 0.5), (c2, -0.4, 0.3)],
    }[kind]


def direct_log_pof(cons, Z):
    total = np.zeros(Z.shape[0])
    for m, lo, hi in cons:
        mu, var = m.posterior(Z)
        s = np.sqrt(var)
        pa = ndtr((lo - mu) / s) if lo is not None else 0.0
        pb = ndtr((hi - mu) / s) if hi is not None else 1.0
        total += np.log(pb - pa)
    return total


@pytest.mark.parametrize("kind", ["upper", "lower", "both", "two"])
def test_pof_values(kind, Z):
    cons = _constraint_set(kind)
    np.testing.assert_allclose(log_prob_feasible(cons, Z), direct_log_pof(cons, Z), rtol=1e-10)
    np.testing.assert_allclose(prob_feasible(cons, Z), np.exp(direct_log_pof(cons, Z)), rtol=1e-10)
    lp, dlp = log_prob_feasible(cons, Z, return_grad=True)
    np.testing.assert_allclose(dlp, fd_grad(lambda X: log_prob_feasible(cons, X), Z), rtol=1e-5, atol=1e-7)
    assert np.all(log_prob_feasible([], Z) == 0)


def test_pof_far_from_feasible_has_useful_gradient():
    c = StubModel(seed=5, shift=50.0, scale=0.1, var_scale=0.01)
    Z = np.random.default_rng(3).uniform(0.1, 0.9, size=(4, 3))
    cons = [(c, None, 0.0)]  # mean ~ 50, sd ~ 0.1 => z ~ -500
    lp, dlp = log_prob_feasible(cons, Z, return_grad=True)
    assert np.all(np.isfinite(lp)) and np.all(lp < -1e4)
    np.testing.assert_allclose(dlp, fd_grad(lambda X: log_prob_feasible(cons, X), Z, h=1e-7), rtol=1e-4)


def test_constrained_values(Z):
    model = StubModel(seed=6)
    cons = _constraint_set("two")
    lp = direct_log_pof(cons, Z)
    best = float(np.max(model.y_train))
    for name in ("logei", "logpi"):
        plain = build_acquisition(name, [model], best_f=best)(Z)
        np.testing.assert_allclose(build_acquisition(name, [model], cons, best_f=best)(Z), plain + lp, rtol=1e-10)
        # feasibility phase: only the PoF term
        feas = build_acquisition(name, [model], cons, best_f=None)
        assert feas.feasibility_phase
        np.testing.assert_allclose(feas(Z), lp, rtol=1e-10)
    for name in ("ei", "pi"):
        plain = build_acquisition(name, [model], best_f=best)(Z)
        np.testing.assert_allclose(build_acquisition(name, [model], cons, best_f=best)(Z), plain * np.exp(lp), rtol=1e-10)
        np.testing.assert_allclose(build_acquisition(name, [model], cons)(Z), np.exp(lp), rtol=1e-10)
    # ucb / pm / std: standardized objective units plus log PoF
    y_mean, y_std = np.mean(model.y_train), np.std(model.y_train)
    mu, var = model.posterior(Z)
    s = np.sqrt(var)
    np.testing.assert_allclose(build_acquisition("pm", [model], cons)(Z), (mu - y_mean) / y_std + lp, rtol=1e-10)
    np.testing.assert_allclose(build_acquisition("std", [model], cons)(Z), s / y_std + lp, rtol=1e-10)
    np.testing.assert_allclose(build_acquisition("ucb", [model], cons, params={"beta": 9.0})(Z),
                               (mu + 3 * s - y_mean) / y_std + lp, rtol=1e-10)


GRAD_NAMES = ["logei", "noisy_logei", "ei", "logpi", "pi", "ucb", "pm", "std", "parego"]


@pytest.mark.parametrize("cons_kind", ["none", "upper", "lower", "both", "two"])
@pytest.mark.parametrize("name", GRAD_NAMES)
def test_fd_gradients(name, cons_kind, Z):
    model = StubModel(seed=7, shift=1.0, scale=2.0)
    cons = _constraint_set(cons_kind)
    best = float(np.max(model.y_train))
    acq = build_acquisition(name, [model], cons, best_f=best)
    assert acq.has_gradient
    assert_grad_ok(acq, Z)


@pytest.mark.parametrize("name", ["logei", "noisy_logei", "ei", "logpi", "pi", "parego"])
def test_fd_gradients_feasibility_phase(name, Z):
    model = StubModel(seed=8)
    acq = build_acquisition(name, [model], _constraint_set("two"), best_f=None)
    assert acq.feasibility_phase
    assert_grad_ok(acq, Z)


def test_fd_gradient_with_variance_floor():
    model = StubModel(seed=9, v0=0.0)
    # var = sum (d cos(e x))^2 vanishes where every cos(e_i x_i) = 0
    z0 = np.pi / (2 * model.e)
    Z = np.vstack([z0, z0 + 1e-3, np.full(3, 0.4)])
    for name in ("logei", "ucb", "std"):
        v, g = build_acquisition(name, [model], best_f=0.0).value_and_grad(Z)
        assert np.all(np.isfinite(v)) and np.all(np.isfinite(g))


def test_scaled_units_are_consistent(Z):
    # logei shifts by log(scale) when the objective is rescaled; logpi is invariant
    m1 = StubModel(seed=11)
    m2 = StubModel(seed=11, scale=1e3, var_scale=1e6)
    b1 = float(np.max(m1.posterior(m1.X_train)[0]))
    a1 = build_acquisition("logei", [m1], best_f=b1)(Z)
    a2 = build_acquisition("logei", [m2], best_f=1e3 * b1)(Z)
    np.testing.assert_allclose(a2, a1 + math.log(1e3), rtol=1e-10)
    np.testing.assert_allclose(build_acquisition("logpi", [m2], best_f=1e3 * b1)(Z),
                               build_acquisition("logpi", [m1], best_f=b1)(Z), rtol=1e-10)


@pytest.mark.parametrize("scale", [1e-10, 1e-14])
def test_tiny_output_scale_keeps_variance_floor_relative(scale, Z):
    # Regression: the variance floor / standardization scale used an absolute std floor
    # (1e-9), so for a tiny-scale objective sigma was floored far above the posterior std
    # (logEI degenerated to the posterior mean) and constrained ucb/pm were divided by 1
    # instead of std(y) (the objective vanished next to log PoF).
    m1 = StubModel(seed=11)
    m2 = StubModel(seed=11, scale=scale, var_scale=scale ** 2)
    b1 = float(np.max(m1.posterior(m1.X_train)[0]))
    a1 = build_acquisition("logei", [m1], best_f=b1)(Z)
    a2 = build_acquisition("logei", [m2], best_f=scale * b1)(Z)
    np.testing.assert_allclose(a2, a1 + math.log(scale), rtol=1e-8)
    cons = [(StubModel(seed=3), None, 0.5)]
    for name in ("pm", "ucb", "std"):
        np.testing.assert_allclose(build_acquisition(name, [m2], cons)(Z),
                                   build_acquisition(name, [m1], cons)(Z), rtol=1e-8, atol=1e-10)
    # the same holds for a tiny-scale constraint metric (its variance floor)
    c1, c2 = StubModel(seed=3), StubModel(seed=3, scale=scale, var_scale=scale ** 2)
    for lo, hi in ((None, 0.5), (-0.5, 0.5)):
        np.testing.assert_allclose(
            log_prob_feasible([(c2, None if lo is None else scale * lo, scale * hi)], Z),
            log_prob_feasible([(c1, lo, hi)], Z), rtol=1e-8, atol=1e-10)


# --------------------------------------------------------------------------- factory errors


def test_names_tuple():
    assert isinstance(ACQF_NAMES, tuple)
    for n in ("logei", "noisy_logei", "ei", "logpi", "pi", "ucb", "pm", "std", "ts", "ehvi", "parego", "sobol", "random"):
        assert n in ACQF_NAMES
    assert set(GRADIENT_ACQFS) < set(POINTWISE_ACQFS) < set(ACQF_NAMES)
    assert "ehvi" in POINTWISE_ACQFS and "ehvi" not in GRADIENT_ACQFS


def test_case_insensitive():
    m = StubModel()
    assert build_acquisition(" LogEI ", [m]).name == "logei"


@pytest.mark.parametrize("name,match", [("bogus", "unknown acquisition"), ("ts", "Thompson"),
                                        ("sobol", "model-free"), ("random", "model-free")])
def test_unbuildable_names(name, match):
    with pytest.raises(LenzError, match=match):
        build_acquisition(name, [StubModel()])


def test_wrong_number_of_models():
    m1, m2, m3 = StubModel(seed=1), StubModel(seed=2), StubModel(seed=3)
    with pytest.raises(LenzError, match="exactly 2 objectives"):
        build_acquisition("ehvi", [m1])
    with pytest.raises(LenzError, match="parego"):
        build_acquisition("ehvi", [m1, m2, m3])
    with pytest.raises(LenzError, match="scalarized"):
        build_acquisition("parego", [m1, m2])
    with pytest.raises(LenzError, match="single-objective"):
        build_acquisition("logei", [m1, m2])
    with pytest.raises(LenzError, match="exactly 1"):
        build_acquisition("ucb", [])


@pytest.mark.parametrize("cons", [
    [(StubModel(), None, None)],
    [(StubModel(), 1.0, 0.5)],
    [(StubModel(), 1.0, 1.0)],
    [(StubModel(), "a", None)],
    [(StubModel(), np.nan, None)],
    [(StubModel(), 1.0)],
])
def test_bad_constraints(cons):
    with pytest.raises(LenzError):
        build_acquisition("logei", [StubModel()], cons, best_f=0.0)


def test_bad_params():
    m = StubModel()
    with pytest.raises(LenzError, match="beta"):
        build_acquisition("ucb", [m], params={"beta": -1.0})
    with pytest.raises(LenzError, match="beta"):
        build_acquisition("ucb", [m], params={"beta": "x"})
    with pytest.raises(LenzError, match="best_f"):
        build_acquisition("logei", [m], best_f=float("nan"))
    with pytest.raises(NotImplementedError):
        build_acquisition("ehvi", [m, StubModel(seed=5)], params={"ref_point": [-5, -5]}).value_and_grad(np.zeros((1, 3)))


# --------------------------------------------------------------------------- EHVI


def _ehvi_setup(seed=0):
    m1 = StubModel(seed=seed, n_train=10)
    m2 = StubModel(seed=seed + 100, n_train=10)
    Y = np.column_stack([m1.y_train, m2.y_train])
    return m1, m2, Y


def test_ehvi_equals_brute_force_expectation(Z):
    m1, m2, Y = _ehvi_setup()
    acq = build_acquisition("ehvi", [m1, m2], Y_observed=Y, rng=np.random.default_rng(5))
    assert not acq.has_gradient
    assert acq.base_samples.shape == (128, 2)
    ref = default_ref_point(Y)
    np.testing.assert_allclose(acq.ref_point, ref)
    front = Y[pareto_mask(Y)]
    hv0 = hypervolume(front, ref)
    vals = acq(Z)
    for i, z in enumerate(Z):
        mu = np.array([m1.posterior(z)[0][0], m2.posterior(z)[0][0]])
        sd = np.sqrt([m1.posterior(z)[1][0], m2.posterior(z)[1][0]])
        samples = mu + sd * acq.base_samples
        brute = np.mean([hypervolume(np.vstack([front, s]), ref) - hv0 for s in samples])
        assert vals[i] == pytest.approx(brute, rel=1e-10, abs=1e-12)
    assert np.all(vals >= 0)


def test_ehvi_empty_front_matches_closed_form():
    # with an empty front EHVI = E[(y1 - r1)^+] E[(y2 - r2)^+] = prod sigma_k h((mu_k - r_k) / sigma_k)
    m1, m2, _ = _ehvi_setup(1)
    Z = np.random.default_rng(2).uniform(size=(5, 3))
    ref = np.array([-0.5, 0.2])
    acq = build_acquisition("ehvi", [m1, m2], Y_observed=np.zeros((0, 2)),
                            params={"ref_point": ref, "n_samples": 1 << 14}, rng=1)
    expected = np.ones(5)
    for k, m in enumerate((m1, m2)):
        mu, var = m.posterior(Z)
        s = np.sqrt(var)
        expected *= s * np.exp(log_h((mu - ref[k]) / s))
    np.testing.assert_allclose(acq(Z), expected, rtol=5e-3)


def test_ehvi_deterministic_and_defaults(Z):
    m1, m2, Y = _ehvi_setup(2)
    a = build_acquisition("ehvi", [m1, m2], Y_observed=Y, rng=3)
    b = build_acquisition("ehvi", [m1, m2], Y_observed=Y, rng=3)
    np.testing.assert_array_equal(a(Z), b(Z))
    # Y_observed defaults to the stacked training targets
    c = build_acquisition("ehvi", [m1, m2], rng=3)
    np.testing.assert_array_equal(a(Z), c(Z))
    with pytest.raises(LenzError, match="ref_point"):
        build_acquisition("ehvi", [m1, m2], Y_observed=np.zeros((0, 2)))
    with pytest.raises(LenzError, match="ref_point"):
        build_acquisition("ehvi", [m1, m2], Y_observed=Y, params={"ref_point": [0.0]})
    with pytest.raises(LenzError, match="Y_observed"):
        build_acquisition("ehvi", [m1, m2], Y_observed=np.zeros((4, 3)))


def test_ehvi_constraints_and_feasibility_phase(Z):
    m1, m2, Y = _ehvi_setup(3)
    cons = _constraint_set("both")
    pof = np.exp(direct_log_pof(cons, Z))
    plain = build_acquisition("ehvi", [m1, m2], Y_observed=Y, rng=0)(Z)
    con = build_acquisition("ehvi", [m1, m2], cons, Y_observed=Y, rng=0)
    np.testing.assert_allclose(con(Z), plain * pof, rtol=1e-10)
    feas = build_acquisition("ehvi", [m1, m2], cons, Y_observed=np.zeros((0, 2)), rng=0)
    assert feas.feasibility_phase
    np.testing.assert_allclose(feas(Z), pof, rtol=1e-10)


def test_ehvi_zero_for_dominated_confident_points():
    m1, m2, _ = _ehvi_setup(4)
    m1.var_scale = m2.var_scale = 1e-12
    Z = np.random.default_rng(0).uniform(size=(4, 3))
    mu = np.column_stack([m1.posterior(Z)[0], m2.posterior(Z)[0]])
    front = np.array([[mu[:, 0].max() + 1, mu[:, 1].max() + 1]])
    acq = build_acquisition("ehvi", [m1, m2], Y_observed=front, params={"ref_point": mu.min(0) - 1})
    np.testing.assert_allclose(acq(Z), 0.0, atol=1e-9)


# --------------------------------------------------------------------------- ParEGO


def test_parego_scalarize_hand_computed():
    Y = np.array([[0.0, 10.0], [1.0, 20.0], [0.5, 15.0], [1.0, 10.0]])
    w = np.array([1.0, 3.0])  # normalized to [0.25, 0.75]
    s = parego_scalarize(Y, w, rho=0.1)
    Yh = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    wy = Yh * np.array([0.25, 0.75])
    np.testing.assert_allclose(s, wy.min(1) + 0.1 * wy.sum(1))
    assert np.argmax(s) == 1


def test_parego_scalarize_edge_cases():
    Y = np.array([[1.0, 5.0], [2.0, 5.0], [np.nan, 1.0]])
    s = parego_scalarize(Y, [0.5, 0.5])
    # constant second column maps to 1; NaN rows give NaN
    np.testing.assert_allclose(s[:2], [0.0 + 0.05 * 0.5, 0.5 + 0.05 * 1.0])
    assert np.isnan(s[2])
    w = parego_weights(3, 0)
    assert w.shape == (3,) and np.all(w >= 0) and w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(parego_scalarize(np.ones((2, 3)), None, rng=0),
                               parego_scalarize(np.ones((2, 3)), parego_weights(3, 0)))
    for bad in ([1.0], [-1.0, 2.0], [0.0, 0.0], [np.nan, 1.0]):
        with pytest.raises(LenzError):
            parego_scalarize(Y, bad)
    with pytest.raises(LenzError):
        parego_scalarize(Y, [0.5, 0.5], rho=-1)


def test_parego_acquisition_is_logei_on_scalarized_model(Z):
    m = StubModel(seed=12)
    best = float(np.max(m.y_train))
    p = build_acquisition("parego", [m], best_f=best, params={"weights": [0.3, 0.7]})
    assert p.name == "parego"
    np.testing.assert_allclose(p(Z), build_acquisition("logei", [m], best_f=best)(Z))


# --------------------------------------------------------------------------- real GP (optional)


def test_with_real_gp_if_available():
    gp_mod = pytest.importorskip("agentic_bo.lenz.gp")
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(15, 3))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 - 0.5 * X[:, 2]
    gp = gp_mod.GP(seed=0).fit(X, y)
    c = gp_mod.GP(seed=1).fit(X, X.sum(axis=1))
    Zt = rng.uniform(0.05, 0.95, size=(6, 3))
    for name in ("logei", "ei", "logpi", "ucb", "pm", "std", "noisy_logei"):
        assert_grad_ok(build_acquisition(name, [gp], best_f=float(y.max())), Zt, rtol=1e-4)
    assert_grad_ok(build_acquisition("logei", [gp], [(c, None, 1.5)], best_f=float(y.max())), Zt, rtol=1e-4)
