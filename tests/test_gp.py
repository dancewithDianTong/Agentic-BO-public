"""Tests for the exact GP surrogate (agentic_bo.lenz.gp)."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from scipy.stats import norm

from agentic_bo.lenz.gp import (
    GP,
    KERNELS,
    _cholesky,
    hyperprior,
    kernel_from_sqdist,
    kernel_grad_x,
    kernel_matrix,
    neg_log_posterior,
    scaled_sqdist,
)
from agentic_bo.utils import LenzError

# --------------------------------------------------------------------------- references


def ref_kernel(X1, X2, ls, s, kind):
    """Direct broadcasting implementation of the ARD kernels."""
    d = (X1[:, None, :] - X2[None, :, :]) / ls
    r2 = np.sum(d**2, axis=-1)
    if kind == "matern52":
        r = np.sqrt(r2)
        return s * (1 + math.sqrt(5) * r + 5.0 / 3.0 * r2) * np.exp(-math.sqrt(5) * r)
    return s * np.exp(-0.5 * r2)


def ref_nlp(theta, X, y, kind, noise=None):
    n, D = X.shape
    ls = np.exp(theta[:D])
    s = np.exp(theta[D])
    nv = np.exp(theta[D + 1]) if noise is None else noise
    K = ref_kernel(X, X, ls, s, kind) + nv * np.eye(n)
    _, logdet = np.linalg.slogdet(K)
    lml = -0.5 * y @ np.linalg.solve(K, y) - 0.5 * logdet - 0.5 * n * math.log(2 * math.pi)
    mu, sigma, _, _ = hyperprior(D, noise is None)
    return -(lml + np.sum(norm.logpdf(theta, mu, sigma)))


def ref_posterior(gp, Xs, X=None, y=None):
    """Posterior (orig units) computed from scratch with gp's hyperparameters/standardization."""
    hp = gp.hyperparameters()
    X = gp.X_train if X is None else X
    y = gp.y_train if y is None else y
    ls, s, nv = np.array(hp["lengthscales"]), hp["outputscale"], hp["noise"]
    ys = (y - gp.y_mean) / gp.y_std
    K = ref_kernel(X, X, ls, s, gp.kernel) + (nv + gp._jitter) * np.eye(len(X))
    Ks = ref_kernel(Xs, X, ls, s, gp.kernel)
    Kss = ref_kernel(Xs, Xs, ls, s, gp.kernel)
    mean = Ks @ np.linalg.solve(K, ys)
    cov = Kss - Ks @ np.linalg.solve(K, Ks.T)
    return gp.y_mean + gp.y_std * mean, gp.y_std**2 * cov


def branin_like(X):
    return np.sin(6 * X[:, 0]) + 0.5 * X[:, 0]


def smooth3(X):
    return np.sin(3 * X[:, 0]) + np.cos(2 * X[:, 1]) * X[:, 2] + 0.5 * X[:, 1] ** 2


@pytest.fixture
def data3():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(25, 3))
    y = smooth3(X) + 0.01 * rng.standard_normal(25)
    return X, y


# --------------------------------------------------------------------------- kernels


@pytest.mark.parametrize("kind", KERNELS)
def test_kernel_matrix_matches_reference(kind):
    rng = np.random.default_rng(0)
    X1, X2 = rng.uniform(size=(7, 3)), rng.uniform(size=(5, 3))
    ls, s = np.array([0.3, 1.2, 0.07]), 2.5
    np.testing.assert_allclose(kernel_matrix(X1, X2, ls, s, kind), ref_kernel(X1, X2, ls, s, kind), rtol=1e-12)
    K = kernel_matrix(X1, None, ls, s, kind)
    np.testing.assert_allclose(K, ref_kernel(X1, X1, ls, s, kind), rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(np.diag(K), s)
    np.testing.assert_array_equal(K, K.T)
    assert kernel_matrix(X1[:1], None, ls, s, kind).shape == (1, 1)
    assert kernel_matrix(X1[:0], X2, ls, s, kind).shape == (0, 5)


@pytest.mark.parametrize("kind", KERNELS)
def test_kernel_dr2_and_grad_x_finite_differences(kind):
    r2 = np.array([0.0, 1e-6, 0.3, 2.0, 9.0])
    k, dk = kernel_from_sqdist(r2, kind, 1.7)
    h = 1e-7
    kp, _ = kernel_from_sqdist(r2 + h, kind, 1.7)
    km, _ = kernel_from_sqdist(np.maximum(r2 - h, 0), kind, 1.7)
    fd = np.where(r2 > h, (kp - km) / (2 * h), (kp - k) / h)
    np.testing.assert_allclose(dk, fd, rtol=1e-4, atol=1e-6)

    rng = np.random.default_rng(3)
    X1, X2 = rng.uniform(size=(4, 2)), rng.uniform(size=(6, 2))
    ls, s = np.array([0.4, 0.9]), 1.3
    G = kernel_grad_x(X1, X2, ls, s, kind)
    assert G.shape == (4, 6, 2)
    eps = 1e-6
    for d in range(2):
        e = np.zeros(2)
        e[d] = eps
        fd = (kernel_matrix(X1 + e, X2, ls, s, kind) - kernel_matrix(X1 - e, X2, ls, s, kind)) / (2 * eps)
        np.testing.assert_allclose(G[:, :, d], fd, rtol=1e-6, atol=1e-9)


def test_matern_grad_formula_from_design():
    """DESIGN: dk/dx_i = -s (5/3)(1 + sqrt5 r) exp(-sqrt5 r) (x_i - x'_i) / l_i^2."""
    x, xp = np.array([[0.2, 0.7]]), np.array([[0.5, 0.1]])
    ls, s = np.array([0.3, 0.8]), 2.0
    r = math.sqrt(np.sum(((x - xp) / ls) ** 2))
    expected = -s * 5 / 3 * (1 + math.sqrt(5) * r) * math.exp(-math.sqrt(5) * r) * (x - xp)[0] / ls**2
    np.testing.assert_allclose(kernel_grad_x(x, xp, ls, s)[0, 0], expected, rtol=1e-12)


def test_scaled_sqdist_symmetric_zero_diag():
    X = np.random.default_rng(0).uniform(size=(6, 4))
    R = scaled_sqdist(X, None, np.full(4, 0.01))
    assert np.all(np.diag(R) == 0) and np.all(R >= 0)
    np.testing.assert_allclose(R, scaled_sqdist(X, X, np.full(4, 0.01)), rtol=1e-12, atol=1e-9)


# --------------------------------------------------------------------------- objective


@pytest.mark.parametrize("kind", KERNELS)
@pytest.mark.parametrize("noise", [None, 0.05])
def test_neg_log_posterior_value_and_gradient(kind, noise):
    rng = np.random.default_rng(42)
    n, D = 15, 3
    X = rng.uniform(size=(n, D))
    y = rng.standard_normal(n)
    mu, sigma, lo, hi = hyperprior(D, noise is None)
    for trial in range(3):
        theta = np.clip(mu + 0.5 * sigma * rng.standard_normal(mu.size), lo, hi)
        theta[:D] = np.log(rng.uniform(0.1, 2.0, D))  # informative lengthscales
        val, grad = neg_log_posterior(theta, X, y, kind, noise)
        assert val == pytest.approx(ref_nlp(theta, X, y, kind, noise), rel=1e-9, abs=1e-9)
        h = 1e-6
        fd = np.empty_like(theta)
        for j in range(theta.size):
            e = np.zeros_like(theta)
            e[j] = h
            fd[j] = (neg_log_posterior(theta + e, X, y, kind, noise)[0]
                     - neg_log_posterior(theta - e, X, y, kind, noise)[0]) / (2 * h)
        np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-5 * max(1.0, np.abs(fd).max()))


def test_neg_log_posterior_gradient_short_lengthscales():
    rng = np.random.default_rng(7)
    X = rng.uniform(size=(20, 2))
    y = rng.standard_normal(20)
    theta = np.log([0.03, 0.05, 1.5, 1e-3])
    _, grad = neg_log_posterior(theta, X, y, "matern52")
    h = 1e-6
    fd = np.array([(neg_log_posterior(theta + h * e, X, y)[0] - neg_log_posterior(theta - h * e, X, y)[0]) / (2 * h)
                   for e in np.eye(4)])
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-5)


def test_hyperprior_values():
    mu, sigma, lo, hi = hyperprior(4, learn_noise=True, form="median")
    assert mu.size == 6
    np.testing.assert_allclose(mu[:4], math.sqrt(2) + 0.5 * math.log(4))
    np.testing.assert_allclose(sigma[:4], math.sqrt(3))
    assert (mu[4], sigma[4], mu[5], sigma[5]) == (0.0, 1.0, -4.0, 1.0)
    # default "botorch" form: LogNormal evaluated on l and noise => location shifted by -sigma^2
    mu_b, sigma_b, lo_b, hi_b = hyperprior(4, learn_noise=True)
    np.testing.assert_allclose(mu_b[:4], math.sqrt(2) + 0.5 * math.log(4) - 3.0)
    assert (mu_b[4], mu_b[5]) == (0.0, -5.0)
    np.testing.assert_allclose(sigma_b, sigma)
    np.testing.assert_allclose(np.r_[lo_b, hi_b], np.r_[lo, hi])
    assert hyperprior(4, learn_noise=False, form="botorch")[0][4] == 0.0
    with pytest.raises(LenzError):
        hyperprior(4, form="gpytorch")
    np.testing.assert_allclose(np.exp([lo[0], hi[0], lo[4], hi[4], lo[5], hi[5]]), [1e-2, 1e3, 1e-2, 1e2, 1e-6, 1.0])
    assert hyperprior(4, learn_noise=False)[0].size == 5


def test_botorch_prior_matches_lognormal_density_on_parameters():
    """In theta = log(l) the "botorch" log-prior equals log LogNormal(l; mu0, s) up to a constant."""
    from scipy.stats import lognorm
    D = 3
    mu0 = math.sqrt(2) + 0.5 * math.log(D)
    mu, sigma, _, _ = hyperprior(D, learn_noise=False)
    thetas = np.linspace(-3.0, 3.0, 7)
    ours = -0.5 * ((thetas - mu[0]) / sigma[0]) ** 2
    ref = lognorm.logpdf(np.exp(thetas), s=math.sqrt(3), scale=math.exp(mu0))
    np.testing.assert_allclose(ours - ours[0], ref - ref[0], atol=1e-10)


def test_gp_prior_option_changes_fit_and_validates():
    rng = np.random.default_rng(3)
    X = rng.uniform(size=(12, 4))
    y = np.sin(6 * X[:, 0]) + 0.1 * X[:, 1]
    ls_b = GP(prior="botorch", seed=0).fit(X, y).hyperparameters()["lengthscales"]
    ls_m = GP(prior="median", seed=0).fit(X, y).hyperparameters()["lengthscales"]
    assert np.median(ls_b) < np.median(ls_m)
    with pytest.raises(LenzError):
        GP(prior="flat")


# --------------------------------------------------------------------------- fit


def test_fit_improves_on_starts_and_is_deterministic(data3):
    X, y = data3
    gp = GP(seed=5).fit(X, y)
    ys = (y - gp.y_mean) / gp.y_std
    theta_hat = gp._theta
    mu, _, lo, hi = hyperprior(3)
    val_hat = neg_log_posterior(theta_hat, X, ys)[0]
    assert val_hat <= neg_log_posterior(np.clip(mu, lo, hi), X, ys)[0] + 1e-9
    assert gp.fit_info["n_starts"] == 3
    assert gp.fit_info["neg_log_posterior"] == pytest.approx(val_hat, rel=1e-10)
    gp2 = GP(seed=5).fit(X, y)
    assert gp2.hyperparameters() == gp.hyperparameters()
    assert np.all(theta_hat >= lo - 1e-12) and np.all(theta_hat <= hi + 1e-12)


def test_fit_with_warm_start_and_no_optimization(data3):
    X, y = data3
    gp = GP().fit(X, y)
    hp = gp.hyperparameters()
    gp_w = GP(n_restarts=1).fit(X, y, warm_start=hp)
    for key in ("outputscale", "noise"):
        assert gp_w.hyperparameters()[key] == pytest.approx(hp[key], rel=1e-3)
    # n_restarts=0: condition on the given hyperparameters exactly
    fixed = {"lengthscales": [0.3, 0.4, 0.5], "outputscale": 2.0, "noise": 0.01}
    gp0 = GP(n_restarts=0).fit(X, y, warm_start=fixed)
    hp0 = gp0.hyperparameters()
    np.testing.assert_allclose(hp0["lengthscales"], fixed["lengthscales"], rtol=1e-12)
    assert hp0["outputscale"] == pytest.approx(2.0) and hp0["noise"] == pytest.approx(0.01)
    # out-of-bounds / malformed warm starts are clipped or ignored
    gp_bad = GP(n_restarts=0).fit(X, y, warm_start={"lengthscales": [1e9, -1.0], "outputscale": 1e9})
    hpb = gp_bad.hyperparameters()
    mu = hyperprior(3)[0]
    np.testing.assert_allclose(hpb["lengthscales"], np.exp(mu[:3]))  # wrong length -> prior median
    assert hpb["outputscale"] == pytest.approx(100.0)
    with pytest.raises(LenzError):
        GP().fit(X, y, warm_start=[1, 2, 3])


def test_fixed_noise():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(12, 2))
    y = np.sin(4 * X[:, 0]) + X[:, 1]
    gp = GP(noise=0.02).fit(X, y)
    hp = gp.hyperparameters()
    assert hp["noise"] == 0.02
    assert hp["noise_var"] == pytest.approx(0.02 * gp.y_std**2)
    assert gp._theta.size == 3
    gp0 = GP(noise=0.0).fit(X, y)
    _, v = gp0.posterior(X)
    assert np.all(v > 0)
    np.testing.assert_allclose(gp0.posterior(X)[0], y, atol=1e-4)


@pytest.mark.parametrize("bad", [dict(kernel="linear"), dict(noise="fixed"), dict(noise=-1.0),
                                 dict(noise=float("nan")), dict(noise=[1, 2])])
def test_constructor_validation(bad):
    with pytest.raises(LenzError):
        GP(**bad)


def test_input_validation():
    gp = GP()
    with pytest.raises(LenzError):
        gp.posterior(np.zeros((1, 2)))  # not fitted
    with pytest.raises(LenzError):
        gp.fit(np.zeros((0, 2)), np.zeros(0))
    with pytest.raises(LenzError):
        gp.fit(np.zeros((3, 2)), np.zeros(2))
    with pytest.raises(LenzError):
        gp.fit(np.array([[0.1, np.nan]]), np.array([1.0]))
    with pytest.raises(LenzError):
        gp.fit(np.zeros((2, 2)), np.array([1.0, np.inf]))
    gp.fit(np.random.default_rng(0).uniform(size=(5, 2)), np.arange(5.0))
    with pytest.raises(LenzError):
        gp.posterior(np.zeros((3, 3)))
    with pytest.raises(LenzError):
        gp.posterior(np.zeros((3, 2)), full_cov=True, return_grad=True)
    with pytest.raises(LenzError):
        gp.fantasize(np.zeros((2, 3)), np.zeros(2))
    with pytest.raises(LenzError):
        gp.fantasize(np.zeros((2, 2)), np.zeros(3))
    with pytest.raises(LenzError):
        gp.sample(np.zeros((2, 2)), -1, 0)


# --------------------------------------------------------------------------- posterior


@pytest.mark.parametrize("kind", KERNELS)
def test_posterior_matches_reference(kind, data3):
    X, y = data3
    gp = GP(kernel=kind).fit(X, y)
    Xs = np.random.default_rng(9).uniform(size=(8, 3))
    m_ref, cov_ref = ref_posterior(gp, Xs)
    m, v = gp.posterior(Xs)
    np.testing.assert_allclose(m, m_ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(v, np.diag(cov_ref), rtol=1e-6, atol=1e-12)
    m2, cov = gp.posterior(Xs, full_cov=True)
    np.testing.assert_allclose(m2, m)
    np.testing.assert_allclose(cov, cov_ref, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(np.diag(cov), v, rtol=1e-10)
    nv = gp.hyperparameters()["noise_var"]
    _, vn = gp.posterior(Xs, observation_noise=True)
    np.testing.assert_allclose(vn, v + nv, rtol=1e-12)
    _, covn = gp.posterior(Xs, full_cov=True, observation_noise=True)
    np.testing.assert_allclose(covn, cov + nv * np.eye(8), rtol=1e-12)
    # single point given as a 1-D array
    m1, v1 = gp.posterior(Xs[0])
    assert m1.shape == (1,) and v1.shape == (1,)
    assert m1[0] == pytest.approx(m[0])


@pytest.mark.parametrize("kind", KERNELS)
def test_posterior_gradients_finite_differences(kind, data3):
    X, y = data3
    gp = GP(kernel=kind).fit(X, y)
    rng = np.random.default_rng(11)
    Xs = np.vstack([rng.uniform(size=(6, 3)), X[2] + 1e-3, X[5]])
    m, v, dm, dv = gp.posterior(Xs, return_grad=True)
    m0, v0 = gp.posterior(Xs)
    np.testing.assert_allclose(m, m0)
    np.testing.assert_allclose(v, v0)
    assert dm.shape == dv.shape == (8, 3)
    h = 1e-6
    for d in range(3):
        e = np.zeros(3)
        e[d] = h
        mp, vp = gp.posterior(Xs + e)
        mm, vm = gp.posterior(Xs - e)
        np.testing.assert_allclose(dm[:, d], (mp - mm) / (2 * h), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dv[:, d], (vp - vm) / (2 * h), rtol=1e-5, atol=1e-7)
    # observation noise does not change gradients
    _, _, dm2, dv2 = gp.posterior(Xs, return_grad=True, observation_noise=True)
    np.testing.assert_allclose(dm2, dm)
    np.testing.assert_allclose(dv2, dv)


@pytest.mark.parametrize("kind", KERNELS)
def test_posterior_gradients_fd_scaled_outputs(kind):
    """Gradients are in original units (y scale 1e3, offset -50)."""
    rng = np.random.default_rng(2)
    X = rng.uniform(size=(15, 2))
    y = 1e3 * (np.sin(5 * X[:, 0]) * X[:, 1]) - 50
    gp = GP(kernel=kind).fit(X, y)
    Xs = rng.uniform(size=(5, 2))
    _, _, dm, dv = gp.posterior(Xs, return_grad=True)
    h = 1e-6
    for d in range(2):
        e = np.zeros(2)
        e[d] = h
        mp, vp = gp.posterior(Xs + e)
        mm, vm = gp.posterior(Xs - e)
        np.testing.assert_allclose(dm[:, d], (mp - mm) / (2 * h), rtol=1e-5, atol=1e-6 * gp.y_std)
        np.testing.assert_allclose(dv[:, d], (vp - vm) / (2 * h), rtol=1e-5, atol=1e-6 * gp.y_std**2)


def test_far_away_variance_reverts_to_prior(data3):
    X, y = data3
    gp = GP().fit(X, y)
    hp = gp.hyperparameters()
    m, v = gp.posterior(np.full((1, 3), 1e3))
    assert m[0] == pytest.approx(gp.y_mean, abs=1e-8)
    assert v[0] == pytest.approx(hp["outputscale"] * gp.y_std**2, rel=1e-8)


# --------------------------------------------------------------------------- LOO / CV


@pytest.mark.parametrize("kind", KERNELS)
def test_loo_matches_explicit_leave_one_out(kind, data3):
    X, y = data3
    gp = GP(kernel=kind).fit(X, y)
    mean_loo, var_loo = gp.loo()
    n = len(y)
    for i in range(n):
        keep = np.arange(n) != i
        m_i, cov_i = ref_posterior(gp, X[i:i + 1], X[keep], y[keep])
        assert mean_loo[i] == pytest.approx(m_i[0], rel=1e-6, abs=1e-8)
        expected_var = cov_i[0, 0] + gp.hyperparameters()["noise_var"] + gp._jitter * gp.y_std**2
        assert var_loo[i] == pytest.approx(expected_var, rel=1e-6)
    sse = np.sum((y - mean_loo) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    assert gp.cv_r2() == pytest.approx(1 - sse / sst)


def test_interpolation_1d():
    X = np.linspace(0, 1, 12)[:, None]
    y = branin_like(X)
    Xt = np.linspace(0, 1, 201)[:, None]
    # learned noise (prior median e^-4 in standardized units) smooths slightly
    gp = GP().fit(X, y)
    m, v = gp.posterior(Xt)
    assert np.max(np.abs(m - branin_like(Xt))) < 0.03
    assert np.all(v > 0)
    assert gp.cv_r2() > 0.95
    z = np.abs(m - branin_like(Xt)) / np.sqrt(v)
    assert np.mean(z < 4) > 0.95  # roughly calibrated
    # (near) noise-free model interpolates the data
    gp = GP(noise=1e-6).fit(X, y)
    m, v = gp.posterior(Xt)
    assert np.max(np.abs(m - branin_like(Xt))) < 0.01
    np.testing.assert_allclose(gp.posterior(X)[0], y, atol=1e-4)
    assert gp.cv_r2() > 0.99


@pytest.mark.parametrize("kind", KERNELS)
def test_accuracy_3d(kind):
    from agentic_bo.space import sobol_points

    X = sobol_points(48, 3, 0)
    y = smooth3(X)
    gp = GP(kernel=kind).fit(X, y)
    Xt = np.random.default_rng(1).uniform(size=(500, 3))
    m, _ = gp.posterior(Xt)
    yt = smooth3(Xt)
    r2 = 1 - np.sum((m - yt) ** 2) / np.sum((yt - yt.mean()) ** 2)
    assert r2 > 0.98
    assert gp.cv_r2() > 0.95
    assert np.isfinite(gp.log_marginal_likelihood())


def test_cv_r2_none_cases():
    gp = GP().fit(np.array([[0.1], [0.9]]), np.array([1.0, 2.0]))
    assert gp.cv_r2() is None
    gp = GP().fit(np.array([[0.1], [0.5], [0.9]]), np.array([3.0, 3.0, 3.0]))
    assert gp.cv_r2() is None


def test_lml_matches_reference(data3):
    X, y = data3
    gp = GP().fit(X, y)
    hp = gp.hyperparameters()
    ys = (y - gp.y_mean) / gp.y_std
    K = ref_kernel(X, X, np.array(hp["lengthscales"]), hp["outputscale"], "matern52") \
        + (hp["noise"] + gp._jitter) * np.eye(len(X))
    expected = -0.5 * ys @ np.linalg.solve(K, ys) - 0.5 * np.linalg.slogdet(K)[1] - 0.5 * len(X) * math.log(2 * math.pi)
    assert gp.log_marginal_likelihood() == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- fantasize


@pytest.mark.parametrize("kind", KERNELS)
def test_fantasize_equals_conditioning(kind, data3):
    X, y = data3
    gp = GP(kernel=kind).fit(X, y)
    before = gp.posterior(X[:3])
    rng = np.random.default_rng(4)
    Xn = rng.uniform(size=(3, 3))
    yn = smooth3(Xn) + 5.0  # far from current data mean on purpose
    fg = gp.fantasize(Xn, yn)
    assert fg.n_train == len(y) + 3
    assert fg.hyperparameters() == gp.hyperparameters()
    assert (fg.y_mean, fg.y_std) == (gp.y_mean, gp.y_std)
    Xs = rng.uniform(size=(6, 3))
    m_ref, cov_ref = ref_posterior(gp, Xs, np.vstack([X, Xn]), np.concatenate([y, yn]))
    m, cov = fg.posterior(Xs, full_cov=True)
    np.testing.assert_allclose(m, m_ref, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(cov, cov_ref, rtol=1e-6, atol=1e-11)
    # identical to a fixed-hyperparameter fit on the combined data with the same constants
    _, _, dm, dv = fg.posterior(Xs, return_grad=True)
    assert np.all(np.isfinite(dm)) and np.all(np.isfinite(dv))
    # original untouched
    after = gp.posterior(X[:3])
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    assert gp.n_train == len(y)
    # LOO on the fantasized model uses the extended factor
    lm, lv = fg.loo()
    assert lm.shape == (len(y) + 3,) and np.all(lv > 0)
    # empty fantasy and 1-D single point
    assert gp.fantasize(np.zeros((0, 3)), np.zeros(0)).n_train == len(y)
    assert gp.fantasize(Xn[0], yn[:1]).n_train == len(y) + 1


def test_fantasize_duplicate_point_falls_back_or_updates():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(6, 2))
    y = X.sum(1)
    gp = GP(noise=0.0).fit(X, y)
    fg = gp.fantasize(X[:2], y[:2])  # exact duplicates with zero noise
    m, v = fg.posterior(X)
    assert np.all(np.isfinite(m)) and np.all(v > 0)
    np.testing.assert_allclose(m, y, atol=1e-3)


def test_kriging_believer_fantasy_reduces_variance_only(data3):
    X, y = data3
    gp = GP().fit(X, y)
    xq = np.array([[0.5, 0.5, 0.5]])
    mq, vq = gp.posterior(xq)
    fg = gp.fantasize(xq, mq)
    m2, v2 = fg.posterior(xq)
    assert m2[0] == pytest.approx(mq[0], abs=1e-8)
    assert v2[0] < vq[0]


# --------------------------------------------------------------------------- sampling


def test_sample_moments_match_posterior(data3):
    X, y = data3
    gp = GP().fit(X, y)
    Xs = np.vstack([np.random.default_rng(0).uniform(size=(4, 3)), X[0]])
    mean, cov = gp.posterior(Xs, full_cov=True)
    S = gp.sample(Xs, 40000, np.random.default_rng(1))
    assert S.shape == (40000, 5)
    sd = np.sqrt(np.diag(cov))
    np.testing.assert_allclose(S.mean(0), mean, atol=float(5 * sd.max() / math.sqrt(40000)) + 1e-9)
    emp = np.cov(S.T)
    np.testing.assert_allclose(emp, cov, atol=0.05 * np.max(np.diag(cov)))
    corr = emp / np.outer(np.sqrt(np.diag(emp)), np.sqrt(np.diag(emp)))
    corr_ref = cov / np.outer(sd, sd)
    np.testing.assert_allclose(corr[:4, :4], corr_ref[:4, :4], atol=0.03)
    # deterministic given the rng seed
    np.testing.assert_array_equal(gp.sample(Xs, 3, 7), gp.sample(Xs, 3, 7))
    assert gp.sample(Xs, 0, 0).shape == (0, 5)


def test_sample_many_close_points():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(10, 2))
    gp = GP().fit(X, np.sin(5 * X[:, 0]))
    Xs = 0.5 + 1e-7 * rng.standard_normal((300, 2))  # numerically singular posterior covariance
    S = gp.sample(Xs, 50, rng)
    assert S.shape == (50, 300) and np.all(np.isfinite(S))
    # near-identical inputs give near-identical values within a sample, while samples differ
    within = np.max(np.std(S, axis=1))
    across = np.std(S[:, 0])
    assert across > 0.01 and within < 1e-2 * across


# --------------------------------------------------------------------------- standardization


def test_standardization_round_trip(data3):
    X, y = data3
    gp = GP().fit(X, y)
    assert gp.y_mean == pytest.approx(np.mean(y))
    assert gp.y_std == pytest.approx(np.std(y, ddof=1))
    np.testing.assert_allclose(gp._ys * gp.y_std + gp.y_mean, y, rtol=1e-12, atol=1e-12)
    a, b = -37.5, 1234.0
    gp2 = GP().fit(X, a * y + b)
    np.testing.assert_allclose(gp2.hyperparameters()["lengthscales"], gp.hyperparameters()["lengthscales"], rtol=1e-6)
    assert gp2.hyperparameters()["noise"] == pytest.approx(gp.hyperparameters()["noise"], rel=1e-6)
    assert gp2.hyperparameters()["noise_var"] == pytest.approx(a**2 * gp.hyperparameters()["noise_var"], rel=1e-6)
    Xs = np.random.default_rng(3).uniform(size=(10, 3))
    m1, v1, dm1, dv1 = gp.posterior(Xs, return_grad=True)
    m2, v2, dm2, dv2 = gp2.posterior(Xs, return_grad=True)
    np.testing.assert_allclose(m2, a * m1 + b, rtol=1e-6)
    np.testing.assert_allclose(v2, a**2 * v1, rtol=1e-5)
    np.testing.assert_allclose(dm2, a * dm1, rtol=1e-5, atol=1e-6 * abs(a))
    np.testing.assert_allclose(dv2, a**2 * dv1, rtol=1e-4, atol=1e-6 * a**2)
    l1, lv1 = gp.loo()
    l2, lv2 = gp2.loo()
    np.testing.assert_allclose(l2, a * l1 + b, rtol=1e-6)
    np.testing.assert_allclose(lv2, a**2 * lv1, rtol=1e-5)
    assert gp2.cv_r2() == pytest.approx(gp.cv_r2(), rel=1e-6)
    # same base normals: centred samples scale with |a| (y_std is positive)
    S1 = gp.sample(Xs, 5, 0) - gp.posterior(Xs)[0]
    S2 = gp2.sample(Xs, 5, 0) - gp2.posterior(Xs)[0]
    np.testing.assert_allclose(S2, abs(a) * S1, rtol=1e-4, atol=1e-6 * abs(a))


@pytest.mark.parametrize("scale", [1e-12, 1e-6, 1e6, 1e12])
def test_tiny_and_huge_output_scales(scale, data3):
    X, y = data3
    ref = GP().fit(X, y)
    gp = GP().fit(X, scale * y)
    np.testing.assert_allclose(gp.hyperparameters()["lengthscales"], ref.hyperparameters()["lengthscales"], rtol=1e-5)
    Xs = np.random.default_rng(0).uniform(size=(5, 3))
    m, v = gp.posterior(Xs)
    mr, vr = ref.posterior(Xs)
    np.testing.assert_allclose(m, scale * mr, rtol=1e-5)
    np.testing.assert_allclose(v, scale**2 * vr, rtol=1e-4)
    assert gp.cv_r2() == pytest.approx(ref.cv_r2(), rel=1e-5)


# --------------------------------------------------------------------------- degenerate data


@pytest.mark.parametrize("prior", ["botorch", "median"])
def test_single_observation_uses_prior_medians(prior):
    x = np.array([[0.3, 0.6]])
    gp = GP(prior=prior).fit(x, np.array([2.5]))
    hp = gp.hyperparameters()
    mu = hyperprior(2, form=prior)[0]
    np.testing.assert_allclose(hp["lengthscales"], np.exp(mu[:2]))
    assert hp["outputscale"] == pytest.approx(1.0)
    assert hp["noise"] == pytest.approx(math.exp(-4 if prior == "median" else -5))
    assert gp.y_std == 1.0 and gp.y_mean == 2.5
    assert gp.fit_info["n_starts"] == 0
    m, v = gp.posterior(np.array([[0.3, 0.6], [0.9, 0.1]]))
    assert m[0] == pytest.approx(2.5)
    assert 0 < v[0] < v[1] <= 1.0 + 1e-12
    assert gp.cv_r2() is None
    lm, lv = gp.loo()
    assert lm.shape == (1,) and lv[0] > 0
    _, _, dm, dv = gp.posterior(np.array([[0.5, 0.5]]), return_grad=True)
    assert np.all(np.isfinite(dm)) and np.all(np.isfinite(dv))
    # fixed noise is respected with n == 1
    assert GP(noise=0.1).fit(x, [1.0]).hyperparameters()["noise"] == 0.1


def test_duplicate_inputs():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(8, 2))
    X = np.vstack([X, X[:4], X[:2]])
    y = np.sin(4 * X[:, 0]) + X[:, 1] + 0.05 * rng.standard_normal(len(X))
    for noise in ("learn", 0.0):
        gp = GP(noise=noise).fit(X, y)
        m, v, dm, dv = gp.posterior(X, return_grad=True)
        assert np.all(np.isfinite(m)) and np.all(v > 0) and np.all(np.isfinite(dv))
        lm, lv = gp.loo()
        assert np.all(np.isfinite(lm)) and np.all(lv > 0)
    assert GP(noise=0.0).fit(X, y)._jitter > 0


def test_constant_targets():
    X = np.random.default_rng(0).uniform(size=(6, 2))
    for c in (0.0, 3.0, -1e8):
        gp = GP().fit(X, np.full(6, c))
        assert gp.y_std == 1.0
        m, v = gp.posterior(np.random.default_rng(1).uniform(size=(4, 2)))
        np.testing.assert_allclose(m, c, atol=1e-6 * max(1.0, abs(c)))
        assert np.all(v > 0) and np.all(np.isfinite(v))
        assert gp.cv_r2() is None
    # floating-point noise around a constant is still treated as constant
    gp = GP().fit(X, 1e6 + np.array([0, 1e-10, -1e-10, 0, 2e-10, 0]))
    assert gp.y_std == 1.0


def test_variances_never_negative():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(30, 2))
    y = np.sin(3 * X[:, 0])
    gp = GP(noise=0.0, kernel="rbf").fit(X, y)
    _, v = gp.posterior(X)
    assert np.all(v > 0)
    _, cov = gp.posterior(X, full_cov=True)
    assert np.all(np.diag(cov) > 0)
    _, v, _, dv = gp.posterior(X, return_grad=True)
    assert np.all(v > 0) and np.all(np.isfinite(dv))


def test_cholesky_jitter_escalation():
    v = np.array([1.0, 2.0, 3.0])
    K = np.outer(v, v)  # rank one, PSD
    L, jitter = _cholesky(K)
    assert jitter > 0
    np.testing.assert_allclose(L @ L.T, K + jitter * np.eye(3), atol=1e-12)
    L, jitter = _cholesky(np.eye(3))
    assert jitter == 0.0
    with pytest.raises(np.linalg.LinAlgError):
        _cholesky(-np.eye(3))


# --------------------------------------------------------------------------- misc


def test_hyperparameters_and_sensitivity(data3):
    X, y = data3
    gp = GP().fit(X, y)
    hp = gp.hyperparameters()
    assert set(hp) == {"lengthscales", "outputscale", "noise", "noise_var"}
    assert len(hp["lengthscales"]) == 3
    assert all(isinstance(v, float) for v in hp["lengthscales"])
    assert hp["noise_var"] == pytest.approx(hp["noise"] * gp.y_std**2)
    sens = gp.sensitivity()
    assert sens.sum() == pytest.approx(1.0) and np.all(sens > 0)
    np.testing.assert_allclose(sens, (1 / np.array(hp["lengthscales"])) / np.sum(1 / np.array(hp["lengthscales"])))
    assert gp.n_train == 25 and gp.X_train.shape == (25, 3) and gp.y_train.shape == (25,)
    assert "n=25" in repr(gp)


def test_relevant_dimension_gets_shorter_lengthscale():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(40, 3))
    y = np.sin(6 * X[:, 0]) + 0.01 * X[:, 2]
    gp = GP().fit(X, y)
    ls = np.array(gp.hyperparameters()["lengthscales"])
    assert ls[0] < ls[1] and ls[0] < ls[2]
    assert np.argmax(gp.sensitivity()) == 0


def test_one_dimensional_vector_inputs():
    x = np.linspace(0, 1, 6)
    y = x**2
    gp = GP().fit(x, y)
    assert gp.D == 1
    m, v = gp.posterior(np.array([0.25, 0.5, 0.75]))
    assert m.shape == (3,)
    np.testing.assert_allclose(m, np.array([0.25, 0.5, 0.75]) ** 2, atol=0.02)


def test_empty_prediction_set(data3):
    X, y = data3
    gp = GP().fit(X, y)
    m, v = gp.posterior(np.zeros((0, 3)))
    assert m.shape == (0,) and v.shape == (0,)
    m, v, dm, dv = gp.posterior(np.zeros((0, 3)), return_grad=True)
    assert dm.shape == (0, 3)


def test_refit_resets_state(data3):
    X, y = data3
    gp = GP().fit(X, y)
    _ = gp.loo()
    gp.fit(X[:10], y[:10])
    assert gp.n_train == 10
    assert gp.loo()[0].shape == (10,)


# --------------------------------------------------------------------------- performance


@pytest.mark.slow
def test_performance_targets():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(200, 20))
    y = np.sin(3 * X[:, :5]).sum(1) + X[:, 5:10].sum(1) ** 2 / 10 + 0.05 * rng.standard_normal(200)
    t0 = time.perf_counter()
    gp = GP(n_restarts=3).fit(X, y)
    t_fit = time.perf_counter() - t0
    Z = rng.uniform(size=(2000, 20))
    t0 = time.perf_counter()
    gp.posterior(Z, return_grad=True)
    t_post = time.perf_counter() - t0
    # targets are ~3 s and ~0.5 s; assert loosely to stay robust on slow CI machines
    assert t_fit < 9.0, t_fit
    assert t_post < 1.5, t_post
