import math
import pickle
import re
import time

import numpy as np
import pytest
from scipy.optimize import minimize

from agentic_bo.metrics import dominated_hypervolume
from agentic_bo.problems import Problem, get_problem, list_problems, randomize_problem
from agentic_bo.problems import applied, synthetic
from agentic_bo.space import RealParam, SearchSpace
from agentic_bo.utils import LenzError

FIXED = ["branin", "six_hump_camel", "goldstein_price", "hartmann3", "hartmann6",
         "constrained_hartmann6", "branin_currin", "reaction_yield"]
FAMILY_DIMS = [f"{fam}{d}" for fam in ["ackley", "levy", "rosenbrock", "styblinski_tang", "rastrigin", "griewank"]
               for d in (2, 5, 10, 20)]
ALL = FIXED + FAMILY_DIMS

# literature optimal values (SFU library / BoTorch), with the tolerance of the rounded constant
LITERATURE = {
    "branin": (0.397887, 1e-6), "six_hump_camel": (-1.0316, 1e-4), "goldstein_price": (3.0, 1e-12),
    "hartmann3": (-3.86278, 1e-5), "hartmann6": (-3.32237, 1e-5), "constrained_hartmann6": (-3.32237, 1e-5),
    "branin_currin": (0.397887, 1e-6),
}


def literature_f_star(name):
    if name in LITERATURE:
        return LITERATURE[name]
    fam, d = re.match(r"([a-z_]+?)(\d+)$", name).groups()
    if fam == "styblinski_tang":
        return -39.16617 * int(d), 1e-5 * int(d)
    return 0.0, 1e-12


def regrets_vectorized(p, X):
    """Regret of raw points X (space order) via the vectorized path."""
    vals = p.evaluate_array(X)[p.objective]
    util = -vals if p.minimize else vals
    return p.u_star - util


def random_points(p, n, rng):
    lb, ub = p.space.encoded_bounds()
    U = rng.uniform(size=(n, p.space.D))
    return np.array([[c[name] for name in p.space.names] for c in p.space.decode_many(lb + U * (ub - lb))])


# --------------------------------------------------------------------------- registry


def test_list_problems_contains_required_names():
    names = list_problems()
    assert names == sorted(names)
    for n in FIXED + ["ackley2", "ackley10", "ackley20", "levy5", "rosenbrock10", "styblinski_tang2",
                      "rastrigin5", "griewank10"]:
        assert n in names


def test_get_problem_errors():
    with pytest.raises(LenzError, match="unknown problem"):
        get_problem("nope")
    with pytest.raises(LenzError, match="unknown problem"):
        get_problem("ackley")                      # family without dimension
    with pytest.raises(LenzError, match="dimension"):
        get_problem("rosenbrock1")
    with pytest.raises(LenzError, match="dimension"):
        get_problem("ackley0")
    with pytest.raises(LenzError, match="unknown get_problem options"):
        get_problem("branin", foo=1)
    with pytest.raises(LenzError, match="randomize"):
        get_problem("branin", shift=True)
    with pytest.raises(LenzError, match="shift-safe"):
        get_problem("branin_currin", randomize=True, shift=True)


def test_name_normalization_and_family_regex():
    assert get_problem("Six-Hump-Camel").name == "six_hump_camel"
    p = get_problem("styblinski_tang7")
    assert p.name == "styblinski_tang7" and p.space.d == 7
    assert get_problem("ackley33").space.d == 33


def test_default_budgets_follow_paper():
    expected = {"branin": 50, "hartmann6": 100, "ackley10": 150, "ackley20": 200,
                "constrained_hartmann6": 100, "reaction_yield": 40}
    for name, budget in expected.items():
        assert get_problem(name).default_budget == budget
    assert get_problem("branin", default_budget=7).default_budget == 7


def test_raw_domains():
    assert [(p.lower, p.upper) for p in get_problem("branin").space.params] == [(-5, 10), (0, 15)]
    assert [(p.lower, p.upper) for p in get_problem("six_hump_camel").space.params] == [(-3, 3), (-2, 2)]
    assert all((p.lower, p.upper) == (-2, 2) for p in get_problem("goldstein_price").space.params)
    assert all((p.lower, p.upper) == (-32.768, 32.768) for p in get_problem("ackley3").space.params)
    assert all((p.lower, p.upper) == (-10, 10) for p in get_problem("levy3").space.params)
    assert all((p.lower, p.upper) == (-5, 10) for p in get_problem("rosenbrock3").space.params)
    assert all((p.lower, p.upper) == (-5, 5) for p in get_problem("styblinski_tang3").space.params)
    assert all((p.lower, p.upper) == (-5.12, 5.12) for p in get_problem("rastrigin3").space.params)
    assert all((p.lower, p.upper) == (-600, 600) for p in get_problem("griewank3").space.params)


# --------------------------------------------------------------------------- optima


@pytest.mark.parametrize("name", [n for n in ALL if n != "reaction_yield"])
def test_optimum_values_match_literature(name):
    p = get_problem(name)
    ref, tol = literature_f_star(name)
    assert p.f_star == pytest.approx(ref, abs=tol)
    m = p.evaluate(p.x_star)
    assert m[p.objective] == pytest.approx(p.f_star, abs=1e-12)
    assert abs(p.regret(m)) <= 1e-12
    assert p.space.contains(p.x_star)


def test_known_minimizers():
    hp = get_problem("hartmann6")
    lit = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
    assert np.allclose([hp.x_star[n] for n in hp.space.names], lit, atol=1e-5)
    h3 = get_problem("hartmann3")
    assert np.allclose([h3.x_star[n] for n in h3.space.names], [0.114614, 0.555649, 0.852547], atol=1e-4)
    cam = get_problem("six_hump_camel")
    assert np.allclose([cam.x_star["x0"], cam.x_star["x1"]], [0.0898, -0.7126], atol=1e-4)
    # the symmetric camel minimizer and the three Branin minimizers attain the same value
    assert cam.evaluate({"x0": -cam.x_star["x0"], "x1": -cam.x_star["x1"]})["y"] == pytest.approx(cam.f_star, abs=1e-12)
    br = get_problem("branin")
    for x in [(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)]:
        assert br.evaluate({"x0": x[0], "x1": x[1]})["y"] == pytest.approx(0.397887, abs=1e-6)
    st = get_problem("styblinski_tang3")
    assert all(v == pytest.approx(-2.903534, abs=1e-6) for v in st.x_star.values())


@pytest.mark.parametrize("name", ["branin", "six_hump_camel", "goldstein_price", "hartmann3", "hartmann6",
                                  "ackley2", "levy3", "rosenbrock3", "styblinski_tang3", "rastrigin2", "griewank2"])
def test_x_star_is_a_polished_local_minimum(name):
    """Local optimization from x_star cannot improve f_star beyond rounding."""
    p = get_problem(name)
    x0 = np.array([p.x_star[n] for n in p.space.names])
    bounds = [(q.lower, q.upper) for q in p.space.params]
    res = minimize(lambda x: float(p.evaluate_array(x[None])["y"][0]), x0, method="L-BFGS-B", bounds=bounds,
                   options={"ftol": 1e-16, "gtol": 1e-14})
    assert res.fun >= p.f_star - 1e-12


@pytest.mark.parametrize("name", ["branin", "six_hump_camel", "goldstein_price", "hartmann3", "hartmann6"])
def test_shift_safety_on_extended_box(name):
    """Shift-safe: no point within a quarter range outside the native box beats f_star."""
    p = get_problem(name)
    assert p.shift_safe
    lo = np.array([q.lower for q in p.space.params])
    hi = np.array([q.upper for q in p.space.params])
    span = hi - lo
    if name == "six_hump_camel":           # much larger than the shift can expose
        lo, hi = np.full(2, -10.0), np.full(2, 10.0)
    else:
        lo, hi = lo - 0.25 * span, hi + 0.25 * span
    rng = np.random.default_rng(0)
    X = lo + rng.uniform(size=(20000, p.space.d)) * (hi - lo)
    f = p.evaluate_array(X)["y"]
    assert f.min() >= p.f_star - 1e-9
    fn = lambda x: float(p.evaluate_array(x[None])["y"][0])  # noqa: E731
    for i in np.argsort(f)[:15]:
        res = minimize(fn, X[i], method="L-BFGS-B", bounds=list(zip(lo, hi)))
        assert res.fun >= p.f_star - 1e-9


def test_branin_currin_values_and_max_hv():
    p = get_problem("branin_currin")
    assert p.is_moo and p.ref_point == [18.0, 6.0] and p.max_hv == pytest.approx(59.36011874867746)
    m = p.evaluate({"x0": 0.5, "x1": 0.5})
    expected_f1 = synthetic.branin(np.array([[2.5, 7.5]]))[0]
    factor = 1 - math.exp(-1.0)
    expected_f2 = factor * (2300 * 0.125 + 1900 * 0.25 + 2092 * 0.5 + 60) / (100 * 0.125 + 500 * 0.25 + 2 + 20)
    assert m["f1"] == pytest.approx(expected_f1) and m["f2"] == pytest.approx(expected_f2)
    assert np.isfinite(p.evaluate({"x0": 0.3, "x1": 0.0})["f2"])          # x1 = 0 limit is finite
    # a dense grid over the Pareto-set region (x0 <= 0.15, x1 >= 0.8) approaches max_hv from below
    X = np.array(np.meshgrid(np.linspace(0, 0.15, 801), np.linspace(0.8, 1, 801))).reshape(2, -1).T
    Y = p.evaluate_array(X)
    hv = dominated_hypervolume(np.column_stack([Y["f1"], Y["f2"]]), p.ref_point, [True, True])
    assert 0.998 * p.max_hv < hv <= p.max_hv


def test_constrained_hartmann6_optimum_is_feasible():
    p = get_problem("constrained_hartmann6")
    m = p.evaluate(p.x_star)
    assert p.is_feasible(m) and m["c"] == pytest.approx(sum(v * v for v in p.x_star.values()))
    assert m["c"] < 1.0
    assert not p.is_feasible(p.evaluate({n: 1.0 for n in p.space.names}))
    assert p.constraints == [{"metric": "c", "lower": None, "upper": 1.0}]


# --------------------------------------------------------------------------- regret non-negativity


@pytest.mark.parametrize("name", ALL)
def test_random_point_regret_nonnegative(name):
    p = get_problem(name)
    rng = np.random.default_rng(1)
    X = random_points(p, 3000, rng)
    assert regrets_vectorized(p, X).min() >= -1e-9
    # near the optimum, where an inaccurate f_star would show up
    xs = np.array([p.x_star[n] for n in p.space.names])
    span = np.array([q.upper - q.lower for q in p.space.params])
    lo = np.array([q.lower for q in p.space.params])
    for scale in (1e-2, 1e-4, 1e-7):
        Xn = np.clip(xs + rng.normal(size=(300, len(xs))) * scale * span, lo, lo + span)
        assert regrets_vectorized(p, Xn).min() >= -1e-9
    # the dict path agrees with the vectorized path
    for x in X[:20]:
        cfg = dict(zip(p.space.names, x))
        assert p.regret(p.evaluate(cfg)) == pytest.approx(regrets_vectorized(p, x[None])[0], abs=1e-9)


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("seed", [0, 7])
def test_randomized_regret_nonnegative_and_optimum_preserved(name, seed):
    p0 = get_problem(name)
    p = get_problem(name, seed=seed, randomize=True)
    assert p.name == p0.name and p.f_star == p0.f_star
    assert abs(p.regret(p.evaluate(p.x_star))) <= 1e-9
    rng = np.random.default_rng(seed + 100)
    U = rng.uniform(size=(2000, p.space.d))
    assert regrets_vectorized(p, U).min() >= -1e-9
    xs = np.array([p.x_star[n] for n in p.space.names])
    Un = np.clip(xs + rng.normal(size=(300, len(xs))) * 1e-3, 0, 1)
    assert regrets_vectorized(p, Un).min() >= -1e-9


# --------------------------------------------------------------------------- randomization


def test_randomization_is_deterministic_in_seed():
    a = get_problem("hartmann6", seed=3, randomize=True)
    b = get_problem("hartmann6", seed=3, randomize=True)
    c = get_problem("hartmann6", seed=4, randomize=True)
    assert a.space.names == b.space.names and a.x_star == b.x_star
    assert a.space.names != c.space.names and a.x_star != c.x_star
    cfg = dict(zip(a.space.names, np.linspace(0.1, 0.9, 6)))
    assert a.evaluate(cfg) == b.evaluate(cfg)
    # default seed when randomizing is 0
    default = get_problem("hartmann6", randomize=True)
    assert default.space.names == get_problem("hartmann6", seed=0, randomize=True).space.names
    # different problems with the same seed get different tokens
    assert get_problem("hartmann3", seed=3, randomize=True).space.names != a.space.names[:3]


def test_seed_is_irrelevant_without_randomization():
    a, b = get_problem("branin", seed=1), get_problem("branin", seed=2)
    assert a.space.names == b.space.names == ["x0", "x1"] and a.x_star == b.x_star


def test_randomized_space_is_canonical_with_token_names():
    for name in ["branin", "ackley20", "reaction_yield", "constrained_hartmann6"]:
        p0 = get_problem(name)
        p = get_problem(name, seed=5, randomize=True)
        assert all(re.fullmatch(r"x_[0-9a-f]{3}", n) for n in p.space.names)
        assert len(set(p.space.names)) == p.space.d == p0.space.d
        assert not set(p.space.names) & set(p0.space.names)
        assert all(isinstance(q, RealParam) and (q.lower, q.upper, q.log_scale) == (0.0, 1.0, False)
                   for q in p.space.params)
        assert sorted(p.meta["original_names"]) == sorted(p0.space.names)


def test_shift_keeps_optimum_inside_margin():
    for name in ["branin", "hartmann6", "ackley10", "styblinski_tang5", "rosenbrock4", "levy3", "griewank2"]:
        for seed in range(5):
            p = get_problem(name, seed=seed, randomize=True)
            assert p.meta["shifted"]
            u = np.array(list(p.x_star.values()))
            assert np.all(u >= 0.05 - 1e-12) and np.all(u <= 0.95 + 1e-12)
            assert np.all(np.abs(p.meta["shift"]) <= 0.25 + 1e-12)
    # shifts actually move the optimum away from the canonical image of the native optimum
    p = get_problem("ackley10", seed=0, randomize=True)
    assert np.max(np.abs(np.array(list(p.x_star.values())) - 0.5)) > 0.05


def test_non_shift_safe_problems_are_not_shifted():
    for name in ["branin_currin", "reaction_yield"]:
        p = get_problem(name, seed=2, randomize=True)
        assert not p.meta["shifted"] and np.allclose(p.meta["shift"], 0.0)


def test_randomized_evaluation_matches_original_unshifted():
    """Canonical coordinates are the original encoded coordinates (log axes included), permuted."""
    for name in ["reaction_yield", "hartmann6", "branin"]:
        p0 = get_problem(name)
        p = get_problem(name, seed=9, randomize=True, shift=False)
        rng = np.random.default_rng(0)
        for _ in range(10):
            u = rng.uniform(size=p.space.d)
            cfg = dict(zip(p.space.names, u))
            enc = np.zeros(p0.space.D)
            for k, orig in enumerate(p.meta["original_names"]):
                enc[p0.space.slices[orig].start] = u[k]
            raw = p0.space.decode(enc)
            assert p.evaluate(cfg) == pytest.approx(p0.evaluate(raw), rel=1e-10, abs=1e-12)


def test_randomized_evaluation_matches_original_shifted():
    p0 = get_problem("hartmann6")
    p = get_problem("hartmann6", seed=11, randomize=True)
    shift = np.array(p.meta["shift"])
    rng = np.random.default_rng(1)
    for _ in range(10):
        u = rng.uniform(size=6)
        e = u - shift                              # encoded original coordinates (may leave [0, 1])
        x = np.empty(6)
        for k, orig in enumerate(p.meta["original_names"]):
            x[p0.space.names.index(orig)] = e[k]   # hartmann's native domain is the unit cube
        assert p.evaluate(dict(zip(p.space.names, u)))["y"] == pytest.approx(synthetic.hartmann6(x[None])[0])


def test_constrained_randomized_constraint_in_original_coordinates():
    for seed in range(4):
        p = get_problem("constrained_hartmann6", seed=seed, randomize=True)
        assert p.meta["shifted"]
        m = p.evaluate(p.x_star)
        assert p.is_feasible(m) and m["y"] == pytest.approx(p.f_star, abs=1e-12)
        shift = np.array(p.meta["shift"])
        u = np.random.default_rng(seed).uniform(size=6)
        e = u - shift
        assert p.evaluate(dict(zip(p.space.names, u)))["c"] == pytest.approx(float(np.sum(e ** 2)))
        # feasible random points never beat the optimum
        U = np.random.default_rng(seed).uniform(size=(3000, 6))
        Y = p.evaluate_array(U)
        feas = Y["c"] <= 1.0
        assert feas.any() and Y["y"][feas].min() >= p.f_star - 1e-9


def test_canonical_map_roundtrip_with_log_axes():
    from agentic_bo.problems import CanonicalMap

    params = [RealParam("a", -2.0, 4.0), RealParam("b", 0.1, 10.0, log_scale=True),
              RealParam("c", 1.0, 180.0, log_scale=True)]
    cmap = CanonicalMap(params, perm=[2, 0, 1], shift=[0.1, -0.2, 0.0])
    U = np.random.default_rng(0).uniform(size=(50, 3))
    X = cmap.to_raw(U)
    assert np.allclose(cmap.to_canonical(X), U)
    # canonical column 0 is original param 2 ("c", log axis) shifted by 0.1
    assert np.allclose(np.log(X[:, 2]), np.log(1.0) + (U[:, 0] - 0.1) * np.log(180.0))
    assert cmap.encoded_to_canonical(0, 0.5) == pytest.approx(0.3)      # original "a" -> column 1


def test_shift_requires_x_star_inside_domain():
    space = SearchSpace([RealParam("a", 0, 1)])
    p = Problem("out", space, lambda c: {"y": c["a"]}, "y", True, 0.0, x_star={"a": 2.0}, shift_safe=True)
    with pytest.raises(LenzError, match="inside the domain"):
        randomize_problem(p, seed=0)


def test_randomize_explicit_options():
    p0 = get_problem("goldstein_price")
    p = randomize_problem(p0, seed=1, shift=False, permute=False)
    assert p.meta["original_names"] == ["x0", "x1"]
    assert list(p.x_star.values()) == pytest.approx([0.5, 0.25])      # (0, -1) on [-2, 2]^2
    with pytest.raises(LenzError, match="real"):
        randomize_problem(Problem("ip", SearchSpace.from_spec([{"name": "k", "type": "int", "lower": 0, "upper": 3}]),
                                  lambda c: {"y": c["k"]}, "y", True, 0.0), seed=0)
    unknown = Problem("u", SearchSpace([RealParam("a", 0, 1)]), lambda c: {"y": c["a"]}, "y", True, None,
                      shift_safe=True)
    with pytest.raises(LenzError, match="x_star"):
        randomize_problem(unknown, seed=0, shift=True)


# --------------------------------------------------------------------------- brief


def test_brief_content_rules():
    p = get_problem("hartmann6")
    b = p.brief()
    for section in ("## Task", "## Inputs", "## Outputs"):
        assert section in b
    assert p.space.describe() in b
    assert "`y`: objective; minimize" in b
    assert "budget" not in b.lower()
    assert str(round(p.f_star, 3)) not in b                     # no optimum leak
    noisy = get_problem("hartmann6", noise_std=0.25)
    assert "## Noise" in noisy.brief() and "0.25" in noisy.brief() and "noiseless" not in noisy.brief()


@pytest.mark.parametrize("name", ["hartmann6", "reaction_yield", "branin_currin"])
def test_noiseless_brief_says_so(name):
    """Regression (reconciliation, llm owner's request): the prompts tell Sara to take noise from the
    brief rather than from the fitted noise_std, so a noiseless problem's brief must say it is noiseless."""
    b = get_problem(name, seed=1, randomize=name != "reaction_yield").brief()
    assert "## Noise\nObservations are noiseless" in b
    from agentic_bo.sara.policies.heuristic import declared_noise_std
    assert declared_noise_std(b) == 0.0


def test_brief_constraints_moo_and_prior_excluded():
    c = get_problem("constrained_hartmann6").brief()
    assert "`c`: constraint; feasible iff `c <= 1`" in c and "Only feasible" in c
    m = get_problem("branin_currin").brief()
    assert "`f1`: objective; minimize" in m and "`f2`: objective; minimize" in m and "[18, 6]" in m
    r = get_problem("reaction_yield")
    rb = r.brief()
    assert r.prior and r.prior not in rb and "protodeboronation" not in rb
    assert "`yield`: objective; maximize" in rb and "`cost`: reported for information" in rb
    assert "budget" not in rb.lower()


@pytest.mark.parametrize("name", ALL)
def test_randomized_brief_does_not_name_function(name):
    p = get_problem(name, seed=3, randomize=True)
    b = p.brief().lower()
    words = {"branin", "currin", "camel", "goldstein", "hartmann", "ackley", "levy", "rosenbrock",
             "styblinski", "rastrigin", "griewank", "suzuki", "reaction", "temperature", "catalyst", name}
    assert not any(w in b for w in words)
    assert "anonymous black-box" in b and f"{p.space.d} continuous input" in b
    assert ("lower is better" in b) == p.minimize or p.is_moo
    assert p.prior is None


# --------------------------------------------------------------------------- problem semantics


def test_utility_feasibility_regret_semantics():
    space = SearchSpace([RealParam("a", 0, 1)])
    p = Problem("toy", space, lambda c: {"y": c["a"], "c": 2 * c["a"]}, "y", False, 1.0,
                constraints=[{"metric": "c", "lower": 0.2, "upper": 1.5}])
    assert p.u_star == 1.0 and p.utility({"y": 0.4}) == 0.4 and p.regret({"y": 0.4}) == pytest.approx(0.6)
    assert p.is_feasible({"c": 0.2}) and p.is_feasible({"c": 1.5})
    assert not p.is_feasible({"c": 0.1}) and not p.is_feasible({"c": 1.6})
    assert not p.is_feasible({"c": float("nan")}) and not p.is_feasible({})
    assert p.utility({"y": float("nan")}) == -math.inf and p.utility({}) == -math.inf
    q = Problem("toy-min", space, lambda c: {"y": c["a"]}, "y", True, -2.0)
    assert q.u_star == 2.0 and q.utility({"y": 3.0}) == -3.0 and q.regret({"y": 3.0}) == pytest.approx(5.0)
    unknown = Problem("u", space, lambda c: {"y": c["a"]}, "y", True, None)
    assert unknown.u_star is None and math.isnan(unknown.regret({"y": 0.0}))
    assert not unknown.is_moo and unknown.objective_list == [{"metric": "y", "minimize": True}]


def test_problem_validation():
    space = SearchSpace([RealParam("a", 0, 1)])
    fn = lambda c: {"y": c["a"]}  # noqa: E731
    with pytest.raises(LenzError, match="objectives"):
        Problem("bad", space, fn, "y", True, 0.0, objectives=[{"metric": "z", "minimize": True},
                                                              {"metric": "y", "minimize": True}])
    with pytest.raises(LenzError, match="bound"):
        Problem("bad", space, fn, "y", True, 0.0, constraints=[{"metric": "c"}])
    with pytest.raises(LenzError, match="lower"):
        Problem("bad", space, fn, "y", True, 0.0, constraints=[{"metric": "c", "lower": 2, "upper": 1}])
    with pytest.raises(LenzError, match="noise_std"):
        Problem("bad", space, fn, "y", True, 0.0, noise_std=-1)
    with pytest.raises(LenzError, match="ref_point"):
        Problem("bad", space, fn, "y", True, 0.0, ref_point=[1.0, 2.0])
    p = Problem("ok", space, lambda c: {"z": 1.0}, "y", True, 0.0)
    with pytest.raises(LenzError, match="did not return"):
        p.evaluate({"a": 0.5})


def test_evaluate_validates_configs():
    p = get_problem("branin")
    with pytest.raises(LenzError, match="outside"):
        p.evaluate({"x0": 11.0, "x1": 1.0})
    with pytest.raises(LenzError, match="missing"):
        p.evaluate({"x0": 1.0})
    assert set(p.evaluate({"x0": 1, "x1": 2})) == {"y"}


def test_observe_noise():
    p = get_problem("branin")
    cfg = {"x0": 1.0, "x1": 2.0}
    assert p.observe(cfg, np.random.default_rng(0)) == p.evaluate(cfg)
    q = get_problem("branin", noise_std=0.5)
    assert q.observe(cfg, 3) == q.observe(cfg, 3)
    rng = np.random.default_rng(0)
    obs = np.array([q.observe(cfg, rng)["y"] for _ in range(4000)])
    assert obs.mean() == pytest.approx(q.evaluate(cfg)["y"], abs=0.05)
    assert obs.std() == pytest.approx(0.5, rel=0.05)
    cm = get_problem("constrained_hartmann6", noise_std=0.1).observe({f"x{i}": 0.5 for i in range(6)}, 1)
    assert set(cm) == {"y", "c"} and cm["c"] != pytest.approx(1.5, abs=1e-12)


@pytest.mark.parametrize("name", ["branin", "ackley5", "constrained_hartmann6", "branin_currin", "reaction_yield"])
def test_evaluate_array_matches_evaluate(name):
    for randomize in (False, True):
        p = get_problem(name, seed=1, randomize=randomize)
        X = random_points(p, 25, np.random.default_rng(2))
        batch = p.evaluate_array(X)
        for i, x in enumerate(X):
            m = p.evaluate(dict(zip(p.space.names, x)))
            for k, v in m.items():
                assert batch[k][i] == pytest.approx(v, rel=1e-12, abs=1e-12)


def test_evaluate_array_fallback_without_batch_fn():
    p = Problem("loop", SearchSpace([RealParam("a", 0, 1), RealParam("b", 0, 1)]),
                lambda c: {"y": c["a"] + 2 * c["b"]}, "y", True, 0.0)
    out = p.evaluate_array(np.array([[0.1, 0.2], [0.3, 0.4]]))
    assert out["y"] == pytest.approx([0.5, 1.1])
    r = randomize_problem(Problem("loop", p.space, p.evaluate_fn, "y", True, 0.0, x_star={"a": 0, "b": 0}), seed=0)
    assert r.evaluate_array(np.array([[0.5, 0.5]]))["y"] == pytest.approx([1.5])


@pytest.mark.parametrize("name", ["hartmann6", "ackley3", "branin_currin", "reaction_yield"])
def test_problems_are_picklable(name):
    for randomize in (False, True):
        p = get_problem(name, seed=2, randomize=randomize, noise_std=0.1)
        q = pickle.loads(pickle.dumps(p))
        cfg = p.space.center()
        assert q.evaluate(cfg) == p.evaluate(cfg) and q.space == p.space and q.f_star == p.f_star


def test_info_is_json_safe():
    import json
    info = get_problem("constrained_hartmann6", seed=1, randomize=True).info()
    json.dumps(info, allow_nan=False)
    assert info["name"] == "constrained_hartmann6" and info["meta"]["randomized"] is True


# --------------------------------------------------------------------------- reaction_yield


def test_reaction_yield_structure():
    p = get_problem("reaction_yield")
    assert p.space.names == ["temperature", "pressure", "catalyst_loading", "residence_time", "pH", "water_fraction"]
    bounds = {q.name: (q.lower, q.upper, q.log_scale) for q in p.space.params}
    assert bounds == {"temperature": (0, 150, False), "pressure": (1, 10, False),
                      "catalyst_loading": (0.1, 10, True), "residence_time": (1, 180, True),
                      "pH": (6, 13, False), "water_fraction": (0, 1, False)}
    assert p.objective == "yield" and not p.minimize and p.default_budget == 40
    assert set(p.evaluate(p.x_star)) == {"yield", "cost"}


def test_reaction_yield_ranges_and_optimum():
    p = get_problem("reaction_yield")
    assert 93.5 <= p.f_star <= 95.0
    X = random_points(p, 20000, np.random.default_rng(3))
    out = p.evaluate_array(X)
    assert out["yield"].min() >= 0.0 and out["yield"].max() <= 95.0
    assert out["yield"].max() <= p.f_star + 1e-9
    assert out["cost"].min() > 0.0
    assert np.median(out["yield"]) < 50                       # most of the space is poor
    # corners stay in range too
    corners = np.array(np.meshgrid(*[[q.lower, q.upper] for q in p.space.params])).reshape(6, -1).T
    yc = p.evaluate_array(corners)["yield"]
    assert np.all((yc >= 0) & (yc <= 95))
    assert p.regret(p.evaluate(p.x_star)) == 0.0


def test_reaction_yield_prior_hint_contains_x_star():
    p = get_problem("reaction_yield")
    region = p.prior_hint["region"]
    assert set(region) <= set(p.space.names)
    assert p.space.contains(p.x_star, bounds=region)
    assert p.space.validate_bounds(region)
    # the randomized prior hint is mapped consistently
    r = get_problem("reaction_yield", seed=4, randomize=True)
    assert r.space.contains(r.x_star, bounds=r.prior_hint["region"])


def test_reaction_yield_prior_text():
    p = get_problem("reaction_yield")
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", p.prior.strip()) if s]
    assert 3 <= len(sentences) <= 6
    low = p.prior.lower()
    assert "ph" in low and "water" in low and "mol%" in low and "temperature" in low
    # no exact optimum values leak into the agent-facing texts
    for v in p.x_star.values():
        for txt in (p.prior, p.description):
            assert f"{v:.1f}" not in txt and f"{v:.2f}" not in txt
    assert f"{p.f_star:.0f}" not in p.prior and "94" not in p.description


def test_reaction_yield_chemistry_trends():
    p = get_problem("reaction_yield")
    x = dict(p.x_star)

    def y(**kw):
        return p.evaluate({**x, **kw})["yield"]

    assert y(temperature=20) < y(temperature=50) < y(temperature=70) < p.f_star      # Arrhenius
    assert y(temperature=130, residence_time=180) < 20                                # decomposition
    assert y(pH=6) < 25 < y(pH=9) and y(pH=13) < y(pH=11.5)                           # base activity
    assert max(y(water_fraction=0.0), y(water_fraction=1.0), y(water_fraction=0.6)) < y(water_fraction=0.3) - 10
    assert y(catalyst_loading=0.1) < y(catalyst_loading=0.5) < y(catalyst_loading=1.3)
    assert y(catalyst_loading=10) < y(catalyst_loading=3) - 5                       # inhibition
    assert abs(y(pressure=6) - y(pressure=3)) < 1.0                                   # weak pressure effect
    assert y(temperature=120, pressure=1) < y(temperature=120, pressure=8) - 10       # boiling at 1 bar
    assert y(residence_time=1) < 40                                                   # too short


def test_reaction_yield_is_mildly_multimodal():
    p = get_problem("reaction_yield")
    names = p.space.names
    lb, ub = p.space.encoded_bounds()

    def neg(z):
        return -float(p.evaluate_array(np.array([[p.space.decode(z)[n] for n in names]]))["yield"][0])

    start = dict(p.x_star, water_fraction=0.85)
    res = minimize(neg, p.space.encode(start), method="L-BFGS-B", bounds=list(zip(lb, ub)))
    local = p.space.decode(res.x)
    assert local["water_fraction"] > 0.7 and 75 < -res.fun < p.f_star - 3


def test_reaction_yield_optimum_is_fast_and_cached():
    applied.reaction_yield_optimum.cache_clear()
    t0 = time.perf_counter()
    f1, x1 = applied.reaction_yield_optimum()
    assert time.perf_counter() - t0 < 2.0
    t0 = time.perf_counter()
    f2, x2 = applied.reaction_yield_optimum()
    assert time.perf_counter() - t0 < 0.01 and (f1, x1) == (f2, x2)


@pytest.mark.slow
def test_reaction_yield_f_star_is_global():
    """Independent, much denser multi-start search does not beat the cached optimum."""
    p = get_problem("reaction_yield")
    lb, ub = p.space.encoded_bounds()
    rng = np.random.default_rng(123)
    Z = rng.uniform(size=(100000, 6))
    X = np.array([[c[n] for n in p.space.names] for c in p.space.decode_many(Z[:20000])])
    y = p.evaluate_array(X)["yield"]
    best = -np.inf

    def neg(z):
        return -float(p.evaluate_array(np.array([[p.space.decode(z)[n] for n in p.space.names]]))["yield"][0])

    for i in np.argsort(-y)[:40]:
        res = minimize(neg, Z[i], method="L-BFGS-B", bounds=list(zip(lb, ub)))
        best = max(best, -res.fun)
    assert best <= p.f_star + 1e-9
    assert best >= p.f_star - 1e-6


# --------------------------------------------------------------------------- review regressions


def test_dynamic_reconfiguration_example_reports_the_true_constrained_optimum():
    # the example printed a raw Sobol maximum (~88.6 %) as "the best yield with cost <= 3";
    # the constrained optimum is ~92.0 % (active cost constraint), so the reference understated
    # what the agent could reach by more than 3 points
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "dynamic_reconfiguration.py"
    spec = importlib.util.spec_from_file_location("dynamic_reconfiguration_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value, cfg = module.constrained_optimum(3.0, n=2 ** 12)
    p = get_problem("reaction_yield")
    m = p.evaluate(cfg)
    assert m["cost"] <= 3.0 + 1e-6 and m["yield"] == pytest.approx(value, abs=0.05)
    assert value > 91.9
