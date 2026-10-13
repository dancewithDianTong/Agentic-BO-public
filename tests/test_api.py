"""Tests for agentic_bo.api: optimize(), make_problem(), as_space() and the black-box adapter."""

from __future__ import annotations

import math

import numpy as np
import pytest

import agentic_bo
from agentic_bo.api import BlackBox, as_space, make_problem, optimize
from agentic_bo.sara import CampaignResult
from agentic_bo.sara.policies import FixedBOPolicy, Policy
from agentic_bo.space import CategoricalParam, IntParam, RealParam, SearchSpace
from agentic_bo.utils import LenzError

SPACE = {"x": (-2.0, 2.0), "y": (-3.0, 3.0)}


def quad(cfg: dict) -> float:
    return (cfg["x"] - 0.3) ** 2 + (cfg["y"] + 1.0) ** 2


class Counter:
    """Callable wrapper that counts calls (and can record configs)."""

    def __init__(self, fn):
        self.fn = fn
        self.configs: list[dict] = []

    def __call__(self, cfg):
        self.configs.append(dict(cfg))
        return self.fn(cfg)


# --------------------------------------------------------------------------- search spaces


def test_as_space_accepts_pairs_specs_params_and_spaces():
    sp = as_space({"x": (0, 1), "n": {"type": "int", "lower": 1, "upper": 8},
                   "opt": CategoricalParam("opt", ("adam", "sgd")), "lr": [1e-4, 1e-1]})
    assert sp.names == ["x", "n", "opt", "lr"]
    assert isinstance(sp["x"], RealParam) and sp["x"].lower == 0.0 and sp["x"].upper == 1.0
    assert isinstance(sp["n"], IntParam) and isinstance(sp["opt"], CategoricalParam)
    assert isinstance(sp["lr"], RealParam)
    same = SearchSpace([RealParam("a", 0, 1)])
    assert as_space(same) is same
    assert as_space([{"name": "a", "type": "real", "lower": 0, "upper": 1}]) == same
    assert as_space({"params": [{"name": "a", "type": "real", "lower": 0, "upper": 1}]}) == same


@pytest.mark.parametrize("bad", [
    {"x": (0, "1")}, {"x": (0, 1, 2)}, {"x": (1, 0)}, {"x": (0, math.inf)}, {"x": (True, 2)},
    {"x": RealParam("y", 0, 1)}, {"x": "real"}, 3,
])
def test_as_space_rejects_bad_specs(bad):
    with pytest.raises(LenzError):
        as_space(bad)


# --------------------------------------------------------------------------- the black box


def test_blackbox_converts_return_values():
    bb = BlackBox(lambda c: np.float32(1.5))
    assert bb({"x": 0.1}) == {"y": 1.5}
    assert BlackBox(lambda c: np.array([2.0]))({"x": 0}) == {"y": 2.0}
    assert BlackBox(lambda c: 3)({"x": 0}) == {"y": 3.0}
    out = BlackBox(lambda c: None, required=["c"])({"x": 0})
    assert set(out) == {"y", "c"} and all(math.isnan(v) for v in out.values())
    out = BlackBox(lambda c: {"y": 1, "c": None, "z": np.float64(2)}, required=["c"])({"x": 0})
    assert out["y"] == 1.0 and math.isnan(out["c"]) and out["z"] == 2.0
    loss = BlackBox(lambda c: {"loss": 0.5}, objective="loss")
    assert loss({"x": 0}) == {"loss": 0.5}


@pytest.mark.parametrize("ret, required", [
    (True, ()), ("1.0", ()), ([1.0, 2.0], ()), ({"z": 1.0}, ()), ({"y": "a"}, ()), ({"y": 1.0}, ("c",)),
    (1.0, ("c",)), (np.array([1.0, 2.0]), ()),
])
def test_blackbox_rejects_malformed_results(ret, required):
    bb = BlackBox(lambda c: ret, required=required)
    with pytest.raises(LenzError):
        bb({"x": 0})


def test_blackbox_take_last_is_single_use_and_config_specific():
    bb = BlackBox(lambda c: c["x"] * 2)
    bb({"x": 1.0})
    assert bb.take_last({"x": 2.0}) is None          # different config
    bb({"x": 1.0})
    assert bb.take_last({"x": 1.0}) == {"y": 2.0}
    assert bb.take_last({"x": 1.0}) is None          # consumed
    assert bb.n_calls == 2
    with pytest.raises(LenzError):
        BlackBox(42)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- make_problem


def test_make_problem_fields_and_names():
    p = make_problem(quad, SPACE, f_star=0.0, description="a bowl", budget=17,
                     constraints=[{"metric": "c", "upper": 1.0}], outputs={"cost": "money"})
    assert p.name == "quad" and p.objective == "y" and p.minimize and p.f_star == 0.0
    assert p.default_budget == 17 and p.constraints == [{"metric": "c", "lower": None, "upper": 1.0}]
    assert p.metric_names == ["y", "c", "cost"]
    assert "a bowl" in p.brief() and "`cost`" in p.brief()
    assert "Noise" not in p.brief()          # a user black box may be noisy by itself: no noiseless claim
    assert make_problem(lambda c: 1.0, SPACE).name == "objective"
    assert make_problem(quad, SPACE, name="custom", maximize=True).name == "custom"
    assert make_problem(quad, SPACE, maximize=True).u_star is None


@pytest.mark.parametrize("kwargs", [
    {"f_star": math.nan}, {"f_star": "0"}, {"constraints": [{"metric": "y", "upper": 1}]},
    {"constraints": [{"upper": 1}]}, {"constraints": {"metric": "c"}}, {"outputs": 3}, {"outputs": [1, 2]},
    {"objective": ""}, {"description": None}, {"prior": 3}, {"prior": {"txt": "hi"}}, {"prior": {"text": ""}},
    {"prior": {"config": {"x": 5.0, "y": 0.0}}}, {"prior": {"region": {"x": [1, 0]}}}, {"noise_std": -1.0},
])
def test_make_problem_validates(kwargs):
    with pytest.raises(LenzError):
        make_problem(quad, SPACE, **kwargs)


def test_make_problem_structured_prior_gets_text():
    p = make_problem(quad, SPACE, prior={"region": {"x": [0, 1]}, "config": {"x": 0.5, "y": -1}})
    assert p.prior_hint == {"region": {"x": [0.0, 1.0]}, "config": {"x": 0.5, "y": -1.0}}
    assert "x in [0.0, 1.0]" in p.prior and "x=0.5" in p.prior
    p = make_problem(quad, SPACE, prior={"text": "Look near x = 0.3.", "region": {"x": [0, 1]}})
    assert p.prior == "Look near x = 0.3." and p.prior_hint == {"region": {"x": [0.0, 1.0]}}
    assert make_problem(quad, SPACE, prior="  ").prior is None


def test_noisy_observation_reuses_the_measurement():
    f = Counter(quad)
    p = make_problem(f, SPACE, noise_std=0.5)
    cfg = {"x": 0.1, "y": 0.2}
    true = p.evaluate(cfg)
    obs = p.observe(cfg, np.random.default_rng(0))
    assert len(f.configs) == 1 and obs["y"] != true["y"]
    p.observe(cfg, 0)                                   # nothing cached any more: measures again
    assert len(f.configs) == 2


# --------------------------------------------------------------------------- optimize


def test_optimize_minimizes_a_quadratic_with_the_default_policy():
    f = Counter(quad)
    res = optimize(f, SPACE, budget=14, f_star=0.0, seed=1)
    assert isinstance(res, CampaignResult) and res.policy == "heuristic"
    assert len(f.configs) == 14 and res.n_evals == 14 and res.forced_commits == 0
    values = [quad(c) for c in f.configs]
    assert res.best_value == pytest.approx(min(values))
    assert quad(res.best_config) == pytest.approx(res.best_value)
    assert res.simple_regret[-1] == pytest.approx(res.best_value)
    assert len(res.simple_regret) == 14 and np.all(np.diff(res.simple_regret) <= 1e-12)
    assert res.best_value < 0.05
    assert res.problem_info["meta"]["family"] == "user_function"


def test_optimize_maximize_and_regret_signs():
    res = optimize(lambda c: -quad(c) + 2.0, SPACE, budget=10, maximize=True, f_star=2.0, policy="fixed_bo")
    vals = [r["true_metrics"]["y"] for r in res.records]
    assert res.best_value == pytest.approx(max(vals))
    assert res.simple_regret[-1] == pytest.approx(2.0 - max(vals)) and res.simple_regret[-1] >= 0
    assert res.cumulative_regret[-1] == pytest.approx(sum(2.0 - v for v in vals))


def test_optimize_without_f_star_has_nan_regret():
    res = optimize(quad, SPACE, budget=5, policy="sobol")
    assert all(math.isnan(v) for v in res.simple_regret) and res.best_value is not None


def test_optimize_constraints_and_objective_name():
    def f(c):
        return {"loss": quad(c), "c": c["x"] + c["y"], "extra": 1.0}

    res = optimize(f, SPACE, budget=12, objective="loss", constraints=[{"metric": "c", "lower": 0.0}],
                   policy="fixed_bo", seed=3)
    feasible = [r for r in res.records if r["feasible"]]
    assert all(r["true_metrics"]["c"] >= 0 for r in feasible)
    assert all(r["feasible"] == (r["true_metrics"]["c"] >= 0) for r in res.records)
    assert res.best_config is not None and f(res.best_config)["c"] >= 0
    assert res.best_value == pytest.approx(min(r["true_metrics"]["loss"] for r in feasible))


def test_optimize_reraises_errors_from_f_immediately():
    calls = []

    def bad(c):
        calls.append(c)
        raise KeyError("broken user function")

    with pytest.raises(KeyError, match="broken user function"):
        optimize(bad, SPACE, budget=10)
    assert len(calls) == 1

    calls.clear()
    with pytest.raises(LenzError, match="without"):
        optimize(lambda c: calls.append(c) or {"loss": 1.0}, SPACE, budget=10, policy="fixed_bo")
    assert len(calls) == 1


def test_optimize_catch_errors_counts_failed_evaluations():
    def flaky(c):
        if c["x"] > 0:
            raise RuntimeError("experiment failed")
        return quad(c)

    res = optimize(flaky, SPACE, budget=8, policy="sobol", catch_errors=True, f_star=0.0)
    failed = [r for r in res.records if r["error"]]
    assert res.n_evals == 8 and failed and all("experiment failed" in r["error"] for r in failed)
    assert all(r["utility"] is None and not r["feasible"] for r in failed)   # -inf, JSON-safe


def test_optimize_none_marks_failed_measurements():
    res = optimize(lambda c: None if c["x"] > 0 else quad(c), SPACE, budget=8, policy="random", seed=2)
    assert res.n_evals == 8
    for r in res.records:
        assert (r["config"]["x"] > 0) == (r["true_metrics"]["y"] is None)   # NaN, JSON-safe
        assert (r["config"]["x"] > 0) == (r["observed"]["y"] is None)


def test_optimize_noise_calls_f_once_per_evaluation_and_keeps_regret_noise_free():
    f = Counter(quad)
    res = optimize(f, SPACE, budget=9, noise_std=0.3, f_star=0.0, policy="fixed_bo", seed=4)
    assert len(f.configs) == 9
    assert any(r["observed"]["y"] != r["true_metrics"]["y"] for r in res.records)
    assert res.simple_regret[-1] == pytest.approx(min(r["true_metrics"]["y"] for r in res.records))
    assert "Noise" in res.context[0]["text"]


def test_optimize_is_deterministic_given_the_seed():
    a = optimize(quad, SPACE, budget=10, seed=5)
    b = optimize(quad, SPACE, budget=10, seed=5)
    c = optimize(quad, SPACE, budget=10, seed=6)
    assert [r["config"] for r in a.records] == [r["config"] for r in b.records]
    assert [r["config"] for r in a.records] != [r["config"] for r in c.records]


def test_optimize_config_mutation_by_f_is_harmless():
    def mutating(c):
        v = quad(c)
        c["x"] = 99.0
        c.clear()
        return v

    res = optimize(mutating, SPACE, budget=6, policy="fixed_bo")
    assert all(-2.0 <= r["config"]["x"] <= 2.0 for r in res.records)


def test_optimize_policy_arguments():
    res = optimize(quad, SPACE, budget=6, policy="fixed_bo", policy_kwargs={"n_init": 2})
    rationales = [r["rationale"] for r in res.records]
    assert sum("Initial Sobol" in t for t in rationales) == 2
    res = optimize(quad, SPACE, budget=4, policy=FixedBOPolicy(n_init=4))
    assert res.policy == "fixed_bo" and all("Initial Sobol" in r["rationale"] for r in res.records)
    with pytest.raises(LenzError):
        optimize(quad, SPACE, budget=4, policy=FixedBOPolicy(), policy_kwargs={"n_init": 2})
    with pytest.raises(LenzError):
        optimize(quad, SPACE, budget=4, policy="no_such_policy")
    with pytest.raises(LenzError):
        optimize(quad, SPACE, budget=4, policy=3)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        optimize(quad, SPACE, budget=4, policy="sobol", policy_kwargs={"n_init": 2})


@pytest.mark.parametrize("budget", [0, -1, 2.5, True, "5"])
def test_optimize_validates_budget(budget):
    with pytest.raises(LenzError):
        optimize(quad, SPACE, budget=budget)  # type: ignore[arg-type]


def test_optimize_campaign_options():
    res = optimize(quad, SPACE, budget=5, regret="cumulative", campaign_options={"soft_actions_per_step": 20},
                   policy="sobol")
    assert res.regret_mode == "cumulative" and res.campaign_config["soft_actions_per_step"] == 20
    with pytest.raises(LenzError, match="repeat"):
        optimize(quad, SPACE, budget=5, campaign_options={"budget": 3})
    with pytest.raises(LenzError, match="unknown"):
        optimize(quad, SPACE, budget=5, campaign_options={"bogus": 1})
    with pytest.raises(LenzError):
        optimize(quad, SPACE, budget=5, regret="median")


def test_optimize_prior_region_and_config_steer_the_heuristic():
    region = {"x": [0.0, 1.0], "y": [-2.0, 0.0]}
    res = optimize(quad, SPACE, budget=10, f_star=0.0, prior={"region": region, "config": {"x": 0.25, "y": -0.9}})
    assert res.records[0]["config"] == {"x": 0.25, "y": -0.9}
    assert res.records[0]["source"] == "agent"
    set_bounds = [e for e in res.log if e.get("tool") == "set_bounds"]
    assert set_bounds and set_bounds[0]["ok"] and set_bounds[0]["step"] == 1
    early = [r["config"] for r in res.records[1:5]]
    assert all(0.0 <= c["x"] <= 1.0 and -2.0 <= c["y"] <= 0.0 for c in early)
    assert any(c["source"] == "prior" for c in res.context)


def test_optimize_context_and_outputs_reach_the_agent():
    def f(c):
        return {"y": quad(c), "cost": abs(c["x"])}

    ctx = [{"text": "Keep the cost at or below 0.5.",
            "data": {"op": "set_constraints", "args": {"constraints": [{"metric": "cost", "upper": 0.5}]}}},
           "Plain user note."]
    res = optimize(f, SPACE, budget=8, outputs={"cost": "cost of a run"}, context=ctx, seed=2)
    sources = [c["source"] for c in res.context]
    assert sources.count("user") == 2
    sc = [e for e in res.log if e.get("tool") == "set_constraints"]
    assert sc and sc[0]["ok"]
    # without the declaration the agent cannot name the metric (it is not in the tool enum)
    res2 = optimize(f, SPACE, budget=3, context=ctx[:1], policy="heuristic")
    sc2 = [e for e in res2.log if e.get("tool") == "set_constraints"]
    assert sc2 and not sc2[0]["ok"]


def test_optimize_mixed_space():
    def f(c):
        return (c["x"] - 0.5) ** 2 + (c["k"] - 3) ** 2 * 0.1 + (0.0 if c["opt"] == "b" else 0.5)

    space = {"x": (0, 1), "k": {"type": "int", "lower": 1, "upper": 6},
             "opt": {"type": "categorical", "choices": ["a", "b", "c"]}}
    res = optimize(f, space, budget=12, f_star=0.0, seed=0)
    assert res.n_evals == 12 and res.forced_commits == 0
    assert all(isinstance(r["config"]["k"], int) and r["config"]["opt"] in "abc" for r in res.records)


def test_optimize_verbose_prints_progress(capsys):
    optimize(quad, SPACE, budget=4, policy="sobol", f_star=0.0, verbose=True)
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("[optimize] sobol on 'quad'")
    assert sum("eval " in line for line in out) == 4 and "simple regret" in out[-1]
    assert any("*" in line for line in out[1:5])


def test_optimize_works_with_a_custom_policy():
    class FirstCandidate(Policy):
        name = "first"

        def step(self, session):
            cand = session.call("suggest", {"acqf": "sobol"})["result"]["candidates"][0]["config"]
            session.call("evaluate", {"config": cand, "rationale": "test"})

    res = optimize(quad, SPACE, budget=3, policy=FirstCandidate())
    assert res.policy == "first" and res.n_evals == 3 and res.forced_commits == 0


def test_package_exports_are_lazy_and_complete():
    for name in agentic_bo.__all__:
        assert getattr(agentic_bo, name) is not None
    assert agentic_bo.optimize is optimize
    with pytest.raises(AttributeError):
        agentic_bo.does_not_exist  # noqa: B018


def test_optimize_failed_first_measurement_keeps_cumulative_regret_defined():
    # a failed first measurement used to make the whole cumulative regret curve NaN
    calls = {"n": 0}

    def f(cfg):
        calls["n"] += 1
        return None if calls["n"] == 1 else (cfg["x"] - 0.3) ** 2

    res = optimize(f, {"x": (0.0, 1.0)}, budget=6, f_star=0.0, policy="sobol", seed=0)
    regrets = [-r["utility"] for r in res.records[1:]]                  # u* = 0 for f_star = 0
    assert res.records[0]["utility"] in (None, -math.inf)                 # the failed measurement
    assert math.isnan(res.cumulative_regret[0])
    # the failed evaluation is charged the worst regret seen at the first finite observation
    assert res.cumulative_regret[1:] == pytest.approx(np.cumsum([2 * regrets[0]] + regrets[1:]).tolist())
    assert math.isfinite(res.final_cumulative_regret)


def test_as_space_parameter_named_params():
    # {"params": [...specs]} is the SearchSpace.from_spec form, but a parameter that happens
    # to be called "params" with a (lower, upper) pair used to be misread as that form
    assert as_space({"params": (0, 1)}).names == ["params"]
    assert as_space({"params": [0.0, 2.0], "y": (0, 1)}).names == ["params", "y"]
    assert as_space({"params": [{"name": "a", "type": "real", "lower": 0, "upper": 1}]}).names == ["a"]
