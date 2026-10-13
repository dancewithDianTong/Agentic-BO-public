"""Tests for the Sara harness: tools, AgentSession enforcement, Campaign loop, CampaignResult."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from agentic_bo.lenz import Lenz
from agentic_bo.lenz.engine import DUPLICATE_TOL
from agentic_bo.metrics import cumulative_regret_curve, simple_regret_curve
from agentic_bo.problems import Problem, get_problem
from agentic_bo.sara import (
    AgentSession, Campaign, CampaignConfig, CampaignResult, ContextItem, Policy, PolicyError, TOOLSETS,
    build_tools, compute_regret_curves,
)
from agentic_bo.sara.policies import FixedBOPolicy, SobolPolicy
from agentic_bo.sara.schema import validate
from agentic_bo.sara.tools import (
    compact, limit_size, present, round_param_value, round_sig, trial_row,
)
from agentic_bo.space import RealParam, SearchSpace
from agentic_bo.utils import LenzError, dumps

# --------------------------------------------------------------------------- helpers


def branin_session(budget: int = 10, **cfg) -> AgentSession:
    return AgentSession(get_problem("branin"), CampaignConfig(budget=budget, **cfg))


def evaluate(session: AgentSession, config: dict, rationale: str = "test point") -> dict:
    """Open a step if needed and evaluate ``config``; return the envelope."""
    session.begin_step()
    return session.call("evaluate", {"config": config, "rationale": rationale})


def seeded_session(n: int = 4, budget: int = 12, **cfg) -> AgentSession:
    s = branin_session(budget, **cfg)
    pts = [{"x0": -3.0, "x1": 12.0}, {"x0": 3.0, "x1": 2.0}, {"x0": 9.0, "x1": 3.0}, {"x0": 0.0, "x1": 7.5},
           {"x0": 6.0, "x1": 10.0}, {"x0": -1.0, "x1": 1.0}]
    for cfg_ in pts[:n]:
        assert evaluate(s, cfg_)["ok"]
    s.begin_step()
    return s


def quad_problem(f_star: float | None = 0.0, fail_when=None, noise_std: float = 0.0) -> Problem:
    """Concave 2-D quadratic ``y = -((x0-0.3)^2 + (x1-0.6)^2)`` (maximized), optionally failing."""

    def fn(cfg: dict) -> dict:
        if fail_when is not None and fail_when(cfg):
            raise RuntimeError("simulated instrument failure")
        return {"y": -((cfg["x0"] - 0.3) ** 2 + (cfg["x1"] - 0.6) ** 2)}

    space = SearchSpace([RealParam("x0", 0, 1), RealParam("x1", 0, 1)])
    return Problem("quad", space, fn, "y", False, f_star, noise_std=noise_std, default_budget=8,
                   x_star={"x0": 0.3, "x1": 0.6})


class ScriptedPolicy(Policy):
    """Evaluates fixed points; optional per-step hooks for failure injection."""

    name = "scripted"

    def __init__(self, points=None, fail_at=None, stop_at=None, skip=False, start_error=False):
        self.points = points
        self.fail_at = fail_at
        self.stop_at = stop_at
        self.skip = skip
        self.start_error = start_error
        self.contexts: list[tuple[int, ContextItem]] = []
        self.steps: list[int] = []
        self.started = self.finished = False

    def start(self, session):
        self.started = True
        if self.start_error:
            raise PolicyError("no API key")

    def step(self, session):
        self.steps.append(session.t)
        if self.fail_at is not None and session.t >= self.fail_at:
            raise PolicyError("model refused")
        if self.stop_at is not None and session.t >= self.stop_at:
            session.call("stop", {"reason": "good enough"})
            return
        if self.skip:
            session.call("status")
            return
        if self.points is not None:
            cfg = self.points[session.t % len(self.points)]
        else:
            cfg = session.call("suggest", {"acqf": "sobol"})["result"]["candidates"][0]["config"]
        session.call("evaluate", {"config": cfg, "rationale": "scripted"})

    def on_context(self, session, item):
        self.contexts.append((session.t, item))

    def finish(self, session):
        self.finished = True

    def usage(self):
        return {"api_calls": 0}


# --------------------------------------------------------------------------- tool registry


def test_tool_sets_sorted_and_conditional():
    full = [t.name for t in branin_session().tools]
    assert full == sorted(full)
    assert "stop" not in full and "create" not in full
    assert set(full) == set(TOOLSETS["full"]) - {"stop", "create"}
    both = [t.name for t in branin_session(allow_early_stop=True, agent_setup=True).tools]
    assert "stop" in both and "create" in both
    llm = [t.name for t in branin_session(toolset="llm_only").tools]
    assert llm == ["evaluate", "incumbent", "record_note", "status", "trials"]
    with pytest.raises(LenzError, match="unknown toolset"):
        build_tools(branin_session(), "nope")


def test_tool_definitions_are_anthropic_dicts_and_deterministic():
    s1, s2 = branin_session(allow_early_stop=True), branin_session(allow_early_stop=True)
    defs = s1.tool_definitions()
    assert [d["name"] for d in defs] == sorted(d["name"] for d in defs)
    for d in defs:
        assert set(d) == {"name", "description", "input_schema"}
        assert d["input_schema"]["type"] == "object"
        assert d["input_schema"]["additionalProperties"] is False
        assert len(d["description"]) > 80
    assert json.dumps(defs, sort_keys=True) == json.dumps(s2.tool_definitions(), sort_keys=True)
    by = {d["name"]: d["description"] for d in defs}
    for free in ("suggest", "predict", "score"):
        assert "records nothing" in by[free] and "spends no budget" in by[free]
    assert "only action that spends budget" in by["evaluate"]
    assert "raw units" in by["predict"] and "radius" in by["suggest"]
    assert "[lo, hi]" in by["set_bounds"]
    schema = {d["name"]: d["input_schema"] for d in defs}
    assert schema["evaluate"]["required"] == ["config", "rationale"]
    assert schema["evaluate"]["properties"]["config"]["required"] == ["x0", "x1"]
    assert schema["evaluate"]["properties"]["config"]["properties"]["x0"]["minimum"] == -5.0


def test_tool_schemas_do_not_change_during_a_campaign():
    s = seeded_session(3)
    before = json.dumps(s.tool_definitions(), sort_keys=True)
    s.call("set_bounds", {"bounds": {"x0": [0, 5]}})
    s.call("set_acqf", {"acqf": "ucb", "beta": 2})
    evaluate(s, {"x0": 1.0, "x1": 1.0})
    assert json.dumps(s.tool_definitions(), sort_keys=True) == before


VALID_CALLS = [
    ("status", {}),
    ("trials", {}),
    ("trials", {"last": 2, "sort_by": "value"}),
    ("trials", {"sort_by": "eval_index", "last": None}),
    ("incumbent", {}),
    ("pareto", {}),
    ("diagnostics", {}),
    ("diagnostics", {"metric": "y"}),
    ("predict", {"configs": [{"x0": 1.0, "x1": 2.0}, {"x0": -5, "x1": 15}]}),
    ("score", {"configs": [{"x0": 1.0, "x1": 2.0}]}),
    ("score", {"configs": [{"x0": 1.0, "x1": 2.0}, {"x0": 3, "x1": 3}], "acqfs": ["ucb", "pm", "logei"]}),
    ("set_bounds", {"bounds": {"x0": [-2, 6]}}),
    ("set_bounds", {"bounds": None}),
    ("set_acqf", {"acqf": "ucb", "beta": 9.0}),
    ("set_acqf", {"acqf": "ts", "params": {"n_candidates": 256}}),
    ("set_objectives", {"objectives": [{"metric": "y", "minimize": True}]}),
    ("set_constraints", {"constraints": []}),
    ("set_constraints", {"constraints": [{"metric": "y", "upper": 100.0}]}),
    ("suggest", {}),
    ("suggest", {"q": 3}),
    ("suggest", {"around": "incumbent", "radius": 0.2}),
    ("suggest", {"around": {"x0": 1.0, "x1": 2.0}, "radius": 0.5, "acqf": "pm"}),
    ("suggest", {"bounds": "full", "acqf": "ucb"}),
    ("suggest", {"bounds": {"x1": [0, 5]}, "q": 2, "acqf": "sobol"}),
    ("record_note", {"text": "x1 looks more sensitive than x0"}),
    ("evaluate", {"config": {"x0": 2.0, "x1": 4.0}, "rationale": "probe the middle", "expectation": "y ~ 10"}),
]


@pytest.mark.parametrize("name, args", VALID_CALLS, ids=[f"{n}-{i}" for i, (n, _) in enumerate(VALID_CALLS)])
def test_every_tool_works_with_valid_args(name, args):
    s = seeded_session(4)
    env = s.call(name, args)
    assert env["ok"], env
    json.dumps(env, allow_nan=False)
    entry = s.log[-1]
    assert entry["tool"] == name and entry["ok"] and entry["t"] == 4


INVALID_CALLS = [
    ("status", {"verbose": True}, "unknown property 'verbose'"),
    ("trials", {"last": 0}, "$.last: must be >= 1"),
    ("trials", {"sort_by": "y"}, "$.sort_by"),
    ("diagnostics", {"metric": "nope"}, "$.metric"),
    ("predict", {"configs": []}, "$.configs: must have at least 1 item(s)"),
    ("predict", {"configs": [{"x0": 1.0}]}, "$.configs[0].x1: required property is missing"),
    ("predict", {"configs": [{"x0": 1.0, "x1": 2.0}] * 51}, "at most 50 item(s)"),
    ("score", {"configs": [{"x0": 1, "x1": 1}], "acqfs": ["ts"]}, "$.acqfs[0]"),
    ("set_bounds", {}, "$.bounds: required property is missing"),
    ("set_bounds", {"bounds": {"x0": [5, 1]}}, "lower 5.0 > upper 1.0"),
    ("set_bounds", {"bounds": {"x0": [0, 99]}}, "$.bounds.x0[1]: must be <= 10"),
    ("set_acqf", {"acqf": "magic"}, "$.acqf"),
    ("set_acqf", {"acqf": "logei", "beta": 2.0}, "does not take parameter 'beta'"),
    ("set_acqf", {"acqf": "ehvi"}, "multi-objective"),
    ("set_objectives", {"objectives": [{"metric": "loss", "minimize": True}]}, "$.objectives[0].metric"),
    ("set_objectives", {"objectives": []}, "at least 1 item(s)"),
    ("set_constraints", {"constraints": [{"metric": "y"}]}, "give a lower and/or an upper bound"),
    ("suggest", {"q": 11}, "$.q: must be <= 10"),
    ("suggest", {"radius": 0}, "$.radius: must be > 0"),
    ("suggest", {"around": "best"}, "$.around"),
    ("suggest", {"around": {"x0": 1.0}}, "$.around.x1: required property is missing"),
    ("suggest", {"bounds": "half"}, "$.bounds"),
    ("record_note", {"text": ""}, "$.text: must be a non-empty string"),
    ("record_note", {"text": "   "}, "blank"),
    ("evaluate", {"config": {"x0": 1.0, "x1": 2.0}}, "$.rationale: required property is missing"),
    ("evaluate", {"config": {"x0": 11.0, "x1": 2.0}, "rationale": "r"}, "$.config.x0: must be <= 10, got 11"),
    ("evaluate", {"config": {"x0": 1.0, "x1": 2.0}, "rationale": "  "}, "rationale must not be blank"),
    ("evaluate", {"config": [1.0, 2.0], "rationale": "r"}, "$.config: expected object, got array"),
    ("nonexistent", {}, "unknown tool 'nonexistent'; available tools: diagnostics"),
    ("stop", {"reason": "x"}, "early stopping is not allowed"),
    ("create", {}, "create is not available"),
]


@pytest.mark.parametrize("name, args, needle", INVALID_CALLS,
                         ids=[f"{n}-{i}" for i, (n, _, _) in enumerate(INVALID_CALLS)])
def test_invalid_calls_return_error_envelopes(name, args, needle):
    s = seeded_session(4)
    n_records = len(s.records)
    env = s.call(name, args)
    assert env["ok"] is False and isinstance(env["error"], str), env
    assert needle in env["error"], env["error"]
    assert len(s.records) == n_records          # nothing was evaluated
    assert s.log[-1]["ok"] is False and s.log[-1]["error"] == env["error"]


def test_call_never_raises_on_garbage():
    s = seeded_session(2)
    rng = np.random.default_rng(0)
    garbage = [None, 3, "x", [], [1, 2], {"a": float("nan")}, {"config": None}, {"configs": "all"},
               {"bounds": {"x0": "wide"}}, {"q": "2"}, {"acqf": None, "q": -1}, object()]
    names = [t.name for t in s.tools] + ["", None, 42, "SUGGEST", "set-bounds"]
    for _ in range(150):
        name = names[int(rng.integers(len(names)))]
        args = garbage[int(rng.integers(len(garbage)))]
        env = s.call(name, args)
        assert isinstance(env, dict) and env["ok"] in (True, False)
        if not env["ok"]:
            assert isinstance(env["error"], str) and env["error"]
        json.dumps(env, allow_nan=False)
    assert len(s.log) >= 150


def test_hyphen_and_case_tolerant_names():
    s = seeded_session(2)
    assert s.call("set-bounds", {"bounds": {"x0": [0, 1]}})["ok"]
    assert s.call("STATUS")["ok"]


def test_internal_errors_become_envelopes(monkeypatch):
    s = seeded_session(2)

    def boom(*a, **k):
        raise ZeroDivisionError("oops")

    monkeypatch.setattr(s.lenz, "suggest", boom)
    monkeypatch.setattr(s.lenz, "call", boom)
    env = s.call("suggest", {})
    assert env == {"ok": False, "error": "internal error (ZeroDivisionError): oops"}


def test_lenz_internal_error_type_is_reported():
    s = seeded_session(2)
    s.lenz.call = lambda command, args=None: {"ok": False, "command": command,
                                              "error": {"type": "KeyError", "message": "'x'"}}
    env = s.call("status")
    assert env["ok"] is False and "internal error in lenz status (KeyError)" in env["error"]


def test_set_objectives_and_constraints_check_metrics_against_problem():
    s = AgentSession(get_problem("reaction_yield"), CampaignConfig(budget=5))
    env = s.call("set_objectives", {"objectives": [{"metric": "yield", "minimize": False},
                                                   {"metric": "cost", "minimize": True}]})
    assert env["ok"] and env["result"]["is_moo"] and env["result"]["acqf"] == "ehvi"
    assert s.call("set_constraints", {"constraints": [{"metric": "cost", "upper": 4.0}]})["ok"]
    assert s.lenz.shelf.constraints == [{"metric": "cost", "lower": None, "upper": 4.0}]
    assert s.known_metrics() == ["yield", "cost"]
    env = s.call("set_constraints", {"constraints": [{"metric": "purity", "lower": 0.9}]})
    assert env["ok"] is False and "$.constraints[0].metric" in env["error"]


# --------------------------------------------------------------------------- presentation


def test_round_helpers():
    assert round_sig(3.14159265358979) == 3.14159
    assert round_sig(0.0) == 0.0 and math.isinf(round_sig(math.inf))
    p = RealParam("x", -5, 10)
    assert round_param_value(p, 3.14159265358979) == 3.141593
    q = RealParam("big", 1000, 1001)
    v = round_param_value(q, 1000.123456789)
    assert abs(v - 1000.123456789) < 1e-6 * 1 and v != 1000.123456789
    edge = RealParam("e", 0.0, 0.12345678912)
    assert round_param_value(edge, 0.12345678912) <= 0.12345678912      # stays inside the domain
    lg = RealParam("l", 1.0, 1.001, log_scale=True)
    w = 1.000123456789
    assert abs(math.log(round_param_value(lg, w) / w)) / math.log(1.001) < 1e-6
    assert round_param_value(p, 3) == 3          # ints untouched
    assert round_param_value(RealParam("u", 0, 1), 52.123456789) == 52.1235     # not a value: not clipped


def test_compact_rounds_metrics_and_keeps_configs_resubmittable():
    space = get_problem("branin").space
    raw = {"config": {"x0": 1.23456789012, "x1": 0.1}, "metrics": {"y": 21.599731413359063},
           "bounds": {"x0": [-1.11111111111, 2.0]}, "values": [np.float64(1 / 3), np.nan, np.inf],
           "n": np.int64(3), "flag": np.bool_(True)}
    out = compact(raw, space)
    assert out["config"]["x0"] == 1.234568 and out["metrics"]["y"] == 21.5997
    assert compact({"lengthscales": {"x0": 250.0, "x1": 0.5}}, space)["lengthscales"] == {"x0": 250.0, "x1": 0.5}
    assert out["bounds"]["x0"] == [-1.111111, 2.0]
    assert out["values"] == [0.333333, None, None] and out["n"] == 3 and out["flag"] is True
    json.dumps(out, allow_nan=False)
    assert space.distance(out["config"], raw["config"]) < 1e-6


def test_limit_size_head_tail_and_summary():
    big = {"trials": [{"i": i, "pad": "x" * 50} for i in range(200)], "n": 200}
    head = limit_size(big, max_chars=2000, list_key="trials", keep="head", hint="use last")
    assert len(json.dumps(head)) <= 2000 and head["trials"][0]["i"] == 0
    assert "first" in head["truncated"] and "use last" in head["truncated"]
    tail = limit_size(big, max_chars=2000, list_key="trials", keep="tail")
    assert tail["trials"][-1]["i"] == 199 and "last" in tail["truncated"]
    k = len(tail["trials"])
    assert len(json.dumps({**big, "trials": big["trials"][-(k + 1):], "truncated": tail["truncated"]})) > 2000
    small = {"a": 1}
    assert limit_size(small, max_chars=100) is small
    blob = {"text": "y" * 5000}
    assert "truncated" in limit_size(blob, max_chars=100)
    assert set(present({"x": [1.0] * 10_000}, max_chars=500)) == {"x", "truncated"}


def test_trial_rows_are_compact():
    s = seeded_session(3)
    rows = s.call("trials")["result"]["trials"]
    assert len(rows) == 3
    assert set(rows[0]) == {"trial_id", "eval_index", "source", "config", "metrics", "feasible"}
    assert trial_row({"trial_id": 1, "status": "abandoned"})["status"] == "abandoned"
    # the rationale is not repeated in trial rows (it lives in lenz meta)
    assert "rationale" not in json.dumps(rows)


def test_trials_truncation_keeps_recent_or_best(monkeypatch):
    import agentic_bo.sara.tools as tools_mod

    s = seeded_session(6, budget=10)
    monkeypatch.setattr(tools_mod, "MAX_RESULT_CHARS", 700)
    recent = s.call("trials")["result"]
    assert "truncated" in recent and recent["trials"][-1]["trial_id"] == 5
    best = s.call("trials", {"sort_by": "value"})["result"]
    ys = [r["metrics"]["y"] for r in best["trials"]]
    assert "truncated" in best and ys == sorted(ys)
    assert ys[0] == pytest.approx(min(r["observed"]["y"] for r in s.records), rel=1e-5)


def test_trials_limit_is_top_k_after_sorting_and_last_is_most_recent():
    """Regression (reconciliation of lenz-trials-last-before-sort): the tool exposes lenz's ``limit``,
    so sort_by='value' with limit=k is the k best trials, while ``last`` cuts before sorting."""
    s = seeded_session(6, budget=10)
    ys = {r["trial_id"]: r["observed"]["y"] for r in s.records}
    best2 = sorted(ys, key=ys.get)[:2]
    top = s.call("trials", {"sort_by": "value", "limit": 2})["result"]
    assert [r["trial_id"] for r in top["trials"]] == best2 and top["n"] == 2 and top["n_total"] == 6
    recent = s.call("trials", {"sort_by": "value", "last": 2})["result"]
    assert sorted(r["trial_id"] for r in recent["trials"]) == [4, 5]
    both = s.call("trials", {"sort_by": "value", "last": 3, "limit": 1})["result"]
    assert [r["trial_id"] for r in both["trials"]] == [min((3, 4, 5), key=ys.get)]
    assert not s.call("trials", {"limit": 0})["ok"]
    props = next(d for d in s.tool_definitions() if d["name"] == "trials")["input_schema"]["properties"]
    assert set(props) == {"last", "sort_by", "limit"}


def test_metric_values_keep_their_resolution_under_a_large_offset():
    """Regression (heuristic owner's request): metric values were rounded to six significant digits,
    so on y ~ 1e4 an agent saw 0.1 resolution and could not tell real improvements apart."""
    base = quad_problem()

    def shifted(cfg: dict) -> dict:
        return {"y": 1e4 + base.evaluate_fn(cfg)["y"]}

    p = Problem("quad_offset", base.space, shifted, "y", False, 1e4, default_budget=8)
    s = AgentSession(p, CampaignConfig(budget=8))
    cfgs = [{"x0": 0.31, "x1": 0.6}, {"x0": 0.3, "x1": 0.62}, {"x0": 0.1, "x1": 0.9}]
    shown = [evaluate(s, c)["result"]["observed"]["y"] for c in cfgs]
    true = [shifted(c)["y"] for c in cfgs]
    assert shown == pytest.approx(true, rel=1e-9, abs=0) and shown[0] != shown[1]
    s.begin_step()
    rows = s.call("trials")["result"]["trials"]
    assert [r["metrics"]["y"] for r in rows] == shown
    inc = s.call("incumbent")["result"]["incumbent"]
    assert inc["value"] == pytest.approx(true[0], rel=1e-9, abs=0)
    mean = s.call("predict", {"configs": [cfgs[0]]})["result"]["predictions"][0]["metrics"]["y"]["mean"]
    raw_mean = s.lenz.predict([cfgs[0]])["predictions"][0]["metrics"]["y"]["mean"]
    assert mean == pytest.approx(raw_mean, rel=1e-9, abs=0)
    # other floats keep the short form
    std = s.call("predict", {"configs": [cfgs[0]]})["result"]["predictions"][0]["metrics"]["y"]["std"]
    assert std == round_sig(std, 6)


# --------------------------------------------------------------------------- evaluate & records


def test_evaluate_flow_records_and_lenz_agree():
    p = quad_problem(noise_std=0.05)
    s = AgentSession(p, CampaignConfig(budget=4, seed=3))
    env = evaluate(s, {"x0": 0.1, "x1": 0.2}, rationale="explore corner")
    assert env["ok"]
    r = env["result"]
    assert set(r) >= {"trial_id", "config", "observed", "is_new_best", "incumbent", "t", "remaining"}
    assert r["t"] == 1 and r["remaining"] == 3 and r["is_new_best"] is True
    rec = s.records[0]
    true_y = -((0.1 - 0.3) ** 2 + (0.2 - 0.6) ** 2)
    assert rec["true_metrics"]["y"] == pytest.approx(true_y)
    assert rec["observed"]["y"] != pytest.approx(true_y, abs=1e-9)       # noisy observation
    assert rec["utility"] == pytest.approx(true_y) and rec["regret"] == pytest.approx(-true_y)
    assert rec["feasible"] and rec["source"] == "agent" and rec["rationale"] == "explore corner"
    assert rec["forced"] is False and rec["t"] == 1 and rec["step"] == 1
    trial = s.lenz.frame.trials[0]
    assert trial.status == "completed" and trial.metrics["y"] == pytest.approx(rec["observed"]["y"])
    assert trial.meta["rationale"] == "explore corner" and trial.source == "agent"
    assert r["observed"]["y"] == pytest.approx(rec["observed"]["y"], rel=1e-5)
    assert s.actions_per_step == [0]


def test_noise_is_reproducible_from_the_seed():
    def run(seed):
        s = AgentSession(quad_problem(noise_std=0.1), CampaignConfig(budget=3, seed=seed))
        for x in (0.1, 0.5, 0.9):
            evaluate(s, {"x0": x, "x1": x})
        return [r["observed"]["y"] for r in s.records]

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_one_evaluate_per_step_and_actions_after_commit_count_toward_next_step():
    s = branin_session(5)
    assert evaluate(s, {"x0": 1.0, "x1": 1.0})["ok"]
    second = s.call("evaluate", {"config": {"x0": 2.0, "x1": 2.0}, "rationale": "again"})
    assert second["ok"] is False and "already committed 1 evaluation(s)" in second["error"]
    assert s.t == 1 and len(s.records) == 1
    s.call("status")                      # after the commit: counts toward step 2
    assert s.log[-1]["step"] == 2 and s.log[-1]["t"] == 1
    s.begin_step()
    assert s.step == 2 and s.actions_this_step == 2   # rejected evaluate + status carried over
    s.begin_step()                         # no-op while the step is open
    assert s.step == 2
    assert s.call("evaluate", {"config": {"x0": 2.0, "x1": 2.0}, "rationale": "now"})["ok"]
    assert s.actions_per_step == [0, 2]


def test_max_evals_per_step_allows_batches():
    s = branin_session(6, max_evals_per_step=3)
    s.begin_step()
    for i in range(3):
        assert s.call("evaluate", {"config": {"x0": float(i), "x1": 1.0}, "rationale": "batch"})["ok"]
    assert s.call("evaluate", {"config": {"x0": 4.0, "x1": 1.0}, "rationale": "batch"})["ok"] is False
    assert s.t == 3 and s.actions_per_step == [0]


def test_budget_exhaustion():
    s = branin_session(2)
    assert evaluate(s, {"x0": 0.0, "x1": 0.0})["ok"]
    env = evaluate(s, {"x0": 1.0, "x1": 0.0})
    assert env["ok"] and "last evaluation" in env["result"]["note"] and env["result"]["remaining"] == 0
    env = evaluate(s, {"x0": 2.0, "x1": 0.0})
    assert env["ok"] is False and "budget is exhausted" in env["error"]
    assert s.commit_fallback("test")["ok"] is False
    assert s.is_done and s.t == 2
    assert "Campaign finished: 2/2" in s.status_line()


def test_outside_region_and_duplicate_notes():
    s = seeded_session(2)
    s.call("set_bounds", {"bounds": {"x0": [0, 5]}})
    env = s.call("evaluate", {"config": {"x0": 8.0, "x1": 1.0}, "rationale": "override lenz"})
    assert "outside the active region" in env["result"]["note"]
    assert s.records[-1]["in_region"] is False
    env = evaluate(s, {"x0": -3.0, "x1": 12.0})
    assert "repeats trial 0" in env["result"]["note"] and s.records[-1]["duplicate_of"] == 0


def test_copied_rounded_suggestion_is_the_same_point():
    s = seeded_session(3)
    clone = Lenz.from_dict(json.loads(dumps(s.lenz.to_dict())))     # reproduces the next suggestion exactly
    raw = clone.suggest()["candidates"][0]["config"]
    shown = s.call("suggest", {})["result"]["candidates"][0]["config"]
    assert shown != raw and s.lenz.space.distance(raw, shown) <= DUPLICATE_TOL
    assert s.call("evaluate", {"config": shown, "rationale": "accept the suggestion"})["ok"]
    evaluate(s, raw)
    assert s.records[-1]["duplicate_of"] == s.records[-2]["trial_id"]
    assert s.lenz.frame.trials[-1].config == raw


def test_failed_black_box_spends_budget_and_is_infeasible():
    p = quad_problem(fail_when=lambda c: c["x0"] > 0.9)
    s = AgentSession(p, CampaignConfig(budget=3))
    env = evaluate(s, {"x0": 0.95, "x1": 0.5})
    assert env["ok"] and "evaluation failed" in env["result"]["note"]
    assert env["result"]["observed"] == {"y": None}
    rec = s.records[0]
    assert rec["error"] and rec["feasible"] is False and rec["utility"] == -math.inf
    assert s.t == 1 and s.lenz.frame.trials[0].metrics == {"y": None}
    evaluate(s, {"x0": 0.3, "x1": 0.5})
    simple, cumulative = compute_regret_curves(p, s.records)
    assert math.isnan(simple[0]) and simple[1] == pytest.approx(0.01)
    assert s.status_line().endswith("best y=-0.01 (trial 1)")


# --------------------------------------------------------------------------- limits


def test_soft_limit_warning():
    s = branin_session(5, soft_actions_per_step=2, hard_actions_per_step=10)
    assert "warning" not in s.call("status")["result"]
    assert "warning" not in s.call("status")["result"]
    third = s.call("status")
    assert "action 3 of step 1 (soft limit 2)" in third["result"]["warning"]
    bad = s.call("trials", {"last": -1})
    assert bad["ok"] is False and "soft limit" in bad["warning"]
    ok_eval = s.call("evaluate", {"config": {"x0": 0.0, "x1": 0.0}, "rationale": "commit"})
    assert ok_eval["ok"] and "warning" not in ok_eval["result"]


def test_hard_limit_forces_a_fallback_commit():
    s = seeded_session(3, budget=8, soft_actions_per_step=2, hard_actions_per_step=3)
    for _ in range(3):
        assert s.call("status")["ok"]
    env = s.call("predict", {"configs": [{"x0": 0.0, "x1": 0.0}]})
    assert env["ok"] is False and env["forced"] is True
    assert "harness evaluated a fallback point" in env["error"] and env["fallback"]["trial_id"] == 3
    assert s.step_committed and s.t == 4
    rec = s.records[-1]
    assert rec["forced"] and rec["source"] == "harness-fallback" and "hard limit" in rec["rationale"]
    forced_entries = [e for e in s.log if e["forced"]]
    assert len(forced_entries) == 1 and forced_entries[0]["tool"] == "evaluate"
    assert s.log[-2]["tool"] == "predict" and s.log[-2]["ok"] is False   # the call that was not run
    # the agent may still commit its own evaluate as the 4th action of a fresh step
    s.begin_step()
    for _ in range(3):
        s.call("status")
    assert s.call("evaluate", {"config": {"x0": 5.0, "x1": 5.0}, "rationale": "late but mine"})["ok"]


def test_commit_fallback_prefers_top_suggestion_then_sobol(monkeypatch):
    s = seeded_session(3)
    env = s.commit_fallback("policy ended the step without evaluating")
    assert env["ok"] and s.records[-1]["forced"]
    assert "top logei suggestion" in s.records[-1]["rationale"]
    s.begin_step()
    calls = []
    original = s.lenz.suggest

    def flaky(*a, **kw):
        calls.append(kw)
        if kw.get("acqf") != "sobol":
            raise RuntimeError("model fit failed")
        return original(*a, **kw)

    monkeypatch.setattr(s.lenz, "suggest", flaky)
    assert s.commit_fallback("again")["ok"]
    assert "Sobol point" in s.records[-1]["rationale"] and calls[-1] == {"q": 1, "acqf": "sobol"}
    s.begin_step()
    monkeypatch.setattr(s.lenz, "suggest", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    assert s.commit_fallback("third")["ok"]
    assert "uniform random point" in s.records[-1]["rationale"]


# --------------------------------------------------------------------------- early stop


def test_stop_only_when_allowed():
    s = branin_session(5, allow_early_stop=True)
    early = s.call("stop", {"reason": "nothing to do"})
    assert early["ok"] is False and "at least one evaluation" in early["error"] and not s.stopped
    evaluate(s, {"x0": 1.0, "x1": 1.0})
    s.begin_step()
    env = s.call("stop", {"reason": "converged"})
    assert env["ok"] and env["result"]["unused_budget"] == 4
    assert s.stopped and s.is_done and s.stop_reason == "converged"
    assert s.call("evaluate", {"config": {"x0": 2.0, "x1": 2.0}, "rationale": "r"})["ok"] is False
    assert s.call("stop", {"reason": "again"})["ok"] is False
    assert s.call("status")["ok"]           # probes still work


@pytest.mark.parametrize("mode", ["simple", "cumulative"])
def test_early_stop_padding(mode):
    p = quad_problem()
    pts = [{"x0": 0.9, "x1": 0.1}, {"x0": 0.35, "x1": 0.55}, {"x0": 0.0, "x1": 1.0}]
    res = Campaign(p, ScriptedPolicy(points=pts, stop_at=3),
                   CampaignConfig(budget=8, allow_early_stop=True, regret=mode)).run()
    assert res.stopped_early and res.n_evals == 3 and len(res.simple_regret) == 8
    regrets = [r["regret"] for r in res.records]
    best = min(regrets)
    assert res.simple_regret[-1] == pytest.approx(best) and res.simple_regret[3:] == [res.simple_regret[2]] * 5
    assert res.cumulative_regret[2] == pytest.approx(sum(regrets))
    assert res.cumulative_regret[-1] == pytest.approx(sum(regrets) + 5 * best)
    assert res.stop_reason == "good enough" and res.actions_per_step[-1] == 0


def test_stop_not_allowed_falls_back():
    res = Campaign(quad_problem(), ScriptedPolicy(stop_at=2), CampaignConfig(budget=5)).run()
    assert not res.stopped_early and res.n_evals == 5 and res.forced_commits == 3
    assert len(res.simple_regret) == 5


# --------------------------------------------------------------------------- context


def test_brief_contents_and_goal_modes():
    s = branin_session(17)
    b = s.brief()
    assert "Branin" in b and "| `x0` | real | [-5, 10] |" in b and "budget of 17 evaluations" in b
    assert "simple regret" in b and "cumulative" not in b and "Prior knowledge" not in b
    assert "0.397887" not in b and "regret" in b
    c = branin_session(17, regret="cumulative", allow_early_stop=True).brief()
    assert "cumulative regret: every evaluation counts" in c and "charged as if it repeated" in c
    y = AgentSession(get_problem("reaction_yield"), CampaignConfig(context=["Prefer low cost.",
                                                                            {"text": "Team note", "source": "pi"}]))
    yb = y.brief()
    assert "## Prior knowledge" in yb and "70-100 °C" in yb and "budget of 40" in yb
    assert yb.index("## Prior knowledge") < yb.index("## Additional context")
    assert "Prefer low cost." in yb and "Team note" in yb
    assert [c.source for c in y.context] == ["problem", "prior", "user", "pi"]
    assert y.context[1].data["prior_hint"]["region"]["temperature"] == [70.0, 100.0]
    hidden = AgentSession(get_problem("reaction_yield"), CampaignConfig(show_prior=False))
    assert "Prior knowledge" not in hidden.brief() and [c.source for c in hidden.context] == ["problem"]
    moo = AgentSession(get_problem("branin_currin"), CampaignConfig(budget=9)).brief()
    assert "hypervolume" in moo


def test_status_line_reports_observed_best():
    s = seeded_session(3)
    line = s.status_line()
    assert line.startswith("Step 4/12 | remaining 9 | best y=")
    best = min(r["observed"]["y"] for r in s.records)
    assert f"{best:.6g}" in line
    moo = AgentSession(get_problem("branin_currin"), CampaignConfig(budget=4))
    assert moo.status_line() == "Step 1/4 | remaining 4 | no evaluations yet"
    evaluate(moo, {"x0": 0.5, "x1": 0.1})
    assert "feasible evaluation(s), hypervolume" in moo.status_line()


def test_context_schedule_injection_timing():
    schedule = {0: "Initial instruction", 2: {"text": "Now keep y below 50", "op": "set_constraints",
                                              "args": {"constraints": [{"metric": "y", "upper": 50}]}},
                "3": ["first", {"text": "second", "source": "pi"}], 99: "never"}
    policy = ScriptedPolicy()
    camp = Campaign(get_problem("branin"), policy, CampaignConfig(budget=5, context_schedule=schedule))
    res = camp.run()
    got = [(t, item.text) for t, item in policy.contexts]
    assert got == [(0, "Initial instruction"), (2, "Now keep y below 50"), (3, "first"), (3, "second")]
    item = policy.contexts[1][1]
    assert item.data == {"op": "set_constraints", "args": {"constraints": [{"metric": "y", "upper": 50}]}}
    assert item.source == "instructor" and item.t == 2 and policy.contexts[3][1].source == "pi"
    # injected before the step that follows t evaluations
    assert policy.steps == [0, 1, 2, 3, 4]
    assert [c["text"] for c in res.context][-4:] == ["Initial instruction", "Now keep y below 50", "first", "second"]
    assert all(c["text"] != "never" for c in res.context)


def test_add_context_variants():
    s = seeded_session(2)
    item = s.add_context("Budget cut: be careful")
    assert isinstance(item, ContextItem) and item.t == 2 and item.source == "instructor"
    assert s.context[-1] is item
    item2 = s.add_context({"op": "set_bounds", "args": {"bounds": None}})
    assert item2.text.startswith("Instruction: ") and item2.data["op"] == "set_bounds"
    own = ContextItem(5, "user", "hello")
    assert s.add_context(own) is own
    with pytest.raises(LenzError):
        s.add_context(["a", "b"])
    assert s.brief().count("Budget cut") == 0      # mid-run context is not part of the initial brief


def test_campaign_config_validation_and_round_trip():
    for bad in ({"regret": "average"}, {"toolset": "tiny"}, {"budget": 0}, {"hard_actions_per_step": 0},
                {"on_policy_error": "ignore"}, {"context_schedule": {"-1": "x"}},
                {"context_schedule": {"a": "x"}}, {"context_schedule": {1: ""}}, {"context": 5},
                {"lenz_options": [1]}, {"seed": 1.5}, {"budget": "10"}, {"budget": True},
                {"max_evals_per_step": 0.5}):
        with pytest.raises(LenzError):
            CampaignConfig(**bad)
    cfg = CampaignConfig(budget=7, regret="cumulative", context="hi", context_schedule={1: "x", 2: {"op": "y"}},
                         lenz_options={"acqf": "ucb"}, seed=4)
    d = json.loads(dumps(cfg.to_dict()))
    again = CampaignConfig.from_dict(d)
    assert again == cfg
    with pytest.raises(LenzError, match="unknown CampaignConfig fields"):
        CampaignConfig.from_dict({"bogus": 1})


def test_lenz_options_are_applied_and_validated():
    s = branin_session(5, lenz_options={"acqf": "ucb", "acqf_params": {"beta": 1.0}, "model": {"kernel": "rbf"}})
    assert s.lenz.shelf.acqf == "ucb" and s.lenz.shelf.model["kernel"] == "rbf"
    with pytest.raises(LenzError, match="invalid lenz_options"):
        branin_session(5, lenz_options={"warp": True})
    moo = AgentSession(get_problem("branin_currin"), CampaignConfig(budget=4))
    assert [o["ref"] for o in moo.lenz.shelf.objectives] == [18.0, 6.0]


# --------------------------------------------------------------------------- notebook / setup / llm_only


def test_record_note_and_status_recent_notes():
    s = seeded_session(1)
    for i in range(4):
        assert s.call("record_note", {"text": f"note {i}"})["result"]["n_notes"] == i + 1
    st = s.call("status")["result"]
    assert st["recent_notes"] == ["note 1", "note 2", "note 3"]
    assert s.notebook[0] == {"t": 1, "step": 2, "text": "note 0"}
    assert st["lenz_config"]["acqf"] == "logei" and st["remaining"] == 11


def test_agent_setup_create_carries_trials_over():
    s = AgentSession(get_problem("constrained_hartmann6"), CampaignConfig(budget=8, agent_setup=True))
    for x in (0.1, 0.3, 0.6):
        evaluate(s, {f"x{i}": x for i in range(6)})
    s.begin_step()
    env = s.call("create", {"acqf": "ucb", "bounds": {"x0": [0, 0.5]}, "model": {"kernel": "rbf"},
                            "constraints": [{"metric": "c", "upper": 0.8}]})
    assert env["ok"], env
    assert env["result"]["carried_over"] == 3 and env["result"]["n_observed"] == 3
    shelf = s.lenz.shelf
    assert shelf.acqf == "ucb" and shelf.model["kernel"] == "rbf" and shelf.bounds == {"x0": [0.0, 0.5]}
    assert shelf.constraints == [{"metric": "c", "lower": None, "upper": 0.8}]
    assert [t.metrics for t in s.lenz.frame.trials] == [r["observed"] for r in s.records]
    bad = s.call("create", {"objectives": [{"metric": "zzz", "minimize": True}]})
    assert bad["ok"] is False and "zzz" in bad["error"]
    assert s.call("evaluate", {"config": {f"x{i}": 0.2 for i in range(6)}, "rationale": "after setup"})["ok"]
    assert s.lenz.n_observed == 4


def test_llm_only_toolset_hides_the_surrogate():
    s = AgentSession(get_problem("hartmann3"), CampaignConfig(budget=5, toolset="llm_only"))
    for x in (0.2, 0.5, 0.8):
        evaluate(s, {"x0": x, "x1": x, "x2": x})
    s.begin_step()
    inc = s.call("incumbent")["result"]
    assert "best_posterior" not in inc and inc["incumbent"]["trial_id"] in (0, 1, 2)
    st = s.call("status")["result"]
    assert "lenz_config" not in st
    assert s.call("suggest")["ok"] is False
    by = {d["name"]: d["description"] for d in s.tool_definitions()}
    assert "active region" not in by["evaluate"] and "GP" not in by["incumbent"]
    assert "looking at the trial log" in s.brief()


def test_policy_toolset_overrides_config():
    class LLMOnlyStub(ScriptedPolicy):
        toolset = "llm_only"

        def step(self, session):
            assert [t.name for t in session.tools] == ["evaluate", "incumbent", "record_note", "status", "trials"]
            session.call("evaluate", {"config": {"x0": 0.5, "x1": 0.5}, "rationale": "r"})

    camp = Campaign(quad_problem(), LLMOnlyStub(), CampaignConfig(budget=2))
    res = camp.run()
    assert camp.config.toolset == "llm_only" and res.campaign_config["toolset"] == "llm_only"


def test_log_reasoning_is_excluded_from_action_counts():
    class Thinker(ScriptedPolicy):
        def step(self, session):
            session.log_reasoning("I think x0 matters", kind="thinking", tokens=12)
            super().step(session)

    res = Campaign(quad_problem(), Thinker(points=[{"x0": 0.1, "x1": 0.1}, {"x0": 0.2, "x1": 0.2}]),
                   CampaignConfig(budget=2)).run()
    traces = [e for e in res.log if e["action_class"] == "reasoning"]
    assert len(traces) == 2 and traces[0]["text"] == "I think x0 matters" and traces[0]["tokens"] == 12
    assert res.action_counts == {"evaluate": 2}
    assert res.action_class_counts == {"evaluate": 2}


# --------------------------------------------------------------------------- campaign loop & errors


def test_policy_error_mid_run_switches_to_fallbacks():
    policy = ScriptedPolicy(fail_at=3)
    res = Campaign(get_problem("branin"), policy, CampaignConfig(budget=8, seed=2)).run()
    assert res.n_evals == 8 and res.policy_error == "PolicyError: model refused"
    assert [r["forced"] for r in res.records] == [False] * 3 + [True] * 5
    assert res.forced_commits == 5 and policy.steps == [0, 1, 2, 3]   # never called again after failing
    assert policy.finished and all("policy failed" in r["rationale"] for r in res.records[3:])
    assert res.usage == {"api_calls": 0}


def test_policy_error_in_start_and_raise_mode():
    res = Campaign(quad_problem(), ScriptedPolicy(start_error=True), CampaignConfig(budget=3)).run()
    assert res.forced_commits == 3 and res.policy_error == "PolicyError: no API key"
    with pytest.raises(PolicyError, match="model refused"):
        Campaign(quad_problem(), ScriptedPolicy(fail_at=1),
                 CampaignConfig(budget=3, on_policy_error="raise")).run()


def test_policy_that_never_evaluates_is_carried_by_fallbacks():
    res = Campaign(quad_problem(), ScriptedPolicy(skip=True), CampaignConfig(budget=4)).run()
    assert res.n_evals == 4 and res.forced_commits == 4
    assert res.actions_per_step == [1, 1, 1, 1] and res.action_counts == {"status": 4}
    assert all(r["source"] == "harness-fallback" for r in res.records)


def test_non_policy_exceptions_are_recorded_and_fallbacks_finish_the_campaign():
    # (formerly test_non_policy_errors_propagate, which encoded the crash this fixes: any
    # exception escaping a policy is a policy failure; the campaign keeps its data and finishes)
    class Buggy(ScriptedPolicy):
        def step(self, session):
            if session.t >= 2:
                self.steps.append(session.t)
                raise TypeError("bug in policy")
            super().step(session)

    policy = Buggy()
    res = Campaign(quad_problem(), policy, CampaignConfig(budget=5)).run()
    assert res.n_evals == 5 and res.policy_error == "TypeError: bug in policy"
    assert [r["forced"] for r in res.records] == [False, False, True, True, True]
    assert policy.steps == [0, 1, 2] and policy.finished
    with pytest.raises(TypeError, match="bug in policy"):
        Campaign(quad_problem(), Buggy(), CampaignConfig(budget=5, on_policy_error="raise")).run()


@pytest.mark.parametrize("hook", ["start", "on_context", "finish"])
def test_exceptions_from_every_policy_hook_are_recorded(hook):
    class Hooky(ScriptedPolicy):
        def start(self, session):
            super().start(session)
            if hook == "start":
                raise ValueError("start broke")

        def on_context(self, session, item):
            super().on_context(session, item)
            if hook == "on_context":
                raise ValueError("on_context broke")

        def finish(self, session):
            super().finish(session)
            if hook == "finish":
                raise ValueError("finish broke")

    res = Campaign(quad_problem(), Hooky(), CampaignConfig(budget=3, context_schedule={1: "note"})).run()
    assert res.n_evals == 3 and res.policy_error == f"ValueError: {hook} broke"


def test_base_exceptions_still_propagate_through_the_campaign():
    class Abort(BaseException):     # like api._Abort or KeyboardInterrupt: must not be swallowed
        pass

    class Interrupted(ScriptedPolicy):
        def step(self, session):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        Campaign(quad_problem(), Interrupted(), CampaignConfig(budget=3)).run()

    def fn(cfg):
        raise Abort("user function bug")

    space = SearchSpace([RealParam("x0", 0, 1), RealParam("x1", 0, 1)])
    p = Problem("abort", space, fn, "y", False, 0.0, default_budget=3)
    with pytest.raises(Abort):
        Campaign(p, SobolPolicy(), CampaignConfig(budget=3)).run()


def test_campaign_accepts_names():
    camp = Campaign("branin", "sobol", CampaignConfig(budget=3))
    res = camp.run()
    assert res.problem == "branin" and res.policy == "sobol" and res.n_evals == 3
    assert camp.session is not None and camp.session.lenz.n_observed == 3
    with pytest.raises(LenzError):
        Campaign("branin", object())


# --------------------------------------------------------------------------- results


def test_regret_curves_match_metrics_module():
    res = Campaign(get_problem("branin"), SobolPolicy(), CampaignConfig(budget=12, seed=5)).run()
    u = [r["utility"] for r in res.records]
    p = get_problem("branin")
    np.testing.assert_allclose(res.simple_regret, simple_regret_curve(u, p.u_star))
    np.testing.assert_allclose(res.cumulative_regret, cumulative_regret_curve(u, p.u_star))
    assert np.all(np.diff(res.simple_regret) <= 1e-12) and min(res.simple_regret) >= -1e-9
    best = min(res.records, key=lambda r: r["regret"])
    assert res.best_value == pytest.approx(best["true_metrics"]["y"]) and res.best_config == best["config"]
    assert res.final_simple_regret == res.simple_regret[-1]


def test_constrained_campaign_uses_feasibility():
    p = get_problem("constrained_hartmann6")
    res = Campaign(p, SobolPolicy(), CampaignConfig(budget=10, seed=0)).run()
    feas = [r["feasible"] for r in res.records]
    assert feas == [p.is_feasible(r["true_metrics"]) for r in res.records]
    u = [r["utility"] for r in res.records]
    np.testing.assert_allclose(res.simple_regret, simple_regret_curve(u, p.u_star, feas))
    if any(feas):
        best = max((r for r in res.records if r["feasible"]), key=lambda r: r["utility"])
        assert res.best_value == pytest.approx(best["true_metrics"]["y"])


def test_unknown_optimum_gives_nan_curves_and_round_trips(tmp_path):
    res = Campaign(quad_problem(f_star=None), SobolPolicy(), CampaignConfig(budget=3)).run()
    assert all(math.isnan(v) for v in res.simple_regret + res.cumulative_regret)
    path = tmp_path / "r.json"
    res.save(path)
    back = CampaignResult.load(path)
    assert all(math.isnan(v) for v in back.simple_regret) and back.best_value == res.best_value


def test_campaign_result_save_load_round_trip(tmp_path):
    cfg = CampaignConfig(budget=6, seed=1, context_schedule={2: "hello"}, allow_early_stop=True)
    res = Campaign(get_problem("branin", noise_std=0.5), FixedBOPolicy(n_init=3), cfg).run()
    path = tmp_path / "sub" / "result.json"
    res.save(path)
    text = path.read_text()
    json.loads(text)
    assert "NaN" not in text and "Infinity" not in text
    back = CampaignResult.load(path)
    assert back.to_dict() == json.loads(dumps(res.to_dict()))
    assert back.simple_regret == pytest.approx(res.simple_regret) and back.records == res.records
    assert back.campaign_config["context_schedule"] == {"2": [{"source": None, "text": "hello", "data": None}]}
    assert CampaignConfig.from_dict(back.campaign_config) == cfg
    assert back.problem_info["name"] == "branin" and back.u_star == pytest.approx(-0.397887, rel=1e-5)
    with pytest.raises(LenzError, match="missing fields"):
        CampaignResult.from_dict({"problem": "x"})


def test_records_true_vs_observed_under_noise():
    p = get_problem("branin", noise_std=1.0)
    res = Campaign(p, SobolPolicy(), CampaignConfig(budget=6, seed=0)).run()
    for r in res.records:
        assert r["true_metrics"]["y"] == pytest.approx(p.evaluate(r["config"])["y"])
        assert r["regret"] == pytest.approx(p.regret(r["true_metrics"]))
        assert abs(r["observed"]["y"] - r["true_metrics"]["y"]) > 0


def test_moo_campaign_hypervolume_regret():
    p = get_problem("branin_currin")
    res = Campaign(p, FixedBOPolicy(n_init=5), CampaignConfig(budget=14, seed=1)).run()
    hv = np.asarray(res.simple_regret)
    assert res.is_moo and len(hv) == 14
    assert np.all(np.diff(hv) <= 1e-9) and np.all(hv >= -1e-9) and np.all(hv <= p.max_hv + 1e-9)
    assert hv[-1] < p.max_hv                     # some hypervolume found
    np.testing.assert_allclose(res.cumulative_regret, np.cumsum(hv))
    assert res.best_value == pytest.approx(p.max_hv - hv[-1], rel=1e-9)
    assert res.best_config is None
    sources = [e["result"]["acqf"] for e in res.log if e["tool"] == "suggest"]
    assert sources[:5] == ["sobol"] * 5 and set(sources[5:]) == {"ehvi"}


def test_moo_early_stop_padding_of_hypervolume():
    p = get_problem("branin_currin")
    pts = [{"x0": 0.5, "x1": 0.1}, {"x0": 0.2, "x1": 0.9}]
    res = Campaign(p, ScriptedPolicy(points=pts, stop_at=2),
                   CampaignConfig(budget=5, allow_early_stop=True, regret="cumulative")).run()
    assert len(res.simple_regret) == 5 and res.simple_regret[2:] == [res.simple_regret[1]] * 3
    np.testing.assert_allclose(res.cumulative_regret, np.cumsum(res.simple_regret))


def test_campaign_is_deterministic():
    def run():
        return Campaign(get_problem("hartmann3"), FixedBOPolicy(n_init=3),
                        CampaignConfig(budget=6, seed=7)).run()

    a, b = run(), run()
    assert [r["config"] for r in a.records] == [r["config"] for r in b.records]
    assert a.simple_regret == b.simple_regret


def test_valid_schema_examples_validate_against_generated_schemas():
    s = seeded_session(2, allow_early_stop=True)
    specs = {t.name: t for t in s.tools}
    for name, args in VALID_CALLS:
        assert validate(args, specs[name].input_schema) == [], (name, args)


# --------------------------------------------------------------------------- review regressions


def test_noisy_evaluation_calls_the_black_box_once_with_problem_observe_noise():
    from agentic_bo.utils import make_rng

    calls: list[dict] = []

    def fn(cfg: dict) -> dict:
        calls.append(dict(cfg))
        return {"y": -((cfg["x0"] - 0.3) ** 2 + (cfg["x1"] - 0.6) ** 2), "cost": cfg["x0"]}

    space = SearchSpace([RealParam("x0", 0, 1), RealParam("x1", 0, 1)])
    p = Problem("quad", space, fn, "y", False, 0.0, noise_std=0.1, default_budget=5)
    res = Campaign(p, SobolPolicy(), CampaignConfig(budget=5, seed=3)).run()
    assert len(calls) == res.n_evals == 5               # one black-box call per evaluation
    rng = make_rng(3, 0x5E55)                            # the session's noise stream
    for r in res.records:                                # Problem.observe semantics on that one call
        eps = rng.standard_normal(2) * 0.1
        assert r["observed"]["y"] == pytest.approx(r["true_metrics"]["y"] + eps[0], abs=1e-12)
        assert r["observed"]["cost"] == pytest.approx(r["true_metrics"]["cost"] + eps[1], abs=1e-12)
    # same noise as Problem.observe with the same stream (results stay reproducible)
    rng = make_rng(3, 0x5E55)
    for r in res.records:
        assert p.observe(r["config"], rng) == pytest.approx(r["observed"], abs=1e-12)


def test_noisy_observation_is_the_measured_value_plus_noise_for_a_stochastic_black_box():
    from agentic_bo.utils import make_rng

    gen = np.random.default_rng(0)

    def fn(cfg: dict) -> dict:
        return {"y": float(gen.normal())}     # every call differs: two calls would decouple true/observed

    space = SearchSpace([RealParam("x0", 0, 1)])
    p = Problem("stoch", space, fn, "y", False, None, noise_std=0.01, default_budget=4)
    res = Campaign(p, SobolPolicy(), CampaignConfig(budget=4, seed=1)).run()
    rng = make_rng(1, 0x5E55)
    for r in res.records:
        eps = float(rng.standard_normal(1)[0]) * 0.01
        assert r["observed"]["y"] - r["true_metrics"]["y"] == pytest.approx(eps, abs=1e-12)


def test_suggest_takes_acquisition_parameters_for_one_call():
    s = seeded_session(4)
    env = s.call("suggest", {"acqf": "ucb", "beta": 25.0})
    assert env["ok"], env
    assert env["result"]["acqf"] == "ucb" and env["result"]["acqf_params"] == {"beta": 25.0}
    clone = Lenz.from_dict(json.loads(dumps(s.lenz.to_dict())))
    env = s.call("suggest", {"acqf": "ts", "params": {"n_candidates": 64}})
    assert env["ok"] and env["result"]["acqf_params"] == {"n_candidates": 64}
    raw = clone.suggest(acqf="ts", acqf_params={"n_candidates": 64})["candidates"][0]["config"]
    assert s.lenz.space.distance(raw, env["result"]["candidates"][0]["config"]) <= DUPLICATE_TOL
    assert s.lenz.shelf.acqf == "logei" and s.lenz.shelf.acqf_params == {}   # nothing reconfigured
    bad = s.call("suggest", {"beta": 2.0})                                   # logei takes no beta
    assert bad["ok"] is False and "beta" in bad["error"]
    assert "beta" in next(t for t in s.tools if t.name == "suggest").description


@pytest.mark.parametrize("bad_call", [("no_such_tool", {}), ("status", "not an object"),
                                      ("evaluate", {"config": {"x0": 100.0, "x1": 0.0}, "rationale": "r"})])
def test_hard_limit_also_stops_invalid_calls(bad_call):
    s = seeded_session(3, budget=8, soft_actions_per_step=2, hard_actions_per_step=3)
    name, args = bad_call
    for _ in range(3):
        assert s.call(name, args)["ok"] is False
        assert s.t == 3
    env = s.call(name, args)                       # action 4 > hard limit 3
    assert env["ok"] is False and env.get("forced") is True, env
    assert "harness evaluated a fallback point" in env["error"]
    assert s.t == 4 and s.step_committed and s.records[-1]["forced"]


def test_valid_evaluate_beyond_the_hard_limit_is_still_honoured():
    s = seeded_session(3, budget=8, soft_actions_per_step=2, hard_actions_per_step=3)
    for _ in range(3):
        s.call("status")
    env = s.call("evaluate", {"config": {"x0": 5.0, "x1": 5.0}, "rationale": "late but mine"})
    assert env["ok"] and not s.records[-1]["forced"]


def test_stop_after_the_budget_is_used_is_rejected():
    s = branin_session(2, allow_early_stop=True)
    evaluate(s, {"x0": 0.0, "x1": 0.0})
    evaluate(s, {"x0": 1.0, "x1": 0.0})
    env = s.call("stop", {"reason": "done"})
    assert env["ok"] is False and "budget" in env["error"] and not s.stopped
    res = CampaignResult.from_session(s, "p")
    assert res.stopped_early is False and res.stop_reason is None


def test_stop_during_on_context_ends_the_campaign_before_the_next_step():
    class StopOnContext(ScriptedPolicy):
        def on_context(self, session, item):
            super().on_context(session, item)
            assert session.call("stop", {"reason": "instructed to stop"})["ok"]

    policy = StopOnContext()
    res = Campaign(get_problem("branin"), policy,
                   CampaignConfig(budget=5, allow_early_stop=True, context_schedule={2: "stop now"})).run()
    assert res.stopped_early and res.n_evals == 2 and policy.steps == [0, 1]
    assert not any(e["tool"] == "evaluate" and not e["ok"] for e in res.log)


def test_ordinal_categorical_and_integer_values_are_shown_exactly_so_they_can_be_resubmitted():
    from agentic_bo.space import CategoricalParam, IntParam, OrdinalParam

    lrs = [float(v) for v in np.logspace(-4, -1, 7)]          # 0.00031622776601683794, ...
    space = SearchSpace([RealParam("x", 0, 1), OrdinalParam("lr", lrs),
                         CategoricalParam("c", [0.123456789, 1.5]), IntParam("n", 0, 10 ** 9)])
    p = Problem("grid", space, lambda c: {"y": -(math.log10(c["lr"]) + 2.5) ** 2 - (c["x"] - 0.5) ** 2},
                "y", False, None, default_budget=6)
    s = AgentSession(p, CampaignConfig(budget=6, seed=0))
    for _ in range(3):
        s.begin_step()
        cfg = s.call("suggest", {"acqf": "sobol"})["result"]["candidates"][0]["config"]
        assert cfg["lr"] in lrs and cfg["c"] in (0.123456789, 1.5)
        env = s.call("evaluate", {"config": cfg, "rationale": "accept the suggestion as shown"})
        assert env["ok"], env
    row = s.call("trials", {"last": 1})["result"]["trials"][0]["config"]
    assert row["lr"] in lrs and s.call("predict", {"configs": [row]})["ok"]
    shown = compact({"lr": lrs[1], "c": 0.123456789, "n": 123456789.0, "x": 0.5}, space)
    assert shown == {"lr": lrs[1], "c": 0.123456789, "n": 123456789.0, "x": 0.5}
    # numbers keyed by a parameter name that are not values of it (e.g. lengthscales) are still rounded
    assert compact({"lengthscales": {"lr": 0.123456789, "n": 0.123456789}}, space) == \
        {"lengthscales": {"lr": 0.123457, "n": 0.123457}}
    assert round_param_value(space.params[1], lrs[1]) == lrs[1]


def test_stop_beyond_the_hard_limit_is_honoured_not_replaced_by_an_evaluation():
    s = seeded_session(3, budget=8, soft_actions_per_step=2, hard_actions_per_step=3, allow_early_stop=True)
    for _ in range(3):
        s.call("status")
    env = s.call("stop", {"reason": "converged"})          # action 4 > hard limit 3: a commit-like action
    assert env["ok"] and s.stopped and s.t == 3 and not any(r["forced"] for r in s.records)


def test_score_and_predict_descriptions_match_the_default_for_non_pointwise_acqfs():
    s = seeded_session(4)
    assert s.call("set_acqf", {"acqf": "ts"})["ok"]
    cfg = {"x0": 1.0, "x1": 2.0}
    assert s.call("score", {"configs": [cfg]})["result"]["acqfs"] == ["logei"]       # not "the current one"
    assert s.call("predict", {"configs": [cfg]})["result"]["acqf"] == "logei"
    desc = {t.name: t.description for t in s.tools}
    for name in ("score", "predict"):
        assert "ts, sobol or random" in desc[name] and "logei" in desc[name], name
