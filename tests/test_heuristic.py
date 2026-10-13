"""Tests for HeuristicSaraPolicy, the scripted agentic BO policy (DESIGN.md §9.3)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from agentic_bo.problems import Problem, get_problem
from agentic_bo.sara import Campaign, CampaignConfig
from agentic_bo.sara.policies import FixedBOPolicy, get_policy
from agentic_bo.sara.policies.heuristic import (
    INSTRUCTION_OPS, HeuristicSaraPolicy, declared_noise_std, space_from_tools,
)
from agentic_bo.sara.session import AgentSession
from agentic_bo.space import CategoricalParam, IntParam, OrdinalParam, RealParam, SearchSpace
from agentic_bo.utils import LenzError


def run(problem: Problem | str, budget: int = 20, seed: int = 0, policy: HeuristicSaraPolicy | None = None,
        **cfg) -> Campaign:
    if isinstance(problem, str):
        problem = get_problem(problem, seed=seed, randomize=True)
    camp = Campaign(problem, policy or HeuristicSaraPolicy(strict=True),
                    CampaignConfig(budget=budget, seed=seed, **cfg))
    camp.run()
    return camp


def calls(camp: Campaign, tool: str) -> list[dict]:
    return [e for e in camp.session.log if e.get("tool") == tool and not e.get("forced")]


def quadratic_problem(center=(0.8, 0.8), **kw) -> Problem:
    """Smooth 2-D bowl on [0, 1]^2 (maximize), optimum at ``center``."""
    space = SearchSpace([RealParam("a", 0.0, 1.0), RealParam("b", 0.0, 1.0)])

    def f(cfg):
        return {"y": 1.0 - (cfg["a"] - center[0]) ** 2 - (cfg["b"] - center[1]) ** 2}

    return Problem(name="bowl", space=space, evaluate_fn=f, objective="y", minimize=False, f_star=1.0,
                   default_budget=20, x_star={"a": center[0], "b": center[1]}, **kw)


def mixed_problem() -> Problem:
    """Mixed real / log-real / int / ordinal / categorical toy problem (minimize)."""
    space = SearchSpace([
        RealParam("lr", 1e-4, 1e-1, log_scale=True),
        RealParam("mom", 0.0, 1.0),
        IntParam("layers", 1, 8),
        OrdinalParam("width", [16, 32, 64, 128]),
        CategoricalParam("opt", ["adam", "sgd", "rmsprop"]),
    ])
    bonus = {"adam": 0.0, "sgd": 0.5, "rmsprop": 0.25}

    def f(cfg):
        loss = ((math.log10(cfg["lr"]) + 2.5) ** 2 + (cfg["mom"] - 0.9) ** 2 + 0.1 * (cfg["layers"] - 4) ** 2
                + 0.2 * abs(math.log2(cfg["width"]) - 6) + bonus[cfg["opt"]])
        return {"loss": loss}

    return Problem(name="mixed_toy", space=space, evaluate_fn=f, objective="loss", minimize=True, f_star=0.0,
                   default_budget=20)


def assert_clean(camp: Campaign, soft: int = 12) -> None:
    """Every commit came from the policy, with a rationale, and steps stayed under the soft limit."""
    res = camp.result
    assert res.n_evals == res.budget
    assert res.forced_commits == 0 and res.policy_error is None
    assert all(r["source"] == "agent" and r["rationale"].strip() for r in res.records)
    assert max(res.actions_per_step) < soft
    assert not [e for e in camp.session.log if e.get("action_class") == "reasoning" and e.get("kind") == "error"]
    assert not [e for e in camp.session.log if e.get("tool") and not e["ok"]]   # no failed tool call
    assert not [r for r in res.records if r["duplicate_of"] is not None]
    # never beyond the soft action limit: no tool result carries the harness warning
    assert not [e for e in camp.session.log if isinstance(e.get("result"), dict) and "warning" in e["result"]]


# --------------------------------------------------------------------------- construction


def test_get_policy_and_knob_validation():
    p = get_policy("heuristic", radius_init=0.3, n_init=4)
    assert isinstance(p, HeuristicSaraPolicy) and p.name == "heuristic" and p.toolset == "full"
    assert p.radius_init == 0.3 and p.n_init == 4
    bad = [dict(n_init=0), dict(max_init=1.5), dict(success_tol=0), dict(radius_min=0.5, radius_init=0.4),
           dict(radius_max=1.5), dict(endgame_frac=1.0), dict(global_ei_ratio=0.5), dict(explore_beta=-1),
           dict(edge_frac=0.6), dict(cumulative_radius=0), dict(n_init=True), dict(global_cooldown=-1),
           dict(global_ref="far"), dict(distrust_explore="random"), dict(distrust_explore_every=-1),
           dict(distrust_patience=-2)]
    for kw in bad:
        with pytest.raises(LenzError):
            HeuristicSaraPolicy(**kw)
    assert "set_constraints" in INSTRUCTION_OPS and "evaluate" in INSTRUCTION_OPS


def test_space_from_tools_matches_problem_space():
    for problem in (mixed_problem(), get_problem("reaction_yield"), get_problem("hartmann6", seed=1, randomize=True)):
        session = AgentSession(problem, CampaignConfig(budget=5))
        assert space_from_tools(session) == problem.space


def test_declared_noise_std_reads_the_brief():
    assert declared_noise_std(get_problem("branin", seed=0, randomize=True, noise_std=0.5).brief()) == 0.5
    assert declared_noise_std(get_problem("hartmann6", noise_std=0.002).brief()) == 0.002
    assert declared_noise_std(get_problem("branin").brief()) == 0.0
    assert declared_noise_std("") == 0.0
    assert declared_noise_std("Noise: standard deviation 1e-2 per metric.") == 0.01


# --------------------------------------------------------------------------- end-to-end runs


@pytest.mark.parametrize("name,budget", [("branin", 25), ("hartmann6", 20)])
def test_runs_cleanly_and_uses_agentic_tools(name, budget):
    camp = run(name, budget=budget, seed=1)
    assert_clean(camp)
    counts = camp.result.action_counts
    for tool in ("diagnostics", "suggest", "evaluate", "record_note", "status"):
        assert counts.get(tool, 0) > 0, tool
    assert counts["evaluate"] == budget
    # model phase: a diagnostics probe and a suggest proposal precede most commits
    assert counts["diagnostics"] >= budget // 2 and counts["suggest"] >= budget
    assert camp.result.action_class_counts.get("probe", 0) >= budget // 2
    # rationales and expectations say what kind of step it was and what the surrogate predicts
    texts = [r["rationale"] for r in camp.result.records]
    assert any("Sobol" in t for t in texts) and any("trust-region" in t or "end game" in t for t in texts)
    assert sum(r["expectation"] is not None for r in camp.result.records) >= budget // 2
    # periodic belief summaries
    assert len(camp.result.notebook) >= 3
    assert any(n["text"].startswith("Summary") for n in camp.result.notebook)


def test_global_and_local_candidates_are_compared_with_score():
    camp = run(quadratic_problem(), budget=16, seed=0)
    assert_clean(camp)
    scores = calls(camp, "score")
    assert scores, "a trusted surrogate should trigger global-vs-local comparisons"
    for e in scores:
        assert 2 <= len(e["args"]["configs"]) <= 3 and "pm" in e["args"]["acqfs"]
    local = [e for e in calls(camp, "suggest") if e["args"].get("around") is not None]
    glob = [e for e in calls(camp, "suggest") if not e["args"]]
    assert local and glob
    assert any("score: local" in r["rationale"] or "global candidate" in r["rationale"] for r in camp.result.records)


def test_mixed_int_categorical_problem():
    camp = run(mixed_problem(), budget=20, seed=0)
    assert_clean(camp)
    space = camp.problem.space
    for r in camp.result.records:
        assert space.validate(r["config"]) == r["config"]
    simple = camp.result.simple_regret
    assert simple[-1] < simple[4]          # the model phase improved on the opening


def test_exhausts_a_small_discrete_space_without_duplicates():
    space = SearchSpace([IntParam("i", 1, 3), IntParam("j", 1, 3)])
    grid = Problem(name="grid", space=space, objective="y", minimize=True, f_star=0.0,
                   evaluate_fn=lambda c: {"y": (c["i"] - 2) ** 2 + 0.5 * (c["j"] - 3) ** 2})
    camp = run(grid, budget=9, seed=0)
    assert_clean(camp)
    assert len({(r["config"]["i"], r["config"]["j"]) for r in camp.result.records}) == 9


def test_beats_random_opening_on_branin():
    camp = run("branin", budget=30, seed=3)
    assert camp.result.final_simple_regret < 0.5


def test_deterministic_given_seed():
    a = run("branin", budget=15, seed=7).result
    b = run("branin", budget=15, seed=7).result
    assert [r["config"] for r in a.records] == [r["config"] for r in b.records]
    assert [r["rationale"] for r in a.records] == [r["rationale"] for r in b.records]
    assert a.notebook == b.notebook
    c = run("branin", budget=15, seed=8).result
    assert [r["config"] for r in a.records] != [r["config"] for r in c.records]


def test_policy_instance_is_reusable():
    policy = HeuristicSaraPolicy(strict=True)
    a = run("branin", budget=12, seed=2, policy=policy).result
    b = run("branin", budget=12, seed=2, policy=policy).result
    assert [r["config"] for r in a.records] == [r["config"] for r in b.records]


# --------------------------------------------------------------------------- opening and priors


def test_opening_lengths_simple_vs_cumulative():
    simple = run("hartmann6", budget=24, seed=0)
    cumul = run("hartmann6", budget=24, seed=0, regret="cumulative")
    n_sobol = lambda c: sum("Sobol" in r["rationale"] for r in c.result.records)  # noqa: E731
    assert n_sobol(cumul) < n_sobol(simple)
    assert n_sobol(cumul) <= 4 and n_sobol(simple) >= 4
    # cumulative mode switches to UCB with a small beta; simple mode keeps logEI
    acqf_calls = calls(cumul, "set_acqf")
    assert acqf_calls and acqf_calls[0]["args"] == {"acqf": "ucb", "beta": 1.0}
    assert cumul.session.lenz.shelf.acqf == "ucb"
    assert simple.session.lenz.shelf.acqf == "logei"
    # cumulative regret is lower in cumulative mode (summed over a few seeds: single seeds can tie)
    regret = {mode: sum(run("hartmann6", budget=24, seed=s, regret=mode).result.final_cumulative_regret
                        for s in range(3)) for mode in ("simple", "cumulative")}
    assert regret["cumulative"] < regret["simple"]


def test_prior_config_is_evaluated_first():
    base = quadratic_problem()
    prior_cfg = {"a": 0.7, "b": 0.75}
    problem = dataclasses.replace(base, prior="Experience says a=0.7, b=0.75 works well.",
                                  prior_hint={"config": prior_cfg})
    camp = run(problem, budget=10)
    assert_clean(camp)
    first = camp.result.records[0]
    assert first["config"] == prior_cfg and "prior" in first["rationale"]
    assert first["t"] == 1


def test_prior_region_is_used_then_widened_when_optimum_is_outside():
    base = quadratic_problem(center=(0.85, 0.85))
    region = {"a": [0.2, 0.6], "b": [0.2, 0.6]}
    problem = dataclasses.replace(base, prior="The good region is a, b in [0.2, 0.6].",
                                  prior_hint={"region": region})
    camp = run(problem, budget=25, policy=HeuristicSaraPolicy(strict=True, region_check_every=2))
    assert_clean(camp)
    set_bounds = calls(camp, "set_bounds")
    assert set_bounds[0]["args"] == {"bounds": region} and set_bounds[0]["step"] == 1
    n_open = camp.policy._st.n_init
    for r in camp.result.records[:n_open]:     # the opening fills the prior region
        assert 0.2 <= r["config"]["a"] <= 0.6 and 0.2 <= r["config"]["b"] <= 0.6
    # the incumbent sits at the region's upper faces, so the region gets widened
    assert len(set_bounds) >= 2
    final = camp.session.lenz.shelf.bounds
    assert final is None or (final["a"][1] > 0.6 and final["b"][1] > 0.6)
    assert any("Widened" in n["text"] or "widening" in n["text"] for n in camp.result.notebook)
    best = max(r["true_metrics"]["y"] for r in camp.result.records)
    assert best > 1.0 - 2 * 0.25 ** 2 + 1e-3    # better than the best point inside the prior region


def test_prior_region_on_reaction_yield():
    camp = run(get_problem("reaction_yield"), budget=12, seed=0)
    assert_clean(camp)
    region = get_problem("reaction_yield").prior_hint["region"]
    assert calls(camp, "set_bounds")[0]["args"]["bounds"] == region
    space = camp.problem.space
    assert space.contains(camp.result.records[0]["config"], region)


# --------------------------------------------------------------------------- context


def test_structured_instruction_adds_constraint_mid_run():
    instr = {"text": "From now on keep c below 0.6.", "op": "set_constraints",
             "args": {"constraints": [{"metric": "c", "upper": 0.6}]}}
    camp = run("constrained_hartmann6", budget=20, seed=0, context_schedule={8: instr})
    assert_clean(camp)
    applied = calls(camp, "set_constraints")
    assert len(applied) == 1 and applied[0]["ok"] and applied[0]["t"] == 8
    assert applied[0]["args"] == instr["args"]
    assert camp.session.lenz.shelf.constraints == [{"metric": "c", "lower": None, "upper": 0.6}]
    assert any("Instruction applied" in n["text"] and "set_constraints" in n["text"] for n in camp.result.notebook)
    # after the instruction the policy steers towards the tighter feasible set
    later = [r for r in camp.result.records if r["t"] > 8]
    assert sum(r["observed"]["c"] <= 0.6 for r in later) > len(later) // 2
    assert camp.session.lenz.incumbent()["incumbent"]["metrics"]["c"] <= 0.6
    assert any("P(feasible)" in (r["expectation"] or "") for r in later)


def test_instructions_for_bounds_acqf_objectives_and_targeted_evaluation():
    target = {"a": 0.25, "b": 0.75}
    schedule = {
        3: [{"text": "Use UCB with beta 2 from now on.", "op": "set_acqf", "args": {"acqf": "ucb", "beta": 2.0}},
            {"text": "Please test a=0.25, b=0.75.", "op": "evaluate", "args": {"config": target}}],
        6: {"text": "Only search a in [0.5, 1].", "op": "set_bounds", "args": {"bounds": {"a": [0.5, 1.0]}}},
        9: "The lab manager says the results look promising.",
        10: {"text": "unknown", "op": "launch_rocket", "args": {}},
    }
    camp = run(quadratic_problem(), budget=16, seed=1, context_schedule=schedule)
    assert_clean(camp)
    shelf = camp.session.lenz.shelf
    assert shelf.acqf == "ucb" and shelf.acqf_params == {"beta": 2.0}          # kept (locked by instruction)
    assert shelf.bounds == {"a": [0.5, 1.0]}                                    # instruction bounds are kept
    rec = camp.result.records[3]
    assert rec["config"] == target and "instructed" in rec["rationale"]
    assert rec["expectation"] is not None and "GP posterior" in rec["expectation"]   # predicted first
    assert [e["args"]["configs"] for e in calls(camp, "predict")] == [[target]]
    texts = [n["text"] for n in camp.result.notebook]
    assert any("lab manager" in t for t in texts)
    assert any("launch_rocket" in t and "not applied" in t for t in texts)
    # after t=6 every suggestion-driven point respects the instructed region
    assert all(r["config"]["a"] >= 0.5 - 1e-9 for r in camp.result.records[7:])


def test_set_objectives_instruction_switches_metric():
    space = SearchSpace([RealParam("a", 0.0, 1.0)])
    problem = Problem(name="two", space=space, evaluate_fn=lambda c: {"y": c["a"], "z": (c["a"] - 0.3) ** 2},
                      objective="y", minimize=False, f_star=1.0, outputs={"y": "", "z": ""})
    instr = {"text": "Minimize z instead.", "op": "set_objectives",
             "args": {"objectives": [{"metric": "z", "minimize": True}]}}
    camp = run(problem, budget=14, seed=0, context_schedule={6: instr})
    assert_clean(camp)
    assert camp.session.lenz.shelf.objectives[0]["metric"] == "z"
    assert camp.policy._st.metric == "z" and camp.policy._st.minimize
    late = [r["config"]["a"] for r in camp.result.records[-4:]]
    assert min(abs(a - 0.3) for a in late) < 0.1


# --------------------------------------------------------------------------- adaptation


def test_stagnation_triggers_exploration_step():
    space = SearchSpace([RealParam("a", 0.0, 1.0), RealParam("b", 0.0, 1.0)])
    flat = Problem(name="flat", space=space, evaluate_fn=lambda c: {"y": 1.0}, objective="y", minimize=False,
                   f_star=1.0)
    camp = run(flat, budget=16, policy=HeuristicSaraPolicy(strict=True, stagnation_steps=5, endgame_frac=0.0))
    res = camp.result
    assert res.forced_commits == 0
    assert any("stagnation" in r["rationale"] for r in res.records)
    acqf = [e["args"] for e in calls(camp, "set_acqf")]
    assert {"acqf": "ucb", "beta": 9.0} in acqf and acqf[-1]["acqf"] == "logei"   # switched, then restored
    assert camp.session.lenz.shelf.acqf == "logei"


def test_unpredictive_surrogate_triggers_exploration():
    space = SearchSpace([RealParam("a", 0.0, 1.0), RealParam("b", 0.0, 1.0)])
    # a hash of the inputs: no surrogate can predict it, not even locally
    wiggly = Problem(name="hash", space=space, f_star=1.0, objective="y", minimize=False,
                     evaluate_fn=lambda c: {"y": math.modf(abs(math.sin(12.9898 * c["a"] + 78.233 * c["b"]))
                                                           * 43758.5453)[0]})
    camp = run(wiggly, budget=24, seed=0)
    assert_clean(camp)
    explore = [r for r in camp.result.records if r["rationale"].startswith("distrust")]
    assert explore and all("not been predictive" in r["rationale"] for r in explore)
    # the opening was extended first, up to max_init
    assert sum("opening extended" in r["rationale"] for r in camp.result.records) > 0
    # cumulative mode never explores because of distrust
    cumul = run(wiggly, budget=24, seed=0, regret="cumulative")
    assert not [r for r in cumul.result.records if r["rationale"].startswith("distrust")]
    # and the behaviour can be switched off
    off = run(wiggly, budget=24, seed=0, policy=HeuristicSaraPolicy(strict=True, distrust_explore_every=0))
    assert not [r for r in off.result.records if r["rationale"].startswith("distrust")]


def test_radius_collapse_triggers_restart():
    policy = HeuristicSaraPolicy(strict=True, radius_init=0.05, radius_min=0.03, failure_tol=1,
                                 stagnation_steps=1000, endgame_frac=0.0)
    camp = run("hartmann6", budget=22, seed=0, policy=policy)
    assert_clean(camp)
    assert any("restart" in r["rationale"] for r in camp.result.records)
    assert camp.policy._st.restarts >= 1


def test_trust_region_radius_adapts():
    camp = run("branin", budget=30, seed=0)
    radii = set()
    for e in calls(camp, "suggest"):
        if e["args"].get("around") is not None:
            radii.add(e["args"]["radius"])
    assert len(radii) >= 2          # successes / failures changed the radius


def test_endgame_exploits_near_incumbent():
    camp = run("hartmann6", budget=30, seed=2)
    last = camp.result.records[-3:]
    assert all("end game" in r["rationale"] for r in last)
    local = [e for e in calls(camp, "suggest") if e["t"] >= 27]
    assert local and all(e["args"].get("around") == "incumbent" and e["args"]["radius"] <= 0.1 for e in local)


def test_multi_objective_problem_runs():
    camp = run("branin_currin", budget=14, seed=0)
    assert_clean(camp)
    assert camp.result.is_moo and camp.result.best_value > 0


def test_noisy_problem_switches_to_noisy_logei():
    problem = dataclasses.replace(quadratic_problem(), noise_std=0.2)
    camp = run(problem, budget=16, seed=0)
    assert_clean(camp)
    assert camp.session.lenz.shelf.acqf == "noisy_logei"
    assert any("noisy" in n["text"] for n in camp.result.notebook)
    assert camp.policy._st.noise_std == 0.2
    # noise-free: logEI stays
    quiet = run(quadratic_problem(), budget=12, seed=0)
    assert quiet.session.lenz.shelf.acqf == "logei" and quiet.policy._st.noise_std == 0


# --------------------------------------------------------------------------- robustness


def test_internal_errors_fall_back_to_harness(monkeypatch):
    def boom(self, session, cv):
        raise RuntimeError("bug")

    monkeypatch.setattr(HeuristicSaraPolicy, "_model_step", boom)
    camp = run("branin", budget=10, policy=HeuristicSaraPolicy())
    res = camp.result
    assert res.n_evals == 10 and res.forced_commits > 0
    errors = [e for e in camp.session.log if e.get("action_class") == "reasoning" and e.get("kind") == "error"]
    assert errors and "RuntimeError: bug" in errors[0]["text"]
    # strict: the error reaches the harness (which records it, or re-raises it on request)
    with pytest.raises(RuntimeError):
        run("branin", budget=10, policy=HeuristicSaraPolicy(strict=True), on_policy_error="raise")


@pytest.mark.slow
def test_matches_fixed_bo_on_branin():
    ours, theirs = [], []
    for seed in range(3):
        ours.append(run("branin", budget=40, seed=seed).result.final_simple_regret)
        problem = get_problem("branin", seed=seed, randomize=True)
        theirs.append(Campaign(problem, FixedBOPolicy(), CampaignConfig(budget=40, seed=seed)).run()
                      .final_simple_regret)
    assert np.median(ours) <= 10 * np.median(theirs) + 1e-3


# --------------------------------------------------------------------------- review regressions


def _offset_bowl(offset: float, minimize: bool, scale: float = 1.0) -> Problem:
    """The 2-D bowl scaled, shifted by an additive constant (and negated when ``minimize``)."""
    base = quadratic_problem()
    sign = -1.0 if minimize else 1.0
    return dataclasses.replace(base, evaluate_fn=lambda c: {"y": sign * (scale * base.evaluate_fn(c)["y"] + offset)},
                               minimize=minimize, f_star=sign * (scale + offset))


@pytest.mark.parametrize("minimize", [False, True])
def test_success_tolerance_is_not_swamped_by_an_objective_offset(minimize):
    # adding a constant to f must not turn every real improvement into a "failure": the GP is
    # offset-invariant, so the trust-region bookkeeping must be too (TuRBO's 1e-3 * |y| is 10
    # here, about the whole observed range of ~13; agent-facing values keep 6 significant digits)
    for seed in (0, 1):
        camp = run(_offset_bowl(1e4, minimize, scale=10.0), budget=20, seed=seed)
        assert_clean(camp)
        st = camp.policy._st
        assert sum(n for kind, n in st.wins.items() if kind != "sobol") >= 1   # model steps improved
        ref = run(_offset_bowl(0.0, minimize, scale=10.0), budget=20, seed=seed).policy._st
        assert sum(st.wins.values()) >= sum(ref.wins.values()) - 1


def test_malformed_instruction_args_are_noted_not_fatal():
    bad = {"text": "Use the full domain.", "op": "set_bounds", "args": "full"}
    for cfg in ({"context": bad}, {"context_schedule": {2: bad}}):   # at start() and mid-run
        camp = run(quadratic_problem(), budget=6, **cfg)
        assert_clean(camp)
        assert any("set_bounds" in n["text"] and "not applied" in n["text"] for n in camp.result.notebook)
        assert camp.session.lenz.shelf.bounds is None


def test_initial_context_errors_do_not_crash_the_campaign(monkeypatch):
    def boom(self, session, item):
        raise RuntimeError("context bug")

    monkeypatch.setattr(HeuristicSaraPolicy, "_apply_context", boom)
    camp = run(quadratic_problem(), budget=5, policy=HeuristicSaraPolicy(), context="Be careful.")
    assert camp.result.n_evals == 5 and camp.result.policy_error is None
    errors = [e for e in camp.session.log if e.get("action_class") == "reasoning" and e.get("kind") == "error"]
    assert errors and "RuntimeError: context bug" in errors[0]["text"]
    assert camp.result.forced_commits == 0                          # the policy kept running the campaign
    with pytest.raises(RuntimeError):                               # strict re-raises to the harness
        run(quadratic_problem(), budget=5, context="Be careful.", on_policy_error="raise")


def _two_output_problem(noise_std: float = 0.0) -> Problem:
    space = SearchSpace([RealParam("a", 0.0, 1.0), RealParam("b", 0.0, 1.0)])

    def f(c):
        return {"y": -((c["a"] - 0.8) ** 2 + (c["b"] - 0.8) ** 2), "z": (c["a"] - 0.3) ** 2 + (c["b"] - 0.5) ** 2}

    return Problem(name="two_out", space=space, evaluate_fn=f, objective="y", minimize=False, f_star=0.0,
                   outputs={"y": "", "z": ""}, noise_std=noise_std)


@pytest.mark.parametrize("regret,noise,expected", [("cumulative", 0.0, ("ucb", {"beta": 1.0})),
                                                   ("simple", 0.05, ("noisy_logei", {}))])
def test_regime_acquisition_survives_an_objectives_round_trip(regret, noise, expected):
    # single -> multi-objective -> single: lenz falls back to logei; the policy must re-apply the
    # acquisition its regime calls for (UCB for cumulative regret, noisy_logei for noisy data)
    moo = {"text": "Also minimize z.", "op": "set_objectives",
           "args": {"objectives": [{"metric": "y", "minimize": False}, {"metric": "z", "minimize": True}]}}
    single = {"text": "Back to y only.", "op": "set_objectives",
              "args": {"objectives": [{"metric": "y", "minimize": False}]}}
    camp = run(_two_output_problem(noise), budget=16, regret=regret, context_schedule={5: moo, 10: single})
    assert_clean(camp)
    shelf = camp.session.lenz.shelf
    assert (shelf.acqf, shelf.acqf_params) == expected
    assert camp.policy._st.acqf == expected[0]


def test_global_move_rationale_reports_raw_predicted_values():
    # 'pm' scores live on the acquisition scale (negated for a minimized metric, raw units without
    # constraints); the rationale must quote the GP predictions in the metric's own units/direction
    from agentic_bo.sara.policies.heuristic import _Candidate, _State

    class Stub:
        def call(self, name, args=None):
            assert name == "score"
            vals = [{"pm": -0.9, "logei": -3.0}, {"pm": -0.4, "logei": -2.5}, {"pm": -0.5, "logei": -9.0}]
            return {"ok": True, "result": {"scores": [{"acquisition_values": v} for v in vals]}}

    policy = HeuristicSaraPolicy(strict=True)
    policy._st = _State(metric="loss", minimize=True, acqf="logei", best_u=-0.45, best_cfg={"a": 0.1})
    local = _Candidate(config={"a": 0.2}, kind="local", acqf="logei", mean=0.9, std=0.1)
    glob = _Candidate(config={"a": 0.7}, kind="global", acqf="logei", mean=0.4, std=0.2)
    cand, why = policy._choose(Stub(), local, glob, "incumbent", 0.2)
    assert cand is glob
    assert "standardized" not in why and "-0.4" not in why and "-0.9" not in why
    assert "0.4" in why and "0.9" in why and "loss" in why


def test_failed_instructed_evaluation_is_acknowledged():
    # an instructed config the evaluate tool rejects must not vanish silently: the policy notes it
    bad = {"text": "Please try a=2, b=0.1.", "op": "evaluate", "args": {"config": {"a": 2.0, "b": 0.1}}}
    camp = run(quadratic_problem(), budget=6, context_schedule={3: bad})
    res = camp.result
    assert res.n_evals == 6 and res.forced_commits == 0 and res.policy_error is None
    assert [e for e in camp.session.log if e.get("tool") == "evaluate" and not e["ok"]]
    notes = [n["text"] for n in res.notebook if "could not be evaluated" in n["text"]]
    assert len(notes) == 1 and "a=2" in notes[0] and "must be <= 1" in notes[0]
