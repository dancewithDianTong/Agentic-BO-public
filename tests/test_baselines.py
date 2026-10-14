"""Tests for the policy interface, get_policy and the baseline policies (Sobol, Random, FixedBO)."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from agentic_bo.problems import get_problem
from agentic_bo.sara import Campaign, CampaignConfig, CampaignResult
from agentic_bo.sara.policies import (
    POLICY_NAMES, FixedBOPolicy, Policy, PolicyError, RandomPolicy, SobolPolicy, get_policy,
)
from agentic_bo.utils import LenzError

BASELINES = [SobolPolicy, RandomPolicy, FixedBOPolicy]
ROOT = Path(__file__).resolve().parents[1]


def run(policy: Policy, problem: str = "branin", budget: int = 20, seed: int = 0, **cfg) -> Campaign:
    camp = Campaign(get_problem(problem), policy, CampaignConfig(budget=budget, seed=seed, **cfg))
    camp.run()
    return camp


# --------------------------------------------------------------------------- interface


def test_policy_is_abstract_and_hooks_default_to_noops():
    with pytest.raises(TypeError):
        Policy()  # type: ignore[abstract]

    class Minimal(Policy):
        def step(self, session):
            pass

    p = Minimal()
    assert p.name == "policy" and p.toolset is None and p.usage() == {}
    assert p.start(None) is None and p.on_context(None, None) is None and p.finish(None) is None
    assert issubclass(PolicyError, RuntimeError)
    assert repr(FixedBOPolicy()) == "FixedBOPolicy(name='fixed_bo')"


def test_get_policy_names_aliases_and_kwargs():
    assert isinstance(get_policy("sobol"), SobolPolicy)
    assert isinstance(get_policy("Random"), RandomPolicy)
    bo = get_policy("fixed-bo", n_init=3, acqf="ucb")
    assert isinstance(bo, FixedBOPolicy) and bo.n_init == 3 and bo.acqf == "ucb"
    assert isinstance(get_policy("standard_bo"), FixedBOPolicy)
    for bad in ("", "bayes", None):
        with pytest.raises(LenzError):
            get_policy(bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        get_policy("sobol", n_init=3)
    assert set(POLICY_NAMES) == {"sobol", "random", "fixed_bo", "heuristic", "sara", "llm_only", "openai",
                                 "openai_llm_only"}


def test_get_policy_imports_optional_modules_lazily(monkeypatch):
    code = ("import sys, agentic_bo.sara, agentic_bo.sara.policies as p; "
            "print(any(m in sys.modules for m in ('agentic_bo.sara.policies.llm', "
            "'agentic_bo.sara.policies.heuristic', 'agentic_bo.sara.policies.openai_llm', 'anthropic', 'openai')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert out.stdout.strip() == "False"

    class Stub(Policy):
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.toolset = kwargs.get("toolset", "full")

        def step(self, session):
            pass

    for mod, cls in (("agentic_bo.sara.policies.llm", "ClaudeSaraPolicy"),
                     ("agentic_bo.sara.policies.heuristic", "HeuristicSaraPolicy")):
        fake = types.ModuleType(mod)
        setattr(fake, cls, Stub)
        monkeypatch.setitem(sys.modules, mod, fake)
    sara = get_policy("sara", model="m")
    assert isinstance(sara, Stub) and sara.kwargs == {"model": "m"}
    only = get_policy("llm-only", effort="high")
    assert only.kwargs == {"effort": "high", "toolset": "llm_only"} and only.toolset == "llm_only"
    assert get_policy("heuristic", seed=3).kwargs == {"seed": 3}


def test_fixed_bo_argument_validation():
    for kwargs in ({"n_init": -1}, {"n_init": 2.5}, {"acqf": "sobol"}, {"acqf": "nope"}):
        with pytest.raises(LenzError):
            FixedBOPolicy(**kwargs)
    assert FixedBOPolicy(acqf=None).acqf is None


# --------------------------------------------------------------------------- campaigns


@pytest.mark.parametrize("policy_cls", BASELINES, ids=lambda c: c.__name__)
def test_baseline_campaign_on_branin(policy_cls):
    camp = run(policy_cls(), budget=20, seed=1)
    res, session = camp.result, camp.session
    p = get_problem("branin")
    assert res.n_evals == 20 and len(res.simple_regret) == 20 and len(res.cumulative_regret) == 20
    assert res.forced_commits == 0 and res.policy_error is None and not res.stopped_early
    sr = np.asarray(res.simple_regret)
    assert np.all(np.diff(sr) <= 1e-12) and np.all(sr >= -1e-9)
    regrets = np.array([r["regret"] for r in res.records])
    np.testing.assert_allclose(regrets, [p.regret(p.evaluate(r["config"])) for r in res.records])
    np.testing.assert_allclose(res.cumulative_regret, np.cumsum(regrets))
    np.testing.assert_allclose(sr, np.minimum.accumulate(regrets))
    # records and the lenz trial log describe the same evaluations
    trials = session.lenz.frame.trials
    assert [t.trial_id for t in trials] == [r["trial_id"] for r in res.records] == list(range(20))
    for t, r in zip(trials, res.records):
        assert t.status == "completed" and t.config == r["config"] and t.metrics == r["observed"]
        assert t.meta["rationale"] == r["rationale"] and t.eval_index == r["t"]
    # each step: one suggest, one evaluate (FixedBO adds one status probe in start)
    assert res.actions_per_step[1:] == [1] * 19
    assert res.action_counts["evaluate"] == 20 and res.action_counts["suggest"] == 20
    assert all(e["ok"] for e in res.log)
    assert len({tuple(r["config"].values()) for r in res.records}) == 20      # no repeats


def test_fixed_bo_phases_and_rationales():
    camp = run(FixedBOPolicy(n_init=4), budget=8, seed=2)
    res = camp.result
    suggest_acqfs = [e["result"]["acqf"] for e in res.log if e["tool"] == "suggest"]
    assert suggest_acqfs == ["sobol"] * 4 + ["logei"] * 4
    assert all("Initial Sobol design point" in r["rationale"] for r in res.records[:4])
    assert all(r["rationale"].startswith("Standard BO: top logei candidate") for r in res.records[4:])
    # never reconfigures
    assert not any(e["action_class"] == "reconfigure" for e in res.log)
    assert camp.session.lenz.shelf.acqf == "logei"


def test_fixed_bo_respects_custom_acqf_and_moo_default():
    res = run(FixedBOPolicy(n_init=2, acqf="ucb"), budget=4).result
    assert [e["result"]["acqf"] for e in res.log if e["tool"] == "suggest"][2:] == ["ucb", "ucb"]
    moo = run(FixedBOPolicy(n_init=2), problem="branin_currin", budget=4).result
    assert [e["result"]["acqf"] for e in moo.log if e["tool"] == "suggest"][2:] == ["ehvi", "ehvi"]
    assert moo.is_moo and moo.forced_commits == 0


def test_sobol_and_random_differ_and_are_seeded():
    a = run(SobolPolicy(), budget=6, seed=3).result
    b = run(SobolPolicy(), budget=6, seed=3).result
    c = run(RandomPolicy(), budget=6, seed=3).result
    d = run(RandomPolicy(), budget=6, seed=4).result
    assert [r["config"] for r in a.records] == [r["config"] for r in b.records]
    assert [r["config"] for r in a.records] != [r["config"] for r in c.records]
    assert [r["config"] for r in c.records] != [r["config"] for r in d.records]


def test_baseline_survives_failing_suggest(monkeypatch):
    """If suggest errors, the baseline returns and the harness commits a forced fallback."""
    from agentic_bo.sara import session as session_mod

    class Broken(session_mod.AgentSession):
        def call(self, name, args=None):
            if name == "suggest":
                return {"ok": False, "error": "lenz unavailable"}
            return super().call(name, args)

    monkeypatch.setattr(session_mod, "AgentSession", Broken)
    res = Campaign(get_problem("branin"), SobolPolicy(), CampaignConfig(budget=3)).run()
    assert res.n_evals == 3 and res.forced_commits == 3 and res.policy_error is None


def test_fixed_bo_beats_random_search_on_average():
    gaps = {}
    for problem in ("branin", "hartmann3"):
        bo = [run(FixedBOPolicy(), problem=problem, budget=20, seed=s).result.final_simple_regret for s in range(3)]
        rnd = [run(RandomPolicy(), problem=problem, budget=20, seed=s).result.final_simple_regret for s in range(3)]
        gaps[problem] = (np.mean(bo), np.mean(rnd))
        assert np.mean(bo) < np.mean(rnd), gaps
        assert np.median(bo) < np.median(rnd), gaps


def test_baseline_result_round_trip(tmp_path):
    res = run(FixedBOPolicy(n_init=3), problem="hartmann3", budget=6, seed=0, regret="cumulative").result
    path = tmp_path / "bo.json"
    res.save(path)
    back = CampaignResult.load(path)
    assert back.policy == "fixed_bo" and back.regret_mode == "cumulative" and back.n_evals == 6
    assert back.simple_regret == pytest.approx(res.simple_regret)
    assert back.cumulative_regret == pytest.approx(res.cumulative_regret)
    assert back.records == res.records and back.actions_per_step == res.actions_per_step
