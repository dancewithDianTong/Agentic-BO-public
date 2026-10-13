"""Tests for agentic_bo.experiment: campaigns, the resumable benchmark runner, statistics, plots and the CLI."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from agentic_bo import experiment as ex
from agentic_bo.experiment import (
    compare, format_report, load_results, main, plot_regret, prepare_problem, result_row, run_benchmark,
    run_campaign, summarize,
)
from agentic_bo.problems import get_problem
from agentic_bo.sara import CampaignResult
from agentic_bo.sara.policies import FixedBOPolicy, Policy
from agentic_bo.utils import LenzError

ROOT = Path(__file__).resolve().parents[1]


def row(problem: str, policy: str, seed: int, simple: float, cumulative: float | None = None, n: int = 5,
        regret_mode: str = "simple", **extra) -> dict:
    """A synthetic result row with linear regret curves ending at ``simple``."""
    cum = simple * n if cumulative is None else cumulative
    curve = list(np.linspace(simple * 3, simple, n))
    return {"problem": problem, "policy": policy, "seed": seed, "budget": n, "regret_mode": regret_mode,
            "is_moo": False, "n_evals": n, "final_simple_regret": simple, "final_cumulative_regret": cum,
            "simple_regret": curve, "cumulative_regret": list(np.cumsum(curve)), "best_value": simple,
            "forced_commits": 0, "wallclock_s": 0.1, "policy_error": None, "mean_actions_per_step": 2.0,
            "usage": {}, **extra}


class Crashing(Policy):
    """A buggy policy: raises a non-PolicyError exception (re-raised with ``on_policy_error="raise"``)."""

    name = "crashing"

    def step(self, session):
        raise ZeroDivisionError("bug in policy")


# --------------------------------------------------------------------------- problems and campaigns


def test_prepare_problem_follows_the_randomization_protocol():
    rnd = prepare_problem("branin", seed=3)
    assert rnd.name == "branin" and rnd.meta.get("randomized") and all(n.startswith("x_") for n in rnd.space.names)
    assert prepare_problem("branin", seed=3).space.names == rnd.space.names
    assert prepare_problem("branin", seed=4).space.names != rnd.space.names
    plain = prepare_problem("branin", seed=3, randomize=False)
    assert plain.space.names == ["x0", "x1"] and not plain.meta.get("randomized")
    ry = prepare_problem("reaction_yield", seed=1)          # prior problems keep their meaning
    assert "temperature" in ry.space.names and ry.prior
    noisy = prepare_problem("hartmann3", noise_std=0.1)
    assert noisy.noise_std == 0.1
    base = get_problem("branin")
    assert prepare_problem(base, seed=9) is base
    assert prepare_problem(base, noise_std=0.2).noise_std == 0.2
    with pytest.raises(LenzError):
        prepare_problem(3)  # type: ignore[arg-type]
    with pytest.raises(LenzError):
        prepare_problem("no_such_problem")


def test_run_campaign_settings_and_validation():
    res = run_campaign("branin", "fixed_bo", seed=2, budget=7, policy_kwargs={"n_init": 3})
    assert isinstance(res, CampaignResult) and res.budget == 7 and res.seed == 2 and res.n_evals == 7
    assert sum("Initial Sobol" in r["rationale"] for r in res.records) == 3
    assert res.problem_info["meta"]["randomized"]
    res2 = run_campaign("branin", "sobol", seed=0, budget=4, randomize=False, regret="cumulative",
                        allow_early_stop=True)
    assert not res2.problem_info["meta"].get("randomized") and res2.regret_mode == "cumulative"
    assert res2.campaign_config["allow_early_stop"] is True
    assert run_campaign("hartmann3", FixedBOPolicy(n_init=2), budget=3).policy == "fixed_bo"
    with pytest.raises(LenzError):
        run_campaign("branin", FixedBOPolicy(), budget=3, policy_kwargs={"n_init": 1})
    with pytest.raises(LenzError, match="unknown campaign settings"):
        run_campaign("branin", "sobol", budget=3, bogus=True)
    # same seed -> same run
    a = run_campaign("hartmann3", "heuristic", seed=1, budget=6)
    b = run_campaign("hartmann3", "heuristic", seed=1, budget=6)
    assert a.simple_regret == b.simple_regret


def test_result_row_round_trips_through_json(tmp_path):
    res = run_campaign("branin", "sobol", budget=4)
    r = result_row(res, tmp_path / "x.json")
    assert r["problem"] == "branin" and r["policy"] == "sobol" and len(r["simple_regret"]) == 4
    assert r["final_simple_regret"] == res.simple_regret[-1] and r["error"] is None
    back = ex._row_from_dict(json.loads(json.dumps(ex.jsonable(r))))
    assert back["simple_regret"] == r["simple_regret"]
    full = ex._row_from_dict(res.to_dict())
    assert full["final_cumulative_regret"] == pytest.approx(r["final_cumulative_regret"])
    with pytest.raises(LenzError):
        ex._row_from_dict({"nothing": 1})


# --------------------------------------------------------------------------- benchmark runner


def test_run_benchmark_writes_runs_and_summary_and_resumes(tmp_path, monkeypatch):
    out = tmp_path / "bench"
    rows = run_benchmark(["branin", "hartmann3"], ["sobol", "fixed_bo"], seeds=2, out_dir=out, budget=6)
    assert [(r["problem"], r["policy"], r["seed"]) for r in rows] == [
        (p, q, s) for p in ("branin", "hartmann3") for q in ("sobol", "fixed_bo") for s in (0, 1)]
    assert all(r["error"] is None and len(r["simple_regret"]) == 6 for r in rows)
    files = sorted(p.name for p in (out / "runs").glob("*.json"))
    assert len(files) == 8 and "branin__sobol__seed0.json" in files
    doc = json.loads((out / "runs" / "branin__fixed_bo__seed1.json").read_text())
    assert doc["run_spec"]["policy"] == "fixed_bo" and doc["run_spec"]["budget"] == 6 and doc["records"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["format"] == ex.SUMMARY_FORMAT and len(summary["runs"]) == 8 and not summary["failed"]
    assert summary["summary"]["table"]["branin"]["sobol"]["n_runs"] == 2
    assert summary["summary"]["comparisons"]
    loaded = load_results(out)      # benchmark order (sobol before fixed_bo), not file-name order
    assert [(r["problem"], r["policy"], r["seed"]) for r in loaded] == [
        (r["problem"], r["policy"], r["seed"]) for r in rows]

    def boom(task):
        raise AssertionError("a finished run was executed again")

    monkeypatch.setattr(ex, "_run_task", boom)
    again = run_benchmark(["branin", "hartmann3"], ["sobol", "fixed_bo"], seeds=2, out_dir=out, budget=6)
    assert [r["final_simple_regret"] for r in again] == [r["final_simple_regret"] for r in rows]
    with pytest.raises(LenzError, match="different settings"):
        run_benchmark(["branin"], ["sobol"], seeds=1, out_dir=out, budget=7)


def test_run_benchmark_reruns_unreadable_files_and_records_failures(tmp_path, capsys):
    out = tmp_path / "bench"
    # the harness turns policy exceptions into fallback commits by default; "raise" makes
    # the run itself crash, which run_benchmark must record without losing the others
    opts = {"on_policy_error": "raise"}
    run_benchmark(["branin"], ["sobol"], seeds=[0, 1], out_dir=out, budget=4, campaign_options=opts)
    target = out / "runs" / "branin__sobol__seed1.json"
    target.write_text("{ not json")
    rows = run_benchmark(["branin"], ["sobol", Crashing()], seeds=[0, 1], out_dir=out, budget=4, verbose=True,
                         campaign_options=opts)
    printed = capsys.readouterr().out
    assert "re-running" in printed and "FAILED" in printed
    assert json.loads(target.read_text())["problem"] == "branin"
    failed = [r for r in rows if r.get("error")]
    assert len(failed) == 2 and all("ZeroDivisionError" in r["error"] for r in failed)
    assert not list((out / "runs").glob("*crashing*"))
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["failed"]) == 2 and summary["summary"]["n_runs"] == 2


def test_run_benchmark_parallel_matches_sequential(tmp_path):
    env_before = {k: os.environ.get(k) for k in ex._BLAS_VARS}
    seq = run_benchmark(["branin"], ["fixed_bo", "heuristic"], seeds=2, out_dir=tmp_path / "a", budget=7)
    par = run_benchmark(["branin"], ["fixed_bo", "heuristic"], seeds=2, out_dir=tmp_path / "b", budget=7,
                        n_jobs=2)
    assert [r["simple_regret"] for r in par] == [r["simple_regret"] for r in seq]
    assert all(r["error"] is None for r in par)
    assert {k: os.environ.get(k) for k in ex._BLAS_VARS} == env_before   # restored after the pool closes


def test_run_benchmark_policy_specs_and_validation(tmp_path):
    rows = run_benchmark(["branin"], [{"name": "fixed_bo", "label": "bo3", "kwargs": {"n_init": 3}}, "sobol"],
                         seeds=1, out_dir=tmp_path, budget=4)
    assert [r["policy"] for r in rows] == ["bo3", "sobol"]
    doc = json.loads((tmp_path / "runs" / "branin__bo3__seed0.json").read_text())
    assert doc["policy"] == "bo3" and doc["run_spec"]["policy_kwargs"] == {"n_init": 3}
    bad = [
        dict(policies=["sobol", "sobol"]), dict(policies=[{"name": "sobol", "nope": 1}]),
        dict(policies=[{"name": "fixed_bo", "kwargs": {"bogus": 1}}]), dict(policies=["no_such_policy"]),
        dict(policies=[3]), dict(seeds=0), dict(seeds=[1, 1]), dict(seeds=True), dict(n_jobs=0),
        dict(problems=["branin", "branin"]), dict(problems=[3]),
        dict(campaign_options={"budget": 3}),
    ]
    for kw in bad:
        args = {"problems": ["branin"], "policies": ["sobol"], "seeds": 1, "out_dir": tmp_path / "x", **kw}
        with pytest.raises((LenzError, TypeError)):
            run_benchmark(**args)


def test_llm_runs_are_sequential_and_get_a_transcript(tmp_path):
    """A fake Anthropic client that is not picklable: n_jobs=2 must still run it in-process."""
    rng = np.random.default_rng(0)
    calls = []

    class Stream:
        """Context manager returning one scripted message: evaluate a random branin config."""

        def __init__(self, kw):
            calls.append(kw)
            cfg = {"x0": float(rng.uniform(-5, 10)), "x1": float(rng.uniform(0, 15))}
            self.msg = NS(content=[NS(type="tool_use", id=f"toolu_{len(calls)}", name="evaluate",
                                      input={"config": cfg, "rationale": "fake model"})],
                          stop_reason="tool_use", model="claude-opus-5-5",
                          usage=NS(input_tokens=100, output_tokens=10, cache_read_input_tokens=0,
                                   cache_creation_input_tokens=0))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return self.msg

    client = NS(beta=NS(messages=NS(stream=lambda **kw: Stream(kw))))   # lambdas: not picklable
    rows = run_benchmark(["branin"], [{"name": "sara", "kwargs": {"client": client}}, "sobol"], seeds=1,
                         out_dir=tmp_path, budget=3, randomize=False, n_jobs=2)
    assert all(r["error"] is None for r in rows), rows
    llm = rows[0]
    assert llm["policy"] == "sara" and llm["usage"]["api_calls"] == 3 and llm["usage"]["cost_usd"] > 0
    assert len(calls) == 3 and json.loads(Path(llm["path"]).read_text())["forced_commits"] == 0
    transcript = tmp_path / "transcripts" / "branin__sara__seed0.jsonl"
    assert transcript.exists() and transcript.read_text().count("\n") >= 4
    assert ex._is_llm("Claude") and ex._is_llm("llm-only") and not ex._is_llm("heuristic")


# --------------------------------------------------------------------------- statistics


def test_compare_mann_whitney_outcomes():
    rows = []
    for s in range(6):
        rows += [row("p", "fixed_bo", s, 1.0 + 0.1 * s), row("p", "better", s, 0.01 * (s + 1)),
                 row("p", "worse", s, 10.0 + s), row("p", "same", s, 1.0 + 0.1 * s)]
    out = {c["policy"]: c for c in compare(rows)}
    assert set(out) == {"better", "worse", "same"}
    assert out["better"]["outcome"] == "win" and out["better"]["p_value"] < 0.05 and out["better"]["effect"] == 0
    assert out["worse"]["outcome"] == "loss" and out["worse"]["effect"] == 1
    assert out["same"]["outcome"] == "tie" and out["same"]["p_value"] == pytest.approx(1.0)
    assert out["better"]["n"] == 6 and out["better"]["n_ref"] == 6 and out["better"]["metric"] == "simple"
    # three seeds per side can never be significant at 0.05
    few = [r for r in rows if r["seed"] < 3]
    assert {c["outcome"] for c in compare(few)} == {"tie"}
    # unknown optimum -> no decision
    nan_rows = [row("q", "fixed_bo", 0, math.nan), row("q", "x", 0, math.nan)]
    assert compare(nan_rows)[0]["outcome"] is None
    assert compare(rows, reference="absent") and all(c["outcome"] is None for c in compare(rows, reference="absent"))


def test_compare_metric_selection():
    rows = [row("p", "fixed_bo", s, 1.0, cumulative=100.0 + s, regret_mode="cumulative") for s in range(5)]
    rows += [row("p", "a", s, 5.0, cumulative=1.0 + s, regret_mode="cumulative") for s in range(5)]
    assert compare(rows)[0]["metric"] == "cumulative" and compare(rows)[0]["outcome"] == "win"
    assert compare(rows, metric="simple")[0]["outcome"] == "loss"
    with pytest.raises(LenzError):
        compare(rows, metric="median")


def test_summarize_table_totals_and_report():
    rows = []
    for s in range(5):
        rows += [row("p1", "fixed_bo", s, 1.0 + s), row("p1", "heuristic", s, 0.001 * (s + 1)),
                 row("p2", "fixed_bo", s, 1.0 + s), row("p2", "heuristic", s, 1.5 + s,
                                                         usage={"cost_usd": 0.5}, forced_commits=1)]
    rows.append({"problem": "p1", "policy": "heuristic", "seed": 9, "error": "Boom"})
    s = summarize(rows)
    assert s["problems"] == ["p1", "p2"] and s["policies"] == ["fixed_bo", "heuristic"]
    assert s["n_runs"] == 20 and s["n_failed"] == 1 and s["reference"] == "fixed_bo"
    e = s["table"]["p1"]["heuristic"]
    assert e["n_runs"] == 5 and e["seeds"] == [0, 1, 2, 3, 4]
    assert e["simple"]["median"] == pytest.approx(0.003) and e["simple"]["q25"] == pytest.approx(0.002)
    assert e["vs_reference"] == "win" and s["table"]["p2"]["heuristic"]["vs_reference"] == "tie"
    assert s["table"]["p2"]["heuristic"]["cost_usd"] == pytest.approx(2.5)
    assert s["table"]["p1"]["fixed_bo"]["vs_reference"] is None
    assert s["totals"] == {"heuristic": {"wins": 1, "ties": 1, "losses": 0, "undecided": 0}}
    text = format_report(s)
    assert "p1 (budget 5)" in text and "win (p=" in text and "(reference)" in text
    assert "1 win(s), 1 tie(s), 0 loss(es)" in text and "1 run(s) failed" in text and "$2.50 API cost" in text
    no_ref = summarize([r for r in rows if r.get("policy") != "fixed_bo"])
    assert no_ref["reference"] is None and no_ref["totals"] == {} and "vs " not in format_report(no_ref)


def test_load_results_accepts_dirs_files_and_mixed_inputs(tmp_path):
    rows = run_benchmark(["branin"], ["sobol"], seeds=2, out_dir=tmp_path, budget=3)
    loaded = load_results(tmp_path)
    assert [r["seed"] for r in loaded] == [0, 1]
    assert [r["simple_regret"] for r in loaded] == [r["simple_regret"] for r in rows]
    assert len(load_results(tmp_path / "runs")) == 2
    assert len(load_results(rows[0]["path"])) == 1
    res = CampaignResult.load(rows[0]["path"])
    mixed = ex._rows([res, rows[1], str(tmp_path / "runs")])
    assert len(mixed) == 4
    with pytest.raises(LenzError):
        load_results(tmp_path / "missing")
    with pytest.raises(LenzError):
        ex._rows([3.0])


# --------------------------------------------------------------------------- plots


def test_plot_regret_writes_images(tmp_path):
    pytest.importorskip("matplotlib")
    rows = [row(p, q, s, (0.0 if q == "a" and s == 0 else 0.1 * (s + 1)) * (1 + i), n=6 + i)
            for i, p in enumerate(["p1", "p2", "p3", "p4"]) for q in ("a", "b", "c") for s in range(3)]
    out = plot_regret(rows, tmp_path / "simple.png")
    assert out == str(tmp_path / "simple.png") and os.path.getsize(out) > 10_000
    out = plot_regret(rows, tmp_path / "sub" / "cum.pdf", kind="cumulative", log_scale=False,
                      problems=["p2"], policies=["c", "a"], title="Cumulative")
    assert os.path.getsize(out) > 1000
    with pytest.raises(LenzError):
        plot_regret(rows, tmp_path / "x.png", kind="median")
    with pytest.raises(LenzError):
        plot_regret(rows, tmp_path / "x.png", problems=["absent"])
    many = [row("p", f"pol{i}", 0, 1.0) for i in range(9)]
    with pytest.raises(LenzError, match="at most 8"):
        plot_regret(many, tmp_path / "x.png")


# --------------------------------------------------------------------------- CLI


def test_cli_run_report_and_list(tmp_path, capsys):
    out = tmp_path / "cli"
    code = main(["run", "--problems", "branin", "--policies", "fixed_bo", "sobol", "--seeds", "2", "--seed-start",
                 "3", "--budget", "6", "--out", str(out), "--policy-kwargs", '{"fixed_bo": {"n_init": 4}}',
                 "--quiet"])
    text = capsys.readouterr().out
    assert code == 0 and "branin (budget 6)" in text and "regret_simple.png" in text
    assert sorted(p.name for p in (out / "runs").glob("*.json")) == [
        "branin__fixed_bo__seed3.json", "branin__fixed_bo__seed4.json", "branin__sobol__seed3.json",
        "branin__sobol__seed4.json"]
    assert (out / "regret_simple.png").exists() and (out / "regret_cumulative.png").exists()
    assert json.loads((out / "runs" / "branin__fixed_bo__seed3.json").read_text())["run_spec"]["policy_kwargs"] \
        == {"n_init": 4}

    assert main(["report", "--results", str(out), "--no-plot"]) == 0
    assert "Totals vs fixed_bo" in capsys.readouterr().out
    assert main(["report", "--results", str(out), "--json", "--no-plot", "--metric", "cumulative"]) == 0
    assert json.loads(capsys.readouterr().out)["metric"] == "cumulative"
    assert main(["list"]) == 0 and "hartmann6" in capsys.readouterr().out

    assert main(["report", "--results", str(tmp_path / "empty")]) == 2
    assert main(["run", "--problems", "branin", "--out", str(out), "--policy-kwargs", "[1]"]) == 2
    assert main(["run", "--problems", "branin", "--out", str(out), "--policy-kwargs", '{"x": {}}']) == 2
    assert main(["run", "--problems", "nope", "--out", str(tmp_path / "n")]) == 2
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main([])


def test_module_entry_points_run_cleanly():
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    for args in (["-m", "agentic_bo.experiment", "list"], ["-m", "agentic_bo", "problems"]):
        proc = subprocess.run([sys.executable, "-W", "error", *args], capture_output=True, text=True, cwd=str(ROOT),
                              env=env, timeout=60)
        assert proc.returncode == 0, proc.stderr
        assert "branin" in proc.stdout and proc.stderr == ""
    proc = subprocess.run([sys.executable, "-m", "agentic_bo", "bogus"], capture_output=True, text=True,
                          cwd=str(ROOT), env=env, timeout=60)
    assert proc.returncode == 2 and "unknown command" in proc.stderr


# --------------------------------------------------------------------------- review regressions


def test_compare_small_samples_with_ties_use_the_exact_test():
    # within-group ties made scipy fall back to the normal approximation, which is
    # anti-conservative for tiny samples: 3 vs 3 seeds came out "significant" (p = 0.047)
    # although the exact permutation p-value is 2 / C(6, 3) = 0.1
    rows = [row("p", "fixed_bo", s, 1.0) for s in range(3)] + [row("p", "a", s, 0.0) for s in range(3)]
    c = compare(rows)[0]
    assert c["p_value"] == pytest.approx(0.1) and c["outcome"] == "tie" and c["effect"] == 0.0
    # 2 vs 6 seeds: exact p = 2 / C(8, 2) = 1 / 14 (the approximation gave 0.015)
    rows = [row("p", "fixed_bo", s, 1.0) for s in range(6)] + [row("p", "a", s, 0.0) for s in range(2)]
    assert compare(rows)[0]["p_value"] == pytest.approx(1 / 14) and compare(rows)[0]["outcome"] == "tie"
    # 4 vs 4 can be significant: p = 2 / C(8, 4) = 1 / 35
    rows = [row("p", "fixed_bo", s, 1.0) for s in range(4)] + [row("p", "a", s, 0.0) for s in range(4)]
    assert compare(rows)[0]["p_value"] == pytest.approx(1 / 35) and compare(rows)[0]["outcome"] == "win"
    # partial ties: exact two-sided p-value = 2 * min(P(U <= u), P(U >= u)), checked by brute force
    ref, x = [1.0, 2.0, 2.0, 3.0], [0.5, 2.0, 0.5]
    rows = [row("p", "fixed_bo", s, v) for s, v in enumerate(ref)] + [row("p", "a", s, v) for s, v in enumerate(x)]
    import itertools

    from scipy.stats import rankdata
    pooled = np.array(x + ref)
    ranks = rankdata(pooled)
    sums = np.array([ranks[list(idx)].sum() for idx in itertools.combinations(range(7), 3)])
    obs = ranks[:3].sum()
    expected = min(1.0, 2 * min(np.mean(sums <= obs + 1e-9), np.mean(sums >= obs - 1e-9)))
    assert compare(rows)[0]["p_value"] == pytest.approx(expected)
    # large samples keep scipy's default (normal approximation with tie correction)
    from scipy.stats import mannwhitneyu
    big_ref, big_x = [float(s % 3) for s in range(12)], [float(s % 2) for s in range(12)]
    rows = [row("p", "fixed_bo", s, v) for s, v in enumerate(big_ref)] + [
        row("p", "a", s, v) for s, v in enumerate(big_x)]
    assert compare(rows)[0]["p_value"] == pytest.approx(
        mannwhitneyu(big_x, big_ref, alternative="two-sided").pvalue)


def test_cli_report_accepts_a_single_result_file(tmp_path, capsys):
    out = tmp_path / "bench"
    run_benchmark(["branin"], ["sobol"], seeds=1, out_dir=out, budget=4)
    run_file = next((out / "runs").glob("*.json"))
    # load_results accepts a single file, but plotting used the file path as a folder and crashed
    assert main(["report", "--results", str(run_file)]) == 0
    assert "branin" in capsys.readouterr().out
    assert (run_file.parent / "regret_simple.png").exists()
