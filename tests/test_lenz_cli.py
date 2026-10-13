"""Tests for the lenz command-line interface (``python -m agentic_bo.lenz``)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agentic_bo.lenz import Lenz
from agentic_bo.lenz.cli import build_parser, main

ROOT = Path(__file__).resolve().parents[1]
SPACE = [{"name": "x0", "type": "real", "lower": -2, "upper": 2},
         {"name": "x1", "type": "real", "lower": -2, "upper": 2}]


def quad(cfg: dict) -> float:
    return -((cfg["x0"] - 0.5) ** 2 + (cfg["x1"] + 0.3) ** 2)


def run(capsys, *argv: str) -> tuple[int, dict]:
    """Run the CLI in-process; assert exactly one JSON line was printed."""
    code = main(list(argv))
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) == 1, out
    return code, json.loads(lines[0])


@pytest.fixture
def study(tmp_path, capsys) -> str:
    path = str(tmp_path / "s.json")
    code, env = run(capsys, "--study", path, "create", "--space", json.dumps(SPACE), "--objectives", "f",
                    "--seed", "3")
    assert code == 0 and env["ok"] and env["command"] == "create"
    return path


def test_create_suggest_submit_observe_status(study, capsys):
    path = study
    env = json.loads(Path(path).read_text())
    assert env["format"] == "agentic_bo.lenz/study"
    for i in range(4):
        code, env = run(capsys, "--study", path, "suggest")
        assert code == 0 and env["command"] == "suggest"
        cfg = env["result"]["candidates"][0]["config"]
        code, env = run(capsys, "--study", path, "submit", "--config", json.dumps(cfg))
        assert code == 0 and env["result"] == {"trial_id": i, "status": "pending", "config": cfg,
                                               "in_region": True, "duplicate_of": None}
        code, env = run(capsys, "--study", path, "observe", "--trial-id", str(i),
                        "--metrics", json.dumps({"f": quad(cfg)}))
        assert code == 0 and env["result"]["eval_index"] == i + 1
    code, env = run(capsys, "--study", path, "status")
    assert code == 0 and env["result"]["n_observed"] == 4 and env["result"]["n_pending"] == 0
    assert env["result"]["incumbent"]["value"] == max(
        t.metrics["f"] for t in Lenz.load(path).frame.trials)
    # every call (probes included) is in the persisted event log
    events = [e.command for e in Lenz.load(path).frame.events]
    assert events == ["create"] + ["suggest", "submit", "observe"] * 4 + ["status"]


def test_cli_matches_library_exactly(study, capsys, tmp_path):
    """Load -> run -> save per call reproduces an in-memory session bit for bit."""
    lib = Lenz(SPACE, "f", seed=3)
    for _ in range(4):
        _, env = run(capsys, "--study", study, "suggest", "--q", "2")
        lib_res = lib.suggest(q=2)
        assert env["result"] == json.loads(json.dumps(lib_res))
        cfg = lib_res["candidates"][0]["config"]
        run(capsys, "--study", study, "observe", "--config", json.dumps(cfg), "--metrics", json.dumps({"f": quad(cfg)}))
        lib.observe({"f": quad(cfg)}, config=cfg)
    _, env = run(capsys, "--study", study, "suggest", "--around", "incumbent", "--radius", "0.05")
    assert env["result"] == json.loads(json.dumps(lib.suggest(around="incumbent", radius=0.05)))


def test_reconfigure_and_probe_commands(study, capsys, tmp_path):
    path = study
    for cfg in ({"x0": 0.0, "x1": 0.0}, {"x0": 1.0, "x1": -1.0}, {"x0": -1.0, "x1": 1.5}, {"x0": 0.4, "x1": -0.2}):
        code, _ = run(capsys, "--study", path, "observe", "--config", json.dumps(cfg),
                      "--metrics", json.dumps({"f": quad(cfg), "c": cfg["x0"]}))
        assert code == 0
    code, env = run(capsys, "--study", path, "set-bounds", "--bounds", json.dumps({"x0": [0, 2]}))
    assert code == 0 and env["result"]["bounds"] == {"x0": [0.0, 2.0]} and env["result"]["n_in_region"] == 3
    code, env = run(capsys, "--study", path, "suggest", "--q", "2", "--acqf", "ucb")
    assert code == 0 and all(c["config"]["x0"] >= 0 for c in env["result"]["candidates"])
    code, env = run(capsys, "--study", path, "set-bounds", "--bounds", "full")
    assert code == 0 and env["result"]["bounds"] is None
    code, env = run(capsys, "--study", path, "set-acqf", "--acqf", "ucb", "--beta", "2.5")
    assert code == 0 and env["result"]["acqf_params"] == {"beta": 2.5}
    code, env = run(capsys, "--study", path, "set-acqf", "--acqf", "ts", "--params", '{"n_candidates": 128}')
    assert code == 0 and env["result"]["acqf_params"] == {"n_candidates": 128}
    code, env = run(capsys, "--study", path, "set-constraints", "--constraints", '[{"metric": "c", "upper": 0.5}]')
    assert code == 0 and env["result"]["n_feasible"] == 3
    code, env = run(capsys, "--study", path, "diagnostics", "--metric", "c")
    assert code == 0 and set(env["result"]["models"]) == {"c"}
    cfgs = json.dumps([{"x0": 0.1, "x1": 0.1}, {"x0": 1.5, "x1": 0.0}])
    code, env = run(capsys, "--study", path, "predict", "--configs", cfgs)
    assert code == 0 and len(env["result"]["predictions"]) == 2
    assert env["result"]["predictions"][0]["prob_feasible"] > env["result"]["predictions"][1]["prob_feasible"]
    code, env = run(capsys, "--study", path, "score", "--configs", cfgs, "--acqf", "logei,ucb")
    assert code == 0 and env["result"]["acqfs"] == ["logei", "ucb"]
    code, env = run(capsys, "--study", path, "incumbent")
    assert code == 0 and env["result"]["incumbent"]["config"] == {"x0": 0.4, "x1": -0.2}
    code, env = run(capsys, "--study", path, "trials", "--last", "2", "--sort-by", "value")
    assert code == 0 and env["result"]["n"] == 2
    code, env = run(capsys, "--study", path, "set-objectives", "--objectives",
                    '[{"metric": "f"}, {"metric": "c", "minimize": true}]')
    assert code == 0 and env["result"]["acqf"] == "ehvi" and env["result"]["acqf_switched"]
    code, env = run(capsys, "--study", path, "pareto")
    assert code == 0 and env["result"]["is_moo"]
    code, env = run(capsys, "--study", path, "set-constraints", "--constraints", "[]")
    assert code == 0 and env["result"]["constraints"] == []
    tid_env = run(capsys, "--study", path, "submit", "--config", '{"x0": 0, "x1": 0}', "--source", "manual",
                  "--meta", '{"why": "check"}')[1]
    code, env = run(capsys, "--study", path, "abandon", "--trial-id", str(tid_env["result"]["trial_id"]))
    assert code == 0 and env["result"]["status"] == "abandoned"
    # JSON from a file with @path, and --study after the command
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text('[{"x0": 0.2, "x1": 0.3}]')
    code, env = run(capsys, "predict", "--configs", f"@{cfg_file}", "--study", path)
    assert code == 0 and env["result"]["predictions"][0]["config"] == {"x0": 0.2, "x1": 0.3}


def test_error_envelopes_and_exit_codes(study, capsys, tmp_path):
    path = study
    cases = [
        (["--study", path, "observe", "--trial-id", "7", "--metrics", '{"f": 1}'], "LenzError", "unknown trial_id"),
        (["--study", path, "submit", "--config", '{"x0": 9, "x1": 0}'], "LenzError", "outside the domain"),
        (["--study", path, "submit", "--config", "{bad json"], "LenzError", "invalid JSON"),
        (["--study", path, "suggest", "--q", "two"], "LenzError", "invalid value"),
        (["--study", path, "set-acqf", "--acqf", "magic"], "LenzError", "unknown acquisition"),
        (["--study", path, "submit"], "UsageError", "--config"),
        (["--study", path, "frobnicate"], "UsageError", "invalid choice"),
        (["--study", path], "UsageError", "no command"),
        (["status"], "UsageError", "--study"),
        (["--study", str(tmp_path / "none.json"), "status"], "LenzError", "not found"),
        (["--study", path, "create", "--space", json.dumps(SPACE), "--objectives", "f"], "LenzError", "already exists"),
        (["--study", str(tmp_path / "n.json"), "create", "--space", "[]", "--objectives", "f"], "LenzError", "parameter"),
        (["--study", path, "predict", "--configs", "@" + str(tmp_path / "missing.json")], "LenzError", "cannot read"),
    ]
    for argv, etype, needle in cases:
        code, env = run(capsys, *argv)
        assert code == 1 and env["ok"] is False, argv
        assert env["error"]["type"] == etype and needle in env["error"]["message"], (argv, env)
    assert not (tmp_path / "n.json").exists()
    # overwrite replaces the study
    code, env = run(capsys, "--study", path, "create", "--space", json.dumps(SPACE[:1]), "--objectives",
                    '{"metric": "g", "minimize": true}', "--overwrite")
    assert code == 0 and env["result"]["config"]["objectives"][0] == {"metric": "g", "minimize": True, "ref": None}


def test_help_is_not_an_error(capsys):
    assert main(["--help"]) == 0
    assert "suggest" in capsys.readouterr().out
    assert {"create", "set-bounds", "observe", "pareto"} <= set(
        build_parser()._subparsers._group_actions[0].choices)  # type: ignore[union-attr]


def _cli(*argv: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", "agentic_bo.lenz", *argv], capture_output=True, text=True,
                          env=env, timeout=120)


def test_subprocess_round_trip(tmp_path):
    path = str(tmp_path / "study.json")
    p = _cli("--study", path, "create", "--space", json.dumps(SPACE), "--objectives", "f",
             "--constraints", '{"c": [null, 1.0]}', "--seed", "1")
    assert p.returncode == 0, p.stderr
    for _ in range(3):
        p = _cli("--study", path, "suggest")
        assert p.returncode == 0 and len(p.stdout.splitlines()) == 1
        cfg = json.loads(p.stdout)["result"]["candidates"][0]["config"]
        p = _cli("--study", path, "submit", "--config", json.dumps(cfg))
        tid = json.loads(p.stdout)["result"]["trial_id"]
        p = _cli("--study", path, "observe", "--trial-id", str(tid),
                 "--metrics", json.dumps({"f": quad(cfg), "c": cfg["x0"] ** 2}))
        assert p.returncode == 0 and json.loads(p.stdout)["result"]["status"] == "completed"
    p = _cli("--study", path, "status")
    res = json.loads(p.stdout)
    assert p.returncode == 0 and res["ok"] and res["result"]["n_observed"] == 3
    # errors: exit code 1, one JSON line on stdout, no traceback
    p = _cli("--study", path, "observe", "--trial-id", "99", "--metrics", '{"f": 1, "c": 0}')
    assert p.returncode == 1 and json.loads(p.stdout)["ok"] is False and "Traceback" not in p.stderr
    p = _cli("--study", path, "observe", "--metrics")
    assert p.returncode == 1 and json.loads(p.stdout)["error"]["type"] == "UsageError" and "Traceback" not in p.stderr


def test_trials_limit_flag_is_top_k_after_sorting(study, capsys):
    path = study
    for i, v in enumerate([3.0, 1.0, 2.0, 0.5]):
        code, _ = run(capsys, "--study", path, "observe", "--config", json.dumps({"x0": i * 0.1, "x1": 0.0}),
                      "--metrics", json.dumps({"f": v}))
        assert code == 0
    code, env = run(capsys, "--study", path, "trials", "--sort-by", "value", "--limit", "2")
    assert code == 0 and [t["value"] for t in env["result"]["trials"]] == [3.0, 2.0]
    code, env = run(capsys, "--study", path, "trials", "--sort-by", "value", "--last", "2")
    assert code == 0 and [t["value"] for t in env["result"]["trials"]] == [2.0, 0.5]
    code, env = run(capsys, "--study", path, "trials", "--limit", "two")
    assert code == 1 and not env["ok"] and "--limit" in env["error"]["message"]
    code, env = run(capsys, "--study", path, "submit", "--config", json.dumps({"x0": 1.0, "x1": 1.0}))
    assert code == 0
    code, env = run(capsys, "--study", path, "trials", "--status", "pending,completed", "--limit", "10")
    assert code == 0 and env["result"]["n"] == 5, env
    code, env = run(capsys, "--study", path, "trials", "--status", "pending")
    assert code == 0 and [t["trial_id"] for t in env["result"]["trials"]] == [4]
