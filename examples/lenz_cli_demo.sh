#!/usr/bin/env bash
# A walk through the lenz command-line interface on a 2-D function (Branin, minimized).
#
# lenz is the Bayesian backend Sara drives; here a shell script plays the agent and a
# python one-liner plays the black box. Every lenz call prints one JSON envelope
# {"ok", "command", "result"} and the study file is the single source of truth: each call
# loads it, runs one command and saves it back.
#
#   bash examples/lenz_cli_demo.sh            # N=10 model-guided evaluations after the opening
#   N=20 KEEP=1 bash examples/lenz_cli_demo.sh
#   PYTHON=/path/to/python bash examples/lenz_cli_demo.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${PYTHON:-python3}"
N="${N:-10}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/lenz-demo.XXXXXX")"
STUDY="$WORK/study.json"
if [[ -z "${KEEP:-}" ]]; then trap 'rm -rf "$WORK"' EXIT; fi

# one lenz CLI call; prints the JSON envelope (the CLI exits 1 on an error envelope)
lenz() { "$PY" -m agentic_bo.lenz --study "$STUDY" "$@"; }
# jget KEY... : read JSON on stdin, walk the keys (digits index lists), print the value as JSON
jget() { "$PY" -c 'import functools,json,sys; print(json.dumps(functools.reduce(lambda v,k: v[int(k)] if k.isdigit() else v[k], sys.argv[1:], json.load(sys.stdin))))' "$@"; }
# the black box: Branin on x1 in [-5, 10], x2 in [0, 15]; minimum 0.397887 (three global minima)
oracle() { "$PY" -c 'import json,math,sys; c=json.loads(sys.argv[1]); x1,x2=c["x1"],c["x2"]; print(json.dumps({"f": (x2-5.1/(4*math.pi**2)*x1**2+5/math.pi*x1-6)**2+10*(1-1/(8*math.pi))*math.cos(x1)+10}))' "$1"; }
say() { printf '\n== %s\n' "$*"; }

say "create a study: 2 real parameters, one minimized objective 'f' (study file: $STUDY)"
lenz create \
  --space '[{"name": "x1", "type": "real", "lower": -5, "upper": 10},
            {"name": "x2", "type": "real", "lower": 0, "upper": 15}]' \
  --objectives '{"metric": "f", "minimize": true}' --seed 0 | jget result config

# One evaluation: suggest -> submit (pending trial) -> run the black box -> observe.
evaluate() {  # evaluate SUGGEST_ARGS...
  local sug cfg mode tid metrics
  sug="$(lenz suggest "$@")"
  cfg="$(jget result candidates 0 config <<<"$sug")"
  mode="$(jget result mode <<<"$sug")"
  tid="$(lenz submit --config "$cfg" | jget result trial_id)"
  metrics="$(oracle "$cfg")"
  lenz observe --trial-id "$tid" --metrics "$metrics" >/dev/null
  printf 'trial %-2s %-15s config=%s -> %s\n' "$tid" "$mode" "$cfg" "$metrics"
}

say "opening: with fewer than 2 observations suggest returns space-filling (Sobol) points"
for _ in 1 2 3; do evaluate; done

say "model-guided loop: logEI (the default acquisition) on a GP fitted to every observation"
for _ in $(seq "$N"); do evaluate; done

say "incumbent (best observed) and best posterior mean"
lenz incumbent | jget result

say "diagnostics: leave-one-out CV R^2, lengthscales and sensitivities of the GP"
lenz diagnostics | jget result models f

say "reconfigure: UCB with a small beta, proposals in a trust region around the incumbent"
lenz set-acqf --acqf ucb --beta 1.0 | jget result
lenz suggest --q 2 --around incumbent --radius 0.1 | jget result candidates

say "probe without spending budget: posterior predictions and acquisition scores at chosen configs"
lenz predict --configs '[{"x1": 3.14159, "x2": 2.275}, {"x1": -3.14159, "x2": 12.275}]' | jget result predictions
lenz score --configs '[{"x1": 3.14159, "x2": 2.275}, {"x1": 9.42478, "x2": 2.475}]' --acqf ucb,pm,std | jget result scores

say "restrict the active region (nothing is dropped: all trials stay in the log and in the model)"
lenz set-bounds --bounds '{"x1": [0, 10], "x2": [0, 6]}' | jget result
evaluate
lenz set-bounds --bounds full | jget result bounds

say "the three best trials (sorted best first) and the study status"
lenz trials --sort-by value --limit 3 | "$PY" -c 'import json,sys; [print("trial", t["trial_id"], t["config"], t["metrics"]) for t in json.load(sys.stdin)["result"]["trials"]]'
lenz status | "$PY" -c 'import json,sys; r=json.load(sys.stdin)["result"]; print({k: r[k] for k in ("n_observed", "n_pending", "n_trials", "feasible_found")})'

say "error envelopes: invalid input never crashes the CLI (exit code 1, one JSON line)"
lenz submit --config '{"x1": 42, "x2": 0}' || true

if [[ -n "${KEEP:-}" ]]; then say "study kept at $STUDY"; fi
