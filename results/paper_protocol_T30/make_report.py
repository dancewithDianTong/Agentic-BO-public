"""Sara-Qwen at 30 evaluations vs the paper-protocol baselines truncated to their first 30 evaluations.

Run from the repo root:  python results/paper_protocol_T30/make_report.py
Baseline rows come from results/paper_protocol (full budgets) cut at evaluation 30.
Truncation is exact for fixed_bo, sobol and random (they do not adapt to the budget);
the scripted `heuristic` adapts its opening and end game to the total budget, so it is
rerun at T=30 (results/paper_protocol_T30/heuristic) instead of truncated.
"""
import copy, glob, json, os, sys
sys.path.insert(0, os.getcwd())
from agentic_bo.experiment import load_results, plot_regret, compare, summarize, format_report

T = 30
HERE = "results/paper_protocol_T30"
PROBS = ["branin", "hartmann6", "ackley10", "ackley20", "constrained_hartmann6", "levy10", "styblinski_tang8",
         "reaction_yield", "branin_currin"]
ORDER = ["qwen_sara", "heuristic", "fixed_bo", "sobol", "random"]

def truncate(row):
    r = copy.deepcopy(row)
    for k in ("simple_regret", "cumulative_regret"):
        r[k] = r[k][:T]
    r["final_simple_regret"], r["final_cumulative_regret"] = r["simple_regret"][-1], r["cumulative_regret"][-1]
    r["budget"] = r["n_evals"] = T
    return r

base = [truncate(r) for r in load_results("results/paper_protocol")
        if r["n_evals"] >= T and r["policy"] in ("fixed_bo", "sobol", "random")]
if os.path.isdir(f"{HERE}/heuristic/runs"):
    base += load_results(f"{HERE}/heuristic")
qwen = load_results(f"{HERE}/qwen") if os.path.isdir(f"{HERE}/qwen/runs") else []
for f in sorted(glob.glob("results/sara_qwen/branin_T30_seed*.json")):   # earlier T=30 branin runs, seeds 0-4
    for r in load_results(f):
        r["policy"] = "qwen_sara"; qwen.append(r)
rows = base + qwen
present = [p for p in ORDER if any(r["policy"] == p for r in rows)]
for kind in ("simple", "cumulative"):
    plot_regret(rows, f"{HERE}/regret_{kind}.png", kind=kind, problems=PROBS, policies=present,
                title=f"Budget {T}: Sara-Qwen (qwen3.8-27b) vs baselines")
report = format_report(summarize(rows, reference="fixed_bo"))
open(f"{HERE}/report.txt", "w").write(report)
q = [r for r in qwen]
health = {"qwen_runs": len(q), "forced_commits": sum(r["forced_commits"] for r in q),
          "policy_errors": sum(bool(r.get("policy_error")) for r in q)}
json.dump({"T": T, "health": health, "summary": summarize(rows), "compare": compare(rows, reference="fixed_bo")},
          open(f"{HERE}/summary.json", "w"), indent=1, default=str)
print(report); print(health)
