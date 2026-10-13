"""A small benchmark: the scripted Sara agent vs standard BO (FixedBO) vs Sobol search.

Runs every problem x policy x seed on randomized problem instances (anonymous
parameter names, shifted optimum), prints median / IQR final regret with two-sided
Mann-Whitney U comparisons against FixedBO, and writes regret plots (median and
25-75 % band per policy, one panel per problem).

    python examples/benchmark.py                                   # 3 problems x 3 policies x 5 seeds
    python examples/benchmark.py --problems branin --seeds 2 --budget 15   # a quick smoke run
    python examples/benchmark.py --problems hartmann6 ackley10 --seeds 10 --n-jobs 8

Runs are stored one JSON file each under ``--out``; re-running the same command
resumes (finished runs are loaded, not repeated). The same is available as
``python -m agentic_bo.experiment run ...`` and ``... report --results DIR``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import agentic_bo  # noqa: F401
except ImportError:  # running from a source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_bo.experiment import format_report, plot_regret, run_benchmark, summarize


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark heuristic Sara vs FixedBO vs Sobol.")
    parser.add_argument("--problems", nargs="+", default=["branin", "hartmann3", "six_hump_camel"])
    parser.add_argument("--policies", nargs="+", default=["heuristic", "fixed_bo", "sobol"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=None, help="default: each problem's own budget")
    parser.add_argument("--n-jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--regret", default="simple", choices=("simple", "cumulative"))
    parser.add_argument("--out", default="results/benchmark")
    args = parser.parse_args()

    rows = run_benchmark(args.problems, args.policies, seeds=args.seeds, out_dir=args.out, budget=args.budget,
                         n_jobs=args.n_jobs, regret=args.regret, verbose=True)
    ok = [r for r in rows if not r.get("error")]
    print()
    print(format_report(summarize(ok, reference="fixed_bo")))
    if not ok:
        return 1
    try:
        simple = plot_regret(ok, Path(args.out) / "regret_simple.png", kind="simple", log_scale=True,
                             title="Simple regret (median, 25-75 % band)")
        cumulative = plot_regret(ok, Path(args.out) / "regret_cumulative.png", kind="cumulative", log_scale=False,
                                 title="Cumulative regret (median, 25-75 % band)")
        print(f"\nplots: {simple}\n       {cumulative}")
    except Exception as exc:  # noqa: BLE001 - e.g. matplotlib is not installed
        print(f"\nplots skipped: {exc}")
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
