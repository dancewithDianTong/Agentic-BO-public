"""Quickstart: optimize a custom 3-D black-box function with the agentic BO loop.

The function is a toy "validation loss" of a training run with three settings: a
learning rate on a log scale, a momentum and an integer number of layers. Its
minimum (0.1 at lr=1e-2, momentum=0.9, 3 layers) is known here only so the script
can report the regret; the optimizer never sees it.

    python examples/quickstart.py            # the scripted Sara agent (no API key needed)
    python examples/quickstart.py --policy fixed_bo --budget 40

Swap ``--policy sara`` in to let Claude drive the same loop (needs ``anthropic`` and
credentials, see examples/run_sara_claude.py).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    import agentic_bo  # noqa: F401
except ImportError:  # running from a source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_bo import optimize

F_STAR = 0.1   # the known optimum, used only to report regret


def validation_loss(cfg: dict) -> float:
    """The black box: lower is better."""
    u = math.log10(cfg["learning_rate"]) + 2.0           # 0 at lr = 1e-2
    return (F_STAR + 0.3 * u ** 2 + 0.05 * math.sin(5.0 * u) ** 2
            + 2.0 * (cfg["momentum"] - 0.9) ** 2 + 0.05 * (cfg["layers"] - 3) ** 2)


SPACE = {
    "learning_rate": {"type": "real", "lower": 1e-4, "upper": 1.0, "log_scale": True},
    "momentum": (0.0, 0.99),                              # a (lower, upper) pair is a real parameter
    "layers": {"type": "int", "lower": 1, "upper": 8},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", default="heuristic", help="heuristic | fixed_bo | sobol | random | sara")
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true", help="no per-evaluation progress lines")
    args = parser.parse_args()

    result = optimize(
        validation_loss, SPACE, budget=args.budget, policy=args.policy, seed=args.seed, f_star=F_STAR,
        description="Validation loss of a small neural network after a fixed training schedule; lower is better.",
        verbose=not args.quiet,
    )

    print()
    print(f"best config : {result.best_config}")
    print(f"best value  : {result.best_value:.6g}   (optimum {F_STAR})")
    print(f"simple regret after {result.n_evals} evaluations: {result.final_simple_regret:.3g}")
    print(f"cumulative regret: {result.final_cumulative_regret:.3g}")
    print(f"agent actions: {result.action_counts}  (forced commits: {result.forced_commits})")
    if result.notebook:
        print(f"last notebook entry: {result.notebook[-1]['text'][:300]}")


if __name__ == "__main__":
    main()
