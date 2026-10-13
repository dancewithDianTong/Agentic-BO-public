"""Agentic Bayesian optimization (Sara agent + lenz backend), after arXiv:2608.00316.

    >>> from agentic_bo import optimize
    >>> result = optimize(lambda c: (c["x"] - 0.3) ** 2, {"x": (0, 1)}, budget=15, f_star=0.0)
    >>> result.best_config, result.simple_regret[-1]

Main entry points:

* :func:`optimize` -- optimize a Python function with the agentic BO loop;
* :class:`Lenz` -- the Bayesian backend (GP surrogates, acquisitions, trial log);
* :class:`Campaign` / :class:`CampaignConfig` / :func:`get_policy` -- the Sara harness
  and its policies (``heuristic``, ``fixed_bo``, ``sobol``, ``random``, ``sara``, ``llm_only``);
* :func:`get_problem` / :func:`list_problems` -- benchmark problems;
* :func:`run_campaign`, :func:`run_benchmark`, :func:`summarize`, :func:`compare`,
  :func:`plot_regret` -- experiments (also ``python -m agentic_bo.experiment``).

Everything except the search-space classes is imported lazily on first access, so
``import agentic_bo`` stays light and the command-line modules run cleanly with ``-m``.
"""

from __future__ import annotations

import importlib
from typing import Any

from .space import CategoricalParam, IntParam, OrdinalParam, RealParam, SearchSpace
from .utils import LenzError

__version__ = "0.1.0"

#: lazily exported name -> defining module
_LAZY = {
    "optimize": ".api", "make_problem": ".api",
    "Lenz": ".lenz", "GP": ".lenz",
    "Problem": ".problems", "get_problem": ".problems", "list_problems": ".problems",
    "Campaign": ".sara", "CampaignConfig": ".sara", "CampaignResult": ".sara", "AgentSession": ".sara",
    "ContextItem": ".sara", "Policy": ".sara", "PolicyError": ".sara", "get_policy": ".sara",
    "run_campaign": ".experiment", "run_benchmark": ".experiment", "summarize": ".experiment",
    "compare": ".experiment", "plot_regret": ".experiment", "load_results": ".experiment",
}

__all__ = ["SearchSpace", "RealParam", "IntParam", "OrdinalParam", "CategoricalParam", "LenzError",
           *_LAZY]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
