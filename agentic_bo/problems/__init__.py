"""Benchmark problems: classic synthetic functions, a constrained and a multi-objective
variant, and a reaction-yield toy problem with a natural-language prior.

    >>> from agentic_bo.problems import get_problem
    >>> p = get_problem("hartmann6", seed=3, randomize=True)
    >>> p.space.names[:2], p.f_star          # anonymized params, unchanged optimum
    >>> p.regret(p.evaluate(p.x_star))       # ~0

Registered names: the fixed problems in ``list_problems()`` plus the dimension-suffixed
families ``ackley{d}``, ``levy{d}``, ``rosenbrock{d}`` (d >= 2), ``styblinski_tang{d}``,
``rastrigin{d}`` and ``griewank{d}`` for any ``1 <= d <= 1000``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from ..utils import LenzError
from .applied import make_reaction_yield
from .base import CanonicalMap, Problem, VectorizedEvaluator, randomize_problem
from .synthetic import FAMILIES, FAMILY_PATTERN, FIXED_BUILDERS, make_family_problem

__all__ = ["Problem", "get_problem", "list_problems", "randomize_problem", "VectorizedEvaluator",
           "CanonicalMap"]

_FIXED: dict[str, Callable[[], Problem]] = {**FIXED_BUILDERS, "reaction_yield": make_reaction_yield}
_LISTED_DIMS = (2, 5, 10, 20)
_KWARGS = ("shift", "permute", "default_budget")


def _canonical_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise LenzError(f"problem name must be a non-empty string, got {name!r}")
    return name.strip().lower().replace("-", "_")


def _build(name: str) -> Problem:
    if name in _FIXED:
        return _FIXED[name]()
    m = FAMILY_PATTERN.match(name)
    if m:
        return make_family_problem(m.group(1), int(m.group(2)))
    raise LenzError(
        f"unknown problem {name!r}; known: {sorted(_FIXED)} and the families "
        f"{', '.join(f + '{d}' for f in sorted(FAMILIES))} (e.g. 'ackley10')")


def get_problem(name: str, seed: int | None = None, randomize: bool = False,
                noise_std: float = 0.0, **kwargs: Any) -> Problem:
    """Build a registered problem.

    Args:
        name: registry name (case-insensitive, ``-`` accepted for ``_``), e.g.
            ``"branin"``, ``"ackley10"``, ``"reaction_yield"``.
        seed: randomization seed; the result is a deterministic function of
            ``(name, seed)``. Ignored unless ``randomize``; defaults to 0 then.
        randomize: anonymize the problem (canonical ``[0, 1]^d`` box, random parameter
            names and order, optimum shift for shift-safe functions). The problem keeps
            its registry ``name`` for reporting.
        noise_std: standard deviation of the Gaussian observation noise.
        **kwargs: ``shift`` (bool; default: shift iff the problem is shift-safe),
            ``permute`` (bool, default True), ``default_budget`` (int override).

    Raises:
        LenzError: unknown name / dimension, unknown keyword, or an invalid option.
    """
    unknown = sorted(set(kwargs) - set(_KWARGS))
    if unknown:
        raise LenzError(f"unknown get_problem options {unknown}; allowed: {list(_KWARGS)}")
    problem = _build(_canonical_name(name))
    if randomize:
        problem = randomize_problem(problem, 0 if seed is None else int(seed),
                                    shift=kwargs.get("shift"), permute=bool(kwargs.get("permute", True)))
    elif kwargs.get("shift"):
        raise LenzError("shift=True requires randomize=True")
    changes: dict[str, Any] = {}
    if noise_std:
        changes["noise_std"] = float(noise_std)
    if kwargs.get("default_budget") is not None:
        changes["default_budget"] = int(kwargs["default_budget"])
    return replace(problem, **changes) if changes else problem


def list_problems() -> list[str]:
    """Registered fixed problems plus the families at common dimensions (2, 5, 10, 20), sorted."""
    names = set(_FIXED)
    for fam in FAMILIES:
        names.update(f"{fam}{d}" for d in _LISTED_DIMS)
    return sorted(names)
