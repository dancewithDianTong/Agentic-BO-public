"""Campaign policies: the classical baselines, the scripted Sara agent and the Claude agent.

    >>> from agentic_bo.sara.policies import get_policy
    >>> get_policy("fixed_bo", n_init=5)
    FixedBOPolicy(name='fixed_bo')

``heuristic`` and the Claude policies live in their own modules and are imported only
when requested (the Claude policy needs the optional ``anthropic`` package).
"""

from __future__ import annotations

from typing import Any

from ...utils import LenzError
from .base import Policy, PolicyError
from .baselines import FixedBOPolicy, RandomPolicy, SobolPolicy

__all__ = ["Policy", "PolicyError", "SobolPolicy", "RandomPolicy", "FixedBOPolicy", "get_policy",
           "POLICY_NAMES"]

#: names accepted by :func:`get_policy`
POLICY_NAMES = ("sobol", "random", "fixed_bo", "heuristic", "sara", "llm_only")
_ALIASES = {"bo": "fixed_bo", "standard_bo": "fixed_bo", "pi0": "fixed_bo", "pi_0": "fixed_bo",
            "claude": "sara", "llm": "sara", "heuristic_sara": "heuristic"}


def get_policy(name: str, **kwargs: Any) -> Policy:
    """Build a policy by name.

    Names: ``sobol``, ``random``, ``fixed_bo`` (standard BO ``Pi_0``), ``heuristic``
    (the scripted Sara agent), ``sara`` (Claude with the full tool set) and ``llm_only``
    (Claude as a direct optimizer, ``toolset="llm_only"``). ``kwargs`` go to the
    policy's constructor. Matching is case-insensitive and ``-`` is accepted for ``_``.

    Raises:
        LenzError: unknown name.
    """
    if not isinstance(name, str) or not name.strip():
        raise LenzError(f"policy name must be a non-empty string, got {name!r}")
    key = name.strip().lower().replace("-", "_")
    key = _ALIASES.get(key, key)
    if key == "sobol":
        return SobolPolicy(**kwargs)
    if key == "random":
        return RandomPolicy(**kwargs)
    if key == "fixed_bo":
        return FixedBOPolicy(**kwargs)
    if key == "heuristic":
        from .heuristic import HeuristicSaraPolicy
        return HeuristicSaraPolicy(**kwargs)
    if key in ("sara", "llm_only"):
        from .llm import ClaudeSaraPolicy
        if key == "llm_only":
            kwargs = {**kwargs, "toolset": "llm_only"}
        return ClaudeSaraPolicy(**kwargs)
    raise LenzError(f"unknown policy {name!r}; choose one of {', '.join(POLICY_NAMES)}")
