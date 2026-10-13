"""Classical baseline policies (DESIGN.md §9.3), acting through the same tools as Sara.

* :class:`SobolPolicy` -- a scrambled Sobol design: ``suggest(acqf="sobol")`` -> evaluate.
* :class:`RandomPolicy` -- uniform random search: ``suggest(acqf="random")`` -> evaluate.
* :class:`FixedBOPolicy` -- standard BO ``Pi_0`` (Ax-like): Sobol for ``n_init`` steps,
  then evaluate the top ``suggest`` candidate of a fixed acquisition function every
  step; it never probes or reconfigures.

Because they use :meth:`AgentSession.call` like an LLM agent would, their traces are
directly comparable with Sara's. If a ``suggest`` call fails, the step returns without
evaluating and the harness commits its fallback (recorded as forced).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...lenz.acquisition import MULTI_OBJECTIVE_ACQFS, SINGLE_OBJECTIVE_ACQFS
from ...utils import LenzError
from .base import Policy

if TYPE_CHECKING:  # pragma: no cover
    from ..session import AgentSession

__all__ = ["SobolPolicy", "RandomPolicy", "FixedBOPolicy"]


def _top_candidate(session: "AgentSession", args: dict) -> tuple[dict, dict] | None:
    """``(candidate, suggest result)`` of the first suggestion, or ``None`` if suggest failed."""
    env = session.call("suggest", args)
    if not env.get("ok"):
        return None
    cands = env["result"].get("candidates") or []
    if not cands:
        return None
    return cands[0], env["result"]


class _SpaceFillingPolicy(Policy):
    """Evaluate one point of a model-free design per step."""

    acqf = "sobol"
    label = "Sobol"

    def step(self, session: "AgentSession") -> None:
        top = _top_candidate(session, {"acqf": self.acqf})
        if top is None:
            return
        cand, _ = top
        session.call("evaluate", {
            "config": cand["config"],
            "rationale": f"{self.label} baseline: space-filling design point {session.t + 1}/{session.budget}; "
                         f"no model is used.",
        })


class SobolPolicy(_SpaceFillingPolicy):
    """Scrambled Sobol design (continues the study's Sobol sequence, skipping repeats)."""

    name = "sobol"
    acqf = "sobol"
    label = "Sobol"


class RandomPolicy(_SpaceFillingPolicy):
    """Uniform random search (lenz's ``random`` design, seeded by the campaign seed)."""

    name = "random"
    acqf = "random"
    label = "Random search"


class FixedBOPolicy(Policy):
    """Standard BO ``Pi_0``: ``n_init`` Sobol points, then the top acquisition candidate every step.

    Args:
        n_init: number of initial Sobol evaluations (Ax uses 5 for small problems).
        acqf: acquisition function of the BO phase (default ``logei``). For a
            multi-objective problem a single-objective name falls back to the study's
            default (``ehvi`` for 2 objectives, ``parego`` beyond); ``None`` always uses
            the study's default.
    """

    name = "fixed_bo"

    def __init__(self, n_init: int = 5, acqf: str | None = "logei"):
        if isinstance(n_init, bool) or not isinstance(n_init, int) or n_init < 0:
            raise LenzError(f"n_init must be an integer >= 0, got {n_init!r}")
        if acqf is not None and acqf not in SINGLE_OBJECTIVE_ACQFS + MULTI_OBJECTIVE_ACQFS:
            raise LenzError(f"FixedBOPolicy acqf must be a model-based acquisition function "
                            f"({', '.join(SINGLE_OBJECTIVE_ACQFS + MULTI_OBJECTIVE_ACQFS)}), got {acqf!r}")
        self.n_init = n_init
        self.acqf = acqf
        self._acqf: str | None = acqf

    def start(self, session: "AgentSession") -> None:
        env = session.call("status")
        is_moo = bool(env.get("ok") and env["result"].get("is_moo"))
        acqf = self.acqf
        if acqf is not None and (acqf in SINGLE_OBJECTIVE_ACQFS) == is_moo:
            acqf = None   # wrong family for this study: use lenz's default for the objectives
        self._acqf = acqf

    def step(self, session: "AgentSession") -> None:
        if session.t < self.n_init:
            top = _top_candidate(session, {"acqf": "sobol"})
            if top is None:
                return
            cand, _ = top
            rationale = f"Initial Sobol design point {session.t + 1}/{self.n_init} (standard BO initialization)."
        else:
            args: dict[str, Any] = {"q": 1}
            if self._acqf is not None:
                args["acqf"] = self._acqf
            top = _top_candidate(session, args)
            if top is None:
                return
            cand, res = top
            name = res.get("acqf")
            value = (cand.get("acquisition_values") or {}).get(name)
            rationale = (f"Standard BO: top {name} candidate"
                         + (f" (acquisition value {value:.4g})" if isinstance(value, (int, float)) else "")
                         + "; fixed configuration, no reconfiguration.")
        session.call("evaluate", {"config": cand["config"], "rationale": rationale})
