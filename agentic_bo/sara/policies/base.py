"""The policy interface (DESIGN.md §9.3): the metalevel decision maker of a campaign.

A policy implements one campaign step, ``Rollout(A, s_t)``: it may take any number of
free computational actions (probe / reconfigure / propose / note) and must then commit
exactly one evaluation (or stop, when the campaign allows it). Every policy, scripted or
LLM-driven, acts **only** through :meth:`AgentSession.call` -- the same tool interface
an LLM sees -- so traces of different policies are directly comparable.

Policies must not read harness-internal ground truth (``session.records`` holds the
noise-free metrics and regret used for scoring); what they may use is the tool results,
``session.context`` (the exogenous context ``K_t``), ``session.brief()``,
``session.status_line()`` and the campaign settings in ``session.config``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - imports for type checkers only
    from ..session import AgentSession, ContextItem

__all__ = ["Policy", "PolicyError"]


class PolicyError(RuntimeError):
    """A policy cannot continue (e.g. a missing API key, a refusal, an exhausted retry budget).

    :class:`~agentic_bo.sara.session.Campaign` records the message and, by default,
    finishes the remaining steps with harness fallback commits
    (``CampaignConfig.on_policy_error = "fallback"``). Any other ``Exception`` escaping a
    policy hook (a bug) is handled the same way, so a failing policy never loses the
    campaign; ``BaseException`` (``KeyboardInterrupt``, abort signals) propagates.
    """


class Policy(ABC):
    """Base class of campaign policies.

    Attributes:
        name: short identifier used in results and reports.
        toolset: tool set this policy needs (``"full"`` or ``"llm_only"``); ``None``
            means "whatever the campaign config says". When set, :class:`Campaign`
            builds the session with this tool set.
    """

    name: str = "policy"
    toolset: str | None = None

    def start(self, session: "AgentSession") -> None:
        """Called once before the first step (read the brief, set up state)."""

    @abstractmethod
    def step(self, session: "AgentSession") -> None:
        """Run one campaign step; must end with ``session.call("evaluate" | "stop", ...)``.

        If the step returns without committing, the harness commits a fallback
        evaluation (``session.commit_fallback``) and marks it ``forced``.
        """

    def on_context(self, session: "AgentSession", item: "ContextItem") -> None:
        """Called when exogenous context (an instruction, new information) arrives mid-run."""

    def finish(self, session: "AgentSession") -> None:
        """Called once after the last step (also after an early stop)."""

    def usage(self) -> dict[str, Any]:
        """Resource usage to report (API calls, tokens, cost); empty for scripted policies."""
        return {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
