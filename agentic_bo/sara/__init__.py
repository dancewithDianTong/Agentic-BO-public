"""Sara: the agentic harness of agentic BO (DESIGN.md §9).

An agent (an LLM, a scripted policy or a classical baseline) drives the lenz backend
through a fixed set of tools; the harness runs the metalevel loop, one evaluation per
step, and scores the campaign by simple or cumulative regret.

    >>> from agentic_bo.problems import get_problem
    >>> from agentic_bo.sara import Campaign, CampaignConfig
    >>> from agentic_bo.sara.policies import FixedBOPolicy
    >>> result = Campaign(get_problem("branin"), FixedBOPolicy(), CampaignConfig(budget=20, seed=0)).run()
    >>> result.simple_regret[-1] >= 0
    True
"""

from .policies import FixedBOPolicy, Policy, PolicyError, RandomPolicy, SobolPolicy, get_policy
from .schema import validate
from .session import AgentSession, Campaign, CampaignConfig, CampaignResult, ContextItem, compute_regret_curves
from .tools import TOOLSETS, ToolSpec, build_tools

__all__ = [
    "AgentSession", "Campaign", "CampaignConfig", "CampaignResult", "ContextItem", "compute_regret_curves",
    "ToolSpec", "TOOLSETS", "build_tools", "validate",
    "Policy", "PolicyError", "SobolPolicy", "RandomPolicy", "FixedBOPolicy", "get_policy",
]
