"""The Sara harness (DESIGN.md §9.2): agent sessions, campaigns and their results.

A **campaign** optimizes one :class:`~agentic_bo.problems.Problem` with one
:class:`~agentic_bo.sara.policies.Policy` under a budget of ``T`` evaluations. Each
*step* is the paper's ``Rollout(A, s_t)``: the policy takes ``k_t >= 0`` free
computational actions through :meth:`AgentSession.call` and then commits exactly one
evaluation (or stops, when allowed). The harness enforces the rules the agent cannot
break: one evaluation per step, the budget, soft / hard limits on actions per step
(a warning, then a forced fallback commit), early stopping only when allowed, and
schema validation of every tool call. No agent input can crash a campaign: every
exception inside :meth:`AgentSession.call` becomes an error envelope.

State ``s_t = (D_t, c_t, K_t, H_t)`` lives in: ``session.lenz`` (trial log ``D_t`` and
configuration ``c_t``), ``session.context`` (exogenous context ``K_t``, append-only)
and ``session.log`` (deliberation history ``H_t``). ``session.records`` holds one
entry per evaluation with the *noise-free* metrics and the regret used for scoring;
policies must not read it.

Regret is computed from noise-free values (``agentic_bo.metrics``): simple regret
``r_t = u* - max_{s<=t} u_s`` and cumulative regret ``R_t = sum_s (u* - u_s)`` for one
objective; for several objectives the "simple" curve is the hypervolume regret
``max_hv - HV_t`` and the "cumulative" curve its running sum. After an early stop the
simple curve keeps its last value and the cumulative curve charges every unused
evaluation the regret of the best point found.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..lenz import Lenz
from ..lenz.engine import DUPLICATE_TOL
from ..metrics import (
    cumulative_regret_curve, dominated_hypervolume, hypervolume_regret_curve, simple_regret_curve,
)
from ..problems.base import Problem
from ..utils import LenzError, dumps, jsonable, make_rng
from .policies.base import Policy, PolicyError
from .schema import format_errors, validate
from .tools import TOOLSETS, ToolSpec, build_tools, present

__all__ = ["ContextItem", "CampaignConfig", "AgentSession", "Campaign", "CampaignResult",
           "compute_regret_curves", "REGRET_MODES", "ON_POLICY_ERROR"]

REGRET_MODES = ("simple", "cumulative")
ON_POLICY_ERROR = ("fallback", "raise")
#: log entries keep agent-facing results up to this many JSON characters (else a preview)
LOG_RESULT_MAX_CHARS = 6000
_LOG_PREVIEW_CHARS = 600
_ERROR_MAX_CHARS = 2000
_SALT_NOISE = 0x5E55      # session stream for observation noise
_SALT_FALLBACK = 0xFA11   # session stream for the last-resort random fallback
RESULT_FORMAT = "agentic_bo.sara/campaign_result"


def _error_text(exc: BaseException) -> str:
    """Agent-facing error message: lenz / validation messages verbatim, other errors labelled."""
    if isinstance(exc, LenzError):
        msg = str(exc)
    else:
        msg = f"internal error ({type(exc).__name__}): {exc}"
    return msg if len(msg) <= _ERROR_MAX_CHARS else msg[: _ERROR_MAX_CHARS - 3] + "..."


def _bounded(obj: Any, max_chars: int = LOG_RESULT_MAX_CHARS) -> Any:
    """JSON-safe copy of ``obj``, replaced by a preview when it is longer than ``max_chars``."""
    try:
        j = jsonable(obj)
        text = json.dumps(j, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        j, text = repr(obj), repr(obj)
    if len(text) <= max_chars:
        return j
    return {"truncated": True, "chars": len(text), "preview": text[:_LOG_PREVIEW_CHARS]}


def _finite_or_none(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{float(v):.6g}"


def _check_int(v: Any, name: str, lo: int | None = None) -> int:
    """``v`` as an int (integral floats accepted, booleans not); ``LenzError`` otherwise."""
    ok = isinstance(v, (int, np.integer)) or (isinstance(v, (float, np.floating)) and float(v).is_integer())
    if isinstance(v, (bool, np.bool_)) or not ok:
        raise LenzError(f"{name} must be an integer, got {v!r}")
    if lo is not None and int(v) < lo:
        raise LenzError(f"{name} must be an integer >= {lo}, got {v!r}")
    return int(v)


# =========================================================================== context


@dataclass
class ContextItem:
    """One entry of the exogenous context ``K_t`` (append-only).

    Attributes:
        t: number of completed evaluations when the item entered the context.
        source: ``"problem"`` (the brief), ``"prior"``, ``"user"`` (initial context),
            ``"instructor"`` (mid-run instructions), ...
        text: natural-language content shown to LLM agents.
        data: optional structured payload for scripted agents, e.g.
            ``{"op": "set_constraints", "args": {...}}`` or ``{"prior_hint": {...}}``.
    """

    t: int
    source: str
    text: str
    data: dict | None = None

    def to_dict(self) -> dict:
        return jsonable({"t": self.t, "source": self.source, "text": self.text, "data": self.data})

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ContextItem":
        return cls(t=int(d.get("t", 0)), source=str(d.get("source", "instructor")), text=str(d.get("text", "")),
                   data=None if d.get("data") is None else dict(d["data"]))


def _context_entries(value: Any, where: str) -> list[dict]:
    """Normalize a context spec (str | dict | ContextItem | list of those) to ``[{"source", "text", "data"}]``."""
    if isinstance(value, (list, tuple)):
        out: list[dict] = []
        for i, v in enumerate(value):
            out.extend(_context_entries(v, f"{where}[{i}]"))
        return out
    if isinstance(value, ContextItem):
        return [{"source": value.source, "text": value.text, "data": value.data}]
    if isinstance(value, str):
        if not value.strip():
            raise LenzError(f"{where}: context text must not be empty")
        return [{"source": None, "text": value, "data": None}]
    if isinstance(value, Mapping):
        d = dict(value)
        source = d.pop("source", None)
        text = d.pop("text", None)
        data = d.pop("data", None)
        if d:  # remaining keys (e.g. op / args) form the structured payload
            if data is not None:
                raise LenzError(f"{where}: give structured fields either inside 'data' or at the top level, "
                                f"not both (extra keys {sorted(d)})")
            data = d
        if data is not None and not isinstance(data, Mapping):
            raise LenzError(f"{where}: 'data' must be an object, got {data!r}")
        if text is None:
            if data is None:
                raise LenzError(f"{where}: a context item needs 'text' (and optionally 'data')")
            text = "Instruction: " + json.dumps(jsonable(data), sort_keys=True)
        if not isinstance(text, str) or not text.strip():
            raise LenzError(f"{where}: 'text' must be a non-empty string")
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise LenzError(f"{where}: 'source' must be a non-empty string")
        return [{"source": source, "text": text, "data": None if data is None else jsonable(dict(data))}]
    raise LenzError(f"{where}: context must be a string, an object {{text, source?, data?}} or a list, "
                    f"got {type(value).__name__}")


# =========================================================================== config


@dataclass
class CampaignConfig:
    """Settings of one campaign (DESIGN §9.2).

    Extensions beyond DESIGN: ``context`` (initial user-provided context: a string,
    a ``{text, source?, data?}`` object or a list of them; part of the brief) and
    ``on_policy_error`` (``"fallback"``: record an exception raised by the policy -- a
    :class:`PolicyError` or any other ``Exception`` -- as ``policy_error`` and finish with
    harness fallback commits; ``"raise"``: re-raise it).

    ``context_schedule`` maps ``t`` (number of completed evaluations) to context that is
    injected before step ``t + 1``; values are strings, ``{text, source?, data?}``
    objects (other top-level keys such as ``op`` / ``args`` become ``data``) or lists.
    """

    budget: int | None = None
    regret: str = "simple"
    allow_early_stop: bool = False
    soft_actions_per_step: int = 12
    hard_actions_per_step: int = 30
    max_evals_per_step: int = 1
    context_schedule: dict = field(default_factory=dict)
    agent_setup: bool = False
    toolset: str = "full"
    seed: int = 0
    lenz_options: dict = field(default_factory=dict)
    show_prior: bool = True
    context: Any = None
    on_policy_error: str = "fallback"

    def __post_init__(self) -> None:
        if self.budget is not None:
            self.budget = _check_int(self.budget, "budget", 1)
        if self.regret not in REGRET_MODES:
            raise LenzError(f"regret must be one of {REGRET_MODES}, got {self.regret!r}")
        if self.toolset not in TOOLSETS:
            raise LenzError(f"toolset must be one of {sorted(TOOLSETS)}, got {self.toolset!r}")
        if self.on_policy_error not in ON_POLICY_ERROR:
            raise LenzError(f"on_policy_error must be one of {ON_POLICY_ERROR}, got {self.on_policy_error!r}")
        for name in ("soft_actions_per_step", "hard_actions_per_step", "max_evals_per_step"):
            setattr(self, name, _check_int(getattr(self, name), name, 1))
        self.seed = _check_int(self.seed, "seed")
        self.allow_early_stop = bool(self.allow_early_stop)
        self.agent_setup = bool(self.agent_setup)
        self.show_prior = bool(self.show_prior)
        if not isinstance(self.lenz_options, Mapping):
            raise LenzError(f"lenz_options must be an object, got {self.lenz_options!r}")
        self.lenz_options = dict(self.lenz_options)
        if not isinstance(self.context_schedule, Mapping):
            raise LenzError(f"context_schedule must map t -> context, got {self.context_schedule!r}")
        schedule: dict[int, list[dict]] = {}
        for k, v in self.context_schedule.items():
            try:
                t = int(k)
                if isinstance(k, bool) or float(k) != t:
                    raise ValueError
            except (TypeError, ValueError):
                raise LenzError(f"context_schedule keys must be integers t >= 0, got {k!r}") from None
            if t < 0:
                raise LenzError(f"context_schedule keys must be >= 0, got {t}")
            schedule.setdefault(t, []).extend(_context_entries(v, f"context_schedule[{t}]"))
        self.context_schedule = dict(sorted(schedule.items()))
        if self.context is not None:
            self.context = _context_entries(self.context, "context")

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        d["context_schedule"] = {str(k): v for k, v in self.context_schedule.items()}
        return jsonable(d)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CampaignConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise LenzError(f"unknown CampaignConfig fields {unknown}")
        return cls(**dict(d))


# =========================================================================== session


class AgentSession:
    """The agent's view of one campaign: tools, budget, context, notebook and history.

    Create one per campaign (``Campaign.run`` does). All agent actions go through
    :meth:`call`; harness-level operations (:meth:`begin_step`, :meth:`add_context`,
    :meth:`commit_fallback`) are driven by :class:`Campaign`.

    Attributes:
        problem, lenz, config, tools: the problem, the lenz study, the settings and the
            agent's tools (sorted by name).
        toolset: ``"full"`` or ``"llm_only"``.
        t: completed evaluations; budget / remaining: ``T`` and ``T - t``.
        step: current step number (1-based; 0 before the first step).
        context: ``K_t`` (list of :class:`ContextItem`, append-only).
        notebook: ``record_note`` entries ``{"t", "step", "text"}``.
        log: ``H_t``; one entry per tool call ``{"t", "step", "j", "tool", "action_class",
            "args", "ok", "result" | "error", "forced", "duration_s"}`` plus reasoning
            entries added with :meth:`log_reasoning`.
        records: one entry per evaluation ``{"t" (1-based evaluation index), "step",
            "trial_id", "config", "observed" (noisy metrics the agent saw), "true_metrics"
            (noise-free), "utility", "feasible", "regret" (raw :meth:`Problem.regret` of the
            noise-free values; negative for an infeasible point that beats the constrained
            optimum, which the regret curves floor at 0),
            "source", "rationale", "expectation", "forced", "in_region", "duplicate_of",
            "is_new_best", "error"}``. Ground truth for scoring: policies must not read it.
        step_committed, stopped, actions_this_step: step bookkeeping.
        actions_per_step: computational actions ``k_t`` taken before each step's commit.
    """

    def __init__(self, problem: Problem, config: CampaignConfig | None = None):
        if not isinstance(problem, Problem):
            raise LenzError(f"problem must be a Problem, got {type(problem).__name__}")
        self.problem = problem
        self.config = config if config is not None else CampaignConfig()
        self.toolset = self.config.toolset
        self.budget = int(self.config.budget if self.config.budget is not None else problem.default_budget)
        self.lenz = self._make_lenz(self.config.lenz_options)

        self.t = 0
        self.step = 0
        self.step_committed = False
        self.stopped = False
        self.stop_reason: str | None = None
        self.actions_this_step = 0
        self._carry = 0                 # actions after a commit, counted toward the next step
        self._evals_this_step = 0
        self.actions_per_step: list[int] = []

        self.context: list[ContextItem] = []
        self.notebook: list[dict] = []
        self.log: list[dict] = []
        self.records: list[dict] = []
        self._Z = np.zeros((0, problem.space.D))   # encoded configs of the records (duplicate notes)
        self._injected: set[int] = set()
        self._noise_rng = make_rng(self.config.seed, _SALT_NOISE)
        self._fallback_rng = make_rng(self.config.seed, _SALT_FALLBACK)

        self.tools: list[ToolSpec] = build_tools(self, self.toolset)
        self._tools = {s.name: s for s in self.tools}

        self.context.append(ContextItem(0, "problem", problem.brief()))
        if self.config.show_prior and problem.prior:
            data = {"prior_hint": problem.prior_hint} if problem.prior_hint else None
            self.context.append(ContextItem(0, "prior", problem.prior, jsonable(data)))
        for e in self.config.context or []:
            self.context.append(ContextItem(0, e["source"] or "user", e["text"], e["data"]))
        self._n_initial = len(self.context)

    def __repr__(self) -> str:
        return (f"AgentSession(problem={self.problem.name!r}, toolset={self.toolset!r}, step={self.step}, "
                f"t={self.t}/{self.budget}, stopped={self.stopped})")

    # ---- construction helpers -------------------------------------------------------------
    def _default_objectives(self) -> list[dict]:
        objs = self.problem.objective_list
        if self.problem.is_moo and self.problem.ref_point is not None:
            return [{**o, "ref": r} for o, r in zip(objs, self.problem.ref_point)]
        return objs

    def _make_lenz(self, options: Mapping[str, Any]) -> Lenz:
        opts = dict(options or {})
        objectives = opts.pop("objectives", None) or self._default_objectives()
        constraints = opts.pop("constraints", None)
        if constraints is None:
            constraints = self.problem.constraints
        seed = opts.pop("seed", self.config.seed)
        try:
            return Lenz(self.problem.space, objectives, constraints=constraints, seed=seed, **opts)
        except TypeError as exc:
            raise LenzError(f"invalid lenz_options {sorted(options)}: {exc}") from None

    # ---- basic properties ------------------------------------------------------------------
    @property
    def remaining(self) -> int:
        """Evaluations left in the budget."""
        return max(self.budget - self.t, 0)

    @property
    def is_done(self) -> bool:
        """True once the budget is used up or the campaign was stopped."""
        return self.stopped or self.t >= self.budget

    def known_metrics(self) -> list[str]:
        """Metric names the problem reports (objectives, constraints, other outputs, observed extras)."""
        names = list(self.problem.metric_names)
        for r in self.records:
            names.extend(r["observed"])
        return list(dict.fromkeys(names))

    def tool_definitions(self) -> list[dict]:
        """Anthropic tool dicts ``{"name", "description", "input_schema"}``, sorted by name."""
        return [s.definition() for s in self.tools]

    # ---- briefing ---------------------------------------------------------------------------
    def brief(self) -> str:
        """Initial agent brief: problem brief, budget and goal, prior knowledge, initial context."""
        initial = self.context[: self._n_initial]
        parts = [c.text.strip() for c in initial if c.source == "problem"]
        parts.append(self._goal_section())
        prior = [c.text.strip() for c in initial if c.source == "prior"]
        if prior:
            parts.append("## Prior knowledge\n" + "\n\n".join(prior))
        extra = [c.text.strip() for c in initial if c.source not in ("problem", "prior")]
        if extra:
            parts.append("## Additional context\n" + "\n\n".join(extra))
        return "\n\n".join(p for p in parts if p)

    def _goal_section(self) -> str:
        T = self.budget
        moo = self.problem.is_moo
        free = ("every other action is free" if self.toolset == "full"
                else "looking at the trial log and the incumbent is free")
        lines = [f"## Budget and goal",
                 f"You have a budget of {T} evaluations of the black-box function, made one at a time "
                 f"(one per step); {free}."]
        quality = "the hypervolume of the feasible evaluated points" if moo else "the best feasible value found"
        if self.config.regret == "simple":
            lines.append(
                f"Your result is judged by simple regret: {quality} at the end of the campaign. Evaluations "
                f"made to learn about the function cost nothing beyond their budget slot, so exploring is "
                f"affordable as long as you converge on a strong result before the budget runs out.")
        else:
            what = "the hypervolume shortfall after each evaluation" if moo else \
                "the gap between the optimum and the value obtained"
            lines.append(
                f"Your result is judged by cumulative regret: every evaluation counts, as the sum over all "
                f"evaluations of {what}. Poor evaluations are charged even if you later find an excellent "
                f"point, so explore only where it is likely to pay off and exploit good regions early.")
        if self.config.allow_early_stop:
            if self.config.regret == "simple":
                lines.append("You may end the campaign early with `stop`; the result is then the best point "
                             "found so far.")
            else:
                lines.append("You may end the campaign early with `stop`; every unused evaluation is then "
                             "charged as if it repeated your best point.")
        else:
            lines.append("The campaign runs until the whole budget is used.")
        return "\n".join(lines)

    def status_line(self) -> str:
        """Compact progress line, e.g. ``"Step 7/50 | remaining 44 | best y=0.41 (trial 5)"`` (observed values)."""
        if self.is_done:
            head = f"Campaign finished: {self.t}/{self.budget} evaluations used"
        else:
            head = f"Step {self.t + 1}/{self.budget} | remaining {self.remaining}"
        return f"{head} | {self._best_summary()}"

    def _best_summary(self) -> str:
        if not self.records:
            return "no evaluations yet"
        p = self.problem
        feas = [r for r in self.records if r["error"] is None and p.is_feasible(r["observed"])]
        if p.is_moo:
            objs = p.objective_list
            Y = np.array([[_finite_or_none(r["observed"].get(o["metric"])) for o in objs] for r in feas],
                         dtype=float).reshape(-1, len(objs))
            if p.ref_point is None:
                return f"{len(feas)} feasible evaluation(s)"
            hv = dominated_hypervolume(Y, p.ref_point, [o["minimize"] for o in objs])
            return f"{len(feas)} feasible evaluation(s), hypervolume {_fmt(hv)}"
        best = None
        for r in feas:
            u = p.utility(r["observed"])
            if math.isfinite(u) and (best is None or u > best[0]):
                best = (u, r)
        if best is None:
            return "no feasible evaluation yet"
        r = best[1]
        return f"best {p.objective}={_fmt(r['observed'].get(p.objective))} (trial {r['trial_id']})"

    # ---- context ----------------------------------------------------------------------------
    def add_context(self, item: ContextItem | str | Mapping[str, Any]) -> ContextItem:
        """Append exogenous context to ``K_t`` (a :class:`ContextItem`, a string or a ``{text, ...}`` object)."""
        if not isinstance(item, ContextItem):
            e = _context_entries(item, "context")
            if len(e) != 1:
                raise LenzError("add_context takes one item at a time")
            item = ContextItem(self.t, e[0]["source"] or "instructor", e[0]["text"], e[0]["data"])
        self.context.append(item)
        return item

    def _take_due_context(self) -> list[ContextItem]:
        """Scheduled context due before the next step (keys ``<= t`` not yet injected)."""
        due = []
        for k, entries in self.config.context_schedule.items():
            if k <= self.t and k not in self._injected:
                self._injected.add(k)
                due.extend(ContextItem(self.t, e["source"] or "instructor", e["text"], e["data"])
                           for e in entries)
        return due

    # ---- steps ------------------------------------------------------------------------------
    def begin_step(self) -> None:
        """Open the next step (a no-op while the current step is still open).

        Actions taken after the previous step committed count toward the new step.
        """
        if self.step > 0 and not self.step_committed:
            return
        self.step += 1
        self.step_committed = False
        self._evals_this_step = 0
        self.actions_this_step = self._carry
        self._carry = 0

    def log_reasoning(self, text: str, kind: str = "thinking", **extra: Any) -> None:
        """Append a reasoning trace (e.g. an LLM's summarized thinking) to ``H_t``."""
        entry: dict[str, Any] = {"t": self.t, "step": max(self.step, 1), "j": None, "tool": None,
                                 "action_class": "reasoning", "kind": str(kind), "text": str(text), "ok": True,
                                 "forced": False}
        for k, v in extra.items():
            entry.setdefault(str(k), _bounded(v))
        self.log.append(entry)

    # ---- the agent interface ----------------------------------------------------------------
    def call(self, name: str, args: Mapping[str, Any] | None = None) -> dict:
        """Run one agent tool call; returns ``{"ok": True, "result"}`` or ``{"ok": False, "error"}``.

        Validates ``args`` against the tool's schema, enforces the per-step limits, logs
        the call in ``H_t`` and never raises. Results are compact (rounded floats,
        bounded size). Beyond ``soft_actions_per_step`` actions a ``warning`` is
        attached; beyond ``hard_actions_per_step`` the harness commits a fallback
        evaluation instead of running the call (any call, including unknown tools and
        malformed arguments); ``evaluate`` and ``stop`` still run there and the fallback
        follows only if they fail.
        """
        try:
            return self._call(name, args)
        except Exception as exc:  # last line of defence: the agent never crashes the campaign
            return {"ok": False, "error": _error_text(exc)}

    def _call(self, name: Any, args: Any) -> dict:
        started = time.perf_counter()
        tool_name = name.strip() if isinstance(name, str) else repr(name)
        spec = self._tools.get(tool_name) or self._tools.get(tool_name.lower().replace("-", "_"))
        if self.step == 0:
            self.begin_step()
        cfg = self.config
        is_eval = spec is not None and spec.name == "evaluate"
        post = self.step_committed and not (is_eval and self._evals_this_step < cfg.max_evals_per_step)
        if post:   # after this step's commit: the action counts toward the next step
            j, step_no = self._carry, self.step + 1
            self._carry += 1
            count = self._carry
        else:
            j, step_no = self.actions_this_step, self.step
            self.actions_this_step += 1
            count = self.actions_this_step
        entry: dict[str, Any] = {
            "t": self.t, "step": step_no, "j": j, "tool": spec.name if spec else tool_name,
            "action_class": spec.action_class if spec else "invalid", "args": _bounded(args),
            "ok": False, "forced": False,
        }

        # Beyond the hard limit every call of an open step ends in a commit: any call but evaluate /
        # stop is replaced by a fallback evaluation (also unknown tools and malformed arguments, so
        # an agent stuck on invalid calls cannot keep the step open); evaluate and stop still run
        # (they are the agent's own way to end the step), followed by the fallback only if they fail.
        ends_step = spec is not None and spec.name in ("evaluate", "stop")
        over_hard = (not post and count > cfg.hard_actions_per_step and not self.step_committed
                     and not self.is_done)
        if over_hard and not ends_step:
            return self._force_commit(entry, count, started)

        result: Any = None
        try:
            if spec is None:
                raise LenzError(self._unknown_tool_message(tool_name))
            if args is None:
                args = {}
            if not isinstance(args, Mapping):
                raise LenzError(f"arguments of {spec.name} must be a JSON object, got {type(args).__name__}")
            errors = validate(dict(args), spec.input_schema)
            if errors:
                raise LenzError(format_errors(errors, prefix=f"invalid arguments for {spec.name}: "))
            result = present(spec.handler(self, dict(args)), self.problem.space, metrics=self.known_metrics())
            envelope: dict[str, Any] = {"ok": True, "result": result}
        except Exception as exc:
            envelope = {"ok": False, "error": _error_text(exc)}

        if over_hard and not envelope["ok"] and not self.step_committed and not self.is_done:
            entry["error"] = envelope["error"]
            entry["duration_s"] = round(time.perf_counter() - started, 6)
            self.log.append(entry)
            return self._hard_limit_fallback(entry["tool"], count, executed_error=envelope["error"])
        if count > cfg.soft_actions_per_step and not (is_eval and envelope["ok"]) and not self.is_done:
            warning = (f"This is action {count} of step {step_no} (soft limit {cfg.soft_actions_per_step}). "
                       f"Commit an evaluation soon: beyond {cfg.hard_actions_per_step} actions the harness "
                       f"evaluates a fallback point for you.")
            if envelope["ok"] and isinstance(result, dict):
                result["warning"] = warning
            else:
                envelope["warning"] = warning
        entry["ok"] = envelope["ok"]
        if envelope["ok"]:
            entry["result"] = _bounded(envelope["result"])
        else:
            entry["error"] = envelope["error"]
        entry["duration_s"] = round(time.perf_counter() - started, 6)
        self.log.append(entry)
        return envelope

    def _unknown_tool_message(self, name: str) -> str:
        available = ", ".join(s.name for s in self.tools)
        if name == "stop" and not self.config.allow_early_stop:
            return f"stop is not available: early stopping is not allowed in this campaign (tools: {available})"
        if name == "create" and not self.config.agent_setup:
            return f"create is not available in this campaign (tools: {available})"
        return f"unknown tool {name!r}; available tools: {available}"

    def _hard_limit_reason(self, count: int) -> str:
        return (f"hard limit: {count - 1} actions in step {self.step} without an evaluation "
                f"(limit {self.config.hard_actions_per_step})")

    def _force_commit(self, entry: dict, count: int, started: float) -> dict:
        """Replace a call beyond the hard limit by a fallback evaluation (the call is not run)."""
        entry["error"] = f"not executed: {self._hard_limit_reason(count)}"
        entry["duration_s"] = round(time.perf_counter() - started, 6)
        self.log.append(entry)
        return self._hard_limit_fallback(entry["tool"], count)

    def _hard_limit_fallback(self, tool: str, count: int, executed_error: str | None = None) -> dict:
        """Commit the fallback evaluation of an over-limit step; the agent-facing error envelope."""
        hard = self.config.hard_actions_per_step
        fb = self.commit_fallback(self._hard_limit_reason(count))
        what = (f"your {tool} call was not executed" if executed_error is None
                else f"your {tool} call failed ({executed_error})")
        if fb.get("ok"):
            tid = fb["result"].get("trial_id")
            msg = (f"Action limit exceeded: step {self.step} already had {hard} actions without an evaluation, "
                   f"so the harness evaluated a fallback point for you (trial {tid}); {what}. "
                   f"This step is now committed.")
        else:
            msg = (f"Action limit exceeded ({hard} actions in step {self.step}); {what} and the fallback "
                   f"commit failed: {fb.get('error')}")
        return {"ok": False, "error": msg, "forced": True, "fallback": fb.get("result")}

    # ---- evaluation -------------------------------------------------------------------------
    def _check_can_evaluate(self) -> None:
        if self.stopped:
            raise LenzError("the campaign has been stopped; no further evaluations are possible")
        if self.remaining <= 0:
            raise LenzError(f"the evaluation budget is exhausted ({self.t} of {self.budget} used)")
        limit = self.config.max_evals_per_step
        if self._evals_this_step >= limit:
            raise LenzError(f"evaluate rejected: step {self.step} already committed {self._evals_this_step} "
                            f"evaluation(s) (limit {limit} per step); the next evaluation belongs to the next step")

    def _tool_evaluate(self, config: Mapping[str, Any], rationale: str, expectation: str | None = None) -> dict:
        self._check_can_evaluate()
        if not rationale or not str(rationale).strip():
            raise LenzError("rationale must not be blank: state your belief about this point and what you "
                            "expect to learn")
        cfg = self.problem.space.validate(config)
        return self._commit(cfg, source="agent", rationale=str(rationale).strip(),
                            expectation=None if expectation is None else str(expectation),
                            forced=False, k=self.actions_this_step - 1)

    def _duplicate_of(self, z: np.ndarray) -> int | None:
        if self._Z.shape[0] == 0:
            return None
        d = np.sqrt(np.mean((self._Z - z) ** 2, axis=1))
        i = int(np.argmin(d))
        return int(self.records[i]["trial_id"]) if d[i] <= DUPLICATE_TOL else None

    def _commit(self, cfg: dict, *, source: str, rationale: str, expectation: str | None, forced: bool,
                k: int) -> dict:
        """Evaluate ``cfg`` on the problem, record it in lenz and in ``records``; return the agent result.

        Flow: ``lenz.submit`` -> ``problem.evaluate`` (noise-free, the black box runs
        exactly once) -> observation noise added by the harness with
        :meth:`Problem.observe` semantics (``N(0, noise_std^2)`` per metric, drawn from
        the session's noise stream) -> ``lenz.observe``; the noise-free metrics give
        utility, feasibility and regret. An exception from the black box counts as a
        failed evaluation: the budget is spent and every metric is reported as missing.
        """
        space, problem = self.problem.space, self.problem
        z = space.encode(cfg)
        notes: list[str] = []
        bounds = self.lenz.shelf.bounds
        in_region = space.contains(cfg, bounds) if bounds else True
        if not in_region:
            notes.append("this config lies outside the active region of suggest (evaluated anyway)")
        dup = self._duplicate_of(z)
        if dup is not None:
            notes.append(f"this config repeats trial {dup}; a repeat only helps to average out observation noise")
        meta: dict[str, Any] = {"rationale": rationale, "step": self.step}
        if expectation is not None:
            meta["expectation"] = expectation
        if forced:
            meta["forced"] = True
        tid = int(self.lenz.submit(cfg, source=source, meta=meta)["trial_id"])

        error = None
        try:
            true = {str(k_): float(v) for k_, v in problem.evaluate(cfg).items()}
        except Exception as exc:  # the black box failed: the evaluation is spent, nothing is measured
            error = _error_text(exc)
            true = {}
            notes.append(f"the evaluation failed and reported no metrics ({error})")
        observed = self._add_noise(true)
        lenz_metrics: dict[str, Any] = {m: None for m in self.lenz.shelf.modelled_metrics}
        lenz_metrics.update({k_: _finite_or_none(v) for k_, v in observed.items()})
        try:
            obs = self.lenz.observe(lenz_metrics, trial_id=tid)
        except Exception:
            self.lenz.abandon(tid)
            raise

        self.t += 1
        self._evals_this_step += 1
        if not self.step_committed:
            self.actions_per_step.append(max(int(k), 0))
        self.step_committed = True
        self._Z = np.vstack([self._Z, z])
        utility = problem.utility(true) if error is None else -math.inf
        feasible = bool(problem.is_feasible(true)) if error is None else False
        regret = problem.regret(true) if error is None else math.nan
        self.records.append({
            "t": self.t, "step": self.step, "trial_id": tid, "config": dict(cfg), "observed": observed,
            "true_metrics": true, "utility": utility, "feasible": feasible, "regret": regret,
            "source": source, "rationale": rationale, "expectation": expectation, "forced": bool(forced),
            "in_region": bool(in_region), "duplicate_of": dup, "is_new_best": bool(obs["is_new_incumbent"]),
            "error": error,
        })
        if self.remaining == 0:
            notes.append("this was the last evaluation of the budget")
        result: dict[str, Any] = {
            "trial_id": tid, "config": dict(cfg), "observed": obs["metrics"], "feasible": obs["feasible"],
            "is_new_best": bool(obs["is_new_incumbent"]), "incumbent": obs["incumbent"],
            "t": self.t, "remaining": self.remaining,
        }
        if notes:
            result["note"] = "; ".join(notes)
        return result

    def _add_noise(self, true: Mapping[str, float]) -> dict[str, float]:
        """What the experimenter sees: ``true`` plus ``N(0, noise_std^2)`` per metric (``Problem.observe``).

        Applied to the single noise-free measurement, so the black box runs once per
        evaluation and ``observed - true`` is exactly the drawn noise.
        """
        sd = self.problem.noise_std
        if sd <= 0 or not true:
            return dict(true)
        eps = self._noise_rng.standard_normal(len(true)) * sd
        return {k: float(v) + float(e) for (k, v), e in zip(true.items(), eps)}

    def commit_fallback(self, reason: str) -> dict:
        """Evaluate a harness-chosen point for the current step (``forced=True``).

        The point is the top ``suggest`` candidate of the current lenz configuration; if
        that fails or repeats an evaluated trial, the next Sobol point; as a last resort
        a uniform random config. Returns the evaluation envelope (``{"ok", "result" |
        "error"}``) and logs it as a forced ``evaluate``.
        """
        started = time.perf_counter()
        entry: dict[str, Any] = {"t": self.t, "step": max(self.step, 1), "j": self.actions_this_step,
                                 "tool": "evaluate", "action_class": "evaluate", "args": None, "ok": False,
                                 "forced": True}
        try:
            if self.step == 0:
                self.begin_step()
            self._check_can_evaluate()
            cfg, how = self._fallback_config()
            entry["args"] = _bounded({"config": cfg, "reason": reason, "choice": how})
            res = self._commit(cfg, source="harness-fallback", rationale=f"harness fallback ({how}): {reason}",
                               expectation=None, forced=True, k=self.actions_this_step)
            envelope: dict[str, Any] = {"ok": True,
                                        "result": present(res, self.problem.space, metrics=self.known_metrics())}
        except Exception as exc:
            envelope = {"ok": False, "error": _error_text(exc)}
        entry["ok"] = envelope["ok"]
        if envelope["ok"]:
            entry["result"] = _bounded(envelope["result"])
        else:
            entry["error"] = envelope["error"]
        entry["duration_s"] = round(time.perf_counter() - started, 6)
        self.log.append(entry)
        return envelope

    def _fallback_config(self) -> tuple[dict, str]:
        for kwargs, label in (({}, None), ({"acqf": "sobol"}, "Sobol point")):
            try:
                r = self.lenz.suggest(q=1, **kwargs)
                cand = r["candidates"][0]
                if cand.get("duplicate_of") is None or label is not None:
                    return dict(cand["config"]), label or f"top {r.get('acqf')} suggestion"
            except Exception:  # noqa: BLE001 - any failure moves on to the next source
                continue
        return self.problem.space.sample_uniform(1, self._fallback_rng)[0], "uniform random point"

    def _tool_stop(self, reason: str) -> dict:
        if not self.config.allow_early_stop:
            raise LenzError("early stopping is not allowed in this campaign")
        if self.stopped:
            raise LenzError("the campaign is already stopped")
        if self.t == 0:
            raise LenzError("stop needs at least one evaluation: nothing has been evaluated yet")
        if self.remaining <= 0:
            raise LenzError(f"nothing to stop: the whole budget is used ({self.t} of {self.budget} evaluations) "
                            f"and the campaign ends by itself")
        self.stopped = True
        self.stop_reason = str(reason).strip()
        if not self.step_committed:
            self.actions_per_step.append(max(self.actions_this_step - 1, 0))
        self.step_committed = True
        return {"stopped": True, "evaluations_done": self.t, "unused_budget": self.remaining,
                "reason": self.stop_reason}

    def _tool_create(self, args: Mapping[str, Any]) -> dict:
        """Re-create the lenz study with agent-chosen settings, carrying every trial over."""
        old = self.lenz
        shelf = old.shelf
        objectives = args.get("objectives") or shelf.objectives
        constraints = args["constraints"] if args.get("constraints") is not None else shelf.constraints
        acqf = args.get("acqf")
        acqf_params = None
        if acqf is None and len(objectives) == len(shelf.objectives):
            acqf, acqf_params = shelf.acqf, shelf.acqf_params
        bounds = args["bounds"] if "bounds" in args else shelf.bounds
        model = {**shelf.model, **dict(args.get("model") or {})}
        new = Lenz(old.space, objectives, constraints=constraints, acqf=acqf, acqf_params=acqf_params,
                   bounds=bounds, model=model, seed=old.seed)
        modelled = new.shelf.modelled_metrics
        for t in old.frame.trials:
            tid = new.submit(t.config, source=t.source, meta=t.meta)["trial_id"]
            if t.status == "completed":
                metrics = {m: None for m in modelled}
                metrics.update(t.metrics or {})
                new.observe(metrics, trial_id=tid)
            elif t.status == "abandoned":
                new.abandon(tid)
        self.lenz = new
        return {"created": True, "config": new.shelf.to_dict(), "n_observed": new.n_observed,
                "carried_over": len(old.frame.trials)}


# =========================================================================== results


def compute_regret_curves(problem: Problem, records: Sequence[Mapping[str, Any]],
                          budget: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Simple and cumulative regret curves of ``records`` (noise-free values).

    Single objective: :func:`~agentic_bo.metrics.simple_regret_curve` /
    :func:`~agentic_bo.metrics.cumulative_regret_curve` of the primary utility.
    Several objectives: hypervolume regret ``max_hv - HV_t`` and its running sum.
    ``budget`` (larger than ``len(records)``) applies the early-stop padding. Unknown
    optima give NaN curves.
    """
    n = len(records)
    pad = budget if budget is not None and budget > n else None
    length = pad if pad is not None else n
    feas = [bool(r["feasible"]) for r in records]
    if problem.is_moo:
        objs = problem.objective_list
        if problem.ref_point is None or problem.max_hv is None:
            nan = np.full(length, np.nan)
            return nan, nan.copy()
        Y = np.array([[_finite_or_none(r["true_metrics"].get(o["metric"])) if r["true_metrics"] else None
                       for o in objs] for r in records], dtype=float).reshape(n, len(objs))
        hv_regret = hypervolume_regret_curve(Y, feas, problem.ref_point, problem.max_hv,
                                             [o["minimize"] for o in objs], budget=pad)
        return hv_regret, np.cumsum(hv_regret)
    u = np.array([r["utility"] for r in records], dtype=float)
    simple = simple_regret_curve(u, problem.u_star, feas, budget=pad)
    cumulative = cumulative_regret_curve(u, problem.u_star, feas, budget=pad)
    return simple, cumulative


def _nan_list(values: Any) -> list[float]:
    return [math.nan if v is None else float(v) for v in values or []]


@dataclass
class CampaignResult:
    """Outcome of one campaign (DESIGN §9.2), JSON round-trippable.

    Regret curves have one entry per evaluation (padded to the budget after an early
    stop) and are computed from noise-free values; for multi-objective problems
    ``simple_regret`` is the hypervolume regret and ``cumulative_regret`` its running
    sum. ``best_value`` / ``best_config``: the best feasible evaluated point by its
    noise-free primary value (raw units); for multi-objective problems ``best_value``
    is the final noise-free hypervolume and ``best_config`` is ``None``.

    Fields after ``policy_error`` are extensions: ``is_moo``, ``u_star``, ``n_evals``,
    ``action_class_counts``, ``campaign_config``, ``problem_info`` and ``stop_reason``.
    """

    problem: str
    policy: str
    seed: int
    budget: int
    regret_mode: str
    records: list[dict]
    simple_regret: list[float]
    cumulative_regret: list[float]
    best_value: float | None
    best_config: dict | None
    actions_per_step: list[int]
    action_counts: dict
    forced_commits: int
    context: list[dict]
    notebook: list[dict]
    log: list[dict]
    usage: dict
    wallclock_s: float
    stopped_early: bool
    policy_error: str | None
    is_moo: bool = False
    u_star: float | None = None
    n_evals: int = 0
    action_class_counts: dict = field(default_factory=dict)
    campaign_config: dict = field(default_factory=dict)
    problem_info: dict = field(default_factory=dict)
    stop_reason: str | None = None

    @property
    def final_simple_regret(self) -> float:
        return float(self.simple_regret[-1]) if self.simple_regret else math.nan

    @property
    def final_cumulative_regret(self) -> float:
        return float(self.cumulative_regret[-1]) if self.cumulative_regret else math.nan

    @classmethod
    def from_session(cls, session: AgentSession, policy_name: str, *, wallclock_s: float = 0.0,
                     usage: Mapping[str, Any] | None = None, policy_error: str | None = None) -> "CampaignResult":
        """Assemble the result of a finished session."""
        problem, records = session.problem, session.records
        stopped = bool(session.stopped)
        simple, cumulative = compute_regret_curves(problem, records, session.budget if stopped else None)
        best_value: float | None = None
        best_config: dict | None = None
        if problem.is_moo:
            objs = problem.objective_list
            if problem.ref_point is not None:
                Y = np.array([[_finite_or_none(r["true_metrics"].get(o["metric"])) for o in objs]
                              for r in records if r["feasible"]], dtype=float).reshape(-1, len(objs))
                best_value = dominated_hypervolume(Y, problem.ref_point, [o["minimize"] for o in objs])
        else:
            best = None
            for r in records:
                if r["feasible"] and math.isfinite(r["utility"]) and (best is None or r["utility"] > best["utility"]):
                    best = r
            if best is not None:
                best_value = _finite_or_none(best["true_metrics"].get(problem.objective))
                best_config = dict(best["config"])
        counts: dict[str, int] = {}
        class_counts: dict[str, int] = {}
        for e in session.log:
            if e.get("forced") or e.get("tool") is None:
                continue
            counts[e["tool"]] = counts.get(e["tool"], 0) + 1
            class_counts[e["action_class"]] = class_counts.get(e["action_class"], 0) + 1
        return cls(
            problem=problem.name, policy=str(policy_name), seed=session.config.seed, budget=session.budget,
            regret_mode=session.config.regret, records=jsonable(records),
            simple_regret=[float(v) for v in simple], cumulative_regret=[float(v) for v in cumulative],
            best_value=best_value, best_config=best_config, actions_per_step=list(session.actions_per_step),
            action_counts=dict(sorted(counts.items())), forced_commits=sum(1 for r in records if r["forced"]),
            context=[c.to_dict() for c in session.context], notebook=jsonable(session.notebook),
            log=jsonable(session.log), usage=jsonable(dict(usage or {})), wallclock_s=float(wallclock_s),
            stopped_early=stopped, policy_error=policy_error, is_moo=problem.is_moo, u_star=problem.u_star,
            n_evals=len(records), action_class_counts=dict(sorted(class_counts.items())),
            campaign_config=session.config.to_dict(), problem_info=problem.info(),
            stop_reason=session.stop_reason,
        )

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return {"format": RESULT_FORMAT, **jsonable(d)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CampaignResult":
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        missing = [f.name for f in dataclasses.fields(cls)
                   if f.name not in kwargs and f.default is dataclasses.MISSING
                   and f.default_factory is dataclasses.MISSING]
        if missing:
            raise LenzError(f"not a campaign result: missing fields {missing}")
        kwargs["simple_regret"] = _nan_list(kwargs["simple_regret"])
        kwargs["cumulative_regret"] = _nan_list(kwargs["cumulative_regret"])
        return cls(**kwargs)

    def save(self, path: str | os.PathLike) -> None:
        """Write the result as one JSON document (atomic replace)."""
        path = os.fspath(path)
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        text = dumps(self.to_dict())
        fd, tmp = tempfile.mkstemp(prefix=".campaign-", suffix=".tmp", dir=folder)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @classmethod
    def load(cls, path: str | os.PathLike) -> "CampaignResult":
        with open(os.fspath(path), encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# =========================================================================== campaign


class Campaign:
    """Run one policy on one problem: the metalevel loop of DESIGN §9.2.

    ``problem`` may be a :class:`Problem` or a registry name, ``policy`` a
    :class:`Policy` or a name for :func:`~agentic_bo.sara.policies.get_policy`. A
    policy with a ``toolset`` attribute overrides ``config.toolset``. After :meth:`run`,
    ``campaign.session`` holds the finished session (with its lenz study).
    """

    def __init__(self, problem: Problem | str, policy: Policy | str, config: CampaignConfig | None = None):
        if isinstance(problem, str):
            from ..problems import get_problem
            problem = get_problem(problem)
        if isinstance(policy, str):
            from .policies import get_policy
            policy = get_policy(policy)
        if not isinstance(policy, Policy):
            raise LenzError(f"policy must be a Policy, got {type(policy).__name__}")
        config = config if config is not None else CampaignConfig()
        toolset = getattr(policy, "toolset", None)
        if toolset and toolset != config.toolset:
            config = dataclasses.replace(config, toolset=toolset)
        self.problem = problem
        self.policy = policy
        self.config = config
        self.session: AgentSession | None = None
        self.result: CampaignResult | None = None

    def _policy_failed(self, exc: Exception) -> str:
        """Record a policy failure (``"fallback"``) or re-raise it (``"raise"``)."""
        if self.config.on_policy_error == "raise":
            raise exc
        return f"{type(exc).__name__}: {exc}"

    def _guarded(self, hook: Any, *args: Any) -> str | None:
        """Run a policy hook; any ``Exception`` it raises is a policy failure (message returned).

        A failing policy must not lose the campaign: the harness records the error and
        finishes the budget with fallback commits. ``BaseException`` (``KeyboardInterrupt``,
        abort signals) is not caught.
        """
        try:
            hook(*args)
        except Exception as exc:  # noqa: BLE001 - PolicyError or a bug in the policy alike
            return self._policy_failed(exc)
        return None

    def run(self) -> CampaignResult:
        """Run the campaign to the end of the budget (or an early stop) and return its result.

        An exception raised by the policy (a :class:`PolicyError` or any other
        ``Exception``) is recorded as ``policy_error`` and the remaining steps are
        committed by the harness (``on_policy_error="fallback"``), or re-raised
        (``"raise"``).
        """
        started = time.perf_counter()
        session = AgentSession(self.problem, self.config)
        self.session = session
        policy = self.policy
        error = self._guarded(policy.start, session)

        while session.t < session.budget and not session.stopped:
            for item in session._take_due_context():
                session.add_context(item)
                if error is None:
                    error = self._guarded(policy.on_context, session, item)
            if session.stopped:   # the policy stopped while handling the new context
                break
            session.begin_step()
            if error is None:
                error = self._guarded(policy.step, session)
            if not session.step_committed and not session.is_done:
                reason = ("policy ended the step without evaluating" if error is None
                          else f"policy failed ({error})")
                if not session.commit_fallback(reason)["ok"]:
                    break  # cannot make progress; end the campaign instead of looping

        msg = self._guarded(policy.finish, session)
        error = error or msg
        try:
            usage = dict(policy.usage() or {})
        except Exception as exc:  # usage reporting must not lose a finished campaign
            usage = {"error": _error_text(exc)}
        self.result = CampaignResult.from_session(session, policy.name, wallclock_s=time.perf_counter() - started,
                                                  usage=usage, policy_error=error)
        return self.result
