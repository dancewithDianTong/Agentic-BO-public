"""Agent-facing tools (DESIGN.md §9.1): the interface between a policy and lenz / the harness.

Every tool is a :class:`ToolSpec` with an LLM-facing description, a JSON input schema
and a handler. Most tools map 1:1 to a lenz command; ``evaluate``, ``stop``,
``record_note`` and ``create`` are harness tools. Schemas are generated once per
problem (one property per parameter, all required, no extra keys) and descriptions
contain nothing volatile, so the tool list is fixed for a whole campaign and the
prompt prefix stays cacheable.

The paper reports that agents are sensitive to the interface, so the descriptions say
what each tool does, when it helps, what it costs (only ``evaluate`` spends budget;
``suggest`` / ``predict`` / ``score`` record nothing) and which units and conventions
apply (raw metric units, ``bounds`` format, normalized trust-region radius).

Results go back to agents in a compact form (:func:`present`): floats are rounded to a
few significant digits (configs keep enough digits to be re-submitted exactly), lenz
bookkeeping fields are dropped, and oversized results are truncated with a note.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

import numpy as np

from ..lenz.acquisition import ACQF_NAMES, POINTWISE_ACQFS
from ..lenz.frame import DEFAULT_MODEL
from ..lenz.gp import KERNELS
from ..space import IntParam, OrdinalParam, RealParam, SearchSpace
from ..utils import LenzError, jsonable

if TYPE_CHECKING:  # pragma: no cover
    from .session import AgentSession

__all__ = [
    "ToolSpec", "TOOLSETS", "ACTION_CLASSES", "build_tools", "config_schema", "bounds_schema",
    "present", "compact", "round_sig", "round_param_value", "limit_size", "trial_row",
    "MAX_PREDICT_CONFIGS", "MAX_SUGGEST_Q", "MAX_RESULT_CHARS", "SIG_DIGITS", "METRIC_SIG_DIGITS",
]

ACTION_CLASSES = ("probe", "reconfigure", "propose", "evaluate", "stop", "note", "setup")
#: tools per tool set; ``stop`` needs ``allow_early_stop`` and ``create`` needs ``agent_setup``
TOOLSETS: dict[str, list[str]] = {
    "full": ["create", "diagnostics", "evaluate", "incumbent", "pareto", "predict", "record_note",
             "score", "set_acqf", "set_bounds", "set_constraints", "set_objectives", "status", "stop",
             "suggest", "trials"],
    "llm_only": ["evaluate", "incumbent", "record_note", "status", "stop", "trials"],
}
MAX_PREDICT_CONFIGS = 50
MAX_SUGGEST_Q = 10
MAX_NOTE_CHARS = 4000
MAX_TEXT_CHARS = 4000
#: agent-facing results longer than this (JSON characters) are truncated with a note
MAX_RESULT_CHARS = 24_000
#: significant digits of floats in agent-facing results (configs keep more, see round_param_value)
SIG_DIGITS = 6
#: significant digits of metric values (observed metrics, objective ``value``, predicted ``mean``), so
#: differences far below a large offset (e.g. y ~ 1e4) stay visible
METRIC_SIG_DIGITS = 10
#: result keys whose floats are metric values whatever the metric is called
METRIC_VALUE_KEYS = ("value", "mean")


@dataclass
class ToolSpec:
    """One agent-facing tool.

    Attributes:
        name: snake_case tool name.
        action_class: ``probe`` | ``reconfigure`` | ``propose`` | ``evaluate`` | ``stop``
            | ``note`` | ``setup`` (the paper's action taxonomy, used in traces).
        description: concise, accurate, LLM-facing description.
        input_schema: JSON schema of the arguments (``type: object``, no extra keys).
        handler: ``handler(session, args) -> dict``; raises ``LenzError`` for input the
            agent can fix. Called by :meth:`AgentSession.call` after schema validation.
    """

    name: str
    action_class: str
    description: str
    input_schema: dict
    handler: Callable[["AgentSession", dict], dict]

    def definition(self) -> dict:
        """Anthropic tool dict ``{"name", "description", "input_schema"}``."""
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


# =========================================================================== schemas


def _obj(properties: Mapping[str, Any], required: list[str] | None = None) -> dict:
    out: dict[str, Any] = {"type": "object", "properties": dict(properties)}
    if required:
        out["required"] = list(required)
    out["additionalProperties"] = False
    return out


def _g(v: float) -> str:
    return f"{v:g}"


def _param_schema(p: Any) -> dict:
    if isinstance(p, RealParam):
        scale = ", log scale" if p.log_scale else ""
        return {"type": "number", "minimum": p.lower, "maximum": p.upper,
                "description": f"real in [{_g(p.lower)}, {_g(p.upper)}]{scale}"}
    if isinstance(p, IntParam):
        scale = ", log scale" if p.log_scale else ""
        return {"type": "integer", "minimum": p.lower, "maximum": p.upper,
                "description": f"integer in [{p.lower}, {p.upper}]{scale}"}
    if isinstance(p, OrdinalParam):
        return {"enum": list(p.values), "description": "ordinal (ordered values)"}
    return {"enum": list(p.choices), "description": "categorical"}


def config_schema(space: SearchSpace, description: str | None = None) -> dict:
    """Schema of one configuration: every parameter required, nothing else allowed."""
    out = _obj({p.name: _param_schema(p) for p in space.params}, list(space.names))
    out["description"] = description or ("A configuration: a value for every parameter, "
                                          "inside its domain (raw units).")
    return out


def bounds_schema(space: SearchSpace) -> dict:
    """Schema of an active-region spec ``{param: [lo, hi] | [allowed values...]}``."""
    props = {}
    for p in space.params:
        if isinstance(p, (RealParam, IntParam)):
            props[p.name] = {"type": "array", "minItems": 2, "maxItems": 2,
                             "items": {"type": "number", "minimum": p.lower, "maximum": p.upper},
                             "description": f"[lo, hi] with {_g(p.lower)} <= lo <= hi <= {_g(p.upper)} (raw units)"}
        else:
            options = list(p.values) if isinstance(p, OrdinalParam) else list(p.choices)
            props[p.name] = {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"enum": options},
                             "description": "non-empty list of allowed values"}
    out = _obj(props)
    out["description"] = ("Active region: {param: [lo, hi]} for numeric parameters (raw units), "
                          "{param: [allowed values]} for categorical / ordinal ones; omitted parameters "
                          "keep their full domain.")
    return out


def _metric_enum(session: "AgentSession") -> dict:
    return {"type": "string", "enum": session.known_metrics()}


# =========================================================================== presentation


def round_sig(x: float, sig: int = SIG_DIGITS) -> float:
    """``x`` rounded to ``sig`` significant digits (non-finite values unchanged)."""
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        return x
    return float(f"{x:.{int(sig)}g}")


def round_param_value(p: Any, v: Any) -> Any:
    """Round a raw value of parameter ``p`` for display without losing identity.

    Real values keep enough digits that the rounding error is below ``1e-6`` of the
    parameter's (log-)range, so a config copied back from a result is recognized as the
    same point (lenz duplicate tolerance), and stay inside the domain. Values of integer,
    ordinal and categorical parameters are returned exactly (a rounded ``0.000316228``
    would not match the ordinal value ``0.00031622776601683794`` and could not be
    re-submitted). Numbers that are not values of the parameter (e.g. a lengthscale
    keyed by the parameter's name) get the usual :data:`SIG_DIGITS`.
    """
    if isinstance(v, bool) or not isinstance(v, (float, np.floating)):
        return v
    x = float(v)
    if not math.isfinite(x) or x == 0.0:
        return x
    if isinstance(p, RealParam):
        if not p.lower <= x <= p.upper:     # not a value of this parameter (e.g. a lengthscale)
            return round_sig(x, SIG_DIGITS)
        if p.log_scale:
            span = math.log(p.upper / p.lower)
            extra = max(0, math.ceil(-math.log10(span))) if span > 0 else 0
        else:
            ratio = abs(x) / (p.upper - p.lower)
            extra = max(0, math.ceil(math.log10(ratio))) if ratio > 0 else 0
        r = round_sig(x, min(7 + extra, 17))
        return min(max(r, p.lower), p.upper)
    if isinstance(p, IntParam):
        exact = x.is_integer() and p.lower <= x <= p.upper
    elif isinstance(p, OrdinalParam):
        exact = any(_same_number(x, o) for o in p.values)
    elif p is not None and hasattr(p, "choices"):
        exact = any(_same_number(x, o) for o in p.choices)
    else:
        exact = False
    return x if exact else round_sig(x, SIG_DIGITS)


def _same_number(x: float, o: Any) -> bool:
    return isinstance(o, (int, float, np.integer, np.floating)) and not isinstance(o, (bool, np.bool_)) \
        and float(o) == x


def compact(obj: Any, space: SearchSpace | None = None, sig: int = SIG_DIGITS,
            metrics: Iterable[str] = ()) -> Any:
    """JSON-safe copy of ``obj`` with floats rounded for agent consumption.

    Values stored under a parameter's name (in configs, bounds, boxes, per-parameter
    diagnostics) use :func:`round_param_value`. Metric values (stored under a name in
    ``metrics``, or under ``value`` / ``mean``) keep :data:`METRIC_SIG_DIGITS`, so the
    resolution does not depend on the metric's offset. Every other float keeps ``sig``
    significant digits. Non-finite floats become ``None`` (``utils.jsonable``).
    """
    params = {p.name: p for p in space.params} if space is not None else {}
    metric_keys = set(metrics) | set(METRIC_VALUE_KEYS)

    def rec(o: Any, key: Any) -> Any:
        if isinstance(o, Mapping):
            return {str(k): rec(v, k) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [rec(v, key) for v in o]
        if isinstance(o, (float, np.floating)):
            x = float(o)
            if not math.isfinite(x):
                return None
            p = params.get(key) if isinstance(key, str) else None
            if p is not None:
                return round_param_value(p, x)
            return round_sig(x, max(sig, METRIC_SIG_DIGITS) if key in metric_keys else sig)
        return o

    return rec(jsonable(obj), None)


def _json_len(obj: Any) -> int:
    return len(json.dumps(obj, sort_keys=True, allow_nan=False))


def limit_size(result: dict, max_chars: int | None = None, list_key: str | None = None,
               keep: str = "head", hint: str = "") -> dict:
    """Shorten an oversized result by cutting one list (``list_key`` or the longest top-level list).

    ``max_chars`` defaults to :data:`MAX_RESULT_CHARS` (JSON characters). ``keep="head"``
    keeps the first items, ``"tail"`` the last ones. A ``truncated`` note says how many
    items are shown and how to get the rest (``hint``). If cutting a list is not enough,
    the result is replaced by a summary of its keys.
    """
    if max_chars is None:
        max_chars = MAX_RESULT_CHARS
    if _json_len(result) <= max_chars:
        return result
    out = dict(result)
    key = list_key
    if key is None or not isinstance(out.get(key), list):
        lists = [(k, v) for k, v in out.items() if isinstance(v, list) and v]
        key = max(lists, key=lambda kv: _json_len(kv[1]))[0] if lists else None
    if key is not None:
        items = out[key]
        n = len(items)

        def build(k: int) -> dict:
            shown = items[:k] if keep == "head" else items[n - k:] if k else []
            note = (f"result too long: showing the {'first' if keep == 'head' else 'last'} {k} of {n} "
                    f"'{key}' entries" + (f"; {hint}" if hint else ""))
            return {**out, key: shown, "truncated": note}

        lo, hi = 0, n
        while lo < hi:  # largest k that fits
            mid = (lo + hi + 1) // 2
            if _json_len(build(mid)) <= max_chars:
                lo = mid
            else:
                hi = mid - 1
        candidate = build(lo)
        if _json_len(candidate) <= max_chars:
            return candidate
    return {"truncated": f"result too long ({_json_len(result)} characters); keys: {sorted(result)}"
                         + (f"; {hint}" if hint else "")}


def present(result: Any, space: SearchSpace | None = None, max_chars: int | None = None,
            metrics: Iterable[str] = ()) -> Any:
    """Compact (:func:`compact`, metric names ``metrics``) and size-limit (:func:`limit_size`) a tool result."""
    out = compact(result, space, metrics=metrics)
    if isinstance(out, dict):
        out = limit_size(out, max_chars)
    return out


def trial_row(t: Mapping[str, Any]) -> dict:
    """Compact agent-facing view of a lenz trial dict (no sequence numbers or meta)."""
    row = {"trial_id": t.get("trial_id"), "eval_index": t.get("eval_index"), "source": t.get("source"),
           "config": t.get("config"), "metrics": t.get("metrics"), "feasible": t.get("feasible")}
    if t.get("status") != "completed":
        row["status"] = t.get("status")
    return row


# =========================================================================== handlers


def _lenz(session: "AgentSession", command: str, **kwargs: Any) -> dict:
    """Run a lenz command through its envelope; raise ``LenzError`` with lenz's message on failure."""
    env = session.lenz.call(command, {k: v for k, v in kwargs.items() if v is not None})
    if env.get("ok"):
        return env["result"]
    err = env.get("error") or {}
    msg = str(err.get("message", "unknown error"))
    if err.get("type") not in (None, "LenzError"):
        msg = f"internal error in lenz {command} ({err.get('type')}): {msg}"
    raise LenzError(msg)


def _llm_only(session: "AgentSession") -> bool:
    return session.toolset == "llm_only"


def _h_status(session: "AgentSession", args: dict) -> dict:
    st = _lenz(session, "status")
    cfg = session.config
    out: dict[str, Any] = {
        "step": session.step, "evaluations_done": session.t, "budget": session.budget,
        "remaining": session.remaining, "regret_mode": cfg.regret, "early_stop_allowed": cfg.allow_early_stop,
        "step_committed": session.step_committed,
        "actions_this_step": session.actions_this_step,
        "action_limits": {"soft": cfg.soft_actions_per_step, "hard": cfg.hard_actions_per_step},
        "n_observed": st["n_observed"], "n_feasible": st["n_feasible"], "feasible_found": st["feasible_found"],
        "is_moo": st["is_moo"], "incumbent": st["incumbent"],
    }
    if not _llm_only(session):
        shelf = st["config"]
        out["lenz_config"] = {k: shelf.get(k) for k in ("objectives", "constraints", "acqf", "acqf_params",
                                                        "bounds", "model")}
    out["recent_notes"] = [n["text"] for n in session.notebook[-3:]]
    return out


def _h_trials(session: "AgentSession", args: dict) -> dict:
    sort_by, limit = args.get("sort_by"), args.get("limit")
    res = _lenz(session, "trials", last=args.get("last"), sort_by=sort_by, limit=limit)
    rows = [trial_row(t) for t in res["trials"]]
    out = {"trials": compact(rows, session.lenz.space, metrics=session.known_metrics()),
           "n": res["n"], "n_total": res["n_total"]}
    keep = "head" if sort_by == "value" or limit is not None else "tail"
    return limit_size(out, list_key="trials", keep=keep,
                      hint="use 'last', or sort_by='value' with 'limit', to see the part you need")


def _h_incumbent(session: "AgentSession", args: dict) -> dict:
    res = dict(_lenz(session, "incumbent"))
    if _llm_only(session):
        res.pop("best_posterior", None)
    if res.get("pareto") is not None:
        res["pareto"] = [trial_row(t) for t in res["pareto"]]
    return res


def _h_pareto(session: "AgentSession", args: dict) -> dict:
    res = dict(_lenz(session, "pareto"))
    res["pareto"] = [trial_row(t) for t in res.get("pareto") or []]
    return res


def _h_diagnostics(session: "AgentSession", args: dict) -> dict:
    return _lenz(session, "diagnostics", metric=args.get("metric"))


def _h_predict(session: "AgentSession", args: dict) -> dict:
    return _lenz(session, "predict", configs=args["configs"])


def _h_score(session: "AgentSession", args: dict) -> dict:
    return _lenz(session, "score", configs=args["configs"], acqfs=args.get("acqfs"))


def _h_set_bounds(session: "AgentSession", args: dict) -> dict:
    bounds = args.get("bounds")
    return _lenz(session, "set_bounds", bounds=bounds if bounds else "full")


def _h_set_acqf(session: "AgentSession", args: dict) -> dict:
    return _lenz(session, "set_acqf", acqf=args["acqf"], beta=args.get("beta"), params=args.get("params"))


def _check_metrics(session: "AgentSession", names: list[str], what: str) -> None:
    known = session.known_metrics()
    unknown = [m for m in names if m not in known]
    if unknown:
        raise LenzError(f"{what}: unknown metric(s) {unknown}; the problem reports {known}")


def _h_set_objectives(session: "AgentSession", args: dict) -> dict:
    objs = args["objectives"]
    _check_metrics(session, [o["metric"] for o in objs], "set_objectives")
    return _lenz(session, "set_objectives", objectives=objs)


def _h_set_constraints(session: "AgentSession", args: dict) -> dict:
    cons = args["constraints"]
    _check_metrics(session, [c["metric"] for c in cons], "set_constraints")
    return _lenz(session, "set_constraints", constraints=cons)


def _h_suggest(session: "AgentSession", args: dict) -> dict:
    params = {k: v for k, v in dict(args.get("params") or {}).items() if v is not None}
    if args.get("beta") is not None:
        params["beta"] = args["beta"]
    res = dict(_lenz(session, "suggest", q=args.get("q"), around=args.get("around"), radius=args.get("radius"),
                     bounds=args.get("bounds"), acqf=args.get("acqf"), acqf_params=params or None))
    # the acquisition name is given once at the top level; the region only lists what is set
    res["candidates"] = [{k: v for k, v in c.items() if k != "acqf"} for c in res.get("candidates") or []]
    region = {k: v for k, v in (res.get("region") or {}).items() if v is not None}
    res["region"] = region or "full domain"
    if not res.get("n_pending"):
        res.pop("n_pending", None)
    return res


def _h_evaluate(session: "AgentSession", args: dict) -> dict:
    return session._tool_evaluate(args["config"], rationale=args["rationale"],
                                  expectation=args.get("expectation"))


def _h_stop(session: "AgentSession", args: dict) -> dict:
    return session._tool_stop(args["reason"])


def _h_record_note(session: "AgentSession", args: dict) -> dict:
    text = args["text"].strip()
    if not text:
        raise LenzError("text must not be blank")
    session.notebook.append({"t": session.t, "step": session.step, "text": text})
    return {"recorded": True, "n_notes": len(session.notebook)}


def _h_create(session: "AgentSession", args: dict) -> dict:
    for key, what in (("objectives", "create objectives"), ("constraints", "create constraints")):
        if args.get(key):
            _check_metrics(session, [o["metric"] for o in args[key]], what)
    return session._tool_create(args)


# =========================================================================== specs


_FREE = "Free: spends no budget"
_READ_ONLY = f"{_FREE} and changes nothing."
_NO_DATA_LOSS = "Reconfiguration never drops data: every trial stays in the log and in the models."


def _specs(session: "AgentSession", toolset: str) -> dict[str, ToolSpec]:
    space = session.problem.space
    llm_only = toolset == "llm_only"
    cfg_s = config_schema(space)
    configs_s = {"type": "array", "minItems": 1, "maxItems": MAX_PREDICT_CONFIGS, "items": cfg_s,
                 "description": f"1 to {MAX_PREDICT_CONFIGS} configurations"}
    metric_s = _metric_enum(session)
    acqf_s = {"type": "string", "enum": list(ACQF_NAMES)}
    beta_s = {"type": ["number", "null"], "minimum": 0, "description": "ucb exploration weight"}
    params_s = {"type": ["object", "null"], "additionalProperties": False,
                "description": "advanced parameters: ehvi ref_point (raw units, one per objective) and "
                               "n_samples; parego weights and rho; ts n_candidates",
                "properties": {
                    "ref_point": {"type": "array", "items": {"type": "number"}},
                    "n_samples": {"type": "integer", "minimum": 1, "maximum": 4096},
                    "weights": {"type": "array", "items": {"type": "number", "minimum": 0}},
                    "rho": {"type": "number", "minimum": 0},
                    "n_candidates": {"type": "integer", "minimum": 16, "maximum": 20000}}}
    region_note = ("" if llm_only else
                   " It may lie outside the active region of `suggest` (the result then carries a note).")
    specs = [
        ToolSpec(
            "status", "probe",
            "Overview of the campaign: current step, evaluations done and remaining budget, regret mode, "
            "number of (feasible) observations, the incumbent, your action count in this step"
            + ("" if llm_only else ", the current lenz configuration (objectives, constraints, acquisition "
               "function, active region, GP model settings)")
            + " and your most recent notebook entries. " + _READ_ONLY
            + " Useful to re-orient at the start of a step.",
            _obj({}), _h_status),
        ToolSpec(
            "trials", "probe",
            "The trial log (the single source of truth): for every evaluated trial its trial_id, eval_index, "
            "source, config and observed metrics (raw units; noisy if the problem is noisy) and whether it is "
            "feasible. `last` keeps only the k most recent trials and is applied before sorting; "
            "sort_by='value' lists the best trials first (feasible before infeasible, in the objective's own "
            "direction); sort_by='eval_index' orders by evaluation; `limit` keeps the first k rows after "
            "sorting, so sort_by='value' with limit=k gives the k best trials. " + _READ_ONLY
            + " Long logs are truncated to fit; use `last`, or sort_by='value' with `limit`, to see the part "
            "you need.",
            _obj({"last": {"type": ["integer", "null"], "minimum": 1,
                           "description": "keep only the k most recent trials (applied before sorting)"},
                  "sort_by": {"type": ["string", "null"], "enum": ["eval_index", "value", None],
                              "description": "'value' = best first; 'eval_index' = evaluation order"},
                  "limit": {"type": ["integer", "null"], "minimum": 1,
                            "description": "keep only the first k rows after sorting (with sort_by='value': "
                                           "the k best trials)"}}),
            _h_trials),
        ToolSpec(
            "incumbent", "probe",
            "The best feasible evaluated trial so far (raw units, observed values)"
            + ("" if llm_only else ", plus best_posterior: the evaluated trial with the best GP posterior mean, "
               "a noise-robust alternative when observations are noisy")
            + ". For several objectives it returns the Pareto set and its hypervolume instead. When no "
            "trial is feasible yet, least_infeasible shows the closest one. " + _READ_ONLY,
            _obj({}), _h_incumbent),
    ]
    if not llm_only:
        specs += [
            ToolSpec(
                "pareto", "probe",
                "Non-dominated feasible evaluated trials, the hypervolume they dominate and the reference "
                "point (raw units, each objective in its own direction). Mainly for multi-objective studies. "
                + _READ_ONLY,
                _obj({}), _h_pareto),
            ToolSpec(
                "diagnostics", "probe",
                "How far to trust the GP surrogate, per modelled metric: cv_r2 (leave-one-out cross-validated "
                "R^2: near 1 = predictive; below about 0.5 = predictions and suggestions are unreliable, so "
                "rely more on space filling, exploration or your own reasoning), noise_std and signal_std "
                "(raw units), lengthscales per parameter (normalized [0, 1] units; small = the metric changes "
                "quickly along that parameter, very large = the parameter barely matters), sensitivity per "
                "parameter (sums to 1; which inputs drive the metric) and the log marginal likelihood. "
                "Free: spends no budget and changes nothing (it may fit the models).",
                _obj({"metric": {**metric_s, "type": ["string", "null"], "enum": metric_s["enum"] + [None],
                                 "description": "one metric only (default: every objective and constraint)"}}),
                _h_diagnostics),
            ToolSpec(
                "predict", "probe",
                f"GP posterior at up to {MAX_PREDICT_CONFIGS} configs of your choice: per metric the mean, "
                "variance and std (raw units), the probability that the config is feasible, its value under "
                "the current acquisition function (when that is ts, sobol or random, which have no pointwise "
                "value: under the default logei, or ehvi / parego for several objectives) and whether it lies "
                "inside the active region. Use it to "
                "check a hypothesis or a hand-crafted candidate before spending an evaluation. Predictions "
                "are beliefs, not observations: they never enter the trial log. Free: spends no budget and "
                "records nothing.",
                _obj({"configs": configs_s}, ["configs"]), _h_predict),
            ToolSpec(
                "score", "probe",
                f"Acquisition values of up to {MAX_PREDICT_CONFIGS} configs under one or more acquisition "
                "functions (default: the current one; when that is ts, sobol or random, which have no "
                "pointwise value, the default logei, or ehvi / parego for several objectives), e.g. to "
                "compare a global candidate, a trust-region "
                "candidate and your own idea on a common scale. Values are 'higher is better' within one "
                "acquisition function; do not compare values across functions. `best` gives the index of the "
                "top config for each function. Free: spends no budget and records nothing.",
                _obj({"configs": configs_s,
                      "acqfs": {"type": ["array", "null"], "minItems": 1,
                                "items": {"type": "string", "enum": list(POINTWISE_ACQFS)},
                                "description": "acquisition functions to evaluate (default: the current one)"}},
                     ["configs"]),
                _h_score),
            ToolSpec(
                "set_bounds", "reconfigure",
                "Set the active region that `suggest` searches. Format: {param: [lo, hi]} for numeric "
                "parameters (raw units, inside the domain) and {param: [allowed values]} for categorical or "
                "ordinal ones; parameters you omit keep their full domain; null resets to the full domain. "
                "Use it to focus on a promising region or to encode prior knowledge, and widen it again if "
                "progress stalls; collapsing the region early risks missing the optimum. `evaluate` may still "
                "go outside the region. " + _NO_DATA_LOSS + " " + _FREE + ".",
                _obj({"bounds": {"anyOf": [bounds_schema(space), {"type": "null"}],
                                 "description": "the new active region, or null for the full domain"}},
                     ["bounds"]),
                _h_set_bounds),
            ToolSpec(
                "set_acqf", "reconfigure",
                "Choose the acquisition function used by `suggest` (and by default by `predict` and `score`). "
                "Single objective: logei (default; log expected improvement, balanced), noisy_logei (for noisy "
                "observations), ei, logpi / pi (probability of improvement; more exploitative), ucb (mean + "
                "sqrt(beta) * std; `beta` sets exploration, default 4, larger explores more), pm (posterior "
                "mean; pure exploitation), std (posterior std; pure exploration), ts (Thompson sampling). "
                "Several objectives: ehvi (exactly 2 objectives), parego. Model-free: sobol (space filling), "
                "random. " + _NO_DATA_LOSS + " " + _FREE + ".",
                _obj({"acqf": acqf_s,
                      "beta": beta_s, "params": params_s},
                     ["acqf"]),
                _h_set_acqf),
            ToolSpec(
                "set_objectives", "reconfigure",
                "Replace the objectives lenz optimizes: a list of {metric, minimize} (two or more = "
                "multi-objective; the acquisition function switches family automatically when needed and the "
                "result says so). Only metrics the problem reports can be used. Use it when instructions "
                "change the goal. " + _NO_DATA_LOSS + " " + _FREE + ".",
                _obj({"objectives": {"type": "array", "minItems": 1, "items": _obj({
                    "metric": metric_s, "minimize": {"type": "boolean"},
                    "ref": {"type": ["number", "null"],
                            "description": "optional hypervolume reference value (raw units)"}},
                    ["metric", "minimize"])}}, ["objectives"]),
                _h_set_objectives),
            ToolSpec(
                "set_constraints", "reconfigure",
                "Replace the outcome constraints: a list of {metric, lower, upper} (feasible iff lower <= "
                "metric <= upper in raw units; give at least one bound, null for an open side); [] removes all "
                "constraints. Acquisition functions then weight candidates by their probability of "
                "feasibility, and incumbents count feasible trials only. " + _NO_DATA_LOSS + " " + _FREE + ".",
                _obj({"constraints": {"type": "array", "items": _obj({
                    "metric": metric_s, "lower": {"type": ["number", "null"]}, "upper": {"type": ["number", "null"]}},
                    ["metric"])}}, ["constraints"]),
                _h_set_constraints),
            ToolSpec(
                "suggest", "propose",
                "Propose candidate configs by optimizing the acquisition function with the GP surrogate over "
                "the active region. Free: it records nothing and spends no budget (candidates have trial_id "
                "null); only `evaluate` commits a point. Options: q = number of candidates (1-10; a greedy "
                "batch with fantasized outcomes, useful as a diverse shortlist); acqf = a different acquisition "
                "function for this call only; beta / params = acquisition parameters for this call only (as in "
                "`set_acqf`, e.g. acqf='ucb' with beta=9 to explore more; without acqf they adjust the current "
                "function); bounds = a region for this call only ('full' = the whole "
                "domain); around = 'incumbent' or a config, to search a local trust region of normalized "
                "half-width `radius` (in (0, 1], a fraction of each parameter's range, scaled by the GP "
                "lengthscales; default 0.1). With fewer than 2 observations it returns space-filling (Sobol) "
                "points. Each candidate lists its acquisition value, predicted metrics (mean, std; raw "
                "units), probability of feasibility and duplicate_of when it repeats an evaluated trial. "
                "Suggestions are only as good as the surrogate: check `diagnostics` when unsure.",
                _obj({"q": {"type": ["integer", "null"], "minimum": 1, "maximum": MAX_SUGGEST_Q},
                      "around": {"anyOf": [{"type": "string", "enum": ["incumbent"]},
                                           config_schema(space, "a config to center the trust region on"),
                                           {"type": "null"}],
                                 "description": "'incumbent' or a config: center of a local trust region"},
                      "radius": {"type": ["number", "null"], "exclusiveMinimum": 0, "maximum": 1,
                                 "description": "trust-region half-width as a fraction of each range"},
                      "bounds": {"anyOf": [{"type": "string", "enum": ["full"]}, bounds_schema(space),
                                           {"type": "null"}],
                                 "description": "region for this call only ('full' = whole domain)"},
                      "acqf": {**acqf_s, "type": ["string", "null"], "enum": list(ACQF_NAMES) + [None]},
                      "beta": {**beta_s, "description": "ucb exploration weight for this call only"},
                      "params": {**params_s, "description": "acquisition parameters for this call only "
                                                            "(same keys as in set_acqf)"}}),
                _h_suggest),
        ]
    specs += [
        ToolSpec(
            "evaluate", "evaluate",
            "Commit one config to the real, expensive black-box function. This is the only action that spends "
            "budget: it uses one of the remaining evaluations and ends the current step (one evaluate per "
            "step). The observed metrics (raw units; noisy if the problem is noisy) are recorded in the trial "
            "log and returned with is_new_best and the incumbent. `config` needs a value for every parameter "
            "inside its domain." + region_note + " `rationale` is required: state what you believe about this "
            "point and what you expect to learn from it (for example 'exploit: predicted 0.12 +/- 0.03 next to "
            "the incumbent' or 'explore: the unexplored corner where uncertainty is highest'). `expectation` "
            "(optional): your predicted outcome, to compare with the observation.",
            _obj({"config": cfg_s,
                  "rationale": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARS,
                                "description": "belief about this point and the learning goal"},
                  "expectation": {"type": ["string", "null"], "maxLength": MAX_TEXT_CHARS,
                                  "description": "predicted outcome, e.g. 'y about 0.4'"}},
                 ["config", "rationale"]),
            _h_evaluate),
        ToolSpec(
            "record_note", "note",
            "Write a short entry in your lab notebook: hypotheses about the landscape, sensitive parameters, "
            "how far the surrogate can be trusted, your plan. Free: it does not touch the trial log. The most "
            "recent notes are shown by `status`.",
            _obj({"text": {"type": "string", "minLength": 1, "maxLength": MAX_NOTE_CHARS}}, ["text"]),
            _h_record_note),
    ]
    if session.config.allow_early_stop:
        specs.append(ToolSpec(
            "stop", "stop",
            "End the campaign now and forfeit the remaining evaluations. Irreversible. Under simple regret "
            "the result is the best point found so far; under cumulative regret every unused evaluation is "
            "charged as if it repeated your best point. Stop only when further evaluations cannot help (and "
            "only after at least one evaluation; once the whole budget is used the campaign ends by itself "
            "and stop is rejected).",
            _obj({"reason": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT_CHARS}}, ["reason"]),
            _h_stop))
    if session.config.agent_setup and not llm_only:
        objective_item = _obj({"metric": metric_s, "minimize": {"type": "boolean"},
                               "ref": {"type": ["number", "null"]}}, ["metric", "minimize"])
        constraint_item = _obj({"metric": metric_s, "lower": {"type": ["number", "null"]},
                                "upper": {"type": ["number", "null"]}}, ["metric"])
        specs.append(ToolSpec(
            "create", "setup",
            "Set up the lenz study yourself: objectives, outcome constraints, acquisition function, active "
            "region and GP model (kernel matern52 or rbf; noise 'learn' or a fixed variance in standardized "
            "units). Omitted fields keep their current setting. Trials already evaluated are carried over, so "
            "no data is lost; the search space is fixed by the problem. " + _FREE + ".",
            _obj({"objectives": {"type": ["array", "null"], "minItems": 1, "items": objective_item},
                  "constraints": {"type": ["array", "null"], "items": constraint_item},
                  "acqf": {**acqf_s, "type": ["string", "null"], "enum": list(ACQF_NAMES) + [None]},
                  "bounds": {"anyOf": [bounds_schema(space), {"type": "null"}]},
                  "model": {"type": ["object", "null"], "additionalProperties": False, "properties": {
                      "kernel": {"type": "string", "enum": list(KERNELS)},
                      "noise": {"anyOf": [{"type": "string", "enum": ["learn"]},
                                          {"type": "number", "minimum": 0}]},
                      "n_restarts": {"type": "integer", "minimum": 1, "maximum": 20}},
                            "description": f"GP settings (default {DEFAULT_MODEL})"}}),
            _h_create))
    return {s.name: s for s in specs}


def build_tools(session: "AgentSession", toolset: str = "full") -> list[ToolSpec]:
    """The tools available to the agent of ``session``, sorted by name.

    ``toolset="full"`` gives every lenz-backed tool plus the harness tools;
    ``"llm_only"`` (the LLM-as-optimizer ablation) gives only ``evaluate``,
    ``incumbent``, ``record_note``, ``status`` and ``trials``. ``stop`` is included only
    when the campaign allows early stopping, ``create`` only with ``agent_setup``.
    """
    if toolset not in TOOLSETS:
        raise LenzError(f"unknown toolset {toolset!r}; choose one of {sorted(TOOLSETS)}")
    specs = _specs(session, toolset)
    names = [n for n in TOOLSETS[toolset] if n in specs]
    return sorted((specs[n] for n in names), key=lambda s: s.name)
