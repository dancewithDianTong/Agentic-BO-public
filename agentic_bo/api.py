"""One-call entry point: optimize a user's black-box function with the agentic BO loop.

    >>> from agentic_bo import optimize
    >>> def f(cfg):
    ...     return (cfg["x"] - 0.3) ** 2 + (cfg["y"] + 1.0) ** 2
    >>> res = optimize(f, {"x": (-2, 2), "y": (-3, 3)}, budget=25, f_star=0.0)
    >>> res.best_config, res.best_value, res.simple_regret[-1]

:func:`optimize` wraps ``f`` into a :class:`~agentic_bo.problems.Problem`
(:func:`make_problem`) and runs one :class:`~agentic_bo.sara.Campaign` with the chosen
policy (the scripted Sara agent by default; ``"sara"`` for Claude). ``f`` receives a
config dict and returns either a number (the objective) or a dict of metrics (the
objective under ``objective``, default ``"y"``, plus constraint and other metrics).

Failed measurements: ``f`` may return ``None`` / NaN (for the objective or any metric);
the evaluation is spent and the agent sees the metric as missing. An *exception*
raised by ``f`` is treated as a bug and aborts the optimization (it is re-raised by
:func:`optimize`) unless ``catch_errors=True``, in which case it counts as a failed
evaluation too.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping, Sequence

import numpy as np

from .problems.base import Problem
from .sara.policies.base import Policy
from .sara.session import Campaign, CampaignConfig, CampaignResult
from .space import CategoricalParam, IntParam, OrdinalParam, RealParam, SearchSpace, param_from_spec
from .utils import LenzError, make_rng

__all__ = ["optimize", "make_problem", "as_space", "BlackBox"]

_PARAM_TYPES = (RealParam, IntParam, OrdinalParam, CategoricalParam)
_PRIOR_KEYS = ("text", "config", "region")
_PREFIX = "[optimize]"


# =========================================================================== search spaces


def _is_real(v: Any) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_)) and math.isfinite(float(v))


def as_space(space: Any) -> SearchSpace:
    """Build a :class:`SearchSpace` from the forms :func:`optimize` accepts.

    Accepted: a ``SearchSpace``; a list of parameter objects or spec dicts (as
    ``SearchSpace.from_spec``); or a mapping ``{name: value}`` where each value is a
    parameter object, a spec dict (``{"type": "int", "lower": 1, "upper": 8}``) or a
    ``(lower, upper)`` pair of numbers, which becomes a real parameter.

    Raises:
        LenzError: for anything else (e.g. a pair with a non-numeric bound).
    """
    if isinstance(space, SearchSpace):
        return space
    # ``{"params": [spec, ...]}`` is SearchSpace.from_spec's form; a parameter that is
    # merely *named* "params" (e.g. ``{"params": (0, 1)}``) is not
    spec_form = (isinstance(space, Mapping) and isinstance(space.get("params"), (list, tuple))
                 and all(isinstance(p, (Mapping, *_PARAM_TYPES)) for p in space["params"]))
    if isinstance(space, Mapping) and not spec_form:
        params = []
        for name, v in space.items():
            if isinstance(v, _PARAM_TYPES):
                if v.name != name:
                    raise LenzError(f"parameter object {v.name!r} is listed under the key {name!r}")
                params.append(v)
            elif isinstance(v, Mapping):
                params.append(param_from_spec(v, name=name))
            elif isinstance(v, (tuple, list)) and len(v) == 2 and all(_is_real(b) for b in v):
                params.append(RealParam(str(name), float(v[0]), float(v[1])))
            else:
                raise LenzError(f"parameter {name!r}: expected (lower, upper), a spec dict or a parameter object, "
                                f"got {v!r}")
        return SearchSpace(params)
    return SearchSpace.from_spec(space)


# =========================================================================== the black box


class _Abort(BaseException):
    """Carries an error out of ``f`` through the harness to :func:`optimize`.

    The harness deliberately turns every ``Exception`` raised by the black box into a
    failed evaluation (a failed experiment is a legitimate outcome). A bug in a user
    function must instead stop the run at once, before more evaluations are spent, so
    it travels as a ``BaseException`` (like ``KeyboardInterrupt``), which the harness'
    ``except Exception`` guards let through.
    """

    def __init__(self, error: BaseException):
        super().__init__(repr(error))
        self.error = error


class BlackBox:
    """``config -> metrics`` adapter around a user function (the ``evaluate_fn`` of :func:`make_problem`).

    Converts the return value of ``f`` to a metrics dict: a number becomes
    ``{objective: value}``; a mapping must contain the objective and every constraint
    metric (values may be ``None`` / NaN for failed measurements). It also keeps the
    last measurement so that a simulated noisy observation of the same config
    (``Problem.observe`` right after ``Problem.evaluate``) does not call ``f`` twice.

    Attributes:
        f: the wrapped function.
        objective: metric name of the objective.
        required: metric names every result must contain.
        abort: re-raise errors from ``f`` as an abort signal (set by :func:`optimize`)
            instead of letting the harness record a failed evaluation.
        on_result: optional ``callback(config, metrics | None, error | None)`` after each call.
        n_calls: number of calls of ``f``.
    """

    def __init__(self, f: Callable[[dict], Any], objective: str = "y", required: Sequence[str] = ()):
        if not callable(f):
            raise LenzError(f"f must be callable, got {type(f).__name__}")
        self.f = f
        self.objective = objective
        self.required = list(dict.fromkeys([objective, *required]))
        self.abort = False
        self.on_result: Callable[[dict, dict | None, BaseException | None], None] | None = None
        self.n_calls = 0
        self._last: tuple[dict, dict] | None = None

    def __call__(self, config: Mapping[str, Any]) -> dict[str, float]:
        cfg = dict(config)
        self._last = None
        self.n_calls += 1
        try:
            metrics = self._convert(self.f(dict(cfg)))
        except Exception as exc:
            if self.on_result is not None:
                self.on_result(cfg, None, exc)
            if self.abort:
                raise _Abort(exc) from exc
            raise
        self._last = (cfg, metrics)
        if self.on_result is not None:
            self.on_result(cfg, metrics, None)
        return dict(metrics)

    def take_last(self, config: Mapping[str, Any]) -> dict[str, float] | None:
        """The metrics of the immediately preceding call if it was for ``config`` (single use)."""
        last, self._last = self._last, None
        if last is not None and last[0] == dict(config):
            return dict(last[1])
        return None

    def _convert(self, out: Any) -> dict[str, float]:
        if out is None:
            return {m: math.nan for m in self.required}
        if isinstance(out, Mapping):
            metrics = {str(k): _metric_value(v, k) for k, v in out.items()}
            missing = [m for m in self.required if m not in metrics]
            if missing:
                raise LenzError(f"f returned metrics {sorted(metrics)} without {missing}; return the objective "
                                f"under {self.objective!r} (or pass objective=...) and every constraint metric")
            return metrics
        if isinstance(out, np.ndarray) and out.size == 1:
            out = out.reshape(()).item()
        if isinstance(out, (bool, np.bool_)) or not isinstance(out, numbers.Real):
            raise LenzError(f"f must return a number, None or a dict of metrics, got {type(out).__name__}")
        if len(self.required) > 1:
            raise LenzError(f"f returned a single number, but the constraint metrics "
                            f"{self.required[1:]} are needed: return a dict of metrics")
        return {self.objective: float(out)}


def _metric_value(v: Any, key: Any) -> float:
    if v is None:
        return math.nan
    if isinstance(v, np.ndarray) and v.size == 1:
        v = v.reshape(()).item()
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
        raise LenzError(f"metric {key!r}: expected a number or None, got {type(v).__name__}")
    return float(v)


@dataclass
class _FunctionProblem(Problem):
    """A :class:`Problem` around a :class:`BlackBox`: a noisy observation reuses the measurement just made.

    The user's function may be noisy by itself (``noise_std`` only adds simulated noise),
    so the brief makes no claim that observations are noiseless.
    """

    deterministic: ClassVar[bool] = False

    def observe(self, config: Mapping[str, Any], rng: np.random.Generator | int | None = None) -> dict[str, float]:
        # The campaign harness adds its own noise to a single evaluate call and never calls
        # observe; a direct caller that runs evaluate and then observe for the same config
        # gets the measurement just made instead of a second run of f.
        bb = self.evaluate_fn
        last = bb.take_last(self.space.validate(config)) if isinstance(bb, BlackBox) else None
        metrics = last if last is not None else self.evaluate(config)
        if self.noise_std > 0:
            eps = make_rng(rng).standard_normal(len(metrics)) * self.noise_std
            metrics = {k: v + float(e) for (k, v), e in zip(metrics.items(), eps)}
        return metrics


# =========================================================================== problem construction


def _prior_parts(prior: Any, space: SearchSpace) -> tuple[str | None, dict | None]:
    """``(text, prior_hint)`` from a prior given as text or as ``{"text"?, "config"?, "region"?}``."""
    if prior is None:
        return None, None
    if isinstance(prior, str):
        return (prior.strip() or None), None
    if not isinstance(prior, Mapping):
        raise LenzError(f"prior must be a string or a dict with keys {_PRIOR_KEYS}, got {type(prior).__name__}")
    unknown = sorted(set(prior) - set(_PRIOR_KEYS))
    if unknown:
        raise LenzError(f"unknown prior keys {unknown}; allowed: {list(_PRIOR_KEYS)}")
    hint: dict[str, Any] = {}
    parts: list[str] = []
    if prior.get("config") is not None:
        hint["config"] = space.validate(prior["config"])
        parts.append("A promising configuration is " + ", ".join(f"{k}={v!r}" for k, v in hint["config"].items())
                     + ".")
    if prior.get("region") is not None:
        region = space.validate_bounds(prior["region"])
        if region:
            hint["region"] = region
            parts.append("Good configurations are expected in the region " + "; ".join(
                f"{k} in {v}" for k, v in region.items()) + ".")
    text = prior.get("text")
    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise LenzError("prior['text'] must be a non-empty string")
    text = text.strip() if text else " ".join(parts) or None
    return text, (hint or None)


def _problem_name(f: Any) -> str:
    name = getattr(f, "__name__", None)
    return name if isinstance(name, str) and name.isidentifier() else "objective"


def make_problem(f: Callable[[dict], Any], space: Any, *, maximize: bool = False, f_star: float | None = None,
                 description: str = "", prior: str | Mapping[str, Any] | None = None,
                 constraints: Sequence[Mapping[str, Any]] | None = None, noise_std: float = 0.0,
                 objective: str = "y", outputs: Mapping[str, str] | Sequence[str] | None = None,
                 budget: int = 50, name: str | None = None) -> Problem:
    """Wrap a user function into a :class:`~agentic_bo.problems.Problem`.

    Args:
        f: ``config dict -> number | dict of metrics | None`` (see :class:`BlackBox`).
        space: search space in any form :func:`as_space` accepts.
        maximize: the objective's raw direction (default: minimize).
        f_star: the optimal objective value in raw units, if known (enables regret).
        description: task description shown to the agent.
        prior: domain knowledge: free text, or ``{"text"?, "config"?, "region"?}`` where
            ``config`` is a promising configuration and ``region`` an active-region spec
            (``{param: [lo, hi] | [allowed values]}``); scripted agents use the
            structured parts, LLM agents read the text.
        constraints: ``[{"metric", "lower"?, "upper"?}]`` on metrics ``f`` returns.
        noise_std: Gaussian noise the harness adds to what the agent observes (a
            simulation device; keep 0 for a black box that is noisy by itself).
        objective: metric name of the objective.
        outputs: other metrics ``f`` returns that the agent may refer to (e.g. as a
            constraint added later): a list of names or ``{name: description}``.
        budget: the problem's default budget.
        name: problem name in results (default: ``f.__name__``).
    """
    sp = as_space(space)
    if not isinstance(objective, str) or not objective.strip():
        raise LenzError(f"objective must be a non-empty metric name, got {objective!r}")
    if constraints is not None and not isinstance(constraints, (list, tuple)):
        raise LenzError(f"constraints must be a list of {{metric, lower?, upper?}} objects, got {constraints!r}")
    cons = []
    for c in constraints or []:
        if not isinstance(c, Mapping) or not isinstance(c.get("metric"), str) or c["metric"] == objective.strip():
            raise LenzError(f"invalid constraint {c!r}: expected {{metric, lower?, upper?}} on a metric other "
                            f"than the objective")
        cons.append(dict(c))
    if outputs is None:
        out_desc: dict[str, str] = {}
    elif isinstance(outputs, Mapping):
        out_desc = {str(k): str(v or "") for k, v in outputs.items()}
    elif isinstance(outputs, (list, tuple)) and all(isinstance(m, str) for m in outputs):
        out_desc = {m: "" for m in outputs}
    else:
        raise LenzError(f"outputs must be a list of metric names or a {{name: description}} dict, got {outputs!r}")
    if f_star is not None and not _is_real(f_star):
        raise LenzError(f"f_star must be a finite number or None, got {f_star!r}")
    if not isinstance(description, str):
        raise LenzError("description must be a string")
    prior_text, prior_hint = _prior_parts(prior, sp)
    box = BlackBox(f, objective.strip(), [c.get("metric") for c in cons])
    return _FunctionProblem(
        name=name or _problem_name(f), space=sp, evaluate_fn=box, objective=objective.strip(),
        minimize=not bool(maximize), f_star=f_star, constraints=cons, noise_std=noise_std,
        description=description, prior=prior_text, prior_hint=prior_hint, default_budget=budget,
        outputs=out_desc, meta={"family": "user_function"},
    )


# =========================================================================== progress printing


def _short_config(cfg: Mapping[str, Any]) -> str:
    def fmt(v: Any) -> str:
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    return "{" + ", ".join(f"{k}={fmt(v)}" for k, v in cfg.items()) + "}"


class _Progress:
    """Prints one line per evaluation (called from the black box)."""

    def __init__(self, problem: Problem, budget: int, stream: Any = None):
        self.problem = problem
        self.budget = budget
        self.stream = stream
        self.n = 0
        self.best: float | None = None
        self.started = time.perf_counter()

    def say(self, text: str) -> None:
        print(f"{_PREFIX} {text}", file=self.stream or sys.stdout, flush=True)

    def __call__(self, cfg: dict, metrics: dict | None, error: BaseException | None) -> None:
        self.n += 1
        p = self.problem
        head = f"eval {self.n:>{len(str(self.budget))}}/{self.budget}"
        if error is not None:
            self.say(f"{head}  failed: {type(error).__name__}: {error}")
            return
        assert metrics is not None
        value = metrics.get(p.objective, math.nan)
        feasible = p.is_feasible(metrics)
        u = p.utility(metrics)
        better = feasible and math.isfinite(u) and (self.best is None or u > self.best)
        if better:
            self.best = u
        best = "n/a" if self.best is None else f"{(-self.best if p.minimize else self.best):.6g}"
        tag = "" if feasible else "  (infeasible)"
        line = f"{head}  {p.objective}={value:.6g}{tag}  best={best}"
        if better:
            line += f"  * {_short_config(cfg)}"
        self.say(line)


# =========================================================================== optimize


def optimize(f: Callable[[dict], Any], space: Any, budget: int, *, maximize: bool = False,
             policy: str | Policy = "heuristic", seed: int = 0, f_star: float | None = None,
             description: str = "", prior: str | Mapping[str, Any] | None = None,
             constraints: Sequence[Mapping[str, Any]] | None = None, noise_std: float = 0.0,
             context: Any = None, regret: str = "simple", policy_kwargs: Mapping[str, Any] | None = None,
             verbose: bool = False, objective: str = "y",
             outputs: Mapping[str, str] | Sequence[str] | None = None, catch_errors: bool = False,
             campaign_options: Mapping[str, Any] | None = None) -> CampaignResult:
    """Optimize a black-box function with the agentic BO loop and return the campaign result.

    Args:
        f: ``config dict -> number | dict of metrics | None``. A number is the
            objective; a dict holds the objective under ``objective`` plus constraint
            and other metrics; ``None`` / NaN marks a failed measurement.
        space: search space (``SearchSpace``, a list of parameter specs, or a mapping
            ``{name: (lower, upper) | spec dict | parameter}``; see :func:`as_space`).
        budget: number of evaluations of ``f``.
        maximize: maximize the objective (default: minimize).
        policy: policy name (``heuristic``, ``fixed_bo``, ``sobol``, ``random``, ``sara``,
            ``llm_only``) or a :class:`~agentic_bo.sara.policies.Policy` instance.
        seed: campaign seed (lenz randomness, noise, scripted policies).
        f_star: optimal objective value (raw units), if known; enables regret curves.
        description: task description shown to the agent.
        prior: domain knowledge (text, or ``{"text"?, "config"?, "region"?}``; see
            :func:`make_problem`).
        constraints: ``[{"metric", "lower"?, "upper"?}]`` on metrics returned by ``f``.
        noise_std: simulated Gaussian observation noise (regret stays noise-free).
        context: initial user context for the agent: a string, a ``{text, source?,
            data?}`` dict or a list of them (``data`` such as ``{"op": "set_bounds",
            "args": {...}}`` is applied by the scripted agent).
        regret: ``"simple"`` or ``"cumulative"``, what the agent is told to minimize.
        policy_kwargs: keyword arguments for the policy constructor (policy names only).
        verbose: print one progress line per evaluation and a final summary.
        objective: metric name of the objective in ``f``'s output.
        outputs: other metrics ``f`` returns that the agent may use (names or
            ``{name: description}``).
        catch_errors: count exceptions raised by ``f`` as failed evaluations instead
            of aborting and re-raising them.
        campaign_options: further :class:`~agentic_bo.sara.CampaignConfig` fields, e.g.
            ``{"allow_early_stop": True, "context_schedule": {10: "..."}}``.

    Returns:
        The :class:`~agentic_bo.sara.CampaignResult`: ``best_config`` / ``best_value``
        (best feasible evaluated point, raw units), ``records`` (every evaluation),
        regret curves (NaN without ``f_star``), the agent's log and notebook.

    Raises:
        LenzError: invalid arguments.
        Exception: whatever ``f`` raised (unless ``catch_errors``).
    """
    if isinstance(budget, bool) or not isinstance(budget, numbers.Integral) or int(budget) < 1:
        raise LenzError(f"budget must be an integer >= 1, got {budget!r}")
    budget = int(budget)
    problem = make_problem(f, space, maximize=maximize, f_star=f_star, description=description, prior=prior,
                           constraints=constraints, noise_std=noise_std, objective=objective, outputs=outputs,
                           budget=budget)
    if isinstance(policy, Policy):
        if policy_kwargs:
            raise LenzError("policy_kwargs only applies when the policy is given by name")
        pol = policy
    elif isinstance(policy, str):
        from .sara.policies import get_policy
        pol = get_policy(policy, **dict(policy_kwargs or {}))
    else:
        raise LenzError(f"policy must be a policy name or a Policy instance, got {type(policy).__name__}")

    options = dict(campaign_options or {})
    clash = sorted({"budget", "seed", "regret", "context"} & set(options))
    if clash:
        raise LenzError(f"campaign_options must not repeat the arguments {clash}; pass them directly")
    known = {fld.name for fld in dataclasses.fields(CampaignConfig)}
    unknown = sorted(set(options) - known)
    if unknown:
        raise LenzError(f"unknown campaign_options {unknown}; CampaignConfig fields are {sorted(known)}")
    config = CampaignConfig(budget=budget, seed=seed, regret=regret, context=context, **options)

    box: BlackBox = problem.evaluate_fn  # type: ignore[assignment]
    box.abort = not catch_errors
    progress: _Progress | None = None
    if verbose:
        progress = _Progress(problem, budget)
        box.on_result = progress
        progress.say(f"{pol.name} on {problem.name!r} ({problem.space.d} parameter"
                     f"{'s' if problem.space.d != 1 else ''}), budget {budget}, {regret} regret, "
                     f"{'maximize' if maximize else 'minimize'} {problem.objective!r}")
    try:
        result = Campaign(problem, pol, config).run()
    except _Abort as abort:
        raise abort.error from None
    finally:
        box.on_result = None
        box.abort = False
    if progress is not None:
        _summary(progress, result)
    return result


def _summary(progress: _Progress, result: CampaignResult) -> None:
    p = progress.problem
    if result.best_value is None:
        best = "no feasible evaluation"
    else:
        best = f"best {p.objective}={result.best_value:.6g} at {_short_config(result.best_config or {})}"
    parts = [f"done in {result.wallclock_s:.1f} s: {best}"]
    if p.f_star is not None and result.simple_regret:
        parts.append(f"simple regret {result.final_simple_regret:.3g}, "
                     f"cumulative regret {result.final_cumulative_regret:.3g}")
    if result.forced_commits:
        parts.append(f"{result.forced_commits} harness fallback commit(s)")
    if result.policy_error:
        parts.append(f"policy error: {result.policy_error}")
    progress.say("; ".join(parts))
