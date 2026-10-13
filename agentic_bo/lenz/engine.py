"""The lenz engine (DESIGN.md §7.2): trial log, surrogate cache and the command interface.

:class:`Lenz` owns a :class:`~agentic_bo.lenz.frame.Frame` and implements every command an
agent (or the CLI) can issue:

* **probe** — ``status``, ``trials``, ``incumbent``, ``pareto``, ``diagnostics``,
  ``predict``, ``score`` (read-only),
* **reconfigure** — ``set_bounds``, ``set_acqf``, ``set_objectives``, ``set_constraints``
  (mutate the shelf ``c_t``; never drop data),
* **propose** — ``suggest`` (acquisition optimization; records nothing),
* **record** — ``submit`` / ``observe`` / ``abandon`` (the only commands that touch ``D_t``).

Modelling conventions. One GP per metric is fitted on the *raw* values of the completed
trials that report a finite value; minimized objectives are negated by a thin wrapper
when the acquisition is built, so everything returned is in raw units and direction
while all acquisition values are "higher is better". Fits are cached per (data content,
model settings) and warm-started from the previous fit; the warm-start chain is part of
the persisted frame, so a saved and reloaded study refits to bit-identical models.
Pending trials and, for ``q > 1``, earlier picks are fantasized at the posterior mean
(kriging believer). Randomness comes from per-call generators derived from
``(seed, counter)``; the space-filling design continues one scrambled Sobol sequence
whose position is persisted too.

Every command returns a JSON-safe dict (``utils.jsonable``) and appends an
:class:`~agentic_bo.lenz.frame.Event`; :meth:`Lenz.call` wraps commands in the
``{"ok", "command", "result" | "error"}`` envelope.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..space import CategoricalParam, IntParam, OrdinalParam, RealParam, SearchSpace, sobol_points
from ..utils import LenzError, dumps, jsonable, make_rng
from .acquisition import (
    ACQF_NAMES, DEFAULT_UCB_BETA, EHVI_N_SAMPLES, Acquisition, MODEL_FREE_ACQFS, MULTI_OBJECTIVE_ACQFS,
    POINTWISE_ACQFS, SINGLE_OBJECTIVE_ACQFS, build_acquisition, parego_scalarize, parego_weights,
    prob_feasible,
)
from .frame import (
    COUNTER_KEYS, Event, Frame, Shelf, Trial, metric_value, normalize_constraints, normalize_model,
    normalize_objectives,
)
from .gp import GP
from .optimize import optimize_acqf, project_box, thompson_sample, trust_region
from .pareto import default_ref_point, hypervolume, pareto_mask

__all__ = ["Lenz", "COMMANDS", "ACQF_PARAMS", "DUPLICATE_TOL", "DEFAULT_RADIUS", "MAX_Q", "STUDY_FORMAT"]

STUDY_FORMAT = "agentic_bo.lenz/study"
STUDY_VERSION = 1
#: commands reachable through :meth:`Lenz.call` (``create`` is the constructor)
COMMANDS = ("suggest", "submit", "observe", "abandon", "set_bounds", "set_acqf", "set_objectives",
            "set_constraints", "diagnostics", "predict", "score", "incumbent", "pareto", "trials",
            "status")
#: parameters each acquisition function accepts (``set_acqf`` / ``suggest(acqf_params=...)``)
ACQF_PARAMS: dict[str, tuple[str, ...]] = {
    "ucb": ("beta",),
    "ehvi": ("ref_point", "n_samples"),
    "parego": ("weights", "rho"),
    "ts": ("n_candidates",),
}
DUPLICATE_TOL = 1e-6        # RMS encoded distance below which two configs count as the same point
DEFAULT_RADIUS = 0.1        # trust-region half-width (encoded units, before lengthscale scaling)
MAX_Q = 100
_MAX_RADIUS = 10.0
_PAREGO_TAG = "__parego__"  # model-cache / warm-start tag of the ParEGO scalarized GP
_SALT_CALL = 101            # stream of per-call generators
_SALT_SOBOL = 202           # the study's space-filling Sobol sequence
_ARG_MAX_CHARS = 2000       # event args longer than this (JSON) are truncated
_ARG_PREVIEW = 300
_SUMMARY_CHARS = 240
_SF_MAX_EXAMINE = 4096      # space-filling points examined per call before giving up on novelty
# acquisitions that reduce to (log) PoF alone while no feasible observation exists
_PURE_POF_IN_FEASIBILITY_PHASE = ("logei", "noisy_logei", "ei", "logpi", "pi", "ehvi", "parego")


# --------------------------------------------------------------------------- small helpers


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _check_int(v: Any, name: str, lo: int, hi: int) -> int:
    if not _is_number(v) or not math.isfinite(float(v)) or float(v) != int(v):
        raise LenzError(f"{name} must be an integer, got {v!r}")
    i = int(v)
    if not lo <= i <= hi:
        raise LenzError(f"{name} must be in [{lo}, {hi}], got {i}")
    return i


def _check_float(v: Any, name: str, lo: float | None = None, hi: float | None = None,
                 lo_open: bool = False) -> float:
    if not _is_number(v) or not math.isfinite(float(v)):
        raise LenzError(f"{name} must be a finite number, got {v!r}")
    x = float(v)
    if lo is not None and (x < lo or (lo_open and x <= lo)):
        raise LenzError(f"{name} must be {'>' if lo_open else '>='} {lo}, got {x}")
    if hi is not None and x > hi:
        raise LenzError(f"{name} must be <= {hi}, got {x}")
    return x


def _default_acqf(n_objectives: int) -> str:
    return "logei" if n_objectives == 1 else ("ehvi" if n_objectives == 2 else "parego")


def _check_acqf(name: Any, n_objectives: int) -> str:
    """Normalize an acquisition name and check it fits the number of objectives."""
    if not isinstance(name, str) or not name.strip():
        raise LenzError(f"acqf must be a non-empty string, got {name!r}")
    key = name.strip().lower()
    if key not in ACQF_NAMES:
        raise LenzError(f"unknown acquisition function {name!r}; choose one of: {', '.join(ACQF_NAMES)}")
    if n_objectives == 1 and key in MULTI_OBJECTIVE_ACQFS:
        raise LenzError(f"{key!r} is a multi-objective acquisition but the study has 1 objective; "
                        f"use one of: {', '.join(SINGLE_OBJECTIVE_ACQFS + MODEL_FREE_ACQFS)}")
    if n_objectives >= 2 and key in SINGLE_OBJECTIVE_ACQFS:
        raise LenzError(f"{key!r} is a single-objective acquisition but the study has {n_objectives} "
                        f"objectives; use 'ehvi' (2 objectives), 'parego', 'sobol' or 'random', "
                        f"or reduce the objectives with set_objectives")
    if key == "ehvi" and n_objectives > 2:
        raise LenzError(f"'ehvi' supports exactly 2 objectives (the study has {n_objectives}); use 'parego'")
    return key


def _check_acqf_params(name: str, params: Any, n_objectives: int) -> dict:
    """Validate the parameters of acquisition ``name``; ``None`` values mean "default"."""
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise LenzError(f"acqf_params must be an object, got {params!r}")
    allowed = ACQF_PARAMS.get(name, ())
    out: dict[str, Any] = {}
    for k, v in params.items():
        if v is None:
            continue
        if k not in allowed:
            hint = f"it accepts {list(allowed)}" if allowed else "it takes no parameters"
            raise LenzError(f"acquisition {name!r} does not take parameter {k!r}; {hint}")
        if k == "beta":
            out[k] = _check_float(v, "ucb beta", lo=0.0)
        elif k == "rho":
            out[k] = _check_float(v, "parego rho", lo=0.0)
        elif k == "n_samples":
            out[k] = _check_int(v, "ehvi n_samples", 1, 4096)
        elif k == "n_candidates":
            out[k] = _check_int(v, "ts n_candidates", 16, 20000)
        elif k in ("ref_point", "weights"):
            if not isinstance(v, (list, tuple, np.ndarray)) or len(v) != n_objectives:
                raise LenzError(f"{name} {k} must be a list of {n_objectives} numbers (one per objective), got {v!r}")
            vals = [_check_float(x, f"{name} {k} entry") for x in v]
            if k == "weights":
                if any(x < 0 for x in vals) or sum(vals) <= 0:
                    raise LenzError(f"parego weights must be non-negative with a positive sum, got {vals}")
                total = sum(vals)
                vals = [x / total for x in vals]
            out[k] = vals
    return out


def _canonical_command(command: Any) -> str:
    if not isinstance(command, str) or not command.strip():
        raise LenzError(f"command must be a non-empty string, got {command!r}")
    key = command.strip().lower().replace("-", "_")
    if key == "create":
        raise LenzError("'create' starts a new study (Lenz.create(...) or the CLI 'create' command); "
                        "this study already exists")
    if key not in COMMANDS:
        raise LenzError(f"unknown command {command!r}; available: {', '.join(COMMANDS)}")
    return key


def _event_args(args: Mapping[str, Any]) -> dict:
    """JSON-safe copy of command arguments; oversized values are replaced by a preview."""
    out: dict[str, Any] = {}
    for k, v in args.items():
        try:
            jv = jsonable(v)
            text = json.dumps(jv, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            jv, text = repr(v), repr(v)
        if len(text) > _ARG_MAX_CHARS:
            jv = {"truncated": True, "chars": len(text), "preview": text[:_ARG_PREVIEW]}
        out[str(k)] = jv
    return out


def _shorten(text: str, n: int = _SUMMARY_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _error_message(exc: BaseException) -> str:
    msg = str(exc)
    if isinstance(exc, KeyError) and len(exc.args) == 1:
        msg = f"missing key {exc.args[0]!r}"
    return msg or type(exc).__name__


def _fmt(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _within(values: np.ndarray, lower: float | None, upper: float | None) -> np.ndarray:
    ok = np.isfinite(values)
    if lower is not None:
        ok &= values >= lower
    if upper is not None:
        ok &= values <= upper
    return ok


def _from_unit(p: RealParam | IntParam, u: float) -> float:
    u = min(max(float(u), 0.0), 1.0)
    if p.log_scale:
        return math.exp(math.log(p.lower) + u * (math.log(p.upper) - math.log(p.lower)))
    return p.lower + u * (p.upper - p.lower)


def _to_unit(p: RealParam | IntParam, v: float) -> float:
    if p.log_scale:
        return (math.log(v) - math.log(p.lower)) / (math.log(p.upper) - math.log(p.lower))
    return (v - p.lower) / (p.upper - p.lower)


def _believe(gp: GP, Zf: np.ndarray) -> tuple[GP, np.ndarray]:
    """Kriging believer: condition ``gp`` on its own posterior mean at ``Zf`` (no refit)."""
    if Zf.shape[0] == 0:
        return gp, np.zeros(0)
    mu = gp.posterior(Zf)[0]
    return gp.fantasize(Zf, mu), mu


class _Signed:
    """Sign-adjusted view of a GP (``sign=-1`` turns a minimized metric into a maximized one)."""

    def __init__(self, gp: GP, sign: float):
        self.gp = gp
        self.sign = float(sign)

    @property
    def X_train(self) -> np.ndarray:
        return self.gp.X_train

    @property
    def y_train(self) -> np.ndarray:
        return self.sign * self.gp.y_train

    def posterior(self, Xs: np.ndarray, full_cov: bool = False, return_grad: bool = False,
                  observation_noise: bool = False):
        out = self.gp.posterior(Xs, full_cov=full_cov, return_grad=return_grad,
                                observation_noise=observation_noise)
        if return_grad:
            mean, var, dmean, dvar = out
            return self.sign * mean, var, self.sign * dmean, dvar
        mean, var = out
        return self.sign * mean, var

    def sample(self, Xs: np.ndarray, n_samples: int, rng: Any) -> np.ndarray:
        return self.sign * self.gp.sample(Xs, n_samples, rng)


def _signed(gp: GP, sign: float) -> Any:
    return gp if sign > 0 else _Signed(gp, sign)


@dataclass
class _Data:
    """Arrays derived from the trial log for one decision."""

    completed: list[Trial]
    pending: list[Trial]
    Zc: np.ndarray            # (n, D) encoded completed trials
    Zp: np.ndarray            # (k, D) encoded pending trials
    Yraw: np.ndarray          # (n, m) raw objective values (NaN = missing / non-finite)
    Craw: np.ndarray          # (n, c) raw constraint values
    signs: np.ndarray         # (m,) +1 maximize, -1 minimize
    valid: np.ndarray         # (n,) every objective finite
    feasible: np.ndarray      # (n,) every constraint satisfied

    @property
    def Y(self) -> np.ndarray:
        """Objectives in the maximize convention."""
        return self.Yraw * self.signs

    @property
    def ok(self) -> np.ndarray:
        """Rows usable as incumbents: all objectives finite and every constraint satisfied."""
        return self.valid & self.feasible


@dataclass
class _Bundle:
    """An acquisition for one greedy step plus what Thompson sampling needs."""

    acq: Any | None
    model: Any | None
    triples: list
    feasibility_phase: bool
    best_f: float | None
    log_values: bool = False      # ``acq`` returns the log of the reported value (empty-front EHVI)


class _EmptyFrontEHVI(Acquisition):
    """Closed-form log EHVI while no front point dominates the reference point.

    With an empty front, ``HVI(y) = prod_i (y_i - r_i)^+`` (maximize convention), so for
    independent objective posteriors ``EHVI(x) = prod_i EI_i(x; best_f = r_i)`` exactly.
    The sum of the per-objective ``logei`` terms (the first carries ``sum_j log PoF_j``)
    is its log: finite and differentiable everywhere, where the Monte Carlo estimate is
    exactly 0 wherever no base sample lands beyond the reference point.
    """

    name = "ehvi"
    has_gradient = True

    def __init__(self, parts: Sequence[Any]):
        self.parts = list(parts)

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        return np.sum([np.asarray(p(Z), dtype=float) for p in self.parts], axis=0)

    def value_and_grad(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vals, grads = zip(*(p.value_and_grad(Z) for p in self.parts))
        return np.sum(vals, axis=0), np.sum(grads, axis=0)


# --------------------------------------------------------------------------- command decorator


def _command(fn: Callable) -> Callable:
    """Log one :class:`Event` per top-level command call and make the result JSON-safe."""
    name = fn.__name__
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(self: "Lenz", *args: Any, **kwargs: Any) -> Any:
        if self._depth:
            return fn(self, *args, **kwargs)
        self._depth += 1
        self._seq_now = int(self._frame.counters["seq"]) + 1
        try:
            result = jsonable(fn(self, *args, **kwargs))
        except Exception as exc:
            self._record(name, _bound_args(sig, args, kwargs), False,
                         f"{type(exc).__name__}: {_error_message(exc)}")
            raise
        finally:
            self._depth -= 1
        self._record(name, _bound_args(sig, args, kwargs), True, _summarize(name, result))
        return result

    return wrapper


def _bound_args(sig: inspect.Signature, args: tuple, kwargs: dict) -> dict:
    try:
        bound = sig.bind(None, *args, **kwargs)
    except TypeError:
        return {"args": list(args), **kwargs}
    out: dict[str, Any] = {}
    for k, v in list(bound.arguments.items())[1:]:
        kind = sig.parameters[k].kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            out.update(v)
        elif kind is inspect.Parameter.VAR_POSITIONAL:
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _summarize(command: str, r: Mapping[str, Any]) -> str:
    try:
        if command == "suggest":
            cands = r.get("candidates") or []
            s = f"{len(cands)} candidate(s), mode={r.get('mode')}, acqf={r.get('acqf')}"
            if r.get("region", {}).get("around") is not None:
                s += f", trust region radius {_fmt(r['region'].get('radius'))}"
        elif command == "submit":
            s = f"trial {r.get('trial_id')} pending"
        elif command == "observe":
            s = f"trial {r.get('trial_id')} completed (eval {r.get('eval_index')}): " + ", ".join(
                f"{k}={_fmt(v)}" for k, v in sorted((r.get("metrics") or {}).items()))
            if r.get("is_new_incumbent"):
                s += " [new incumbent]"
        elif command == "abandon":
            s = f"trial {r.get('trial_id')} abandoned"
        elif command == "set_bounds":
            b = r.get("bounds")
            s = "bounds reset to the full domain" if not b else f"bounds on {sorted(b)}"
            s += f" ({r.get('n_in_region')} of {r.get('observed')} observations inside)"
        elif command == "set_acqf":
            s = f"acqf={r.get('acqf')} params={json.dumps(r.get('acqf_params'), sort_keys=True)}"
        elif command == "set_objectives":
            s = "objectives " + ", ".join(
                f"{'min' if o.get('minimize') else 'max'} {o.get('metric')}" for o in r.get("objectives") or [])
            s += f"; acqf={r.get('acqf')}"
        elif command == "set_constraints":
            cs = r.get("constraints") or []
            s = "no constraints" if not cs else "constraints " + ", ".join(
                f"{_fmt(c.get('lower'))} <= {c.get('metric')} <= {_fmt(c.get('upper'))}" for c in cs)
        elif command == "diagnostics":
            parts = []
            for k, v in sorted((r.get("models") or {}).items()):
                parts.append(f"{k}: n={v.get('n_observed')} cv_r2={_fmt(v.get('cv_r2'))}")
            s = "; ".join(parts) or "no models"
        elif command in ("predict", "score"):
            s = f"{len(r.get('predictions') or r.get('scores') or [])} config(s)"
        elif command == "incumbent":
            inc = r.get("incumbent")
            if r.get("is_moo"):
                s = f"{len(r.get('pareto') or [])} Pareto point(s), hypervolume {_fmt(r.get('hypervolume'))}"
            elif inc:
                s = f"trial {inc.get('trial_id')} value {_fmt(inc.get('value'))}"
            else:
                s = "no feasible completed trial"
        elif command == "pareto":
            s = f"{len(r.get('pareto') or [])} Pareto point(s), hypervolume {_fmt(r.get('hypervolume'))}"
        elif command == "trials":
            s = f"{r.get('n')} of {r.get('n_total')} trial(s)"
        elif command == "status":
            s = f"{r.get('n_observed')} observed, {r.get('n_pending')} pending"
        else:  # pragma: no cover - every command is listed above
            s = command
    except (AttributeError, TypeError, KeyError):  # pragma: no cover - defensive
        s = command
    return _shorten(s)


# --------------------------------------------------------------------------- the engine


class Lenz:
    """A Bayesian-optimization study driven through structured commands.

    Args:
        space: a :class:`~agentic_bo.space.SearchSpace` or its spec.
        objectives: a metric name (maximized), one ``{"metric", "minimize", "ref"?}`` object
            or a list of them (``ref`` = hypervolume reference value, raw units).
        constraints: ``[{"metric", "lower", "upper"}]`` (feasible iff ``lower <= v <= upper``).
        acqf: acquisition function (default ``logei`` / ``ehvi`` / ``parego`` for 1 / 2 / >2
            objectives); see :data:`~agentic_bo.lenz.acquisition.ACQF_NAMES`.
        acqf_params: its parameters (:data:`ACQF_PARAMS`).
        bounds: active region (``{param: [lo, hi] | [allowed...]}``) or ``None`` = full domain.
        model: GP settings ``{"kernel": "matern52" | "rbf", "noise": "learn" | float, "n_restarts": int}``.
        seed: study seed; all randomness derives from it.
    """

    def __init__(self, space: Any, objectives: Any, constraints: Any = None, acqf: str | None = None,
                 acqf_params: Mapping[str, Any] | None = None, bounds: Any = None,
                 model: Mapping[str, Any] | None = None, seed: int = 0):
        space = SearchSpace.from_spec(space)
        objs = normalize_objectives(objectives)
        cons = normalize_constraints(constraints)
        name = _check_acqf(_default_acqf(len(objs)) if acqf is None else acqf, len(objs))
        params = _check_acqf_params(name, acqf_params, len(objs))
        region = space.validate_bounds(bounds)
        mdl = normalize_model(model)
        if isinstance(seed, bool) or not _is_number(seed) or float(seed) != int(seed):
            raise LenzError(f"seed must be an integer, got {seed!r}")
        shelf = Shelf(objectives=objs, constraints=cons, acqf=name, acqf_params=params,
                      bounds=region, model=mdl)
        self._frame = Frame(space=space, shelf=shelf, seed=int(seed))
        self._init_runtime()
        self._record("create", {"space": space.to_spec(), "objectives": objs, "constraints": cons,
                                "acqf": name, "acqf_params": params, "bounds": region,
                                "model": mdl, "seed": int(seed)}, True,
                     _shorten(f"created study: {space.d} parameter(s) ({space.D} encoded dims), "
                              f"objectives {[o['metric'] for o in objs]}, "
                              f"{len(cons)} constraint(s), acqf {name}"))

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> "Lenz":
        """Alias of the constructor (symmetry with the CLI ``create`` command)."""
        return cls(*args, **kwargs)

    def _init_runtime(self) -> None:
        self._gp_cache: dict[str, tuple[str, GP]] = {}
        self._z_cache: dict[int, np.ndarray] = {}
        self._depth = 0
        self._seq_now = int(self._frame.counters.get("seq", 0)) + 1

    # ---- properties -------------------------------------------------------------------
    @property
    def frame(self) -> Frame:
        return self._frame

    @property
    def space(self) -> SearchSpace:
        return self._frame.space

    @property
    def shelf(self) -> Shelf:
        return self._frame.shelf

    @property
    def seed(self) -> int:
        return self._frame.seed

    @property
    def n_observed(self) -> int:
        return sum(1 for t in self._frame.trials if t.status == "completed")

    @property
    def n_pending(self) -> int:
        return sum(1 for t in self._frame.trials if t.status == "pending")

    def __repr__(self) -> str:
        return (f"Lenz(d={self.space.d}, objectives={self.shelf.objective_metrics}, "
                f"acqf={self.shelf.acqf!r}, n_observed={self.n_observed}, n_pending={self.n_pending})")

    # ---- event log --------------------------------------------------------------------
    def _record(self, command: str, args: Mapping[str, Any], ok: bool, summary: str) -> None:
        seq = int(self._frame.counters.get("seq", 0)) + 1
        self._frame.counters["seq"] = seq
        self._frame.events.append(Event(seq=seq, command=command, args=_event_args(args), ok=bool(ok),
                                        summary=_shorten(summary)))

    def _next_rng(self) -> np.random.Generator:
        self._frame.counters["rng"] = int(self._frame.counters.get("rng", 0)) + 1
        return make_rng(self.seed, _SALT_CALL, self._frame.counters["rng"])

    # ---- data ---------------------------------------------------------------------------
    def _z(self, trial: Trial) -> np.ndarray:
        z = self._z_cache.get(trial.trial_id)
        if z is None:
            z = self.space.encode(trial.config)
            self._z_cache[trial.trial_id] = z
        return z

    def _encode_trials(self, trials: Sequence[Trial]) -> np.ndarray:
        if not trials:
            return np.zeros((0, self.space.D))
        return np.vstack([self._z(t) for t in trials])

    def _data(self) -> _Data:
        shelf = self.shelf
        completed = self._frame.by_status("completed")
        pending = self._frame.by_status("pending")

        def column(metric: str) -> np.ndarray:
            vals = [metric_value(t.metrics, metric) for t in completed]
            return np.array([np.nan if v is None else v for v in vals], dtype=float)

        n = len(completed)
        Yraw = (np.column_stack([column(o["metric"]) for o in shelf.objectives])
                if n else np.zeros((0, len(shelf.objectives))))
        Craw = (np.column_stack([column(c["metric"]) for c in shelf.constraints])
                if n and shelf.constraints else np.zeros((n, len(shelf.constraints))))
        feasible = np.ones(n, dtype=bool)
        for j, c in enumerate(shelf.constraints):
            feasible &= _within(Craw[:, j], c["lower"], c["upper"])
        signs = np.array([-1.0 if o["minimize"] else 1.0 for o in shelf.objectives])
        return _Data(completed=completed, pending=pending, Zc=self._encode_trials(completed),
                     Zp=self._encode_trials(pending), Yraw=Yraw, Craw=Craw, signs=signs,
                     valid=np.all(np.isfinite(Yraw), axis=1) if n else np.zeros(0, dtype=bool),
                     feasible=feasible)

    def _is_feasible(self, metrics: Mapping[str, Any] | None) -> bool:
        for c in self.shelf.constraints:
            v = metric_value(metrics, c["metric"])
            if v is None or not bool(_within(np.array([v]), c["lower"], c["upper"])[0]):
                return False
        return True

    def _metric_xy(self, metric: str) -> tuple[np.ndarray, np.ndarray]:
        rows = [(t, metric_value(t.metrics, metric)) for t in self._frame.by_status("completed")]
        rows = [(t, v) for t, v in rows if v is not None]
        if not rows:
            return np.zeros((0, self.space.D)), np.zeros(0)
        return self._encode_trials([t for t, _ in rows]), np.array([v for _, v in rows], dtype=float)

    def _known_metrics(self) -> list[str]:
        names = list(self.shelf.modelled_metrics)
        for t in self._frame.trials:
            for k in (t.metrics or {}):
                if k not in names:
                    names.append(k)
        return names

    # ---- models -------------------------------------------------------------------------
    def _fit(self, tag: str, X: np.ndarray, y: np.ndarray) -> GP:
        """GP for ``tag`` on ``(X, y)``: cached by content, warm-started via ``model_state``.

        The warm start used for a given data key is persisted with it, so a reloaded study
        refits that key to the identical model (GP fits are deterministic given their inputs).
        """
        mdl = self.shelf.model
        h = hashlib.sha1()
        h.update(f"{tag}|{mdl['kernel']}|{mdl['noise']!r}|{mdl['n_restarts']}|{mdl['prior']}|{self.seed}".encode())
        h.update(np.ascontiguousarray(X, dtype=float).tobytes())
        h.update(np.ascontiguousarray(y, dtype=float).tobytes())
        key = h.hexdigest()
        hit = self._gp_cache.get(tag)
        if hit is not None and hit[0] == key:
            return hit[1]
        state = self._frame.model_state.get(tag) or {}
        warm = state.get("warm_start") if state.get("key") == key else state.get("hyperparameters")
        gp = GP(kernel=mdl["kernel"], noise=mdl["noise"], n_restarts=mdl["n_restarts"], prior=mdl["prior"],
                seed=abs(int(self.seed))).fit(X, y, warm_start=warm)
        self._frame.model_state[tag] = {"key": key, "n": int(y.shape[0]), "warm_start": warm,
                                        "hyperparameters": jsonable(gp.hyperparameters())}
        self._gp_cache[tag] = (key, gp)
        return gp

    def _gp(self, metric: str) -> GP | None:
        """Base GP of a metric on the completed trials reporting it (``None`` without data)."""
        X, y = self._metric_xy(metric)
        if y.shape[0] == 0:
            return None
        return self._fit(f"metric:{metric}", X, y)

    def _require_gp(self, metric: str) -> GP:
        gp = self._gp(metric)
        if gp is None:
            raise LenzError(f"no completed trial reports metric {metric!r} yet; the surrogate needs data")
        return gp

    def _ref_point(self, data: _Data, params: Mapping[str, Any] | None) -> np.ndarray | None:
        """Hypervolume reference point in the maximize convention (``None`` if undetermined).

        Per objective: its ``ref`` (raw units), else ``params["ref_point"]`` (raw units), else
        :func:`pareto.default_ref_point` over the completed observations.
        """
        objs = self.shelf.objectives
        ref = np.full(len(objs), np.nan)
        rp = (params or {}).get("ref_point")
        for i, o in enumerate(objs):
            if o.get("ref") is not None:
                ref[i] = data.signs[i] * o["ref"]
            elif rp is not None:
                ref[i] = data.signs[i] * float(rp[i])
        if np.any(np.isnan(ref)):
            Y = data.Y[data.valid]
            if Y.shape[0] == 0:
                return None
            ref = np.where(np.isnan(ref), default_ref_point(Y), ref)
        return ref

    def _prepare(self, name: str, params: Mapping[str, Any], data: _Data, Zf: np.ndarray,
                 weights: np.ndarray | None = None) -> _Bundle:
        """Acquisition ``name`` with every model conditioned on the fantasy points ``Zf``.

        Kriging believer: each GP is conditioned on its own posterior mean at ``Zf``
        (pending trials, then earlier picks of the batch). Believed points that the
        constraint means call feasible also update the incumbent / Pareto front.
        """
        shelf = self.shelf
        k = Zf.shape[0]
        triples: list = []
        fant_ok = np.ones(k, dtype=bool)
        for c in shelf.constraints:
            gp = self._gp(c["metric"])
            if gp is None:
                continue
            gpf, mu = _believe(gp, Zf)
            triples.append((gpf, c["lower"], c["upper"]))
            fant_ok &= _within(mu, c["lower"], c["upper"])
        # incumbent rows: feasible ones when feasibility can be modelled, else every valid row;
        # a believed point whose constraint means are feasible ends the feasibility phase too
        # (otherwise a batch would keep maximizing PoF, which the believer has just sharpened
        # around the first pick, and collapse onto it)
        rows = data.ok if triples else data.valid
        feas_phase = bool(triples) and not bool(rows.any()) and not bool(fant_ok.any())

        if name in SINGLE_OBJECTIVE_ACQFS:
            o = shelf.objectives[0]
            s = float(data.signs[0])
            gpf, mu = _believe(self._require_gp(o["metric"]), Zf)
            model = _signed(gpf, s)
            best = None
            if not feas_phase:
                incumbents = np.concatenate([s * data.Yraw[rows, 0], s * mu[fant_ok]])
                if incumbents.size:
                    best = float(np.max(incumbents))
            if name == "ts":
                return _Bundle(None, model, triples, feas_phase, best)
            X_obs = np.vstack([data.Zc[rows], Zf[fant_ok]]) if not feas_phase else None
            acq = build_acquisition(name, [model], triples or None, best_f=best, X_observed=X_obs,
                                    params=params)
            return _Bundle(acq, model, triples, feas_phase, best)

        if name == "ehvi":
            models, Yf = [], np.zeros((k, len(shelf.objectives)))
            for i, o in enumerate(shelf.objectives):
                gpf, mu = _believe(self._require_gp(o["metric"]), Zf)
                models.append(_signed(gpf, data.signs[i]))
                Yf[:, i] = data.signs[i] * mu
            if feas_phase:
                Y_obs = np.zeros((0, 2))
            else:
                Y_obs = np.vstack([data.Y[rows], Yf[fant_ok]])
            p: dict[str, Any] = {"n_samples": params.get("n_samples", EHVI_N_SAMPLES)}
            ref = self._ref_point(data, params)
            if ref is not None and not feas_phase and not np.any(np.all(Y_obs > ref, axis=1)):
                # hypervolume 0: EHVI has a closed form (see _EmptyFrontEHVI)
                parts = [build_acquisition("logei", [mdl], (triples or None) if i == 0 else None,
                                           best_f=float(ref[i])) for i, mdl in enumerate(models)]
                return _Bundle(_EmptyFrontEHVI(parts), None, triples, feas_phase, None, log_values=True)
            if ref is not None:
                p["ref_point"] = ref
            acq = build_acquisition("ehvi", models, triples or None, params=p, Y_observed=Y_obs)
            return _Bundle(acq, None, triples, feas_phase, None)

        if name == "parego":
            valid = data.valid
            if not valid.any():
                raise LenzError("no completed trial reports every objective yet; parego needs data")
            if weights is None:
                weights = np.full(len(shelf.objectives), 1.0 / len(shelf.objectives))
            svals = parego_scalarize(data.Y[valid], weights, rho=params.get("rho", 0.05))
            gpf, mu = _believe(self._fit(_PAREGO_TAG, data.Zc[valid], svals), Zf)
            best = None
            ok_valid = rows[valid]
            if not feas_phase:
                incumbents = np.concatenate([svals[ok_valid], mu[fant_ok]])
                if incumbents.size:
                    best = float(np.max(incumbents))
            acq = build_acquisition("parego", [gpf], triples or None, best_f=best)
            return _Bundle(acq, gpf, triples, feas_phase, best)

        raise LenzError(f"{name!r} has no pointwise acquisition value")  # sobol / random

    def _probe_acqf(self) -> str:
        """Acquisition used by predict / score by default: the shelf's if it is pointwise."""
        a = self.shelf.acqf
        return a if a in POINTWISE_ACQFS else _default_acqf(len(self.shelf.objectives))

    def _probe_values(self, name: str, params: Mapping[str, Any], data: _Data,
                      Z: np.ndarray) -> tuple[list | None, str | None]:
        """Acquisition values at ``Z`` (pending trials fantasized), or ``(None, reason)``."""
        weights = None
        if name == "parego" and params.get("weights") is not None:
            weights = np.asarray(params["weights"], dtype=float)
        try:
            bundle = self._prepare(name, params, data, data.Zp, weights)
        except LenzError as exc:
            return None, str(exc)
        vals = np.asarray(bundle.acq(Z), dtype=float).reshape(-1)
        if bundle.log_values:
            vals = np.exp(vals)
        return [float(v) for v in vals], None

    def _ref_override_notes(self, name: str, params: Mapping[str, Any] | None) -> list[str]:
        """Say so when an ehvi ``ref_point`` parameter is shadowed by objectives' own ``ref``."""
        if name != "ehvi" or not params or params.get("ref_point") is None:
            return []
        own = [o["metric"] for o in self.shelf.objectives if o.get("ref") is not None]
        if not own:
            return []
        return [f"ref_point ignored for objective(s) {own}: an objective's own 'ref' takes precedence "
                f"(change it with set_objectives)"]

    # ---- geometry -----------------------------------------------------------------------
    def _region(self, bounds: Any) -> dict | None:
        """Region for one call: ``None`` -> active region, ``"full"`` / ``{}`` -> full domain."""
        if bounds is None:
            return self.shelf.bounds
        return self.space.validate_bounds(bounds)

    def _dup_of(self, z: np.ndarray, data: _Data) -> int | None:
        for trials, Z in ((data.completed, data.Zc), (data.pending, data.Zp)):
            if Z.shape[0]:
                d = np.sqrt(np.mean((Z - z) ** 2, axis=1))
                i = int(np.argmin(d))
                if d[i] <= DUPLICATE_TOL:
                    return trials[i].trial_id
        return None

    def _violation(self, data: _Data) -> np.ndarray:
        """Total constraint violation per completed trial, each scaled by the metric's spread."""
        n = len(data.completed)
        total = np.zeros(n)
        for j, c in enumerate(self.shelf.constraints):
            v = data.Craw[:, j]
            fin = v[np.isfinite(v)]
            scale = float(np.std(fin)) if fin.size > 1 and np.std(fin) > 0 else 1.0
            viol = np.zeros(n)
            if c["lower"] is not None:
                viol = np.maximum(viol, c["lower"] - v)
            if c["upper"] is not None:
                viol = np.maximum(viol, v - c["upper"])
            total += np.where(np.isfinite(v), np.maximum(viol, 0.0) / scale, np.inf)
        return total

    def _best_index(self, data: _Data) -> int | None:
        """Index (into ``data.completed``) of the single-objective incumbent."""
        ok = data.ok
        if not ok.any():
            return None
        y = np.where(ok, data.Y[:, 0], -np.inf)
        return int(np.argmax(y))

    def _center_index(self, data: _Data) -> int | None:
        """Completed trial a trust region is centred on for ``around="incumbent"``."""
        if not data.completed:
            return None
        ok = data.ok
        if ok.any():
            if not self.shelf.is_moo:
                return self._best_index(data)
            idx = np.flatnonzero(ok)
            m = len(self.shelf.objectives)
            score = parego_scalarize(data.Y[idx], np.full(m, 1.0 / m))
            return int(idx[int(np.argmax(score))])
        viol = self._violation(data)
        viol = np.where(data.valid, viol, viol + 1e300)  # prefer trials that report every objective
        return int(np.argmin(viol))

    def _center(self, around: Any, data: _Data) -> tuple[np.ndarray, dict]:
        if isinstance(around, str):
            if around.strip().lower() != "incumbent":
                raise LenzError(f"around must be 'incumbent' or a config object, got {around!r}")
            i = self._center_index(data)
            if i is None:
                raise LenzError("around='incumbent' needs at least one completed trial")
            return data.Zc[i].copy(), dict(data.completed[i].config)
        if isinstance(around, Mapping):
            cfg = self.space.validate(around)
            return self.space.encode(cfg), cfg
        raise LenzError(f"around must be 'incumbent' or a config object, got {around!r}")

    def _reach_lattice(self, center_z: np.ndarray, lb: np.ndarray, ub: np.ndarray, rlb: np.ndarray,
                       rub: np.ndarray, region: Mapping | None) -> tuple[np.ndarray, np.ndarray]:
        """Widen a trust box on integer / ordinal dims to the nearest lattice value on each side.

        A half-width below the lattice spacing would otherwise leave only the centre's value
        on that dim (every local candidate then repeats the centre there, and in a purely
        discrete space repeats the centre trial). Neighbours are taken inside the region
        ``[rlb, rub]`` (and its allowed ordinal values), so the region is never left.
        """
        lb, ub = lb.copy(), ub.copy()
        for p in self.space.params:
            if not isinstance(p, (IntParam, OrdinalParam)):
                continue
            i = self.space.slices[p.name].start
            c = min(max(float(center_z[i]), float(rlb[i])), float(rub[i]))
            if isinstance(p, IntParam):
                vlo = math.ceil(_from_unit(p, rlb[i]) - 1e-9)
                vhi = math.floor(_from_unit(p, rub[i]) + 1e-9)
                if vlo > vhi:
                    continue
                cv = min(max(int(round(_from_unit(p, c))), vlo), vhi)
                below = _to_unit(p, cv - 1) if cv - 1 >= vlo else None
                above = _to_unit(p, cv + 1) if cv + 1 <= vhi else None
            else:
                n = len(p.values)
                allowed = region.get(p.name) if region else None
                grid = [j / (n - 1) for j, v in enumerate(p.values)
                        if rlb[i] - 1e-9 <= j / (n - 1) <= rub[i] + 1e-9 and (allowed is None or v in allowed)]
                if not grid:
                    continue
                k = int(np.argmin([abs(g - c) for g in grid]))
                below = grid[k - 1] if k > 0 else None
                above = grid[k + 1] if k + 1 < len(grid) else None
            if below is not None:
                lb[i] = max(float(rlb[i]), min(float(lb[i]), below))
            if above is not None:
                ub[i] = min(float(rub[i]), max(float(ub[i]), above))
        return lb, ub

    def _box_spec(self, lb: np.ndarray, ub: np.ndarray, region: Mapping | None) -> dict:
        """Raw-unit description of an encoded box (numeric ranges / allowed values)."""
        out: dict[str, Any] = {}
        for p in self.space.params:
            sl = self.space.slices[p.name]
            i = sl.start
            if isinstance(p, RealParam):
                out[p.name] = [_from_unit(p, lb[i]), _from_unit(p, ub[i])]
            elif isinstance(p, IntParam):
                lo = math.ceil(_from_unit(p, lb[i]) - 1e-9)
                hi = math.floor(_from_unit(p, ub[i]) + 1e-9)
                if lo > hi:
                    lo = hi = int(round(_from_unit(p, 0.5 * (lb[i] + ub[i]))))
                out[p.name] = [int(lo), int(hi)]
            elif isinstance(p, OrdinalParam):
                n = len(p.values)
                allowed = region.get(p.name) if region else None
                vals = [v for j, v in enumerate(p.values)
                        if lb[i] - 1e-9 <= j / (n - 1) <= ub[i] + 1e-9 and (allowed is None or v in allowed)]
                out[p.name] = vals
            else:
                blo, bhi = lb[sl], ub[sl]
                lo_bad = blo > 1e-9
                vals = []
                for j, ch in enumerate(p.choices):
                    others_ok = not np.any(np.delete(lo_bad, j))
                    if bhi[j] >= 1.0 - 1e-9 and others_ok:
                        vals.append(ch)
                out[p.name] = vals
        return out

    # ---- space filling --------------------------------------------------------------------
    def _space_filling(self, kind: str, q: int, lb: np.ndarray, ub: np.ndarray, region: dict | None,
                       rng: np.random.Generator, data: _Data) -> tuple[list[tuple[dict, np.ndarray]], int]:
        """Up to ``q`` new space-filling points in the box (duplicates only as a last resort).

        ``kind="sobol"`` continues the study's scrambled Sobol sequence (its position is
        persisted in ``counters["sobol"]``); ``"random"`` draws uniformly from ``rng``.
        Returns the picks and the number of them that duplicate existing trials.
        """
        space = self.space
        D = space.D
        E = np.vstack([data.Zc, data.Zp])
        picks: list[tuple[dict, np.ndarray]] = []
        spare: list[tuple[dict, np.ndarray]] = []
        start = int(self._frame.counters.get("sobol", 0))
        examined = 0
        while len(picks) < q and examined < _SF_MAX_EXAMINE:
            n = min(max(2 * (q - len(picks)), 16), _SF_MAX_EXAMINE - examined)
            if kind == "sobol":
                U = sobol_points(n, D, make_rng(self.seed, _SALT_SOBOL), skip=start + examined)
            else:
                U = rng.uniform(size=(n, D))
            P = project_box(space, lb + U * (ub - lb), lb, ub)
            for cfg in space.decode_many(P, region):
                examined += 1
                z = space.encode(cfg)
                if E.shape[0] == 0 or np.min(np.sqrt(np.mean((E - z) ** 2, axis=1))) > DUPLICATE_TOL:
                    picks.append((cfg, z))
                    E = np.vstack([E, z])
                    if len(picks) == q:
                        break
                elif len(spare) < q:
                    spare.append((cfg, z))
        if kind == "sobol":
            self._frame.counters["sobol"] = start + examined
        n_dup = 0
        while len(picks) < q and spare:
            picks.append(spare.pop(0))
            n_dup += 1
        return picks, n_dup

    # ---- predictions ---------------------------------------------------------------------
    def _predict_block(self, Z: np.ndarray, *, with_variance: bool = False) -> tuple[list[dict], list]:
        """Per-row ``{metric: {"mean", "std"(, "variance")}}`` (raw units) and prob_feasible."""
        m = Z.shape[0]
        preds: list[dict] = [dict() for _ in range(m)]
        for metric in self.shelf.modelled_metrics:
            gp = self._gp(metric)
            if gp is None:
                for row in preds:
                    row[metric] = {"mean": None, "std": None, **({"variance": None} if with_variance else {})}
                continue
            mean, var = gp.posterior(Z)
            for row, mu, v in zip(preds, mean, var):
                entry = {"mean": float(mu), "std": float(math.sqrt(max(v, 0.0)))}
                if with_variance:
                    entry["variance"] = float(v)
                row[metric] = entry
        if not self.shelf.constraints:
            return preds, [1.0] * m
        triples = []
        for c in self.shelf.constraints:
            gp = self._gp(c["metric"])
            if gp is None:
                return preds, [None] * m
            triples.append((gp, c["lower"], c["upper"]))
        return preds, [float(v) for v in prob_feasible(triples, Z)]

    def _check_configs(self, configs: Any) -> list[dict]:
        if isinstance(configs, Mapping):
            configs = [configs]
        if not isinstance(configs, (list, tuple)) or not configs:
            raise LenzError("configs must be a non-empty list of config objects")
        out = []
        for i, c in enumerate(configs):
            try:
                out.append(self.space.validate(c))
            except LenzError as exc:
                raise LenzError(f"configs[{i}]: {exc}") from None
        return out

    # ---- summaries used by several commands ----------------------------------------------
    def _trial_dict(self, t: Trial) -> dict:
        d = t.to_dict()
        if t.is_completed:
            d["feasible"] = self._is_feasible(t.metrics)
            d["value"] = metric_value(t.metrics, self.shelf.objectives[0]["metric"])
        else:
            d["feasible"] = None
            d["value"] = None
        return d

    def _incumbent_light(self, data: _Data) -> dict | None:
        i = self._best_index(data)
        if i is None:
            return None
        t = data.completed[i]
        return {"trial_id": t.trial_id, "config": dict(t.config), "metrics": dict(t.metrics or {}),
                "value": float(data.Yraw[i, 0])}

    def _front(self, data: _Data) -> tuple[list[int], float, np.ndarray | None]:
        """Pareto front (indices into ``data.completed``), hypervolume and ref point (max conv.)."""
        params = self.shelf.acqf_params if self.shelf.acqf == "ehvi" else {}
        ref = self._ref_point(data, params)
        ok = np.flatnonzero(data.ok)
        if ok.size == 0:
            return [], 0.0, ref
        Y = data.Y[ok]
        mask = pareto_mask(Y)
        hv = float(hypervolume(Y[mask], ref)) if ref is not None else 0.0
        return [int(i) for i in ok[mask]], hv, ref

    def _pareto_summary(self, data: _Data) -> dict:
        idx, hv, ref = self._front(data)
        return {"pareto_trial_ids": [data.completed[i].trial_id for i in idx], "n_pareto": len(idx),
                "hypervolume": hv, "ref_point": None if ref is None else (ref * data.signs).tolist()}

    def _incumbent_payload(self, data: _Data) -> dict | None:
        return self._pareto_summary(data) if self.shelf.is_moo else self._incumbent_light(data)

    # ================================================================== commands
    @_command
    def suggest(self, q: int = 1, around: Any = None, radius: float = DEFAULT_RADIUS, bounds: Any = None,
                acqf: str | None = None, acqf_params: Mapping[str, Any] | None = None) -> dict:
        """Propose ``q`` configurations (records nothing; see DESIGN §7.2)."""
        space, shelf = self.space, self.shelf
        m = len(shelf.objectives)
        q = 1 if q is None else _check_int(q, "q", 1, MAX_Q)
        radius = DEFAULT_RADIUS if radius is None else _check_float(radius, "radius", lo=0.0, hi=_MAX_RADIUS,
                                                                     lo_open=True)
        name = shelf.acqf if acqf is None else _check_acqf(acqf, m)
        if acqf_params is not None and not isinstance(acqf_params, Mapping):
            raise LenzError(f"acqf_params must be an object, got {acqf_params!r}")
        base = shelf.acqf_params if name == shelf.acqf else {}
        params = _check_acqf_params(name, {**base, **dict(acqf_params or {})}, m)
        region = self._region(bounds)
        lb, ub = space.encoded_bounds(region)
        data = self._data()
        n_model = int(data.valid.sum())
        space_filling = name in MODEL_FREE_ACQFS or n_model < 2
        notes: list[str] = self._ref_override_notes(name, acqf_params)

        center_cfg = None
        if around is not None:
            center_z, center_cfg = self._center(around, data)
            ls = None
            if not space_filling:
                logs = [np.log(self._require_gp(o["metric"]).hyperparameters()["lengthscales"])
                        for o in shelf.objectives]
                ls = np.exp(np.mean(logs, axis=0))
            tlb, tub = trust_region(center_z, radius, lb, ub, ls)
            lb, ub = self._reach_lattice(center_z, tlb, tub, lb, ub, region)
        rng = self._next_rng()

        cands: list[dict] = []
        feas_phase = False
        if space_filling:
            kind = name if name in MODEL_FREE_ACQFS else "sobol"
            if name not in MODEL_FREE_ACQFS:
                notes.append(f"fewer than 2 completed trials report every objective ({n_model}): "
                             f"space-filling (Sobol) design instead of {name!r}")
            picks, n_dup = self._space_filling(kind, q, lb, ub, region, rng, data)
            if n_dup:
                notes.append(f"only {q - n_dup} new distinct point(s) found in the region; "
                             f"{n_dup} candidate(s) repeat existing trials")
            Z = np.vstack([z for _, z in picks]) if picks else np.zeros((0, space.D))
            if name in MODEL_FREE_ACQFS:
                preds, pof = [None] * len(picks), [None] * len(picks)
                notes.append("model-free acquisition: predictions omitted (use predict)")
            else:
                preds, pof = self._predict_block(Z)
            for (cfg, z), pr, pf in zip(picks, preds, pof):
                cands.append({"config": cfg, "acqf": kind, "acquisition_values": {kind: None},
                              "predicted": pr, "prob_feasible": pf, "trial_id": None,
                              "duplicate_of": self._dup_of(z, data)})
            name = kind
        else:
            if any(self._gp(c["metric"]) is None for c in shelf.constraints):
                missing = [c["metric"] for c in shelf.constraints if self._gp(c["metric"]) is None]
                notes.append(f"constraint metric(s) {missing} have no observations yet and are ignored "
                             f"by the acquisition")
            picks_m: list[tuple[dict, np.ndarray, float, np.ndarray | None]] = []
            empty_front = batch_left_phase = False
            Zf = data.Zp.copy()
            E = np.vstack([data.Zc, data.Zp])
            for _ in range(q):
                w = None
                if name == "parego":
                    w = (np.asarray(params["weights"], dtype=float) if "weights" in params
                         else parego_weights(m, rng))
                bundle = self._prepare(name, params, data, Zf, w)
                feas_phase = feas_phase or bundle.feasibility_phase
                batch_left_phase = feas_phase and not bundle.feasibility_phase
                if name == "ts":
                    z, v = thompson_sample(bundle.model, space, lb, ub, rng,
                                           n_candidates=params.get("n_candidates", 1024),
                                           X_baseline=data.Zc, exclude=E, exclude_tol=DUPLICATE_TOL,
                                           constraint_models=bundle.triples or None)
                else:
                    z, v = optimize_acqf(bundle.acq, space, lb, ub, rng, X_baseline=data.Zc,
                                         exclude=E, exclude_tol=DUPLICATE_TOL)
                cfg = space.decode(z, region)
                z2 = space.encode(cfg)
                if bundle.acq is not None and np.max(np.abs(z2 - z)) > 1e-12:
                    v = float(np.asarray(bundle.acq(z2[None, :])).reshape(-1)[0])
                if bundle.log_values:
                    empty_front = True
                    v = float(np.exp(v))
                picks_m.append((cfg, z2, float(v), w))
                Zf = np.vstack([Zf, z2])
                E = np.vstack([E, z2])
            if data.pending:
                notes.append(f"{len(data.pending)} pending trial(s) fantasized at their posterior mean")
            if empty_front:
                ref = self._ref_point(data, params)
                notes.append(f"no {'feasible ' if shelf.constraints else ''}completed trial dominates the "
                             f"reference point {(ref * data.signs).tolist()} (hypervolume 0): ehvi uses its "
                             f"closed form for an empty front, prod_i EI_i(x; ref_i), optimized in log space")
            if feas_phase:
                if name in _PURE_POF_IN_FEASIBILITY_PHASE:
                    notes.append("no feasible completed trial yet: feasibility phase "
                                 "(the acquisition maximizes the probability of feasibility)")
                elif name == "ts":
                    notes.append("no feasible completed trial yet: Thompson sampling picks the candidate "
                                 "with the smallest sampled constraint violation")
                else:
                    notes.append(f"no feasible completed trial yet: {name!r} is weighted by the "
                                 f"log probability of feasibility")
                if batch_left_phase:
                    notes.append("later picks of the batch treat earlier picks believed feasible "
                                 "(kriging believer) as incumbents")
            Z = np.vstack([z for _, z, _, _ in picks_m])
            preds, pof = self._predict_block(Z)
            for (cfg, z, v, w), pr, pf in zip(picks_m, preds, pof):
                cand = {"config": cfg, "acqf": name, "acquisition_values": {name: v}, "predicted": pr,
                        "prob_feasible": pf, "trial_id": None, "duplicate_of": self._dup_of(z, data)}
                if w is not None:
                    cand["weights"] = [float(x) for x in w]
                cands.append(cand)
        dups = [c["duplicate_of"] for c in cands if c["duplicate_of"] is not None]
        if dups and not space_filling:
            notes.append(f"no distinct optimum found: candidate(s) repeat trial(s) {sorted(set(dups))}")
        return {
            "candidates": cands,
            "mode": "space_filling" if space_filling else "model",
            "acqf": name,
            "acqf_params": params,
            "region": {"bounds": region, "around": center_cfg,
                       "radius": radius if around is not None else None,
                       "box": self._box_spec(lb, ub, region) if around is not None else None},
            "n_observed": len(data.completed),
            "n_pending": len(data.pending),
            "feasible_found": bool(data.ok.any()),
            "feasibility_phase": bool(feas_phase),
            "notes": notes,
        }

    @_command
    def submit(self, config: Mapping[str, Any], source: str = "agent", meta: Mapping[str, Any] | None = None) -> dict:
        """Record a pending trial for ``config`` (validated against the domain)."""
        cfg = self.space.validate(config)
        if not isinstance(source, str) or not source.strip():
            raise LenzError(f"source must be a non-empty string, got {source!r}")
        if meta is not None and not isinstance(meta, Mapping):
            raise LenzError(f"meta must be an object, got {meta!r}")
        try:
            meta_j = jsonable(dict(meta or {}))
        except TypeError as exc:
            raise LenzError(f"meta must be JSON-serializable: {exc}") from None
        data = self._data()
        z = self.space.encode(cfg)
        dup = self._dup_of(z, data)
        t = Trial(trial_id=len(self._frame.trials), config=cfg, status="pending", metrics=None,
                  source=source.strip(), created_seq=self._seq_now, meta=meta_j)
        self._frame.trials.append(t)
        self._z_cache[t.trial_id] = z
        return {"trial_id": t.trial_id, "status": "pending", "config": cfg,
                "in_region": self.space.contains(cfg, self.shelf.bounds), "duplicate_of": dup}

    def _check_metrics(self, metrics: Any) -> dict:
        if not isinstance(metrics, Mapping) or not metrics:
            raise LenzError(f"metrics must be a non-empty object {{metric: value}}, got {metrics!r}")
        out: dict[str, float | None] = {}
        for k, v in metrics.items():
            if not isinstance(k, str) or not k.strip():
                raise LenzError(f"metric names must be non-empty strings, got {k!r}")
            if v is None:
                out[k] = None
            elif not _is_number(v):
                raise LenzError(f"metric {k!r} must be a number (or null for a failed measurement), got {v!r}")
            else:
                x = float(v)
                out[k] = x if math.isfinite(x) else None
        required = self.shelf.modelled_metrics
        missing = [r for r in required if r not in out]
        if missing:
            raise LenzError(f"metrics are missing {missing}; every objective and constraint metric "
                            f"({required}) must be reported")
        return {k: out[k] for k in sorted(out)}

    @_command
    def observe(self, metrics: Mapping[str, Any], trial_id: int | None = None,
                config: Mapping[str, Any] | None = None) -> dict:
        """Attach metrics to a pending trial (by id, or by matching ``config``), or log a new evaluation."""
        vals = self._check_metrics(metrics)
        new_trial = None
        if trial_id is not None:
            t = self._frame.trial(trial_id)
            if t.status != "pending":
                raise LenzError(f"trial {t.trial_id} is {t.status}, not pending; only pending trials can be observed "
                                f"(observe with a config to log a repeat evaluation)")
            if config is not None:
                cfg = self.space.validate(config)
                if self.space.distance(cfg, t.config) > DUPLICATE_TOL:
                    raise LenzError(f"config does not match pending trial {t.trial_id} ({t.config})")
        elif config is not None:
            cfg = self.space.validate(config)
            z = self.space.encode(cfg)
            t = None
            for p in self._frame.by_status("pending"):
                if float(np.sqrt(np.mean((self._z(p) - z) ** 2))) <= DUPLICATE_TOL:
                    t = p
                    break
            if t is None:
                new_trial = Trial(trial_id=len(self._frame.trials), config=cfg, status="pending",
                                  source="observe", created_seq=self._seq_now)
                t = new_trial
        else:
            raise LenzError("observe needs a trial_id or a config")

        if new_trial is not None:
            self._frame.trials.append(new_trial)
        self._frame.counters["eval"] = int(self._frame.counters.get("eval", 0)) + 1
        t.status = "completed"
        t.metrics = vals
        t.completed_seq = self._seq_now
        t.eval_index = self._frame.counters["eval"]

        data = self._data()
        if self.shelf.is_moo:
            idx, _, _ = self._front(data)
            is_new = any(data.completed[i].trial_id == t.trial_id for i in idx)
        else:
            inc = self._incumbent_light(data)
            is_new = inc is not None and inc["trial_id"] == t.trial_id
        return {"trial_id": t.trial_id, "status": "completed", "metrics": vals, "eval_index": t.eval_index,
                "feasible": self._is_feasible(vals), "is_new_incumbent": bool(is_new),
                "incumbent": self._incumbent_payload(data), "new_trial": new_trial is not None}

    @_command
    def abandon(self, trial_id: int) -> dict:
        """Mark a pending trial as abandoned (it stays in the log, unused by models)."""
        t = self._frame.trial(trial_id)
        if t.status != "pending":
            raise LenzError(f"trial {t.trial_id} is {t.status}; only pending trials can be abandoned")
        t.status = "abandoned"
        return {"trial_id": t.trial_id, "status": "abandoned", "config": dict(t.config)}

    @_command
    def set_bounds(self, bounds: Any) -> dict:
        """Set the active region (``None`` / ``"full"`` = full domain); data is never dropped."""
        region = self.space.validate_bounds(bounds)
        self.shelf.bounds = region
        completed = self._frame.by_status("completed")
        return {"bounds": region, "is_moo": self.shelf.is_moo, "observed": len(completed),
                "n_in_region": sum(1 for t in completed if self.space.contains(t.config, region))}

    @_command
    def set_acqf(self, acqf: str, beta: float | None = None, **params: Any) -> dict:
        """Choose the acquisition function and its parameters (see :data:`ACQF_PARAMS`)."""
        nested = params.pop("params", None)
        if nested is not None:
            if not isinstance(nested, Mapping):
                raise LenzError(f"params must be an object, got {nested!r}")
            params = {**dict(nested), **params}
        if beta is not None:
            params["beta"] = beta
        m = len(self.shelf.objectives)
        name = _check_acqf(acqf, m)
        p = _check_acqf_params(name, params, m)
        self.shelf.acqf, self.shelf.acqf_params = name, p
        return {"acqf": name, "acqf_params": p, "is_moo": self.shelf.is_moo, "observed": self.n_observed,
                "notes": self._ref_override_notes(name, p)}

    @_command
    def set_objectives(self, objectives: Any) -> dict:
        """Replace the objectives; switches between single- and multi-objective acquisitions if needed."""
        objs = normalize_objectives(objectives)
        m = len(objs)
        old, old_params = self.shelf.acqf, self.shelf.acqf_params
        new = old
        if old not in MODEL_FREE_ACQFS:
            if m == 1 and old in MULTI_OBJECTIVE_ACQFS:
                new = "logei"
            elif m == 2 and old in SINGLE_OBJECTIVE_ACQFS:
                new = "ehvi"
            elif m > 2 and (old in SINGLE_OBJECTIVE_ACQFS or old == "ehvi"):
                new = "parego"
        notes: list[str] = []
        params: dict = {}
        same_objs = ([(o["metric"], o["minimize"]) for o in objs]
                     == [(o["metric"], o["minimize"]) for o in self.shelf.objectives])
        if new != old:
            notes.append(f"acquisition switched from {old!r} to {new!r} for {m} objective(s)")
        else:
            for k, v in old_params.items():
                if k == "ref_point" and not same_objs:  # raw units of the previous objectives
                    notes.append("dropped acquisition parameter 'ref_point' (raw reference values of the "
                                 "previous objectives); give a new one with set_acqf or per-objective 'ref'")
                    continue
                try:
                    params.update(_check_acqf_params(new, {k: v}, m))
                except LenzError:
                    notes.append(f"dropped acquisition parameter {k!r} (does not fit {m} objective(s))")
        self.shelf.objectives = objs
        self.shelf.acqf, self.shelf.acqf_params = new, params
        data = self._data()
        return {"objectives": objs, "is_moo": self.shelf.is_moo, "acqf": new, "acqf_switched": new != old,
                "previous_acqf": old, "acqf_params": params, "n_with_objectives": int(data.valid.sum()),
                "observed": len(data.completed), "notes": notes}

    @_command
    def set_constraints(self, constraints: Any) -> dict:
        """Replace the outcome constraints (``None`` / ``[]`` removes them)."""
        cons = normalize_constraints(constraints)
        self.shelf.constraints = cons
        data = self._data()
        n_feas = int(data.ok.sum())
        return {"constraints": cons, "is_moo": self.shelf.is_moo, "feasible_found": n_feas > 0,
                "n_feasible": n_feas, "observed": len(data.completed)}

    @_command
    def diagnostics(self, metric: str | None = None) -> dict:
        """Surrogate quality and hyperparameters per modelled metric (DESIGN §7.2)."""
        shelf, space = self.shelf, self.space
        if metric is None:
            names = shelf.modelled_metrics
        else:
            known = self._known_metrics()
            if not isinstance(metric, str) or metric not in known:
                raise LenzError(f"unknown metric {metric!r}; known metrics: {known}")
            names = [metric]
        models = {}
        for name in names:
            roles = [r for r, ms in (("objective", shelf.objective_metrics),
                                     ("constraint", shelf.constraint_metrics)) if name in ms]
            gp = self._gp(name)
            entry: dict[str, Any] = {"role": "+".join(roles) or "reported",
                                     "n_observed": 0 if gp is None else gp.n_train}
            if gp is None:
                entry.update({"cv_r2": None, "noise_std": None, "signal_std": None, "outputscale": None,
                              "lengthscales": None, "sensitivity": None, "log_marginal_likelihood": None,
                              "note": "no observations yet"})
                models[name] = entry
                continue
            hp = gp.hyperparameters()
            ls = np.asarray(hp["lengthscales"], dtype=float)
            sens = gp.sensitivity()
            lengthscales, sensitivity = {}, {}
            for p in space.params:
                sl = space.slices[p.name]
                lengthscales[p.name] = float(np.min(ls[sl]))
                sensitivity[p.name] = float(np.sum(sens[sl]))
            entry.update({
                "cv_r2": gp.cv_r2(),
                "noise_std": math.sqrt(max(hp["noise_var"], 0.0)),
                "signal_std": math.sqrt(hp["outputscale"]) * gp.y_std,
                "outputscale": hp["outputscale"],
                "lengthscales": lengthscales,
                "sensitivity": sensitivity,
                "log_marginal_likelihood": gp.log_marginal_likelihood(),
            })
            models[name] = entry
        return {"models": models, "objective": shelf.objectives[0]["metric"]}

    @_command
    def predict(self, configs: Any) -> dict:
        """Posterior of every modelled metric (raw units), PoF and the current acquisition value."""
        cfgs = self._check_configs(configs)
        Z = self.space.encode_many(cfgs)
        data = self._data()
        preds, pof = self._predict_block(Z, with_variance=True)
        name = self._probe_acqf()
        params = self.shelf.acqf_params if name == self.shelf.acqf else {}
        vals, reason = self._probe_values(name, params, data, Z)
        out = []
        for i, cfg in enumerate(cfgs):
            out.append({"config": cfg, "metrics": preds[i], "prob_feasible": pof[i],
                        "acquisition_values": {name: None if vals is None else vals[i]},
                        "in_region": self.space.contains(cfg, self.shelf.bounds)})
        res: dict[str, Any] = {"predictions": out, "acqf": name, "n_observed": len(data.completed)}
        if reason:
            res["note"] = f"acquisition values unavailable: {reason}"
        return res

    @_command
    def score(self, configs: Any, acqfs: Any = None) -> dict:
        """Acquisition values of ``configs`` under each requested acquisition (default: current)."""
        cfgs = self._check_configs(configs)
        m = len(self.shelf.objectives)
        if acqfs is None:
            names = [self._probe_acqf()]
        else:
            raw = acqfs.split(",") if isinstance(acqfs, str) else acqfs
            if not isinstance(raw, (list, tuple)) or not raw:
                raise LenzError(f"acqfs must be a name, a comma-separated string or a list, got {acqfs!r}")
            names = []
            for a in raw:
                key = _check_acqf(a, m)
                if key not in POINTWISE_ACQFS:
                    raise LenzError(f"{key!r} has no pointwise value and cannot be scored; "
                                    f"scorable: {', '.join(POINTWISE_ACQFS)}")
                if key not in names:
                    names.append(key)
        Z = self.space.encode_many(cfgs)
        data = self._data()
        values, notes, best = {}, [], {}
        for name in names:
            params = self.shelf.acqf_params if name == self.shelf.acqf else {}
            vals, reason = self._probe_values(name, params, data, Z)
            values[name] = vals
            if reason:
                notes.append(f"{name}: {reason}")
                best[name] = None
            else:
                arr = np.where(np.isfinite(vals), vals, -np.inf)
                best[name] = int(np.argmax(arr))
        scores = [{"config": cfg, "acquisition_values": {n: (None if values[n] is None else values[n][i])
                                                         for n in names}}
                  for i, cfg in enumerate(cfgs)]
        return {"scores": scores, "acqfs": names, "best": best, "notes": notes}

    @_command
    def incumbent(self) -> dict:
        """Best completed feasible trial (single objective) or the Pareto summary (multi-objective)."""
        data = self._data()
        feasible_found = bool(data.ok.any())
        if self.shelf.is_moo:
            idx, hv, ref = self._front(data)
            return {"is_moo": True, "incumbent": None, "feasible_found": feasible_found,
                    "pareto": [self._trial_dict(data.completed[i]) for i in idx], "hypervolume": hv,
                    "ref_point": None if ref is None else (ref * data.signs).tolist(), "best_posterior": None}
        out: dict[str, Any] = {"is_moo": False, "incumbent": self._incumbent_light(data),
                               "best_posterior": None, "feasible_found": feasible_found}
        ok = np.flatnonzero(data.ok)
        gp = self._gp(self.shelf.objectives[0]["metric"])
        if ok.size and gp is not None:
            mean = gp.posterior(data.Zc[ok])[0]
            j = int(np.argmax(data.signs[0] * mean))
            t = data.completed[int(ok[j])]
            out["best_posterior"] = {"trial_id": t.trial_id, "config": dict(t.config), "mean": float(mean[j])}
        if self.shelf.constraints and not feasible_found and data.completed:
            i = self._center_index(data)
            t = data.completed[i]
            out["least_infeasible"] = {"trial_id": t.trial_id, "config": dict(t.config),
                                       "metrics": dict(t.metrics or {})}
        return out

    @_command
    def pareto(self) -> dict:
        """Non-dominated feasible completed trials, their hypervolume and the reference point (raw units)."""
        data = self._data()
        idx, hv, ref = self._front(data)
        return {"pareto": [self._trial_dict(data.completed[i]) for i in idx], "hypervolume": hv,
                "ref_point": None if ref is None else (ref * data.signs).tolist(),
                "n_feasible": int(data.ok.sum()), "is_moo": self.shelf.is_moo}

    @_command
    def trials(self, status: Any = None, last: int | None = None, sort_by: str | None = None,
               limit: int | None = None) -> dict:
        """The trial log: filtered by ``status``, cut to the ``last`` k (most recent), sorted, cut to ``limit``.

        ``last`` keeps the k most recently created trials *before* sorting; ``limit`` keeps
        the first k *after* sorting, so ``sort_by="value", limit=k`` is the top k.
        """
        trials = list(self._frame.trials)
        if status is not None and not (isinstance(status, str) and status.strip().lower() == "all"):
            wanted = [status] if isinstance(status, str) else status
            if not isinstance(wanted, (list, tuple)) or not wanted:
                raise LenzError(f"status must be one of pending, completed, abandoned (or a list), got {status!r}")
            wanted = [str(s).strip().lower() for s in wanted]
            bad = [s for s in wanted if s not in ("pending", "completed", "abandoned")]
            if bad:
                raise LenzError(f"unknown trial status {bad}; use pending, completed or abandoned")
            trials = [t for t in trials if t.status in wanted]
        if last is not None:
            k = _check_int(last, "last", 0, 10**9)
            trials = trials[len(trials) - k:] if k else []
        if sort_by is not None:
            key = str(sort_by).strip().lower()
            if key == "trial_id":
                pass
            elif key == "eval_index":
                trials.sort(key=lambda t: (t.eval_index is None, t.eval_index or 0, t.trial_id))
            elif key == "value":
                o = self.shelf.objectives[0]
                s = -1.0 if o["minimize"] else 1.0

                def rank(t: Trial) -> tuple:
                    v = t.value(o["metric"])
                    feas = t.is_completed and self._is_feasible(t.metrics)
                    return (v is None, not feas, -(s * v) if v is not None else 0.0, t.trial_id)

                trials.sort(key=rank)
            else:
                raise LenzError(f"sort_by must be one of trial_id, eval_index, value, got {sort_by!r}")
        if limit is not None:
            trials = trials[:_check_int(limit, "limit", 0, 10**9)]
        return {"trials": [self._trial_dict(t) for t in trials], "n": len(trials),
                "n_total": len(self._frame.trials)}

    @_command
    def status(self) -> dict:
        """Overview of the study: counts, configuration, space, incumbent and recent events."""
        data = self._data()
        counts = {s: 0 for s in ("pending", "completed", "abandoned")}
        for t in self._frame.trials:
            counts[t.status] += 1
        return {
            "n_observed": counts["completed"], "n_pending": counts["pending"],
            "n_abandoned": counts["abandoned"], "n_trials": len(self._frame.trials),
            "pending_trial_ids": [t.trial_id for t in data.pending],
            "is_moo": self.shelf.is_moo, "feasible_found": bool(data.ok.any()),
            "n_feasible": int(data.ok.sum()), "config": self.shelf.to_dict(),
            "space": self.space.to_spec(), "incumbent": self._incumbent_payload(data),
            "seed": self.seed, "n_events": len(self._frame.events),
            "recent_events": [e.to_dict() for e in self._frame.events[-5:]],
        }

    # ================================================================== infrastructure
    def call(self, command: str, args: Mapping[str, Any] | None = None) -> dict:
        """Run a command by name and wrap the outcome in an envelope.

        Returns ``{"ok": True, "command", "result"}`` or ``{"ok": False, "command",
        "error": {"type", "message"}}``; never raises for bad input. Hyphenated names
        (``"set-bounds"``) are accepted. Unknown or missing arguments are ``LenzError``.
        """
        seq_before = int(self._frame.counters.get("seq", 0))
        label = command.strip().lower().replace("-", "_") if isinstance(command, str) else repr(command)
        try:
            cmd = _canonical_command(command)
            if args is None:
                args = {}
            if not isinstance(args, Mapping):
                raise LenzError(f"args must be an object, got {args!r}")
            method = getattr(self, cmd)
            _check_call_args(cmd, method, args)
            result = method(**{str(k): v for k, v in args.items()})
            return {"ok": True, "command": cmd, "result": result}
        except Exception as exc:  # the envelope is the error channel for agents and the CLI
            if int(self._frame.counters.get("seq", 0)) == seq_before:
                self._record(label, args if isinstance(args, Mapping) else {"args": args}, False,
                             f"{type(exc).__name__}: {_error_message(exc)}")
            return {"ok": False, "command": label,
                    "error": {"type": type(exc).__name__, "message": _error_message(exc)}}

    def to_dict(self) -> dict:
        """Complete study state as JSON values (deterministic)."""
        return {"format": STUDY_FORMAT, "version": STUDY_VERSION, "frame": self._frame.to_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Lenz":
        if not isinstance(d, Mapping):
            raise LenzError("a study must be a JSON object")
        if "frame" in d:
            if d.get("format", STUDY_FORMAT) != STUDY_FORMAT:
                raise LenzError(f"not a lenz study (format {d.get('format')!r})")
            if int(d.get("version", STUDY_VERSION)) > STUDY_VERSION:
                raise LenzError(f"study version {d.get('version')} is newer than this lenz ({STUDY_VERSION})")
            frame_d = d["frame"]
        else:
            frame_d = d
        frame = Frame.from_dict(frame_d)
        _check_acqf(frame.shelf.acqf, len(frame.shelf.objectives))
        for k in COUNTER_KEYS:
            frame.counters.setdefault(k, 0)
        obj = cls.__new__(cls)
        obj._frame = frame
        obj._init_runtime()
        return obj

    def save(self, path: str | os.PathLike) -> None:
        """Write the study to ``path`` as one JSON document (atomic replace)."""
        path = os.fspath(path)
        text = dumps(self.to_dict())
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".lenz-", suffix=".tmp", dir=folder)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Lenz":
        """Read a study written by :meth:`save`."""
        path = os.fspath(path)
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except FileNotFoundError:
            raise LenzError(f"study file {path!r} not found (run 'create' first)") from None
        except OSError as exc:
            raise LenzError(f"cannot read study file {path!r}: {exc}") from None
        try:
            d = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LenzError(f"study file {path!r} is not valid JSON: {exc}") from None
        return cls.from_dict(d)


def _check_call_args(command: str, method: Callable, args: Mapping[str, Any]) -> None:
    sig = inspect.signature(method)
    params = sig.parameters
    var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    named = [n for n, p in params.items()
             if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]
    bad_keys = [k for k in args if not isinstance(k, str)]
    if bad_keys:
        raise LenzError(f"{command}: argument names must be strings, got {bad_keys}")
    unknown = [k for k in args if k not in named]
    if unknown and not var_kw:
        raise LenzError(f"{command}: unknown argument(s) {unknown}; allowed: {named}")
    missing = [n for n in named if params[n].default is inspect.Parameter.empty and n not in args]
    if missing:
        raise LenzError(f"{command}: missing required argument(s) {missing}")
