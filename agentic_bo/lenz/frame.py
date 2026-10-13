"""Data model of a lenz study (DESIGN.md §7.1): the trial log and the configuration shelf.

A study's state is a :class:`Frame`:

* ``trials`` — the trial log ``D_t``, the single source of truth. Trials are only ever
  appended; their ``status`` moves from ``"pending"`` to ``"completed"`` or ``"abandoned"``.
* ``shelf`` — the configuration ``c_t`` (objectives, constraints, acquisition function,
  active region, model settings). Reconfiguration replaces shelf entries and never
  touches the trials.
* ``events`` — one :class:`Event` per command call (the command-level history).
* ``seed`` / ``counters`` / ``model_state`` — what makes the study reproducible: the
  per-call random streams, the position in the space-filling sequence, and the GP
  warm-start chain all persist, so a saved and reloaded study behaves exactly like the
  original.

``to_dict`` / ``from_dict`` round-trip through plain JSON values deterministically.
This module also holds the validators for the user-facing shelf specs
(:func:`normalize_objectives`, :func:`normalize_constraints`, :func:`normalize_model`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..space import SearchSpace
from ..utils import LenzError, jsonable
from .gp import KERNELS, PRIOR_FORMS

__all__ = [
    "Trial", "Event", "Shelf", "Frame", "TRIAL_STATUSES", "DEFAULT_MODEL", "COUNTER_KEYS",
    "normalize_objectives", "normalize_constraints", "normalize_model", "metric_value",
]

TRIAL_STATUSES = ("pending", "completed", "abandoned")
DEFAULT_MODEL = {"kernel": "matern52", "noise": "learn", "n_restarts": 3, "prior": "botorch"}
#: ``seq`` last event number, ``eval`` completed evaluations, ``rng`` per-call generators
#: handed out, ``sobol`` points consumed from the study's space-filling sequence
COUNTER_KEYS = ("seq", "eval", "rng", "sobol")
_MAX_RESTARTS = 20


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def metric_value(metrics: Mapping[str, Any] | None, metric: str) -> float | None:
    """The finite float value of ``metric`` in ``metrics``, or ``None`` if missing / non-finite."""
    if not metrics:
        return None
    v = metrics.get(metric)
    if not _is_number(v):
        return None
    x = float(v)
    return x if math.isfinite(x) else None


# --------------------------------------------------------------------------- records


@dataclass
class Trial:
    """One entry of the trial log.

    ``eval_index`` is the 1-based position among completed trials; ``created_seq`` and
    ``completed_seq`` are the sequence numbers of the events that created / completed it.
    ``metrics`` maps metric names to floats (``None`` = failed / non-finite measurement).
    """

    trial_id: int
    config: dict
    status: str = "pending"
    metrics: dict | None = None
    source: str = "agent"
    created_seq: int = 0
    completed_seq: int | None = None
    eval_index: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def is_pending(self) -> bool:
        return self.status == "pending"

    def value(self, metric: str) -> float | None:
        """Finite value of ``metric`` (``None`` if not completed, missing or non-finite)."""
        return metric_value(self.metrics, metric) if self.is_completed else None

    def to_dict(self) -> dict:
        return jsonable({
            "trial_id": self.trial_id, "config": self.config, "status": self.status,
            "metrics": self.metrics, "source": self.source, "created_seq": self.created_seq,
            "completed_seq": self.completed_seq, "eval_index": self.eval_index, "meta": self.meta,
        })

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Trial":
        try:
            status = str(d.get("status", "pending"))
            if status not in TRIAL_STATUSES:
                raise LenzError(f"invalid trial status {status!r}")
            metrics = d.get("metrics")
            return cls(
                trial_id=int(d["trial_id"]), config=dict(d["config"]), status=status,
                metrics=None if metrics is None else dict(metrics),
                source=str(d.get("source", "agent")), created_seq=int(d.get("created_seq", 0)),
                completed_seq=None if d.get("completed_seq") is None else int(d["completed_seq"]),
                eval_index=None if d.get("eval_index") is None else int(d["eval_index"]),
                meta=dict(d.get("meta") or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, LenzError):
                raise
            raise LenzError(f"malformed trial record {dict(d) if isinstance(d, Mapping) else d!r}: {exc}") from None


@dataclass
class Event:
    """One command call: its arguments (JSON values, possibly truncated), outcome and a short summary."""

    seq: int
    command: str
    args: dict
    ok: bool
    summary: str

    def to_dict(self) -> dict:
        return jsonable({"seq": self.seq, "command": self.command, "args": self.args,
                         "ok": self.ok, "summary": self.summary})

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Event":
        try:
            return cls(seq=int(d["seq"]), command=str(d["command"]), args=dict(d.get("args") or {}),
                       ok=bool(d.get("ok", True)), summary=str(d.get("summary", "")))
        except (KeyError, TypeError, ValueError) as exc:
            raise LenzError(f"malformed event record: {exc}") from None


@dataclass
class Shelf:
    """The configuration ``c_t = (M, alpha, B, O, C)`` of a study."""

    objectives: list[dict]
    constraints: list[dict]
    acqf: str
    acqf_params: dict
    bounds: dict | None
    model: dict

    @property
    def is_moo(self) -> bool:
        return len(self.objectives) >= 2

    @property
    def objective_metrics(self) -> list[str]:
        return [o["metric"] for o in self.objectives]

    @property
    def constraint_metrics(self) -> list[str]:
        return [c["metric"] for c in self.constraints]

    @property
    def modelled_metrics(self) -> list[str]:
        """Objective then constraint metrics, without duplicates (order preserved)."""
        out: list[str] = []
        for m in self.objective_metrics + self.constraint_metrics:
            if m not in out:
                out.append(m)
        return out

    def to_dict(self) -> dict:
        return jsonable({"objectives": self.objectives, "constraints": self.constraints,
                         "acqf": self.acqf, "acqf_params": self.acqf_params,
                         "bounds": self.bounds, "model": self.model})

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], space: SearchSpace | None = None) -> "Shelf":
        try:
            bounds = d.get("bounds")
            if space is not None:
                bounds = space.validate_bounds(bounds)
            return cls(objectives=normalize_objectives(d["objectives"]),
                       constraints=normalize_constraints(d.get("constraints")),
                       acqf=str(d["acqf"]), acqf_params=dict(d.get("acqf_params") or {}),
                       bounds=bounds, model=normalize_model(d.get("model")))
        except KeyError as exc:
            raise LenzError(f"malformed shelf: missing {exc}") from None


@dataclass
class Frame:
    """Complete, serializable state of a study."""

    space: SearchSpace
    shelf: Shelf
    trials: list[Trial] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    seed: int = 0
    counters: dict = field(default_factory=lambda: {k: 0 for k in COUNTER_KEYS})
    #: per-model GP warm-start chain: ``{tag: {"key", "n", "warm_start", "hyperparameters"}}``
    model_state: dict = field(default_factory=dict)

    def trial(self, trial_id: Any) -> Trial:
        """Look up a trial by id (``LenzError`` if unknown)."""
        if isinstance(trial_id, bool) or not _is_number(trial_id) or float(trial_id) != int(trial_id):
            raise LenzError(f"trial_id must be an integer, got {trial_id!r}")
        i = int(trial_id)
        if 0 <= i < len(self.trials) and self.trials[i].trial_id == i:
            return self.trials[i]
        for t in self.trials:  # ids are positional by construction; this only guards odd files
            if t.trial_id == i:
                return t
        raise LenzError(f"unknown trial_id {i}; known ids are 0..{len(self.trials) - 1}"
                        if self.trials else f"unknown trial_id {i}: no trials yet")

    def by_status(self, status: str) -> list[Trial]:
        return [t for t in self.trials if t.status == status]

    def to_dict(self) -> dict:
        return {
            "space": jsonable(self.space.to_spec()),
            "shelf": self.shelf.to_dict(),
            "trials": [t.to_dict() for t in self.trials],
            "events": [e.to_dict() for e in self.events],
            "seed": int(self.seed),
            "counters": jsonable(self.counters),
            "model_state": jsonable(self.model_state),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Frame":
        if not isinstance(d, Mapping):
            raise LenzError("a study frame must be a JSON object")
        try:
            space = SearchSpace.from_spec(d["space"])
            shelf = Shelf.from_dict(d["shelf"], space)
        except KeyError as exc:
            raise LenzError(f"malformed study: missing {exc}") from None
        trials = [Trial.from_dict(t) for t in d.get("trials") or []]
        for i, t in enumerate(trials):
            if t.trial_id != i:
                raise LenzError(f"malformed study: trial ids must be 0..n-1 in order (got {t.trial_id} at {i})")
            try:
                t.config = space.validate(t.config)
            except LenzError as exc:
                raise LenzError(f"malformed study: trial {i}: {exc}") from None
        counters = {k: 0 for k in COUNTER_KEYS}
        counters.update({str(k): int(v) for k, v in dict(d.get("counters") or {}).items()})
        return cls(space=space, shelf=shelf, trials=trials,
                   events=[Event.from_dict(e) for e in d.get("events") or []],
                   seed=int(d.get("seed", 0)), counters=counters,
                   model_state=dict(d.get("model_state") or {}))


# --------------------------------------------------------------------------- spec validation


def _check_metric_name(name: Any, what: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise LenzError(f"{what}: metric names must be non-empty strings, got {name!r}")
    return name.strip()


def _optional_number(v: Any, what: str) -> float | None:
    if v is None:
        return None
    if not _is_number(v) or not math.isfinite(float(v)):
        raise LenzError(f"{what} must be a finite number or null, got {v!r}")
    return float(v)


_DIRECTIONS = {"minimize": True, "min": True, "maximize": False, "max": False}


def _objective_item(spec: Any) -> dict:
    if isinstance(spec, str):
        return {"metric": _check_metric_name(spec, "objective"), "minimize": False, "ref": None}
    if not isinstance(spec, Mapping):
        raise LenzError(f"an objective must be a metric name or an object "
                        f"{{\"metric\", \"minimize\"}}, got {spec!r}")
    unknown = sorted(set(spec) - {"metric", "name", "minimize", "direction", "ref", "ref_point"})
    if unknown:
        raise LenzError(f"objective {spec!r}: unknown keys {unknown} (allowed: metric, minimize, ref)")
    metric = _check_metric_name(spec.get("metric", spec.get("name")), "objective")
    minimize = spec.get("minimize")
    if "direction" in spec:
        d = spec["direction"]
        if not isinstance(d, str) or d.strip().lower() not in _DIRECTIONS:
            raise LenzError(f"objective {metric!r}: direction must be 'minimize' or 'maximize', got {d!r}")
        by_dir = _DIRECTIONS[d.strip().lower()]
        if minimize is not None and bool(minimize) != by_dir:
            raise LenzError(f"objective {metric!r}: 'minimize' and 'direction' disagree")
        minimize = by_dir
    if minimize is None:
        minimize = False
    if not isinstance(minimize, (bool, np.bool_)):
        raise LenzError(f"objective {metric!r}: 'minimize' must be true or false, got {minimize!r}")
    ref = _optional_number(spec.get("ref", spec.get("ref_point")), f"objective {metric!r}: ref")
    return {"metric": metric, "minimize": bool(minimize), "ref": ref}


def normalize_objectives(spec: Any) -> list[dict]:
    """Normalize objectives to ``[{"metric", "minimize", "ref"}]``.

    Accepts a metric name (maximized), one ``{"metric", "minimize"?, "ref"?}`` object
    (``"direction": "minimize" | "maximize"`` is accepted too) or a non-empty list of
    those. ``ref`` is an optional hypervolume reference value in raw metric units.
    """
    if spec is None:
        raise LenzError("at least one objective is required")
    items = list(spec) if isinstance(spec, (list, tuple)) else [spec]
    if not items:
        raise LenzError("at least one objective is required")
    out = [_objective_item(s) for s in items]
    names = [o["metric"] for o in out]
    if len(set(names)) != len(names):
        raise LenzError(f"duplicate objective metrics in {names}")
    return out


def _constraint_item(spec: Any, metric: str | None = None) -> dict:
    if isinstance(spec, (list, tuple)) and metric is not None:
        if len(spec) != 2:
            raise LenzError(f"constraint {metric!r}: expected [lower, upper], got {spec!r}")
        spec = {"metric": metric, "lower": spec[0], "upper": spec[1]}
    if not isinstance(spec, Mapping):
        raise LenzError(f"a constraint must be an object {{\"metric\", \"lower\", \"upper\"}}, got {spec!r}")
    if metric is not None and "metric" not in spec:
        spec = {**spec, "metric": metric}
    unknown = sorted(set(spec) - {"metric", "name", "lower", "upper"})
    if unknown:
        raise LenzError(f"constraint {spec!r}: unknown keys {unknown} (allowed: metric, lower, upper)")
    name = _check_metric_name(spec.get("metric", spec.get("name")), "constraint")
    lower = _optional_number(spec.get("lower"), f"constraint {name!r}: lower")
    upper = _optional_number(spec.get("upper"), f"constraint {name!r}: upper")
    if lower is None and upper is None:
        raise LenzError(f"constraint {name!r}: give a lower and/or an upper bound")
    if lower is not None and upper is not None and not lower < upper:
        raise LenzError(f"constraint {name!r}: need lower < upper, got [{lower}, {upper}]")
    return {"metric": name, "lower": lower, "upper": upper}


def normalize_constraints(spec: Any) -> list[dict]:
    """Normalize outcome constraints to ``[{"metric", "lower", "upper"}]``.

    Accepts ``None`` / ``[]`` (no constraints), one ``{"metric", "lower"?, "upper"?}``
    object, a list of those, or a mapping ``{metric: [lower, upper]}`` (``null`` for an
    open side). Feasible means ``lower <= value <= upper``.
    """
    if spec is None:
        return []
    if isinstance(spec, Mapping):
        if "metric" in spec or "name" in spec:
            items = [_constraint_item(spec)]
        else:
            items = [_constraint_item(v, _check_metric_name(k, "constraint")) for k, v in spec.items()]
    elif isinstance(spec, (list, tuple)):
        items = [_constraint_item(s) for s in spec]
    else:
        raise LenzError(f"constraints must be a list of {{\"metric\", \"lower\", \"upper\"}} objects, got {spec!r}")
    names = [c["metric"] for c in items]
    if len(set(names)) != len(names):
        raise LenzError(f"duplicate constraint metrics in {names}: use one constraint with both bounds")
    return items


def normalize_model(spec: Any) -> dict:
    """Normalize GP settings ``{"kernel", "noise", "n_restarts", "prior"}`` (missing keys get defaults)."""
    if spec is None:
        return dict(DEFAULT_MODEL)
    if not isinstance(spec, Mapping):
        raise LenzError(f"model must be an object like {DEFAULT_MODEL}, got {spec!r}")
    unknown = sorted(set(spec) - set(DEFAULT_MODEL))
    if unknown:
        raise LenzError(f"model: unknown keys {unknown} (allowed: {sorted(DEFAULT_MODEL)})")
    out = dict(DEFAULT_MODEL)
    kernel = spec.get("kernel", out["kernel"])
    if not isinstance(kernel, str) or kernel.strip().lower() not in KERNELS:
        raise LenzError(f"model kernel must be one of {list(KERNELS)}, got {kernel!r}")
    out["kernel"] = kernel.strip().lower()
    noise = spec.get("noise", out["noise"])
    if isinstance(noise, str):
        if noise.strip().lower() != "learn":
            raise LenzError(f"model noise must be 'learn' or a fixed variance >= 0, got {noise!r}")
        out["noise"] = "learn"
    else:
        if not _is_number(noise) or not math.isfinite(float(noise)) or float(noise) < 0:
            raise LenzError(f"model noise must be 'learn' or a fixed variance >= 0, got {noise!r}")
        out["noise"] = float(noise)
    nr = spec.get("n_restarts", out["n_restarts"])
    if isinstance(nr, bool) or not _is_number(nr) or float(nr) != int(nr) or not 1 <= int(nr) <= _MAX_RESTARTS:
        raise LenzError(f"model n_restarts must be an integer in [1, {_MAX_RESTARTS}], got {nr!r}")
    out["n_restarts"] = int(nr)
    prior = spec.get("prior", out["prior"])
    if not isinstance(prior, str) or prior.strip().lower() not in PRIOR_FORMS:
        raise LenzError(f"model prior must be one of {list(PRIOR_FORMS)}, got {prior!r}")
    out["prior"] = prior.strip().lower()
    return out
