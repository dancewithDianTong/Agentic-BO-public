"""Search spaces: bounded reals (optionally log-scaled), integers, ordinals, categoricals.

Every space has an *encoded* representation in ``[0, 1]^D`` that the GP and the
acquisition optimizer work in: one dimension per real/int/ordinal parameter and a
one-hot block per categorical parameter. ``decode`` is the inverse up to rounding, and
``project`` snaps arbitrary encoded points onto representable ones.

An *active region* (the paper's ``B_t``) is a dict ``{name: [lo, hi]}`` for numeric
parameters (raw units) and ``{name: [allowed, values]}`` for ordinal / categorical
parameters; parameters that are not mentioned keep their full domain.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Union

import numpy as np
from scipy.stats import qmc

from .utils import LenzError, derive_seed, make_rng

_TOL = 1e-9


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _same_value(a: Any, b: Any) -> bool:
    a_bool, b_bool = isinstance(a, (bool, np.bool_)), isinstance(b, (bool, np.bool_))
    if a_bool or b_bool:
        return a_bool and b_bool and bool(a) == bool(b)
    if _is_number(a) and _is_number(b):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


# --------------------------------------------------------------------------- params


@dataclass(frozen=True)
class RealParam:
    name: str
    lower: float
    upper: float
    log_scale: bool = False

    def __post_init__(self) -> None:
        _check_name(self.name)
        if not (_is_number(self.lower) and _is_number(self.upper)):
            raise LenzError(f"{self.name}: lower/upper must be numbers")
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)) or not self.lower < self.upper:
            raise LenzError(f"{self.name}: need finite lower < upper, got [{self.lower}, {self.upper}]")
        if self.log_scale and self.lower <= 0:
            raise LenzError(f"{self.name}: log scale requires lower > 0")
        object.__setattr__(self, "lower", float(self.lower))
        object.__setattr__(self, "upper", float(self.upper))

    kind = "real"


@dataclass(frozen=True)
class IntParam:
    name: str
    lower: int
    upper: int
    log_scale: bool = False

    def __post_init__(self) -> None:
        _check_name(self.name)
        for v in (self.lower, self.upper):
            if not _is_number(v) or float(v) != round(float(v)):
                raise LenzError(f"{self.name}: integer bounds required, got {v!r}")
        lo, hi = int(round(float(self.lower))), int(round(float(self.upper)))
        if not lo < hi:
            raise LenzError(f"{self.name}: need lower < upper, got [{lo}, {hi}]")
        if self.log_scale and lo <= 0:
            raise LenzError(f"{self.name}: log scale requires lower >= 1")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)

    kind = "int"


@dataclass(frozen=True)
class OrdinalParam:
    name: str
    values: tuple

    def __post_init__(self) -> None:
        _check_name(self.name)
        vals = tuple(self.values)
        if len(vals) < 2:
            raise LenzError(f"{self.name}: an ordinal needs at least 2 values")
        for i, a in enumerate(vals):
            for b in vals[i + 1:]:
                if _same_value(a, b):
                    raise LenzError(f"{self.name}: duplicate ordinal value {a!r}")
        object.__setattr__(self, "values", vals)

    kind = "ordinal"


@dataclass(frozen=True)
class CategoricalParam:
    name: str
    choices: tuple

    def __post_init__(self) -> None:
        _check_name(self.name)
        ch = tuple(self.choices)
        if len(ch) < 1:
            raise LenzError(f"{self.name}: a categorical needs at least 1 choice")
        if len(set(map(repr, ch))) != len(ch):
            raise LenzError(f"{self.name}: duplicate categorical choices")
        object.__setattr__(self, "choices", ch)

    kind = "categorical"


Param = Union[RealParam, IntParam, OrdinalParam, CategoricalParam]


def _check_name(name: Any) -> None:
    if not isinstance(name, str) or not name:
        raise LenzError(f"parameter names must be non-empty strings, got {name!r}")


_TYPE_ALIASES = {
    "real": "real", "float": "real", "continuous": "real", "range": "real",
    "int": "int", "integer": "int",
    "ordinal": "ordinal", "ordered": "ordinal",
    "categorical": "categorical", "choice": "categorical", "cat": "categorical",
}


def param_from_spec(spec: Mapping[str, Any], name: str | None = None) -> Param:
    """Build a parameter from a dict such as ``{"name": "lr", "type": "real", "lower": 1e-4, "upper": 1, "log": True}``."""
    if not isinstance(spec, Mapping):
        raise LenzError(f"parameter spec must be an object, got {spec!r}")
    name = spec.get("name", name)
    kind = _TYPE_ALIASES.get(str(spec.get("type", "real")).lower())
    if kind is None:
        raise LenzError(f"{name}: unknown parameter type {spec.get('type')!r}")
    log = bool(spec.get("log_scale", spec.get("log", False)))
    if kind in ("real", "int"):
        lo = spec.get("lower", spec.get("low"))
        hi = spec.get("upper", spec.get("high"))
        if "bounds" in spec and (lo is None or hi is None):
            lo, hi = spec["bounds"]
        if lo is None or hi is None:
            raise LenzError(f"{name}: numeric parameters need lower and upper")
        return RealParam(name, lo, hi, log) if kind == "real" else IntParam(name, lo, hi, log)
    if kind == "ordinal":
        if "values" not in spec:
            raise LenzError(f"{name}: ordinal parameters need 'values'")
        return OrdinalParam(name, tuple(spec["values"]))
    if "choices" not in spec and "values" not in spec:
        raise LenzError(f"{name}: categorical parameters need 'choices'")
    return CategoricalParam(name, tuple(spec.get("choices", spec.get("values"))))


def param_to_spec(p: Param) -> dict:
    if isinstance(p, RealParam):
        return {"name": p.name, "type": "real", "lower": p.lower, "upper": p.upper, "log_scale": p.log_scale}
    if isinstance(p, IntParam):
        return {"name": p.name, "type": "int", "lower": p.lower, "upper": p.upper, "log_scale": p.log_scale}
    if isinstance(p, OrdinalParam):
        return {"name": p.name, "type": "ordinal", "values": list(p.values)}
    return {"name": p.name, "type": "categorical", "choices": list(p.choices)}


# --------------------------------------------------------------------------- space


def sobol_points(n: int, dim: int, rng: np.random.Generator | int | None, skip: int = 0) -> np.ndarray:
    """``n`` scrambled Sobol points in ``[0,1]^dim`` (after skipping ``skip`` points)."""
    if n <= 0:
        return np.zeros((0, dim))
    engine = qmc.Sobol(d=max(dim, 1), scramble=True, seed=derive_seed(make_rng(rng)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if skip:
            engine.fast_forward(int(skip))
        pts = engine.random(int(n))
    return pts[:, :dim] if dim > 0 else np.zeros((n, 0))


class SearchSpace:
    """An ordered collection of parameters with an encoding into ``[0, 1]^D``."""

    def __init__(self, params: Sequence[Param]):
        params = list(params)
        if not params:
            raise LenzError("a search space needs at least one parameter")
        names = [p.name for p in params]
        if len(set(names)) != len(names):
            raise LenzError(f"duplicate parameter names in {names}")
        self.params: list[Param] = params
        self.names: list[str] = names
        self._by_name = {p.name: p for p in params}
        self.slices: dict[str, slice] = {}
        i = 0
        for p in params:
            width = len(p.choices) if isinstance(p, CategoricalParam) else 1
            self.slices[p.name] = slice(i, i + width)
            i += width
        self.D = i
        self.d = len(params)

    # ---- construction / serialization -------------------------------------------------
    @classmethod
    def from_spec(cls, spec: Any) -> "SearchSpace":
        if isinstance(spec, SearchSpace):
            return spec
        if isinstance(spec, Mapping):
            if "params" in spec and isinstance(spec["params"], (list, tuple)):
                return cls.from_spec(spec["params"])
            return cls([param_from_spec(v, name=k) for k, v in spec.items()])
        if isinstance(spec, (list, tuple)):
            return cls([p if isinstance(p, (RealParam, IntParam, OrdinalParam, CategoricalParam))
                        else param_from_spec(p) for p in spec])
        raise LenzError(f"cannot build a search space from {type(spec).__name__}")

    def to_spec(self) -> list[dict]:
        return [param_to_spec(p) for p in self.params]

    def __getitem__(self, name: str) -> Param:
        try:
            return self._by_name[name]
        except KeyError:
            raise LenzError(f"unknown parameter {name!r}; known: {self.names}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return self.d

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SearchSpace) and self.to_spec() == other.to_spec()

    def __repr__(self) -> str:
        return f"SearchSpace({self.to_spec()!r})"

    @property
    def is_continuous(self) -> bool:
        return all(isinstance(p, RealParam) for p in self.params)

    # ---- per-parameter numeric transforms -----------------------------------------------
    @staticmethod
    def _num_to_unit(p: RealParam | IntParam, v: float) -> float:
        if p.log_scale:
            return (math.log(v) - math.log(p.lower)) / (math.log(p.upper) - math.log(p.lower))
        return (v - p.lower) / (p.upper - p.lower)

    @staticmethod
    def _unit_to_num(p: RealParam | IntParam, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 0.0, 1.0)
        if p.log_scale:
            return np.exp(math.log(p.lower) + u * (math.log(p.upper) - math.log(p.lower)))
        return p.lower + u * (p.upper - p.lower)

    # ---- validation ----------------------------------------------------------------------
    def validate_value(self, name: str, v: Any) -> Any:
        p = self[name]
        if isinstance(p, (RealParam, IntParam)):
            if not _is_number(v) or not math.isfinite(float(v)):
                raise LenzError(f"{name}: expected a finite number, got {v!r}")
            x = float(v)
            span = p.upper - p.lower
            if x < p.lower - _TOL * span - 1e-12 or x > p.upper + _TOL * span + 1e-12:
                raise LenzError(f"{name}={x!r} is outside the domain [{p.lower}, {p.upper}]")
            x = min(max(x, p.lower), p.upper)
            if isinstance(p, IntParam):
                if abs(x - round(x)) > 1e-6:
                    raise LenzError(f"{name}: expected an integer, got {v!r}")
                return int(round(x))
            return x
        options = p.values if isinstance(p, OrdinalParam) else p.choices
        for o in options:
            if _same_value(o, v):
                return o
        raise LenzError(f"{name}={v!r} is not one of {list(options)}")

    def validate(self, config: Mapping[str, Any]) -> dict:
        """Return a normalized copy of ``config`` (param order, canonical types)."""
        if not isinstance(config, Mapping):
            raise LenzError(f"a config must be an object mapping parameter names to values, got {config!r}")
        missing = [n for n in self.names if n not in config]
        extra = [k for k in config if k not in self._by_name]
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing parameters {missing}")
            if extra:
                parts.append(f"unknown parameters {extra}")
            raise LenzError("invalid config: " + "; ".join(parts) + f" (expected exactly {self.names})")
        return {n: self.validate_value(n, config[n]) for n in self.names}

    def contains(self, config: Mapping[str, Any], bounds: Mapping | None = None) -> bool:
        try:
            cfg = self.validate(config)
        except LenzError:
            return False
        if not bounds:
            return True
        b = self.validate_bounds(bounds) or {}
        for name, allowed in b.items():
            p = self[name]
            v = cfg[name]
            if isinstance(p, (RealParam, IntParam)):
                lo, hi = allowed
                span = p.upper - p.lower
                if v < lo - _TOL * span or v > hi + _TOL * span:
                    return False
            elif not any(_same_value(v, a) for a in allowed):
                return False
        return True

    def validate_bounds(self, bounds: Any) -> dict | None:
        """Normalize an active-region spec. ``None`` / ``"full"`` / ``{}`` mean the full domain."""
        if bounds is None or (isinstance(bounds, str) and bounds.lower() == "full"):
            return None
        if not isinstance(bounds, Mapping):
            raise LenzError(f"bounds must be an object {{param: [lo, hi] | [allowed...]}}, got {bounds!r}")
        if not bounds:
            return None
        out: dict[str, list] = {}
        for name, spec in bounds.items():
            p = self[name]
            if isinstance(p, (RealParam, IntParam)):
                if isinstance(spec, Mapping):
                    spec = [spec.get("lower", spec.get("low")), spec.get("upper", spec.get("high"))]
                if not isinstance(spec, (list, tuple)) or len(spec) != 2:
                    raise LenzError(f"bounds for {name} must be [lo, hi], got {spec!r}")
                lo, hi = spec
                if not (_is_number(lo) and _is_number(hi)):
                    raise LenzError(f"bounds for {name} must be numbers, got {spec!r}")
                lo, hi = float(lo), float(hi)
                span = p.upper - p.lower
                if lo > hi:
                    raise LenzError(f"bounds for {name}: lower {lo} > upper {hi}")
                if lo < p.lower - _TOL * span or hi > p.upper + _TOL * span:
                    raise LenzError(f"bounds for {name} [{lo}, {hi}] exceed the domain [{p.lower}, {p.upper}]")
                lo, hi = max(lo, p.lower), min(hi, p.upper)
                if isinstance(p, IntParam):
                    lo, hi = math.ceil(lo - 1e-9), math.floor(hi + 1e-9)
                    if lo > hi:
                        raise LenzError(f"bounds for {name} contain no integer")
                    out[name] = [int(lo), int(hi)]
                else:
                    out[name] = [lo, hi]
            else:
                if not isinstance(spec, (list, tuple)) or len(spec) == 0:
                    raise LenzError(f"bounds for {name} must be a non-empty list of allowed values")
                allowed = [self.validate_value(name, v) for v in spec]
                options = p.values if isinstance(p, OrdinalParam) else p.choices
                out[name] = [o for o in options if any(_same_value(o, a) for a in allowed)]
        return out

    # ---- encoding ------------------------------------------------------------------------
    def encode(self, config: Mapping[str, Any]) -> np.ndarray:
        cfg = self.validate(config)
        z = np.zeros(self.D)
        for p in self.params:
            sl = self.slices[p.name]
            v = cfg[p.name]
            if isinstance(p, (RealParam, IntParam)):
                z[sl.start] = self._num_to_unit(p, float(v))
            elif isinstance(p, OrdinalParam):
                idx = next(i for i, o in enumerate(p.values) if _same_value(o, v))
                z[sl.start] = idx / (len(p.values) - 1)
            else:
                idx = next(i for i, o in enumerate(p.choices) if _same_value(o, v))
                z[sl.start + idx] = 1.0
        return np.clip(z, 0.0, 1.0)

    def encode_many(self, configs: Iterable[Mapping[str, Any]]) -> np.ndarray:
        rows = [self.encode(c) for c in configs]
        return np.array(rows).reshape(len(rows), self.D)

    def _allowed_indices(self, p: OrdinalParam | CategoricalParam, bounds: Mapping | None) -> list[int]:
        options = p.values if isinstance(p, OrdinalParam) else p.choices
        if bounds and p.name in bounds:
            allowed = bounds[p.name]
            return [i for i, o in enumerate(options) if any(_same_value(o, a) for a in allowed)]
        return list(range(len(options)))

    def decode_many(self, Z: np.ndarray, bounds: Mapping | None = None) -> list[dict]:
        """Decode encoded rows (clipping, rounding, nearest allowed ordinal, argmax categorical)."""
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        if Z.shape[1] != self.D:
            raise LenzError(f"expected encoded points with {self.D} columns, got {Z.shape[1]}")
        b = self.validate_bounds(bounds) if bounds else None
        out: list[dict] = [dict() for _ in range(Z.shape[0])]
        for p in self.params:
            sl = self.slices[p.name]
            if isinstance(p, (RealParam, IntParam)):
                vals = self._unit_to_num(p, Z[:, sl.start])
                if b and p.name in b:
                    vals = np.clip(vals, b[p.name][0], b[p.name][1])
                for row, v in zip(out, vals):
                    if isinstance(p, IntParam):
                        iv = int(round(float(v)))
                        lo, hi = (b[p.name] if b and p.name in b else (p.lower, p.upper))
                        row[p.name] = int(min(max(iv, lo), hi))
                    else:
                        row[p.name] = float(min(max(float(v), p.lower), p.upper))
            elif isinstance(p, OrdinalParam):
                allowed = np.array(self._allowed_indices(p, b))
                pos = np.clip(Z[:, sl.start], 0, 1) * (len(p.values) - 1)
                idx = allowed[np.argmin(np.abs(pos[:, None] - allowed[None, :]), axis=1)]
                for row, i in zip(out, idx):
                    row[p.name] = p.values[int(i)]
            else:
                allowed = self._allowed_indices(p, b)
                block = Z[:, sl][:, allowed]
                idx = np.asarray(allowed)[np.argmax(block, axis=1)]
                for row, i in zip(out, idx):
                    row[p.name] = p.choices[int(i)]
        return out

    def decode(self, z: np.ndarray, bounds: Mapping | None = None) -> dict:
        return self.decode_many(np.asarray(z, dtype=float).reshape(1, -1), bounds)[0]

    def project(self, Z: np.ndarray, bounds: Mapping | None = None) -> np.ndarray:
        """Snap encoded points onto representable configurations (``encode(decode(Z))``)."""
        Z = np.asarray(Z, dtype=float)
        single = Z.ndim == 1
        if self.is_continuous:
            # nothing to round: just clip into the (active) box, keeping full precision
            lb, ub = self.encoded_bounds(bounds)
            out = np.clip(np.atleast_2d(Z), lb, ub)
        else:
            out = self.encode_many(self.decode_many(np.atleast_2d(Z), bounds))
        return out[0] if single else out

    def encoded_bounds(self, bounds: Mapping | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Box ``(lb, ub)`` in encoded coordinates for an active region."""
        b = self.validate_bounds(bounds) if bounds else None
        lb, ub = np.zeros(self.D), np.ones(self.D)
        if not b:
            return lb, ub
        for name, allowed in b.items():
            p = self[name]
            sl = self.slices[name]
            if isinstance(p, (RealParam, IntParam)):
                lb[sl.start] = self._num_to_unit(p, float(allowed[0]))
                ub[sl.start] = self._num_to_unit(p, float(allowed[1]))
            elif isinstance(p, OrdinalParam):
                idx = self._allowed_indices(p, b)
                lb[sl.start] = min(idx) / (len(p.values) - 1)
                ub[sl.start] = max(idx) / (len(p.values) - 1)
            else:
                idx = set(self._allowed_indices(p, b))
                for k in range(len(p.choices)):
                    if k not in idx:
                        ub[sl.start + k] = 0.0
        return np.clip(lb, 0, 1), np.clip(np.maximum(ub, lb), 0, 1)

    # ---- sampling / geometry -------------------------------------------------------------
    def sobol(self, n: int, rng: np.random.Generator | int | None, bounds: Mapping | None = None,
              skip: int = 0) -> list[dict]:
        """Space-filling configs: scrambled Sobol points mapped into the active region."""
        lb, ub = self.encoded_bounds(bounds)
        U = sobol_points(n, self.D, rng, skip=skip)
        return self.decode_many(lb + U * (ub - lb), bounds)

    def sample_uniform(self, n: int, rng: np.random.Generator | int | None,
                       bounds: Mapping | None = None) -> list[dict]:
        lb, ub = self.encoded_bounds(bounds)
        U = make_rng(rng).uniform(size=(n, self.D))
        return self.decode_many(lb + U * (ub - lb), bounds)

    def distance(self, a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
        """RMS distance between two configs in encoded space (0 = identical, at most 1)."""
        za, zb = self.encode(a), self.encode(b)
        return float(np.sqrt(np.mean((za - zb) ** 2)))

    def center(self, bounds: Mapping | None = None) -> dict:
        lb, ub = self.encoded_bounds(bounds)
        z = (lb + ub) / 2
        for p in self.params:
            if isinstance(p, CategoricalParam):  # pick the first allowed choice
                sl = self.slices[p.name]
                block = ub[sl].copy()
                z[sl] = 0.0
                z[sl.start + int(np.argmax(block))] = 1.0
        return self.decode(z, bounds)

    def describe(self, bounds: Mapping | None = None) -> str:
        """Markdown table of the parameters (used in agent briefs)."""
        lines = ["| parameter | type | domain | scale |", "|---|---|---|---|"]
        for p in self.params:
            if isinstance(p, (RealParam, IntParam)):
                dom = f"[{p.lower:g}, {p.upper:g}]"
                scale = "log" if p.log_scale else "linear"
            elif isinstance(p, OrdinalParam):
                dom, scale = "{" + ", ".join(map(str, p.values)) + "} (ordered)", "-"
            else:
                dom, scale = "{" + ", ".join(map(str, p.choices)) + "}", "-"
            lines.append(f"| `{p.name}` | {p.kind} | {dom} | {scale} |")
        return "\n".join(lines)
