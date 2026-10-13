"""Small shared helpers: JSON conversion, RNG derivation, the library error type."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


class LenzError(ValueError):
    """Invalid input to the lenz backend or the agent harness.

    Raised for problems the caller (often an LLM agent) can fix, e.g. an unknown
    parameter name or an out-of-domain value. Command dispatchers turn it into an
    error envelope instead of crashing.
    """


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy / tuple / set values into JSON-safe python values.

    Non-finite floats become ``None`` so the result survives ``json.dumps`` with
    ``allow_nan=False``.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        seq = sorted(obj, key=repr) if isinstance(obj, (set, frozenset)) else obj
        return [jsonable(v) for v in seq]
    if hasattr(obj, "to_dict"):
        return jsonable(obj.to_dict())
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dumps(obj: Any, **kwargs: Any) -> str:
    """Deterministic JSON serialization (sorted keys, no NaN)."""
    kwargs.setdefault("sort_keys", True)
    kwargs.setdefault("allow_nan", False)
    return json.dumps(jsonable(obj), **kwargs)


def make_rng(seed: int | np.random.Generator | None, *salt: int) -> np.random.Generator:
    """Return a Generator; integer seeds are combined with ``salt`` deterministically."""
    if isinstance(seed, np.random.Generator):
        return seed
    base = 0 if seed is None else int(seed)
    return np.random.default_rng([abs(base), *[abs(int(s)) for s in salt]])


def derive_seed(rng: np.random.Generator) -> int:
    """Draw a fresh 31-bit integer seed from ``rng`` (for scipy QMC engines etc.)."""
    return int(rng.integers(0, 2**31 - 1))
