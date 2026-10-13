"""Benchmark problems: the :class:`Problem` container and the randomization wrapper.

A :class:`Problem` bundles a search space, a noise-free evaluation function returning
a dict of raw metrics, the primary objective (name + raw direction), its known optimum
(for regret), optional constraints / extra objectives, and the natural-language brief
shown to agents. Regret is always computed from noise-free metrics in the
*maximization* convention: ``utility = -value`` for a minimized objective and
``regret = u_star - utility >= 0``.

:func:`randomize_problem` implements the paper's anti-memorization protocol (App. A):
the domain is canonicalized to ``[0, 1]^d``, parameters get deterministic random names
(and a random order), and -- only for functions whose known optimum is global over
``R^d`` ("shift-safe") -- the optimum is shifted by up to a quarter of each range while
staying inside ``[0.05, 0.95]``. The optimal value is unchanged.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field, replace
from typing import Any, Callable, ClassVar, Mapping, Sequence

import numpy as np

from ..space import RealParam, SearchSpace
from ..utils import LenzError, jsonable, make_rng

__all__ = ["Problem", "VectorizedEvaluator", "CanonicalMap", "randomize_problem"]

MetricsFn = Callable[[dict], Mapping[str, Any]]
BatchFn = Callable[[np.ndarray], Mapping[str, np.ndarray]]

SHIFT_FRACTION = 0.25          # max optimum shift, as a fraction of each canonical range
SHIFT_MARGIN = 0.05            # the shifted optimum stays inside [margin, 1 - margin]


# --------------------------------------------------------------------------- helpers


def _as_float(v: Any) -> float:
    """``float(v)``, mapping ``None`` / non-numeric values to NaN."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _normalize_constraints(constraints: Sequence[Mapping[str, Any]] | None) -> list[dict]:
    out: list[dict] = []
    for c in constraints or []:
        if not isinstance(c, Mapping) or not isinstance(c.get("metric"), str):
            raise LenzError("a constraint must be {'metric': str, 'lower': float|None, 'upper': float|None}, "
                            f"got {c!r}")
        lo = None if c.get("lower") is None else float(c["lower"])
        hi = None if c.get("upper") is None else float(c["upper"])
        if lo is None and hi is None:
            raise LenzError(f"constraint on {c['metric']!r} needs a lower and/or an upper bound")
        if lo is not None and hi is not None and lo > hi:
            raise LenzError(f"constraint on {c['metric']!r}: lower {lo} > upper {hi}")
        out.append({"metric": c["metric"], "lower": lo, "upper": hi})
    return out


def _normalize_objectives(objectives: Sequence[Mapping[str, Any]] | None) -> list[dict] | None:
    if objectives is None:
        return None
    out = []
    for o in objectives:
        if not isinstance(o, Mapping) or not isinstance(o.get("metric"), str):
            raise LenzError(f"an objective must be {{'metric': str, 'minimize': bool}}, got {o!r}")
        out.append({"metric": o["metric"], "minimize": bool(o.get("minimize", False))})
    if len({o["metric"] for o in out}) != len(out):
        raise LenzError(f"duplicate objective metrics in {out}")
    return out


def _to_unit(p: RealParam, v: float) -> float:
    """Encoded (``[0, 1]``) coordinate of a raw value, without clipping."""
    if p.log_scale:
        return (math.log(v) - math.log(p.lower)) / (math.log(p.upper) - math.log(p.lower))
    return (v - p.lower) / (p.upper - p.lower)


def _fmt(v: float) -> str:
    return f"{v:g}"


def _constraint_text(c: Mapping[str, Any]) -> str:
    m, lo, hi = c["metric"], c["lower"], c["upper"]
    if lo is not None and hi is not None:
        return f"{_fmt(lo)} <= {m} <= {_fmt(hi)}"
    if hi is not None:
        return f"{m} <= {_fmt(hi)}"
    return f"{m} >= {_fmt(lo)}"


# --------------------------------------------------------------------------- problem


@dataclass
class Problem:
    """A black-box benchmark problem (see DESIGN.md §8).

    ``evaluate_fn`` receives a *validated* config (``space.validate``) and returns the
    noise-free metrics. ``objective`` / ``minimize`` / ``f_star`` describe the primary
    metric; for multi-objective problems they refer to the first entry of
    ``objectives`` (single-objective views such as regret use it).

    Fields after ``max_hv`` are extensions of the DESIGN contract:

    * ``outputs`` -- ``{metric: short description}`` listed in :meth:`brief` (lets the
      brief mention reported-but-not-optimized metrics such as ``cost``).
    * ``batch_fn`` -- optional vectorized evaluator on raw ``(n, d)`` arrays (columns in
      ``space.names`` order) used by :meth:`evaluate_array`.
    * ``shift_safe`` -- the known optimum is global over the region a randomized shift
      can expose, so :func:`randomize_problem` may shift it.
    * ``meta`` -- bookkeeping that is never shown to agents (family, dimension,
      randomization mapping, ...).

    The class attribute ``deterministic`` (True) records that ``evaluate_fn`` is
    noise-free, so with ``noise_std == 0`` the brief states that observations are
    noiseless; wrappers around a user black box that may be noisy by itself set it to
    False and the brief then says nothing about noise.
    """

    deterministic: ClassVar[bool] = True

    name: str
    space: SearchSpace
    evaluate_fn: MetricsFn
    objective: str
    minimize: bool
    f_star: float | None
    constraints: list[dict] = field(default_factory=list)
    noise_std: float = 0.0
    description: str = ""
    prior: str | None = None
    prior_hint: dict | None = None
    default_budget: int = 50
    x_star: dict | None = None
    objectives: list[dict] | None = None
    ref_point: list[float] | None = None
    max_hv: float | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    batch_fn: BatchFn | None = None
    shift_safe: bool = False
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.space, SearchSpace):
            self.space = SearchSpace.from_spec(self.space)
        self.minimize = bool(self.minimize)
        self.f_star = None if self.f_star is None else float(self.f_star)
        self.constraints = _normalize_constraints(self.constraints)
        self.objectives = _normalize_objectives(self.objectives)
        if self.objectives is not None:
            first = self.objectives[0]
            if first["metric"] != self.objective or first["minimize"] != self.minimize:
                raise LenzError(f"{self.name}: objective/minimize must match objectives[0] = {first}")
        if self.ref_point is not None:
            self.ref_point = [float(v) for v in self.ref_point]
            n_obj = len(self.objectives) if self.objectives else 1
            if len(self.ref_point) != n_obj:
                raise LenzError(f"{self.name}: ref_point needs {n_obj} entries, got {len(self.ref_point)}")
        self.max_hv = None if self.max_hv is None else float(self.max_hv)
        self.noise_std = float(self.noise_std)
        if not (self.noise_std >= 0 and math.isfinite(self.noise_std)):
            raise LenzError(f"noise_std must be a finite number >= 0, got {self.noise_std}")
        self.default_budget = int(self.default_budget)
        if self.default_budget < 1:
            raise LenzError(f"default_budget must be >= 1, got {self.default_budget}")
        if self.x_star is not None:
            self.x_star = dict(self.x_star)

    # ---- structure ---------------------------------------------------------------------
    @property
    def is_moo(self) -> bool:
        return self.objectives is not None and len(self.objectives) >= 2

    @property
    def u_star(self) -> float | None:
        """Optimal value of the primary objective in the maximization convention."""
        if self.f_star is None:
            return None
        return -self.f_star if self.minimize else self.f_star

    @property
    def objective_list(self) -> list[dict]:
        """``objectives`` if set, else the single primary objective, as ``[{"metric", "minimize"}]``."""
        if self.objectives:
            return [dict(o) for o in self.objectives]
        return [{"metric": self.objective, "minimize": self.minimize}]

    @property
    def metric_names(self) -> list[str]:
        """Objective metrics, then constraint metrics, then other declared outputs (no duplicates)."""
        names = [o["metric"] for o in self.objective_list]
        names += [c["metric"] for c in self.constraints]
        names += list(self.outputs)
        return list(dict.fromkeys(names))

    # ---- evaluation --------------------------------------------------------------------
    def evaluate(self, config: Mapping[str, Any]) -> dict[str, float]:
        """Noise-free metrics of ``config`` (validated against the space; raises ``LenzError``)."""
        cfg = self.space.validate(config)
        raw = self.evaluate_fn(cfg)
        if not isinstance(raw, Mapping):
            raise TypeError(f"{self.name}: evaluate_fn must return a dict of metrics, got {type(raw).__name__}")
        out = {str(k): _as_float(v) for k, v in raw.items()}
        required = [o["metric"] for o in self.objective_list] + [c["metric"] for c in self.constraints]
        missing = [m for m in required if m not in out]
        if missing:
            raise LenzError(f"{self.name}: evaluation did not return metrics {missing}")
        return out

    def evaluate_array(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Vectorized noise-free evaluation of raw points ``X`` (``(n, d)``, ``space.names`` order).

        Uses ``batch_fn`` when available; otherwise loops over :meth:`evaluate`. Points
        are not validated on the fast path, so callers must keep them inside the domain.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.space.d:
            raise LenzError(f"expected arrays with {self.space.d} columns, got {X.shape[1]}")
        if self.batch_fn is not None:
            return {k: np.asarray(v, dtype=float).reshape(X.shape[0]) for k, v in self.batch_fn(X).items()}
        rows = [self.evaluate(dict(zip(self.space.names, map(float, x)))) for x in X]
        keys = list(rows[0]) if rows else self.metric_names
        return {k: np.array([r.get(k, math.nan) for r in rows], dtype=float) for k in keys}

    def observe(self, config: Mapping[str, Any], rng: np.random.Generator | int | None = None) -> dict[str, float]:
        """Metrics as an experimenter sees them: noise-free values plus ``N(0, noise_std^2)`` per metric."""
        metrics = self.evaluate(config)
        if self.noise_std > 0:
            gen = make_rng(rng)
            eps = gen.standard_normal(len(metrics)) * self.noise_std
            metrics = {k: v + float(e) for (k, v), e in zip(metrics.items(), eps)}
        return metrics

    # ---- scoring -----------------------------------------------------------------------
    def utility(self, metrics: Mapping[str, Any]) -> float:
        """Primary objective in the maximization convention (``-inf`` if missing / non-finite)."""
        v = _as_float(metrics.get(self.objective)) if metrics is not None else math.nan
        if not math.isfinite(v):
            return -math.inf
        return -v if self.minimize else v

    def is_feasible(self, metrics: Mapping[str, Any]) -> bool:
        """True iff every constraint metric is finite and inside its ``[lower, upper]`` band."""
        for c in self.constraints:
            v = _as_float(metrics.get(c["metric"])) if metrics is not None else math.nan
            if not math.isfinite(v):
                return False
            if c["lower"] is not None and v < c["lower"]:
                return False
            if c["upper"] is not None and v > c["upper"]:
                return False
        return True

    def regret(self, metrics: Mapping[str, Any]) -> float:
        """Raw per-point regret ``u_star - utility(metrics)``; NaN when the optimum is unknown.

        It is >= 0 (up to rounding) for feasible points. An infeasible point can beat the
        constrained optimum and then has a negative raw value; the regret curves in
        :mod:`agentic_bo.metrics` charge infeasible points with a floor at 0 instead.
        """
        if self.u_star is None:
            return math.nan
        return self.u_star - self.utility(metrics)

    # ---- agent-facing text -------------------------------------------------------------
    def brief(self) -> str:
        """Markdown brief: task description, inputs, outputs (objective(s), constraints, other metrics), noise.

        The budget and the natural-language ``prior`` are deliberately left out; the
        harness adds them (the prior only when ``show_prior`` is on).
        """
        desc = self.description.strip() or (
            f"Optimize a black-box function of {self.space.d} inputs.")
        lines = ["## Task", desc, "", "## Inputs", self.space.describe(), "", "## Outputs"]
        objs = self.objective_list
        for o in objs:
            direction = "minimize (lower is better)" if o["minimize"] else "maximize (higher is better)"
            lines.append(self._output_line(o["metric"], f"objective; {direction}"))
        for c in self.constraints:
            lines.append(self._output_line(c["metric"], f"constraint; feasible iff `{_constraint_text(c)}`"))
        listed = {o["metric"] for o in objs} | {c["metric"] for c in self.constraints}
        for m in self.outputs:
            if m not in listed:
                lines.append(self._output_line(m, "reported for information; not optimized"))
        notes = []
        if self.is_moo:
            goal = "Goal: find a good set of trade-offs (an approximation of the Pareto front)"
            if self.ref_point is not None:
                ref = ", ".join(_fmt(v) for v in self.ref_point)
                order = ", ".join(o["metric"] for o in objs)
                goal += (f"; quality is the hypervolume dominated by the feasible evaluated points "
                         f"relative to the reference point [{ref}] (order: {order})")
            notes.append(goal + ".")
        if self.constraints:
            notes.append("Only feasible evaluations count toward the best value found.")
        if notes:
            lines += ["", " ".join(notes)]
        if self.noise_std > 0:
            lines += ["", "## Noise",
                      f"Observed metric values include independent Gaussian noise with standard deviation "
                      f"{_fmt(self.noise_std)} (in metric units); repeated evaluations of the same input can differ."]
        elif self.deterministic:
            lines += ["", "## Noise",
                      "Observations are noiseless: repeated evaluations of the same input return the same value."]
        return "\n".join(lines)

    def _output_line(self, metric: str, role: str) -> str:
        text = self.outputs.get(metric, "").strip()
        return f"- `{metric}`: {role}." + (f" {text[0].upper()}{text[1:]}." if text else "")

    def info(self) -> dict:
        """JSON-safe summary for reports (no callables, nothing agent-facing)."""
        return jsonable({
            "name": self.name, "d": self.space.d, "objective": self.objective, "minimize": self.minimize,
            "f_star": self.f_star, "u_star": self.u_star, "x_star": self.x_star, "constraints": self.constraints,
            "objectives": self.objectives, "ref_point": self.ref_point, "max_hv": self.max_hv,
            "noise_std": self.noise_std, "default_budget": self.default_budget, "shift_safe": self.shift_safe,
            "is_moo": self.is_moo, "has_prior": self.prior is not None, "meta": self.meta,
        })


# --------------------------------------------------------------------------- vectorized adapter


class VectorizedEvaluator:
    """Picklable ``config -> metrics`` adapter around a vectorized function of raw inputs.

    ``fn`` maps an ``(n, d)`` array (columns in ``names`` order) to either an ``(n,)``
    array (reported under ``metric``) or a dict of ``(n,)`` arrays. Module-level
    functions keep problems picklable for multiprocessing benchmark runs.
    """

    def __init__(self, names: Sequence[str], fn: Callable[[np.ndarray], Any], metric: str = "y"):
        self.names = list(names)
        self.fn = fn
        self.metric = metric

    def batch(self, X: np.ndarray) -> dict[str, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = self.fn(X)
        if isinstance(out, Mapping):
            return {k: np.asarray(v, dtype=float).reshape(X.shape[0]) for k, v in out.items()}
        return {self.metric: np.asarray(out, dtype=float).reshape(X.shape[0])}

    def __call__(self, config: Mapping[str, Any]) -> dict[str, float]:
        x = np.array([[float(config[n]) for n in self.names]])
        return {k: float(v[0]) for k, v in self.batch(x).items()}


# --------------------------------------------------------------------------- randomization


class CanonicalMap:
    """Affine (log-affine for log-scale params) map between a canonical ``[0,1]^d`` box and raw units.

    Canonical column ``k`` corresponds to original parameter ``perm[k]``; its encoded
    (``[0, 1]``) coordinate is ``u_k - shift_k``. Unlike ``SearchSpace.decode`` the map
    does not clip, so a shifted problem evaluates the original function slightly
    outside its native domain (which is why shifting needs ``shift_safe``).
    """

    def __init__(self, params: Sequence[RealParam], perm: Sequence[int], shift: Sequence[float]):
        self.names = [p.name for p in params]
        self.perm = np.asarray(perm, dtype=int)
        self.shift = np.asarray(shift, dtype=float)
        self.log = np.array([p.log_scale for p in params], dtype=bool)
        lo = np.array([p.lower for p in params], dtype=float)
        hi = np.array([p.upper for p in params], dtype=float)
        self.a = np.where(self.log, np.log(np.where(self.log, lo, 1.0)), lo)
        self.b = np.where(self.log, np.log(np.where(self.log, hi, 1.0)), hi)

    def to_raw(self, U: np.ndarray) -> np.ndarray:
        """Canonical ``(n, d)`` -> raw ``(n, d)`` in the ORIGINAL parameter order."""
        U = np.atleast_2d(np.asarray(U, dtype=float))
        E = np.empty_like(U)
        E[:, self.perm] = U - self.shift
        Z = self.a + E * (self.b - self.a)
        return np.where(self.log, np.exp(np.where(self.log, Z, 0.0)), Z)

    def to_canonical(self, X: np.ndarray) -> np.ndarray:
        """Raw ``(n, d)`` (original order) -> canonical ``(n, d)``."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Z = np.where(self.log, np.log(np.where(self.log, np.maximum(X, 1e-300), 1.0)), X)
        E = (Z - self.a) / (self.b - self.a)
        return E[:, self.perm] + self.shift

    def encoded_to_canonical(self, j: int, e: float) -> float:
        """Encoded coordinate of original param ``j`` -> canonical coordinate of its column."""
        k = int(np.flatnonzero(self.perm == j)[0])
        return float(e + self.shift[k])


class _CanonicalEvaluator:
    """Evaluate a canonical config by mapping it back to the original problem's raw units."""

    def __init__(self, tokens: Sequence[str], cmap: CanonicalMap, base_fn: MetricsFn, base_batch: BatchFn | None):
        self.tokens = list(tokens)
        self.cmap = cmap
        self.base_fn = base_fn
        self.base_batch = base_batch

    def batch(self, U: np.ndarray) -> dict[str, np.ndarray]:
        X = self.cmap.to_raw(U)
        if self.base_batch is not None:
            return {k: np.asarray(v, dtype=float) for k, v in self.base_batch(X).items()}
        rows = [self.base_fn(dict(zip(self.cmap.names, map(float, x)))) for x in X]
        keys = list(rows[0]) if rows else []
        return {k: np.array([_as_float(r.get(k)) for r in rows]) for k in keys}

    def __call__(self, config: Mapping[str, Any]) -> dict[str, float]:
        u = np.array([[float(config[t]) for t in self.tokens]])
        x = self.cmap.to_raw(u)[0]
        return dict(self.base_fn(dict(zip(self.cmap.names, map(float, x)))))


def _random_tokens(d: int, rng: np.random.Generator) -> list[str]:
    width = max(3, math.ceil(math.log(16 * d, 16)))
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < d:
        tok = f"x_{int(rng.integers(0, 16 ** width)):0{width}x}"
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def _anonymous_description(problem: Problem) -> str:
    d = problem.space.d
    inputs = f"{d} continuous input{'s' if d != 1 else ''} in [0, 1]"
    if problem.is_moo:
        objs = ", ".join(f"`{o['metric']}` ({'minimize' if o['minimize'] else 'maximize'})"
                         for o in problem.objective_list)
        n_obj = len(problem.objective_list)
        text = f"An anonymous black-box problem with {inputs} and {n_obj} competing objectives: {objs}."
    else:
        better = "lower" if problem.minimize else "higher"
        text = f"An anonymous black-box function of {inputs}; {better} is better."
    if problem.constraints:
        names = ", ".join(f"`{c['metric']}`" for c in problem.constraints)
        text += f" Each evaluation also returns the constraint metric(s) {names}; see Outputs for the feasibility rule."
    return text


def _map_prior_hint(hint: dict | None, space: SearchSpace, cmap: CanonicalMap, tokens: list[str]) -> dict | None:
    if not hint:
        return None
    idx = {n: j for j, n in enumerate(space.names)}
    token_of = {int(cmap.perm[k]): tokens[k] for k in range(len(tokens))}

    def to_c(name: str, v: float) -> float:
        j = idx[name]
        e = _to_unit(space[name], float(v))  # type: ignore[arg-type]
        return float(np.clip(cmap.encoded_to_canonical(j, e), 0.0, 1.0))

    out: dict = {}
    if isinstance(hint.get("config"), Mapping):
        out["config"] = {token_of[idx[n]]: to_c(n, v) for n, v in hint["config"].items() if n in idx}
    if isinstance(hint.get("region"), Mapping):
        region = {}
        for n, (lo, hi) in hint["region"].items():
            if n in idx:
                a, b = sorted((to_c(n, lo), to_c(n, hi)))
                region[token_of[idx[n]]] = [a, b]
        out["region"] = region
    return out or None


def randomize_problem(problem: Problem, seed: int = 0, *, shift: bool | None = None,
                      permute: bool = True) -> Problem:
    """Anonymized copy of ``problem`` on ``[0, 1]^d`` (paper App. A anti-memorization).

    * every parameter becomes ``RealParam(token, 0, 1)`` with a deterministic random
      name (``"x_3fa"``) and, if ``permute``, a random position;
    * ``shift`` (default: ``problem.shift_safe``) moves the optimum by up to
      ``SHIFT_FRACTION`` of each range, keeping it inside ``[0.05, 0.95]``;
      requesting a shift for a problem that is not shift-safe raises ``LenzError``;
    * ``f_star``, constraints, objectives and the reference point are unchanged
      (constraints keep being computed in the original coordinates); ``x_star`` and
      ``prior_hint`` are mapped; the description no longer names the function and the
      natural-language ``prior`` (which refers to the original parameters) is dropped.

    Everything is a deterministic function of ``(problem.name, seed)``.
    """
    if not all(isinstance(p, RealParam) for p in problem.space.params):
        raise LenzError(f"{problem.name}: randomization supports continuous (real) parameters only")
    do_shift = problem.shift_safe if shift is None else bool(shift)
    if do_shift and not problem.shift_safe:
        raise LenzError(f"{problem.name} is not shift-safe: its known optimum is only optimal on the native domain")
    if do_shift and problem.x_star is None:
        raise LenzError(f"{problem.name}: shifting needs a known x_star")

    d = problem.space.d
    rng = make_rng(int(seed), zlib.crc32(problem.name.encode()) & 0x7FFFFFFF, 0xB0)
    tokens = _random_tokens(d, rng)
    perm = rng.permutation(d) if permute else np.arange(d)
    params: list[RealParam] = list(problem.space.params)  # type: ignore[arg-type]

    s = np.zeros(d)
    e_star = None   # encoded optimum, in canonical column order
    if problem.x_star is not None:
        e_star = np.array([_to_unit(p, float(problem.x_star[p.name])) for p in params])[perm]
    if do_shift:
        assert e_star is not None
        # admissible shifts keep the optimum inside [margin, 1 - margin]; the interval is
        # non-empty for any x_star inside the domain
        lo = np.maximum(-SHIFT_FRACTION, SHIFT_MARGIN - e_star)
        hi = np.minimum(SHIFT_FRACTION, (1.0 - SHIFT_MARGIN) - e_star)
        if np.any(lo > hi):
            raise LenzError(f"{problem.name}: x_star must lie inside the domain to be shifted")
        s = lo + rng.uniform(size=d) * (hi - lo)

    cmap = CanonicalMap(params, perm, s)
    space = SearchSpace([RealParam(t, 0.0, 1.0) for t in tokens])
    evaluator = _CanonicalEvaluator(tokens, cmap, problem.evaluate_fn, problem.batch_fn)
    x_star = None
    if e_star is not None:
        x_star = {t: float(v) for t, v in zip(tokens, e_star + s)}

    outputs = {m: "" for m in problem.metric_names}
    meta = dict(problem.meta)
    meta.update({
        "randomized": True, "seed": int(seed), "shifted": bool(do_shift), "permuted": bool(permute),
        "original_names": [problem.space.names[int(j)] for j in perm], "shift": s.tolist(),
    })
    return replace(
        problem,
        space=space,
        evaluate_fn=evaluator,
        batch_fn=evaluator.batch,
        description=_anonymous_description(problem),
        prior=None,
        prior_hint=_map_prior_hint(problem.prior_hint, problem.space, cmap, tokens),
        x_star=x_star,
        outputs=outputs,
        meta=meta,
    )
