"""Classic synthetic test functions (raw domains, minimization, known optima).

Definitions, domains and optima follow the SFU Virtual Library of Simulation
Experiments (Surjanovic & Bingham) and BoTorch's ``test_functions``. Optimal values are
evaluated at high-precision minimizers (polished with a Newton solve on the gradient),
so ``regret(evaluate(x_star))`` is zero up to rounding; the rounded literature values
are checked in the tests.

Every function maps a raw ``(n, d)`` array to ``(n,)`` values and is vectorized.

**Shift safety** (see ``randomize_problem``): a randomized shift evaluates a function
up to a quarter of each range outside its native box, so shifting is only allowed when
the known optimum is the global minimum over (at least) that extended box:

* Branin: ``f >= s*t = 5/(4 pi)`` everywhere (the squared term is >= 0 and
  ``cos >= -1``), attained on infinitely many points -> safe.
* Six-hump camel: a coercive polynomial (``x1^6/3`` and ``4 x2^4`` dominate) whose
  stationary points all lie in ``[-2, 2] x [-1, 1]``; its two global minimizers
  ``+-(0.0898, -0.7127)`` are global over ``R^2`` (also checked numerically on
  ``[-10, 10]^2`` in the tests) -> safe.
* Goldstein-Price: ``f = (1 + A)(30 + B)`` with
  ``A = (x1+x2+1)^2 (3 s^2 - 14 s + 19)``, ``s = x1 + x2``, and the quadratic in ``s``
  has negative discriminant, so ``1 + A >= 1`` (equality iff ``x1 + x2 = -1``);
  ``B = t^2 (3 t^2 - 16 t + 18)`` with ``t = 2 x1 - 3 x2`` has minimum ``-27`` at
  ``t = 3``, so ``30 + B >= 3``. Hence ``f >= 3`` on ``R^2`` with equality only at
  ``(0, -1)`` -> safe.
* Hartmann-3/6: all centres ``P_i`` lie in the unit cube and ``A_ij > 0``; projecting
  any point onto the cube shrinks every ``|x_j - P_ij|``, so each exponential term grows
  and ``f`` decreases. The minimum over ``R^d`` is attained inside the cube -> safe.
* Ackley, Levy, Rosenbrock, Rastrigin, Griewank: sums of non-negative terms that vanish
  at the optimum (``f >= 0 = f*``); Styblinski-Tang is separable with each 1-D term
  minimized globally at ``-2.9035`` -> all safe.
* Constrained Hartmann-6: the objective is safe as above and the constraint
  ``||x||^2 <= 1`` is evaluated in the original coordinates, so the (feasible) optimum
  ``x*`` stays feasible after a shift and remains optimal over the feasible set -> safe.
* Branin-Currin: Currin is undefined / unbounded below for ``x2 < 0`` and the
  hypervolume optimum is tied to the unit square -> NOT safe.
"""

from __future__ import annotations

import math
import re
from typing import Callable

import numpy as np

from ..space import RealParam, SearchSpace
from ..utils import LenzError
from .base import Problem, VectorizedEvaluator

__all__ = [
    "branin", "six_hump_camel", "goldstein_price", "hartmann3", "hartmann6", "ackley", "levy",
    "rosenbrock", "styblinski_tang", "rastrigin", "griewank", "constrained_hartmann6_metrics",
    "branin_currin_metrics", "FIXED_BUILDERS", "FAMILIES", "FAMILY_PATTERN", "make_family_problem",
    "MAX_FAMILY_DIM",
]

MAX_FAMILY_DIM = 1000

# --------------------------------------------------------------------------- functions


def branin(X: np.ndarray) -> np.ndarray:
    """Branin-Hoo on ``[-5, 10] x [0, 15]``; ``f* = 5 / (4 pi) = 0.397887`` at three minimizers."""
    x1, x2 = X[:, 0], X[:, 1]
    b, c, t = 5.1 / (4.0 * math.pi ** 2), 5.0 / math.pi, 1.0 / (8.0 * math.pi)
    return (x2 - b * x1 ** 2 + c * x1 - 6.0) ** 2 + 10.0 * (1.0 - t) * np.cos(x1) + 10.0


def six_hump_camel(X: np.ndarray) -> np.ndarray:
    """Six-hump camel on ``[-3, 3] x [-2, 2]``; ``f* = -1.0316`` at ``+-(0.0898, -0.7126)``."""
    x1, x2 = X[:, 0], X[:, 1]
    x1s, x2s = x1 ** 2, x2 ** 2
    return (4.0 - 2.1 * x1s + x1s ** 2 / 3.0) * x1s + x1 * x2 + (-4.0 + 4.0 * x2s) * x2s


def goldstein_price(X: np.ndarray) -> np.ndarray:
    """Goldstein-Price on ``[-2, 2]^2``; ``f* = 3`` at ``(0, -1)``."""
    x1, x2 = X[:, 0], X[:, 1]
    a = 1.0 + (x1 + x2 + 1.0) ** 2 * (19.0 - 14.0 * x1 + 3.0 * x1 ** 2 - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * x2 ** 2)
    b = 30.0 + (2.0 * x1 - 3.0 * x2) ** 2 * (
        18.0 - 32.0 * x1 + 12.0 * x1 ** 2 + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * x2 ** 2)
    return a * b


_H_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H3_A = np.array([[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]])
_H3_P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]], dtype=float)
_H6_A = np.array([[10.0, 3.0, 17.0, 3.5, 1.7, 8.0], [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                  [3.0, 3.5, 1.7, 10.0, 17.0, 8.0], [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]])
_H6_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]], dtype=float)


def _hartmann(X: np.ndarray, A: np.ndarray, P: np.ndarray) -> np.ndarray:
    inner = np.einsum("ij,nij->ni", A, (X[:, None, :] - P[None, :, :]) ** 2)
    return -np.exp(-inner) @ _H_ALPHA


def hartmann3(X: np.ndarray) -> np.ndarray:
    """Hartmann-3 on ``[0, 1]^3``; ``f* = -3.86278`` at ``(0.114614, 0.555649, 0.852547)``."""
    return _hartmann(X, _H3_A, _H3_P)


def hartmann6(X: np.ndarray) -> np.ndarray:
    """Hartmann-6 on ``[0, 1]^6``; ``f* = -3.32237`` at
    ``(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)``."""
    return _hartmann(X, _H6_A, _H6_P)


def ackley(X: np.ndarray) -> np.ndarray:
    """Ackley (a=20, b=0.2, c=2 pi) on ``[-32.768, 32.768]^d``; ``f* = 0`` at the origin."""
    a, b, c = 20.0, 0.2, 2.0 * math.pi
    r = np.sqrt(np.mean(X ** 2, axis=1))
    return -a * np.exp(-b * r) - np.exp(np.mean(np.cos(c * X), axis=1)) + a + math.e


def levy(X: np.ndarray) -> np.ndarray:
    """Levy on ``[-10, 10]^d``; ``f* = 0`` at ``(1, ..., 1)``."""
    w = 1.0 + (X - 1.0) / 4.0
    head = np.sin(math.pi * w[:, 0]) ** 2
    mid = np.sum((w[:, :-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(math.pi * w[:, :-1] + 1.0) ** 2), axis=1)
    tail = (w[:, -1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * math.pi * w[:, -1]) ** 2)
    return head + mid + tail


def rosenbrock(X: np.ndarray) -> np.ndarray:
    """Rosenbrock on ``[-5, 10]^d`` (``d >= 2``); ``f* = 0`` at ``(1, ..., 1)``."""
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (X[:, :-1] - 1.0) ** 2, axis=1)


ST_ARGMIN = -2.903534027771177   # root of 4 x^3 - 32 x + 5 = 0 (Newton-polished)


def styblinski_tang(X: np.ndarray) -> np.ndarray:
    """Styblinski-Tang on ``[-5, 5]^d``; ``f* = -39.16617 d`` at ``(-2.903534, ...)``."""
    return 0.5 * np.sum(X ** 4 - 16.0 * X ** 2 + 5.0 * X, axis=1)


def rastrigin(X: np.ndarray) -> np.ndarray:
    """Rastrigin on ``[-5.12, 5.12]^d``; ``f* = 0`` at the origin."""
    return 10.0 * X.shape[1] + np.sum(X ** 2 - 10.0 * np.cos(2.0 * math.pi * X), axis=1)


def griewank(X: np.ndarray) -> np.ndarray:
    """Griewank on ``[-600, 600]^d``; ``f* = 0`` at the origin."""
    i = np.arange(1, X.shape[1] + 1, dtype=float)
    return np.sum(X ** 2, axis=1) / 4000.0 - np.prod(np.cos(X / np.sqrt(i)), axis=1) + 1.0


def constrained_hartmann6_metrics(X: np.ndarray) -> dict[str, np.ndarray]:
    """BoTorch's ConstrainedHartmann: objective ``y`` = Hartmann-6, constraint ``c = ||x||^2 <= 1``."""
    return {"y": hartmann6(X), "c": np.sum(X ** 2, axis=1)}


def _currin(X: np.ndarray) -> np.ndarray:
    x0, x1 = X[:, 0], X[:, 1]
    with np.errstate(divide="ignore", over="ignore"):
        factor = 1.0 - np.exp(-1.0 / (2.0 * x1))      # x1 = 0 -> exp(-inf) = 0 -> factor 1 (BoTorch limit)
    factor = np.where(x1 == 0.0, 1.0, factor)
    num = 2300.0 * x0 ** 3 + 1900.0 * x0 ** 2 + 2092.0 * x0 + 60.0
    den = 100.0 * x0 ** 3 + 500.0 * x0 ** 2 + 4.0 * x0 + 20.0
    return factor * num / den


def branin_currin_metrics(X: np.ndarray) -> dict[str, np.ndarray]:
    """BoTorch's BraninCurrin on ``[0, 1]^2``: ``f1`` = rescaled Branin, ``f2`` = Currin (both minimized)."""
    raw = np.column_stack([15.0 * X[:, 0] - 5.0, 15.0 * X[:, 1]])
    return {"f1": branin(raw), "f2": _currin(X)}


# --------------------------------------------------------------------------- problem builders


def _space(names: list[str], bounds: list[tuple[float, float]]) -> SearchSpace:
    return SearchSpace([RealParam(n, lo, hi) for n, (lo, hi) in zip(names, bounds)])


def _names(d: int) -> list[str]:
    return [f"x{i}" for i in range(d)]


def _single(name: str, fn: Callable[[np.ndarray], np.ndarray], bounds: list[tuple[float, float]],
            x_star: list[float], description: str, budget: int, *, family: str | None = None) -> Problem:
    names = _names(len(bounds))
    ev = VectorizedEvaluator(names, fn, metric="y")
    x = np.asarray([x_star], dtype=float)
    return Problem(
        name=name, space=_space(names, bounds), evaluate_fn=ev, objective="y", minimize=True,
        f_star=float(fn(x)[0]), description=description, default_budget=budget,
        x_star={n: float(v) for n, v in zip(names, x_star)},
        outputs={"y": "function value"}, batch_fn=ev.batch, shift_safe=True,
        meta={"family": family or name, "dim": len(bounds), "randomized": False},
    )


def make_branin() -> Problem:
    return _single("branin", branin, [(-5.0, 10.0), (0.0, 15.0)], [math.pi, 2.275],
                   "Minimize the 2-D Branin-Hoo test function on its standard domain.", 50)


def make_six_hump_camel() -> Problem:
    return _single("six_hump_camel", six_hump_camel, [(-3.0, 3.0), (-2.0, 2.0)],
                   [0.08984201310031807, -0.7126564030207396],
                   "Minimize the 2-D six-hump camel test function on its standard domain.", 50)


def make_goldstein_price() -> Problem:
    return _single("goldstein_price", goldstein_price, [(-2.0, 2.0), (-2.0, 2.0)], [0.0, -1.0],
                   "Minimize the 2-D Goldstein-Price test function on its standard domain.", 50)


def make_hartmann3() -> Problem:
    return _single("hartmann3", hartmann3, [(0.0, 1.0)] * 3,
                   [0.11458887665506896, 0.5556488946169301, 0.8525469846866774],
                   "Minimize the 3-D Hartmann test function on the unit cube.", 75)


HARTMANN6_ARGMIN = [0.20168951100670543, 0.15001069182345797, 0.47687397422189703,
                    0.2753324304940561, 0.31165161660011326, 0.6573005340656204]


def make_hartmann6() -> Problem:
    return _single("hartmann6", hartmann6, [(0.0, 1.0)] * 6, HARTMANN6_ARGMIN,
                   "Minimize the 6-D Hartmann test function on the unit cube.", 100)


def make_constrained_hartmann6() -> Problem:
    names = _names(6)
    ev = VectorizedEvaluator(names, constrained_hartmann6_metrics)
    return Problem(
        name="constrained_hartmann6", space=_space(names, [(0.0, 1.0)] * 6), evaluate_fn=ev,
        objective="y", minimize=True, f_star=float(hartmann6(np.array([HARTMANN6_ARGMIN]))[0]),
        constraints=[{"metric": "c", "lower": None, "upper": 1.0}],
        description=("Minimize the 6-D Hartmann test function on the unit cube subject to the "
                     "constraint that the squared Euclidean norm of the input is at most 1."),
        default_budget=100, x_star=dict(zip(names, HARTMANN6_ARGMIN)),
        outputs={"y": "function value", "c": "squared Euclidean norm of the input vector"},
        batch_fn=ev.batch, shift_safe=True,
        meta={"family": "constrained_hartmann6", "dim": 6, "randomized": False},
    )


BRANIN_CURRIN_REF_POINT = [18.0, 6.0]
BRANIN_CURRIN_MAX_HV = 59.36011874867746


def make_branin_currin() -> Problem:
    names = _names(2)
    ev = VectorizedEvaluator(names, branin_currin_metrics)
    x_star = [(math.pi + 5.0) / 15.0, 2.275 / 15.0]   # a minimizer of f1 (rescaled Branin)
    return Problem(
        name="branin_currin", space=_space(names, [(0.0, 1.0)] * 2), evaluate_fn=ev,
        objective="f1", minimize=True,
        f_star=float(branin_currin_metrics(np.array([x_star]))["f1"][0]),
        description=("Two-objective problem on the unit square: minimize both the rescaled Branin "
                     "function (f1) and the Currin exponential function (f2)."),
        default_budget=50, x_star=dict(zip(names, x_star)),
        objectives=[{"metric": "f1", "minimize": True}, {"metric": "f2", "minimize": True}],
        ref_point=list(BRANIN_CURRIN_REF_POINT), max_hv=BRANIN_CURRIN_MAX_HV,
        outputs={"f1": "first objective", "f2": "second objective"},
        batch_fn=ev.batch, shift_safe=False,
        meta={"family": "branin_currin", "dim": 2, "randomized": False},
    )


FIXED_BUILDERS: dict[str, Callable[[], Problem]] = {
    "branin": make_branin,
    "six_hump_camel": make_six_hump_camel,
    "goldstein_price": make_goldstein_price,
    "hartmann3": make_hartmann3,
    "hartmann6": make_hartmann6,
    "constrained_hartmann6": make_constrained_hartmann6,
    "branin_currin": make_branin_currin,
}

# family -> (function, (lower, upper), optimal coordinate, display name, minimum dimension)
FAMILIES: dict[str, tuple[Callable[[np.ndarray], np.ndarray], tuple[float, float], float, str, int]] = {
    "ackley": (ackley, (-32.768, 32.768), 0.0, "Ackley", 1),
    "levy": (levy, (-10.0, 10.0), 1.0, "Levy", 1),
    "rosenbrock": (rosenbrock, (-5.0, 10.0), 1.0, "Rosenbrock", 2),
    "styblinski_tang": (styblinski_tang, (-5.0, 5.0), ST_ARGMIN, "Styblinski-Tang", 1),
    "rastrigin": (rastrigin, (-5.12, 5.12), 0.0, "Rastrigin", 1),
    "griewank": (griewank, (-600.0, 600.0), 0.0, "Griewank", 1),
}
FAMILY_PATTERN = re.compile(r"^(" + "|".join(sorted(FAMILIES, key=len, reverse=True)) + r")(\d+)$")


def family_budget(d: int) -> int:
    """Default budget for a ``d``-dimensional family member (ackley10 -> 150, ackley20 -> 200)."""
    return int(min(200, max(50, 15 * d)))


def make_family_problem(family: str, d: int) -> Problem:
    """Build ``{family}{d}``, e.g. ``make_family_problem("ackley", 10)``."""
    if family not in FAMILIES:
        raise LenzError(f"unknown function family {family!r}; known: {sorted(FAMILIES)}")
    fn, (lo, hi), opt, label, d_min = FAMILIES[family]
    d = int(d)
    if not d_min <= d <= MAX_FAMILY_DIM:
        raise LenzError(f"{family} needs a dimension in [{d_min}, {MAX_FAMILY_DIM}], got {d}")
    desc = f"Minimize the {d}-D {label} test function on [{lo:g}, {hi:g}]^{d}."
    return _single(f"{family}{d}", fn, [(lo, hi)] * d, [opt] * d, desc, family_budget(d), family=family)
