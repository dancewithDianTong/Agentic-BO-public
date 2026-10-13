import itertools
import json
import math

import numpy as np
import pytest

from agentic_bo.metrics import (best_so_far_curve, cumulative_regret_curve, dominated_hypervolume,
                                hypervolume_regret_curve, simple_regret_curve, summarize_curves)
from agentic_bo.utils import LenzError

NAN = float("nan")


# --------------------------------------------------------------------------- simple regret


def test_simple_regret_all_feasible():
    r = simple_regret_curve([1.0, 3.0, 2.0, 5.0], u_star=6.0)
    assert r.tolist() == [5.0, 3.0, 3.0, 1.0]
    assert simple_regret_curve(np.array([6.0]), 6.0).tolist() == [0.0]


def test_simple_regret_infeasible_convention():
    # raw regrets [2, 5, 3, 1, 4]; before the first feasible point: worst regret so far
    u = [4.0, 1.0, 3.0, 5.0, 2.0]
    feas = [False, False, True, False, True]
    assert simple_regret_curve(u, 6.0, feas).tolist() == [2.0, 5.0, 3.0, 3.0, 3.0]
    # an infeasible point better than everything never lowers the regret
    assert simple_regret_curve([1.0, 10.0], 6.0, [True, False]).tolist() == [5.0, 5.0]
    # no feasible point at all: worst point so far throughout
    assert simple_regret_curve([3.0, 5.0, 0.0], 6.0, [False] * 3).tolist() == [3.0, 3.0, 6.0]


def test_simple_regret_nonfinite_and_unknown_optimum():
    r = simple_regret_curve([NAN, -math.inf, 2.0, NAN], 6.0)
    assert math.isnan(r[0]) and math.isnan(r[1]) and r[2:].tolist() == [4.0, 4.0]
    assert np.all(np.isnan(simple_regret_curve([1.0, 2.0], None)))
    assert simple_regret_curve([], 1.0).size == 0


def test_simple_regret_early_stop_and_budget_checks():
    assert simple_regret_curve([1.0, 4.0], 6.0, budget=5).tolist() == [5.0, 2.0, 2.0, 2.0, 2.0]
    assert np.all(np.isnan(simple_regret_curve([], 6.0, budget=3)))
    with pytest.raises(LenzError, match="budget"):
        simple_regret_curve([1.0, 2.0, 3.0], 6.0, budget=2)
    with pytest.raises(LenzError, match="feasible"):
        simple_regret_curve([1.0, 2.0], 6.0, feasible=[True])


def test_simple_regret_is_nonincreasing_once_feasible():
    rng = np.random.default_rng(0)
    for _ in range(50):
        u = rng.normal(size=30)
        feas = rng.uniform(size=30) < 0.6
        r = simple_regret_curve(u, 5.0, feas)
        first = np.argmax(feas) if feas.any() else 30
        assert np.all(np.diff(r[first:]) <= 0)
        assert np.all(np.diff(r[:first]) >= 0)          # worst-so-far before feasibility
        assert np.all(r >= 0)


# --------------------------------------------------------------------------- cumulative regret


def test_cumulative_regret_all_feasible():
    assert cumulative_regret_curve([1.0, 3.0, 2.0, 5.0], 6.0).tolist() == [5.0, 8.0, 12.0, 13.0]


def test_cumulative_regret_infeasible_charges():
    # raw regrets [2, 5, 3, 1, 4]; infeasible charges: max(raw, max seen) = [2, 5, -, 5, -]
    u = [4.0, 1.0, 3.0, 5.0, 2.0]
    feas = [False, False, True, False, True]
    assert cumulative_regret_curve(u, 6.0, feas).tolist() == [2.0, 7.0, 10.0, 15.0, 19.0]


def test_cumulative_regret_infeasible_never_cheaper():
    rng = np.random.default_rng(1)
    for _ in range(50):
        u = rng.normal(size=25)
        feas = rng.uniform(size=25) < 0.5
        rho_inf = np.diff(np.r_[0.0, cumulative_regret_curve(u, 4.0, feas)])
        rho_all = np.diff(np.r_[0.0, cumulative_regret_curve(u, 4.0)])
        assert np.all(rho_inf >= rho_all - 1e-12)
        assert np.all(rho_inf[~feas] >= np.maximum.accumulate(4.0 - u)[~feas] - 1e-12)


def test_cumulative_regret_failed_evaluations():
    r = cumulative_regret_curve([2.0, NAN, 5.0], 6.0)        # NaN charged the max regret seen (4)
    assert r.tolist() == [4.0, 8.0, 9.0]
    # nothing seen yet -> undefined at t = 1; from the first finite observation on, the
    # leading failure is charged that observation's worst regret (1)
    r = cumulative_regret_curve([NAN, 5.0], 6.0)
    assert math.isnan(r[0]) and r[1] == 2.0
    assert np.all(np.isnan(cumulative_regret_curve([1.0], None)))


def test_cumulative_regret_early_stop():
    # unused evaluations are charged the regret of the best feasible point (u = 4 -> 2)
    assert cumulative_regret_curve([1.0, 4.0], 6.0, budget=5).tolist() == [5.0, 7.0, 9.0, 11.0, 13.0]
    # the best *feasible* point counts, not a better infeasible one
    r = cumulative_regret_curve([1.0, 5.5], 6.0, [True, False], budget=3)
    assert r.tolist() == [5.0, 10.0, 15.0]
    # no feasible point: charge the max regret seen
    assert cumulative_regret_curve([1.0, 4.0], 6.0, [False, False], budget=4).tolist() == [5.0, 10.0, 15.0, 20.0]
    # stopping early is never cheaper than continuing to exploit the incumbent
    full = cumulative_regret_curve([1.0, 4.0, 4.0, 4.0, 4.0], 6.0)
    assert np.allclose(cumulative_regret_curve([1.0, 4.0], 6.0, budget=5), full)
    assert np.all(np.isnan(cumulative_regret_curve([], 6.0, budget=2)))
    with pytest.raises(LenzError, match="budget"):
        cumulative_regret_curve([1.0, 2.0], 6.0, budget=1)


# --------------------------------------------------------------------------- best so far


def test_best_so_far_curve():
    assert best_so_far_curve([1.0, 3.0, 2.0]).tolist() == [1.0, 3.0, 3.0]
    b = best_so_far_curve([5.0, 1.0, 3.0], [False, True, True], budget=4)
    assert math.isnan(b[0]) and b[1:].tolist() == [1.0, 3.0, 3.0]


# --------------------------------------------------------------------------- hypervolume


def brute_force_hv_2d(Y, ref, n=1201):
    """Grid estimate of the dominated area (maximization), for cross-checking."""
    lo = ref
    hi = np.max(Y, axis=0)
    gx = np.linspace(lo[0], hi[0], n)
    gy = np.linspace(lo[1], hi[1], n)
    cx, cy = (gx[:-1] + gx[1:]) / 2, (gy[:-1] + gy[1:]) / 2
    GX, GY = np.meshgrid(cx, cy)
    dom = np.zeros(GX.shape, dtype=bool)
    for y in Y:
        dom |= (GX <= y[0]) & (GY <= y[1])
    return dom.mean() * (hi[0] - lo[0]) * (hi[1] - lo[1])


def test_dominated_hypervolume_known_values():
    # minimization: union of boxes [y, ref] = 16 + 1 + 1
    Y = np.array([[1.0, 5.0], [2.0, 2.0], [5.0, 1.0]])
    assert dominated_hypervolume(Y, [6.0, 6.0], [True, True]) == pytest.approx(18.0)
    # maximization with points outside the reference box ignored
    assert dominated_hypervolume([[2.0, 3.0], [-1.0, 10.0]], [0.0, 0.0]) == pytest.approx(6.0)
    # 1-D and 3-D
    assert dominated_hypervolume([[3.0], [5.0]], [1.0]) == pytest.approx(4.0)
    Y3 = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
    assert dominated_hypervolume(Y3, [0.0, 0.0, 0.0]) == pytest.approx(4.0)  # 6 - 3 + 1
    assert dominated_hypervolume(np.zeros((0, 2)), [0.0, 0.0]) == 0.0
    # duplicate and dominated points change nothing
    assert dominated_hypervolume(np.vstack([Y, Y, [[3.0, 3.0]]]), [6.0, 6.0], [True, True]) == pytest.approx(18.0)


def test_dominated_hypervolume_matches_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(5):
        Y = rng.uniform(0, 1, size=(12, 2))
        assert dominated_hypervolume(Y, [0.0, 0.0]) == pytest.approx(brute_force_hv_2d(Y, np.zeros(2)), abs=5e-3)
    # 3-D: inclusion-exclusion over all subsets
    Y = rng.uniform(0.1, 1, size=(6, 3))
    ie = 0.0
    for k in range(1, 7):
        for sub in itertools.combinations(range(6), k):
            ie += (-1) ** (k + 1) * np.prod(np.min(Y[list(sub)], axis=0))
    assert dominated_hypervolume(Y, [0.0, 0.0, 0.0]) == pytest.approx(ie, rel=1e-10)
    # mixed directions: flipping a column and the reference point preserves the volume
    assert dominated_hypervolume(Y * [1, -1, 1], [0.0, 0.0, 0.0], [False, True, False]) == pytest.approx(ie)


def test_hypervolume_regret_curve():
    Y = np.array([[1.0, 5.0], [2.0, 2.0], [5.0, 1.0]])
    r = hypervolume_regret_curve(Y, None, [6.0, 6.0], 20.0, [True, True])
    assert r.tolist() == pytest.approx([15.0, 3.0, 2.0])
    # infeasible / out-of-reference / non-finite rows add nothing
    Y2 = np.array([[2.0, 2.0], [1.0, 5.0], [7.0, 0.5], [NAN, 1.0], [5.0, 1.0]])
    r2 = hypervolume_regret_curve(Y2, [True, False, True, True, True], [6.0, 6.0], 20.0, [True, True])
    assert r2.tolist() == pytest.approx([4.0, 4.0, 4.0, 4.0, 3.0])
    # early stop: last value repeated; unknown max_hv -> NaN
    r3 = hypervolume_regret_curve(Y[:2], None, [6.0, 6.0], 20.0, [True, True], budget=4)
    assert r3.tolist() == pytest.approx([15.0, 3.0, 3.0, 3.0])
    assert np.all(np.isnan(hypervolume_regret_curve(Y, None, [6.0, 6.0], None, [True, True])))
    with pytest.raises(LenzError, match="minimize_flags"):
        hypervolume_regret_curve(Y, None, [6.0, 6.0], 20.0, [True])
    with pytest.raises(LenzError, match="feasible"):
        hypervolume_regret_curve(Y, [True], [6.0, 6.0], 20.0, [True, True])


def test_hypervolume_regret_curve_is_monotone_and_matches_batch():
    rng = np.random.default_rng(5)
    Y = rng.uniform(0, 10, size=(40, 2))
    feas = rng.uniform(size=40) < 0.7
    r = hypervolume_regret_curve(Y, feas, [10.0, 10.0], 100.0, [True, True])
    assert np.all(np.diff(r) <= 1e-12)
    for t in (1, 10, 25, 40):
        hv = dominated_hypervolume(Y[:t][feas[:t]], [10.0, 10.0], [True, True])
        assert r[t - 1] == pytest.approx(100.0 - hv)
    # 3 objectives (recursive slicing path), mixed directions
    Y3 = rng.uniform(0, 1, size=(15, 3))
    r3 = hypervolume_regret_curve(Y3, None, [0.0, 1.0, 0.0], 1.0, [False, True, False])
    assert r3[-1] == pytest.approx(1.0 - dominated_hypervolume(Y3, [0.0, 1.0, 0.0], [False, True, False]))
    assert np.all(np.diff(r3) <= 1e-12)


# --------------------------------------------------------------------------- summaries


def test_summarize_curves_values():
    curves = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0, 5.0]), np.array([2.0, 6.0, 4.0])]
    s = summarize_curves(curves)
    assert set(s) == {"median", "q25", "q75", "mean", "sem"}
    assert s["median"] == [2.0, 4.0, 4.0]
    assert s["mean"] == pytest.approx([2.0, 4.0, 4.0])
    assert s["q25"] == pytest.approx([1.5, 3.0, 3.5]) and s["q75"] == pytest.approx([2.5, 5.0, 4.5])
    assert s["sem"] == pytest.approx([1.0 / math.sqrt(3), 2.0 / math.sqrt(3), 1.0 / math.sqrt(3)])
    json.dumps(s)
    assert all(isinstance(v, float) for v in s["median"])


def test_summarize_curves_padding_and_edge_cases():
    curves = [np.array([3.0, 1.0]), np.array([4.0, 2.0, 0.5])]
    last = summarize_curves(curves)                    # [3, 1, 1] vs [4, 2, 0.5]
    assert last["median"] == pytest.approx([3.5, 1.5, 0.75])
    nan = summarize_curves(curves, pad="nan")          # third point: only the second run
    assert nan["median"] == pytest.approx([3.5, 1.5, 0.5]) and nan["sem"][2] == 0.0
    single = summarize_curves([np.array([1.0, 2.0])])
    assert single["median"] == [1.0, 2.0] and single["sem"] == [0.0, 0.0]
    assert summarize_curves([]) == {k: [] for k in ("median", "q25", "q75", "mean", "sem")}
    with_nan = summarize_curves([np.array([NAN, 1.0]), np.array([2.0, 3.0])])
    assert with_nan["median"] == pytest.approx([2.0, 2.0])
    all_nan = summarize_curves([np.array([NAN]), np.array([NAN])])
    assert math.isnan(all_nan["median"][0]) and all_nan["sem"] == [0.0]
    with pytest.raises(LenzError, match="pad"):
        summarize_curves(curves, pad="zero")


def test_curves_from_a_problem_run():
    """End-to-end: regret curves of random evaluations on a constrained benchmark."""
    from agentic_bo.problems import get_problem

    p = get_problem("constrained_hartmann6", seed=1, randomize=True)
    rng = np.random.default_rng(0)
    cfgs = p.space.sample_uniform(30, rng) + [p.x_star]
    metrics = [p.evaluate(c) for c in cfgs]
    u = np.array([p.utility(m) for m in metrics])
    feas = np.array([p.is_feasible(m) for m in metrics])
    sr = simple_regret_curve(u, p.u_star, feas)
    cr = cumulative_regret_curve(u, p.u_star, feas, budget=40)
    assert sr[-1] == pytest.approx(0.0, abs=1e-9) and np.all(sr >= -1e-9)
    assert cr.size == 40 and np.all(np.diff(cr) >= -1e-9)
    assert cr[-1] == pytest.approx(cr[30])                # remaining budget charged regret 0 (optimum found)


# --------------------------------------------------------------------------- review regressions


def test_cumulative_regret_survives_a_failed_first_evaluation():
    # a failed first evaluation used to turn the whole curve into NaN (cumsum of a NaN);
    # it is charged the worst regret at the first finite observation (here 1), and only the
    # entries before that observation are undefined
    r = cumulative_regret_curve([NAN, 5.0, 4.0], 6.0)
    assert math.isnan(r[0]) and r[1:].tolist() == [2.0, 4.0]
    r = cumulative_regret_curve([-math.inf, NAN, 3.0, 5.0], 6.0, [False, False, True, True])
    assert np.all(np.isnan(r[:2])) and r[2:].tolist() == [9.0, 10.0]
    # an infeasible first finite point: leading failures pay max(raw, seen) of that point
    r = cumulative_regret_curve([NAN, 2.0, 5.0], 6.0, [False, False, True])
    assert math.isnan(r[0]) and r[1:].tolist() == [8.0, 9.0]
    # with early stop the padding still applies; nothing finite at all stays undefined
    r = cumulative_regret_curve([NAN, 5.0], 6.0, budget=4)
    assert math.isnan(r[0]) and r[1:].tolist() == [2.0, 3.0, 4.0]
    assert np.all(np.isnan(cumulative_regret_curve([NAN, -math.inf], 6.0, budget=3)))


def test_regret_of_infeasible_points_is_never_negative():
    # minimize x s.t. x >= 0.5 (f* = 0.5, u* = -0.5): infeasible points x < 0.5 have u > u*,
    # and used to give negative simple / cumulative regret (an infeasible policy beat the optimum)
    u = [-0.1, -0.2, -0.3, -0.9]
    feas = [False, False, False, True]
    assert simple_regret_curve(u, -0.5, feas).tolist() == pytest.approx([0.0, 0.0, 0.0, 0.4])
    assert cumulative_regret_curve(u, -0.5, feas).tolist() == pytest.approx([0.0, 0.0, 0.0, 0.4])
    # raw regrets [-0.4, 0.3, -0.3], all infeasible: charges max(raw, worst seen, 0)
    u2 = [-0.1, -0.8, -0.2]
    assert simple_regret_curve(u2, -0.5, [False] * 3).tolist() == pytest.approx([0.0, 0.3, 0.3])
    assert cumulative_regret_curve(u2, -0.5, [False] * 3).tolist() == pytest.approx([0.0, 0.3, 0.6])
    assert cumulative_regret_curve(u2, -0.5, [False] * 3, budget=5).tolist() == pytest.approx(
        [0.0, 0.3, 0.6, 0.9, 1.2])
    rng = np.random.default_rng(3)
    for _ in range(50):
        uu = rng.normal(size=20)
        ff = rng.uniform(size=20) < 0.4
        assert np.all(simple_regret_curve(uu, 0.0, ff)[~np.cumsum(ff).astype(bool)] >= 0)
        rho = np.diff(np.r_[0.0, cumulative_regret_curve(uu, 0.0, ff)])
        assert np.all(rho[~ff] >= 0)


def test_hypervolume_regret_curve_without_evaluations_is_max_hv():
    # no evaluation dominates nothing: HV_t = 0, so an early stop at t = 0 is charged max_hv
    # (it used to be NaN, which dropped such runs from every statistic)
    r = hypervolume_regret_curve(np.zeros((0, 2)), None, [6.0, 6.0], 20.0, [True, True], budget=3)
    assert r.tolist() == [20.0, 20.0, 20.0]
    assert hypervolume_regret_curve(np.zeros((0, 2)), [], [6.0, 6.0], 20.0, [True, True]).size == 0
    assert np.all(np.isnan(hypervolume_regret_curve(np.zeros((0, 2)), None, [6.0, 6.0], None, [True, True],
                                                    budget=2)))
