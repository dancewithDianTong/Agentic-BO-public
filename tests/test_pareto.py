import itertools

import numpy as np
import pytest

from agentic_bo.lenz.pareto import default_ref_point, hvi_2d, hypervolume, pareto_mask
from agentic_bo.utils import LenzError


# --------------------------------------------------------------------------- brute-force references


def brute_pareto(Y):
    Y = np.asarray(Y, float)
    out = []
    for i, y in enumerate(Y):
        if not np.all(np.isfinite(y)):
            out.append(False)
            continue
        dominated = any(
            np.all(np.isfinite(o)) and np.all(o >= y) and np.any(o > y) for j, o in enumerate(Y) if j != i
        )
        out.append(not dominated)
    return np.array(out)


def grid_hypervolume(Y, ref):
    """Exact HV for integer-valued points and an integer ref point by counting unit cells."""
    Y = np.asarray(Y, int)
    ref = np.asarray(ref, int)
    hi = Y.max(axis=0)
    axes = [np.arange(r, h) + 0.5 for r, h in zip(ref, hi)]
    cells = np.array(list(itertools.product(*axes)))
    if cells.size == 0:
        return 0.0
    dom = np.any(np.all(Y[None, :, :] >= cells[:, None, :], axis=2), axis=1)
    return float(dom.sum())


# --------------------------------------------------------------------------- pareto_mask


def test_pareto_mask_simple():
    Y = np.array([[1, 5], [2, 4], [3, 3], [2, 2], [0, 6], [3, 1]])
    np.testing.assert_array_equal(pareto_mask(Y), [True, True, True, False, True, False])


def test_pareto_mask_duplicates_kept_or_dropped_together():
    Y = np.array([[1.0, 2.0], [1.0, 2.0], [0.5, 0.5], [0.5, 0.5], [2.0, 1.0]])
    m = pareto_mask(Y)
    np.testing.assert_array_equal(m, [True, True, False, False, True])


def test_pareto_mask_weak_domination():
    # equal in one objective, better in the other => dominated
    Y = np.array([[1.0, 1.0], [1.0, 2.0]])
    np.testing.assert_array_equal(pareto_mask(Y), [False, True])


def test_pareto_mask_nonfinite_rows():
    Y = np.array([[np.nan, 10.0], [1.0, 1.0], [np.inf, np.inf], [0.0, 2.0]])
    np.testing.assert_array_equal(pareto_mask(Y), [False, True, False, True])


def test_pareto_mask_single_objective_and_empty():
    np.testing.assert_array_equal(pareto_mask(np.array([1.0, 3.0, 3.0, 2.0])), [False, True, True, False])
    assert pareto_mask(np.zeros((0, 2))).shape == (0,)


@pytest.mark.parametrize("m", [2, 3, 4])
def test_pareto_mask_matches_brute_force(m):
    rng = np.random.default_rng(m)
    Y = rng.integers(0, 6, size=(60, m)).astype(float)  # many ties and duplicates
    np.testing.assert_array_equal(pareto_mask(Y), brute_pareto(Y))


def test_pareto_mask_large_chunked():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(3000, 2))
    mask = pareto_mask(Y)
    # verify the front by a sort-based sweep
    order = np.lexsort((-Y[:, 1], -Y[:, 0]))
    best2 = -np.inf
    ref_mask = np.zeros(len(Y), bool)
    for i in order:
        if Y[i, 1] > best2:
            ref_mask[i] = True
            best2 = Y[i, 1]
    np.testing.assert_array_equal(mask, ref_mask)


# --------------------------------------------------------------------------- hypervolume


def test_hypervolume_2d_hand_computed():
    Y = np.array([[5.0, 3.0], [10.0, 1.0]])
    assert hypervolume(Y, [0.0, 0.0]) == pytest.approx(20.0)
    # dominated points, duplicates and points outside the reference box add nothing
    Y2 = np.vstack([Y, [[4.0, 2.0], [5.0, 3.0], [-1.0, 100.0], [100.0, 0.0]]])
    assert hypervolume(Y2, [0.0, 0.0]) == pytest.approx(20.0)
    assert hypervolume(Y, [5.0, 0.0]) == pytest.approx(5.0)
    assert hypervolume(Y, [20.0, 20.0]) == 0.0
    assert hypervolume(np.zeros((0, 2)), [0.0, 0.0]) == 0.0


def test_hypervolume_1d():
    assert hypervolume(np.array([[1.0], [3.0]]), [0.5]) == pytest.approx(2.5)


@pytest.mark.parametrize("seed", range(4))
def test_hypervolume_2d_vs_grid(seed):
    rng = np.random.default_rng(seed)
    Y = rng.integers(-3, 8, size=(15, 2))
    ref = np.array([-2, -1])
    assert hypervolume(Y.astype(float), ref) == pytest.approx(grid_hypervolume(np.maximum(Y, ref), ref))


@pytest.mark.parametrize("seed", range(4))
def test_hypervolume_3d_vs_grid(seed):
    rng = np.random.default_rng(10 + seed)
    Y = rng.integers(0, 9, size=(12, 3))
    ref = np.array([1, 0, 2])
    assert hypervolume(Y.astype(float), ref) == pytest.approx(grid_hypervolume(np.maximum(Y, ref), ref))


def test_hypervolume_3d_hand_computed():
    # two boxes [0,2]x[0,1]x[0,1] and [0,1]x[0,2]x[0,1] overlap in a unit cube; plus a tall thin one
    Y = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 3.0]])
    assert hypervolume(Y, [0, 0, 0]) == pytest.approx(2 + 2 - 1 + 2)


def test_hypervolume_mc_4d_seeded():
    rng = np.random.default_rng(3)
    Y = rng.integers(0, 4, size=(6, 4))
    ref = np.zeros(4, int)
    exact = grid_hypervolume(Y, ref)
    est = hypervolume(Y.astype(float), ref, n_samples=200_000, rng=1)
    assert est == pytest.approx(exact, rel=0.02)
    assert hypervolume(Y.astype(float), ref) == hypervolume(Y.astype(float), ref)  # deterministic default seed


def test_hypervolume_bad_ref():
    with pytest.raises(LenzError):
        hypervolume(np.ones((2, 2)), [0.0])
    with pytest.raises(LenzError):
        hypervolume(np.ones((2, 2)), [0.0, np.nan])


# --------------------------------------------------------------------------- hvi_2d


def test_hvi_2d_hand_computed():
    front = np.array([[5.0, 3.0], [10.0, 1.0]])
    S = np.array([[7.0, 4.0], [5.0, 3.0], [1.0, 1.0], [12.0, 0.5], [-1.0, 10.0], [11.0, 4.0]])
    expected = [11.0, 0.0, 0.0, 1.0, 0.0, 44.0 - 20.0]
    np.testing.assert_allclose(hvi_2d(S, front, [0.0, 0.0]), expected)


def test_hvi_2d_empty_front_is_box_volume():
    S = np.array([[2.0, 3.0], [-1.0, 3.0], [0.5, 0.5]])
    np.testing.assert_allclose(hvi_2d(S, np.zeros((0, 2)), [0.0, 0.0]), [6.0, 0.0, 0.25])


@pytest.mark.parametrize("seed", range(3))
def test_hvi_2d_matches_hypervolume_difference(seed):
    rng = np.random.default_rng(seed)
    front = rng.normal(size=(12, 2))   # includes dominated points
    ref = np.array([-2.0, -2.5])
    S = rng.normal(scale=1.5, size=(200, 2))
    hv0 = hypervolume(front, ref)
    brute = np.array([hypervolume(np.vstack([front, s]), ref) - hv0 for s in S])
    np.testing.assert_allclose(hvi_2d(S, front, ref), brute, atol=1e-12)
    assert np.all(hvi_2d(S, front, ref) >= 0)


def test_hvi_2d_bad_shapes():
    with pytest.raises(LenzError):
        hvi_2d(np.ones((3, 3)), np.ones((2, 2)), [0, 0])


# --------------------------------------------------------------------------- default_ref_point


def test_default_ref_point_formula():
    Y = np.array([[1.0, 10.0], [3.0, 30.0], [2.0, np.nan]])
    np.testing.assert_allclose(default_ref_point(Y), [1.0 - 0.2, 10.0 - 2.0])


def test_default_ref_point_degenerate():
    ref = default_ref_point(np.array([[5.0, -0.2]]))
    np.testing.assert_allclose(ref, [5.0 - 0.5, -0.2 - 0.1])
    assert hypervolume(np.array([[5.0, -0.2]]), ref) > 0
    with pytest.raises(LenzError):
        default_ref_point(np.zeros((0, 2)))
