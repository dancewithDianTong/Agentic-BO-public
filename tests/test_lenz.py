"""Tests for the lenz engine (frame data model, commands, envelopes, persistence)."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from agentic_bo.lenz import COMMANDS, Event, Frame, Lenz, Shelf, Trial
from agentic_bo.lenz.frame import normalize_constraints, normalize_model, normalize_objectives
from agentic_bo.lenz.pareto import hypervolume
from agentic_bo.space import SearchSpace
from agentic_bo.utils import LenzError, dumps

# --------------------------------------------------------------------------- helpers


def real_space(d: int = 2, lo: float = 0.0, hi: float = 1.0, prefix: str = "x") -> list[dict]:
    return [{"name": f"{prefix}{i}", "type": "real", "lower": lo, "upper": hi} for i in range(d)]


def quad(cfg: dict) -> float:
    """Concave 2-D quadratic, maximum 0 at (0.5, -0.3)."""
    return -((cfg["x0"] - 0.5) ** 2 + (cfg["x1"] + 0.3) ** 2)


def step(lz: Lenz, fn, **suggest_kw) -> tuple[dict, dict]:
    """One suggest -> submit -> observe cycle; returns (suggest result, observe result)."""
    r = lz.suggest(**suggest_kw)
    cfg = r["candidates"][0]["config"]
    tid = lz.submit(cfg, source=f"suggest:{r['acqf']}")["trial_id"]
    out = fn(cfg)
    metrics = out if isinstance(out, dict) else {lz.shelf.objectives[0]["metric"]: out}
    return r, lz.observe(metrics, trial_id=tid)


def seeded(lz: Lenz, fn, configs) -> None:
    for cfg in configs:
        out = fn(cfg)
        lz.observe(out if isinstance(out, dict) else {lz.shelf.objectives[0]["metric"]: out}, config=cfg)


def rms(a: dict, b: dict, space: SearchSpace) -> float:
    return space.distance(a, b)


def roundtrip(lz: Lenz) -> Lenz:
    return Lenz.from_dict(json.loads(dumps(lz.to_dict())))


@pytest.fixture
def quad_study() -> Lenz:
    lz = Lenz(real_space(2, -2, 2), "f", seed=1)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2, 2, size=(8, 2))
    seeded(lz, quad, [{"x0": float(a), "x1": float(b)} for a, b in pts])
    return lz


def branin_currin(cfg: dict) -> dict:
    a, b = cfg["x0"], cfg["x1"]
    x1, x2 = 15 * a - 5, 15 * b
    f1 = (x2 - 5.1 / (4 * math.pi ** 2) * x1 ** 2 + 5 / math.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * math.pi)) * math.cos(x1) + 10
    bb = max(b, 1e-9)
    f2 = (1 - math.exp(-1 / (2 * bb))) * (2300 * a ** 3 + 1900 * a ** 2 + 2092 * a + 60) \
        / (100 * a ** 3 + 500 * a ** 2 + 4 * a + 20)
    return {"f1": f1, "f2": f2}


MOO_OBJECTIVES = [{"metric": "f1", "minimize": True, "ref": 18.0}, {"metric": "f2", "minimize": True, "ref": 6.0}]


def hv_minimize_2d(F: np.ndarray, ref) -> float:
    """Brute-force 2-D hypervolume for minimization (independent of lenz.pareto)."""
    pts = [p for p in F if p[0] < ref[0] and p[1] < ref[1]]
    nd = [p for p in pts if not any((q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1]) for q in pts)]
    nd = sorted({(float(p[0]), float(p[1])) for p in nd})
    area = 0.0
    for i, (x, y) in enumerate(nd):
        x_next = nd[i + 1][0] if i + 1 < len(nd) else ref[0]
        area += (x_next - x) * (ref[1] - y)
    return area


# --------------------------------------------------------------------------- frame data model


def test_trial_event_shelf_frame_roundtrip():
    t = Trial(3, {"x": 0.5}, "completed", {"y": 1.0, "z": None}, "suggest:logei", 4, 6, 2, {"rationale": "r"})
    assert Trial.from_dict(json.loads(json.dumps(t.to_dict()))) == t
    assert t.value("y") == 1.0 and t.value("z") is None and t.value("missing") is None
    e = Event(5, "suggest", {"q": 2}, True, "ok")
    assert Event.from_dict(e.to_dict()) == e
    space = SearchSpace.from_spec([{"name": "x", "type": "real", "lower": 0, "upper": 1}])
    shelf = Shelf(normalize_objectives("y"), normalize_constraints({"c": [None, 1.0]}), "logei", {},
                  {"x": [0.0, 0.5]}, normalize_model(None))
    fr = Frame(space, shelf, [Trial(0, {"x": 0.25}, "pending")], [e], seed=7,
               model_state={"metric:y": {"key": "k", "hyperparameters": {"lengthscales": [0.3]}}})
    d = json.loads(json.dumps(fr.to_dict()))
    fr2 = Frame.from_dict(d)
    assert fr2.to_dict() == fr.to_dict()
    assert fr2.shelf.is_moo is False and fr2.shelf.modelled_metrics == ["y", "c"]
    assert fr2.trial(0).config == {"x": 0.25}
    with pytest.raises(LenzError):
        fr2.trial(5)
    with pytest.raises(LenzError):
        fr2.trial("a")
    with pytest.raises(LenzError):
        Trial.from_dict({"config": {}})
    with pytest.raises(LenzError):
        Trial.from_dict({"trial_id": 0, "config": {}, "status": "weird"})
    bad = dict(d)
    bad["trials"] = [{"trial_id": 0, "config": {"x": 9.0}, "status": "pending"}]
    with pytest.raises(LenzError):
        Frame.from_dict(bad)


def test_normalize_objectives():
    assert normalize_objectives("y") == [{"metric": "y", "minimize": False, "ref": None}]
    assert normalize_objectives({"metric": "y", "minimize": True}) == [{"metric": "y", "minimize": True, "ref": None}]
    assert normalize_objectives([{"name": "a", "direction": "min", "ref": 3}, "b"]) == [
        {"metric": "a", "minimize": True, "ref": 3.0}, {"metric": "b", "minimize": False, "ref": None}]
    for bad in (None, [], [""], ["a", "a"], {"metric": "a", "minimize": "yes"}, {"metric": "a", "foo": 1},
                {"metric": "a", "direction": "down"}, {"metric": "a", "ref": float("nan")}, 3):
        with pytest.raises(LenzError):
            normalize_objectives(bad)


def test_normalize_constraints_and_model():
    assert normalize_constraints(None) == [] and normalize_constraints([]) == []
    assert normalize_constraints({"c": [None, 1]}) == [{"metric": "c", "lower": None, "upper": 1.0}]
    assert normalize_constraints({"metric": "c", "lower": 0}) == [{"metric": "c", "lower": 0.0, "upper": None}]
    assert normalize_constraints({"c": {"lower": 0, "upper": 2}})[0]["upper"] == 2.0
    for bad in ({"metric": "c"}, {"metric": "c", "lower": 2, "upper": 1}, [{"metric": "c", "upper": 1}] * 2,
                {"metric": "c", "upper": 1, "op": "<="}, "c", {"c": [1, 2, 3]}):
        with pytest.raises(LenzError):
            normalize_constraints(bad)
    assert normalize_model(None) == {"kernel": "matern52", "noise": "learn", "n_restarts": 3, "prior": "botorch"}
    assert normalize_model({"prior": "Median"})["prior"] == "median"
    assert normalize_model({"kernel": "RBF", "noise": 1e-4})["kernel"] == "rbf"
    for bad in ({"kernel": "linear"}, {"noise": -1}, {"noise": "fixed"}, {"n_restarts": 0}, {"lr": 1}, "x", {"prior": "flat"}):
        with pytest.raises(LenzError):
            normalize_model(bad)


# --------------------------------------------------------------------------- construction


def test_construction_defaults_and_validation():
    sp = real_space(2)
    assert Lenz(sp, "y").shelf.acqf == "logei"
    assert Lenz(sp, ["a", "b"]).shelf.acqf == "ehvi"
    assert Lenz(sp, ["a", "b", "c"]).shelf.acqf == "parego"
    lz = Lenz(sp, "y", acqf="UCB", acqf_params={"beta": 9})
    assert lz.shelf.acqf == "ucb" and lz.shelf.acqf_params == {"beta": 9.0}
    assert lz.frame.events[0].command == "create" and lz.frame.events[0].seq == 1
    for kwargs in ({"acqf": "ehvi"}, {"acqf": "nope"}, {"acqf": "logei", "acqf_params": {"beta": 2}},
                   {"bounds": {"x0": [0.5, 2.0]}}, {"seed": 1.5}, {"model": {"kernel": "x"}},
                   {"constraints": [{"metric": "c"}]}):
        with pytest.raises(LenzError):
            Lenz(sp, "y", **kwargs)
    with pytest.raises(LenzError):
        Lenz(sp, ["a", "b", "c"], acqf="ehvi")
    with pytest.raises(LenzError):
        Lenz(sp, ["a", "b"], acqf="logei")
    assert Lenz.create(sp, "y").n_observed == 0
    assert set(COMMANDS) >= {"suggest", "submit", "observe", "status", "pareto"}


# --------------------------------------------------------------------------- lifecycle


def test_lifecycle_quadratic_improves():
    lz = Lenz(real_space(2, -2, 2), "f", seed=1)
    modes, best = [], []
    for _ in range(16):
        r, o = step(lz, quad)
        modes.append(r["mode"])
        best.append(lz.incumbent()["incumbent"]["value"])
    assert modes[:2] == ["space_filling"] * 2 and set(modes[2:]) == {"model"}
    assert all(b2 >= b1 for b1, b2 in zip(best, best[1:]))
    assert best[-1] > best[2] and best[-1] > -0.02
    tr = lz.trials()["trials"]
    assert [t["eval_index"] for t in tr] == list(range(1, 17))
    assert all(t["status"] == "completed" and t["source"].startswith("suggest:") for t in tr)
    assert lz.n_observed == 16 and lz.n_pending == 0
    inc = lz.incumbent()
    assert inc["feasible_found"] and inc["best_posterior"]["mean"] <= 0.05
    assert inc["incumbent"]["value"] == max(t["metrics"]["f"] for t in tr)


def test_observe_new_incumbent_flags(quad_study):
    lz = quad_study
    inc = lz.incumbent()["incumbent"]
    worse = lz.observe({"f": inc["value"] - 1.0}, config={"x0": -2.0, "x1": 2.0})
    assert not worse["is_new_incumbent"] and worse["incumbent"]["trial_id"] == inc["trial_id"]
    better = lz.observe({"f": inc["value"] + 1.0}, config={"x0": 0.5, "x1": -0.3})
    assert better["is_new_incumbent"] and better["incumbent"]["trial_id"] == better["trial_id"]
    assert better["new_trial"] and lz.frame.trial(better["trial_id"]).source == "observe"


def test_minimize_sign_correctness():
    def g(cfg):  # minimum 1.0 at (0.2, 0.7)
        return 1.0 + 3 * (cfg["x0"] - 0.2) ** 2 + 3 * (cfg["x1"] - 0.7) ** 2

    lz = Lenz(real_space(2), {"metric": "g", "minimize": True}, seed=4)
    for _ in range(14):
        step(lz, g)
    tr = lz.trials()["trials"]
    observed = [t["metrics"]["g"] for t in tr]
    inc = lz.incumbent()
    assert inc["incumbent"]["value"] == min(observed) and inc["incumbent"]["value"] < 1.05
    assert inc["best_posterior"]["mean"] < 1.1
    pred = lz.predict([t["config"] for t in tr])["predictions"]
    means = np.array([p["metrics"]["g"]["mean"] for p in pred])
    assert np.max(np.abs(means - np.array(observed))) < 0.05 * (max(observed) - min(observed))
    cand = lz.suggest()["candidates"][0]
    assert cand["predicted"]["g"]["mean"] > 0.9  # raw units and raw direction
    srt = lz.trials(sort_by="value")["trials"]
    assert [t["metrics"]["g"] for t in srt] == sorted(observed)


def test_maximize_and_minimize_are_mirror_images():
    def h(cfg):
        return (cfg["x0"] - 0.2) ** 2 + (cfg["x1"] - 0.7) ** 2

    A = Lenz(real_space(2), {"metric": "g", "minimize": True}, seed=3)
    B = Lenz(real_space(2), "h", seed=3)
    for _ in range(7):
        ca = A.suggest()["candidates"][0]
        cb = B.suggest()["candidates"][0]
        assert ca["config"] == cb["config"]
        if A.n_observed:
            assert ca["predicted"]["g"]["mean"] == pytest.approx(-cb["predicted"]["h"]["mean"], abs=1e-9)
        A.observe({"g": h(ca["config"])}, config=ca["config"])
        B.observe({"h": -h(cb["config"])}, config=cb["config"])


def test_pending_trials_are_fantasized(quad_study):
    lz = quad_study
    first = lz.suggest()["candidates"][0]["config"]
    tid = lz.submit(first)["trial_id"]
    r = lz.suggest()
    second = r["candidates"][0]["config"]
    assert r["n_pending"] == 1 and any("pending" in n for n in r["notes"])
    assert rms(first, second, lz.space) > 1e-3
    assert r["candidates"][0]["duplicate_of"] is None
    lz.abandon(tid)
    assert lz.n_pending == 0 and lz.frame.trial(tid).status == "abandoned"
    with pytest.raises(LenzError):
        lz.abandon(tid)
    with pytest.raises(LenzError):
        lz.observe({"f": 0.0}, trial_id=tid)


def test_q_batch_is_diverse(quad_study):
    lz = quad_study
    for acqf in ("logei", "ucb", "ts", "pm"):
        r = lz.suggest(q=4, acqf=acqf)
        cfgs = [c["config"] for c in r["candidates"]]
        assert len(cfgs) == 4 and r["mode"] == "model"
        d = [rms(a, b, lz.space) for i, a in enumerate(cfgs) for b in cfgs[i + 1:]]
        assert min(d) > 1e-4, (acqf, d)
        for c in r["candidates"]:
            assert c["trial_id"] is None and c["acqf"] == acqf and acqf in c["acquisition_values"]
    assert lz.n_observed == 8 and lz.n_pending == 0  # suggest records nothing


def test_suggest_records_nothing_and_override_keeps_shelf(quad_study):
    lz = quad_study
    before = [t.to_dict() for t in lz.frame.trials]
    r = lz.suggest(acqf="ucb", acqf_params={"beta": 1.0})
    assert r["acqf"] == "ucb" and r["acqf_params"] == {"beta": 1.0}
    assert lz.shelf.acqf == "logei" and lz.shelf.acqf_params == {}
    assert [t.to_dict() for t in lz.frame.trials] == before
    with pytest.raises(LenzError):
        lz.suggest(acqf="logei", acqf_params={"beta": 2})
    for bad in ({"q": 0}, {"q": 1.5}, {"q": True}, {"radius": 0}, {"radius": -1}, {"acqf": "ehvi"},
                {"around": "best"}, {"around": {"x0": 9, "x1": 0}}, {"bounds": {"x0": [3, 4]}}):
        with pytest.raises(LenzError):
            lz.suggest(**bad)


# --------------------------------------------------------------------------- regions


def test_trust_region_around_incumbent_and_config(quad_study):
    lz = quad_study
    inc = lz.incumbent()["incumbent"]["config"]
    r = lz.suggest(q=2, around="incumbent", radius=0.05)
    box = r["region"]["box"]
    assert r["region"]["around"] == inc and r["region"]["radius"] == 0.05
    for name in ("x0", "x1"):
        lo, hi = box[name]
        assert lo <= inc[name] <= hi
        assert hi - lo <= 2 * 0.05 * 4 * 4 + 1e-9  # encoded half-width <= 4 * radius, span 4
    for c in r["candidates"]:
        for name in ("x0", "x1"):
            assert box[name][0] - 1e-9 <= c["config"][name] <= box[name][1] + 1e-9
    center = {"x0": -1.5, "x1": 1.5}
    r2 = lz.suggest(around=center, radius=0.02)
    c = r2["candidates"][0]["config"]
    assert r2["region"]["around"] == center
    assert abs(c["x0"] + 1.5) <= 0.02 * 4 * 4 + 1e-9 and abs(c["x1"] - 1.5) <= 0.02 * 4 * 4 + 1e-9


def test_bounds_override_and_set_bounds(quad_study):
    lz = quad_study
    r = lz.suggest(q=3, bounds={"x0": [0.0, 0.5]})
    assert all(0.0 <= c["config"]["x0"] <= 0.5 for c in r["candidates"])
    assert r["region"]["bounds"] == {"x0": [0.0, 0.5]} and lz.shelf.bounds is None
    res = lz.set_bounds({"x0": [-2.0, 0.0], "x1": [1.0, 2.0]})
    assert res["observed"] == 8 and res["n_in_region"] == sum(
        1 for t in lz.frame.trials if t.config["x0"] <= 0 and t.config["x1"] >= 1)
    for c in lz.suggest(q=2)["candidates"]:
        assert c["config"]["x0"] <= 0.0 and c["config"]["x1"] >= 1.0
    # the bounds argument overrides the active region; "full" means the whole domain for this call
    assert lz.suggest(bounds="full")["region"]["bounds"] is None
    assert lz.set_bounds("full")["bounds"] is None and lz.shelf.bounds is None
    with pytest.raises(LenzError):
        lz.set_bounds({"zz": [0, 1]})


def test_space_filling_respects_region_and_never_duplicates():
    space = [{"name": "k", "type": "categorical", "choices": ["a", "b", "c"]},
             {"name": "n", "type": "int", "lower": 1, "upper": 2}]
    lz = Lenz(space, "y", acqf="sobol", seed=0)
    seen = []
    for _ in range(6):
        r = lz.suggest()
        cfg = r["candidates"][0]["config"]
        assert r["mode"] == "space_filling" and r["candidates"][0]["duplicate_of"] is None
        assert cfg not in seen
        seen.append(cfg)
        lz.observe({"y": 1.0}, config=cfg)
    r = lz.suggest()  # all 6 configs observed: a duplicate is returned with a note
    assert r["candidates"][0]["duplicate_of"] is not None and any("repeat" in n for n in r["notes"])
    r = lz.suggest(q=3, bounds={"k": ["b"], "n": [2, 2]})
    assert all(c["config"] == {"k": "b", "n": 2} for c in r["candidates"])


def test_sobol_sequence_continues_across_calls():
    lz = Lenz(real_space(3), "y", seed=11)
    a = [c["config"] for c in lz.suggest(q=2)["candidates"]]
    b = [c["config"] for c in lz.suggest(q=3, acqf="sobol")["candidates"]]
    fresh = Lenz(real_space(3), "y", seed=11)
    together = [c["config"] for c in fresh.suggest(q=5, acqf="sobol")["candidates"]]
    assert a + b == together
    assert lz.frame.counters["sobol"] == 5
    other = Lenz(real_space(3), "y", seed=12)
    assert other.suggest()["candidates"][0]["config"] != a[0]
    r = lz.suggest(acqf="random", q=2)
    assert r["acqf"] == "random" and r["candidates"][0]["predicted"] is None
    assert lz.frame.counters["sobol"] == 5


# --------------------------------------------------------------------------- reconfiguration


def test_reconfiguration_never_drops_data(quad_study):
    lz = quad_study
    lz.submit({"x0": 0.0, "x1": 0.0})
    before = [t.to_dict() for t in lz.frame.trials]
    lz.set_bounds({"x0": [0, 1]})
    lz.set_acqf("ucb", beta=2.0)
    lz.set_objectives([{"metric": "f"}, {"metric": "g", "minimize": True}])
    lz.set_constraints([{"metric": "c", "upper": 1.0}])
    lz.set_constraints(None)
    lz.set_objectives("f")
    lz.set_bounds(None)
    assert [t.to_dict() for t in lz.frame.trials] == before
    assert lz.n_observed == 8 and lz.n_pending == 1
    st = lz.status()
    assert st["n_observed"] == 8 and st["n_pending"] == 1 and st["pending_trial_ids"] == [8]


def test_set_acqf_validation_and_params():
    lz = Lenz(real_space(2), "y")
    r = lz.set_acqf("ucb", beta=9.0)
    assert r == {"acqf": "ucb", "acqf_params": {"beta": 9.0}, "is_moo": False, "observed": 0, "notes": []}
    assert lz.set_acqf("ts", params={"n_candidates": 256})["acqf_params"] == {"n_candidates": 256}
    assert lz.set_acqf("logei")["acqf_params"] == {}
    for args, kw in ((("logei",), {"beta": 2.0}), (("bogus",), {}), (("ehvi",), {}),
                     (("ucb",), {"beta": -1}), (("ucb",), {"gamma": 1}), (("ts",), {"n_candidates": 2})):
        with pytest.raises(LenzError):
            lz.set_acqf(*args, **kw)
    m = Lenz(real_space(2), ["a", "b"])
    assert m.set_acqf("parego", weights=[3, 1])["acqf_params"] == {"weights": [0.75, 0.25]}
    with pytest.raises(LenzError):
        m.set_acqf("parego", weights=[1, 1, 1])
    with pytest.raises(LenzError):
        m.set_acqf("ehvi", ref_point=[1.0])


def test_set_objectives_switches_acqf_family():
    lz = Lenz(real_space(2), "y", acqf="ucb", acqf_params={"beta": 2})
    r = lz.set_objectives(["a", {"metric": "b", "minimize": True}])
    assert r["acqf"] == "ehvi" and r["acqf_switched"] and r["previous_acqf"] == "ucb" and r["is_moo"]
    assert r["objectives"][1] == {"metric": "b", "minimize": True, "ref": None} and r["acqf_params"] == {}
    assert lz.set_objectives(["a", "b", "c"])["acqf"] == "parego"
    r = lz.set_objectives("a")
    assert r["acqf"] == "logei" and not r["is_moo"] and r["notes"]
    lz.set_acqf("sobol")
    assert lz.set_objectives(["a", "b"])["acqf"] == "sobol"
    lz.set_acqf("ehvi", ref_point=[1.0, 2.0])
    r = lz.set_objectives(["a", "b", "c"])
    assert r["acqf"] == "parego" and r["acqf_params"] == {}
    lz.set_acqf("parego", weights=[1, 1, 2])
    r = lz.set_objectives(["a", "b", "d"])  # same family, same arity: params kept
    assert r["acqf_params"] == {"weights": [0.25, 0.25, 0.5]} and not r["acqf_switched"]


# --------------------------------------------------------------------------- constraints


def _cons_fn(cfg):
    a, b = cfg["x0"], cfg["x1"]
    return {"y": a + b, "c": a * a + b * b}


def test_constrained_feasibility_phase_then_constrained_logei():
    lz = Lenz(real_space(2), "y", constraints=[{"metric": "c", "upper": 0.3}], seed=2)
    infeasible = [{"x0": 0.9, "x1": 0.9}, {"x0": 0.8, "x1": 0.6}, {"x0": 0.6, "x1": 0.9}, {"x0": 1.0, "x1": 0.5}]
    seeded(lz, _cons_fn, infeasible)
    inc = lz.incumbent()
    assert inc["incumbent"] is None and not inc["feasible_found"] and inc["least_infeasible"]["trial_id"] == 1
    assert lz.status()["feasible_found"] is False
    assert all(t["feasible"] is False for t in lz.trials()["trials"])
    r = lz.suggest()
    cand = r["candidates"][0]
    assert r["feasibility_phase"] and any("feasibility" in n for n in r["notes"])
    # PoF maximization heads for the unexplored low-c corner (the GP mean reverts there, so
    # judge by the true constraint value rather than by the predicted mean)
    assert _cons_fn(cand["config"])["c"] < min(_cons_fn(c)["c"] for c in infeasible)
    pof_obs = [p["prob_feasible"] for p in lz.predict(infeasible)["predictions"]]
    assert cand["prob_feasible"] > max(pof_obs)
    # follow the suggestions until a feasible point is found
    for _ in range(6):
        if lz.incumbent()["feasible_found"]:
            break
        step(lz, _cons_fn)
    assert lz.incumbent()["feasible_found"]
    for _ in range(5):
        r, _ = step(lz, _cons_fn)
    assert not r["feasibility_phase"] and r["acqf"] == "logei"
    inc = lz.incumbent()["incumbent"]
    assert inc["metrics"]["c"] <= 0.3 and inc["value"] > 0.5  # optimum: a = b = sqrt(0.15), y = 0.775
    feas = [t for t in lz.trials()["trials"] if t["feasible"]]
    assert inc["value"] == max(t["metrics"]["y"] for t in feas)
    for acqf in ("ucb", "ts", "noisy_logei", "ei", "pi", "logpi"):
        r = lz.suggest(acqf=acqf)
        assert 0 <= r["candidates"][0]["prob_feasible"] <= 1


def test_observe_requires_constraint_metrics():
    lz = Lenz(real_space(2), "y", constraints={"c": [None, 1.0]})
    with pytest.raises(LenzError, match="missing"):
        lz.observe({"y": 1.0}, config={"x0": 0.1, "x1": 0.1})
    lz.observe({"y": 1.0, "c": None, "extra": 3}, config={"x0": 0.1, "x1": 0.1})
    t = lz.frame.trial(0)
    assert t.metrics == {"c": None, "extra": 3.0, "y": 1.0}
    assert lz.trials()["trials"][0]["feasible"] is False


# --------------------------------------------------------------------------- multi-objective


def test_moo_ehvi_parego_and_pareto():
    lz = Lenz(real_space(2), MOO_OBJECTIVES, seed=0)
    assert lz.shelf.acqf == "ehvi"
    for _ in range(6):
        step(lz, branin_currin)
    for _ in range(3):
        r, o = step(lz, branin_currin)
        assert r["mode"] == "model" and r["acqf"] == "ehvi"
        assert set(r["candidates"][0]["predicted"]) == {"f1", "f2"}
    tr = lz.trials()["trials"]
    F = np.array([[t["metrics"]["f1"], t["metrics"]["f2"]] for t in tr])
    par = lz.pareto()
    assert par["ref_point"] == [18.0, 6.0] and par["is_moo"]
    front_ids = {p["trial_id"] for p in par["pareto"]}
    expected = {i for i in range(len(F)) if not any(
        np.all(F[j] <= F[i]) and np.any(F[j] < F[i]) for j in range(len(F)))}
    assert front_ids == expected
    assert par["hypervolume"] == pytest.approx(hv_minimize_2d(F, (18.0, 6.0)), rel=1e-12)
    inc = lz.incumbent()
    assert inc["is_moo"] and inc["incumbent"] is None and {p["trial_id"] for p in inc["pareto"]} == front_ids
    st = lz.status()["incumbent"]
    assert set(st["pareto_trial_ids"]) == front_ids and st["hypervolume"] == pytest.approx(par["hypervolume"])
    # parego: one Dirichlet weight vector per candidate, drawn from the per-call stream
    lz.set_acqf("parego")
    r = lz.suggest(q=2)
    w = [c["weights"] for c in r["candidates"]]
    assert all(abs(sum(x) - 1) < 1e-12 for x in w) and w[0] != w[1]
    fixed = lz.suggest(acqf="parego", acqf_params={"weights": [1, 0]})
    assert fixed["candidates"][0]["weights"] == [1.0, 0.0]
    before_hv = par["hypervolume"]
    _, o = step(lz, branin_currin)
    after = lz.pareto()
    assert o["is_new_incumbent"] == (o["trial_id"] in {p["trial_id"] for p in after["pareto"]})
    assert after["hypervolume"] >= before_hv - 1e-12
    for acqf in ("logei", "ts", "ucb"):
        with pytest.raises(LenzError):
            lz.suggest(acqf=acqf)
    sc = lz.score([t["config"] for t in tr[:3]], acqfs=["ehvi", "parego"])
    assert sc["acqfs"] == ["ehvi", "parego"] and all(len(s["acquisition_values"]) == 2 for s in sc["scores"])


def test_moo_default_ref_point_and_constraints():
    objs = [{"metric": "f1", "minimize": True}, {"metric": "f2", "minimize": True}]
    lz = Lenz(real_space(2), objs, constraints=[{"metric": "c", "upper": 0.5}], seed=1)

    def fn(cfg):
        return {**branin_currin(cfg), "c": cfg["x0"]}

    seeded(lz, fn, [{"x0": 0.9, "x1": 0.2}, {"x0": 0.7, "x1": 0.8}, {"x0": 0.8, "x1": 0.5}])
    r = lz.suggest()
    assert r["feasibility_phase"] and r["acqf"] == "ehvi"
    assert lz.pareto()["pareto"] == [] and lz.pareto()["hypervolume"] == 0.0
    seeded(lz, fn, [{"x0": 0.2, "x1": 0.3}, {"x0": 0.4, "x1": 0.1}])
    par = lz.pareto()
    tr = lz.trials()["trials"]
    F = np.array([[t["metrics"]["f1"], t["metrics"]["f2"]] for t in tr])
    ref = F.max(axis=0) + 0.1 * (F.max(axis=0) - F.min(axis=0))  # default ref over all observations
    assert par["ref_point"] == pytest.approx(ref.tolist())
    feas = F[[t["feasible"] for t in tr]]
    assert par["hypervolume"] == pytest.approx(hv_minimize_2d(feas, ref))
    assert not lz.suggest()["feasibility_phase"]
    lz3 = Lenz(real_space(2), ["a", "b", "c"], seed=0)
    seeded(lz3, lambda c: {"a": c["x0"], "b": c["x1"], "c": -c["x0"] - c["x1"]},
           [{"x0": 0.1, "x1": 0.9}, {"x0": 0.5, "x1": 0.5}, {"x0": 0.9, "x1": 0.2}])
    r = lz3.suggest()
    assert r["acqf"] == "parego" and len(r["candidates"][0]["weights"]) == 3
    assert lz3.pareto()["hypervolume"] > 0


# --------------------------------------------------------------------------- probes


def test_diagnostics_fields():
    space = [{"name": "a", "type": "real", "lower": 0, "upper": 1},
             {"name": "k", "type": "categorical", "choices": ["p", "q", "r"]}]
    lz = Lenz(space, "y", constraints=[{"metric": "c", "upper": 2.0}], seed=0)
    rng = np.random.default_rng(1)
    for i in range(10):
        cfg = {"a": float(rng.uniform()), "k": ["p", "q", "r"][i % 3]}
        lz.observe({"y": cfg["a"] ** 2 + {"p": 0, "q": 1, "r": 2}[cfg["k"]], "c": cfg["a"], "cost": 1.0 + i},
                   config=cfg)
    d = lz.diagnostics()
    assert d["objective"] == "y" and set(d["models"]) == {"y", "c"}
    for name, m in d["models"].items():
        for key in ("n_observed", "cv_r2", "noise_std", "outputscale", "lengthscales", "sensitivity",
                    "log_marginal_likelihood"):
            assert key in m, (name, key)
        assert m["n_observed"] == 10
        assert set(m["lengthscales"]) == {"a", "k"} and set(m["sensitivity"]) == {"a", "k"}
        assert sum(m["sensitivity"].values()) == pytest.approx(1.0)
        assert m["noise_std"] >= 0 and m["outputscale"] > 0
    assert d["models"]["y"]["role"] == "objective" and d["models"]["c"]["role"] == "constraint"
    assert d["models"]["y"]["cv_r2"] > 0.5
    single = lz.diagnostics("cost")
    assert set(single["models"]) == {"cost"} and single["models"]["cost"]["role"] == "reported"
    with pytest.raises(LenzError):
        lz.diagnostics("nope")
    empty = Lenz(real_space(2), "y").diagnostics()
    assert empty["models"]["y"]["n_observed"] == 0 and empty["models"]["y"]["cv_r2"] is None


def test_predict_and_score_shapes(quad_study):
    lz = quad_study
    cfgs = [{"x0": 0.5, "x1": -0.3}, {"x0": -2.0, "x1": 2.0}, {"x0": 1.0, "x1": 1.0}]
    lz.set_bounds({"x0": [0.0, 2.0]})
    p = lz.predict(cfgs)
    assert len(p["predictions"]) == 3 and p["acqf"] == "logei"
    for row, cfg in zip(p["predictions"], cfgs):
        assert row["config"] == cfg and row["prob_feasible"] == 1.0
        m = row["metrics"]["f"]
        assert m["std"] == pytest.approx(math.sqrt(m["variance"]))
        assert isinstance(row["acquisition_values"]["logei"], float)
    assert [r["in_region"] for r in p["predictions"]] == [True, False, True]
    assert p["predictions"][0]["metrics"]["f"]["mean"] > p["predictions"][1]["metrics"]["f"]["mean"]
    s = lz.score(cfgs, acqfs=["logei", "ucb", "pm", "std"])
    assert s["acqfs"] == ["logei", "ucb", "pm", "std"] and len(s["scores"]) == 3
    pm = [row["acquisition_values"]["pm"] for row in s["scores"]]
    assert s["best"]["pm"] == int(np.argmax(pm))
    assert pm == pytest.approx([row["metrics"]["f"]["mean"] for row in p["predictions"]])
    assert lz.score(cfgs[0])["acqfs"] == ["logei"]
    assert lz.score(cfgs, acqfs="ucb,ei")["acqfs"] == ["ucb", "ei"]
    for bad in ({"acqfs": ["ts"]}, {"acqfs": ["sobol"]}, {"acqfs": ["ehvi"]}, {"acqfs": []}):
        with pytest.raises(LenzError):
            lz.score(cfgs, **bad)
    for bad in ([], [{"x0": 5, "x1": 0}], [{"x0": 0}], "x"):
        with pytest.raises(LenzError):
            lz.predict(bad)


def test_score_matches_suggest_acquisition_value(quad_study):
    lz = quad_study
    lz.submit({"x0": 0.0, "x1": 0.0})  # pending trials are fantasized in both
    cand = lz.suggest()["candidates"][0]
    sc = lz.score([cand["config"]])["scores"][0]["acquisition_values"]["logei"]
    assert sc == pytest.approx(cand["acquisition_values"]["logei"], rel=1e-9, abs=1e-9)
    lz.set_acqf("ucb", beta=4.0)
    cand = lz.suggest()["candidates"][0]
    pr = lz.predict([cand["config"]])["predictions"][0]
    assert pr["acquisition_values"]["ucb"] == pytest.approx(cand["acquisition_values"]["ucb"], rel=1e-9)


def test_probes_without_data():
    lz = Lenz(real_space(2), "y", constraints=[{"metric": "c", "upper": 1}])
    p = lz.predict([{"x0": 0.1, "x1": 0.2}])
    assert p["predictions"][0]["metrics"]["y"]["mean"] is None and "note" in p
    assert p["predictions"][0]["prob_feasible"] is None
    s = lz.score([{"x0": 0.1, "x1": 0.2}])
    assert s["scores"][0]["acquisition_values"]["logei"] is None and s["best"]["logei"] is None and s["notes"]
    inc = lz.incumbent()
    assert inc["incumbent"] is None and inc["best_posterior"] is None and not inc["feasible_found"]
    assert lz.pareto() == {"pareto": [], "hypervolume": 0.0, "ref_point": None, "n_feasible": 0, "is_moo": False}
    r = lz.suggest(q=2)
    assert r["mode"] == "space_filling" and r["candidates"][0]["predicted"]["y"]["mean"] is None
    with pytest.raises(LenzError):
        lz.suggest(around="incumbent")
    st = lz.status()
    assert st["n_observed"] == 0 and st["incumbent"] is None and st["space"] == lz.space.to_spec()


def test_trials_filters_and_sorting(quad_study):
    lz = quad_study
    lz.submit({"x0": 0.0, "x1": 0.0})
    tid = lz.submit({"x0": 0.1, "x1": 0.1})["trial_id"]
    lz.abandon(tid)
    lz.observe({"f": float("nan")}, config={"x0": 1.9, "x1": 1.9})
    assert lz.trials()["n"] == 11 and lz.trials()["n_total"] == 11
    assert [t["trial_id"] for t in lz.trials(status="pending")["trials"]] == [8]
    assert [t["trial_id"] for t in lz.trials(status=["abandoned", "pending"])["trials"]] == [8, 9]
    assert lz.trials(status="all")["n"] == 11
    assert [t["trial_id"] for t in lz.trials(last=2)["trials"]] == [9, 10]
    assert lz.trials(last=0)["n"] == 0
    by_val = lz.trials(sort_by="value")["trials"]
    vals = [t["value"] for t in by_val if t["value"] is not None]
    assert vals == sorted(vals, reverse=True) and by_val[0]["trial_id"] == lz.incumbent()["incumbent"]["trial_id"]
    assert by_val[-1]["value"] is None
    by_eval = lz.trials(sort_by="eval_index")["trials"]
    assert [t["eval_index"] for t in by_eval[:9]] == list(range(1, 10))
    nan_trial = lz.frame.trial(10)
    assert nan_trial.metrics == {"f": None} and lz.diagnostics()["models"]["f"]["n_observed"] == 8
    for bad in ({"status": "done"}, {"sort_by": "speed"}, {"last": -1}):
        with pytest.raises(LenzError):
            lz.trials(**bad)


def test_observe_by_config_and_validation(quad_study):
    lz = quad_study
    cfg = {"x0": 0.25, "x1": 0.75}
    t1 = lz.submit(cfg)["trial_id"]
    t2 = lz.submit(cfg)["trial_id"]
    assert lz.submit(cfg)["duplicate_of"] == t1
    r = lz.observe({"f": quad(cfg)}, config=cfg)
    assert r["trial_id"] == t1 and not r["new_trial"]  # the earliest matching pending trial
    with pytest.raises(LenzError, match="does not match"):
        lz.observe({"f": 1.0}, trial_id=t2, config={"x0": 0.0, "x1": 0.0})
    with pytest.raises(LenzError, match="not pending"):
        lz.observe({"f": 1.0}, trial_id=t1)
    for bad in ({"metrics": {}}, {"metrics": {"f": "high"}}, {"metrics": {"f": True}},
                {"metrics": {"g": 1.0}, "trial_id": t2}, {"metrics": {"f": 1.0}}):
        with pytest.raises(LenzError):
            lz.observe(**bad)
    with pytest.raises(LenzError):
        lz.submit({"x0": 3.0, "x1": 0.0})
    with pytest.raises(LenzError):
        lz.submit({"x0": 0.0})
    with pytest.raises(LenzError):
        lz.submit({"x0": 0.0, "x1": 0.0}, meta="not a dict")
    sub = lz.submit({"x0": 0.0, "x1": 0.0}, source="harness-fallback", meta={"rationale": "test"})
    assert lz.frame.trial(sub["trial_id"]).meta == {"rationale": "test"}


# --------------------------------------------------------------------------- mixed spaces


MIXED = [
    {"name": "lr", "type": "real", "lower": 1e-4, "upper": 1.0, "log_scale": True},
    {"name": "layers", "type": "int", "lower": 1, "upper": 8},
    {"name": "width", "type": "int", "lower": 8, "upper": 512, "log_scale": True},
    {"name": "drop", "type": "ordinal", "values": [0.0, 0.1, 0.2, 0.5]},
    {"name": "opt", "type": "categorical", "choices": ["adam", "sgd", "rmsprop"]},
]


def mixed_fn(c: dict) -> float:
    return (-(math.log10(c["lr"]) + 2.5) ** 2 - 0.1 * (c["layers"] - 4) ** 2
            - 0.05 * (math.log2(c["width"]) - 6) ** 2 - 2 * c["drop"]
            + {"adam": 0.0, "sgd": -1.0, "rmsprop": -0.5}[c["opt"]])


def test_mixed_space_end_to_end():
    lz = Lenz(MIXED, "acc", seed=5)
    space = lz.space
    for _ in range(14):
        r, _ = step(lz, mixed_fn)
        cfg = r["candidates"][0]["config"]
        assert space.validate(cfg) == cfg
        assert isinstance(cfg["layers"], int) and isinstance(cfg["width"], int)
        assert cfg["drop"] in (0.0, 0.1, 0.2, 0.5) and cfg["opt"] in ("adam", "sgd", "rmsprop")
    inc = lz.incumbent()["incumbent"]
    first = [t["metrics"]["acc"] for t in lz.trials()["trials"][:2]]
    assert inc["value"] >= max(first)
    d = lz.diagnostics()["models"]["acc"]
    assert set(d["lengthscales"]) == {"lr", "layers", "width", "drop", "opt"}
    assert sum(d["sensitivity"].values()) == pytest.approx(1.0)
    r = lz.suggest(q=2, bounds={"opt": ["sgd", "rmsprop"], "drop": [0.0, 0.5], "layers": [2, 3]})
    for c in r["candidates"]:
        assert c["config"]["opt"] in ("sgd", "rmsprop") and c["config"]["drop"] in (0.0, 0.5)
        assert c["config"]["layers"] in (2, 3)
    r = lz.suggest(around="incumbent", radius=0.1)
    assert r["candidates"][0]["config"]["opt"] == inc["config"]["opt"]
    assert r["region"]["box"]["opt"] == [inc["config"]["opt"]]
    lz2 = roundtrip(lz)
    assert lz2.suggest(q=2) == lz.suggest(q=2)


# --------------------------------------------------------------------------- persistence


def test_save_load_reproduces_suggestions(tmp_path, quad_study):
    lz = quad_study
    lz.suggest()                        # warms the in-memory GP cache of the original only
    lz.submit({"x0": 0.0, "x1": 0.0})
    path = tmp_path / "study.json"
    lz.save(path)
    loaded = Lenz.load(path)
    assert loaded.suggest(q=3) == lz.suggest(q=3)
    # continue both after an observation: the persisted warm-start chain keeps them in lockstep
    for z in (lz, loaded):
        z.observe({"f": -0.25}, trial_id=8)
    assert loaded.suggest(around="incumbent") == lz.suggest(around="incumbent")
    assert dumps(loaded.to_dict()) == dumps(lz.to_dict())


@pytest.mark.parametrize("acqf", ["parego", "ehvi"])
def test_save_load_reproduces_moo(acqf):
    lz = Lenz(real_space(2), MOO_OBJECTIVES, acqf=acqf, seed=5)
    for _ in range(5):
        step(lz, branin_currin)
    lz.submit(lz.suggest()["candidates"][0]["config"])
    other = roundtrip(lz)
    assert other.suggest(q=2) == lz.suggest(q=2)


def test_save_load_reproduces_ts_and_space_filling(tmp_path):
    lz = Lenz(real_space(3), "y", acqf="ts", seed=2)
    a = lz.suggest(q=2)
    b = roundtrip(lz).suggest(q=2)
    assert a != b  # the round trip happened after a; the next call continues the stream
    assert roundtrip(lz).suggest(q=2) == lz.suggest(q=2)
    for _ in range(4):
        step(lz, lambda c: -sum((v - 0.3) ** 2 for v in c.values()))
    assert roundtrip(lz).suggest(q=2) == lz.suggest(q=2)


def test_to_dict_is_deterministic_and_validated(tmp_path, quad_study):
    lz = quad_study
    s1 = dumps(lz.to_dict())
    lz2 = roundtrip(lz)
    assert dumps(lz2.to_dict()) == s1
    json.loads(s1)  # strict JSON (no NaN)
    with pytest.raises(LenzError):
        Lenz.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(LenzError):
        Lenz.load(bad)
    with pytest.raises(LenzError):
        Lenz.from_dict({"format": "something-else", "frame": {}})
    with pytest.raises(LenzError):
        Lenz.from_dict([1, 2])
    d = lz.to_dict()
    assert Lenz.from_dict(d["frame"]).n_observed == 8  # a bare frame is accepted too


# --------------------------------------------------------------------------- envelopes and events


def test_call_envelopes(quad_study):
    lz = quad_study
    ok = lz.call("suggest", {"q": 2})
    assert ok["ok"] and ok["command"] == "suggest" and len(ok["result"]["candidates"]) == 2
    json.loads(dumps(ok))
    assert lz.call("set-bounds", {"bounds": "full"})["ok"]
    assert lz.call("status")["result"]["n_observed"] == 8
    cases = [
        ("bogus", {}, "unknown command"),
        ("create", {}, "already exists"),
        ("suggest", {"qq": 2}, "unknown argument"),
        ("observe", {}, "missing required"),
        ("suggest", "q=2", "args must be an object"),
        ("set_acqf", {"acqf": "bogus"}, "unknown acquisition"),
        ("submit", {"config": {"x0": 9.0, "x1": 0.0}}, "outside the domain"),
        ("observe", {"trial_id": 99, "metrics": {"f": 1.0}}, "unknown trial_id"),
        ("observe", {"config": {"x0": 0.0, "x1": 0.0}, "metrics": {"g": 1.0}}, "missing"),
        ("abandon", {"trial_id": 0}, "only pending"),
        ("diagnostics", {"metric": "nope"}, "unknown metric"),
    ]
    for cmd, args, needle in cases:
        n_events = len(lz.frame.events)
        env = lz.call(cmd, args)
        assert env["ok"] is False and env["error"]["type"] == "LenzError", (cmd, env)
        assert needle in env["error"]["message"], (cmd, env["error"]["message"])
        assert set(env) == {"ok", "command", "error"}
        assert len(lz.frame.events) == n_events + 1 and lz.frame.events[-1].ok is False
        json.loads(dumps(env))
    assert lz.n_observed == 8  # failed calls change nothing


def test_call_reports_unexpected_exceptions(quad_study, monkeypatch):
    lz = quad_study

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(lz, "_data", boom)
    env = lz.call("status")
    assert env == {"ok": False, "command": "status", "error": {"type": "RuntimeError", "message": "boom"}}
    assert lz.frame.events[-1].command == "status" and not lz.frame.events[-1].ok
    with pytest.raises(RuntimeError):  # library calls do not swallow bugs
        lz.status()


def test_event_log(quad_study):
    lz = quad_study
    n0 = len(lz.frame.events)
    assert n0 == 1 + 8 and [e.seq for e in lz.frame.events] == list(range(1, n0 + 1))
    lz.suggest(q=2)
    lz.call("trials", {"last": 1})
    lz.call("bogus")
    many = [{"x0": 0.0, "x1": float(v)} for v in np.linspace(-2, 2, 200)]
    lz.predict(many)
    ev = lz.frame.events[n0:]
    assert [e.command for e in ev] == ["suggest", "trials", "bogus", "predict"]
    assert [e.ok for e in ev] == [True, True, False, True]
    assert ev[0].args == {"q": 2} and "2 candidate(s)" in ev[0].summary
    assert ev[1].args == {"last": 1}
    assert ev[3].args["configs"]["truncated"] is True and ev[3].args["configs"]["chars"] > 2000
    sub = lz.submit({"x0": 0.0, "x1": 0.0})
    t = lz.frame.trial(sub["trial_id"])
    assert t.created_seq == lz.frame.events[-1].seq
    lz.observe({"f": -0.1}, trial_id=t.trial_id)
    assert t.completed_seq == lz.frame.events[-1].seq and "completed" in lz.frame.events[-1].summary
    st = lz.status()
    assert st["n_events"] == len(lz.frame.events) - 1 and len(st["recent_events"]) == 5


# --------------------------------------------------------------------------- performance


def _bulk_study(n: int, D: int) -> Lenz:
    rng = np.random.default_rng(0)
    lz = Lenz(real_space(D), "y", seed=0)
    X = rng.uniform(size=(n, D))
    y = -np.sum((X - 0.3) ** 2, axis=1) + 0.1 * np.sin(5 * X).sum(axis=1)
    for x, v in zip(X, y):
        lz.observe({"y": float(v)}, config={f"x{i}": float(a) for i, a in enumerate(x)})
    return lz


def test_suggest_performance_n100_d6():
    lz = _bulk_study(100, 6)
    t0 = time.perf_counter()
    r = lz.suggest()
    elapsed = time.perf_counter() - t0
    assert r["mode"] == "model" and elapsed < 2.0, elapsed


@pytest.mark.slow
def test_suggest_performance_n200_d20():
    lz = _bulk_study(200, 20)
    t0 = time.perf_counter()
    r = lz.suggest()
    elapsed = time.perf_counter() - t0
    assert r["mode"] == "model" and elapsed < 6.0, elapsed


# --------------------------------------------------------------------------- review-round regressions


def test_trials_limit_is_top_k_after_sorting_and_last_is_most_recent():
    lz = Lenz(real_space(1), {"metric": "y", "minimize": True}, seed=0)
    for i, v in enumerate([5, 1, 9, 2, 8, 7, 6]):
        lz.observe({"y": float(v)}, config={"x0": i / 10})
    top = lz.trials(sort_by="value", limit=3)
    assert [t["value"] for t in top["trials"]] == [1.0, 2.0, 5.0] and top["n"] == 3
    # 'last' keeps its meaning: the k most recent trials (here sorted by value afterwards)
    recent = lz.trials(sort_by="value", last=3)
    assert [t["trial_id"] for t in recent["trials"]] == [6, 5, 4]
    both = lz.trials(sort_by="value", last=4, limit=2)
    assert [t["value"] for t in both["trials"]] == [2.0, 6.0]
    assert [t["trial_id"] for t in lz.trials(limit=2)["trials"]] == [0, 1]
    assert lz.trials(limit=0)["n"] == 0
    with pytest.raises(LenzError):
        lz.trials(limit=-1)


def _int_grid_study() -> Lenz:
    space = [{"name": "k", "type": "int", "lower": 0, "upper": 10},
             {"name": "j", "type": "int", "lower": 0, "upper": 10}]
    lz = Lenz(space, "y", seed=0)
    rng = np.random.default_rng(1)
    for _ in range(8):
        c = {"k": int(rng.integers(0, 11)), "j": int(rng.integers(0, 11))}
        lz.observe({"y": float(-((c["k"] - 6) ** 2 + (c["j"] - 3) ** 2))}, config=c)
    return lz


@pytest.mark.parametrize("radius", [0.01, 0.02, 0.04])
def test_trust_region_narrower_than_integer_lattice_reaches_neighbours(radius):
    lz = _int_grid_study()
    inc = lz.incumbent()["incumbent"]["config"]
    r = lz.suggest(around="incumbent", radius=radius, q=3)
    box = r["region"]["box"]
    for name in ("k", "j"):  # the box reaches at least one lattice step on each side (inside the domain)
        assert box[name][0] <= max(inc[name] - 1, 0) and box[name][1] >= min(inc[name] + 1, 10)
    assert all(c["duplicate_of"] is None for c in r["candidates"]), r["notes"]
    for c in r["candidates"]:
        assert all(box[n][0] <= c["config"][n] <= box[n][1] for n in ("k", "j"))
    # the active region still bounds the snapped box
    lz.set_bounds({"k": [inc["k"], 10], "j": [inc["j"], 10]})
    r = lz.suggest(around="incumbent", radius=radius)
    assert r["region"]["box"]["k"][0] == inc["k"] and r["region"]["box"]["j"][0] == inc["j"]
    assert r["candidates"][0]["duplicate_of"] is None
    # ordinal lattice too
    sp = [{"name": "o", "type": "ordinal", "values": list(range(21))},
          {"name": "x", "type": "real", "lower": 0, "upper": 1}]
    lz2 = Lenz(sp, "y", seed=0)
    for o, x in [(3, 0.1), (10, 0.5), (15, 0.9), (7, 0.3), (12, 0.6), (18, 0.2)]:
        lz2.observe({"y": -abs(o - 10) - (x - 0.5) ** 2}, config={"o": o, "x": x})
    r = lz2.suggest(around="incumbent", radius=radius * 0.1, q=2)
    assert {9, 10, 11} <= set(r["region"]["box"]["o"])
    assert all(c["duplicate_of"] is None for c in r["candidates"])


# 11 branin-currin evaluations (minimize f1, f2) from a real campaign; none dominates ref (18, 6),
# and the Monte Carlo EHVI was exactly 0 on the whole domain
_NO_DOMINATION_DATA = [
    ((0.7775576738640666, 0.18458870332688093), (21.599733140812436, 9.81609133713628)),
    ((0.20872606802731752, 0.5160166751593351), (10.019934576854594, 8.558459765647354)),
    ((0.30125480704009533, 0.44460189156234264), (18.527849396429104, 9.01580700111096)),
    ((0.748018242418766, 0.8536673961207271), (156.6253103409222, 4.69592689853129)),
    ((0.6015030182898045, 0.3306954875588417), (14.596310319061818, 8.680498841209642)),
    ((1.0, 0.5496740865947262), (29.4233264956965, 6.0804803558988985)),
    ((1.0, 0.5807633599186252), (34.530042054122816, 5.875943566962549)),
    ((1.0, 0.544674422118976), (28.642680302836034, 6.114563167792055)),
    ((1.0, 0.5404708861119192), (27.995045729756335, 6.143481849689268)),
    ((0.6183958887563578, 0.615377623519908), (64.83225250624469, 6.150620454161267)),
    ((1.0, 0.5436314893308659), (28.481255028308723, 6.121715617425861)),
]


def _ei_below(mean: float, std: float, ref: float) -> float:
    """E[(ref - F)^+] for F ~ N(mean, std^2) (improvement of a minimized metric below ``ref``)."""
    from scipy.stats import norm
    z = (ref - mean) / std
    return (ref - mean) * norm.cdf(z) + std * norm.pdf(z)


def test_ehvi_is_informative_before_any_point_dominates_the_ref_point():
    lz = Lenz(real_space(2), MOO_OBJECTIVES, seed=0)
    for (a, b), (f1, f2) in _NO_DOMINATION_DATA:
        lz.observe({"f1": f1, "f2": f2}, config={"x0": a, "x1": b})
    assert lz.pareto()["hypervolume"] == 0.0
    grid = [{"x0": float(a), "x1": float(b)} for a in np.linspace(0, 1, 9) for b in np.linspace(0, 1, 9)]
    vals = np.array([s["acquisition_values"]["ehvi"] for s in lz.score(grid)["scores"]])
    # informative: positive over most of the space and varying (exact zeros only where the
    # product of two Gaussian tail expectations underflows, e.g. ~1e-350 far above the ref)
    assert np.mean(vals > 0) > 0.75 and vals.max() > 0 and np.unique(vals[vals > 0]).size > 10
    # empty front: HVI(y) = prod_i (ref_i - y_i)^+, so with independent posteriors
    # EHVI = prod_i E[(ref_i - f_i)^+] exactly
    preds = lz.predict(grid)["predictions"]
    expected = np.array([_ei_below(p["metrics"]["f1"]["mean"], p["metrics"]["f1"]["std"], 18.0)
                         * _ei_below(p["metrics"]["f2"]["mean"], p["metrics"]["f2"]["std"], 6.0) for p in preds])
    assert vals == pytest.approx(expected, rel=1e-6)
    r = lz.suggest(q=2)
    assert r["mode"] == "model" and r["acqf"] == "ehvi"
    assert any("reference point" in n for n in r["notes"])
    v = [c["acquisition_values"]["ehvi"] for c in r["candidates"]]
    assert v[0] > 0 and v[0] >= vals.max() * (1 - 1e-6)
    assert all(c["duplicate_of"] is None for c in r["candidates"])
    # once a point dominates the ref point, the Monte Carlo EHVI is back in charge
    lz.observe({"f1": 10.0, "f2": 5.0}, config={"x0": 0.1, "x1": 0.9})
    r2 = lz.suggest()
    assert lz.pareto()["hypervolume"] > 0 and not any("reference point" in n for n in r2["notes"])


def test_set_acqf_says_when_objective_refs_override_ref_point():
    lz = Lenz(real_space(2), MOO_OBJECTIVES, seed=0)
    res = lz.set_acqf("ehvi", ref_point=[300.0, 15.0])
    assert any("ref" in n and "ignored" in n for n in res["notes"]), res
    # objectives without their own ref take the acquisition's ref_point, silently
    lz2 = Lenz(real_space(2), [{"metric": "f1", "minimize": True}, {"metric": "f2", "minimize": True}], seed=0)
    assert lz2.set_acqf("ehvi", ref_point=[300.0, 15.0])["notes"] == []
    seeded(lz2, branin_currin, [{"x0": 0.2, "x1": 0.5}, {"x0": 0.6, "x1": 0.3}])
    assert lz2.pareto()["ref_point"] == [300.0, 15.0]


def _feasibility_phase_study(objectives, seed: int = 0, model: dict | None = None) -> Lenz:
    sp = real_space(2)
    lz = Lenz(sp, objectives, constraints=[{"metric": "g", "upper": 0.6}], seed=seed, model=model)
    rng = np.random.default_rng(seed)
    for _ in range(6):
        a, b = rng.uniform(0.35, 1, size=2)
        lz.observe({"f": -(a - 0.7) ** 2 - (b - 0.7) ** 2, "h": -(a - 0.2) ** 2 - b ** 2, "g": a + b},
                   config={"x0": float(a), "x1": float(b)})
    return lz


@pytest.mark.parametrize("prior", ["botorch", "median"])
@pytest.mark.parametrize("objectives", ["f", ["f", "h"]])
def test_feasibility_phase_batch_does_not_collapse_onto_believed_feasible_pick(objectives, prior):
    lz = _feasibility_phase_study(objectives, model={"prior": prior})
    assert not lz.incumbent()["feasible_found"]
    r = lz.suggest(q=4)
    assert r["feasibility_phase"] and any("feasibility phase" in n for n in r["notes"])
    Z = np.array([[c["config"]["x0"], c["config"]["x1"]] for c in r["candidates"]])
    d = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    assert d.min() > 0.02, Z
    if prior != "median":
        return
    # a pending trial believed feasible is an incumbent for the next pick (kriging believer);
    # this scenario's data make the median-prior model believe its first pick is feasible
    first = r["candidates"][0]
    assert first["prob_feasible"] > 0.5
    lz.submit(first["config"])
    r2 = lz.suggest()
    assert not r2["feasibility_phase"]
    assert r2["candidates"][0]["config"] != first["config"]


def test_set_objectives_drops_per_objective_params_when_objectives_change():
    lz = Lenz(real_space(2), ["a", "b"], seed=0)
    lz.set_acqf("ehvi", ref_point=[1.0, 2.0])
    same = lz.set_objectives(["a", "b"])
    assert same["acqf_params"] == {"ref_point": [1.0, 2.0]} and same["notes"] == []
    r = lz.set_objectives(["a", "c"])  # 2.0 was a reference value for 'b', not for 'c'
    assert r["acqf"] == "ehvi" and "ref_point" not in r["acqf_params"]
    assert any("ref_point" in n for n in r["notes"])
    lz.set_acqf("ehvi", ref_point=[1.0, 2.0])
    r = lz.set_objectives([{"metric": "a", "minimize": True}, "c"])  # direction changed
    assert r["acqf_params"] == {} and any("ref_point" in n for n in r["notes"])
    lz.set_acqf("parego", weights=[3, 1])  # normalized-objective weights are unit-free: kept
    assert lz.set_objectives(["a", "d"])["acqf_params"] == {"weights": [0.75, 0.25]}
