import numpy as np
import pytest

from agentic_bo.space import SearchSpace
from agentic_bo.utils import LenzError, dumps, jsonable

SPEC = [
    {"name": "x", "type": "real", "lower": -5, "upper": 10},
    {"name": "lr", "type": "real", "lower": 1e-4, "upper": 1.0, "log_scale": True},
    {"name": "k", "type": "int", "lower": 1, "upper": 8},
    {"name": "width", "type": "ordinal", "values": [16, 32, 64, 128]},
    {"name": "opt", "type": "categorical", "choices": ["adam", "sgd", "rmsprop"]},
]


@pytest.fixture
def space():
    return SearchSpace.from_spec(SPEC)


def test_dims(space):
    assert space.d == 5 and space.D == 7
    assert space.slices["opt"] == slice(4, 7)


def test_roundtrip(space):
    cfg = {"x": 2.5, "lr": 0.01, "k": 3, "width": 64, "opt": "sgd"}
    z = space.encode(cfg)
    assert z.shape == (7,) and np.all((z >= 0) & (z <= 1))
    dec = space.decode(z)
    assert dec.pop("lr") == pytest.approx(0.01)
    exp = space.validate(cfg); exp.pop("lr")
    assert dec == exp
    assert np.isclose(z[1], 0.5)  # log scale midpoint of [1e-4, 1]


def test_validate_errors(space):
    with pytest.raises(LenzError, match="missing"):
        space.validate({"x": 1.0})
    with pytest.raises(LenzError, match="outside"):
        space.validate({"x": 11, "lr": 0.1, "k": 2, "width": 16, "opt": "adam"})
    with pytest.raises(LenzError, match="integer"):
        space.validate({"x": 1, "lr": 0.1, "k": 2.5, "width": 16, "opt": "adam"})
    with pytest.raises(LenzError, match="not one of"):
        space.validate({"x": 1, "lr": 0.1, "k": 2, "width": 17, "opt": "adam"})
    with pytest.raises(LenzError, match="unknown"):
        space.validate({"x": 1, "lr": 0.1, "k": 2, "width": 16, "opt": "adam", "zz": 1})
    assert space.validate({"x": 1, "lr": 0.1, "k": 2.0, "width": 16.0, "opt": "adam"})["k"] == 2


def test_bounds_and_sampling(space):
    b = {"x": [0, 1], "k": [2.2, 5.9], "opt": ["adam", "rmsprop"], "width": [32, 128]}
    nb = space.validate_bounds(b)
    assert nb["k"] == [3, 5] and nb["opt"] == ["adam", "rmsprop"]
    lb, ub = space.encoded_bounds(b)
    assert ub[space.slices["opt"].start + 1] == 0.0  # sgd excluded
    pts = space.sobol(64, 0, bounds=b)
    for c in pts:
        assert space.contains(c, bounds=b)
        assert c["width"] in (32, 128)
    assert {c["opt"] for c in pts} == {"adam", "rmsprop"}
    assert space.validate_bounds("full") is None
    with pytest.raises(LenzError):
        space.validate_bounds({"x": [5, 1]})
    with pytest.raises(LenzError):
        space.validate_bounds({"x": [-6, 1]})


def test_sobol_sequence_continuation(space):
    a = space.sobol(8, 3)
    b = space.sobol(4, 3, skip=4)
    assert a[4:] == b


def test_project_and_distance(space):
    rng = np.random.default_rng(0)
    Z = rng.uniform(size=(10, space.D))
    P = space.project(Z)
    assert np.allclose(space.project(P), P)
    c = space.decode(P[0])
    assert space.distance(c, c) == 0.0
    assert 0 < space.distance(space.decode(P[0]), space.decode(P[1])) <= 1


def test_continuous_project_keeps_precision():
    s = SearchSpace.from_spec({"a": {"type": "real", "lower": 0, "upper": 1}, "b": {"type": "real", "lower": 0, "upper": 2}})
    z = np.array([0.123456789, 1.2])
    assert np.allclose(s.project(z), [0.123456789, 1.0])


def test_spec_roundtrip_and_describe(space):
    s2 = SearchSpace.from_spec(space.to_spec())
    assert s2 == space
    assert "| `lr` | real" in space.describe()


def test_jsonable():
    out = jsonable({"a": np.float64(1.5), "b": np.array([1, 2]), "c": float("nan"), "d": (np.int64(3),)})
    assert out == {"a": 1.5, "b": [1, 2], "c": None, "d": [3]}
    assert dumps({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'
