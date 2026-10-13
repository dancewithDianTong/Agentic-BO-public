"""Tests for the minimal JSON-schema validator used to check agent tool inputs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from agentic_bo.sara.schema import MAX_ERRORS, format_errors, is_valid, validate
from agentic_bo.sara.tools import bounds_schema, config_schema
from agentic_bo.space import SearchSpace


def obj(props: dict, required: list | None = None, extra=False) -> dict:
    s = {"type": "object", "properties": props, "additionalProperties": extra}
    if required:
        s["required"] = required
    return s


# --------------------------------------------------------------------------- types


@pytest.mark.parametrize("value, type_, ok", [
    ("a", "string", True), (1, "string", False),
    (1, "number", True), (1.5, "number", True), (True, "number", False), (False, "integer", False),
    (3, "integer", True), (3.0, "integer", True), (3.5, "integer", False), (np.int64(4), "integer", True),
    (np.float64(2.0), "integer", True), (np.float32(2.5), "number", True), (np.bool_(True), "boolean", True),
    (True, "boolean", True), (0, "boolean", False), (None, "null", True), (0, "null", False),
    ({}, "object", True), ([], "object", False), ([], "array", True), ((1, 2), "array", True),
    ("x", "array", False), ({"a": 1}, "array", False),
])
def test_type_checks(value, type_, ok):
    assert is_valid(value, {"type": type_}) is ok


def test_type_list_is_nullable_and_message_names_both():
    s = {"type": ["number", "null"]}
    assert is_valid(None, s) and is_valid(2.5, s)
    errs = validate("x", s)
    assert errs == ['$: expected number or null, got string "x"']


def test_integer_message_mentions_whole_number():
    errs = validate({"q": 2.5}, obj({"q": {"type": "integer"}}))
    assert errs == ["$.q: expected integer, got 2.5 (not a whole number)"]


def test_bool_is_never_a_number():
    errs = validate(True, {"type": "number"})
    assert errs == ["$: expected number, got boolean true"]


def test_non_finite_numbers_rejected():
    for bad in (math.nan, math.inf, -math.inf):
        errs = validate(bad, {"type": "number"})
        assert errs and "finite" in errs[0]


def test_unsupported_schema_type_raises():
    with pytest.raises(ValueError, match="unsupported schema type"):
        validate(1, {"type": "float"})


def test_true_false_schemas():
    assert validate(123, True) == []
    assert validate(123, {}) == []
    assert validate(1, False) == ["$: no value is allowed here"]


# --------------------------------------------------------------------------- enum / const


def test_enum_uses_json_equality():
    s = {"enum": [1, "a", None]}
    assert is_valid(1, s) and is_valid(1.0, s) and is_valid(None, s) and is_valid("a", s)
    assert not is_valid(True, s)          # True is not 1 in JSON
    assert not is_valid("b", s)
    errs = validate("b", s)
    assert errs == ['$: "b" is not one of the allowed values [1, "a", null]']
    assert not is_valid(1, {"enum": [True, False]})
    assert is_valid([1, {"a": 2}], {"enum": [[1.0, {"a": 2.0}]]})


def test_long_enum_is_abbreviated():
    errs = validate("zz", {"enum": [f"o{i}" for i in range(30)]})
    assert "(30 options)" in errs[0]


def test_const():
    assert is_valid("incumbent", {"const": "incumbent"})
    assert validate("center", {"const": "incumbent"}) == ['$: must be "incumbent", got "center"']


# --------------------------------------------------------------------------- numbers / strings


def test_numeric_bounds():
    s = {"type": "number", "minimum": 0, "maximum": 10, "exclusiveMinimum": -1, "exclusiveMaximum": 11}
    assert is_valid(0, s) and is_valid(10, s)
    assert validate(-0.5, s) == ["$: must be >= 0, got -0.5"]
    assert validate(12, s) == ["$: must be <= 10, got 12", "$: must be < 11, got 12"]
    r = {"type": "number", "exclusiveMinimum": 0, "maximum": 1}
    assert validate(0, r) == ["$: must be > 0, got 0"]
    assert is_valid(1e-12, r)
    assert validate(0.123456789012, {"type": "number", "maximum": 0.1}) == [
        "$: must be <= 0.1, got 0.123456789012"]


def test_string_length():
    s = {"type": "string", "minLength": 1, "maxLength": 3}
    assert validate("", s) == ["$: must be a non-empty string"]
    assert validate("abcd", s) == ["$: must be at most 3 characters long, got 4"]
    assert validate("a", {"type": "string", "minLength": 2}) == [
        "$: must be a string of at least 2 character(s), got 1"]


# --------------------------------------------------------------------------- arrays


def test_arrays_items_and_lengths_with_paths():
    s = {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "number", "minimum": 0}}
    assert is_valid([0, 1.5], s)
    assert validate([], s) == ["$: must have at least 1 item(s), got 0"]
    assert validate([1, 2, 3, 4], s) == ["$: must have at most 3 item(s), got 4"]
    assert validate([1, -2, "x"], s) == ["$[1]: must be >= 0, got -2", '$[2]: expected number, got string "x"']


def test_unique_items():
    s = {"type": "array", "uniqueItems": True}
    assert is_valid([1, "1", True], s)
    errs = validate(["a", "b", "a"], s)
    assert errs == ['$: items must be unique, but [2] repeats [0] ("a")']
    assert not is_valid([1, 1.0], s)


def test_items_false_rejects_every_item():
    assert validate([1], {"type": "array", "items": False}) == ["$[0]: no value is allowed here"]
    assert validate([], {"type": "array", "items": False}) == []


# --------------------------------------------------------------------------- objects


def test_required_unknown_and_nested_paths():
    s = obj({"config": obj({"x0": {"type": "number", "maximum": 1}, "c": {"enum": ["a", "b"]}}, ["x0", "c"]),
             "rationale": {"type": "string", "minLength": 1}}, ["config", "rationale"])
    assert is_valid({"config": {"x0": 0.5, "c": "a"}, "rationale": "r"}, s)
    errs = validate({"config": {"x0": 2, "zz": 1}, "extra": True}, s)
    assert "$.config.x0: must be <= 1, got 2" in errs
    assert any(e.startswith("$.config.c: required property is missing (one of \"a\", \"b\")") for e in errs)
    assert "$.config: unknown property 'zz' (allowed: c, x0)" in errs
    assert any(e.startswith("$.rationale: required property is missing (a string)") for e in errs)
    assert "$: unknown property 'extra' (allowed: config, rationale)" in errs


def test_additional_properties_schema_and_default_allows_extra():
    s = {"type": "object", "properties": {"a": {"type": "integer"}}, "additionalProperties": {"type": "string"}}
    assert is_valid({"a": 1, "b": "x"}, s)
    assert validate({"a": 1, "b": 2}, s) == ["$.b: expected string, got integer 2"]
    assert is_valid({"anything": [1, 2]}, {"type": "object"})


def test_path_quoting_for_odd_keys():
    s = {"type": "object", "properties": {"a b": {"type": "integer"}, "0x": {"type": "integer"}}}
    errs = validate({"a b": "x", "0x": "y"}, s)
    assert '$["a b"]: expected integer, got string "x"' in errs
    assert '$["0x"]: expected integer, got string "y"' in errs


def test_min_max_properties_and_non_string_keys():
    s = {"type": "object", "minProperties": 1, "maxProperties": 1}
    assert validate({}, s) == ["$: must have at least 1 propert(ies), got 0"]
    assert validate({"a": 1, "b": 2}, s) == ["$: must have at most 1 propert(ies), got 2"]
    assert validate({1: "x"}, {"type": "object"}) == ["$: property names must be strings, got 1"]


def test_type_error_short_circuits_structure_checks():
    errs = validate("abc", obj({"a": {"type": "integer"}}, ["a"]))
    assert errs == ['$: expected object, got string "abc"']


# --------------------------------------------------------------------------- combinators


AROUND = {"anyOf": [{"type": "string", "enum": ["incumbent"]},
                    obj({"x": {"type": "number", "minimum": 0, "maximum": 1}}, ["x"]),
                    {"type": "null"}]}


def test_any_of_reports_the_branch_with_matching_type():
    assert is_valid("incumbent", AROUND) and is_valid({"x": 0.3}, AROUND) and is_valid(None, AROUND)
    assert validate("center", AROUND) == ['$: "center" is not one of the allowed values ["incumbent"]']
    assert validate({"x": 3}, AROUND) == ["$.x: must be <= 1, got 3"]
    errs = validate(5, AROUND)
    assert len(errs) == 1
    assert errs[0].startswith('$: 5 matches none of the allowed forms: (1) one of "incumbent"; '
                              '(2) an object with keys x; (3) null')


def test_one_of_requires_exactly_one():
    s = {"oneOf": [{"type": "number"}, {"type": "integer"}]}
    assert is_valid(2.5, s)
    errs = validate(2, s)
    assert errs == ["$: 2 matches more than one alternative (1, 2); it must match exactly one"]


def test_all_of_collects_every_error():
    s = {"allOf": [{"type": "number", "minimum": 0}, {"type": "number", "maximum": -1}]}
    assert validate(0.5, s) == ["$: must be <= -1, got 0.5"]


def test_error_cap_and_format():
    s = {"type": "array", "items": {"type": "string"}}
    errs = validate(list(range(40)), s)
    assert len(errs) == MAX_ERRORS + 1
    assert errs[-1] == f"... and {40 - MAX_ERRORS} more error(s)"
    assert format_errors(["a", "b"], prefix="bad: ") == "bad: a; b"
    assert format_errors(["a"]) == "a"


# --------------------------------------------------------------------------- generated tool schemas


@pytest.fixture
def mixed_space() -> SearchSpace:
    return SearchSpace.from_spec([
        {"name": "lr", "type": "real", "lower": 1e-4, "upper": 1.0, "log_scale": True},
        {"name": "x", "type": "real", "lower": -5, "upper": 10},
        {"name": "k", "type": "int", "lower": 1, "upper": 8},
        {"name": "bs", "type": "ordinal", "values": [16, 32, 64]},
        {"name": "opt", "type": "categorical", "choices": ["adam", "sgd"]},
    ])


def test_config_schema_accepts_valid_and_explains_invalid(mixed_space):
    s = config_schema(mixed_space)
    assert s["required"] == mixed_space.names and s["additionalProperties"] is False
    good = {"lr": 0.01, "x": 2.5, "k": 3, "bs": 32, "opt": "adam"}
    assert validate(good, s) == []
    assert validate({**good, "k": 3.0}, s) == []           # integral float is an integer
    errs = validate({**good, "lr": 2.0, "k": 2.5, "bs": 48, "opt": "rmsprop", "extra": 1}, s)
    assert "$.lr: must be <= 1, got 2" in errs
    assert "$.k: expected integer, got 2.5 (not a whole number)" in errs
    assert "$.bs: 48 is not one of the allowed values [16, 32, 64]" in errs
    assert '$.opt: "rmsprop" is not one of the allowed values ["adam", "sgd"]' in errs
    assert any("unknown property 'extra'" in e for e in errs)
    missing = validate({"lr": 0.01}, s)
    assert sum("required property is missing" in e for e in missing) == 4


def test_bounds_schema(mixed_space):
    s = bounds_schema(mixed_space)
    assert validate({"x": [0, 5], "opt": ["sgd"], "bs": [16, 64]}, s) == []
    assert validate({}, s) == []
    errs = validate({"x": [0, 50], "opt": [], "k": [1], "nope": [0, 1]}, s)
    assert "$.x[1]: must be <= 10, got 50" in errs
    assert "$.opt: must have at least 1 item(s), got 0" in errs
    assert "$.k: must have at least 2 item(s), got 1" in errs
    assert any("unknown property 'nope'" in e for e in errs)


def test_integers_beyond_float_range_get_errors_not_exceptions():
    # json.loads("1" + "0" * 400) is a valid JSON integer; the validator must report, not raise
    big = 10 ** 400
    assert validate(big, {"type": "integer"}) == []
    errs = validate(big, {"type": "integer", "minimum": 1, "maximum": 10})
    assert len(errs) == 1 and errs[0].startswith("$: must be <= 10, got 1000") and len(errs[0]) < 120
    assert validate(-big, {"type": "number", "minimum": 0})[0].startswith("$: must be >= 0, got -1000")
    assert validate(-big, {"type": "number", "exclusiveMaximum": 0}) == []
    assert validate(big, {"enum": [1, 2.5]})[0].startswith("$: 1000")
    assert validate(big, {"const": 1.0})[0].startswith("$: must be 1.0, got 1000")
    assert validate([big, big], {"type": "array", "uniqueItems": True})[0].startswith("$: items must be unique")
    assert validate({"q": big}, {"type": "object", "properties": {"q": {"type": "integer", "maximum": 10}}}) \
        == [validate(big, {"type": "integer", "maximum": 10})[0].replace("$", "$.q", 1)]
