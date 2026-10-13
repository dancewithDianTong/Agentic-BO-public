"""A minimal JSON-schema validator for agent tool inputs.

The harness validates every tool call against the tool's ``input_schema`` before
dispatching it (DESIGN.md §9.1). With eager input streaming the API does not enforce
schemas, so this validator is the guard, and its messages go straight back to the
agent: each error names the offending location as a JSON path (``$.config.x0``) and
says what was expected and what was received, so an LLM can fix its call.

Supported keywords: ``type`` (a name or a list of names; ``integer`` accepts integral
numbers, booleans are never numbers), ``enum``, ``const``, ``properties``,
``required``, ``additionalProperties`` (``false`` or a schema), ``minProperties`` /
``maxProperties``, ``minimum`` / ``maximum`` / ``exclusiveMinimum`` /
``exclusiveMaximum``, ``minLength`` / ``maxLength``, ``items`` (one schema for every
element), ``minItems`` / ``maxItems``, ``uniqueItems``, ``anyOf``, ``oneOf`` and
``allOf``. Annotation keywords (``description``, ``title``, ``default``, ...) are
ignored. "Nullable" is spelled with a type list, e.g. ``{"type": ["number", "null"]}``.

    >>> validate({"q": 0}, {"type": "object", "properties": {"q": {"type": "integer", "minimum": 1}}})
    ['$.q: must be >= 1, got 0']
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = ["validate", "is_valid", "format_errors", "SCHEMA_TYPES", "MAX_ERRORS"]

#: JSON-schema type names understood by :func:`validate`
SCHEMA_TYPES = ("object", "array", "string", "number", "integer", "boolean", "null")
#: errors reported per call (the rest are summarized), so messages stay readable
MAX_ERRORS = 12
_PREVIEW_CHARS = 60


# --------------------------------------------------------------------------- value helpers


def _is_bool(v: Any) -> bool:
    return isinstance(v, (bool, np.bool_))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not _is_bool(v)


def _is_integer(v: Any) -> bool:
    if isinstance(v, (int, np.integer)) and not _is_bool(v):
        return True
    return isinstance(v, (float, np.floating)) and math.isfinite(float(v)) and float(v).is_integer()


def _num(v: Any) -> int | float:
    """A number for exact comparisons: python ints stay ints (any size, no float overflow)."""
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    return float(v)


def _type_of(v: Any) -> str:
    """JSON type name of a python value (``"unsupported"`` for anything else)."""
    if v is None:
        return "null"
    if _is_bool(v):
        return "boolean"
    if _is_integer(v):
        return "integer"
    if _is_number(v):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, Mapping):
        return "object"
    if isinstance(v, (list, tuple)):
        return "array"
    return type(v).__name__


def _matches_type(v: Any, name: str) -> bool:
    if name == "null":
        return v is None
    if name == "boolean":
        return _is_bool(v)
    if name == "integer":
        return _is_integer(v)
    if name == "number":
        return _is_number(v)
    if name == "string":
        return isinstance(v, str)
    if name == "object":
        return isinstance(v, Mapping)
    if name == "array":
        return isinstance(v, (list, tuple))
    raise ValueError(f"unsupported schema type {name!r}; supported: {', '.join(SCHEMA_TYPES)}")


def _json_equal(a: Any, b: Any) -> bool:
    """Equality with JSON semantics: ``True != 1``, ``1 == 1.0``, deep for arrays / objects."""
    if _is_bool(a) or _is_bool(b):
        return _is_bool(a) and _is_bool(b) and bool(a) == bool(b)
    if _is_number(a) or _is_number(b):
        return _is_number(a) and _is_number(b) and _num(a) == _num(b)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b)
                and all(_json_equal(x, y) for x, y in zip(a, b)))
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        return (isinstance(a, Mapping) and isinstance(b, Mapping) and set(a) == set(b)
                and all(_json_equal(a[k], b[k]) for k in a))
    return a == b


def _preview(v: Any) -> str:
    """Short JSON-ish rendering of a value for error messages."""
    try:
        text = json.dumps(v, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        text = repr(v)
    return text if len(text) <= _PREVIEW_CHARS else text[: _PREVIEW_CHARS - 3] + "..."


def _fmt_num(x: Any) -> str:
    if isinstance(x, (int, np.integer)) and not _is_bool(x):
        text = str(int(x))
        return text if len(text) <= 20 else text[:17] + "..."
    if _is_integer(x) and abs(float(x)) < 1e15:
        return str(int(x))
    return f"{float(x):.12g}" if _is_number(x) else str(x)


_ARTICLE = {"object": "an object", "array": "an array", "integer": "an integer", "string": "a string",
            "number": "a number", "boolean": "a boolean", "null": "null"}


def _child(path: str, key: Any) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    key = str(key)
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{json.dumps(key)}]"


def _describe(schema: Any) -> str:
    """One-phrase description of what a (sub)schema accepts, for anyOf / oneOf messages."""
    if not isinstance(schema, Mapping):
        return "anything" if schema is True else "nothing"
    if "const" in schema:
        return f"the constant {_preview(schema['const'])}"
    if "enum" in schema:
        return "one of " + ", ".join(_preview(v) for v in schema["enum"])
    t = schema.get("type")
    if isinstance(t, (list, tuple)):
        return " or ".join(_ARTICLE.get(str(x), str(x)) for x in t)
    if t == "object" and schema.get("required"):
        return "an object with keys " + ", ".join(map(str, schema["required"]))
    if t:
        return _ARTICLE.get(str(t), str(t))
    if "anyOf" in schema or "oneOf" in schema:
        return " or ".join(_describe(s) for s in schema.get("anyOf", schema.get("oneOf", [])))
    return "a value matching a sub-schema"


# --------------------------------------------------------------------------- validation


def validate(instance: Any, schema: Any, path: str = "$") -> list[str]:
    """Validate ``instance`` against ``schema``; return a list of human-readable errors.

    An empty list means the instance is valid. Each error starts with the JSON path of
    the offending value (``$`` is the whole instance). At most :data:`MAX_ERRORS`
    errors are listed; a final entry says how many more were found.

    Raises:
        ValueError: if the *schema* itself uses an unsupported type name.
    """
    errors: list[str] = []
    _validate(instance, schema, path, errors)
    if len(errors) > MAX_ERRORS:
        extra = len(errors) - MAX_ERRORS
        errors = errors[:MAX_ERRORS] + [f"... and {extra} more error(s)"]
    return errors


def is_valid(instance: Any, schema: Any) -> bool:
    """``True`` iff :func:`validate` finds no error."""
    return not validate(instance, schema)


def format_errors(errors: Sequence[str], prefix: str = "") -> str:
    """Join validation errors into one message (``prefix`` is prepended when given)."""
    body = "; ".join(errors)
    return f"{prefix}{body}" if prefix else body


def _validate(v: Any, schema: Any, path: str, errors: list[str]) -> None:
    if schema is True or schema is None:
        return
    if schema is False:
        errors.append(f"{path}: no value is allowed here")
        return
    if not isinstance(schema, Mapping):
        raise ValueError(f"a schema must be an object or a boolean, got {schema!r}")

    # ---- type / enum / const: if these fail, the structural checks below are noise
    t = schema.get("type")
    if t is not None:
        names = list(t) if isinstance(t, (list, tuple)) else [t]
        if not any(_matches_type(v, n) for n in names):
            want = " or ".join(names)
            got = _type_of(v)
            if "integer" in names and _is_number(v):
                errors.append(f"{path}: expected {want}, got {_preview(v)} (not a whole number)")
            else:
                errors.append(f"{path}: expected {want}, got {got} {_preview(v)}")
            return
    if "const" in schema and not _json_equal(v, schema["const"]):
        errors.append(f"{path}: must be {_preview(schema['const'])}, got {_preview(v)}")
        return
    if "enum" in schema:
        options = list(schema["enum"])
        if not any(_json_equal(v, o) for o in options):
            shown = ", ".join(_preview(o) for o in options[:20])
            if len(options) > 20:
                shown += f", ... ({len(options)} options)"
            errors.append(f"{path}: {_preview(v)} is not one of the allowed values [{shown}]")
            return

    # ---- numbers
    if _is_number(v):
        x = _num(v)   # exact: a huge JSON integer is compared as an int, never overflowed into a float
        if isinstance(x, float) and not math.isfinite(x):
            errors.append(f"{path}: must be a finite number, got {_preview(x)}")
        else:
            lo, hi = schema.get("minimum"), schema.get("maximum")
            xlo, xhi = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
            if lo is not None and x < _num(lo):
                errors.append(f"{path}: must be >= {_fmt_num(lo)}, got {_fmt_num(v)}")
            if hi is not None and x > _num(hi):
                errors.append(f"{path}: must be <= {_fmt_num(hi)}, got {_fmt_num(v)}")
            if _is_number(xlo) and x <= _num(xlo):
                errors.append(f"{path}: must be > {_fmt_num(xlo)}, got {_fmt_num(v)}")
            if _is_number(xhi) and x >= _num(xhi):
                errors.append(f"{path}: must be < {_fmt_num(xhi)}, got {_fmt_num(v)}")

    # ---- strings
    if isinstance(v, str):
        if "minLength" in schema and len(v) < int(schema["minLength"]):
            n = int(schema["minLength"])
            errors.append(f"{path}: must be a string of at least {n} character(s), got {len(v)}"
                          if v else f"{path}: must be a non-empty string")
        if "maxLength" in schema and len(v) > int(schema["maxLength"]):
            errors.append(f"{path}: must be at most {int(schema['maxLength'])} characters long, got {len(v)}")

    # ---- arrays
    if isinstance(v, (list, tuple)):
        if "minItems" in schema and len(v) < int(schema["minItems"]):
            errors.append(f"{path}: must have at least {int(schema['minItems'])} item(s), got {len(v)}")
        if "maxItems" in schema and len(v) > int(schema["maxItems"]):
            errors.append(f"{path}: must have at most {int(schema['maxItems'])} item(s), got {len(v)}")
        if schema.get("uniqueItems"):
            for i in range(len(v)):
                dup = next((j for j in range(i) if _json_equal(v[i], v[j])), None)
                if dup is not None:
                    errors.append(f"{path}: items must be unique, but [{i}] repeats [{dup}] ({_preview(v[i])})")
                    break
        items = schema.get("items")
        if items is not None and items is not True:
            for i, item in enumerate(v):
                _validate(item, items, _child(path, i), errors)

    # ---- objects
    if isinstance(v, Mapping):
        props: Mapping = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in v:
                sub = props.get(key)
                hint = f" ({_describe(sub)})" if isinstance(sub, Mapping) and sub else ""
                errors.append(f"{_child(path, key)}: required property is missing{hint}")
        if "minProperties" in schema and len(v) < int(schema["minProperties"]):
            errors.append(f"{path}: must have at least {int(schema['minProperties'])} propert(ies), got {len(v)}")
        if "maxProperties" in schema and len(v) > int(schema["maxProperties"]):
            errors.append(f"{path}: must have at most {int(schema['maxProperties'])} propert(ies), got {len(v)}")
        extra_schema = schema.get("additionalProperties", True)
        unknown = []
        for key, val in v.items():
            if not isinstance(key, str):
                errors.append(f"{path}: property names must be strings, got {_preview(key)}")
                continue
            if key in props:
                _validate(val, props[key], _child(path, key), errors)
            elif extra_schema is False:
                unknown.append(key)
            elif isinstance(extra_schema, Mapping):
                _validate(val, extra_schema, _child(path, key), errors)
        if unknown:
            allowed = ", ".join(sorted(map(str, props))) or "none"
            errors.append(f"{path}: unknown propert{'y' if len(unknown) == 1 else 'ies'} "
                          f"{', '.join(map(repr, unknown))} (allowed: {allowed})")

    # ---- combinators
    for sub in schema.get("allOf") or []:
        _validate(v, sub, path, errors)
    if "anyOf" in schema:
        _check_alternatives(v, list(schema["anyOf"]), path, errors, exactly_one=False)
    if "oneOf" in schema:
        _check_alternatives(v, list(schema["oneOf"]), path, errors, exactly_one=True)


def _check_alternatives(v: Any, options: list, path: str, errors: list[str], *, exactly_one: bool) -> None:
    results = []
    for sub in options:
        errs: list[str] = []
        _validate(v, sub, path, errs)
        results.append(errs)
    n_ok = sum(1 for e in results if not e)
    if n_ok == 0:
        # report the alternative whose structural type matches (if any): its errors are
        # the most useful; otherwise list what each alternative accepts
        typed = [i for i, sub in enumerate(options) if isinstance(sub, Mapping) and "type" in sub
                 and any(_matches_type(v, n) for n in (sub["type"] if isinstance(sub["type"], (list, tuple))
                                                         else [sub["type"]]))]
        if len(typed) == 1:
            errors.extend(results[typed[0]])
            return
        forms = "; ".join(f"({k + 1}) {_describe(sub)}" for k, sub in enumerate(options))
        detail = "; ".join(e[0] for e in results if e)
        errors.append(f"{path}: {_preview(v)} matches none of the allowed forms: {forms}"
                      + (f" [details: {detail}]" if detail else ""))
    elif exactly_one and n_ok > 1:
        idx = [str(i + 1) for i, e in enumerate(results) if not e]
        errors.append(f"{path}: {_preview(v)} matches more than one alternative ({', '.join(idx)}); "
                      f"it must match exactly one")
