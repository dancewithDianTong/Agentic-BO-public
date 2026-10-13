"""Command-line interface to a lenz study (DESIGN.md §7.3, paper App. H).

::

    python -m agentic_bo.lenz --study s.json create --space '[{"name": "x", "type": "real", "lower": 0, "upper": 1}]' --objectives y
    python -m agentic_bo.lenz --study s.json suggest --q 2
    python -m agentic_bo.lenz --study s.json submit --config '{"x": 0.25}'
    python -m agentic_bo.lenz --study s.json observe --trial-id 0 --metrics '{"y": 1.3}'
    python -m agentic_bo.lenz --study s.json status

Every invocation prints exactly one JSON line, the command envelope
``{"ok": true, "command", "result"}`` or ``{"ok": false, "command", "error": {"type", "message"}}``,
and exits with status 0 on success and 1 otherwise (usage errors included; no tracebacks).
The study file is authoritative: each call loads it, runs one command and saves it back
(the event log records every call, probes included). JSON-valued flags also accept
``@path`` to read the JSON from a file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Sequence

from ..utils import LenzError, dumps
from .engine import Lenz

__all__ = ["main", "build_parser"]


class _UsageError(Exception):
    """Raised instead of argparse's print-and-exit on invalid command lines."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise _UsageError(f"{self.prog}: {message}")


# --------------------------------------------------------------------------- value parsing


def _load_text(text: str, flag: str) -> str:
    if text.startswith("@"):
        try:
            with open(text[1:], encoding="utf-8") as fh:
                return fh.read()
        except OSError as exc:
            raise LenzError(f"{flag}: cannot read {text[1:]!r}: {exc}") from None
    return text


def _json(flag: str) -> Callable[[str], Any]:
    def parse(text: str) -> Any:
        try:
            return json.loads(_load_text(text, flag))
        except json.JSONDecodeError as exc:
            raise LenzError(f"{flag}: invalid JSON ({exc.msg} at position {exc.pos})") from None
    return parse


def _json_or_word(flag: str) -> Callable[[str], Any]:
    """JSON if it parses, else the bare word (metric names, ``incumbent``, ``full``)."""
    def parse(text: str) -> Any:
        body = _load_text(text, flag)
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            stripped = body.strip()
            if stripped and stripped[0] in "[{\"":
                raise LenzError(f"{flag}: invalid JSON {stripped[:80]!r}") from None
            return stripped
    return parse


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _name_or_names(text: str) -> str | list[str]:
    """``a`` -> ``"a"``; ``a,b`` -> ``["a", "b"]`` (e.g. ``--status pending,completed``)."""
    names = _names(text)
    return names if len(names) > 1 else text


# (flag, argument name, converter, required); converters run after argparse so that bad
# JSON becomes a LenzError envelope rather than a usage error
_Spec = tuple[str, str, Callable[[str], Any] | None, bool]
_COMMANDS: dict[str, tuple[str, list[_Spec]]] = {
    "create": ("create a new study file", [
        ("--space", "space", _json("--space"), True),
        ("--objectives", "objectives", _json_or_word("--objectives"), True),
        ("--constraints", "constraints", _json("--constraints"), False),
        ("--acqf", "acqf", None, False),
        ("--acqf-params", "acqf_params", _json("--acqf-params"), False),
        ("--bounds", "bounds", _json_or_word("--bounds"), False),
        ("--model", "model", _json("--model"), False),
        ("--seed", "seed", int, False),
    ]),
    "suggest": ("propose candidates (records nothing)", [
        ("--q", "q", int, False),
        ("--around", "around", _json_or_word("--around"), False),
        ("--radius", "radius", float, False),
        ("--bounds", "bounds", _json_or_word("--bounds"), False),
        ("--acqf", "acqf", None, False),
        ("--acqf-params", "acqf_params", _json("--acqf-params"), False),
    ]),
    "submit": ("record a pending trial", [
        ("--config", "config", _json("--config"), True),
        ("--source", "source", None, False),
        ("--meta", "meta", _json("--meta"), False),
    ]),
    "observe": ("attach metrics to a pending trial (or log a new evaluation by config)", [
        ("--trial-id", "trial_id", int, False),
        ("--config", "config", _json("--config"), False),
        ("--metrics", "metrics", _json("--metrics"), True),
    ]),
    "abandon": ("abandon a pending trial", [("--trial-id", "trial_id", int, True)]),
    "set-bounds": ("set the active region (JSON or 'full')", [
        ("--bounds", "bounds", _json_or_word("--bounds"), True)]),
    "set-acqf": ("choose the acquisition function", [
        ("--acqf", "acqf", None, True),
        ("--beta", "beta", float, False),
        ("--params", "params", _json("--params"), False),
    ]),
    "set-objectives": ("replace the objectives", [
        ("--objectives", "objectives", _json_or_word("--objectives"), True)]),
    "set-constraints": ("replace the outcome constraints ('[]' removes them)", [
        ("--constraints", "constraints", _json("--constraints"), True)]),
    "diagnostics": ("surrogate diagnostics per metric", [("--metric", "metric", None, False)]),
    "predict": ("posterior predictions at configs", [("--configs", "configs", _json("--configs"), True)]),
    "score": ("acquisition values at configs", [
        ("--configs", "configs", _json("--configs"), True),
        ("--acqf", "acqfs", _names, False),
    ]),
    "incumbent": ("best feasible completed trial", []),
    "pareto": ("Pareto front and hypervolume", []),
    "trials": ("the trial log (--last: k most recent, before sorting; --limit: first k after sorting)", [
        ("--last", "last", int, False),
        ("--status", "status", _name_or_names, False),
        ("--sort-by", "sort_by", None, False),
        ("--limit", "limit", int, False),
    ]),
    "status": ("study overview", []),
}


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser (``--study`` may come before or after the command)."""
    parser = _Parser(prog="lenz", description="lenz: Bayesian optimization backend for agentic BO "
                                              "(one JSON envelope per call).")
    parser.add_argument("--study", help="path of the study JSON file")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, (help_text, specs) in _COMMANDS.items():
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("--study", default=argparse.SUPPRESS, help="path of the study JSON file")
        for flag, _, _, required in specs:
            p.add_argument(flag, required=required)
        if name == "create":
            p.add_argument("--overwrite", action="store_true", help="replace an existing study file")
    return parser


def _collect_args(command: str, ns: argparse.Namespace) -> dict:
    out: dict[str, Any] = {}
    for flag, dest, conv, _ in _COMMANDS[command][1]:
        raw = getattr(ns, flag.lstrip("-").replace("-", "_"), None)
        if raw is None:
            continue
        if conv is None:
            out[dest] = raw
            continue
        try:
            out[dest] = conv(raw)
        except ValueError as exc:  # int / float conversion (LenzError is a ValueError too)
            if isinstance(exc, LenzError):
                raise
            raise LenzError(f"{flag}: invalid value {raw!r}") from None
    return out


def _envelope_error(command: str | None, exc: BaseException) -> dict:
    return {"ok": False, "command": command,
            "error": {"type": type(exc).__name__, "message": str(exc) or type(exc).__name__}}


def _run(argv: Sequence[str] | None) -> dict:
    parser = build_parser()
    try:
        ns = parser.parse_args(argv)
    except _UsageError as exc:
        return {"ok": False, "command": None, "error": {"type": "UsageError", "message": str(exc)}}
    command = ns.command
    if command is None:
        return {"ok": False, "command": None,
                "error": {"type": "UsageError", "message": "no command given; run with --help for the list"}}
    canonical = command.replace("-", "_")
    if not ns.study:
        return {"ok": False, "command": canonical,
                "error": {"type": "UsageError", "message": "--study STUDY.json is required"}}
    path = ns.study
    try:
        args = _collect_args(command, ns)
        if command == "create":
            if os.path.exists(path) and not ns.overwrite:
                raise LenzError(f"study file {path!r} already exists (pass --overwrite to replace it)")
            lenz = Lenz.create(**args)
            lenz.save(path)
            result = {"study": path, "space": lenz.space.to_spec(), "config": lenz.shelf.to_dict(),
                      "seed": lenz.seed, "n_trials": 0}
            return {"ok": True, "command": "create", "result": result}
        lenz = Lenz.load(path)
    except Exception as exc:  # usage-level failures: report, never raise
        return _envelope_error(canonical, exc)
    envelope = lenz.call(canonical, args)
    try:
        lenz.save(path)
    except Exception as exc:
        return _envelope_error(canonical, exc)
    return envelope


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI command; prints one JSON line and returns the exit code (0 ok, 1 error)."""
    try:
        envelope = _run(argv)
    except SystemExit as exc:  # --help / --version from argparse
        return int(exc.code or 0) if isinstance(exc.code, int) else 0
    except Exception as exc:  # pragma: no cover - last-resort guard: never print a traceback
        envelope = _envelope_error(None, exc)
    try:
        line = dumps(envelope)
    except (TypeError, ValueError) as exc:  # pragma: no cover - results are jsonable by construction
        line = dumps(_envelope_error(envelope.get("command"), exc))
        envelope = {"ok": False}
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
    return 0 if envelope.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
