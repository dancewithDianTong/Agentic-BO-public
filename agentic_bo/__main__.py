"""``python -m agentic_bo``: entry point that dispatches to the package's command-line tools.

::

    python -m agentic_bo lenz --study s.json status        # the lenz CLI (same as python -m agentic_bo.lenz)
    python -m agentic_bo experiment run --problems branin --out results/   # benchmarks
    python -m agentic_bo problems                           # registered problems and policies
"""

from __future__ import annotations

import sys
from typing import Sequence

_USAGE = """usage: python -m agentic_bo COMMAND [ARGS...]

commands:
  lenz ARGS...         lenz backend CLI (python -m agentic_bo.lenz --help)
  experiment ARGS...   benchmarks: run | report | list (python -m agentic_bo.experiment --help)
  problems             list registered problems and policies
  version              print the package version
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch ``argv[0]`` to a sub-tool; returns the exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        sys.stdout.write(_USAGE)
        return 0 if args else 2
    command, rest = args[0], args[1:]
    if command == "lenz":
        from agentic_bo.lenz.cli import main as lenz_main
        return lenz_main(rest)
    if command == "experiment":
        from agentic_bo.experiment import main as experiment_main
        return experiment_main(rest)
    if command == "problems":
        from agentic_bo.experiment import main as experiment_main
        return experiment_main(["list"])
    if command == "version":
        from agentic_bo import __version__
        print(__version__)
        return 0
    sys.stderr.write(f"unknown command {command!r}\n\n{_USAGE}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
