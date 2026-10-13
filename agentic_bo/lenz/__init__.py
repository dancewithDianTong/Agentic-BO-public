"""lenz: the Bayesian backend of agentic BO (GP surrogates, acquisitions, trial log, commands).

    >>> from agentic_bo.lenz import Lenz
    >>> lz = Lenz([{"name": "x", "type": "real", "lower": 0, "upper": 1}], objectives="y")
    >>> cand = lz.suggest()["candidates"][0]["config"]
    >>> tid = lz.submit(cand)["trial_id"]
    >>> lz.observe({"y": 0.3}, trial_id=tid)["is_new_incumbent"]
    True

The same commands are available through :meth:`Lenz.call` (JSON envelopes) and the CLI
``python -m agentic_bo.lenz --study STUDY.json <command> ...``.
"""

from ..utils import LenzError
from .engine import ACQF_PARAMS, COMMANDS, Lenz
from .frame import Event, Frame, Shelf, Trial
from .gp import GP

__all__ = ["Lenz", "LenzError", "GP", "Frame", "Trial", "Event", "Shelf", "COMMANDS", "ACQF_PARAMS"]
