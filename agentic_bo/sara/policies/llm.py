"""Sara as an LLM: Claude drives the campaign through the Anthropic API (DESIGN.md §10).

:class:`ClaudeSaraPolicy` is the paper's metalevel policy: Claude reads the brief,
calls the agent tools (probes, reconfigurations, proposals, notes) and commits one
evaluation per step. It runs a manual, append-only tool-use loop over
``client.beta.messages.stream(...)``:

* **Frozen prefix.** ``system`` (``SYSTEM.md`` + ``LENZ_REF.md``, or
  ``SYSTEM_LLM_ONLY.md`` for the LLM-as-optimizer ablation) and ``tools`` (sorted by
  name, built once per campaign) never change, with a cache breakpoint on the last
  system block plus top-level automatic caching for the growing tail.
* **Append-only history.** Each assistant turn is appended exactly as returned
  (thinking, text, tool_use and compaction blocks), all tool results of one assistant
  turn go into one user message (results first, then text such as the step status
  line), and exogenous instructions arrive as ``{"role": "system"}`` messages after
  that user message. Nothing earlier is ever edited, so preserved thinking and the
  prompt cache stay valid. Content prepared for the next request is *staged* and
  committed to the history only when a request succeeds with a non-empty reply, so a
  rejected request (or an empty reply) never leaves a trace.
* **Robustness.** Tool inputs stream eagerly and are validated by
  :meth:`AgentSession.call`; unparseable JSON from the stream re-issues the request
  (capped); ``max_tokens`` (and ``model_context_window_exceeded``, which also cuts the
  output off) / ``refusal`` / ``pause_turn`` / ``end_turn`` are handled per the contract; an SDK that lacks a keyword argument gets it through
  ``extra_body`` instead; an API that rejects mid-conversation system messages gets
  the instruction as a ``<system-reminder>`` text block in the user turn.

The ``anthropic`` package is optional: it is imported only when the policy has to
build its own client (``client=None``) and, guarded, to recognize SDK error types.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ...utils import LenzError, jsonable
from ..tools import TOOLSETS
from .base import Policy, PolicyError

if TYPE_CHECKING:  # pragma: no cover - imports for type checkers only
    from ..session import AgentSession, ContextItem

__all__ = [
    "ClaudeSaraPolicy", "load_prompt", "estimate_cost", "DEFAULT_MODEL", "EFFORT_LEVELS", "THINKING_DISPLAYS",
    "FALLBACK_BETA", "COMPACTION_BETA", "COMPACTION_EDIT_TYPE", "PRICING_PER_MTOK", "PROMPTS_DIR",
    "MAX_STREAM_RETRIES", "MAX_PAUSE_CONTINUATIONS", "MIN_COMPACTION_TRIGGER",
]

DEFAULT_MODEL = "claude-opus-5-5"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
#: ``thinking.display`` values; ``None`` sends ``{"type": "adaptive"}`` alone (the API default display)
THINKING_DISPLAYS = ("summarized", "omitted", None)
FALLBACK_BETA = "server-side-fallback-2026-07-01"     # pairs with fallbacks="default"
COMPACTION_BETA = "compact-2026-01-12"
COMPACTION_EDIT_TYPE = "compact_20260112"
#: the API's minimum compaction trigger (input tokens); smaller values are a 400 on every request
MIN_COMPACTION_TRIGGER = 50_000
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
#: USD per million tokens: (input, output, cache read, 5-minute cache write)
PRICING_PER_MTOK: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-opus-5": (5.00, 25.00, 0.50, 6.25),
    "claude-opus-4-8": (5.00, 25.00, 0.50, 6.25),
}
#: re-issues of one request after the stream raised ValueError (unparseable tool-input JSON)
MAX_STREAM_RETRIES = 3
#: consecutive pause_turn re-sends within one step
MAX_PAUSE_CONTINUATIONS = 5
INSTALL_HINT = "pip install anthropic   (or: pip install 'agentic-bo[llm]')"

_TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
_UNEXPECTED_KWARG = re.compile(r"unexpected keyword argument '([A-Za-z_][A-Za-z0-9_]*)'")
_UNMOVABLE_KWARGS = frozenset({"model", "max_tokens", "messages", "extra_body", "extra_headers"})
_TEXT_PREVIEW = 300
#: stop reasons that cut the output off, possibly in the middle of a tool_use input (never executed)
_TRUNCATING_STOPS = ("max_tokens", "model_context_window_exceeded")
#: the 400 of a model without mid-conversation system messages ("role 'system' is not supported on this model")
_SYSTEM_ROLE_UNSUPPORTED = re.compile(
    r"(?:\brole\W{0,3}system|\bsystem\W{0,3}role)\b.{0,40}?(?:not\s+supported|unsupported)", re.S)


# =========================================================================== helpers


def load_prompt(name: str) -> str:
    """Text of a bundled prompt file (``SYSTEM.md``, ``LENZ_REF.md``, ``SYSTEM_LLM_ONLY.md``, ``COMPACTION.md``)."""
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def estimate_cost(model: str, input_tokens: int = 0, output_tokens: int = 0, cache_read_input_tokens: int = 0,
                  cache_creation_input_tokens: int = 0) -> float:
    """Estimated cost in USD of one request's token counts (5-minute cache writes).

    Unknown models are priced like :data:`DEFAULT_MODEL`.
    """
    p_in, p_out, p_read, p_write = PRICING_PER_MTOK.get(model, PRICING_PER_MTOK[DEFAULT_MODEL])
    return (input_tokens * p_in + output_tokens * p_out + cache_read_input_tokens * p_read
            + cache_creation_input_tokens * p_write) / 1e6


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Field ``name`` of an SDK object or a plain dict."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _plain(obj: Any) -> Any:
    """JSON-friendly copy of SDK objects (``to_dict``), dicts, lists and simple namespaces (for transcripts)."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    for attr in ("to_dict", "model_dump"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _plain(fn())
            except Exception:  # noqa: BLE001 - fall through to the generic forms
                pass
    if hasattr(obj, "__dict__"):
        return {k: _plain(v) for k, v in vars(obj).items() if not k.startswith("_") and not callable(v)}
    return repr(obj)


def _dump(obj: Any) -> str:
    """Compact deterministic JSON for tool results (agent-facing)."""
    try:
        return json.dumps(jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False)
    except (TypeError, ValueError):
        return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=repr)


def _short(text: Any, n: int = _TEXT_PREVIEW) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def _is_api_error(exc: BaseException) -> bool:
    """True for the SDK's typed API errors (never mistaken for unparseable stream JSON)."""
    try:
        import anthropic  # noqa: PLC0415 - optional dependency, imported only to recognize its errors
    except Exception:  # noqa: BLE001 - absent or broken install: fall back to duck typing
        anthropic = None
    if anthropic is not None:
        api_error = getattr(anthropic, "APIError", None)
        if isinstance(api_error, type) and isinstance(exc, api_error):
            return True
    return getattr(exc, "status_code", None) is not None


def _rejects_system_role(exc: BaseException) -> bool:
    """A 400 that refuses ``role: "system"`` messages (models without mid-conversation system messages).

    Matches the documented rejection (``role 'system' is not supported on this model``)
    and close variants only; every other 400 is a real request error and must surface.
    """
    if getattr(exc, "status_code", None) != 400:
        return False
    text = f"{exc} {getattr(exc, 'message', '')} {getattr(exc, 'body', '')}".lower()
    return _SYSTEM_ROLE_UNSUPPORTED.search(text) is not None


def _echo_content(content: Sequence[Any]) -> list[Any]:
    """The assistant content to append and act on, normally the content unchanged.

    After a server-side fallback switched models mid-output, the blocks the declined
    model produced before the last ``fallback`` marker must not be echoed except text
    and paired server-tool blocks (claude-api skill, "Echoing fallback turns back");
    its tool calls are not executed either.
    """
    blocks = list(content)
    marks = [i for i, b in enumerate(blocks) if _get(b, "type") == "fallback"]
    if not marks:
        return blocks
    boundary = marks[-1]
    result_ids = {_get(b, "tool_use_id") for b in blocks[:boundary]
                  if str(_get(b, "type", "")).endswith("_tool_result")}
    kept = []
    for i, b in enumerate(blocks):
        kind = _get(b, "type")
        if i < boundary and kind not in ("text", "fallback"):
            paired_use = kind == "server_tool_use" and _get(b, "id") in result_ids
            paired_result = str(kind).endswith("_tool_result")
            if not (paired_use or paired_result):
                continue
        kept.append(b)
    return kept


def _check_int(value: Any, name: str, lo: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < lo:
        raise LenzError(f"{name} must be an integer >= {lo}, got {value!r}")
    return value


# =========================================================================== policy


class ClaudeSaraPolicy(Policy):
    """Sara driven by Claude (DESIGN §10): an LLM metalevel policy with an append-only tool loop.

    Args:
        model: Claude model id.
        effort: ``output_config.effort`` (``low`` ... ``max``); the paper ran ``high``.
        max_tokens: output limit per request (thinking included).
        thinking_display: ``"summarized"`` (readable reasoning summaries, logged to
            ``H_t``), ``"omitted"`` or ``None`` (send ``{"type": "adaptive"}`` only).
            Thinking is always adaptive; it is never disabled or budgeted.
        use_fallbacks: server-side refusal fallbacks (``fallbacks="default"`` with the
            ``server-side-fallback-2026-07-01`` beta).
        use_compaction: server-side compaction (``compact-2026-01-12`` beta) with
            ``COMPACTION.md`` as the summarization instructions.
        compaction_trigger_tokens: input-token count that triggers compaction (the API
            minimum is 50,000).
        cache: prompt caching (breakpoint on the last system block + top-level automatic
            caching).
        eager_input_streaming: stream tool inputs eagerly (inputs are validated by the
            session either way).
        max_api_calls_per_step: requests per step before the step is left to the
            harness fallback.
        max_nudges: reminders sent when Claude ends its turn without evaluating, before
            the step is left to the harness fallback.
        client: an Anthropic client (anything with ``beta.messages.stream(**kwargs)``);
            ``None`` creates ``anthropic.Anthropic()`` at :meth:`start`.
        transcript_path: optional JSONL file receiving the session header (system and
            tools once), every request's newly appended messages and every response.
        toolset: ``"full"`` (Sara) or ``"llm_only"`` (LLM-as-optimizer ablation).
        system_prompt: replaces the text of ``SYSTEM.md`` / ``SYSTEM_LLM_ONLY.md``.
        verbose: print reasoning and tool calls to stderr.
    """

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "high", max_tokens: int = 32000,
                 thinking_display: str | None = "summarized", use_fallbacks: bool = True,
                 use_compaction: bool = True, compaction_trigger_tokens: int = 150_000, cache: bool = True,
                 eager_input_streaming: bool = True, max_api_calls_per_step: int = 40, max_nudges: int = 3,
                 client: Any = None, transcript_path: str | os.PathLike | None = None, toolset: str = "full",
                 system_prompt: str | None = None, verbose: bool = False):
        if not isinstance(model, str) or not model.strip():
            raise LenzError(f"model must be a non-empty string, got {model!r}")
        if effort not in EFFORT_LEVELS:
            raise LenzError(f"effort must be one of {EFFORT_LEVELS}, got {effort!r}")
        if thinking_display not in THINKING_DISPLAYS:
            raise LenzError(f"thinking_display must be one of {THINKING_DISPLAYS}, got {thinking_display!r}")
        if toolset not in TOOLSETS:
            raise LenzError(f"toolset must be one of {sorted(TOOLSETS)}, got {toolset!r}")
        if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
            raise LenzError("system_prompt must be a non-empty string or None")
        self.model = model.strip()
        self.effort = effort
        self.max_tokens = _check_int(max_tokens, "max_tokens", 1)
        self.thinking_display = thinking_display
        self.use_fallbacks = bool(use_fallbacks)
        self.use_compaction = bool(use_compaction)
        self.compaction_trigger_tokens = _check_int(compaction_trigger_tokens, "compaction_trigger_tokens",
                                                    MIN_COMPACTION_TRIGGER)
        self.cache = bool(cache)
        self.eager_input_streaming = bool(eager_input_streaming)
        self.max_api_calls_per_step = _check_int(max_api_calls_per_step, "max_api_calls_per_step", 1)
        self.max_nudges = _check_int(max_nudges, "max_nudges", 0)
        self.transcript_path = None if transcript_path is None else os.fspath(transcript_path)
        self.toolset = toolset
        self.name = "sara" if toolset == "full" else "llm_only"
        self.system_prompt = system_prompt
        self.verbose = bool(verbose)
        self._client = client
        self._session: AgentSession | None = None
        self._reset()

    def __repr__(self) -> str:
        return f"ClaudeSaraPolicy(name={self.name!r}, model={self.model!r}, effort={self.effort!r})"

    # ---- state -----------------------------------------------------------------------------
    def _reset(self) -> None:
        self._system: list[dict] = []
        self._tools: list[dict] = []
        self._thinking: dict = {"type": "adaptive"}
        self._context_management: dict | None = None
        self._messages: list[dict] = []        # committed history (append-only)
        self._staged_user: list[dict] = []     # content of the next user message
        self._staged_system: list[str] = []    # system messages to follow it
        self._pending_context: list[Any] = []
        self._opened = False                   # first user message staged
        self._system_role_ok = True
        self._moved: dict[str, str] = {}       # request kwarg -> "extra_body" | "extra_headers"
        self._log_seen = 0
        self._n_logged_messages = 0
        self._usage = self._fresh_usage()

    def _fresh_usage(self) -> dict[str, Any]:
        return {"model": self.model, "effort": self.effort, "api_calls": 0,
                **{k: 0 for k in _TOKEN_KEYS}, "cost_usd": 0.0, "stop_reasons": {}, "steps": 0, "nudges": 0,
                "stream_retries": 0, "pause_turns": 0, "max_tokens_stops": 0, "refusals": 0,
                "api_call_limit_hits": 0, "fallback_responses": 0, "compactions": 0,
                "system_role_fallback": False, "moved_kwargs": {}}

    @property
    def messages(self) -> list[dict]:
        """The committed conversation history (a shallow copy; the history itself is append-only)."""
        return list(self._messages)

    @property
    def system_blocks(self) -> list[dict]:
        """The frozen ``system`` parameter of this campaign."""
        return copy.deepcopy(self._system)

    @property
    def tool_params(self) -> list[dict]:
        """The frozen ``tools`` parameter of this campaign."""
        return copy.deepcopy(self._tools)

    def usage(self) -> dict[str, Any]:
        """API calls, token counts (``input_tokens`` is the uncached part), stop reasons and estimated cost."""
        out = copy.deepcopy(self._usage)
        out["cost_usd"] = round(float(out["cost_usd"]), 6)
        return out

    # ---- Policy hooks --------------------------------------------------------------------------
    def start(self, session: "AgentSession") -> None:
        """Create the client if needed and freeze ``system`` / ``tools`` (the brief opens the first request)."""
        self._session = session
        self._reset()
        if self._client is None:
            self._client = self._make_client()
        toolset = session.toolset
        try:
            if toolset == "llm_only":
                texts = [self.system_prompt or load_prompt("SYSTEM_LLM_ONLY.md")]
            else:
                texts = [self.system_prompt or load_prompt("SYSTEM.md"), load_prompt("LENZ_REF.md")]
            compaction = load_prompt("COMPACTION.md") if self.use_compaction else None
        except OSError as exc:
            raise PolicyError(f"cannot read the Sara prompt files in {PROMPTS_DIR}: {exc}") from exc
        self._system = [{"type": "text", "text": t} for t in texts]
        if self.cache:
            self._system[-1]["cache_control"] = {"type": "ephemeral"}
        tools = sorted(copy.deepcopy(session.tool_definitions()), key=lambda d: d["name"])
        if self.eager_input_streaming:
            for d in tools:
                d["eager_input_streaming"] = True
        self._tools = tools
        self._thinking = {"type": "adaptive"}
        if self.thinking_display is not None:
            self._thinking["display"] = self.thinking_display
        if compaction is not None:
            self._context_management = {"edits": [{
                "type": COMPACTION_EDIT_TYPE,
                "trigger": {"type": "input_tokens", "value": self.compaction_trigger_tokens},
                "instructions": compaction}]}
        self._log_seen = len(session.log)
        self._write_transcript({
            "event": "start", "policy": self.name, "problem": session.problem.name, "budget": session.budget,
            "toolset": toolset, "settings": self._settings(), "system": self._system, "tools": self._tools})

    def on_context(self, session: "AgentSession", item: "ContextItem") -> None:
        """Queue exogenous context; it is sent as a system message at the start of the next step."""
        self._pending_context.append(item)

    def step(self, session: "AgentSession") -> None:
        """One campaign step: request, execute tools, repeat until an evaluation (or stop) is committed.

        Returns without committing (leaving the step to the harness fallback) after
        ``max_nudges`` turns that end without an evaluation or after
        ``max_api_calls_per_step`` requests. Raises :class:`PolicyError` on a refusal
        or an API failure.
        """
        if self._session is not session or not self._system:
            raise PolicyError("ClaudeSaraPolicy.step() called before start() for this session")
        if session.is_done:
            return
        try:
            self._run_step(session)
        finally:
            self._log_seen = len(session.log)

    def finish(self, session: "AgentSession") -> None:
        """Close the transcript with the usage summary."""
        if self._session is session:
            self._write_transcript({"event": "finish", "t": session.t, "stopped": session.stopped,
                                    "usage": self.usage()})

    # ---- the step loop -------------------------------------------------------------------------
    def _run_step(self, session: "AgentSession") -> None:
        self._usage["steps"] += 1
        self._begin_turn(session)
        calls = nudges = pauses = 0
        while True:
            if calls >= self.max_api_calls_per_step:
                self._usage["api_call_limit_hits"] += 1
                self._note(session, f"API call limit reached ({self.max_api_calls_per_step} requests in step "
                                    f"{session.step}); the harness evaluates a fallback point")
                return
            response, content = self._send(session)
            calls += 1
            stop = _get(response, "stop_reason")
            if stop == "refusal":   # checked before the content is read: run nothing, log nothing of it
                self._usage["refusals"] += 1
                msg = self._refusal_message(response)
                self._note(session, msg)
                raise PolicyError(msg)
            self._log_content(session, content, stop)
            if not content:
                # Nothing was appended and the staged turn stays staged (see _send): re-send as is. If the
                # step is abandoned, the next step's opening joins that same user message, so a staged system
                # message still ends the request instead of being followed by a second user message.
                nudges += 1
                if nudges > self.max_nudges:
                    self._note(session, "empty responses; the harness evaluates a fallback point")
                    return
                continue
            tool_uses = [b for b in content if _get(b, "type") == "tool_use"]
            if stop in _TRUNCATING_STOPS:   # output cut off: a tool_use may hold a truncated input
                if stop == "max_tokens":
                    self._usage["max_tokens_stops"] += 1
                self._stage_truncated(tool_uses, stop)
                continue
            if stop == "pause_turn" and not tool_uses:
                self._usage["pause_turns"] += 1
                pauses += 1
                if pauses > MAX_PAUSE_CONTINUATIONS:
                    self._note(session, f"turn still paused after {MAX_PAUSE_CONTINUATIONS} continuations")
                    return
                continue  # re-send: the history ends with the paused assistant turn
            pauses = 0
            if tool_uses:
                self._staged_user.extend(self._execute(session, tool_uses))
                if session.step_committed or session.is_done:
                    return
                continue
            # end_turn (or any other stop) without a tool call
            if session.step_committed or session.is_done:
                return
            if nudges >= self.max_nudges:
                self._note(session, f"turn ended {nudges + 1} time(s) without an evaluation; the harness "
                                    f"evaluates a fallback point")
                return
            nudges += 1
            self._usage["nudges"] += 1
            nudge = self._nudge_text(session, last=nudges == self.max_nudges)
            self._staged_user.append({"type": "text", "text": nudge})

    def _begin_turn(self, session: "AgentSession") -> None:
        """Stage the user message that opens the step, then any new context as system messages."""
        texts: list[str] = []
        if not self._opened:
            self._opened = True
            texts.append(session.brief())
            texts.append(f"[harness] {session.status_line()}. The campaign starts now: plan your opening, "
                         f"then commit the first evaluation with `evaluate`.")
        else:
            texts.extend(self._harness_notes(session))
            texts.append(f"[harness] {session.status_line()}")
        self._staged_user.extend({"type": "text", "text": t} for t in texts)
        for item in self._pending_context:
            self._staged_system.append(self._context_text(session, item))
        self._pending_context = []

    def _harness_notes(self, session: "AgentSession") -> list[str]:
        """Fallback evaluations the harness committed since this policy last acted."""
        notes = []
        for e in session.log[self._log_seen:]:
            if e.get("tool") != "evaluate" or not e.get("forced"):
                continue
            args = e.get("args") if isinstance(e.get("args"), Mapping) else {}
            reason = args.get("reason") or "no evaluation was committed"
            if e.get("ok"):
                res = e.get("result") if isinstance(e.get("result"), Mapping) else {}
                keys = ("trial_id", "config", "observed", "feasible", "is_new_best")
                summary = {k: res[k] for k in keys if k in res} or res
                notes.append(f"[harness] Step {e.get('step')} ended without an evaluation from you ({reason}), "
                             f"so the harness evaluated a fallback point: {_dump(summary)}")
            else:
                notes.append(f"[harness] Step {e.get('step')}: the harness fallback evaluation failed: "
                             f"{e.get('error')}")
        return notes

    @staticmethod
    def _context_text(session: "AgentSession", item: Any) -> str:
        source = _get(item, "source") or "operator"
        t = _get(item, "t", session.t)
        return (f"New context from {source} (received after {t} of {session.budget} evaluations):\n\n"
                f"{str(_get(item, 'text', '')).strip()}")

    @staticmethod
    def _nudge_text(session: "AgentSession", last: bool) -> str:
        options = "`evaluate` with a config and a rationale"
        if session.config.allow_early_stop:
            options += " (or `stop` to end the campaign)"
        text = (f"[harness] Step {session.step} is still open: no evaluation has been committed. Call {options}. "
                f"Free tools remain available, but only an evaluation completes the step.")
        if last:
            text += " If this turn also ends without one, the harness evaluates a fallback point for you."
        return text

    def _stage_truncated(self, tool_uses: list[Any], stop: str) -> None:
        """Answer a response cut off by ``max_tokens`` or the context window; its tool calls are not run."""
        limit = (f"the max_tokens limit ({self.max_tokens})" if stop == "max_tokens"
                 else "the limit of the model's context window")
        if tool_uses:
            msg = (f"Not executed: the response reached {limit} while this call was being written, so its input "
                   f"may be truncated. Re-issue the call with shorter reasoning or arguments.")
            self._staged_user.extend({"type": "tool_result", "tool_use_id": _get(b, "id"), "content": msg,
                                      "is_error": True} for b in tool_uses)
        else:
            self._staged_user.append({"type": "text", "text": (
                f"[harness] Your previous response was cut off at {limit}. Continue more concisely and call "
                f"a tool.")})

    def _execute(self, session: "AgentSession", tool_uses: list[Any]) -> list[dict]:
        """Run the tool calls of one assistant message in order; one tool_result per call."""
        results = []
        for b in tool_uses:
            name, raw, block_id = _get(b, "name"), _get(b, "input"), _get(b, "id")
            args = dict(raw) if isinstance(raw, Mapping) else raw
            envelope = session.call(name, args)
            results.append(self._tool_result(block_id, envelope, raw))
            if self.verbose:
                status = "ok" if envelope.get("ok") else f"error: {_short(envelope.get('error'), 160)}"
                self._say(f"step {session.step} | {name}({_short(_dump(args), 160)}) -> {status}")
        return results

    @staticmethod
    def _tool_result(block_id: Any, envelope: Mapping[str, Any], raw_input: Any) -> dict:
        if envelope.get("ok"):
            return {"type": "tool_result", "tool_use_id": block_id, "content": _dump(envelope.get("result"))}
        parts = [str(envelope.get("error") or "the tool call failed")]
        if envelope.get("warning"):
            parts.append(str(envelope["warning"]))
        if envelope.get("fallback") is not None:
            parts.append("Fallback evaluation: " + _dump(envelope["fallback"]))
        text = "\n".join(parts)
        if not isinstance(raw_input, Mapping):  # eager streaming delivered input that is not a JSON object
            text = json.dumps({"INVALID_JSON": raw_input if isinstance(raw_input, str) else repr(raw_input),
                               "error": text}, ensure_ascii=False)
        return {"type": "tool_result", "tool_use_id": block_id, "content": text, "is_error": True}

    @staticmethod
    def _refusal_message(response: Any) -> str:
        details = _get(response, "stop_details")
        category = _get(details, "category") if details is not None else None
        explanation = _get(details, "explanation") if details is not None else None
        msg = f"Claude declined to continue (stop_reason 'refusal', category {category!r})"
        if explanation:
            msg += f": {_short(explanation, 400)}"
        return msg

    # ---- requests ------------------------------------------------------------------------------
    def _request_kwargs(self, messages: list[dict]) -> dict[str, Any]:
        """Every request is built here (and only here)."""
        kw: dict[str, Any] = {
            "model": self.model, "max_tokens": self.max_tokens, "system": self._system, "tools": self._tools,
            "tool_choice": {"type": "auto"}, "thinking": self._thinking,
            "output_config": {"effort": self.effort}, "messages": messages,
        }
        betas = []
        if self.use_fallbacks:
            betas.append(FALLBACK_BETA)
            kw["fallbacks"] = "default"
        if self._context_management is not None:
            betas.append(COMPACTION_BETA)
            kw["context_management"] = self._context_management
        if betas:
            kw["betas"] = betas
        if self.cache:
            kw["cache_control"] = {"type": "ephemeral"}
        for key in self._moved:
            if key in kw:
                self._move_kwarg(kw, key)
        return kw

    def _move_kwarg(self, kw: dict[str, Any], key: str) -> None:
        value = kw.pop(key)
        if key == "betas":
            headers = dict(kw.get("extra_headers") or {})
            headers["anthropic-beta"] = ",".join(value)
            kw["extra_headers"] = headers
            self._moved[key] = "extra_headers"
        else:
            body = dict(kw.get("extra_body") or {})
            body[key] = value
            kw["extra_body"] = body
            self._moved[key] = "extra_body"

    def _staged_messages(self) -> list[dict]:
        """The messages this request appends: the staged user turn, then at most one system message.

        A ``role: "system"`` message must follow a user message and be last or followed
        by an assistant turn, so several context items due before the same step share
        one system message (in arrival order) rather than forming a run of them.
        """
        user = list(self._staged_user)
        system = list(self._staged_system)
        if system and not self._system_role_ok:
            user.extend({"type": "text", "text": f"<system-reminder>\n{t}\n</system-reminder>"} for t in system)
            system = []
        out: list[dict] = []
        if user:
            out.append({"role": "user", "content": user})
        if system:
            out.append({"role": "system", "content": "\n\n".join(system)})
        return out

    def _send(self, session: "AgentSession") -> tuple[Any, list[Any]]:
        """Send the history plus the staged turn; commit both with the reply when the reply has content."""
        stream_failures = 0
        while True:
            new = self._staged_messages()
            kw = self._request_kwargs(self._messages + new)
            try:
                with self._client.beta.messages.stream(**kw) as stream:
                    response = stream.get_final_message()
            except TypeError as exc:
                key = self._unexpected_kwarg(exc, kw)
                if key is None:
                    raise PolicyError(f"the Anthropic client rejected the request: {exc}") from exc
                self._move_kwarg(kw, key)   # remembered: later requests move it too
                self._usage["moved_kwargs"] = dict(sorted(self._moved.items()))
                self._write_transcript({"event": "retry", "reason": f"SDK lacks keyword {key!r}",
                                        "moved_to": self._moved[key]})
                continue
            except Exception as exc:  # noqa: BLE001 - classified below; never escapes as a non-PolicyError
                if _is_api_error(exc):
                    has_system = any(m["role"] == "system" for m in new)
                    if has_system and self._system_role_ok and _rejects_system_role(exc):
                        self._system_role_ok = False
                        self._usage["system_role_fallback"] = True
                        self._write_transcript({"event": "retry", "reason": "role 'system' rejected",
                                                "error": _short(exc)})
                        continue
                    raise PolicyError(f"Claude API error ({type(exc).__name__}): {_short(exc, 600)}") from exc
                if isinstance(exc, ValueError):  # unparseable tool-input JSON from an eager stream
                    stream_failures += 1
                    self._usage["stream_retries"] += 1
                    self._write_transcript({"event": "retry", "reason": "stream ValueError", "error": _short(exc)})
                    if stream_failures > MAX_STREAM_RETRIES:
                        raise PolicyError(f"the response stream could not be parsed after {stream_failures} "
                                          f"attempts: {_short(exc)}") from exc
                    continue
                raise PolicyError(f"calling the Claude API failed ({type(exc).__name__}): {_short(exc, 600)}") \
                    from exc
            content = _echo_content(_get(response, "content") or [])
            if content:   # an empty reply commits nothing: its request is re-sent (or extended) as staged
                self._messages.extend(new)
                self._messages.append({"role": "assistant", "content": content})
                self._staged_user = []
                self._staged_system = []
            self._account(response)
            self._log_exchange(session, kw, response)
            return response, content

    def _unexpected_kwarg(self, exc: TypeError, kw: Mapping[str, Any]) -> str | None:
        """The request keyword an older SDK does not accept, if it can move to ``extra_body``."""
        m = _UNEXPECTED_KWARG.search(str(exc))
        if m is None:
            return None
        key = m.group(1)
        if key not in kw or key in _UNMOVABLE_KWARGS or key in self._moved:
            return None
        return key

    def _make_client(self) -> Any:
        try:
            import anthropic  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise PolicyError(f"ClaudeSaraPolicy needs the 'anthropic' package, which is not installed: "
                              f"{INSTALL_HINT}") from exc
        try:
            return anthropic.Anthropic()
        except Exception as exc:  # noqa: BLE001 - e.g. no credentials configured
            raise PolicyError(f"could not create the Anthropic client ({type(exc).__name__}: {_short(exc)}); "
                              f"set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN, or run `ant auth login`) "
                              f"or pass client=...") from exc

    # ---- accounting and traces -----------------------------------------------------------------
    def _account(self, response: Any) -> None:
        u = self._usage
        u["api_calls"] += 1
        stop = str(_get(response, "stop_reason") or "unknown")
        u["stop_reasons"][stop] = u["stop_reasons"].get(stop, 0) + 1
        usage = _get(response, "usage")
        if usage is None:
            return
        served_by = _get(response, "model") or self.model
        iterations = _get(usage, "iterations") or []
        # usage.iterations itemizes compaction and fallback attempts; the top level covers one attempt only
        for entry in (list(iterations) if iterations else [usage]):
            kind = _get(entry, "type")
            if kind == "compaction":
                u["compactions"] += 1
            elif kind == "fallback_message":
                u["fallback_responses"] += 1
            model = _get(entry, "model") or (served_by if kind == "fallback_message" else self.model)
            counts = {}
            for key in _TOKEN_KEYS:
                v = _get(entry, key)
                counts[key] = int(v) if isinstance(v, (int, float)) and math.isfinite(v) else 0
                u[key] += counts[key]
            u["cost_usd"] += estimate_cost(str(model), **counts)

    def _log_content(self, session: "AgentSession", content: list[Any], stop: Any) -> None:
        """Append the turn's summarized thinking, text and compaction summaries to ``H_t``."""
        call = self._usage["api_calls"]
        for b in content:
            kind = _get(b, "type")
            if kind == "thinking":
                text = _get(b, "thinking")
            elif kind == "text":
                text = _get(b, "text")
            elif kind == "compaction":
                text = _get(b, "content")
            else:
                continue
            if isinstance(text, str) and text.strip():
                session.log_reasoning(text, kind=str(kind), stop_reason=stop, api_call=call)
                if self.verbose:
                    self._say(f"step {session.step} | {kind}: {_short(text)}")

    def _note(self, session: "AgentSession", text: str) -> None:
        session.log_reasoning(text, kind="policy")
        self._write_transcript({"event": "note", "step": session.step, "t": session.t, "text": text})
        if self.verbose:
            self._say(f"step {session.step} | {text}")

    def _log_exchange(self, session: "AgentSession", kw: Mapping[str, Any], response: Any) -> None:
        if not self.transcript_path:
            return
        msgs = kw["messages"]
        params = {k: v for k, v in kw.items() if k not in ("system", "tools", "messages")}
        self._write_transcript({"event": "request", "step": session.step, "t": session.t,
                                "n_messages": len(msgs), "new_messages": msgs[self._n_logged_messages:],
                                "params": params})
        self._write_transcript({"event": "response", "step": session.step, "t": session.t,
                                "stop_reason": _get(response, "stop_reason"), "message": response})
        self._n_logged_messages = len(self._messages)

    def _write_transcript(self, event: Mapping[str, Any]) -> None:
        if not self.transcript_path:
            return
        path = self.transcript_path
        try:
            folder = os.path.dirname(os.path.abspath(path))
            os.makedirs(folder, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(_plain(event), sort_keys=True, ensure_ascii=False, default=repr) + "\n")
        except OSError as exc:  # a transcript must never stop the campaign
            print(f"[sara] transcript disabled: {exc}", file=sys.stderr, flush=True)
            self.transcript_path = None

    def _settings(self) -> dict[str, Any]:
        return {"model": self.model, "effort": self.effort, "max_tokens": self.max_tokens,
                "thinking": dict(self._thinking), "use_fallbacks": self.use_fallbacks,
                "use_compaction": self.use_compaction, "compaction_trigger_tokens": self.compaction_trigger_tokens,
                "cache": self.cache, "eager_input_streaming": self.eager_input_streaming,
                "max_api_calls_per_step": self.max_api_calls_per_step, "max_nudges": self.max_nudges}

    def _say(self, text: str) -> None:
        if self.verbose:
            print(f"[sara] {text}", file=sys.stderr, flush=True)
