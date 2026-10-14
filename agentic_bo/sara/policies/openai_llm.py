"""Sara driven by an OpenAI-compatible chat model (OpenAI API, or a self-hosted server).

:class:`OpenAISaraPolicy` runs the same metalevel loop as
:class:`~agentic_bo.sara.policies.llm.ClaudeSaraPolicy` (same prompts, same tools,
one committed evaluation per step) over the **Chat Completions** API
(``client.chat.completions.create``). Because vLLM, SGLang, Ollama and similar servers
expose that API, the same class drives e.g. a Qwen model on your own machine: pass
``base_url="http://localhost:8000/v1"``.

* **Frozen prefix.** One leading ``system`` message (``SYSTEM.md`` + ``LENZ_REF.md``, or
  ``SYSTEM_LLM_ONLY.md``) and the tools (``{"type": "function", ...}``, sorted by name)
  never change, which keeps prefix caching effective on OpenAI and on vLLM.
* **Append-only history.** Assistant messages are appended as returned (content and
  ``tool_calls``); every tool call is answered by its own ``{"role": "tool"}`` message,
  in call order, followed by one ``user`` message with harness text (step status,
  reminders). Content for the next request is staged and committed only when a request
  returns a non-empty reply.
* **Exogenous instructions** go out as a ``system`` message (``context_as_system=True``,
  the default for the OpenAI API) or as a ``<system-reminder>`` block inside the user
  message (the default with ``base_url``, since many open-model chat templates accept a
  system message only at the start).
* **Robustness.** Tool arguments arrive as a JSON string; unparseable arguments are
  answered with an ``INVALID_JSON`` error instead of being run. ``finish_reason ==
  "length"`` leaves every tool call of that message unexecuted (its arguments may be
  cut off). A refusal (``message.refusal`` or ``finish_reason == "content_filter"``)
  raises :class:`PolicyError`, so the harness continues with fallback evaluations.
* **Reasoning traces.** Reasoning text some servers return (``reasoning_content`` on
  vLLM/SGLang, ``reasoning`` elsewhere) is logged to ``H_t`` but not sent back. OpenAI's
  Chat Completions API does not return reasoning text, so only visible text is logged.

There is no server-side compaction here: very long campaigns can outgrow the model's
context window (``finish_reason == "length"`` or a 400). Keep budgets moderate, or
use a model with a long context.

The ``openai`` package is optional and imported only when the policy builds its own
client (``client=None``).
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ...utils import LenzError
from ..tools import TOOLSETS
from .base import Policy, PolicyError
from .llm import PROMPTS_DIR, _check_int, _dump, _get, _plain, _short, load_prompt

if TYPE_CHECKING:  # pragma: no cover - imports for type checkers only
    from ..session import AgentSession, ContextItem

__all__ = ["OpenAISaraPolicy", "to_openai_tools", "INSTALL_HINT", "LOCAL_API_KEY"]

INSTALL_HINT = "pip install openai   (or: pip install 'agentic-bo[openai]')"
#: placeholder key for self-hosted servers that ignore authentication (vLLM / Ollama convention)
LOCAL_API_KEY = "EMPTY"
_USAGE_TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
                     "reasoning_tokens")
_REASONING_FIELDS = ("reasoning_content", "reasoning")


def to_openai_tools(definitions: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Agent tool definitions (``name``/``description``/``input_schema``) as Chat Completions tools.

    Sorted by name; ``strict`` mode is not used because the Sara schemas contain
    optional fields and null unions that strict mode rejects (the session validates
    every call anyway).
    """
    tools = []
    for d in sorted(definitions, key=lambda d: d["name"]):
        tools.append({"type": "function", "function": {
            "name": d["name"], "description": d.get("description", ""),
            "parameters": copy.deepcopy(d.get("input_schema") or {"type": "object", "properties": {}})}})
    return tools


class OpenAISaraPolicy(Policy):
    """Sara with an OpenAI-compatible chat model and the agent tools.

    Args:
        model: model name on the endpoint (required; e.g. an OpenAI model id, or the name
            a vLLM/Ollama server serves, such as ``"Qwen/Qwen3-32B"``).
        base_url: API base URL; ``None`` uses the OpenAI API (or ``OPENAI_BASE_URL``).
        api_key: API key; ``None`` reads ``OPENAI_API_KEY``, falling back to a placeholder
            when ``base_url`` points at a self-hosted server.
        reasoning_effort: sent as ``reasoning_effort`` for reasoning models (omitted when
            ``None``; non-reasoning models may reject it).
        max_completion_tokens: output limit per request (omitted when ``None``).
        temperature: sampling temperature (omitted when ``None``; reasoning models reject it).
        parallel_tool_calls: sent when not ``None``.
        extra_body: extra JSON fields for the request, e.g. ``{"chat_template_kwargs":
            {"enable_thinking": True}}`` for Qwen3 on vLLM.
        context_as_system: deliver new instructions as ``system`` messages (``True``) or as
            ``<system-reminder>`` blocks in the user message (``False``); ``None`` picks
            ``True`` for the OpenAI API and ``False`` with a ``base_url``.
        pricing: optional USD per million tokens ``(input, output, cached input)`` for the
            cost estimate; without it ``cost_usd`` stays 0.
        max_api_calls_per_step, max_nudges, transcript_path, toolset, system_prompt,
        verbose: as in :class:`~agentic_bo.sara.policies.llm.ClaudeSaraPolicy`.
        client: an OpenAI client (anything with ``chat.completions.create(**kwargs)``);
            ``None`` builds ``openai.OpenAI(base_url=..., api_key=...)`` at :meth:`start`.
        name: label in results (default ``"openai"``, or ``"openai_llm_only"``).
    """

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 reasoning_effort: str | None = None, max_completion_tokens: int | None = None,
                 temperature: float | None = None, parallel_tool_calls: bool | None = None,
                 extra_body: Mapping[str, Any] | None = None, context_as_system: bool | None = None,
                 pricing: Sequence[float] | None = None, max_api_calls_per_step: int = 40, max_nudges: int = 3,
                 client: Any = None, transcript_path: str | os.PathLike | None = None, toolset: str = "full",
                 system_prompt: str | None = None, name: str | None = None, verbose: bool = False):
        if not isinstance(model, str) or not model.strip():
            raise LenzError("OpenAISaraPolicy needs model=..., the model name on the endpoint "
                            "(e.g. an OpenAI model id, or the model a vLLM/Ollama server serves)")
        if toolset not in TOOLSETS:
            raise LenzError(f"toolset must be one of {sorted(TOOLSETS)}, got {toolset!r}")
        if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
            raise LenzError("system_prompt must be a non-empty string or None")
        if reasoning_effort is not None and (not isinstance(reasoning_effort, str) or not reasoning_effort.strip()):
            raise LenzError(f"reasoning_effort must be a non-empty string or None, got {reasoning_effort!r}")
        if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                                        or not math.isfinite(temperature) or temperature < 0):
            raise LenzError(f"temperature must be a finite number >= 0 or None, got {temperature!r}")
        if extra_body is not None and not isinstance(extra_body, Mapping):
            raise LenzError(f"extra_body must be a mapping or None, got {type(extra_body).__name__}")
        if pricing is not None:
            prices = list(pricing) if isinstance(pricing, (list, tuple)) else None
            if (prices is None or len(prices) != 3 or any(isinstance(p, bool) or not isinstance(p, (int, float))
                                                          or not math.isfinite(p) or p < 0 for p in prices)):
                raise LenzError(f"pricing must be (input, output, cached input) USD per million tokens, got {pricing!r}")
        self.model = model.strip()
        self.base_url = base_url
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = (None if max_completion_tokens is None
                                      else _check_int(max_completion_tokens, "max_completion_tokens", 1))
        self.temperature = None if temperature is None else float(temperature)
        self.parallel_tool_calls = None if parallel_tool_calls is None else bool(parallel_tool_calls)
        self.extra_body = None if extra_body is None else copy.deepcopy(dict(extra_body))
        self.context_as_system = (base_url is None) if context_as_system is None else bool(context_as_system)
        self.pricing = None if pricing is None else tuple(float(p) for p in pricing)
        self.max_api_calls_per_step = _check_int(max_api_calls_per_step, "max_api_calls_per_step", 1)
        self.max_nudges = _check_int(max_nudges, "max_nudges", 0)
        self.transcript_path = None if transcript_path is None else os.fspath(transcript_path)
        self.toolset = toolset
        self.name = name or ("openai" if toolset == "full" else "openai_llm_only")
        self.system_prompt = system_prompt
        self.verbose = bool(verbose)
        self._client = client
        self._session: AgentSession | None = None
        self._reset()

    def __repr__(self) -> str:
        return f"OpenAISaraPolicy(name={self.name!r}, model={self.model!r}, base_url={self.base_url!r})"

    # ---- state -----------------------------------------------------------------------------
    def _reset(self) -> None:
        self._system_text = ""
        self._tools: list[dict] = []
        self._messages: list[dict] = []          # committed history (append-only), system message first
        self._staged_tool: list[dict] = []       # {"role": "tool"} answers to the last assistant message
        self._staged_text: list[str] = []        # text of the next user message
        self._staged_system: list[str] = []      # instructions sent as one system message after it
        self._pending_context: list[Any] = []
        self._opened = False
        self._log_seen = 0
        self._n_logged_messages = 0
        self._usage = {"model": self.model, "base_url": self.base_url, "api_calls": 0,
                       **{k: 0 for k in _USAGE_TOKEN_KEYS}, "cost_usd": 0.0, "priced": self.pricing is not None,
                       "finish_reasons": {}, "steps": 0, "nudges": 0, "truncated": 0, "refusals": 0,
                       "invalid_arguments": 0, "api_call_limit_hits": 0}

    @property
    def messages(self) -> list[dict]:
        """The committed conversation history (a shallow copy)."""
        return list(self._messages)

    @property
    def tool_params(self) -> list[dict]:
        """The frozen ``tools`` parameter of this campaign."""
        return copy.deepcopy(self._tools)

    def usage(self) -> dict[str, Any]:
        """API calls, token counts (``input_tokens`` excludes cached input), finish reasons, cost."""
        out = copy.deepcopy(self._usage)
        out["cost_usd"] = round(float(out["cost_usd"]), 6)
        return out

    # ---- Policy hooks --------------------------------------------------------------------------
    def start(self, session: "AgentSession") -> None:
        """Create the client if needed and freeze the system message and the tools."""
        self._session = session
        self._reset()
        if self._client is None:
            self._client = self._make_client()
        try:
            if session.toolset == "llm_only":
                texts = [self.system_prompt or load_prompt("SYSTEM_LLM_ONLY.md")]
            else:
                texts = [self.system_prompt or load_prompt("SYSTEM.md"), load_prompt("LENZ_REF.md")]
        except OSError as exc:
            raise PolicyError(f"cannot read the Sara prompt files in {PROMPTS_DIR}: {exc}") from exc
        self._system_text = "\n\n".join(texts)
        self._messages = [{"role": "system", "content": self._system_text}]
        self._tools = to_openai_tools(session.tool_definitions())
        self._log_seen = len(session.log)
        self._write_transcript({"event": "start", "policy": self.name, "problem": session.problem.name,
                                "budget": session.budget, "toolset": session.toolset,
                                "settings": self._settings(), "system": self._system_text, "tools": self._tools})

    def on_context(self, session: "AgentSession", item: "ContextItem") -> None:
        """Queue exogenous context; it is sent at the start of the next step."""
        self._pending_context.append(item)

    def step(self, session: "AgentSession") -> None:
        """One campaign step: request, run the tool calls, repeat until an evaluation (or stop) is committed."""
        if self._session is not session or not self._system_text:
            raise PolicyError("OpenAISaraPolicy.step() called before start() for this session")
        if session.is_done:
            return
        try:
            self._run_step(session)
        finally:
            self._log_seen = len(session.log)

    def finish(self, session: "AgentSession") -> None:
        if self._session is session:
            self._write_transcript({"event": "finish", "t": session.t, "stopped": session.stopped,
                                    "usage": self.usage()})

    # ---- the step loop -------------------------------------------------------------------------
    def _run_step(self, session: "AgentSession") -> None:
        self._usage["steps"] += 1
        self._begin_turn(session)
        calls = nudges = 0
        while True:
            if calls >= self.max_api_calls_per_step:
                self._usage["api_call_limit_hits"] += 1
                self._note(session, f"API call limit reached ({self.max_api_calls_per_step} requests in step "
                                    f"{session.step}); the harness evaluates a fallback point")
                return
            message, finish, appended = self._send(session)
            calls += 1
            refusal = _get(message, "refusal")
            if finish == "content_filter" or (isinstance(refusal, str) and refusal.strip()):
                self._usage["refusals"] += 1
                msg = f"the model declined to continue (finish_reason {finish!r})"
                if isinstance(refusal, str) and refusal.strip():
                    msg += f": {_short(refusal, 400)}"
                self._note(session, msg)
                raise PolicyError(msg)
            self._log_reasoning(session, message, finish)
            if not appended:
                nudges += 1
                if nudges > self.max_nudges:
                    self._note(session, "empty responses; the harness evaluates a fallback point")
                    return
                continue
            tool_calls = list(_get(message, "tool_calls") or [])
            if finish == "length":   # output cut off: tool arguments may be truncated, run nothing
                self._usage["truncated"] += 1
                self._stage_truncated(tool_calls)
                continue
            if tool_calls:
                self._staged_tool.extend(self._execute(session, tool_calls))
                if session.step_committed or session.is_done:
                    return
                continue
            if session.step_committed or session.is_done:
                return
            if nudges >= self.max_nudges:
                self._note(session, f"turn ended {nudges + 1} time(s) without an evaluation; the harness "
                                    f"evaluates a fallback point")
                return
            nudges += 1
            self._usage["nudges"] += 1
            self._staged_text.append(self._nudge_text(session, last=nudges == self.max_nudges))

    def _begin_turn(self, session: "AgentSession") -> None:
        if not self._opened:
            self._opened = True
            self._staged_text.append(session.brief())
            self._staged_text.append(f"[harness] {session.status_line()}. The campaign starts now: plan your "
                                     f"opening, then commit the first evaluation with `evaluate`.")
        else:
            self._staged_text.extend(self._harness_notes(session))
            self._staged_text.append(f"[harness] {session.status_line()}")
        for item in self._pending_context:
            source = _get(item, "source") or "operator"
            t = _get(item, "t", session.t)
            self._staged_system.append(f"New context from {source} (received after {t} of {session.budget} "
                                       f"evaluations):\n\n{str(_get(item, 'text', '')).strip()}")
        self._pending_context = []

    def _harness_notes(self, session: "AgentSession") -> list[str]:
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
    def _nudge_text(session: "AgentSession", last: bool) -> str:
        options = "`evaluate` with a config and a rationale"
        if session.config.allow_early_stop:
            options += " (or `stop` to end the campaign)"
        text = (f"[harness] Step {session.step} is still open: no evaluation has been committed. Call {options} "
                f"as a tool call. Free tools remain available, but only an evaluation completes the step.")
        if last:
            text += " If this turn also ends without one, the harness evaluates a fallback point for you."
        return text

    def _stage_truncated(self, tool_calls: list[Any]) -> None:
        if tool_calls:
            msg = ("Not executed: the response was cut off (finish_reason 'length') while this call was being "
                   "written, so its arguments may be truncated. Re-issue the call with shorter reasoning or arguments.")
            self._staged_tool.extend({"role": "tool", "tool_call_id": _get(c, "id"), "content": msg}
                                     for c in tool_calls)
        else:
            self._staged_text.append("[harness] Your previous response was cut off (finish_reason 'length'). "
                                     "Continue more concisely and call a tool.")

    def _execute(self, session: "AgentSession", tool_calls: list[Any]) -> list[dict]:
        """Run the tool calls of one assistant message in order; one ``tool`` message per call."""
        out = []
        for c in tool_calls:
            fn = _get(c, "function")
            name, raw = _get(fn, "name"), _get(fn, "arguments")
            args, parse_error = _parse_arguments(raw)
            if parse_error is not None:
                self._usage["invalid_arguments"] += 1
                text = json.dumps({"INVALID_JSON": raw if isinstance(raw, str) else repr(raw),
                                   "error": f"the arguments are not a JSON object ({parse_error}); re-issue the "
                                            f"call with valid JSON arguments"}, ensure_ascii=False)
                out.append({"role": "tool", "tool_call_id": _get(c, "id"), "content": text})
                if self.verbose:
                    self._say(f"step {session.step} | {name}: invalid arguments ({parse_error})")
                continue
            envelope = session.call(name, args)
            out.append({"role": "tool", "tool_call_id": _get(c, "id"), "content": _result_text(envelope)})
            if self.verbose:
                status = "ok" if envelope.get("ok") else f"error: {_short(envelope.get('error'), 160)}"
                self._say(f"step {session.step} | {name}({_short(_dump(args), 160)}) -> {status}")
        return out

    # ---- requests ------------------------------------------------------------------------------
    def _request_kwargs(self, messages: list[dict]) -> dict[str, Any]:
        """Every request is built here (and only here)."""
        kw: dict[str, Any] = {"model": self.model, "messages": messages, "tools": self._tools, "tool_choice": "auto"}
        if self.reasoning_effort is not None:
            kw["reasoning_effort"] = self.reasoning_effort
        if self.max_completion_tokens is not None:
            kw["max_completion_tokens"] = self.max_completion_tokens
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.parallel_tool_calls is not None:
            kw["parallel_tool_calls"] = self.parallel_tool_calls
        if self.extra_body:
            kw["extra_body"] = copy.deepcopy(self.extra_body)
        return kw

    def _staged_messages(self) -> list[dict]:
        """Tool answers first (they must follow the assistant message), then one user and one system message."""
        out = list(self._staged_tool)
        texts = list(self._staged_text)
        system = list(self._staged_system)
        if system and not self.context_as_system:
            texts.extend(f"<system-reminder>\n{t}\n</system-reminder>" for t in system)
            system = []
        if texts:
            out.append({"role": "user", "content": "\n\n".join(texts)})
        if system:
            out.append({"role": "system", "content": "\n\n".join(system)})
        return out

    def _send(self, session: "AgentSession") -> tuple[Any, Any, bool]:
        """Send history + staged messages; commit both with the reply when the reply is non-empty."""
        new = self._staged_messages()
        kw = self._request_kwargs(self._messages + new)
        try:
            response = self._client.chat.completions.create(**kw)
        except Exception as exc:  # noqa: BLE001 - every client failure becomes a PolicyError
            status = getattr(exc, "status_code", None)
            label = f"HTTP {status}" if status is not None else type(exc).__name__
            raise PolicyError(f"chat completion request failed ({label}): {_short(exc, 600)}") from exc
        choices = _get(response, "choices") or []
        choice = choices[0] if choices else None
        message = _get(choice, "message") if choice is not None else None
        finish = _get(choice, "finish_reason") if choice is not None else None
        assistant = _assistant_message(message)
        appended = assistant is not None
        if appended:
            self._messages.extend(new)
            self._messages.append(assistant)
            self._staged_tool, self._staged_text, self._staged_system = [], [], []
        self._account(response, finish)
        self._log_exchange(session, kw, response)
        return message, finish, appended

    def _make_client(self) -> Any:
        try:
            import openai  # noqa: PLC0415 - optional dependency
        except ImportError as exc:
            raise PolicyError(f"OpenAISaraPolicy needs the 'openai' package, which is not installed: "
                              f"{INSTALL_HINT}") from exc
        api_key = self.api_key
        if api_key is None and self.base_url is not None and not os.environ.get("OPENAI_API_KEY"):
            api_key = LOCAL_API_KEY   # self-hosted servers usually ignore the key, but the SDK requires one
        try:
            return openai.OpenAI(base_url=self.base_url, api_key=api_key)
        except Exception as exc:  # noqa: BLE001 - e.g. no credentials configured
            raise PolicyError(f"could not create the OpenAI client ({type(exc).__name__}: {_short(exc)}); set "
                              f"OPENAI_API_KEY, pass api_key=... / base_url=..., or pass client=...") from exc

    # ---- accounting and traces -----------------------------------------------------------------
    def _account(self, response: Any, finish: Any) -> None:
        u = self._usage
        u["api_calls"] += 1
        key = str(finish or "unknown")
        u["finish_reasons"][key] = u["finish_reasons"].get(key, 0) + 1
        usage = _get(response, "usage")
        if usage is None:
            return
        prompt = _count(_get(usage, "prompt_tokens"))
        cached = _count(_get(_get(usage, "prompt_tokens_details"), "cached_tokens"))
        output = _count(_get(usage, "completion_tokens"))
        reasoning = _count(_get(_get(usage, "completion_tokens_details"), "reasoning_tokens"))
        uncached = max(prompt - cached, 0)
        u["input_tokens"] += uncached
        u["cache_read_input_tokens"] += cached
        u["output_tokens"] += output
        u["reasoning_tokens"] += reasoning
        if self.pricing is not None:
            p_in, p_out, p_cached = self.pricing
            u["cost_usd"] += (uncached * p_in + output * p_out + cached * p_cached) / 1e6

    def _log_reasoning(self, session: "AgentSession", message: Any, finish: Any) -> None:
        call = self._usage["api_calls"]
        for field in _REASONING_FIELDS:
            text = _get(message, field)
            if isinstance(text, str) and text.strip():
                session.log_reasoning(text, kind="thinking", finish_reason=finish, api_call=call)
                if self.verbose:
                    self._say(f"step {session.step} | thinking: {_short(text)}")
                break
        text = _get(message, "content")
        if isinstance(text, str) and text.strip():
            session.log_reasoning(text, kind="text", finish_reason=finish, api_call=call)
            if self.verbose:
                self._say(f"step {session.step} | text: {_short(text)}")

    def _note(self, session: "AgentSession", text: str) -> None:
        session.log_reasoning(text, kind="policy")
        self._write_transcript({"event": "note", "step": session.step, "t": session.t, "text": text})
        if self.verbose:
            self._say(f"step {session.step} | {text}")

    def _log_exchange(self, session: "AgentSession", kw: Mapping[str, Any], response: Any) -> None:
        if not self.transcript_path:
            return
        msgs = kw["messages"]
        params = {k: v for k, v in kw.items() if k not in ("tools", "messages")}
        self._write_transcript({"event": "request", "step": session.step, "t": session.t, "n_messages": len(msgs),
                                "new_messages": msgs[max(self._n_logged_messages, 1):], "params": params})
        self._write_transcript({"event": "response", "step": session.step, "t": session.t, "response": response})
        self._n_logged_messages = len(self._messages)

    def _write_transcript(self, event: Mapping[str, Any]) -> None:
        if not self.transcript_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.transcript_path)), exist_ok=True)
            with open(self.transcript_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(_plain(event), sort_keys=True, ensure_ascii=False, default=repr) + "\n")
        except OSError as exc:  # a transcript must never stop the campaign
            print(f"[sara] transcript disabled: {exc}", file=sys.stderr, flush=True)
            self.transcript_path = None

    def _settings(self) -> dict[str, Any]:
        return {"model": self.model, "base_url": self.base_url, "reasoning_effort": self.reasoning_effort,
                "max_completion_tokens": self.max_completion_tokens, "temperature": self.temperature,
                "parallel_tool_calls": self.parallel_tool_calls, "extra_body": self.extra_body,
                "context_as_system": self.context_as_system, "pricing": self.pricing,
                "max_api_calls_per_step": self.max_api_calls_per_step, "max_nudges": self.max_nudges}

    def _say(self, text: str) -> None:
        if self.verbose:
            print(f"[sara] {text}", file=sys.stderr, flush=True)


# =========================================================================== helpers


def _count(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else 0


def _parse_arguments(raw: Any) -> tuple[Any, str | None]:
    """Tool-call arguments as a dict, or ``(raw, reason)`` when they are not a JSON object."""
    if isinstance(raw, Mapping):   # some clients/servers hand back parsed arguments
        return dict(raw), None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None             # a call without arguments (e.g. `status`)
    if not isinstance(raw, str):
        return raw, f"unexpected type {type(raw).__name__}"
    try:
        value = json.loads(raw)
    except ValueError as exc:
        return raw, f"invalid JSON: {exc}"
    if not isinstance(value, dict):
        return raw, f"expected a JSON object, got {type(value).__name__}"
    return value, None


def _result_text(envelope: Mapping[str, Any]) -> str:
    """Tool-message content for a session envelope (results as JSON, failures as plain error text)."""
    if envelope.get("ok"):
        return _dump(envelope.get("result"))
    parts = ["ERROR: " + str(envelope.get("error") or "the tool call failed")]
    if envelope.get("warning"):
        parts.append(str(envelope["warning"]))
    if envelope.get("fallback") is not None:
        parts.append("Fallback evaluation: " + _dump(envelope["fallback"]))
    return "\n".join(parts)


def _assistant_message(message: Any) -> dict | None:
    """The assistant message to append (content + tool calls), or ``None`` for an empty reply.

    Reasoning fields are deliberately left out: they are logged, not sent back.
    """
    if message is None:
        return None
    content = _get(message, "content")
    tool_calls = []
    for c in _get(message, "tool_calls") or []:
        fn = _get(c, "function")
        args = _get(fn, "arguments")
        tool_calls.append({"id": _get(c, "id"), "type": "function", "function": {
            "name": _get(fn, "name"),
            "arguments": args if isinstance(args, str) else json.dumps(_plain(args), ensure_ascii=False)}})
    has_text = isinstance(content, str) and content.strip() != ""
    if not has_text and not tool_calls:
        return None
    out: dict[str, Any] = {"role": "assistant", "content": content if isinstance(content, str) else ""}
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out
