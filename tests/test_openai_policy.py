"""Tests for OpenAISaraPolicy with a scripted fake Chat Completions client (no network).

``FakeClient.chat.completions.create(**kwargs)`` records every request (with a JSON
snapshot of its messages taken at call time) and replays scripted responses shaped like
the openai SDK's ``ChatCompletion`` objects.
"""

from __future__ import annotations

import itertools
import json
import sys
from types import SimpleNamespace as NS
from typing import Any, Callable

import pytest

from agentic_bo.experiment import _is_llm
from agentic_bo.problems import get_problem
from agentic_bo.sara import Campaign, CampaignConfig
from agentic_bo.sara.policies import PolicyError, get_policy
from agentic_bo.sara.policies.llm import load_prompt
from agentic_bo.sara.policies.openai_llm import LOCAL_API_KEY, OpenAISaraPolicy, to_openai_tools
from agentic_bo.sara.tools import TOOLSETS
from agentic_bo.utils import LenzError

_ids = itertools.count()


# =========================================================================== fakes


def call(name: str, args: Any) -> NS:
    raw = args if isinstance(args, str) else json.dumps(args)
    return NS(id=f"call_{name}_{next(_ids)}", type="function", function=NS(name=name, arguments=raw))


def usage(prompt: int = 100, completion: int = 20, cached: int = 0, reasoning: int = 0) -> NS:
    return NS(prompt_tokens=prompt, completion_tokens=completion,
              prompt_tokens_details=NS(cached_tokens=cached),
              completion_tokens_details=NS(reasoning_tokens=reasoning))


def reply(*calls: NS, content: str | None = None, finish: str | None = None, use: NS | None = None,
          **message_fields: Any) -> NS:
    finish = finish or ("tool_calls" if calls else "stop")
    msg = NS(role="assistant", content=content, tool_calls=list(calls) or None, refusal=None, **message_fields)
    return NS(choices=[NS(index=0, message=msg, finish_reason=finish)], usage=use or usage())


class FakeAPIError(Exception):
    def __init__(self, msg: str, status_code: int):
        super().__init__(msg)
        self.status_code = status_code


class FakeClient:
    """``responder(client, kwargs)`` returns a response or raises."""

    def __init__(self, responder: Callable[["FakeClient", dict], Any]):
        self.responder = responder
        self.calls: list[dict] = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw: Any) -> Any:
        self.calls.append({"kw": kw, "messages": [json.dumps(m, sort_keys=True) for m in kw["messages"]]})
        out = self.responder(self, kw)
        if isinstance(out, BaseException):
            raise out
        return out


def script(*outcomes: Any) -> Callable[[FakeClient, dict], Any]:
    it = iter(outcomes)

    def responder(client: FakeClient, kw: dict) -> Any:
        try:
            return next(it)
        except StopIteration:
            raise AssertionError("the fake model ran out of scripted responses") from None
    return responder


def suggest_then_evaluate() -> Callable[[FakeClient, dict], Any]:
    """A well-behaved model: suggest, then evaluate the top candidate, every step."""
    def responder(client: FakeClient, kw: dict) -> Any:
        last = kw["messages"][-1]
        if last["role"] == "tool":
            try:
                result = json.loads(last["content"])
            except ValueError:   # an error answer in plain text
                result = {}
            if isinstance(result, dict) and "candidates" in result:
                cfg = result["candidates"][0]["config"]
                return reply(call("evaluate", {"config": cfg, "rationale": "top logEI candidate"}),
                             content="Evaluating the suggestion.")
        return reply(call("suggest", {}))
    return responder


def run(policy: OpenAISaraPolicy, budget: int = 3, problem: str = "branin", **cfg: Any) -> Any:
    prob = get_problem(problem, seed=0, randomize=True)
    return Campaign(prob, policy, CampaignConfig(budget=budget, seed=0, **cfg)).run()


def policy(responder: Callable, **kw: Any) -> tuple[OpenAISaraPolicy, FakeClient]:
    client = FakeClient(responder)
    return OpenAISaraPolicy(model="test-model", client=client, **kw), client


# =========================================================================== construction


def test_model_is_required_and_arguments_are_validated():
    with pytest.raises(LenzError, match="model"):
        OpenAISaraPolicy()
    for bad in ({"toolset": "nope"}, {"temperature": -1}, {"pricing": (1, 2)}, {"max_completion_tokens": 0},
                {"reasoning_effort": ""}, {"extra_body": [1]}):
        with pytest.raises(LenzError):
            OpenAISaraPolicy(model="m", **bad)


def test_registry_names_and_llm_detection():
    p = get_policy("openai", model="m", client=FakeClient(script()))
    assert isinstance(p, OpenAISaraPolicy) and p.name == "openai" and p.toolset == "full"
    q = get_policy("openai-llm-only", model="m", client=FakeClient(script()))
    assert q.toolset == "llm_only" and q.name == "openai_llm_only"
    assert get_policy("sara_openai", model="m").name == "openai"
    assert _is_llm("openai") and _is_llm(p) and not _is_llm("fixed_bo")


def test_context_delivery_default_depends_on_base_url():
    assert OpenAISaraPolicy(model="m").context_as_system is True
    assert OpenAISaraPolicy(model="m", base_url="http://localhost:8000/v1").context_as_system is False
    assert OpenAISaraPolicy(model="m", base_url="http://x/v1", context_as_system=True).context_as_system is True


def test_tool_conversion_is_sorted_and_non_strict():
    defs = [{"name": "b", "description": "B", "input_schema": {"type": "object", "properties": {}}},
            {"name": "a", "description": "A", "input_schema": {"type": "object", "properties": {"x": {}}}}]
    tools = to_openai_tools(defs)
    assert [t["function"]["name"] for t in tools] == ["a", "b"]
    assert tools[0] == {"type": "function", "function": {"name": "a", "description": "A",
                                                         "parameters": {"type": "object", "properties": {"x": {}}}}}
    assert all("strict" not in t["function"] for t in tools)


# =========================================================================== request contract


def test_request_contract_and_full_campaign():
    pol, client = policy(suggest_then_evaluate())
    res = run(pol, budget=3)
    assert res.n_evals == 3 and res.forced_commits == 0 and res.policy_error is None
    assert all(r["rationale"] == "top logEI candidate" for r in res.records)
    kw = client.calls[0]["kw"]
    assert kw["model"] == "test-model" and kw["tool_choice"] == "auto"
    for absent in ("reasoning_effort", "max_completion_tokens", "temperature", "parallel_tool_calls", "extra_body"):
        assert absent not in kw
    sys_msg = kw["messages"][0]
    assert sys_msg["role"] == "system"
    assert sys_msg["content"] == load_prompt("SYSTEM.md") + "\n\n" + load_prompt("LENZ_REF.md")
    names = [t["function"]["name"] for t in kw["tools"]]
    assert names == sorted(names) and "evaluate" in names and "suggest" in names
    # the prefix (system message + tools) never changes
    assert all(c["kw"]["tools"] == kw["tools"] and c["messages"][0] == client.calls[0]["messages"][0]
               for c in client.calls)
    assert kw["messages"][1]["role"] == "user" and "campaign starts now" in kw["messages"][1]["content"]
    assert len(kw["messages"]) == 2


def test_optional_parameters_are_sent_when_set():
    pol, client = policy(suggest_then_evaluate(), reasoning_effort="high", max_completion_tokens=4096,
                         temperature=0.2, parallel_tool_calls=False,
                         extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    run(pol, budget=1)
    kw = client.calls[0]["kw"]
    assert (kw["reasoning_effort"], kw["max_completion_tokens"], kw["temperature"], kw["parallel_tool_calls"]) == \
        ("high", 4096, 0.2, False)
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


def test_history_is_append_only():
    pol, client = policy(suggest_then_evaluate())
    run(pol, budget=3)
    for prev, cur in zip(client.calls, client.calls[1:]):
        assert cur["messages"][: len(prev["messages"])] == prev["messages"]


def test_each_tool_call_gets_its_own_tool_message_in_order_then_user_text():
    def responder(client: FakeClient, kw: dict) -> Any:
        n = len(client.calls)
        if n == 1:
            return reply(call("status", {}), call("trials", {}), call("suggest", {}))
        if n == 2:
            cand = json.loads(kw["messages"][-1]["content"])["candidates"][0]["config"]
            return reply(call("evaluate", {"config": cand, "rationale": "r"}))
        return reply(call("suggest", {}))
    pol, client = policy(responder)
    run(pol, budget=1)
    msgs = client.calls[1]["kw"]["messages"]
    assistant = msgs[-4]
    assert assistant["role"] == "assistant" and len(assistant["tool_calls"]) == 3
    tools = msgs[-3:]   # mid-step the tool answers end the request; harness text opens the next step
    assert [m["role"] for m in tools] == ["tool"] * 3
    assert [m["tool_call_id"] for m in tools] == [c["id"] for c in assistant["tool_calls"]]
    assert msgs[-5]["role"] == "user" and "[harness]" in msgs[-5]["content"]


# =========================================================================== robustness


def test_invalid_json_arguments_are_answered_not_run():
    def responder(client: FakeClient, kw: dict) -> Any:
        n = len(client.calls)
        if n == 1:
            return reply(call("evaluate", '{"config": {"x": 1'))   # truncated JSON
        return suggest_then_evaluate()(client, kw)
    pol, client = policy(responder)
    res = run(pol, budget=1)
    err = client.calls[1]["kw"]["messages"][-1]
    assert err["role"] == "tool" and "INVALID_JSON" in err["content"]
    assert res.n_evals == 1 and res.forced_commits == 0
    assert pol.usage()["invalid_arguments"] == 1


def test_length_finish_leaves_tool_calls_unexecuted():
    def responder(client: FakeClient, kw: dict) -> Any:
        if len(client.calls) == 1:
            return reply(call("evaluate", {"config": {}, "rationale": "cut"}), finish="length")
        return suggest_then_evaluate()(client, kw)
    pol, client = policy(responder)
    res = run(pol, budget=1)
    answer = client.calls[1]["kw"]["messages"][-1]
    assert answer["role"] == "tool" and "Not executed" in answer["content"]
    assert res.n_evals == 1 and res.records[0]["rationale"] == "top logEI candidate"
    assert pol.usage()["truncated"] == 1


def test_turns_without_evaluation_get_nudged_then_fall_back():
    pol, client = policy(lambda c, kw: reply(content="Thinking about it..."), max_nudges=2)
    res = run(pol, budget=1)
    assert res.forced_commits == 1 and res.n_evals == 1
    nudges = [m for m in client.calls[-1]["kw"]["messages"] if m["role"] == "user" and "still open" in m["content"]]
    assert len(nudges) == 2


def test_refusal_and_api_errors_become_policy_errors_and_harness_continues():
    refusal = reply(content=None)
    refusal.choices[0].message.refusal = "I can't help with that."
    pol, _ = policy(script(refusal))
    res = run(pol, budget=2)
    assert res.policy_error and "declined" in res.policy_error and res.forced_commits == 2

    pol2, _ = policy(script(FakeAPIError("bad request", 400)))
    res2 = run(pol2, budget=1)
    assert "HTTP 400" in res2.policy_error and res2.forced_commits == 1

    pol3, _ = policy(script(reply(finish="content_filter", content="...")))
    with pytest.raises(PolicyError):
        run(pol3, budget=1, on_policy_error="raise")


def test_empty_replies_are_resent_without_touching_history():
    outcomes = iter([reply(content=""), reply(content=None)])

    def responder(client: FakeClient, kw: dict) -> Any:
        nxt = next(outcomes, None)
        return nxt if nxt is not None else suggest_then_evaluate()(client, kw)
    pol, client = policy(responder)
    res = run(pol, budget=1)
    assert res.n_evals == 1
    assert client.calls[0]["messages"] == client.calls[1]["messages"] == client.calls[2]["messages"]


# =========================================================================== context and traces


@pytest.mark.parametrize("as_system", [True, False])
def test_context_injection(as_system):
    pol, client = policy(suggest_then_evaluate(), context_as_system=as_system)
    run(pol, budget=2, context_schedule={1: "Also keep x0 below 0.5."})
    step2 = [c for c in client.calls if any("keep x0 below" in m for m in c["messages"])][0]
    msgs = step2["kw"]["messages"]
    holder = [m for m in msgs if "keep x0 below" in str(m.get("content"))][0]
    if as_system:
        assert holder["role"] == "system" and msgs[msgs.index(holder) - 1]["role"] == "user"
    else:
        assert holder["role"] == "user" and "<system-reminder>" in holder["content"]
    assert sum(m["role"] == "system" for m in msgs) == (2 if as_system else 1)


def test_reasoning_text_is_logged_but_not_sent_back():
    def responder(client: FakeClient, kw: dict) -> Any:
        r = suggest_then_evaluate()(client, kw)
        r.choices[0].message.reasoning_content = "The model is uncertain near the corner."
        return r
    pol, client = policy(responder)
    res = run(pol, budget=1)
    thoughts = [e for e in res.log if e.get("tool") == "reasoning" or e.get("kind") == "thinking"]
    assert any("uncertain near the corner" in json.dumps(e) for e in thoughts)
    assert all("uncertain near the corner" not in m for c in client.calls for m in c["messages"])


def test_usage_and_cost_accounting():
    pol, _ = policy(lambda c, kw: _with_usage(suggest_then_evaluate()(c, kw)), pricing=(2.0, 8.0, 0.5))
    run(pol, budget=1)
    u = pol.usage()
    assert u["api_calls"] == 2
    assert (u["input_tokens"], u["cache_read_input_tokens"], u["output_tokens"], u["reasoning_tokens"]) == \
        (2 * 60, 2 * 40, 2 * 20, 2 * 5)
    assert u["cost_usd"] == pytest.approx(2 * (60 * 2.0 + 20 * 8.0 + 40 * 0.5) / 1e6)
    assert u["priced"] is True
    assert OpenAISaraPolicy(model="m").usage()["priced"] is False


def _with_usage(r: NS) -> NS:
    r.usage = usage(prompt=100, completion=20, cached=40, reasoning=5)
    return r


def test_llm_only_toolset_uses_its_prompt_and_tools():
    def responder(client: FakeClient, kw: dict) -> Any:
        return reply(call("evaluate", {"config": {n: 0.5 for n in _names(kw)}, "rationale": "center"}))
    pol, client = policy(responder, toolset="llm_only")
    res = run(pol, budget=1)
    kw = client.calls[0]["kw"]
    assert kw["messages"][0]["content"] == load_prompt("SYSTEM_LLM_ONLY.md")
    assert {t["function"]["name"] for t in kw["tools"]} <= set(TOOLSETS["llm_only"])
    assert res.n_evals == 1 and res.forced_commits == 0


def _names(kw: dict) -> list[str]:
    ev = [t for t in kw["tools"] if t["function"]["name"] == "evaluate"][0]
    return list(ev["function"]["parameters"]["properties"]["config"]["properties"])


def test_transcript_is_written(tmp_path):
    path = tmp_path / "t.jsonl"
    pol, _ = policy(suggest_then_evaluate(), transcript_path=path)
    run(pol, budget=1)
    events = [json.loads(line)["event"] for line in path.read_text().splitlines()]
    assert events[0] == "start" and events[-1] == "finish" and "request" in events and "response" in events


# =========================================================================== client creation


def test_missing_openai_package_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    res = run(OpenAISaraPolicy(model="m"), budget=1)
    assert "pip install openai" in res.policy_error


def test_local_server_gets_placeholder_key(monkeypatch):
    seen = {}

    class FakeOpenAI:
        def __init__(self, base_url=None, api_key=None):
            seen.update(base_url=base_url, api_key=api_key)
            self.chat = NS(completions=NS(create=lambda **kw: suggest_then_evaluate()(None, kw)))

    monkeypatch.setitem(sys.modules, "openai", NS(OpenAI=FakeOpenAI))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    res = run(OpenAISaraPolicy(model="Qwen/Qwen3-32B", base_url="http://localhost:8000/v1"), budget=1)
    assert res.policy_error is None and seen == {"base_url": "http://localhost:8000/v1", "api_key": LOCAL_API_KEY}
