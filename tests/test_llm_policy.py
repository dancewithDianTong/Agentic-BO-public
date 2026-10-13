"""Tests for ClaudeSaraPolicy (DESIGN §10) and the Sara prompts, with a scripted fake Anthropic client.

No network: ``FakeClient.beta.messages.stream(**kwargs)`` records every request (and a
JSON snapshot of its messages taken at call time) and returns a context manager whose
``get_final_message()`` yields scripted message objects.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
import types
from types import SimpleNamespace as NS
from typing import Any, Callable

import pytest

from agentic_bo.problems import get_problem
from agentic_bo.sara import Campaign, CampaignConfig
from agentic_bo.sara.policies import PolicyError, get_policy
from agentic_bo.sara.policies.llm import (
    COMPACTION_BETA, COMPACTION_EDIT_TYPE, DEFAULT_MODEL, FALLBACK_BETA, MAX_STREAM_RETRIES, PROMPTS_DIR,
    ClaudeSaraPolicy, estimate_cost, load_prompt,
)
from agentic_bo.sara.session import AgentSession, ContextItem
from agentic_bo.sara.tools import TOOLSETS
from agentic_bo.utils import LenzError

# =========================================================================== fakes

_ids = itertools.count()


def _plain(o: Any) -> Any:
    if isinstance(o, NS):
        return vars(o)
    raise TypeError(f"not serializable: {type(o).__name__}")


def snap(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=_plain)


def text(t: str) -> NS:
    return NS(type="text", text=t)


def thinking(t: str) -> NS:
    return NS(type="thinking", thinking=t, signature="sig-" + str(next(_ids)))


def tool(name: str, inp: Any, tag: str | None = None) -> NS:
    return NS(type="tool_use", id=f"toolu_{tag or name}_{next(_ids)}", name=name, input=inp)


def usage(i: int = 10, o: int = 5, r: int = 0, w: int = 0, **extra: Any) -> NS:
    return NS(input_tokens=i, output_tokens=o, cache_read_input_tokens=r, cache_creation_input_tokens=w, **extra)


def message(*blocks: Any, stop: str = "tool_use", use: NS | None = None, **extra: Any) -> NS:
    return NS(content=list(blocks), stop_reason=stop, usage=use or usage(), model=DEFAULT_MODEL, **extra)


class FakeAPIError(Exception):
    """Duck-typed SDK error (status_code attribute), as raised by the real client."""

    def __init__(self, msg: str, status_code: int):
        super().__init__(msg)
        self.message = msg
        self.status_code = status_code


class FakeStream:
    def __init__(self, record: dict, outcome: Any):
        self.record, self.outcome = record, outcome

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def get_final_message(self) -> Any:
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        self.record["ok"] = True
        return self.outcome


class FakeClient:
    """``responder(client, kwargs)`` returns a message, returns an exception (raised by the stream) or raises."""

    def __init__(self, responder: Callable[["FakeClient", dict], Any]):
        self.responder = responder
        self.calls: list[dict] = []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kw: Any) -> FakeStream:
        record = {"kw": kw, "ok": False, "messages": [snap(m) for m in kw["messages"]],
                  "system": snap(kw.get("system")), "tools": snap(kw.get("tools"))}
        self.calls.append(record)
        return FakeStream(record, self.responder(self, kw))

    @property
    def ok_calls(self) -> list[dict]:
        return [c for c in self.calls if c["ok"]]


def script(*outcomes: Any) -> Callable[[FakeClient, dict], Any]:
    """Responder replaying ``outcomes`` in order (then failing loudly)."""
    it = iter(outcomes)

    def responder(client: FakeClient, kw: dict) -> Any:
        try:
            return next(it)
        except StopIteration:
            raise AssertionError("the fake model ran out of scripted responses") from None
    return responder


def last_user(kw: dict) -> dict:
    return next(m for m in reversed(kw["messages"]) if m["role"] == "user")


def tool_results(msg: dict) -> list[dict]:
    return [b for b in msg["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]


class SuggestEvaluateBot:
    """A scripted 'model': ask lenz for a suggestion, then evaluate its top candidate."""

    def __init__(self) -> None:
        self.n = 0

    def __call__(self, client: FakeClient, kw: dict) -> Any:
        self.n += 1
        results = tool_results(last_user(kw))
        sug = [r for r in results if r["tool_use_id"].startswith("toolu_suggest")]
        if sug and not sug[-1].get("is_error"):
            cfg = json.loads(sug[-1]["content"])["candidates"][0]["config"]
            return message(thinking(f"evaluating suggestion {self.n}"),
                           tool("evaluate", {"config": cfg, "rationale": "top suggestion", "expectation": "better"}),
                           use=usage(100, 40, 900, 60))
        return message(thinking(f"plan {self.n}"), text("Asking lenz."), tool("suggest", {"q": 1}),
                       use=usage(100, 40, 900, 60))


def make_policy(responder: Callable, **kw: Any) -> tuple[ClaudeSaraPolicy, FakeClient]:
    client = FakeClient(responder)
    return ClaudeSaraPolicy(client=client, **kw), client


def run_campaign(policy: ClaudeSaraPolicy, problem: str = "branin", budget: int = 3, **cfg: Any) -> Campaign:
    camp = Campaign(get_problem(problem), policy, CampaignConfig(budget=budget, **cfg))
    camp.run()
    return camp


def started(policy: ClaudeSaraPolicy, problem: str = "branin", budget: int = 3, **cfg: Any) -> AgentSession:
    session = AgentSession(get_problem(problem), CampaignConfig(budget=budget, toolset=policy.toolset, **cfg))
    policy.start(session)
    session.begin_step()
    return session


def center(session: AgentSession) -> dict:
    return session.problem.space.center()


def assert_append_only(calls: list[dict]) -> None:
    assert len(calls) >= 2
    for prev, cur in zip(calls, calls[1:]):
        n = len(prev["messages"])
        assert len(cur["messages"]) > n
        assert cur["messages"][:n] == prev["messages"], "history was edited between requests"
        assert cur["system"] == prev["system"] and cur["tools"] == prev["tools"]


# =========================================================================== construction


def test_import_and_construction_without_anthropic(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)  # constructing must not import the SDK
    p = ClaudeSaraPolicy()
    assert p.name == "sara" and p.toolset == "full" and p.model == "claude-opus-5-5" and p.effort == "high"
    assert p.usage()["api_calls"] == 0
    q = get_policy("llm_only")
    assert isinstance(q, ClaudeSaraPolicy) and q.toolset == "llm_only" and q.name == "llm_only"
    assert isinstance(get_policy("claude", effort="medium"), ClaudeSaraPolicy)


@pytest.mark.parametrize("kw", [{"effort": "extreme"}, {"thinking_display": "full"}, {"toolset": "nope"},
                                {"max_tokens": 0}, {"max_nudges": -1}, {"max_api_calls_per_step": 0},
                                {"model": ""}, {"system_prompt": "  "}, {"compaction_trigger_tokens": 1.5},
                                {"compaction_trigger_tokens": 10_000}])   # below the API minimum (50,000)
def test_invalid_constructor_arguments(kw):
    with pytest.raises(LenzError):
        ClaudeSaraPolicy(**kw)


def test_missing_anthropic_package_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)  # import anthropic -> ImportError
    policy = ClaudeSaraPolicy()
    session = AgentSession(get_problem("branin"), CampaignConfig(budget=2))
    with pytest.raises(PolicyError, match="pip install anthropic"):
        policy.start(session)
    res = Campaign(get_problem("branin"), ClaudeSaraPolicy(), CampaignConfig(budget=2)).run()
    assert "anthropic" in res.policy_error and res.n_evals == 2 and res.forced_commits == 2


def test_client_created_from_anthropic_module(monkeypatch):
    fake = FakeClient(SuggestEvaluateBot())
    mod = types.ModuleType("anthropic")
    mod.Anthropic = lambda: fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    camp = run_campaign(ClaudeSaraPolicy(), budget=2)
    assert camp.result.policy_error is None and len(fake.ok_calls) == 4

    def broken() -> None:
        raise RuntimeError("no credentials")
    mod.Anthropic = broken  # type: ignore[attr-defined]
    with pytest.raises(PolicyError, match="could not create the Anthropic client"):
        ClaudeSaraPolicy().start(AgentSession(get_problem("branin"), CampaignConfig(budget=2)))


def test_step_before_start_raises():
    policy, _ = make_policy(SuggestEvaluateBot())
    session = AgentSession(get_problem("branin"), CampaignConfig(budget=2))
    with pytest.raises(PolicyError, match="before start"):
        policy.step(session)


# =========================================================================== request contract


def test_request_contract():
    policy, client = make_policy(SuggestEvaluateBot())
    session = started(policy)
    policy.step(session)
    assert session.t == 1 and len(client.calls) == 2
    kw = client.calls[0]["kw"]
    assert kw["model"] == "claude-opus-5-5" and kw["max_tokens"] == 32000
    assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kw["output_config"] == {"effort": "high"}
    assert kw["tool_choice"] == {"type": "auto"}
    assert kw["betas"] == [FALLBACK_BETA, COMPACTION_BETA]
    assert kw["fallbacks"] == "default"
    assert kw["context_management"] == {"edits": [{
        "type": COMPACTION_EDIT_TYPE, "trigger": {"type": "input_tokens", "value": 150_000},
        "instructions": load_prompt("COMPACTION.md")}]}
    assert kw["cache_control"] == {"type": "ephemeral"}
    assert "extra_body" not in kw and "temperature" not in kw
    blob = json.dumps({k: v for k, v in kw.items() if k != "messages"}, default=_plain)
    assert "budget_tokens" not in blob and '"disabled"' not in blob
    # frozen system: SYSTEM.md + LENZ_REF.md, cache breakpoint on the last block only
    assert kw["system"] == [{"type": "text", "text": load_prompt("SYSTEM.md")},
                            {"type": "text", "text": load_prompt("LENZ_REF.md"),
                             "cache_control": {"type": "ephemeral"}}]
    # tools: sorted by name, eager input streaming, schemas exactly the session's
    names = [t["name"] for t in kw["tools"]]
    assert names == sorted(names) == [d["name"] for d in session.tool_definitions()]
    for t, d in zip(kw["tools"], session.tool_definitions()):
        assert t["eager_input_streaming"] is True
        assert {k: v for k, v in t.items() if k != "eager_input_streaming"} == d
    # first request: one user message with the brief and the kickoff line
    (first,) = kw["messages"]
    assert first["role"] == "user" and first["content"][0]["text"] == session.brief()
    assert first["content"][1]["text"].startswith("[harness] Step 1/3")
    # the same objects are reused: identical bytes in every request
    assert client.calls[1]["system"] == client.calls[0]["system"]
    assert client.calls[1]["tools"] == client.calls[0]["tools"]


def test_request_toggles():
    policy, client = make_policy(SuggestEvaluateBot(), use_fallbacks=False, use_compaction=False, cache=False,
                                 eager_input_streaming=False, thinking_display=None, effort="medium",
                                 max_tokens=4000, compaction_trigger_tokens=90_000)
    session = started(policy)
    policy.step(session)
    kw = client.calls[0]["kw"]
    for key in ("betas", "fallbacks", "context_management", "cache_control"):
        assert key not in kw
    assert kw["thinking"] == {"type": "adaptive"} and kw["output_config"] == {"effort": "medium"}
    assert kw["max_tokens"] == 4000
    assert all("cache_control" not in b for b in kw["system"])
    assert all("eager_input_streaming" not in t for t in kw["tools"])

    policy2, client2 = make_policy(SuggestEvaluateBot(), use_fallbacks=False, compaction_trigger_tokens=90_000,
                                   thinking_display="omitted")
    session2 = started(policy2)
    policy2.step(session2)
    kw2 = client2.calls[0]["kw"]
    assert kw2["betas"] == [COMPACTION_BETA] and "fallbacks" not in kw2
    assert kw2["context_management"]["edits"][0]["trigger"]["value"] == 90_000
    assert kw2["thinking"] == {"type": "adaptive", "display": "omitted"}


def test_custom_system_prompt_replaces_system_md():
    policy, client = make_policy(SuggestEvaluateBot(), system_prompt="You are a careful optimizer.")
    policy.step(started(policy))
    system = client.calls[0]["kw"]["system"]
    assert system[0] == {"type": "text", "text": "You are a careful optimizer."}
    assert system[1]["text"] == load_prompt("LENZ_REF.md")


# =========================================================================== full campaigns


def test_full_campaign_on_branin_with_scripted_model():
    bot = SuggestEvaluateBot()
    policy, client = make_policy(bot)
    camp = run_campaign(policy, budget=5)
    res = camp.result
    assert res.policy_error is None and res.n_evals == 5 and res.forced_commits == 0
    assert res.action_counts == {"evaluate": 5, "suggest": 5}
    assert res.actions_per_step == [1] * 5
    assert len(client.calls) == 10 and all(c["ok"] for c in client.calls)
    assert all(r["rationale"] == "top suggestion" and r["expectation"] == "better" for r in res.records)
    assert all(v == v for v in res.simple_regret)  # finite (no NaN)
    assert_append_only(client.calls)
    # usage: 10 calls with fixed token counts; cost at Opus 5.5 prices
    u = res.usage
    assert u["api_calls"] == 10 and u["input_tokens"] == 1000 and u["output_tokens"] == 400
    assert u["cache_read_input_tokens"] == 9000 and u["cache_creation_input_tokens"] == 600
    assert u["stop_reasons"] == {"tool_use": 10}
    assert u["cost_usd"] == pytest.approx(10 * (100 * 4 + 40 * 20 + 900 * 0.2 + 60 * 5) / 1e6)
    # the final history extends the last request
    final = [snap(m) for m in policy.messages]
    assert final[: len(client.calls[-1]["messages"])] == client.calls[-1]["messages"]
    # reasoning traces (summarized thinking + text) are in H_t
    reasoning = [e for e in res.log if e.get("action_class") == "reasoning"]
    assert {e["kind"] for e in reasoning} == {"thinking", "text"}
    assert any(e["text"] == "plan 1" and e["step"] == 1 for e in reasoning)


def test_append_only_across_a_campaign_with_every_kind_of_turn():
    bot = SuggestEvaluateBot()
    special = {3: message(thinking("Let me think about this."), text("I will wait."), stop="end_turn"),
               6: message(thinking("long server-side work"), stop="pause_turn")}

    def responder(client: FakeClient, kw: dict) -> Any:
        return special.get(len(client.calls)) or bot(client, kw)

    policy, client = make_policy(responder)
    camp = run_campaign(policy, budget=4, context_schedule={0: "Prefer small x0.", 2: "Now minimize carefully."})
    assert camp.result.policy_error is None and camp.result.n_evals == 4 and camp.result.forced_commits == 0
    assert_append_only(client.calls)
    roles = [json.loads(m)["role"] for m in client.calls[-1]["messages"]]
    assert roles.count("system") == 2
    # every system message follows a user message and is followed by an assistant turn (or ends the request)
    for i, r in enumerate(roles):
        if r == "system":
            assert i > 0 and roles[i - 1] == "user"
            assert i == len(roles) - 1 or roles[i + 1] == "assistant"
    # the pause_turn re-send appended nothing but the paused assistant turn
    paused = client.calls[5]
    resend = client.calls[6]
    assert resend["messages"][: len(paused["messages"])] == paused["messages"]
    assert len(resend["messages"]) == len(paused["messages"]) + 1
    assert json.loads(resend["messages"][-1])["role"] == "assistant"
    assert camp.result.usage["nudges"] == 1 and camp.result.usage["pause_turns"] == 1


def test_parallel_tool_calls_answered_in_one_user_message_in_order():
    policy, client = make_policy(lambda c, kw: None)
    session = started(policy, budget=3)
    cfg = center(session)
    calls = [tool("status", {}), tool("trials", {"last": 3}), tool("evaluate", {"config": cfg, "rationale": "r"})]
    client.responder = script(message(thinking("plan"), text("Checking status first."), *calls),
                              message(tool("evaluate", {"config": {**cfg, "x0": 1.0}, "rationale": "next"})))
    policy.step(session)
    assert session.t == 1 and session.step_committed
    session.begin_step()
    policy.step(session)
    assert session.t == 2
    msgs = client.calls[1]["kw"]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [b.type for b in msgs[1]["content"]] == ["thinking", "text", "tool_use", "tool_use", "tool_use"]
    user = msgs[2]["content"]
    results = tool_results(msgs[2])
    assert [r["tool_use_id"] for r in results] == [c.id for c in calls]
    assert user[: len(results)] == results  # tool results first ...
    assert all(not r.get("is_error") for r in results)
    assert user[-1]["type"] == "text" and user[-1]["text"].startswith("[harness] Step 2/3")  # ... then the status
    assert json.loads(results[2]["content"])["trial_id"] == 0
    assert json.loads(results[0]["content"])["step"] == 1


def test_second_evaluate_in_one_message_is_rejected_and_counts_toward_next_step():
    policy, client = make_policy(lambda c, kw: None)
    session = started(policy, budget=3)
    cfg = center(session)
    first, second = (tool("evaluate", {"config": cfg, "rationale": "a"}),
                     tool("evaluate", {"config": {**cfg, "x1": 2.0}, "rationale": "b"}))
    client.responder = script(message(first, second))
    policy.step(session)
    assert session.t == 1
    results = policy._staged_user[:2]
    assert not results[0].get("is_error")
    assert results[1]["is_error"] and "rejected" in results[1]["content"]
    entry = [e for e in session.log if e.get("tool") == "evaluate"][-1]
    assert entry["step"] == 2 and not entry["ok"]


def test_end_turn_nudges_then_harness_fallback():
    policy, client = make_policy(lambda c, kw: message(text("Thinking out loud, no tool."), stop="end_turn"),
                                 max_nudges=2)
    camp = run_campaign(policy, budget=2)
    res = camp.result
    assert res.policy_error is None and res.forced_commits == 2 and res.n_evals == 2
    assert len(client.calls) == 6 and res.usage["nudges"] == 4  # 1 + 2 nudges per step
    nudges = [b["text"] for c in client.calls for m in [last_user(c["kw"])] for b in m["content"]
              if b.get("type") == "text" and "is still open" in b["text"]]
    assert len(nudges) == 4 and [n.startswith("[harness] Step 1") for n in nudges] == [True, True, False, False]
    assert ["fallback point for you" in n for n in nudges] == [False, True, False, True]
    step2 = last_user(client.calls[3]["kw"])["content"]
    note = next(b["text"] for b in step2 if "ended without an evaluation" in b["text"])
    assert note.startswith("[harness] Step 1") and '"trial_id":0' in note
    assert_append_only(client.calls)


def test_max_api_calls_per_step_leaves_the_step_to_the_harness():
    policy, client = make_policy(lambda c, kw: message(tool("status", {})), max_api_calls_per_step=3)
    camp = run_campaign(policy, budget=2)
    res = camp.result
    assert res.forced_commits == 2 and len(client.calls) == 6 and res.usage["api_call_limit_hits"] == 2
    content = last_user(client.calls[3]["kw"])["content"]
    assert content[0]["type"] == "tool_result"  # the last status result of step 1 ...
    assert "ended without an evaluation" in content[1]["text"]  # ... then the fallback notice
    assert content[2]["text"].startswith("[harness] Step 2/2")
    assert_append_only(client.calls)
    assert any(e.get("kind") == "policy" and "API call limit" in e["text"] for e in res.log)


def test_max_tokens_with_tool_use_is_not_executed():
    policy, client = make_policy(lambda c, kw: None)
    session = started(policy)
    cfg = center(session)
    cut = tool("evaluate", {"config": cfg, "rationale": "trunc"})
    client.responder = script(message(thinking("long"), cut, stop="max_tokens"),
                              message(tool("evaluate", {"config": cfg, "rationale": "short"})))
    policy.step(session)
    assert session.t == 1 and session.records[0]["rationale"] == "short"
    results = tool_results(last_user(client.calls[1]["kw"]))
    assert len(results) == 1 and results[0]["tool_use_id"] == cut.id and results[0]["is_error"]
    assert "max_tokens" in results[0]["content"]
    assert policy.usage()["max_tokens_stops"] == 1
    assert not any(e.get("tool") == "evaluate" and e.get("args", {}).get("rationale") == "trunc"
                   for e in session.log)


def test_context_window_stop_with_tool_use_is_not_executed():
    """Regression: model_context_window_exceeded can cut a tool_use off mid-input like max_tokens."""
    policy, client = make_policy(lambda c, kw: None)
    session = started(policy)
    cfg = center(session)
    cut = tool("evaluate", {"config": {k: 1.0 for k in cfg}, "rationale": "trunc"})
    client.responder = script(message(thinking("long"), cut, stop="model_context_window_exceeded"),
                              message(tool("evaluate", {"config": cfg, "rationale": "short"})))
    policy.step(session)
    assert session.t == 1 and session.records[0]["rationale"] == "short"
    results = tool_results(last_user(client.calls[1]["kw"]))
    assert len(results) == 1 and results[0]["tool_use_id"] == cut.id and results[0]["is_error"]
    assert "context window" in results[0]["content"]
    assert policy.usage()["stop_reasons"] == {"model_context_window_exceeded": 1, "tool_use": 1}
    assert_append_only(client.calls)


def test_max_tokens_without_tool_use_asks_to_continue():
    policy, client = make_policy(lambda c, kw: None)
    session = started(policy)
    client.responder = script(message(thinking("very long"), stop="max_tokens"),
                              message(tool("evaluate", {"config": center(session), "rationale": "ok"})))
    policy.step(session)
    assert session.t == 1
    content = last_user(client.calls[1]["kw"])["content"]
    assert content[-1]["type"] == "text" and "cut off" in content[-1]["text"]


def test_refusal_executes_nothing_and_raises_policy_error():
    policy, client = make_policy(lambda c, kw: None)
    session = started(policy)
    client.responder = script(message(tool("evaluate", {"config": center(session), "rationale": "x"}),
                                      stop="refusal", stop_details=NS(category="cyber", explanation="declined")))
    with pytest.raises(PolicyError, match="refusal.*cyber"):
        policy.step(session)
    assert session.t == 0 and not any(e.get("tool") == "evaluate" for e in session.log)
    assert policy.usage()["refusals"] == 1

    policy2, client2 = make_policy(script(message(stop="refusal", stop_details=None)))
    res = run_campaign(policy2, budget=3).result
    assert "refusal" in res.policy_error and res.forced_commits == 3 and len(client2.calls) == 1


def test_refused_content_is_not_read_into_the_reasoning_log():
    """Regression: stop_reason is checked before content, so a refused turn's partial output never
    reaches H_t; the refusal itself is logged as a policy note."""
    policy, client = make_policy(script(message(thinking("partial refused reasoning"), text("partial refused text"),
                                                stop="refusal", stop_details=NS(category="bio", explanation="x"))))
    session = started(policy)
    with pytest.raises(PolicyError, match="bio"):
        policy.step(session)
    texts = [e.get("text", "") for e in session.log if e.get("action_class") == "reasoning"]
    assert not any("partial refused" in t for t in texts), texts
    assert any("refusal" in t and "bio" in t for t in texts)


def test_pause_turn_is_resent_unchanged():
    policy, client = make_policy(lambda c, kw: None)
    session = started(policy)
    client.responder = script(message(thinking("working"), stop="pause_turn"),
                              message(tool("evaluate", {"config": center(session), "rationale": "go"})))
    policy.step(session)
    assert session.t == 1 and len(client.calls) == 2
    a, b = client.calls
    assert b["messages"][:-1] == a["messages"]
    assert json.loads(b["messages"][-1])["role"] == "assistant"
    assert policy.usage()["pause_turns"] == 1


def test_empty_response_is_resent_without_a_new_user_turn():
    policy, client = make_policy(lambda c, kw: None)
    session = started(policy)
    client.responder = script(message(stop="end_turn"),
                              message(tool("evaluate", {"config": center(session), "rationale": "go"})))
    policy.step(session)
    assert session.t == 1
    assert client.calls[1]["messages"] == client.calls[0]["messages"]


def test_empty_responses_after_a_context_message_keep_the_next_step_valid():
    """Regression: a step abandoned on empty replies left [user, system] last, so the next step sent
    [user, system, user] (a system message must be last or followed by an assistant turn)."""
    bot = SuggestEvaluateBot()

    empty: list[int] = []

    def responder(client: FakeClient, kw: dict) -> Any:
        if any(m["role"] == "system" for m in kw["messages"]) and len(client.calls) <= 6:
            empty.append(len(client.calls) - 1)
            return message(stop="end_turn")   # step 2: nothing but empty replies
        return bot(client, kw)

    policy, client = make_policy(responder)
    camp = run_campaign(policy, budget=3, context_schedule={1: "Operator: keep x1 below 5."})
    assert camp.result.policy_error is None and camp.result.forced_commits == 1
    for call in client.calls:
        assert_system_placement(call["kw"]["messages"])
    final = client.calls[-1]["kw"]["messages"]
    assert sum(m["role"] == "system" for m in final) == 1
    roles = [m["role"] for m in final]
    assert all(not (a == b == "user") for a, b in zip(roles, roles[1:])), roles
    # the harness note about step 2's fallback reaches the model before step 3
    texts = [b.get("text", "") for m in final if m["role"] == "user" for b in m["content"]]
    assert any("Step 2 ended without an evaluation from you" in t for t in texts)
    assert len(empty) == 4   # max_nudges + 1 empty replies, all with the same body
    assert len({tuple(client.calls[i]["messages"]) for i in empty}) == 1
    assert_append_only([c for i, c in enumerate(client.calls) if i not in empty])


# =========================================================================== context injection


def test_context_arrives_as_system_message_after_the_tool_result_user_message():
    bot = SuggestEvaluateBot()
    policy, client = make_policy(bot)
    camp = run_campaign(policy, budget=3, context_schedule={1: "From now on keep x1 below 5."})
    assert camp.result.policy_error is None
    # the first request of step 2 (call index 2) ends with [user (tool results + status), system]
    msgs = client.calls[2]["kw"]["messages"]
    user, system = msgs[-2], msgs[-1]
    assert system["role"] == "system" and isinstance(system["content"], str)
    assert "keep x1 below 5" in system["content"] and "instructor" in system["content"]
    assert "after 1 of 3 evaluations" in system["content"]
    assert user["role"] == "user" and user["content"][0]["type"] == "tool_result"
    assert sum(m["role"] == "system" for m in client.calls[-1]["kw"]["messages"]) == 1
    assert_append_only(client.calls)


def test_context_at_t0_follows_the_first_user_message():
    policy, client = make_policy(SuggestEvaluateBot())
    run_campaign(policy, budget=1, context_schedule={0: {"text": "Budget is tight.", "source": "manager"}})
    msgs = client.calls[0]["kw"]["messages"]
    assert [m["role"] for m in msgs] == ["user", "system"]
    assert "manager" in msgs[1]["content"] and "Budget is tight." in msgs[1]["content"]


def test_system_role_rejection_falls_back_to_user_text():
    bot = SuggestEvaluateBot()
    rejections = []

    def responder(client: FakeClient, kw: dict) -> Any:
        if any(m["role"] == "system" for m in kw["messages"]):
            err = FakeAPIError("messages.3: role 'system' is not supported on this model", 400)
            rejections.append(err)
            return err
        return bot(client, kw)

    policy, client = make_policy(responder)
    camp = run_campaign(policy, budget=4, context_schedule={1: "First instruction.", 2: "Second instruction."})
    assert camp.result.policy_error is None and camp.result.forced_commits == 0
    assert len(rejections) == 1 and camp.result.usage["system_role_fallback"] is True
    ok = client.ok_calls
    assert_append_only(ok)
    # both instructions became text blocks in the user turn that opens their step
    blocks = [b["text"] for c in ok for b in last_user(c["kw"])["content"] if b.get("type") == "text"]
    for instruction in ("First instruction.", "Second instruction."):
        hit = [b for b in blocks if instruction in b]
        assert hit and hit[0].startswith("<system-reminder>")
    # the failed request was the previous successful one plus the staged turn: nothing leaked into history
    failed = next(c for c in client.calls if not c["ok"])
    retry = client.calls[client.calls.index(failed) + 1]
    assert retry["messages"][:-1] == failed["messages"][:-2]


def assert_system_placement(messages: list[dict]) -> None:
    """claude-api skill: a role:system message follows a user message and is last or followed by an assistant."""
    for i, m in enumerate(messages):
        if m["role"] != "system":
            continue
        assert i > 0 and messages[i - 1]["role"] == "user", f"messages[{i}]: system not after a user message"
        assert i == len(messages) - 1 or messages[i + 1]["role"] == "assistant", \
            f"messages[{i}]: system followed by {messages[i + 1]['role']}"


def test_several_context_items_for_one_step_share_one_system_message():
    """Regression (SDK-1): two items due before the same step produced [user, system, system]."""
    bot = SuggestEvaluateBot()
    policy, client = make_policy(bot)
    camp = run_campaign(policy, budget=3, context_schedule={
        0: {"text": "Opening note.", "source": "manager"},
        1: ["Operator: context A.", "Operator: context B."]})
    assert camp.result.policy_error is None and camp.result.usage["system_role_fallback"] is False
    for call in client.calls:
        assert_system_placement(call["kw"]["messages"])
    final = client.calls[-1]["kw"]["messages"]
    systems = [m["content"] for m in final if m["role"] == "system"]
    assert len(systems) == 2 and "Opening note." in systems[0]
    a, b = systems[1].index("Operator: context A."), systems[1].index("Operator: context B.")
    assert a < b   # both items, in arrival order, in one message
    assert_append_only(client.calls)


def test_unrelated_400_is_not_mistaken_for_a_system_role_rejection():
    """Regression (SDK-2): any 400 mentioning 'system' or 'role' switched to <system-reminder> text for good."""
    bot = SuggestEvaluateBot()
    errors = []

    def responder(client: FakeClient, kw: dict) -> Any:
        if any(m["role"] == "system" for m in kw["messages"]):
            err = FakeAPIError("system: text content blocks must contain non-whitespace text", 400)
            errors.append(err)
            return err
        return bot(client, kw)

    policy, client = make_policy(responder)
    camp = run_campaign(policy, budget=2, context_schedule={1: "Operator: context."})
    assert camp.result.usage["system_role_fallback"] is False
    assert len(errors) == 1   # not re-sent as if role:system were the problem
    assert "non-whitespace" in (camp.result.policy_error or "")
    assert not any("<system-reminder>" in c["messages"][-1] for c in client.calls)


@pytest.mark.parametrize("msg", [
    "messages.3: role 'system' is not supported on this model",
    'messages: role "system" is not supported for this model',
    "Messages with role system are not supported on claude-sonnet-5",
])
def test_system_role_rejection_wordings_are_recognized(msg):
    from agentic_bo.sara.policies.llm import _rejects_system_role
    assert _rejects_system_role(FakeAPIError(msg, 400))
    assert not _rejects_system_role(FakeAPIError(msg, 500))
    for other in ("system: text content blocks must contain non-whitespace text",
                  "messages.2: roles must alternate between user and assistant",
                  "messages.4: a message with role 'system' must follow a user message",
                  "tools.0: role-based access is not supported"):
        assert not _rejects_system_role(FakeAPIError(other, 400)), other


def test_other_api_errors_raise_policy_error():
    policy, client = make_policy(lambda c, kw: FakeAPIError("overloaded", 529))
    session = started(policy)
    with pytest.raises(PolicyError, match="Claude API error.*overloaded"):
        policy.step(session)
    assert len(client.calls) == 1


# =========================================================================== SDK robustness


def test_unexpected_kwarg_moves_into_extra_body_and_is_remembered():
    bot = SuggestEvaluateBot()
    raised = []

    def responder(client: FakeClient, kw: dict) -> Any:
        for key in ("fallbacks", "context_management"):
            if key in kw:
                raised.append(key)
                raise TypeError(f"Messages.stream() got an unexpected keyword argument '{key}'")
        return bot(client, kw)

    policy, client = make_policy(responder)
    camp = run_campaign(policy, budget=2)
    assert camp.result.policy_error is None and camp.result.n_evals == 2
    assert raised == ["fallbacks", "context_management"]
    kw = client.ok_calls[-1]["kw"]
    assert "fallbacks" not in kw and "context_management" not in kw
    assert kw["extra_body"] == {"fallbacks": "default",
                                "context_management": policy._context_management}
    assert kw["betas"] == [FALLBACK_BETA, COMPACTION_BETA]
    assert camp.result.usage["moved_kwargs"] == {"context_management": "extra_body", "fallbacks": "extra_body"}
    assert len(client.ok_calls) == 4


def test_unexpected_betas_kwarg_moves_into_headers():
    bot = SuggestEvaluateBot()

    def responder(client: FakeClient, kw: dict) -> Any:
        if "betas" in kw:
            raise TypeError("stream() got an unexpected keyword argument 'betas'")
        return bot(client, kw)

    policy, client = make_policy(responder)
    policy.step(started(policy))
    kw = client.ok_calls[-1]["kw"]
    assert kw["extra_headers"] == {"anthropic-beta": f"{FALLBACK_BETA},{COMPACTION_BETA}"}


def test_other_type_errors_are_not_retried():
    def responder(client: FakeClient, kw: dict) -> Any:
        raise TypeError("unsupported operand type(s)")

    policy, client = make_policy(responder)
    with pytest.raises(PolicyError, match="rejected the request"):
        policy.step(started(policy))
    assert len(client.calls) == 1


def test_stream_value_error_reissues_the_same_request():
    bot = SuggestEvaluateBot()
    state = {"n": 0}

    def responder(client: FakeClient, kw: dict) -> Any:
        state["n"] += 1
        if state["n"] == 1:
            return ValueError("Unable to parse tool parameter JSON from model")
        return bot(client, kw)

    policy, client = make_policy(responder)
    session = started(policy)
    policy.step(session)
    assert session.t == 1 and policy.usage()["stream_retries"] == 1
    assert client.calls[0]["messages"] == client.calls[1]["messages"] and not client.calls[0]["ok"]
    assert_append_only(client.ok_calls)


def test_stream_value_errors_are_capped():
    policy, client = make_policy(lambda c, kw: ValueError("bad json"))
    with pytest.raises(PolicyError, match="could not be parsed"):
        policy.step(started(policy))
    assert len(client.calls) == MAX_STREAM_RETRIES + 1


def test_typed_api_errors_are_never_treated_as_json_errors():
    class APIValueError(ValueError):  # an SDK error that happens to subclass ValueError
        status_code = 500

    policy, client = make_policy(lambda c, kw: APIValueError("server error"))
    with pytest.raises(PolicyError, match="Claude API error"):
        policy.step(started(policy))
    assert len(client.calls) == 1


def test_eager_streaming_invalid_input_returns_invalid_json_error():
    policy, client = make_policy(lambda c, kw: None)
    session = started(policy)
    bad = tool("evaluate", '{"config": {"x0": 1.0, "x1": ')
    client.responder = script(message(bad), message(tool("evaluate", {"config": center(session), "rationale": "ok"})))
    policy.step(session)
    assert session.t == 1
    (res,) = tool_results(last_user(client.calls[1]["kw"]))
    assert res["is_error"] and res["tool_use_id"] == bad.id
    payload = json.loads(res["content"])
    assert payload["INVALID_JSON"].startswith('{"config"') and "JSON object" in payload["error"]


def test_schema_invalid_input_is_an_error_result():
    policy, client = make_policy(lambda c, kw: None)
    session = started(policy)
    client.responder = script(message(tool("evaluate", {"config": {"x0": 1.0}, "rationale": "missing x1"})),
                              message(tool("evaluate", {"config": center(session), "rationale": "ok"})))
    policy.step(session)
    (res,) = tool_results(last_user(client.calls[1]["kw"]))
    assert res["is_error"] and "invalid arguments for evaluate" in res["content"]
    assert session.t == 1


def test_hard_limit_force_commit_mid_message_ends_the_step():
    policy, client = make_policy(lambda c, kw: None)
    session = started(policy, hard_actions_per_step=2, soft_actions_per_step=1)
    calls = [tool("status", {}) for _ in range(4)]
    client.responder = script(message(*calls))
    policy.step(session)
    assert session.t == 1 and session.records[0]["forced"] and len(client.calls) == 1
    results = policy._staged_user
    assert [r["tool_use_id"] for r in results] == [c.id for c in calls]
    assert "warning" in json.loads(results[1]["content"])  # past the soft limit
    assert results[2]["is_error"] and "Action limit exceeded" in results[2]["content"]
    assert "Fallback evaluation:" in results[2]["content"]
    assert not results[3].get("is_error")  # after the commit: counts toward the next step


def test_fallback_switch_point_drops_declined_blocks():
    policy, client = make_policy(lambda c, kw: None)
    session = started(policy)
    cfg = center(session)
    declined = tool("evaluate", {"config": cfg, "rationale": "declined model"})
    served = tool("evaluate", {"config": {**cfg, "x0": 2.0}, "rationale": "fallback model"})
    client.responder = script(message(thinking("declined reasoning"), declined,
                                      NS(type="fallback", to=NS(model="claude-opus-5")), text("continuing"), served,
                                      use=usage(10, 5, iterations=[
                                          NS(type="message", input_tokens=10, output_tokens=2,
                                             cache_read_input_tokens=0, cache_creation_input_tokens=0),
                                          NS(type="fallback_message", model="claude-opus-5", input_tokens=10,
                                             output_tokens=5, cache_read_input_tokens=0,
                                             cache_creation_input_tokens=0)])))
    policy.step(session)
    assert session.t == 1 and session.records[0]["rationale"] == "fallback model"
    content = policy.messages[-1]["content"]
    assert [b.type for b in content] == ["fallback", "text", "tool_use"] and content[-1] is served
    u = policy.usage()
    assert u["fallback_responses"] == 1 and u["input_tokens"] == 20 and u["output_tokens"] == 7
    assert u["cost_usd"] == pytest.approx(estimate_cost(DEFAULT_MODEL, 10, 2) + estimate_cost("claude-opus-5", 10, 5))


# =========================================================================== usage, traces, transcript


def test_usage_sums_iterations_and_counts_compactions():
    policy, client = make_policy(lambda c, kw: None)
    session = started(policy)
    it = [NS(type="compaction", input_tokens=5000, output_tokens=300, cache_read_input_tokens=0,
             cache_creation_input_tokens=0),
          NS(type="message", input_tokens=200, output_tokens=50, cache_read_input_tokens=1000,
             cache_creation_input_tokens=100)]
    client.responder = script(
        message(NS(type="compaction", content="Summary: incumbent trial 3 ..."),
                tool("evaluate", {"config": center(session), "rationale": "r"}), use=usage(200, 50, iterations=it)))
    policy.step(session)
    u = policy.usage()
    assert u["compactions"] == 1 and u["api_calls"] == 1
    assert (u["input_tokens"], u["output_tokens"], u["cache_read_input_tokens"],
            u["cache_creation_input_tokens"]) == (5200, 350, 1000, 100)
    assert u["cost_usd"] == pytest.approx((5200 * 4 + 350 * 20 + 1000 * 0.2 + 100 * 5) / 1e6)
    assert any(e.get("kind") == "compaction" and e["text"].startswith("Summary") for e in session.log)


def test_estimate_cost_prices():
    assert estimate_cost(DEFAULT_MODEL, 1_000_000, 0) == pytest.approx(4.0)
    assert estimate_cost(DEFAULT_MODEL, 0, 1_000_000) == pytest.approx(20.0)
    assert estimate_cost(DEFAULT_MODEL, 0, 0, 1_000_000) == pytest.approx(0.2)
    assert estimate_cost(DEFAULT_MODEL, 0, 0, 0, 1_000_000) == pytest.approx(5.0)
    assert estimate_cost("unknown-model", 1_000_000) == pytest.approx(4.0)


def test_transcript_jsonl(tmp_path):
    path = tmp_path / "logs" / "transcript.jsonl"
    policy, client = make_policy(SuggestEvaluateBot(), transcript_path=path)
    run_campaign(policy, budget=2)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "finish"
    assert kinds.count("request") == kinds.count("response") == len(client.calls) == 4
    start = events[0]
    assert start["system"][0]["text"] == load_prompt("SYSTEM.md") and len(start["tools"]) == len(TOOLSETS["full"]) - 2
    requests = [e for e in events if e["event"] == "request"]
    assert all("system" not in e and "tools" not in e and "system" not in e["params"] for e in requests)
    assert requests[0]["params"]["model"] == DEFAULT_MODEL
    # request deltas + response messages reconstruct the history exactly once
    non_assistant = [m for m in policy.messages if m["role"] != "assistant"]
    assert sum(len(e["new_messages"]) for e in requests) == len(non_assistant)
    responses = [e for e in events if e["event"] == "response"]
    assert all(e["message"]["stop_reason"] == "tool_use" for e in responses)
    assert events[-1]["usage"]["api_calls"] == 4


def test_policy_reuse_resets_the_conversation():
    bot = SuggestEvaluateBot()
    policy, client = make_policy(bot)
    run_campaign(policy, budget=2)
    n_first = len(client.calls)
    run_campaign(policy, budget=2)
    second = client.calls[n_first]
    assert len(second["kw"]["messages"]) == 1
    assert policy.usage()["api_calls"] == 4


def test_stop_ends_the_campaign():
    policy, client = make_policy(lambda c, kw: None)
    state = {"n": 0}

    def responder(client: FakeClient, kw: dict) -> Any:
        state["n"] += 1
        if state["n"] == 1:
            return message(tool("evaluate", {"config": {"x0": 0.0, "x1": 5.0}, "rationale": "probe"}))
        return message(tool("stop", {"reason": "good enough"}))

    client.responder = responder
    res = run_campaign(policy, budget=5, allow_early_stop=True).result
    assert res.stopped_early and res.n_evals == 1 and res.stop_reason == "good enough"
    assert len(client.calls) == 2 and res.forced_commits == 0


# =========================================================================== llm_only


def test_llm_only_toolset_campaign():
    counter = itertools.count()

    def responder(client: FakeClient, kw: dict) -> Any:
        i = next(counter)
        return message(thinking("reasoning directly"),
                       tool("evaluate", {"config": {"x0": -5.0 + 3.0 * i, "x1": 1.0 + 2.0 * i}, "rationale": "grid"}))

    client = FakeClient(responder)
    policy = get_policy("llm_only", client=client)
    camp = run_campaign(policy, budget=3)
    res = camp.result
    assert res.policy == "llm_only" and res.policy_error is None and res.n_evals == 3 and res.forced_commits == 0
    assert camp.session.toolset == "llm_only"
    kw = client.calls[0]["kw"]
    assert [t["name"] for t in kw["tools"]] == ["evaluate", "incumbent", "record_note", "status", "trials"]
    assert kw["system"] == [{"type": "text", "text": load_prompt("SYSTEM_LLM_ONLY.md"),
                             "cache_control": {"type": "ephemeral"}}]
    assert_append_only(client.calls)


# =========================================================================== prompts


PROMPT_FILES = ("SYSTEM.md", "LENZ_REF.md", "SYSTEM_LLM_ONLY.md", "COMPACTION.md")
_ACRONYMS = {"JSON", "RMS"}


@pytest.mark.parametrize("name", PROMPT_FILES)
def test_prompts_exist_and_use_calm_language(name):
    body = load_prompt(name)
    assert len(body) > 500 and (PROMPTS_DIR / name).is_file()
    shouting = set(re.findall(r"\b[A-Z]{3,}\b", body)) - _ACRONYMS
    assert not shouting, f"all-caps emphasis in {name}: {shouting}"
    assert "!" not in body.replace("!=", "")
    for task_specific in ("branin", "hartmann", "ackley", "rosenbrock", "currin", "suzuki", "reaction"):
        assert task_specific not in body.lower()


def test_lenz_reference_covers_every_tool_and_acquisition_function():
    from agentic_bo.lenz.acquisition import ACQF_NAMES
    ref = load_prompt("LENZ_REF.md")
    for name in TOOLSETS["full"]:
        assert f"### {name}" in ref, name
    for acqf in ACQF_NAMES:
        assert f"`{acqf}`" in ref or acqf in ref, acqf
    for arg in ("last", "sort_by", "configs", "acqfs", "bounds", "beta", "params", "objectives", "constraints",
                "q", "around", "radius", "rationale", "expectation", "duplicate_of", "cv_r2", "lengthscales"):
        assert f"`{arg}`" in ref or arg in ref, arg


def test_system_prompts_cover_the_directives():
    sara = load_prompt("SYSTEM.md").lower()
    for phrase in ("not an autopilot", "opening", "diagnostics", "anti-patterns", "rationale", "simple regret",
                   "cumulative regret", "duplicate", "never deletes data", "basins", "system messages"):
        assert phrase in sara, phrase
    direct = load_prompt("SYSTEM_LLM_ONLY.md")
    for name in TOOLSETS["llm_only"]:
        assert f"`{name}`" in direct, name
    for absent in ("`suggest`", "`diagnostics`", "`predict`", "`set_bounds`"):
        assert absent not in direct
    compaction = load_prompt("COMPACTION.md").lower()
    for phrase in ("instructions", "incumbent", "lenz configuration", "hypotheses", "sensitive", "model trust",
                   "plan", "budget", "`trials`"):
        assert phrase in compaction, phrase


def _line_with(body: str, anchor: str) -> str:
    lines = [ln for ln in body.splitlines() if anchor in ln]
    assert lines, f"no line mentions {anchor!r}"
    return " ".join(lines).lower()


@pytest.mark.parametrize("name", ["SYSTEM.md", "LENZ_REF.md"])
def test_prompts_carry_the_operational_guidance(name):
    """Regression: guidance measured on lenz itself (scratch runs on goldstein_price, branin, saturated
    integer regions) was missing: what to do with duplicate_of, cv_r2 on heavy-tailed objectives, and
    the fitted noise_std, which absorbs misfit on noiseless functions."""
    body = load_prompt(name)
    ref = name == "LENZ_REF.md"
    dup = _line_with(body, "`duplicate_of` names" if ref else "duplicate_of")
    assert "radius" in dup and "global" in dup, dup
    cv = _line_with(body, "`cv_r2`:" if ref else "cv_r2")
    assert "heavy-tailed" in cv and "expectation" in cv, cv
    noise = _line_with(body, "`noise_std` and" if ref else "noise_std")
    assert "brief" in noise and "misfit" in noise, noise


def test_lenz_reference_matches_the_tool_definitions():
    """Regression: every argument of every tool schema is documented in its LENZ_REF section."""
    ref = load_prompt("LENZ_REF.md")
    sections = dict(re.findall(r"^### (\w+)\n(.*?)(?=^### |^## |\Z)", ref, flags=re.S | re.M))
    session = AgentSession(get_problem("branin"), CampaignConfig(budget=3, allow_early_stop=True))
    for d in session.tool_definitions():
        sec = sections[d["name"]]
        args_line = next((ln for ln in sec.splitlines() if ln.startswith(("Arguments", "Available only"))), "")
        for arg in d["input_schema"].get("properties", {}):
            assert f"`{arg}`" in args_line, (d["name"], arg)
    assert "`notes`" in sections["set_acqf"]
    assert "before sorting" in sections["trials"]


def test_compaction_instructions_ask_for_text_only():
    """Regression: custom compaction instructions replace the server's summarization prompt whole
    (claude-api skill), so they must ask for a plain-text summary and no tool call themselves."""
    compaction = load_prompt("COMPACTION.md").lower()
    assert "text only" in compaction and "tool call" in compaction


def _load_example(name: str):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_examples_refuse_a_client_without_credentials(monkeypatch, capsys):
    """Regression (reconciliation): anthropic>=1.13 builds a client without any credentials and fails
    only at the first request, so the Claude examples started a campaign that ended in harness
    fallbacks (run_sara_claude exit 1, dynamic_reconfiguration --llm exit 0 with a bogus report)."""
    class FakeClient:
        def __init__(self, **creds):
            self.api_key = creds.get("api_key")
            self.auth_token = creds.get("auth_token")
            self.credentials = creds.get("credentials")

    for creds, ok in (({}, False), ({"api_key": "k"}, True), ({"credentials": object()}, True)):
        monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=lambda c=creds: FakeClient(**c)))
        run = _load_example("run_sara_claude")
        assert (run.make_client() is not None) == ok, creds
        assert ("No API calls were made" in capsys.readouterr().err) != ok
        dyn = _load_example("dynamic_reconfiguration")
        if ok:
            assert dyn.make_llm_policy("high") is not None
        else:
            with pytest.raises(SystemExit, match="No API calls were made"):
                dyn.make_llm_policy("high")
