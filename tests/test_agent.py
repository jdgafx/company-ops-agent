import json
from dataclasses import dataclass, field
from types import SimpleNamespace as NS

import pytest

from ops_agent.agent import run_agent
from ops_agent.config import Settings
from ops_agent.tools import ToolRegistry, ToolSpec


def text(t):
    return NS(type="text", text=t)


def use(id_, name, **inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def reply(content, stop, i=10, o=5):
    return NS(content=content, stop_reason=stop, usage=NS(input_tokens=i, output_tokens=o))


@dataclass
class FakeClient:
    script: list
    calls: list = field(default_factory=list)

    def __post_init__(self):
        self.messages = self

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.script.pop(0)


def make_tools():
    reg = ToolRegistry()
    schema = {"type": "object", "properties": {"id": {"type": "string"}}}

    def get_order(id):
        if id == "bad":
            raise KeyError("no such order")
        return {"id": id, "status": "shipped"}

    reg.register(ToolSpec("get_order", "Look up an order", schema, get_order))
    reg.register(ToolSpec("create_ticket", "Open a ticket", schema, lambda id: {"ticket": id}, True))
    return reg


S = Settings(max_steps=3)


def test_direct_answer_no_tools():
    client = FakeClient([reply([text("Hello")], "end_turn")])
    out = run_agent("hi", tools=make_tools(), client=client, settings=S)
    assert out["answer"] == "Hello"
    assert out["stop_reason"] == "end_turn"
    assert out["steps"] == 1 and out["trace"] == []
    sent = client.calls[0]
    assert sent["model"] == "claude-haiku-5-5"
    assert "temperature" not in sent
    assert {t["name"] for t in sent["tools"]} == {"get_order", "create_ticket"}
    assert "Lumen Outdoor Co." in sent["system"]


def test_two_tool_chain_and_trace():
    client = FakeClient(
        [
            reply([use("t1", "get_order", id="A1")], "tool_use"),
            reply([use("t2", "get_order", id="A2")], "tool_use"),
            reply([text("Both shipped")], "end_turn"),
        ]
    )
    out = run_agent("status?", tools=make_tools(), client=client, settings=S)
    assert out["answer"] == "Both shipped" and out["steps"] == 3
    assert [(t["step"], t["tool"], t["arguments"]) for t in out["trace"]] == [
        (1, "get_order", {"id": "A1"}),
        (2, "get_order", {"id": "A2"}),
    ]
    assert out["trace"][0]["result"] == {"ok": True, "data": {"id": "A1", "status": "shipped"}}
    assert all(t["latency_ms"] >= 0 for t in out["trace"]) and out["latency_ms"] >= 0
    last = client.calls[2]["messages"]
    assert last[-1]["content"][0]["tool_use_id"] == "t2"
    assert json.loads(last[-1]["content"][0]["content"])["ok"] is True


def test_tool_error_fed_back_and_recovery():
    client = FakeClient(
        [
            reply([use("t1", "get_order", id="bad")], "tool_use"),
            reply([use("t2", "get_order", id="A1")], "tool_use"),
            reply([text("Recovered")], "end_turn"),
        ]
    )
    out = run_agent("x", tools=make_tools(), client=client, settings=S)
    assert out["answer"] == "Recovered"
    first = client.calls[1]["messages"][-1]["content"][0]
    assert first["is_error"] is True and first["tool_use_id"] == "t1"
    assert out["trace"][0]["result"]["ok"] is False
    second = client.calls[2]["messages"][-1]["content"][0]
    assert "is_error" not in second


def test_write_tool_without_confirm_is_error_result():
    client = FakeClient(
        [reply([use("t1", "create_ticket", id="Z")], "tool_use"), reply([text("ok")], "end_turn")]
    )
    run_agent("x", tools=make_tools(), client=client, settings=S)
    assert client.calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_step_cap_returns_step_limit():
    script = [reply([use(f"t{i}", "get_order", id="A")], "tool_use") for i in range(5)]
    client = FakeClient(script)
    out = run_agent("loop", tools=make_tools(), client=client, settings=S)
    assert out["stop_reason"] == "step_limit"
    assert out["steps"] == 3 and len(client.calls) == 3
    assert "limit" in out["answer"].lower()
    assert len(out["trace"]) == 2  # the third request is not executed


def test_usage_summed():
    client = FakeClient(
        [
            reply([use("t1", "get_order", id="A")], "tool_use", 100, 20),
            reply([text("done")], "end_turn", 150, 30),
        ]
    )
    out = run_agent("x", tools=make_tools(), client=client, settings=S)
    assert out["usage"] == {"input_tokens": 250, "output_tokens": 50}


def test_settings_defaults_and_env(monkeypatch):
    for k in ("OPS_AGENT_MODEL", "OPS_AGENT_MAX_STEPS", "OPS_AGENT_MAX_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    assert Settings.from_env() == Settings("claude-haiku-5-5", 8, 1024)
    monkeypatch.setenv("OPS_AGENT_MODEL", "m")
    monkeypatch.setenv("OPS_AGENT_MAX_STEPS", "4")
    monkeypatch.setenv("OPS_AGENT_MAX_TOKENS", "256")
    assert Settings.from_env() == Settings("m", 4, 256)


@pytest.mark.parametrize(
    "name,value",
    [
        ("OPS_AGENT_MAX_STEPS", "0"),
        ("OPS_AGENT_MAX_STEPS", "13"),
        ("OPS_AGENT_MAX_STEPS", "abc"),
        ("OPS_AGENT_MAX_TOKENS", "63"),
        ("OPS_AGENT_MAX_TOKENS", "4097"),
    ],
)
def test_settings_bad_env_raises(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        Settings.from_env()
