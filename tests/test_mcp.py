"""MCP server tests. In-process client: mcp.Client(server) (InMemoryTransport)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp import Client

from ops_agent.mcp_server import create_server
from ops_agent.tools import ToolRegistry, ToolSpec

_OBJ = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec("double", "Double a number.", _OBJ, lambda x: {"value": x * 2}))
    reg.register(
        ToolSpec("save", "Save a number.", _OBJ, lambda x: {"saved": x}, write=True)
    )
    return reg


def _with_client(reg: ToolRegistry, fn: Any) -> Any:
    async def run() -> Any:
        async with Client(create_server(reg)) as client:
            return await fn(client)

    return asyncio.run(run())


def test_exposes_exactly_registry_tools_with_schemas() -> None:
    reg = _registry()
    result = _with_client(reg, lambda c: c.list_tools())
    got = {t.name: t for t in result.tools}
    assert set(got) == {s.name for s in reg.specs()}
    for spec in reg.specs():
        assert got[spec.name].input_schema == spec.input_schema
        assert got[spec.name].description == spec.description


def test_call_matches_registry_call() -> None:
    reg = _registry()
    result = _with_client(reg, lambda c: c.call_tool("double", {"x": 21}))
    assert json.loads(result.content[0].text) == reg.call("double", {"x": 21})
    assert not result.is_error


def test_write_without_confirm_is_refused() -> None:
    reg = _registry()
    result = _with_client(reg, lambda c: c.call_tool("save", {"x": 1}))
    payload = json.loads(result.content[0].text)
    assert payload["error"]["code"] == "confirmation_required"
    assert result.is_error


def test_write_with_confirm_goes_through() -> None:
    result = _with_client(_registry(), lambda c: c.call_tool("save", {"x": 1, "confirm": True}))
    assert json.loads(result.content[0].text) == {"ok": True, "data": {"saved": 1}}


def test_unknown_tool() -> None:
    result = _with_client(_registry(), lambda c: c.call_tool("nope", {}))
    assert json.loads(result.content[0].text)["error"]["code"] == "unknown_tool"
    assert result.is_error


def test_real_catalog_tools_exposed(tmp_path: Any) -> None:
    from ops_agent.catalog import build_registry
    from ops_agent.store import Store

    reg = build_registry(Store(tmp_path))
    result = _with_client(reg, lambda c: c.list_tools())
    assert {t.name for t in result.tools} == {s.name for s in reg.specs()}
