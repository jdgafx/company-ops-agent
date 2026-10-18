"""MCP server exposing every tool in a ToolRegistry (stdio by default, --http optional).

Built on the low-level mcp Server so each tool keeps its exact input_schema.
All calls go through registry.call, so the write-confirm rule lives in one place.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import anyio
import mcp_types as types
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server

from ops_agent.tools import ToolRegistry

HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8765


def create_server(registry: ToolRegistry) -> Server[Any]:
    async def list_tools(
        ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(name=s.name, description=s.description, input_schema=s.input_schema)
                for s in registry.specs()
            ]
        )

    async def call_tool(
        ctx: ServerRequestContext[Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        result = registry.call(params.name, params.arguments)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result))],
            is_error=not result.get("ok", False),
        )

    return Server("ops-agent", on_list_tools=list_tools, on_call_tool=call_tool)


async def _run_stdio(server: Server[Any]) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def _run_http(server: Server[Any]) -> None:
    import uvicorn

    app = server.streamable_http_app(host=HTTP_HOST)
    uvicorn.run(app, host=HTTP_HOST, port=HTTP_PORT)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Company ops agent MCP server")
    parser.add_argument("--http", action="store_true", help=f"serve streamable HTTP on {HTTP_HOST}:{HTTP_PORT}")
    args = parser.parse_args(argv)

    from ops_agent.catalog import build_registry
    from ops_agent.store import Store

    tickets = os.environ.get("OPS_TICKETS_PATH")
    store = Store(Path(os.environ.get("OPS_DATA_DIR", "./data")), Path(tickets) if tickets else None)
    server = create_server(build_registry(store))
    if args.http:
        _run_http(server)
    else:
        anyio.run(_run_stdio, server)


if __name__ == "__main__":
    main()
