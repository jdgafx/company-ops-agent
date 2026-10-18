"""Tool registry shared by the agent loop and the MCP server.

Every tool is a plain function returning a JSON-serialisable dict. Calls never
raise to the caller: unknown tools and bad arguments come back as
{"ok": False, "error": {"code": ..., "message": ...}}.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., dict[str, Any]]
    write: bool = False  # write tools need confirm=true in their arguments


@dataclass
class ToolRegistry:
    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self._tools[spec.name] = spec

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def call(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        spec = self._tools.get(name)
        if spec is None:
            return _error("unknown_tool", f"no tool named {name!r}")
        args = arguments or {}
        if not isinstance(args, dict):
            return _error("bad_arguments", "arguments must be an object")
        if spec.write and args.get("confirm") is not True:
            return _error("confirmation_required", "write tools need confirm=true")
        try:
            result = spec.fn(**{k: v for k, v in args.items() if k != "confirm"})
        except TypeError as exc:  # wrong or missing keyword arguments
            return _error("bad_arguments", str(exc))
        except Exception as exc:  # noqa: BLE001 - tool bugs must not crash the agent loop
            return _error("tool_failed", f"{type(exc).__name__}: {exc}")
        return {"ok": True, "data": result}


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}
