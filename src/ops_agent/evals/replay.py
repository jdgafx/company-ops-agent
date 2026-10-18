"""Record and replay model calls so the eval gate runs offline, with no key and no spend.

A recording is one JSON file per task: {"task": id, "steps": [{"request": {...}, "response":
{...}}]}. "request" is the exact kwargs the agent sent to messages.create; "response" holds the
content blocks, stop_reason and usage. Recordings never contain credentials: the key lives on the
client object, not in the request kwargs.

Replay compares every incoming request with the recorded one as canonical JSON. A changed tool
output, system prompt or tool schema therefore raises ReplayMismatch, which run_task turns into a
failed task. That comparison is what makes the replayed gate catch regressions without a model.
"""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace as NS
from typing import Any


class ReplayError(Exception):
    """Base class for replay failures (a failed task, never a crash of the whole run)."""


class ReplayMismatch(ReplayError):
    """The agent's request differs from the recording, or ran past the recorded steps."""


def recording_path(recordings_dir: pathlib.Path, task_id: str) -> pathlib.Path:
    return pathlib.Path(recordings_dir) / f"{task_id}.json"


def to_plain(obj: Any) -> Any:
    """Convert SDK models, namespaces, dicts and lists to plain JSON-compatible data."""
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return to_plain(obj.model_dump(exclude_none=True))
    if isinstance(obj, NS):
        return to_plain(vars(obj))
    return obj


def _slim(block: Any) -> Any:
    """Keep only the fields the agent loop uses, so live SDK blocks and replayed ones compare equal."""
    if isinstance(block, dict):
        if block.get("type") == "text":
            return {"type": "text", "text": block.get("text", "")}
        if block.get("type") == "tool_use":
            return {k: block.get(k) for k in ("type", "id", "name", "input")}
    return block


def normalize_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = to_plain(kwargs)  # every kwarg, so an added option such as tool_choice is a mismatch
    for message in out.get("messages", []):
        if isinstance(message.get("content"), list):
            message["content"] = [_slim(b) for b in message["content"]]
    return out


def normalize_response(response: Any) -> dict[str, Any]:
    usage = response.usage
    return {
        "content": [_slim(to_plain(b)) for b in response.content],
        "stop_reason": response.stop_reason,
        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
    }


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _to_response(data: dict[str, Any]) -> Any:
    blocks = [NS(**b) for b in data["content"]]
    return NS(content=blocks, stop_reason=data["stop_reason"], usage=NS(**data["usage"]))


class _Messages:
    def __init__(self, owner: Any):
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        return self._owner.create(**kwargs)


class Recorder:
    """Wraps a live client; begin(task_id) starts a fresh recording file for that task."""

    def __init__(self, client: Any, recordings_dir: pathlib.Path):
        self._client = client
        self._dir = pathlib.Path(recordings_dir)
        self._task = ""
        self._steps: list[dict[str, Any]] = []
        self.messages = _Messages(self)

    def begin(self, task_id: str) -> None:
        self._task, self._steps = task_id, []

    def create(self, **kwargs: Any) -> Any:
        response = self._client.messages.create(**kwargs)
        self._steps.append(
            {"request": normalize_request(kwargs), "response": normalize_response(response)}
        )
        path = recording_path(self._dir, self._task)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"task": self._task, "steps": self._steps}, indent=2) + "\n"
        )
        return response


class ReplayClient:
    """Fake client that answers from recordings and checks every request against them."""

    def __init__(self, recordings_dir: pathlib.Path):
        self._dir = pathlib.Path(recordings_dir)
        self._task = ""
        self._steps: list[dict[str, Any]] | None = None
        self._next = 0
        self.model: str | None = None  # model of the first replayed request, for the report
        self.messages = _Messages(self)

    def begin(self, task_id: str) -> None:
        self._task, self._next = task_id, 0
        path = recording_path(self._dir, task_id)
        self._steps = json.loads(path.read_text())["steps"] if path.exists() else None

    def create(self, **kwargs: Any) -> Any:
        task, n = self._task, self._next + 1
        if self._steps is None:
            raise ReplayError(
                f"task {task}: no recording at {recording_path(self._dir, task)} "
                "(run the eval with --record to create it)"
            )
        if self._next >= len(self._steps):
            raise ReplayMismatch(
                f"task {task} step {n}: more model calls than the {len(self._steps)} recorded "
                "(the agent flow changed)"
            )
        step = self._steps[self._next]
        got, want = normalize_request(kwargs), step["request"]
        if canonical(got) != canonical(want):
            keys = sorted(k for k in got.keys() | want.keys()
                          if canonical(got.get(k)) != canonical(want.get(k)))
            raise ReplayMismatch(
                f"task {task} step {n}: request differs from recording "
                f"(tool output, prompt or tool schema changed; differs in: {', '.join(keys)})"
            )
        self._next += 1
        if self.model is None:
            self.model = want.get("model")
        return _to_response(step["response"])
