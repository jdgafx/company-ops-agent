"""Tool-use loop: ask the model, run the tools it requests, feed results back."""

from __future__ import annotations

import json
import time
from typing import Any

from ops_agent.config import Settings

SYSTEM_PROMPT = """You are the operations assistant for Lumen Outdoor Co., a fictional outdoor \
gear company.

Rules:
- Use the tools for any fact about customers, orders or inventory. Never answer those from memory.
- Never invent ids, names, dates or numbers. Only report what the tools returned.
- Call create_ticket only when the user explicitly asks you to open a ticket, and only with \
confirm=true.
- If the data you need is missing or a tool fails, say so plainly instead of guessing.
- Keep answers short and factual."""

STEP_LIMIT_ANSWER = "I reached the step limit before finishing, so I cannot give a complete answer."


def _tool_defs(tools: Any) -> list[dict[str, Any]]:
    return [
        {"name": s.name, "description": s.description, "input_schema": s.input_schema}
        for s in tools.specs()
    ]


def _text(content: list[Any]) -> str:
    return "\n".join(b.text for b in content if b.type == "text" and b.text).strip()


def run_agent(question: str, *, tools: Any, client: Any, settings: Settings) -> dict[str, Any]:
    started = time.perf_counter()
    tool_defs = _tool_defs(tools)
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    trace: list[dict[str, Any]] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    steps = 0

    def finish(answer: str, stop_reason: str) -> dict[str, Any]:
        return {
            "answer": answer,
            "stop_reason": stop_reason,
            "steps": steps,
            "trace": trace,
            "usage": usage,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    while steps < settings.max_steps:
        steps += 1
        # No temperature/top_p: claude-haiku-5-5 rejects non-default sampling params.
        response = client.messages.create(
            model=settings.model,
            max_tokens=settings.max_tokens,
            system=SYSTEM_PROMPT,
            tools=tool_defs,
            messages=messages,
        )
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens

        calls = [b for b in response.content if b.type == "tool_use"]
        if not calls:
            answer = _text(response.content)
            if not answer and response.stop_reason == "refusal":
                answer = "The model declined to answer this request."
            return finish(answer, "end_turn")

        if steps == settings.max_steps:
            break

        # Replay the full assistant content (incl. thinking blocks) unchanged.
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for call in calls:
            arguments = dict(call.input or {})
            t0 = time.perf_counter()
            result = tools.call(call.name, arguments)
            latency = (time.perf_counter() - t0) * 1000
            trace.append(
                {
                    "step": steps,
                    "tool": call.name,
                    "arguments": arguments,
                    "result": result,
                    "latency_ms": latency,
                }
            )
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": json.dumps(result, default=str),
            }
            if not result.get("ok", True):
                block["is_error"] = True
            results.append(block)
        messages.append({"role": "user", "content": results})

    return finish(STEP_LIMIT_ANSWER, "step_limit")
