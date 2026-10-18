"""Per-token prices of the pinned model, for the cost_usd field of an answer.

Source: OpenRouter public models list, https://openrouter.ai/api/v1/models, read 2026-10-18,
entry "anthropic/claude-haiku-5.5": prompt 0.0000001 and completion 0.0000005 USD per token;
for a call whose prompt has 100000 tokens or more the entry lists an override of 0.0000005 and
0.0000025. Prices are looked up per model id; an id that is not listed here has no known price
and cost_usd is None (never a guess).
"""

from __future__ import annotations

# model id -> (prompt USD/token, completion USD/token, long-prompt threshold in tokens,
#              long-prompt prompt USD/token, long-prompt completion USD/token)
PRICES: dict[str, tuple[float, float, int, float, float]] = {
    "anthropic/claude-haiku-5.5": (0.0000001, 0.0000005, 100_000, 0.0000005, 0.0000025),
}


def call_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """USD cost of one model call, or None when the model's price is not known."""
    price = PRICES.get(model)
    if price is None:
        return None
    p_in, p_out, threshold, long_in, long_out = price
    if input_tokens >= threshold:
        p_in, p_out = long_in, long_out
    return input_tokens * p_in + output_tokens * p_out
