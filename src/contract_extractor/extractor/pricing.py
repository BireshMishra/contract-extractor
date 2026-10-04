"""Price per million tokens in USD, as (input, output)."""

PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    # Anthropic's published rate for Claude Sonnet 5.5 (cached 2026-09-25).
    "claude-sonnet-5-5": (2.00, 10.00),
    # OpenAI's published list price for GPT-4.1, entered from memory and NOT checked
    # against OpenAI's pricing page. Verify it before trusting the cost column.
    "gpt-4.1": (2.00, 8.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """(input tokens x input price) + (output tokens x output price)."""
    input_price, output_price = PRICES_PER_MILLION[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
