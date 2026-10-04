"""Run every sample contract through both providers and print a results table.

Makes real, paid API calls: one per contract per provider (plus any retries).
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .__main__ import PACKAGE_DIR, PROJECT_ROOT, fail
from .errors import ExtractionError
from .extract import extract
from .pricing import cost_usd
from .providers import ClaudeClient, LLMClient, OpenAIClient


@dataclass
class Row:
    contract: str
    provider: str
    seconds: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None
    retries: int | None = None
    null_fields: list[str] | None = None
    error: str | None = None


def run_one(client: LLMClient, provider: str, model: str, path: Path) -> Row:
    row = Row(path.name, provider)
    try:
        result = extract(client, path.read_text(encoding="utf-8"))
    except ExtractionError as e:
        row.error = str(e)
        return row
    row.seconds = result.seconds
    row.input_tokens = result.input_tokens
    row.output_tokens = result.output_tokens
    row.cost = cost_usd(model, result.input_tokens, result.output_tokens)
    row.retries = result.retries
    row.null_fields = [
        name for name, value in result.summary.model_dump().items() if value is None
    ]
    return row


def format_table(rows: list[Row]) -> str:
    header = "| Contract | Provider | Seconds | Tokens (in/out) | Cost (USD) | Retries | Null fields |"
    lines = [header, "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        if r.error:
            lines.append(f"| {r.contract} | {r.provider} | - | - | - | - | FAILED: {r.error} |")
            continue
        nulls = ", ".join(r.null_fields) if r.null_fields else "none"
        lines.append(
            f"| {r.contract} | {r.provider} | {r.seconds:.1f} | "
            f"{r.input_tokens:,} / {r.output_tokens:,} | ${r.cost:.4f} | {r.retries} | {nulls} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m contract_extractor.extractor.benchmark",
        description="Run all sample contracts through both providers (paid API calls).",
    )
    parser.add_argument("--out", type=Path, help="also write the table to this file")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        fail("no API key found; set ANTHROPIC_API_KEY in your environment or .env")
    if not os.environ.get("OPENAI_API_KEY"):
        fail("no API key found; set OPENAI_API_KEY in your environment or .env")

    claude, openai_client = ClaudeClient(), OpenAIClient()
    providers = [("claude", claude, claude.model), ("openai", openai_client, openai_client.model)]
    rows = []
    for path in sorted((PACKAGE_DIR / "contracts").glob("*.txt")):
        for name, client, model in providers:
            print(f"running {path.name} on {name}...", file=sys.stderr)
            rows.append(run_one(client, name, model, path))

    table = format_table(rows)
    print(table)
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
