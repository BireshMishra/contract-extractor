"""Compare how often the JSON-schema mode and the tool-use mode fail.

Runs every sample contract through both modes `--runs` times and reports, per mode,
how many runs needed a retry (first reply invalid), how many failed outright (still
invalid after MAX_RETRIES retries) and how many errored (API error, refusal, no tool
call, truncation), which says nothing about the mode's schema reliability and is kept
out of the other columns. Makes real API calls: at least 2 modes x samples x runs.
"""

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

from .__main__ import MODES, PACKAGE_DIR, PROJECT_ROOT, fail
from .errors import ExtractionError, SchemaError
from .extract import MAX_RETRIES, extract
from .providers import ClaudeClient, LLMClient


def compare(
    make_client: Callable[[str], LLMClient], contracts: list[Path], runs: int
) -> dict[str, dict[str, int]]:
    stats = {
        m: {"runs": 0, "retried": 0, "failed": 0, "errored": 0, "attempts": 0}
        for m in MODES
    }
    seen_errors: set[str] = set()
    for mode in MODES:
        client = make_client(mode)
        for path in contracts:
            text = path.read_text(encoding="utf-8")
            for _ in range(runs):
                s = stats[mode]
                s["runs"] += 1
                try:
                    _, attempts = extract(client, text)
                except SchemaError:
                    s["failed"] += 1
                    s["retried"] += 1
                    s["attempts"] += MAX_RETRIES + 1
                except ExtractionError as e:
                    s["errored"] += 1
                    if str(e) not in seen_errors:
                        seen_errors.add(str(e))
                        print(f"[{mode}] {path.name}: {e}", file=sys.stderr)
                else:
                    s["retried"] += attempts > 1
                    s["attempts"] += attempts
    return stats


def format_report(stats: dict[str, dict[str, int]]) -> str:
    lines = [
        f"{'mode':<6} {'runs':>5} {'retried':>8} {'failed':>7} {'errored':>8} {'calls':>6}"
    ]
    for mode, s in stats.items():
        lines.append(
            f"{mode:<6} {s['runs']:>5} {s['retried']:>8} {s['failed']:>7} "
            f"{s['errored']:>8} {s['attempts']:>6}"
        )
    lines.append(
        "retried = first reply invalid; failed = invalid after all retries; "
        "errored = API error or no usable reply (not counted in calls)"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m contract_extractor.extractor.compare",
        description="Compare failure rates of the json and tool modes (paid API calls).",
    )
    parser.add_argument("--runs", type=int, default=5, help="runs per contract per mode")
    parser.add_argument("--out", type=Path, help="also write the report to this file")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        fail("no API key found; set ANTHROPIC_API_KEY in your environment or .env")

    contracts = sorted((PACKAGE_DIR / "contracts").glob("*.txt"))
    stats = compare(lambda mode: ClaudeClient(mode=mode), contracts, args.runs)
    report = format_report(stats)
    print(report)
    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
