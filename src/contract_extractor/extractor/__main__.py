import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn

import anthropic
from dotenv import load_dotenv
from pydantic import ValidationError

from .models import ContractSummary
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
DEFAULT_CONTRACT = PACKAGE_DIR / "contracts" / "sample1_saas_subscription.txt"
MODEL = "claude-sonnet-5-5"


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def read_contract(path: Path) -> str:
    if path.is_dir():
        fail(f"contract path is a directory, not a file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail(f"contract file not found: {path}")
    except PermissionError:
        fail(f"permission denied reading contract file: {path}")
    except UnicodeDecodeError:
        fail(f"contract file is not valid UTF-8 text: {path}")
    except OSError as e:
        fail(f"could not read contract file {path}: {e.strerror or e}")
    if not text.strip():
        fail(f"contract file is empty: {path}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m contract_extractor.extractor",
        description="Extract a structured summary from a contract text file.",
    )
    parser.add_argument(
        "contract",
        nargs="?",
        type=Path,
        default=DEFAULT_CONTRACT,
        help=f"path to the contract text file (default: {DEFAULT_CONTRACT.name})",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    text_content = read_contract(args.contract)

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        fail("no API key found; set ANTHROPIC_API_KEY in your environment or .env")

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(contract=text_content),
                }
            ],
            output_config={
                "effort": "low",
                "format": {
                    "type": "json_schema",
                    "schema": anthropic.transform_schema(ContractSummary),
                },
            },
        )
    except anthropic.AuthenticationError:
        fail("invalid API key (authentication failed); check ANTHROPIC_API_KEY")
    except anthropic.PermissionDeniedError:
        fail("API key does not have permission for this request")
    except anthropic.RateLimitError:
        fail("rate limit exceeded; wait a moment and try again")
    except anthropic.APITimeoutError:
        fail("request to the Anthropic API timed out")
    except anthropic.APIConnectionError:
        fail("could not connect to the Anthropic API; check your network connection")
    except anthropic.APIStatusError as e:
        fail(f"Anthropic API returned HTTP {e.status_code}: {e.message}")
    except anthropic.APIError as e:
        fail(f"Anthropic API error: {e.message}")

    print(response.usage, file=sys.stderr)
    if response.stop_reason != "end_turn":
        fail(f"no complete structured output (stop_reason={response.stop_reason})")
    print("done", file=sys.stderr)

    text = "".join(block.text for block in response.content if block.type == "text")
    try:
        summary = ContractSummary.model_validate_json(text)
    except ValidationError as e:
        fail(
            f"response did not match the ContractSummary schema "
            f"(stop_reason={response.stop_reason}, {e.error_count()} validation errors)"
        )
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
