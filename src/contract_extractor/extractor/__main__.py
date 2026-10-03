import argparse
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from .models import ContractSummary

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
DEFAULT_CONTRACT = PACKAGE_DIR / "contracts" / "sample1_saas_subscription.txt"


SYSTEM_PROMPT = """You extract structured data from contracts.
Use null for any field whose value is not stated in the contract."""


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def read_contract(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail(f"contract file not found: {path}")
    except IsADirectoryError:
        fail(f"contract path is a directory, not a file: {path}")
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

    try:
        client = anthropic.Anthropic()
        response = client.messages.parse(
            model="claude-sonnet-5-5",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Extract the contract summary from this contract:\n\n{text_content}\n",
                }
            ],
            output_format=ContractSummary,
        )
    except TypeError as e:
        # The SDK raises TypeError when no API key/auth token is configured.
        if "authentication method" in str(e):
            fail("no API key found; set ANTHROPIC_API_KEY in your environment or .env")
        raise
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

    if response.stop_reason == "end_turn":
        print("done", file=sys.stderr)
    print(response.usage, file=sys.stderr)

    if response.parsed_output is None:
        raise SystemExit(f"No structured output (stop_reason={response.stop_reason})")
    print(response.parsed_output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
