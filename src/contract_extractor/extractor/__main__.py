import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn

from dotenv import load_dotenv

from .errors import ExtractionError
from .extract import extract
from .providers import CLAUDE_MODEL as MODEL
from .providers import ClaudeClient, LLMClient, OpenAIClient

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
DEFAULT_CONTRACT = PACKAGE_DIR / "contracts" / "sample1_saas_subscription.txt"
MODES = ("json", "tool")
PROVIDERS = ("claude", "openai")


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
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default="claude",
        help="which LLM to use (default: claude)",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        help="claude only. json: schema-constrained output (default); tool: strict tool call",
    )
    args = parser.parse_args()
    if args.provider == "openai" and args.mode:
        parser.error("--mode applies to --provider claude only")

    load_dotenv(PROJECT_ROOT / ".env")
    text_content = read_contract(args.contract)

    if args.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            fail("no API key found; set OPENAI_API_KEY in your environment or .env")
    elif not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        fail("no API key found; set ANTHROPIC_API_KEY in your environment or .env")

    client: LLMClient
    if args.provider == "openai":
        client = OpenAIClient()
    else:
        client = ClaudeClient(mode=args.mode or "json")
    try:
        result = extract(client, text_content, verbose=True)
    except ExtractionError as e:
        fail(str(e))
    print(result.summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
