import argparse
import os
import sys
from pathlib import Path
from typing import NoReturn

import anthropic
from dotenv import load_dotenv
from pydantic import ValidationError

from .errors import ExtractionError, SchemaError
from .extract import MAX_RETRIES, extract_text
from .models import ContractSummary
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .providers import ClaudeClient, OpenAIClient, translate_anthropic_errors

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
DEFAULT_CONTRACT = PACKAGE_DIR / "contracts" / "sample1_saas_subscription.txt"
MODEL = "claude-sonnet-5-5"
MODES = ("json", "tool")
PROVIDERS = ("claude", "openai")
TOOL_NAME = "record_summary"
TOOL_INSTRUCTION = (
    f"Return the result by calling the {TOOL_NAME} tool exactly once. "
    "Do not answer in plain text."
)


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


def request_options(mode: str) -> dict:
    """Mode-specific arguments: a JSON-schema output format, or a strict tool.

    Sonnet 5.5 rejects a forced tool_choice (HTTP 400), so tool mode uses "auto" and
    the system prompt tells the model to call the tool.
    """
    if mode == "tool":
        tool = {
            "name": TOOL_NAME,
            "description": "Record the structured summary extracted from the contract.",
            "strict": True,
            "input_schema": anthropic.transform_schema(ContractSummary),
        }
        return {
            "system": f"{SYSTEM_PROMPT}\n\n{TOOL_INSTRUCTION}",
            "tools": [tool],
            "tool_choice": {"type": "auto"},
            "output_config": {"effort": "low"},
        }
    return {
        "system": SYSTEM_PROMPT,
        "output_config": {
            "effort": "low",
            "format": {
                "type": "json_schema",
                "schema": anthropic.transform_schema(ContractSummary),
            },
        }
    }


def send(client, messages: list[dict], mode: str):
    with translate_anthropic_errors():
        return client.messages.create(
            model=MODEL,
            max_tokens=4000,
            messages=messages,
            **request_options(mode),
        )


def parse_reply(response, mode: str) -> ContractSummary:
    """Validate the reply; raises ValidationError on bad output."""
    if mode == "tool":
        block = next((b for b in response.content if b.type == "tool_use"), None)
        if block is None:
            raise ExtractionError("model did not call the extraction tool")
        return ContractSummary.model_validate(block.input)
    text = "".join(b.text for b in response.content if b.type == "text")
    return ContractSummary.model_validate_json(text)


def retry_turn(response, mode: str, error: ValidationError) -> list[dict]:
    """Messages that show the model its bad output and the validation error.

    The assistant turn is the response content unchanged (thinking blocks included).
    """
    problem = f"That output failed schema validation:\n{error}\nFix it and try again."
    if mode == "tool":
        # Every tool_use needs a tool_result, and the results must come first.
        content = [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "is_error": True,
                "content": problem,
            }
            for block in response.content
            if block.type == "tool_use"
        ]
    else:
        content = problem
    return [
        {"role": "assistant", "content": response.content},
        {"role": "user", "content": content},
    ]


def extract(
    client, contract: str, mode: str = "json", verbose: bool = False
) -> tuple[ContractSummary, int]:
    """Extract a summary; returns it with the number of attempts used.

    On invalid output, sends the bad output and the validation error back and asks
    for a fix, up to MAX_RETRIES times, then raises SchemaError.
    """
    expected_stop = "tool_use" if mode == "tool" else "end_turn"
    messages = [
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(contract=contract)}
    ]
    for attempt in range(1, MAX_RETRIES + 2):
        response = send(client, messages, mode)
        if verbose:
            print(response.usage, file=sys.stderr)
        if mode == "tool" and response.stop_reason == "end_turn":
            raise ExtractionError(
                f"model answered in text instead of calling {TOOL_NAME} "
                f"(stop_reason={response.stop_reason})"
            )
        if response.stop_reason != expected_stop:
            raise ExtractionError(
                f"no complete structured output (stop_reason={response.stop_reason})"
            )
        if verbose:
            print("done", file=sys.stderr)
        try:
            summary = parse_reply(response, mode)
        except ValidationError as e:
            if attempt > MAX_RETRIES:
                raise SchemaError(
                    f"response did not match the ContractSummary schema after "
                    f"{attempt} attempts ({e.error_count()} validation errors, "
                    f"first: {e.errors()[0]['msg']})"
                )
            messages += retry_turn(response, mode, e)
        else:
            return summary, attempt
    raise AssertionError("unreachable")


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
        choices=(*MODES, "text"),
        help=(
            "claude only. json: schema-constrained JSON (default); tool: strict tool "
            "call; text: the provider-neutral prompt-only path that openai always uses"
        ),
    )
    args = parser.parse_args()
    if args.provider == "openai" and args.mode not in (None, "text"):
        parser.error("--mode json/tool needs --provider claude; openai uses text mode")
    mode = args.mode or ("text" if args.provider == "openai" else "json")

    load_dotenv(PROJECT_ROOT / ".env")
    text_content = read_contract(args.contract)

    if args.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            fail("no API key found; set OPENAI_API_KEY in your environment or .env")
    elif not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        fail("no API key found; set ANTHROPIC_API_KEY in your environment or .env")

    try:
        if mode == "text":
            client = OpenAIClient() if args.provider == "openai" else ClaudeClient()
            summary, _ = extract_text(client, text_content, verbose=True)
        else:
            summary, _ = extract(anthropic.Anthropic(), text_content, mode, verbose=True)
    except ExtractionError as e:
        fail(str(e))
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
