import argparse
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from .models import ContractSummary

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
DEFAULT_CONTRACT = PACKAGE_DIR / "contracts" / "sample1_saas_subscription.txt"


SYSTEM_PROMPT = """You extract structured data from contracts.
Use null for any field whose value is not stated in the contract."""


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
    client = anthropic.Anthropic()

    text_content = args.contract.read_text(encoding="utf-8")

    response = client.messages.parse(
        model="claude-sonnet-4-5",
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

    if response.stop_reason == "end_turn":
        print("done")
        print(response.usage)

    if response.parsed_output is None:
        raise SystemExit(f"No structured output (stop_reason={response.stop_reason})")
    print(response.parsed_output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
