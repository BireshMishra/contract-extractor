"""Provider-agnostic extraction: prompt for JSON, validate, retry on bad output."""

import json
import sys

from pydantic import ValidationError

from .errors import ExtractionError, SchemaError
from .models import ContractSummary
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .providers import LLMClient

MAX_RETRIES = 2

JSON_INSTRUCTION = (
    "Reply with only a JSON object that matches this JSON schema. "
    "No prose and no code fences.\n"
)


def strip_code_fence(text: str) -> str:
    """Models sometimes wrap JSON in a ```json fence despite being told not to."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.removesuffix("```")
    return text.strip()


def extract_text(
    client: LLMClient, contract: str, verbose: bool = False
) -> tuple[ContractSummary, int]:
    """Extract a summary through any LLMClient; returns it with the attempts used.

    complete() takes a single user string, so a retry resends the original request
    with the model's bad output and the validation error appended. Gives up with a
    SchemaError after MAX_RETRIES retries.
    """
    schema = json.dumps(ContractSummary.model_json_schema())
    system = f"{SYSTEM_PROMPT}\n\n{JSON_INSTRUCTION}{schema}"
    base_user = USER_PROMPT_TEMPLATE.format(contract=contract)
    user = base_user
    for attempt in range(1, MAX_RETRIES + 2):
        response = client.complete(system, user)
        if verbose:
            print(
                f"tokens: input={response.input_tokens} output={response.output_tokens}",
                file=sys.stderr,
            )
            print("done", file=sys.stderr)
        try:
            return ContractSummary.model_validate_json(strip_code_fence(response.text)), attempt
        except ValidationError as e:
            if attempt > MAX_RETRIES:
                raise SchemaError(
                    f"response did not match the ContractSummary schema after "
                    f"{attempt} attempts ({e.error_count()} validation errors, "
                    f"first: {e.errors()[0]['msg']})"
                )
            user = (
                f"{base_user}\n\nYour previous reply was:\n{response.text}\n\n"
                f"It failed schema validation:\n{e}\n"
                "Reply again with only the corrected JSON object."
            )
    raise AssertionError("unreachable")
