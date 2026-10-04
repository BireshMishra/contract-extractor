"""Extraction through any LLMClient: one place for validation and retries."""

import sys
import time
from dataclasses import dataclass

from pydantic import ValidationError

from .errors import ExtractionError, SchemaError
from .models import ContractSummary
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .providers import LLMClient

MAX_RETRIES = 2


@dataclass
class Extraction:
    """A successful extraction, with what it took (summed over all attempts)."""

    summary: ContractSummary
    attempts: int
    seconds: float
    input_tokens: int
    output_tokens: int

    @property
    def retries(self) -> int:
        return self.attempts - 1


def strip_code_fence(text: str) -> str:
    """Models sometimes wrap JSON in a ```json fence despite being told not to."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.removesuffix("```")
    return text.strip()


def extract(
    client: LLMClient, contract: str, verbose: bool = False
) -> Extraction:
    """Extract a summary, timing each call with perf_counter and summing tokens.

    complete() takes one user string, so a retry resends the original request with
    the model's bad output and the validation error appended. After MAX_RETRIES
    retries it raises SchemaError; a cut-off or refused reply raises ExtractionError.
    """
    schema = ContractSummary.model_json_schema()
    base_user = USER_PROMPT_TEMPLATE.format(contract=contract)
    user = base_user
    seconds = 0.0
    input_tokens = output_tokens = 0
    for attempt in range(1, MAX_RETRIES + 2):
        started = time.perf_counter()
        response = client.complete(SYSTEM_PROMPT, user, schema)
        seconds += time.perf_counter() - started
        input_tokens += response.input_tokens
        output_tokens += response.output_tokens
        if verbose:
            print(
                f"tokens: input={response.input_tokens} output={response.output_tokens}",
                file=sys.stderr,
            )
        if response.stop_reason != "end":
            raise ExtractionError(
                f"no complete structured output (stop_reason={response.stop_reason})"
            )
        if verbose:
            print("done", file=sys.stderr)
        try:
            summary = ContractSummary.model_validate_json(strip_code_fence(response.text))
            return Extraction(summary, attempt, seconds, input_tokens, output_tokens)
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
