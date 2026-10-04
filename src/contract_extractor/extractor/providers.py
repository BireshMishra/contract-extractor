import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal, Protocol

import anthropic
import openai
from pydantic import BaseModel

from .errors import ExtractionError

CLAUDE_MODEL = "claude-sonnet-5-5"
OPENAI_MODEL = "gpt-4.1"
MAX_TOKENS = 4000
TOOL_NAME = "record_summary"
TOOL_INSTRUCTION = (
    f"Return the result by calling the {TOOL_NAME} tool exactly once. "
    "Do not answer in plain text."
)

StopReason = Literal["end", "max_tokens", "refusal", "other"]

CLAUDE_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end",
    "tool_use": "end",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}
OPENAI_STOP_REASONS: dict[str, StopReason] = {
    "stop": "end",
    "length": "max_tokens",
    "content_filter": "refusal",
}


@contextmanager
def translate_anthropic_errors() -> Iterator[None]:
    """Turn Anthropic SDK errors into one-line ExtractionErrors."""
    try:
        yield
    except anthropic.AuthenticationError:
        raise ExtractionError(
            "invalid API key (authentication failed); check ANTHROPIC_API_KEY"
        )
    except anthropic.PermissionDeniedError:
        raise ExtractionError("API key does not have permission for this request")
    except anthropic.RateLimitError:
        raise ExtractionError("rate limit exceeded; wait a moment and try again")
    except anthropic.APITimeoutError:
        raise ExtractionError("request to the Anthropic API timed out")
    except anthropic.APIConnectionError:
        raise ExtractionError(
            "could not connect to the Anthropic API; check your network connection"
        )
    except anthropic.APIStatusError as e:
        raise ExtractionError(f"Anthropic API returned HTTP {e.status_code}: {e.message}")
    except anthropic.APIError as e:
        raise ExtractionError(f"Anthropic API error: {e.message}")


@contextmanager
def translate_openai_errors() -> Iterator[None]:
    """Turn OpenAI SDK errors into one-line ExtractionErrors."""
    try:
        yield
    except openai.AuthenticationError:
        raise ExtractionError(
            "invalid API key (authentication failed); check OPENAI_API_KEY"
        )
    except openai.PermissionDeniedError:
        raise ExtractionError("API key does not have permission for this request")
    except openai.RateLimitError:
        raise ExtractionError("rate limit exceeded; wait a moment and try again")
    except openai.APITimeoutError:
        raise ExtractionError("request to the OpenAI API timed out")
    except openai.APIConnectionError:
        raise ExtractionError(
            "could not connect to the OpenAI API; check your network connection"
        )
    except openai.APIStatusError as e:
        raise ExtractionError(f"OpenAI API returned HTTP {e.status_code}: {e.message}")
    except openai.APIError as e:
        raise ExtractionError(f"OpenAI API error: {e.message}")


class LLMResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: StopReason


class LLMClient(Protocol):
    """One provider-neutral call.

    `schema` is a JSON schema the reply must follow; each client hands it to its
    provider's own structured-output feature. Clients raise ExtractionError for
    provider failures and report a cut-off reply through `stop_reason`.
    """

    def complete(
        self, system: str, user: str, schema: dict[str, Any] | None = None
    ) -> LLMResponse: ...


class ClaudeClient:
    """Claude. mode="json" uses schema-constrained output; "tool" a strict tool call."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = CLAUDE_MODEL,
        mode: Literal["json", "tool"] = "json",
    ):
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.mode = mode

    def complete(
        self, system: str, user: str, schema: dict[str, Any] | None = None
    ) -> LLMResponse:
        # Low effort keeps thinking from eating max_tokens, which truncates the reply.
        output_config: dict[str, Any] = {"effort": "low"}
        extra: dict[str, Any] = {}
        use_tool = schema is not None and self.mode == "tool"
        if use_tool:
            # Sonnet 5.5 rejects a forced tool_choice (HTTP 400), so ask for the tool.
            system = f"{system}\n\n{TOOL_INSTRUCTION}"
            extra["tool_choice"] = {"type": "auto"}
            extra["tools"] = [
                {
                    "name": TOOL_NAME,
                    "description": "Record the structured summary extracted from the contract.",
                    "strict": True,
                    "input_schema": anthropic.transform_schema(schema),
                }
            ]
        elif schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": anthropic.transform_schema(schema),
            }
        with translate_anthropic_errors():
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config=output_config,
                **extra,
            )
        stop_reason = CLAUDE_STOP_REASONS.get(response.stop_reason, "other")
        text = "".join(b.text for b in response.content if b.type == "text")
        tool_call = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_call is not None:
            text = json.dumps(tool_call.input)
        elif use_tool and stop_reason == "end":
            raise ExtractionError(
                f"model answered in text instead of calling {TOOL_NAME} "
                f"(stop_reason={response.stop_reason})"
            )
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=stop_reason,
        )


def strict_schema(schema: Any) -> Any:
    """OpenAI strict mode: every object closed, every property required."""
    if isinstance(schema, list):
        return [strict_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out = {key: strict_schema(value) for key, value in schema.items()}
    if isinstance(out.get("properties"), dict):
        out["additionalProperties"] = False
        out["required"] = list(out["properties"])
    return out


class OpenAIClient:
    def __init__(self, client: openai.OpenAI | None = None, model: str = OPENAI_MODEL):
        self.client = client or openai.OpenAI()
        self.model = model

    def complete(
        self, system: str, user: str, schema: dict[str, Any] | None = None
    ) -> LLMResponse:
        extra: dict[str, Any] = {}
        if schema is not None:
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": strict_schema(schema),
                },
            }
        with translate_openai_errors():
            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **extra,
            )
        choice = response.choices[0]
        if choice.message.refusal:
            stop_reason: StopReason = "refusal"
        else:
            stop_reason = OPENAI_STOP_REASONS.get(choice.finish_reason, "other")
        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            stop_reason=stop_reason,
        )
