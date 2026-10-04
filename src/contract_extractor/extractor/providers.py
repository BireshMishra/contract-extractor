from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

import anthropic
import openai
from pydantic import BaseModel

from .errors import ExtractionError

CLAUDE_MODEL = "claude-sonnet-5-5"
OPENAI_MODEL = "gpt-4.1"
MAX_TOKENS = 4000


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


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> LLMResponse: ...


class ClaudeClient:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str = CLAUDE_MODEL):
        self.client = client or anthropic.Anthropic()
        self.model = model

    def complete(self, system: str, user: str) -> LLMResponse:
        with translate_anthropic_errors():
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class OpenAIClient:
    def __init__(self, client: openai.OpenAI | None = None, model: str = OPENAI_MODEL):
        self.client = client or openai.OpenAI()
        self.model = model

    def complete(self, system: str, user: str) -> LLMResponse:
        with translate_openai_errors():
            response = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        return LLMResponse(
            text=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
