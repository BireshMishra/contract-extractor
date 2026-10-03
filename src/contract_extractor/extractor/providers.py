from typing import Protocol

import anthropic
import openai
from pydantic import BaseModel

CLAUDE_MODEL = "claude-sonnet-5-5"
OPENAI_MODEL = "gpt-4.1"
MAX_TOKENS = 4000


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
