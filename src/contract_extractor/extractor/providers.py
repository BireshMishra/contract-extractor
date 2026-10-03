from typing import Protocol
from pydantic import BaseModel
import anthropic

MODEL = "claude-sonnet-5-5"

class LLMResponse(BaseModel):
    text : str
    input_token : str
    output_token : str

class LLMClient(Protocol):
    def complete(system: str, user: str) -> LLMResponse:
        ...


class ClaudeClient:
    def complete(system: str, user: str) -> LLMResponse:
        client = anthropic.Anthropic()
        response = client.messages.create(
             model=MODEL,
             max_tokens=4000,
             system_prompt=system,
             messages=[
                {
                     "role": "user",
                     "content": user
                }
             ],
        )

