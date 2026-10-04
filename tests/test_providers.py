from types import SimpleNamespace

import httpx2
import openai
import pytest

from contract_extractor.extractor.errors import ExtractionError
from contract_extractor.extractor.providers import (
    ClaudeClient,
    LLMClient,
    LLMResponse,
    OpenAIClient,
    strict_schema,
)

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
}


class Recorder:
    def __init__(self, reply):
        self.reply = reply
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def claude_reply(content, stop_reason="end_turn"):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )


def claude(reply, **kwargs):
    rec = Recorder(reply)
    return ClaudeClient(client=SimpleNamespace(messages=rec), model="m", **kwargs), rec


def openai_reply(content, finish_reason="stop", refusal=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, refusal=refusal),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


def openai_client(reply):
    rec = Recorder(reply)
    client = OpenAIClient(client=SimpleNamespace(chat=SimpleNamespace(completions=rec)), model="m")
    return client, rec


def test_claude_plain_request_and_response():
    reply = claude_reply(
        [
            SimpleNamespace(type="thinking", thinking="hmm"),
            SimpleNamespace(type="text", text="hel"),
            SimpleNamespace(type="text", text="lo"),
        ]
    )
    client, rec = claude(reply)

    result = client.complete("be brief", "hi")

    assert result == LLMResponse(text="hello", input_tokens=11, output_tokens=7, stop_reason="end")
    assert rec.kwargs["system"] == "be brief"
    assert rec.kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert rec.kwargs["output_config"] == {"effort": "low"}  # no schema, no format
    assert "tools" not in rec.kwargs


def test_claude_json_mode_passes_schema_as_output_format():
    client, rec = claude(claude_reply([SimpleNamespace(type="text", text="{}")]))
    client.complete("s", "u", SCHEMA)

    config = rec.kwargs["output_config"]
    assert config["effort"] == "low"
    assert config["format"]["type"] == "json_schema"
    assert "name" in config["format"]["schema"]["properties"]


def test_claude_tool_mode_passes_schema_as_strict_tool():
    block = SimpleNamespace(type="tool_use", id="t1", name="record_summary", input={"name": "x"})
    client, rec = claude(claude_reply([block], stop_reason="tool_use"), mode="tool")

    result = client.complete("s", "u", SCHEMA)

    (tool,) = rec.kwargs["tools"]
    assert tool["strict"] is True
    assert "name" in tool["input_schema"]["properties"]
    assert rec.kwargs["tool_choice"] == {"type": "auto"}
    assert "record_summary" in rec.kwargs["system"]
    assert rec.kwargs["output_config"] == {"effort": "low"}
    assert result.text == '{"name": "x"}'
    assert result.stop_reason == "end"


@pytest.mark.parametrize(
    ("sdk_reason", "expected"),
    [("end_turn", "end"), ("max_tokens", "max_tokens"), ("refusal", "refusal"), ("pause_turn", "other")],
)
def test_claude_stop_reasons_are_normalised(sdk_reason, expected):
    client, _ = claude(claude_reply([SimpleNamespace(type="text", text="x")], sdk_reason))
    assert client.complete("s", "u").stop_reason == expected


def test_claude_tool_mode_text_answer_is_an_error():
    client, _ = claude(claude_reply([SimpleNamespace(type="text", text="hi")]), mode="tool")
    with pytest.raises(ExtractionError, match="instead of calling record_summary"):
        client.complete("s", "u", SCHEMA)


def test_claude_translates_sdk_errors():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    import anthropic

    error = anthropic.RateLimitError("boom", response=httpx2.Response(429, request=request), body=None)
    client, _ = claude(error)
    with pytest.raises(ExtractionError, match="rate limit"):
        client.complete("s", "u")


def test_openai_request_and_response():
    client, rec = openai_client(openai_reply("hello"))

    result = client.complete("be brief", "hi")

    assert result == LLMResponse(text="hello", input_tokens=11, output_tokens=7, stop_reason="end")
    assert rec.kwargs["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert "response_format" not in rec.kwargs


def test_openai_passes_schema_as_strict_response_format():
    client, rec = openai_client(openai_reply("{}"))
    client.complete("s", "u", SCHEMA)

    fmt = rec.kwargs["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["name", "age"]


def test_strict_schema_closes_nested_objects():
    nested = {"properties": {"inner": {"properties": {"a": {"type": "string"}}}}}
    out = strict_schema(nested)
    assert out["properties"]["inner"]["additionalProperties"] is False
    assert out["properties"]["inner"]["required"] == ["a"]
    assert "additionalProperties" not in nested  # input untouched


@pytest.mark.parametrize(
    ("finish", "refusal", "expected"),
    [("stop", None, "end"), ("length", None, "max_tokens"), ("content_filter", None, "refusal"),
     ("stop", "I can't", "refusal"), ("tool_calls", None, "other")],
)
def test_openai_stop_reasons_are_normalised(finish, refusal, expected):
    client, _ = openai_client(openai_reply("x", finish_reason=finish, refusal=refusal))
    assert client.complete("s", "u").stop_reason == expected


def test_openai_empty_content_becomes_empty_text():
    client, _ = openai_client(openai_reply(None))
    assert client.complete("s", "u").text == ""


def test_openai_translates_sdk_errors():
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    error = openai.RateLimitError("boom", response=httpx2.Response(429, request=request), body=None)
    client, _ = openai_client(error)
    with pytest.raises(ExtractionError, match="rate limit"):
        client.complete("s", "u")


def test_both_clients_satisfy_the_protocol():
    a: LLMClient = ClaudeClient(client=SimpleNamespace())
    b: LLMClient = OpenAIClient(client=SimpleNamespace())
    assert callable(a.complete) and callable(b.complete)
