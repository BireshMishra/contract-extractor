from types import SimpleNamespace

from contract_extractor.extractor.providers import (
    ClaudeClient,
    LLMClient,
    LLMResponse,
    OpenAIClient,
)


class Recorder:
    def __init__(self, reply):
        self.reply = reply
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.reply


def test_claude_client_maps_request_and_response():
    reply = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="hmm"),
            SimpleNamespace(type="text", text="hel"),
            SimpleNamespace(type="text", text="lo"),
        ],
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )
    rec = Recorder(reply)
    client = ClaudeClient(client=SimpleNamespace(messages=rec), model="m")

    result = client.complete("be brief", "hi")

    assert result == LLMResponse(text="hello", input_tokens=11, output_tokens=7)
    assert rec.kwargs["system"] == "be brief"
    assert rec.kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert rec.kwargs["model"] == "m"


def test_openai_client_maps_request_and_response():
    reply = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )
    rec = Recorder(reply)
    client = OpenAIClient(
        client=SimpleNamespace(chat=SimpleNamespace(completions=rec)), model="m"
    )

    result = client.complete("be brief", "hi")

    assert result == LLMResponse(text="hello", input_tokens=11, output_tokens=7)
    assert rec.kwargs["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]


def test_openai_empty_content_becomes_empty_text():
    reply = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0),
    )
    client = OpenAIClient(
        client=SimpleNamespace(chat=SimpleNamespace(completions=Recorder(reply)))
    )
    assert client.complete("s", "u").text == ""


def test_both_clients_satisfy_the_protocol():
    claude: LLMClient = ClaudeClient(client=SimpleNamespace())
    openai_client: LLMClient = OpenAIClient(client=SimpleNamespace())
    assert callable(claude.complete) and callable(openai_client.complete)
