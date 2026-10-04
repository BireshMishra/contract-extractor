import json
import sys
from pathlib import Path

import httpx2
import openai
import pytest

from contract_extractor.extractor import __main__ as cli
from contract_extractor.extractor.errors import ExtractionError, SchemaError
from contract_extractor.extractor.extract import extract_text, strip_code_fence
from contract_extractor.extractor.providers import LLMResponse, OpenAIClient

CONTRACTS_DIR = Path(cli.__file__).resolve().parent.parent / "contracts"
EXPECTED = json.loads((Path(__file__).parent / "expected.json").read_text("utf-8"))
SAMPLE = "sample1_saas_subscription.txt"
GOOD = json.dumps(EXPECTED[SAMPLE])


class FakeLLM:
    """An LLMClient that replays canned texts and records (system, user) calls."""

    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return LLMResponse(text=self.texts.pop(0), input_tokens=3, output_tokens=4)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def run_cli(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["extractor", *argv])
    code = 0
    try:
        cli.main()
    except SystemExit as e:
        code = e.code
    out, err = capsys.readouterr()
    return code, out, err


def test_strip_code_fence():
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence('  {"a": 1}  ') == '{"a": 1}'


def test_extract_text_sends_schema_and_contract():
    llm = FakeLLM(GOOD)
    summary, attempts = extract_text(llm, "THE CONTRACT")

    assert attempts == 1
    assert summary.model_dump(mode="json") == EXPECTED[SAMPLE]
    ((system, user),) = llm.calls
    assert "governing_law" in system  # the schema is in the system prompt
    assert "<contract>\nTHE CONTRACT\n</contract>" in user


def test_extract_text_accepts_fenced_json():
    summary, _ = extract_text(FakeLLM(f"```json\n{GOOD}\n```"), "c")
    assert summary.parties == EXPECTED[SAMPLE]["parties"]


def test_extract_text_retry_includes_bad_output_and_error():
    llm = FakeLLM('{"parties": "nope"}', GOOD)
    _, attempts = extract_text(llm, "c")

    assert attempts == 2
    retry_user = llm.calls[1][1]
    assert '{"parties": "nope"}' in retry_user
    assert "failed schema validation" in retry_user
    assert "parties" in retry_user
    assert "<contract>" in retry_user


def test_extract_text_stops_after_two_retries():
    llm = FakeLLM("{}", "{}", "{}")
    with pytest.raises(SchemaError, match="after 3 attempts"):
        extract_text(llm, "c")
    assert len(llm.calls) == 3


def test_cli_openai_routes_through_openai_client(monkeypatch, capsys):
    llm = FakeLLM(GOOD)
    monkeypatch.setattr(cli, "OpenAIClient", lambda: llm)

    code, out, err = run_cli(
        monkeypatch, capsys, str(CONTRACTS_DIR / SAMPLE), "--provider", "openai"
    )

    assert code == 0
    assert json.loads(out) == EXPECTED[SAMPLE]
    assert err.splitlines() == ["tokens: input=3 output=4", "done"]
    assert len(llm.calls) == 1


def test_cli_claude_text_mode_routes_through_claude_client(monkeypatch, capsys):
    llm = FakeLLM(GOOD)
    monkeypatch.setattr(cli, "ClaudeClient", lambda: llm)

    code, out, _ = run_cli(
        monkeypatch, capsys, str(CONTRACTS_DIR / SAMPLE), "--mode", "text"
    )

    assert code == 0
    assert json.loads(out) == EXPECTED[SAMPLE]


def test_cli_openai_rejects_native_modes(monkeypatch, capsys):
    code, _, err = run_cli(
        monkeypatch, capsys, str(CONTRACTS_DIR / SAMPLE),
        "--provider", "openai", "--mode", "tool",
    )
    assert code == 2
    assert "needs --provider claude" in err


def test_cli_openai_needs_its_own_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY")
    code, _, err = run_cli(
        monkeypatch, capsys, str(CONTRACTS_DIR / SAMPLE), "--provider", "openai"
    )
    assert code == 1
    assert "OPENAI_API_KEY" in err


def test_cli_reports_provider_schema_failure(monkeypatch, capsys):
    monkeypatch.setattr(cli, "OpenAIClient", lambda: FakeLLM("{}", "{}", "{}"))
    code, out, err = run_cli(
        monkeypatch, capsys, str(CONTRACTS_DIR / SAMPLE), "--provider", "openai"
    )
    assert code == 1
    assert out == ""
    assert "ContractSummary schema" in err


def test_openai_client_translates_errors():
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    error = openai.RateLimitError(
        "boom", response=httpx2.Response(429, request=request), body=None
    )

    class Boom:
        def create(self, **kwargs):
            raise error

    client = OpenAIClient(client=type("C", (), {"chat": type("Ch", (), {"completions": Boom()})()})())
    with pytest.raises(ExtractionError, match="rate limit"):
        client.complete("s", "u")
