"""The retry loop and CLI routing, written against the provider-neutral interface."""

import json
import sys
from pathlib import Path

import pytest

from contract_extractor.extractor import __main__ as cli
from contract_extractor.extractor.errors import ExtractionError, SchemaError
from contract_extractor.extractor.extract import extract, strip_code_fence
from contract_extractor.extractor.providers import LLMResponse

CONTRACTS_DIR = Path(cli.__file__).resolve().parent.parent / "contracts"
EXPECTED = json.loads((Path(__file__).parent / "expected.json").read_text("utf-8"))
SAMPLE = "sample1_saas_subscription.txt"
GOOD = json.dumps(EXPECTED[SAMPLE])


class FakeLLM:
    """An LLMClient that replays canned replies and records every call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user, schema=None):
        self.calls.append((system, user, schema))
        reply = self.replies.pop(0)
        if isinstance(reply, str):
            reply = LLMResponse(text=reply, input_tokens=3, output_tokens=4, stop_reason="end")
        return reply


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


def test_extract_passes_the_schema_to_the_client():
    llm = FakeLLM(GOOD)
    summary, attempts = extract(llm, "THE CONTRACT")

    assert attempts == 1
    assert summary.model_dump(mode="json") == EXPECTED[SAMPLE]
    ((_, user, schema),) = llm.calls
    assert "governing_law" in schema["properties"]
    assert "<contract>\nTHE CONTRACT\n</contract>" in user


def test_extract_accepts_fenced_json():
    summary, _ = extract(FakeLLM(f"```json\n{GOOD}\n```"), "c")
    assert summary.parties == EXPECTED[SAMPLE]["parties"]


def test_retry_includes_bad_output_and_error():
    llm = FakeLLM('{"parties": "nope"}', GOOD)
    _, attempts = extract(llm, "c")

    assert attempts == 2
    retry_user = llm.calls[1][1]
    assert '{"parties": "nope"}' in retry_user
    assert "failed schema validation" in retry_user
    assert "<contract>" in retry_user
    assert llm.calls[1][2] is not None  # the schema goes with every attempt


def test_stops_after_two_retries():
    llm = FakeLLM("{}", "{}", "{}")
    with pytest.raises(SchemaError, match="after 3 attempts"):
        extract(llm, "c")
    assert len(llm.calls) == 3


@pytest.mark.parametrize("reason", ["max_tokens", "refusal", "other"])
def test_cut_off_reply_is_reported_not_retried(reason):
    llm = FakeLLM(LLMResponse(text='{"parties": ["A"', input_tokens=1, output_tokens=1, stop_reason=reason))
    with pytest.raises(ExtractionError, match=f"stop_reason={reason}") as info:
        extract(llm, "c")
    assert not isinstance(info.value, SchemaError)
    assert len(llm.calls) == 1


def test_cli_openai_routes_through_openai_client(monkeypatch, capsys):
    llm = FakeLLM(GOOD)
    monkeypatch.setattr(cli, "OpenAIClient", lambda: llm)

    code, out, err = run_cli(monkeypatch, capsys, str(CONTRACTS_DIR / SAMPLE), "--provider", "openai")

    assert code == 0
    assert json.loads(out) == EXPECTED[SAMPLE]
    assert err.splitlines() == ["tokens: input=3 output=4", "done"]


def test_cli_claude_mode_reaches_the_claude_client(monkeypatch, capsys):
    made = {}

    def fake_claude(mode):
        made["mode"] = mode
        return FakeLLM(GOOD)

    monkeypatch.setattr(cli, "ClaudeClient", fake_claude)
    code, _, _ = run_cli(monkeypatch, capsys, str(CONTRACTS_DIR / SAMPLE), "--mode", "tool")

    assert code == 0
    assert made["mode"] == "tool"


def test_cli_openai_rejects_mode(monkeypatch, capsys):
    code, _, err = run_cli(
        monkeypatch, capsys, str(CONTRACTS_DIR / SAMPLE), "--provider", "openai", "--mode", "tool"
    )
    assert code == 2
    assert "--provider claude only" in err


def test_cli_openai_needs_its_own_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY")
    code, _, err = run_cli(monkeypatch, capsys, str(CONTRACTS_DIR / SAMPLE), "--provider", "openai")
    assert code == 1
    assert "OPENAI_API_KEY" in err


def test_cli_reports_provider_errors_in_one_line(monkeypatch, capsys):
    class Boom:
        def complete(self, system, user, schema=None):
            raise ExtractionError("rate limit exceeded; wait a moment and try again")

    monkeypatch.setattr(cli, "OpenAIClient", Boom)
    code, out, err = run_cli(monkeypatch, capsys, str(CONTRACTS_DIR / SAMPLE), "--provider", "openai")
    assert code == 1
    assert out == ""
    assert err == "error: rate limit exceeded; wait a moment and try again\n"
