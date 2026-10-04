"""CLI behaviour against a fake Anthropic SDK client (Claude path)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from contract_extractor.extractor import __main__ as cli
from contract_extractor.extractor import compare, providers
from contract_extractor.extractor.models import ContractSummary

CONTRACTS_DIR = Path(cli.__file__).resolve().parent.parent / "contracts"
EXPECTED = json.loads((Path(__file__).parent / "expected.json").read_text("utf-8"))
SAMPLE = "sample1_saas_subscription.txt"
USAGE = SimpleNamespace(input_tokens=1, output_tokens=2)
TOKENS_LINE = "tokens: input=1 output=2"


class FakeClient:
    """Stands in for anthropic.Anthropic; replays replies (or raises errors) in turn."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


def text_reply(text, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        usage=USAGE,
    )


def tool_reply(data, stop_reason="tool_use"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[
            SimpleNamespace(type="tool_use", id="toolu_1", name=providers.TOOL_NAME, input=data)
        ],
        usage=USAGE,
    )


def status_error(cls, status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    return cls("boom", response=response, body=None)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


@pytest.fixture
def run(monkeypatch, capsys):
    """Run the CLI against a fake client; returns (client, exit_code, stdout, stderr)."""

    def _run(contract_path, client, *flags):
        monkeypatch.setattr(anthropic, "Anthropic", lambda: client)
        monkeypatch.setattr(sys, "argv", ["extractor", str(contract_path), *flags])
        code = 0
        try:
            cli.main()
        except SystemExit as e:
            code = e.code
        out, err = capsys.readouterr()
        return client, code, out, err

    return _run


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_sample_contract_matches_expected(run, name):
    client = FakeClient(text_reply(json.dumps(EXPECTED[name])))

    _, code, out, err = run(CONTRACTS_DIR / name, client)

    assert code == 0
    assert json.loads(out) == EXPECTED[name]
    assert err.splitlines() == [TOKENS_LINE, "done"]


def test_stdout_is_pure_json(run):
    name = "sample2_consulting_msa.txt"
    _, _, out, _ = run(CONTRACTS_DIR / name, FakeClient(text_reply(json.dumps(EXPECTED[name]))))
    json.loads(out)  # raises if anything but JSON was printed


def test_json_mode_request_shape(run):
    path = CONTRACTS_DIR / SAMPLE
    client = FakeClient(text_reply(json.dumps(EXPECTED[SAMPLE])))

    run(path, client)

    (request,) = client.requests
    assert request["model"] == cli.MODEL
    assert request["max_tokens"] >= 4000
    assert request["output_config"]["effort"] == "low"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "tools" not in request
    content = request["messages"][0]["content"]
    assert f"<contract>\n{path.read_text('utf-8')}\n</contract>" in content


def test_tool_mode_request_shape(run):
    client = FakeClient(tool_reply(EXPECTED[SAMPLE]))

    _, code, out, _ = run(CONTRACTS_DIR / SAMPLE, client, "--mode", "tool")

    assert code == 0
    assert json.loads(out) == EXPECTED[SAMPLE]
    (request,) = client.requests
    (tool,) = request["tools"]
    assert tool["name"] == providers.TOOL_NAME
    assert tool["strict"] is True
    assert tool["input_schema"] == anthropic.transform_schema(ContractSummary.model_json_schema())
    assert request["tool_choice"] == {"type": "auto"}  # a forced tool is a 400
    assert providers.TOOL_NAME in request["system"]
    assert request["output_config"] == {"effort": "low"}


def test_tool_mode_text_reply_is_reported(run):
    client = FakeClient(text_reply("here you go"))
    _, code, out, err = run(CONTRACTS_DIR / SAMPLE, client, "--mode", "tool")

    assert code == 1
    assert out == ""
    assert "instead of calling record_summary" in err


def test_truncated_reply_reports_stop_reason(run):
    client = FakeClient(text_reply('{"parties": ["A"', stop_reason="max_tokens"))
    _, code, out, err = run(CONTRACTS_DIR / "sample2_consulting_msa.txt", client)

    assert code == 1
    assert out == ""
    assert "stop_reason=max_tokens" in err
    assert len(client.requests) == 1  # truncation is not retried


def test_truncated_tool_reply_reports_stop_reason(run):
    client = FakeClient(tool_reply({"parties": ["A"]}, stop_reason="max_tokens"))
    _, code, _, err = run(CONTRACTS_DIR / SAMPLE, client, "--mode", "tool")

    assert code == 1
    assert "stop_reason=max_tokens" in err


def test_json_retry_sends_bad_output_and_error(run):
    client = FakeClient(text_reply('{"parties": "nope"}'), text_reply(json.dumps(EXPECTED[SAMPLE])))

    _, code, out, _ = run(CONTRACTS_DIR / SAMPLE, client)

    assert code == 0
    assert json.loads(out) == EXPECTED[SAMPLE]
    first, second = client.requests
    retry = second["messages"][0]["content"]
    assert retry.startswith(first["messages"][0]["content"])
    assert '{"parties": "nope"}' in retry
    assert "failed schema validation" in retry


def test_stops_after_two_retries(run):
    client = FakeClient(text_reply('{"parties": "x"}'))
    _, code, out, err = run(CONTRACTS_DIR / SAMPLE, client)

    assert code == 1
    assert out == ""
    assert len(client.requests) == 3  # first try + 2 retries
    assert "after 3 attempts" in err
    assert "ContractSummary schema" in err
    assert "stop_reason" not in err


def test_missing_file(run):
    client = FakeClient()
    _, code, _, err = run(CONTRACTS_DIR / "nope.txt", client)

    assert code == 1
    assert "not found" in err
    assert client.requests == []


def test_directory_path(run, tmp_path):
    client = FakeClient()
    _, code, _, err = run(tmp_path, client)

    assert code == 1
    assert "is a directory" in err
    assert client.requests == []


def test_empty_file(run, tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("  \n")
    client = FakeClient()
    _, code, _, err = run(empty, client)

    assert code == 1
    assert "empty" in err
    assert client.requests == []


def test_missing_api_key(run, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    client = FakeClient()
    _, code, _, err = run(CONTRACTS_DIR / "sample2_consulting_msa.txt", client)

    assert code == 1
    assert "no API key" in err
    assert client.requests == []


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (status_error(anthropic.AuthenticationError, 401), "invalid API key"),
        (status_error(anthropic.PermissionDeniedError, 403), "permission"),
        (status_error(anthropic.RateLimitError, 429), "rate limit"),
        (status_error(anthropic.InternalServerError, 500), "HTTP 500"),
        (
            anthropic.APIConnectionError(
                request=httpx2.Request("POST", "https://api.anthropic.com")
            ),
            "could not connect",
        ),
        (
            anthropic.APITimeoutError(
                request=httpx2.Request("POST", "https://api.anthropic.com")
            ),
            "timed out",
        ),
    ],
)
def test_api_errors_give_one_line_message(run, error, message):
    client = FakeClient(error)
    _, code, out, err = run(CONTRACTS_DIR / "sample2_consulting_msa.txt", client)

    assert code == 1
    assert out == ""
    assert message in err
    assert err.startswith("error: ")
    assert len(err.strip().splitlines()) == 1


def test_compare_counts_retries_and_failures():
    good = json.dumps(EXPECTED[SAMPLE])
    json_client = FakeClient(
        text_reply(good),  # run 1: clean
        text_reply("{}"), text_reply(good),  # run 2: one retry
    )
    tool_client = FakeClient(
        tool_reply({}), tool_reply({}), tool_reply({}),  # run 1: fails
        tool_reply(EXPECTED[SAMPLE]),  # run 2: clean
    )
    clients = {"json": json_client, "tool": tool_client}

    stats = compare.compare(
        lambda mode: providers.ClaudeClient(client=clients[mode], mode=mode),
        [CONTRACTS_DIR / SAMPLE],
        runs=2,
    )

    assert stats["json"] == {"runs": 2, "retried": 1, "failed": 0, "errored": 0, "attempts": 3}
    assert stats["tool"] == {"runs": 2, "retried": 1, "failed": 1, "errored": 0, "attempts": 4}


def test_compare_keeps_going_after_an_error(capsys):
    good = json.dumps(EXPECTED[SAMPLE])
    clients = {
        "json": FakeClient(status_error(anthropic.BadRequestError, 400), text_reply(good)),
        "tool": FakeClient(tool_reply(EXPECTED[SAMPLE]), text_reply("no", stop_reason="refusal")),
    }
    stats = compare.compare(
        lambda mode: providers.ClaudeClient(client=clients[mode], mode=mode),
        [CONTRACTS_DIR / SAMPLE],
        runs=2,
    )

    assert stats["json"] == {"runs": 2, "retried": 0, "failed": 0, "errored": 1, "attempts": 1}
    assert stats["tool"] == {"runs": 2, "retried": 0, "failed": 0, "errored": 1, "attempts": 1}
    assert "errored" in compare.format_report(stats)
    err = capsys.readouterr().err
    assert "HTTP 400" in err and "refusal" in err
