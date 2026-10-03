import json
import sys
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from contract_extractor.extractor import __main__ as cli

CONTRACTS_DIR = Path(cli.__file__).resolve().parent.parent / "contracts"
EXPECTED = json.loads((Path(__file__).parent / "expected.json").read_text("utf-8"))


class FakeClient:
    """Stands in for anthropic.Anthropic; records requests, replies with a canned result."""

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.reply


def text_reply(text, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        usage="Usage(input_tokens=1, output_tokens=2)",
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

    def _run(contract_path, client):
        monkeypatch.setattr(cli.anthropic, "Anthropic", lambda: client)
        monkeypatch.setattr(sys, "argv", ["extractor", str(contract_path)])
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
    path = CONTRACTS_DIR / name
    client = FakeClient(text_reply(json.dumps(EXPECTED[name])))

    _, code, out, err = run(path, client)

    assert code == 0
    assert json.loads(out) == EXPECTED[name]
    assert err.splitlines() == [client.reply.usage, "done"]


def test_stdout_is_pure_json(run):
    name = "sample2_consulting_msa.txt"
    _, _, out, _ = run(
        CONTRACTS_DIR / name, FakeClient(text_reply(json.dumps(EXPECTED[name])))
    )
    json.loads(out)  # raises if anything but JSON was printed


def test_request_shape(run):
    name = "sample1_saas_subscription.txt"
    path = CONTRACTS_DIR / name
    client = FakeClient(text_reply(json.dumps(EXPECTED[name])))

    run(path, client)

    (request,) = client.requests
    assert request["model"] == cli.MODEL
    assert request["max_tokens"] >= 4000
    assert request["output_config"]["effort"] == "low"
    assert request["output_config"]["format"]["type"] == "json_schema"
    content = request["messages"][0]["content"]
    assert f"<contract>\n{path.read_text('utf-8')}\n</contract>" in content


def test_truncated_reply_reports_stop_reason(run):
    client = FakeClient(text_reply('{"parties": ["A"', stop_reason="max_tokens"))
    _, code, out, err = run(CONTRACTS_DIR / "sample2_consulting_msa.txt", client)

    assert code == 1
    assert out == ""
    assert "stop_reason=max_tokens" in err


def test_schema_mismatch_is_reported(run):
    client = FakeClient(text_reply('{"parties": "not a list"}'))
    _, code, out, err = run(CONTRACTS_DIR / "sample2_consulting_msa.txt", client)

    assert code == 1
    assert out == ""
    assert "ContractSummary schema" in err
    assert "validation errors" in err
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
    client = FakeClient(error=error)
    _, code, out, err = run(CONTRACTS_DIR / "sample2_consulting_msa.txt", client)

    assert code == 1
    assert out == ""
    assert message in err
    assert err.startswith("error: ")
    assert len(err.strip().splitlines()) == 1


class SequenceClient(FakeClient):
    """Replies with each canned reply in turn, recording a copy of every request."""

    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.replies.pop(0)


def tool_reply(data, block_id="toolu_1"):
    return SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(
                type="tool_use", id=block_id, name=cli.TOOL_NAME, input=data
            )
        ],
        usage="Usage(input_tokens=1, output_tokens=2)",
    )


SAMPLE = "sample1_saas_subscription.txt"


def test_json_retry_sends_bad_output_and_error(run):
    client = SequenceClient(
        [text_reply('{"parties": "not a list"}'), text_reply(json.dumps(EXPECTED[SAMPLE]))]
    )
    _, code, out, _ = run(CONTRACTS_DIR / SAMPLE, client)

    assert code == 0
    assert json.loads(out) == EXPECTED[SAMPLE]
    assert len(client.requests) == 2
    roles = [m["role"] for m in client.requests[1]["messages"]]
    assert roles == ["user", "assistant", "user"]
    first, bad, fix = client.requests[1]["messages"]
    assert "<contract>" in first["content"]
    assert bad["content"] == '{"parties": "not a list"}'
    assert "failed schema validation" in fix["content"]
    assert "parties" in fix["content"]


def test_stops_after_two_retries(run):
    client = SequenceClient([text_reply('{"parties": "x"}')] * 3)
    _, code, out, err = run(CONTRACTS_DIR / SAMPLE, client)

    assert code == 1
    assert out == ""
    assert len(client.requests) == 3  # first try + 2 retries
    assert "after 3 attempts" in err
    assert "ContractSummary schema" in err


def test_tool_mode_request_and_output(run, monkeypatch):
    client = FakeClient(tool_reply(EXPECTED[SAMPLE]))
    monkeypatch.setattr(cli.anthropic, "Anthropic", lambda: client)
    monkeypatch.setattr(
        sys, "argv", ["extractor", str(CONTRACTS_DIR / SAMPLE), "--mode", "tool"]
    )

    cli.main()

    (request,) = client.requests
    (tool,) = request["tools"]
    assert tool["input_schema"] == cli.ContractSummary.model_json_schema()
    assert request["tool_choice"] == {"type": "tool", "name": cli.TOOL_NAME}
    assert "format" not in request["output_config"]


def test_tool_mode_retry_uses_tool_result(monkeypatch, capsys):
    bad = {"parties": "not a list"}
    client = SequenceClient([tool_reply(bad), tool_reply(EXPECTED[SAMPLE], "toolu_2")])
    monkeypatch.setattr(cli.anthropic, "Anthropic", lambda: client)
    monkeypatch.setattr(
        sys, "argv", ["extractor", str(CONTRACTS_DIR / SAMPLE), "--mode", "tool"]
    )

    cli.main()

    assert json.loads(capsys.readouterr().out) == EXPECTED[SAMPLE]
    _, assistant, result = client.requests[1]["messages"]
    assert assistant["content"][0]["type"] == "tool_use"
    assert assistant["content"][0]["input"] == bad
    (block,) = result["content"]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_1"
    assert block["is_error"] is True
    assert "parties" in block["content"]


def test_compare_counts_retries_and_failures():
    from contract_extractor.extractor import compare

    good = json.dumps(EXPECTED[SAMPLE])
    client = SequenceClient(
        [
            text_reply(good),  # json run 1: clean
            text_reply("{}"), text_reply(good),  # json run 2: one retry
            tool_reply({}), tool_reply({}), tool_reply({}),  # tool run 1: fails
            tool_reply(EXPECTED[SAMPLE]),  # tool run 2: clean
        ]
    )
    stats = compare.compare(client, [CONTRACTS_DIR / SAMPLE], runs=2)

    assert stats["json"] == {"runs": 2, "retried": 1, "failed": 0, "attempts": 3}
    assert stats["tool"] == {"runs": 2, "retried": 1, "failed": 1, "attempts": 4}
