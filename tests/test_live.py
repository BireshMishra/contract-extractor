"""Live eval: sends each sample contract to the real API and scores the extraction.

Opt in with `pytest --live`. Exact match for dates, ints and nulls; a key-phrase
containment check for free text (phrases in key_phrases.json; a phrase given as a
list is satisfied by any one of its alternatives). A free-text field whose expected
value is null must come back null.
"""

import json
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

from contract_extractor.extractor import __main__ as cli

pytestmark = pytest.mark.live

TESTS_DIR = Path(__file__).parent
CONTRACTS_DIR = cli.PACKAGE_DIR / "contracts"
EXPECTED = json.loads((TESTS_DIR / "expected.json").read_text("utf-8"))
KEY_PHRASES = json.loads((TESTS_DIR / "key_phrases.json").read_text("utf-8"))

EXACT_FIELDS = ["effective_date", "term_months", "termination_notice_days"]
FREE_TEXT_FIELDS = ["governing_law", "liability_cap"]


@pytest.fixture(scope="module", autouse=True)
def require_credentials():
    load_dotenv(cli.PROJECT_ROOT / ".env")
    if not (
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        pytest.skip("no Anthropic credentials available")


@pytest.mark.parametrize("mode", cli.MODES)
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_live_extraction(name, mode, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["extractor", str(CONTRACTS_DIR / name), "--mode", mode]
    )
    try:
        cli.main()
    except SystemExit as e:
        pytest.fail(f"extractor exited {e.code}: {capsys.readouterr().err.strip()}")
    actual = json.loads(capsys.readouterr().out)
    expected = EXPECTED[name]

    problems = []

    if {p.casefold() for p in actual["parties"]} != {
        p.casefold() for p in expected["parties"]
    }:
        problems.append(
            f"parties: expected {expected['parties']}, got {actual['parties']}"
        )

    for field in EXACT_FIELDS:
        if actual[field] != expected[field]:
            problems.append(
                f"{field}: expected {expected[field]!r}, got {actual[field]!r}"
            )

    for field in FREE_TEXT_FIELDS:
        value = actual[field]
        if expected[field] is None:
            if value is not None:
                problems.append(f"{field}: expected null, got {value!r}")
            continue
        if value is None:
            problems.append(f"{field}: expected text, got null")
            continue
        for phrase in KEY_PHRASES[name][field]:
            alternatives = [phrase] if isinstance(phrase, str) else phrase
            if not any(alt.casefold() in value.casefold() for alt in alternatives):
                problems.append(f"{field}: missing phrase {phrase!r} in {value!r}")

    assert not problems, f"{name} ({mode} mode):\n  " + "\n  ".join(problems)
