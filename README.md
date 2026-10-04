# contract-extractor

A small CLI that reads a contract as plain text and returns a structured summary as JSON, using Claude Sonnet 5.5 with schema-constrained output.

```console
$ uv run contract-extractor src/contract_extractor/contracts/sample1_saas_subscription.txt
{
  "parties": ["NORTHWIND ANALYTICS LTD", "BRIGHTFIELD RETAIL PLC"],
  "effective_date": "2025-04-01",
  "term_months": 24,
  "termination_notice_days": 90,
  "governing_law": "England and Wales",
  "liability_cap": "Each party's total aggregate liability is capped at the total fees paid by Customer in the twelve (12) months preceding the event ..."
}
```

Fields that the contract doesn't state come back as `null` rather than guessed.

## Setup

```console
uv sync
echo ANTHROPIC_API_KEY=sk-ant-... > .env
```

With no path argument it runs on the first sample contract.

## Design

- **Schema-constrained output.** The `ContractSummary` model ([models.py](src/contract_extractor/extractor/models.py)) is sent as a JSON schema, and the reply is validated with Pydantic. Each field has a description.
- **Prompt rules.** [prompts.py](src/contract_extractor/extractor/prompts.py) says which entities count as parties, how to derive dates and terms, and how to treat amendment letters, which must not borrow terms from the agreement they amend.
- **Pipe-friendly.** stdout is only the JSON. Token usage and `done` go to stderr, so `contract-extractor file.txt > out.json` works.
- **Failures are one line.** A bad path, an empty file, a missing or invalid key, rate limits, timeouts, network errors, a truncated reply and a schema mismatch each print a single `error: ...` line to stderr and exit 1. A truncated reply reports the `stop_reason`.

## Tests

```console
uv run pytest              # 19 offline tests with a fake client, no API calls
uv run pytest --live       # also runs the live eval (about 5 API calls)
```

The offline tests cover the plumbing: request shape, output cleanliness and every error path. They replay the answer key, so they don't measure extraction quality.

The live eval ([tests/test_live.py](tests/test_live.py)) sends the 5 sample contracts to the real API. Dates, integers and nulls must match exactly. Free text (`governing_law`, `liability_cap`) must contain key phrases ([key_phrases.json](tests/key_phrases.json)), and parties are compared as a case-insensitive set. The last recorded run passed 5 of 5 ([evidence/live_eval.txt](evidence/live_eval.txt)).

## Limitations

- The answer key ([expected.json](tests/expected.json)) was hand-written by the author, with no independent review. Some entries are judgment calls, such as using the last signature date as the effective date.
- Five short contracts is a smoke test, not a benchmark. There is no measurement of variance across runs or of performance on long or messy documents.
- Input is plain UTF-8 text only. There is no PDF or DOCX parsing and no chunking, so a very long contract goes into a single request.

## Output modes and failure-rate comparison

`--mode json` (default) asks for schema-constrained JSON; `--mode tool` offers a strict
`record_summary` tool whose `input_schema` is the `ContractSummary` JSON schema
(`transform_schema`), with `tool_choice` set to `auto` (Sonnet 5.5 returns HTTP 400 for a
forced tool) and the system prompt telling the model to call it. In both
modes, invalid output is sent back (the bad output plus the validation error) for up to 2
retries, then the run fails with a clear error.

## Providers

`--provider claude` (default) or `--provider openai` (needs `OPENAI_API_KEY`). Both implement
`LLMClient` ([providers.py](src/contract_extractor/extractor/providers.py)): one method,
`complete(system, user, schema)` returning text, token counts and a normalised `stop_reason`
(`end`, `max_tokens`, `refusal`, `other`). The schema goes to each provider's own
structured-output feature: Claude's `output_config` JSON schema (`--mode json`, default) or a
strict tool (`--mode tool`, Claude only), and OpenAI's strict `response_format`. Each client
turns its SDK's errors into `ExtractionError`, so the validate-and-retry loop
([extract.py](src/contract_extractor/extractor/extract.py)) is written once: a cut-off or refused
reply fails with its stop reason, and invalid output is retried up to twice. Claude runs at low
effort so thinking does not eat `max_tokens`. Because `complete()` takes one user string, a
retry resends the request with the bad output and the error appended, rather than passing the
reply back as a real assistant turn (so no thinking-block or `tool_result` passthrough). The
OpenAI path has not been run against the real API.

`python -m contract_extractor.extractor.compare --runs 5 --out evidence/mode_comparison.txt`
runs every sample through both modes and reports how many runs needed a retry or failed
outright. It makes real, paid API calls; no results are recorded yet.
