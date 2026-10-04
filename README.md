# contract-extractor

A small CLI that reads a contract as plain text and returns a structured summary as JSON. It runs on Claude Sonnet 5.5 or OpenAI, both through one `LLMClient` interface and one validate-and-retry loop.

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

## What it does

- Sends the contract and the `ContractSummary` JSON schema ([models.py](src/contract_extractor/extractor/models.py)) to the model, using each provider's own structured-output feature.
- Validates the reply with Pydantic. If it is invalid, it sends the request again with the bad output and the validation error appended, up to 2 retries, then fails with a clear error.
- Reports a cut-off or refused reply as `stop_reason=...` instead of retrying it.
- Keeps stdout as pure JSON. Token usage goes to stderr, so `contract-extractor file.txt > out.json` works.
- Prints a single `error: ...` line and exits 1 for a bad path, empty file, missing or invalid key, rate limit, timeout, network error, truncated reply or schema mismatch.

## How to run

```console
uv sync
echo ANTHROPIC_API_KEY=sk-ant-... >> .env
echo OPENAI_API_KEY=sk-...        >> .env      # only for --provider openai

uv run contract-extractor [path]                       # Claude, schema-constrained JSON (default)
uv run contract-extractor [path] --mode tool           # Claude, strict tool call
uv run contract-extractor [path] --provider openai     # OpenAI, strict response_format
```

With no path it runs on the first sample contract. `--mode tool` offers a strict `record_summary` tool and sets `tool_choice` to `auto`, because Sonnet 5.5 returns HTTP 400 for a forced tool.

```console
# Time, token and cost table for all 5 samples on both providers (about 10 paid calls)
uv run python -m contract_extractor.extractor.benchmark --out evidence/benchmark.md

# Failure rate of --mode json vs --mode tool on Claude (paid; --runs per contract per mode)
uv run python -m contract_extractor.extractor.compare --runs 5 --out evidence/mode_comparison.txt
```

## Results

One run of each sample on each provider ([evidence/benchmark.md](evidence/benchmark.md)). Seconds are wall-clock time per call, measured with `time.perf_counter()`, summed over retries. Cost is `input tokens × input price + output tokens × output price` ([pricing.py](src/contract_extractor/extractor/pricing.py)).

| Contract | Provider | Seconds | Tokens (in/out) | Cost (USD) | Retries | Null fields |
|---|---|---:|---:|---:|---:|---|
| sample1_saas_subscription.txt | claude | 4.1 | 2,239 / 183 | $0.0063 | 0 | none |
| sample1_saas_subscription.txt | openai | 3.0 | 1,180 / 78 | $0.0030 | 0 | none |
| sample2_consulting_msa.txt | claude | 3.0 | 2,246 / 129 | $0.0058 | 0 | governing_law |
| sample2_consulting_msa.txt | openai | 1.4 | 1,130 / 66 | $0.0028 | 0 | governing_law |
| sample3_mutual_nda.txt | claude | 2.2 | 1,955 / 110 | $0.0050 | 0 | liability_cap |
| sample3_mutual_nda.txt | openai | 1.5 | 1,010 / 60 | $0.0025 | 0 | liability_cap |
| sample4_supply_agreement.txt | claude | 2.7 | 2,057 / 151 | $0.0056 | 0 | termination_notice_days |
| sample4_supply_agreement.txt | openai | 1.2 | 1,051 / 80 | $0.0027 | 0 | termination_notice_days |
| sample5_amendment_letter.txt | claude | 2.1 | 1,721 / 81 | $0.0043 | 0 | term_months, governing_law, liability_cap |
| sample5_amendment_letter.txt | openai | 2.5 | 858 / 51 | $0.0021 | 0 | term_months, governing_law, liability_cap |

Totals: Claude 14.1 s and $0.0270, OpenAI 9.6 s and $0.0131. No call needed a retry. In every row the null fields match the answer key's nulls ([expected.json](tests/expected.json)). That only checks which fields are null; the other values were not compared in this run.

Read this with care:
- It is one run per cell, so the timings are noisy and say nothing about variance.
- The OpenAI price in `pricing.py` ($2 / $8 per million tokens for `gpt-4.1`) was entered from memory and not checked against OpenAI's pricing page. The Claude price ($2 / $10) is Anthropic's published Sonnet 5.5 rate. The OpenAI cost column is only as good as that entry.
- The two providers use different tokenizers, so the token columns are not comparable like for like.

## Three things I learned

1. **A fake client hides what the real API rejects.** I forced the tool call with `tool_choice: {"type": "tool"}`, and the offline tests passed because the fake client accepts anything. The real API returns HTTP 400 on Sonnet 5.5. The fix was `tool_choice: auto` plus an instruction in the system prompt. Offline tests prove the plumbing; only a live call proves a request is valid, which is why the live eval runs both modes.
2. **A thin interface drops provider features.** `complete(system, user)` returning text could not carry structured output, a stop reason or error handling. Widening it (a `schema` argument, a normalised `stop_reason`, errors mapped to `ExtractionError` inside each client) let one retry loop serve both providers. The price was that a retry is now one user string with the bad output appended, not a real assistant turn, so thinking blocks and `tool_result` messages are no longer passed back.
3. **The same text costs different token counts on different providers.** Claude counted about twice as many input tokens as OpenAI for the same contract (for example 2,239 against 1,180 for sample 1). Cost has to be computed from each provider's own usage numbers, and a per-token price comparison alone would mislead.

## Design

- **One interface, two clients.** `LLMClient.complete(system, user, schema)` returns text, token counts and a normalised `stop_reason` (`end`, `max_tokens`, `refusal`, `other`) ([providers.py](src/contract_extractor/extractor/providers.py)). Each client turns its SDK's errors into `ExtractionError`, so the loop in [extract.py](src/contract_extractor/extractor/extract.py) is written once.
- **Provider-native structured output.** Claude uses `output_config` with a JSON schema (`--mode json`) or a strict tool (`--mode tool`). OpenAI uses a strict `response_format`, with every object closed and every property required.
- **Low effort for Claude.** Without it, thinking can use up `max_tokens` and truncate the reply.
- **Prompt rules.** [prompts.py](src/contract_extractor/extractor/prompts.py) says which entities count as parties, how to derive dates and terms, and how to treat amendment letters, which must not borrow terms from the agreement they amend.

## Tests

```console
uv run pytest              # 60 offline tests with fake clients, no API calls
uv run pytest --live       # also runs the live eval (10 API calls: 5 samples x json/tool mode)
```

The offline tests cover how each client builds its request, the stop-reason and error mapping, the single retry loop, the CLI routing, the timing and cost arithmetic, and every error path. They replay the answer key, so they don't measure extraction quality.

The live eval ([tests/test_live.py](tests/test_live.py)) sends the 5 sample contracts to the real Claude API in both modes. Dates, integers and nulls must match exactly. Free text (`governing_law`, `liability_cap`) must contain key phrases ([key_phrases.json](tests/key_phrases.json)), and parties are compared as a case-insensitive set. The last recorded run, 5 of 5 passing ([evidence/live_eval.txt](evidence/live_eval.txt)), predates the provider refactor and the tool mode. The live eval has not been re-run since, and it has no OpenAI case.

## Limitations

- The answer key ([expected.json](tests/expected.json)) was hand-written by the author, with no independent review. Some entries are judgment calls, such as using the last signature date as the effective date.
- Five short contracts is a smoke test, not a benchmark. There is no measurement of variance across runs or of performance on long or messy documents.
- The OpenAI path has only been run through the benchmark above. `OPENAI_MODEL` (`gpt-4.1`) is a default I chose, not a recommendation.
- `--mode json` against `--mode tool` failure rates ([compare.py](src/contract_extractor/extractor/compare.py)) have not been measured; no results are recorded yet.
- Input is plain UTF-8 text only. There is no PDF or DOCX parsing and no chunking, so a very long contract goes into a single request.
