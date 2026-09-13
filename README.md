# Buy or Wait?

An AI-assisted financial affordability agent for the HackerRank Orchestrate hackathon.

For each request, it reconstructs a user’s cash position, forecasts the next 90 days, and recommends one of:

- full payment
- partial payment
- installments
- wait
- do not proceed

The recommendation must keep the balance above the user’s required minimum.

## Current result

On the 25 labelled requests:

| Metric | Result |
|---|---:|
| Headline score | **63.3%** |
| Valid full-dataset rows | **250/250** |
| Model calls | **464** |
| Estimated model cost | **$0.0495** |

See [`code/evaluation/usage_report.md`](code/evaluation/usage_report.md) for token and cost details.

## How it works

```text
CSV data
  → currency-normalised events
  → image/message extraction when enabled
  → claim reconciliation
  → recurring-series detection
  → 90-day cash-flow timeline
  → safety checks
  → plan enumeration and ranking
  → validated output.csv
```

The code makes decisions deterministically. Models are used only to:

1. read linked financial images;
2. extract claims from messages;
3. write the final explanation.

Model responses are schema-validated, cached under `.cache/`, and rejected when malformed. Message and image content is treated as untrusted data, never as instructions.

## Quick start

```bash
source venv/bin/activate
make check
make score
make run
```

The full output is written to `dataset/output.csv`.

## Optional model run

Put the key in a local, gitignored `.env` file:

```text
OPENAI_API_KEY=your_key_here
```

Test access before spending model calls:

```bash
make smoke
```

Run image extraction, message reconciliation, and model-written explanations with:

```bash
make run-llm
```

Cached results are reused on later runs. Never commit `.env`, API keys, or `.cache/`.

## Useful commands

```bash
# Run a different request file
python3 code/main.py --requests dataset/requests.csv --out dataset/output.csv

# Process only the first N requests
python3 code/main.py --requests dataset/requests.csv --out /tmp/pred.csv --limit 10

# Try alternate recurrence strategies
python3 code/main.py --strategy monthly_only
python3 code/main.py --strategy monthly_plus_average
```

## Important design rules

- Blank event amounts are unresolved—not zero.
- Foreign-currency events use the dated exchange-rate table.
- Pending debits are reserved; pending credits are ignored.
- Failed, cancelled, duplicate, unrealized, and non-cash records do not become available cash.
- Recurring series require at least three historical occurrences.
- Payments and essential projected expenses must stay above `minimum_balance_to_keep`.
- Spending changes may target only flexible, non-protected categories the user permits.
- Installment plans must exactly match a supplied payment option.
- Validation runs before any output row is written.

## Repository layout

```text
dataset/
  requests.csv                 evaluation requests
  sample_requests.csv          labelled examples
  financial_profiles.csv       balances and preferences
  financial_events.csv         transaction history
  exchange_rates.csv           dated currency rates
  request_payment_options.csv  seller payment schedules
  messages.csv                 supporting text evidence
  images.csv                   image/event links
  media/images/                linked PNG documents

code/
  main.py                      pipeline entry point
  build_context.py             loading and currency normalisation
  ledger.py                    recurrence detection and timeline building
  safety.py                    balance replay and safe-amount math
  plans.py                     candidate plans and ranking
  reconcile.py                 message-claim precedence and audit trail
  extract/                     image and message extraction
  validate.py                  output invariants
  writeout.py                  CSV serialization
  evaluation/                  scorer, smoke test, usage report
```

## Submission checklist

```bash
make check
make score
make run
wc -l dataset/output.csv       # 251, including the header
```

Submit the required code archive, final output, and chat transcript. Do not include credentials or private cache files.
