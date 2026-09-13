# Buy or Wait? — a deterministic financial decision agent

A personal-finance affordability engine built with **Python**, **pandas**, **pydantic**, and **GPT-4o-mini**, written for the HackerRank Orchestrate 24-hour hackathon (September 2026).

Given a user asking *"can I afford this laptop?"*, the system reconstructs their financial position from 25,342 transaction records, forecasts their balance 90 days forward, and decides whether they should pay in full, pay part now, take an installment plan, wait, or not proceed — while keeping them above the minimum balance they asked to protect.

> **Result:** 63.3% on the 25 labelled requests, 250/250 valid output rows, 464 model calls, **$0.0495 total inference cost** ($0.000198 per request), 4.2-second deterministic run.

---

## The result

Scored against the 25 labelled requests with `make score`:

| Output column | Score |
|---|---|
| `amount_safe_to_pay` | 12% exact, **44% within 5%**, 190% MAPE |
| `affordability_status` | **72%** (4-class) |
| `recommended_payment_method` | **76%** (5-class) |
| `payment_plan` | **72%** exact string match |
| `earliest_date_for_full_payment` | **60%** exact date |
| `spending_changes_needed` | **88%** set equality |
| **Headline** (mean of six) | **63.3%** |

25 labelled rows means one request is worth 4 points. I treat every number here as ±8 and made decisions on measured deltas, not single runs.

### Ablation — what each layer was actually worth

| Configuration | Score | Δ |
|---|---|---|
| Deterministic only, no model calls | 60.7% | — |
| \+ vision extraction of 16 blank amounts | 60.7% | **0.0** |
| \+ message claim extraction and reconciliation | **63.3%** | **+2.6** |

The vision layer resolves all 16 blank amounts correctly and changes the score by nothing, because only 11 of 250 requests touch one and 8 of those sit in history. The message layer touches **97 of 250 requests**. Building the ablation harness first is the only reason I know which of the two deserved the remaining hours.

---

## The problem

| Input | Scale |
|---|---|
| Requests to decide | 250 (+25 labelled) |
| Financial events | 25,342 |
| User profiles | 275 |
| Currencies | 5 (EUR, IDR, INR, USD, ZAR) |
| Dated FX rates | 134 |
| Seller payment options | 790 |
| Free-text messages | 215 (multilingual) |
| Document images | 16 PNGs |
| Events with a blank amount | 16 |

Every scored field is an exact value — a number, an enum, a date, or a plan string that must reproduce a supplied option character-for-character.

---

## Design principle: the LLM perceives, the code decides

```text
CSVs ──► currency normalisation ──► history / future split
                                          │
        16 PNGs ──► [VISION MODEL] ───────┤   schema-validated JSON
   215 messages ──► [LANGUAGE MODEL] ─────┤   schema-validated JSON
                                          ▼
                                   claim reconciliation
                                   (precedence rules)
                                          ▼
                              recurring-series detection
                                          ▼
                            90-day dated cashflow timeline
                                          ▼
                      safety replay ──► amount_safe_to_pay
                                   └──► earliest_safe_date
                                          ▼
                    plan enumeration ──► filter ──► rank ──► status
                                          ▼
                                 [LANGUAGE MODEL] explanation
                                          ▼
                              invariant assertions ──► output.csv
```

Model calls are confined to three jobs: reading a PNG, interpreting a message, and writing one sentence. Each emits JSON validated against a pydantic schema, and anything that fails validation is **discarded rather than repaired**.

Three consequences:

- **Reproducible.** Cache the perception layer and the rest is a pure function. Reruns are free and identical.
- **Injection-resistant.** A malicious instruction in `messages.csv` cannot reach a decision. The extraction prompt states the content is data to be described, never instructions to follow, and the model is never shown the output enums — it is not in a position to leak a decision even if it wanted to. The worst an injected string can do is produce a claim that fails validation and gets dropped.
- **Cheap.** Cost scales with 215 distinct messages and 16 distinct images, not with 250 requests.

---

## The 90-day safety check

A plan is safe only if the projected balance never falls below `minimum_balance_to_keep` at any point in the 90-day window.

The naive way to find the largest safe payment is a binary search over the amount. It isn't necessary. Paying `X` today shifts **every** later balance down by exactly `X`, so safety is linear in today's outflow:

```text
amount_safe_to_pay = clamp(
    min(projected_balance) - minimum_balance,
    0,
    requested_amount
)
```

One pass instead of ~40 replays per request. `earliest_date_for_full_payment` falls out of the same replay using prefix and suffix minima — the first day where every balance before it clears the minimum and every balance after it clears `minimum + requested`, again linear rather than re-replaying per candidate date.

This closed form reproduces the ground truth exactly on every `affordable_now` case.

---

## Recurrence detection — the actual hard part

The dataset gives each user **0–2 events dated after their request** against roughly 100 rows of history. The 90-day forecast cannot be filtered out of the file; it has to be *generated*.

```text
group history by (description, category, event_type, direction)
        ↓
require ≥ 3 occurrences
        ↓
cadence = mean inter-arrival gap
        ↓
if 27–32 days and occurrences cluster on one day-of-month
        → calendar-monthly, anchored to that day
   else → fixed day-count stepping
        ↓
project forward, skipping cycles a real event already covers
```

Two decisions here were worth measurable points:

**Calendar months, not day counts.** Stepping a monthly series by its median 30-day gap drifts a salary paid on the 15th to the 13th two months out. Every date-valued output landed 1–2 days early. Anchoring monthly series to their day-of-month took `earliest_date_for_full_payment` from **36% → 56%** and `payment_plan` from **60% → 68%**.

**Mean gap, not median.** Gap distributions are right-skewed, so the median sits below the true average interval and manufactures extra occurrences. Switching to the mean moved `amount_safe_to_pay` within-5% from **32% → 40%** and brought the predicted `not_affordable` share on the full 250 from 107 down to 86, against a labelled base rate near 24%.

---

## Perception layer

### Vision — 16 blank amounts

A blank `amount` is not zero; the value lives in a linked PNG. The documents are genuinely adversarial: a payslip showing **Total Earnings 4,780,800** *and* **Net Pay 4,365,000**; a rent receipt showing **invoice total 200,000**, **amount received 100,000** and **balance due 100,000**.

The event description decides which line is correct — "August 2019 net salary" means net pay, "Outstanding rent balance" means balance due. The extractor passes the entry's description, category, direction, currency and date alongside the image. Without that context the model is guessing between three plausible totals.

All 16 resolve. An extraction whose reported currency disagrees with the event row is dropped rather than reconciled, because guessing which side is wrong would put an unsupported number into a safety calculation.

### Language — 215 messages, 122 applied claims

Messages clarify, amend, cancel, delay or confirm financial facts, in several languages:

```text
"Rincian penggajian Anda di Cobalt Systems telah berubah.
 Gaji bulanan Anda naik menjadi IDR 42750000.
 Perubahan ini berlaku mulai 2025-08-15."

        ↓ extracted claim

{ "kind": "amend_amount",
  "target_event_id": "event_136",
  "new_amount": 42750000,
  "new_date": "2025-08-15" }
```

That single claim corrected one request's earliest safe date, status, method and payment plan simultaneously.

Of 158 claims emitted across 215 messages, **122 apply**:

| Claim kind | Applied |
|---|---|
| `amend_amount` | 97 |
| `cancel` | 11 |
| `delay` | 6 |
| `confirm` | 4 |
| `none` | 4 |

The 36 rejections are all principled: 16 name an event ID that does not exist, and 20 are confirmations or settlements against already-settled events, which the spec says must lose to the settled record. **Zero schema failures.**

Conflicts resolve by the spec's precedence order — explicit cancellation or amendment first, then the newer assertion from the same source, then settled over estimate, then the financially safer reading. Every applied and rejected claim is retained in an audit trail.

---

## Measured negative result

Recurrence needs 3+ occurrences, so every one-off and twice-seen debit is dropped. I measured how much that discards: **13.4% of each user's recent spending**. Since 16 of 25 predictions were over-optimistic, carrying it forward as a flat daily rate looked like the obvious correction.

It isn't:

| Residual weight | Score |
|---|---|
| 0.0 (off) | **63.3%** |
| 0.5 | 64.0% |
| 1.0 | 51.3% |
| 1.5 | 42.7% |

Adding spending the model is provably missing makes it **12 points worse**. The ground-truth forecast evidently models only detected recurring series and ignores irregular spending entirely.

The code is still in the repo behind `--residual-scale`, defaulting to zero, with the measurement recorded in its docstring. The negative result is more useful than the feature would have been.

---

## Four bugs worth naming

**1. 83% of installment options were silently discarded.** A window guard rejected any plan whose payments extended past `request_date + 90`. That killed 428 of 515 installment options and *every* option for 194 of 275 requests — they could never win regardless of quality. The safety replay already ignores out-of-window payments, so the guard was pure damage.

**2. The message extractor never showed the model any event IDs.** It validated claims against the real ID set afterward but never put that set in the prompt. The model had to invent IDs, and reconciliation would have rejected essentially all of them — 215 calls to change nothing. Supplying a candidate list of the user's events turned the stage from inert into +2.6 points.

**3. A required `asserted_at` field the model always returned as null.** Every claim failed schema validation. The fix was not to loosen the schema but to stop asking: `messages.csv` already carries `sent_at`, so requesting a value already known only added a failure mode and cost tokens. Caught by a three-call smoke test before spending 465.

**4. Cost reported as $0.0000.** The API echoes a dated model id (`gpt-4o-mini-2024-07-18`) and the pricing table was keyed on the base name, so every lookup silently fell through to zero. Now resolved by longest-prefix match, and unpriced models are named explicitly rather than reported as free.

---

## Cost and performance

| Metric | Value |
|---|---|
| Model calls | 464 (214 messages, 250 explanations) |
| Input / output tokens | 266,682 / 15,912 |
| Total tokens | 282,594 |
| **Estimated cost** | **$0.0495** |
| Cost per request | $0.000198 |
| Model-written explanations | 249 / 250 |
| Deterministic runtime, 250 requests | **4.2 s** |
| Cached artifacts | 480 (16 images, 215 messages, 249 explanations) |

Runtime came down from 5.5s by profiling rather than guessing. The largest single win: a residual-spend calculation ran for all 250 requests and was then multiplied by a default weight of zero — a quarter of the runtime spent on a discarded result. Three more followed the same shape: wrapping every row in a fresh `pandas.Series` to read four fields, scanning and re-stringifying all 25,342 event rows once per claim, and re-reading the requests CSV on every request.

Every extraction is cached per artifact and explanations are keyed by prompt hash, so a rerun with unchanged decisions costs nothing and a changed decision is re-explained.

---

## Project structure

4,048 lines across 21 modules.

```text
code/
├── main.py             212   pipeline entry point and CLI
├── build_context.py    305   CSV loading, FX normalisation, per-request context
├── ledger.py           499   recurrence detection, 90-day timeline
├── safety.py           170   balance replay, closed-form safe amount
├── plans.py            396   candidate enumeration, ranking, status derivation
├── reconcile.py        168   claim precedence and audit trail
├── validate.py         154   output invariants, asserted before write
├── writeout.py          32   submission serialisation
├── usage.py            177   unified token ledger and cost report
├── explain.py          365   explanation generation, prompt-hash cache
├── env.py               75   .env loading, secrets never logged
├── extract/
│   ├── images.py       263   vision extraction
│   ├── messages.py     426   claim extraction with candidate grounding
│   ├── schemas.py       53   strict pydantic models
│   ├── cache.py         84   per-artifact response cache
│   └── resolve.py       56   extraction → event amounts
└── evaluation/
    ├── score.py        492   per-column scoring, confusion matrices, diffs
    └── smoke.py        121   one call per stage before spending 465
```

---

## Running it

```bash
python3 -m venv venv
venv/bin/pip install pandas pydantic
cp .env.example .env          # OPENAI_API_KEY=sk-...
```

```bash
make check     # byte-compile every module
make test      # unit tests
make run       # deterministic only, no API key needed
make smoke     # 3 calls, verifies config before spending 465
make run-llm   # full pipeline with claims and explanations
make score     # score the 25 labelled requests
```

`make run` produces a complete, valid `output.csv` with no API key at all. The model layers are additive.

Secrets are read from the environment only. `.env` is gitignored and was verified absent from git history.

---

## Limitations

**`amount_safe_to_pay` is 12% exact.** The error is bimodal: seven requests land within 3% and four are catastrophically wrong. Both failure modes trace to recurrence fidelity — the projected amount or cadence of a series — not to the safety mathematics, which is exact.

**The labelled set is 25 rows.** One request is 4 points. Late-stage parameter tuning produced differences inside that noise, so I stopped rather than overfit.

**Recurrence is monthly-or-fixed-interval.** Users whose income is irregular gig payouts rather than a fixed-day salary are modelled poorly; one such user in the labelled set is among the four catastrophic misses.

**Explanations are unverified.** The sentence is generated after the decision is frozen and cannot alter any scored field, but its wording is not checked against the numbers beyond asserting which dates it may name.

---

## What I took away

**Build the scorer before the solver.** The first module written was `score.py`, with a self-test that scoring the labelled file against itself must return exactly 100%. Every subsequent decision was a measured delta. Three of the changes I was most confident about — residual spending, median gaps, parameter tuning — made things worse or did nothing, and I would have shipped all three on intuition.

**Measure the layer before building it.** Vision extraction was the headline feature in my design document and it is worth zero. Ten minutes counting how many requests actually touch a blank amount (11 of 250) would have reordered the whole plan.

**A smoke test that makes three calls.** A required field the model always returned as null would have wasted all 215 message calls. Three calls found it.

**Profile, don't guess.** Every one of the four performance wins was invisible to reading the code and obvious in `cProfile` output.

**Negative results are results.** The residual-spending experiment failed and taught me more about the ground-truth generator than any successful change did.

---

## Tech stack

```text
Python 3.12
pandas          dataframe pipeline over 25k events
pydantic        strict schema validation of all model output
OpenAI API      gpt-4o-mini, vision and text
cProfile        performance work
Make            reproducible entry points
```
