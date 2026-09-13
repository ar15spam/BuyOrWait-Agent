"""One call per stage, so a broken config costs three calls instead of 465.

Run this before the full extraction:

    python3 -m code.evaluation.smoke

It makes at most one image call, one message call and one explanation call,
prints what came back, and writes each response to the normal cache so the
full run reuses it. Nothing here prints or stores a key.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.env import describe, load_env  # noqa: E402

_LOADED = load_env()

import os  # noqa: E402

import pandas as pd  # noqa: E402


def check_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "")
    if key.strip():
        source = ".env" if "OPENAI_API_KEY" in _LOADED else "shell environment"
        print(f"OPENAI_API_KEY: found ({len(key)} chars, from {source})")
        return True
    print("OPENAI_API_KEY: MISSING. What the loader can see:")
    print(describe())
    print("\n  Fix: make sure .env sits next to the Makefile and contains one line,")
    print("       OPENAI_API_KEY=sk-...   (quotes optional, no spaces around =)")
    print("       and that your shell does not export a blank OPENAI_API_KEY.")
    return False


def smoke_image() -> None:
    from code.extract.images import (OpenAICompatibleVisionClient, event_context,
                                     extract_image, linked_blank_amounts)
    row = linked_blank_amounts().iloc[0]
    image_id = str(row.image_id)
    print(f"\n[image] {image_id} (event {row.event_id}: {row.description})")
    try:
        result = extract_image(image_id, client=OpenAICompatibleVisionClient.from_environment(),
                               context=event_context(row))
    except Exception as error:  # noqa: BLE001
        print(f"  FAILED: {type(error).__name__}: {error}")
        return
    if result is None:
        print("  returned nothing usable (response cached for inspection)")
        return
    print(f"  amount={result.amount} {result.currency} date={result.date} "
          f"confidence={result.confidence}\n  evidence: {result.evidence}")


def smoke_message() -> None:
    from code.extract.messages import (OpenAICompatibleMessageClient, candidate_events,
                                       extract_message_claims)
    events = pd.read_csv(ROOT / "dataset" / "financial_events.csv")
    messages = pd.read_csv(ROOT / "dataset" / "messages.csv")
    row = messages.iloc[0]
    print(f"\n[message] {row.message_id} ({row.source_type})")
    print(f"  text: {str(row.message_text)[:110]}...")
    related = "" if pd.isna(row.related_event_id) else str(row.related_event_id)
    try:
        result = extract_message_claims(
            str(row.message_id), str(row.message_text),
            set(events["event_id"].astype(str)),
            client=OpenAICompatibleMessageClient.from_environment(),
            candidates=candidate_events(events, str(row.user_id), related),
        )
    except Exception as error:  # noqa: BLE001
        print(f"  FAILED: {type(error).__name__}: {error}")
        return
    for claim in result.claims:
        print(f"  claim: {claim.kind} -> {claim.target_event_id} "
              f"amount={claim.new_amount} date={claim.new_date}")
    for rejection in result.rejected:
        print(f"  rejected: {rejection.reason}")
    if not result.claims and not result.rejected:
        print("  no claims (valid if the message asserts nothing about a supplied event)")


def smoke_explanation() -> None:
    from code.build_context import RATES, build_context, normalize_event_amounts
    from code.explain import explain_decision
    from code.ledger import build_timeline, detect_series
    from code.plans import choose_plan
    print("\n[explanation] request_01")
    context = build_context("request_01", ROOT / "dataset" / "sample_requests.csv")
    history = normalize_event_amounts(context.past_events, context.home_currency, RATES)
    series = detect_series(history, "mean")
    timeline = build_timeline(context, series, context.request["request_date"])
    options = pd.read_csv(ROOT / "dataset" / "request_payment_options.csv")
    decision = choose_plan(context, timeline, series, options)
    try:
        print("  " + explain_decision(decision, context.home_currency))
    except Exception as error:  # noqa: BLE001
        print(f"  FAILED: {type(error).__name__}: {error}")


def main() -> None:
    if not check_key():
        raise SystemExit(1)
    smoke_image()
    smoke_message()
    smoke_explanation()
    print("\nIf all three printed sensible output, run the full extraction:")
    print("  python3 code/main.py --requests dataset/requests.csv "
          "--out dataset/output.csv --messages --explain")


if __name__ == "__main__":
    main()
