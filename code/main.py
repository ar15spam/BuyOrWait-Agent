"""Run the deterministic Buy or Wait? decision pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.env import load_env  # noqa: E402

load_env()

from code.build_context import (  # noqa: E402
    RATES,
    apply_resolved_amounts,
    build_context,
    load_requests,
    normalize_event_amounts,
)
from code.ledger import build_timeline, detect_series, residual_daily_spend  # noqa: E402
from code.plans import choose_plan  # noqa: E402
from code.safety import amount_safe_to_pay, earliest_full_payment_date  # noqa: E402
from code.validate import validate_rows  # noqa: E402
from code.usage import Call, Ledger, write_report  # noqa: E402
from code.writeout import write_output  # noqa: E402


class _LedgerBridge:
    """Adapt explain.py's per-request usage record to the unified ledger."""

    def __init__(self, ledger: Ledger, request_id: str) -> None:
        self._ledger = ledger
        self._request_id = request_id

    def add(self, record) -> None:  # noqa: ANN001 - explain.ExplanationUsage
        self._ledger.add(
            Call(
                "explanations",
                getattr(record, "request_id", self._request_id),
                getattr(record, "provider", "unknown"),
                getattr(record, "model", "unknown"),
                int(getattr(record, "input_tokens", 0)),
                int(getattr(record, "output_tokens", 0)),
            )
        )


def _apply_message_claims(enabled: bool, ledger: Ledger | None = None) -> int:
    """Reconcile message claims into the event table. Returns claims applied."""
    if not enabled:
        return 0
    try:
        from code.build_context import events as current_events
        from code.build_context import replace_events
        from code.extract.messages import extract_all_messages
        from code.reconcile import reconcile_events
    except ImportError as error:
        print(f"[messages] extraction unavailable: {error}")
        return 0
    try:
        results = extract_all_messages(ledger=ledger)
    except Exception as error:  # noqa: BLE001 - degrade, never fail the run
        print(f"[messages] extraction skipped ({type(error).__name__}: {error})")
        return 0
    outcome = reconcile_events(current_events, results)
    replace_events(outcome.events)
    applied = sum(1 for entry in outcome.audit if entry.outcome == "applied")
    print(f"[messages] {applied} claims applied, {len(outcome.audit) - applied} rejected")
    return applied


def _add_explanations(rows: list[dict[str, object]], decisions: list, currencies: dict[str, str],
                      ledger: Ledger, enabled: bool) -> int:
    """Replace templated explanations with model-written ones, one call each."""
    if not enabled:
        return 0
    try:
        from code.explain import explain_decision, replace_explanation
    except ImportError as error:
        print(f"[explain] unavailable, keeping templated text: {error}")
        return 0

    written = 0
    for index, decision in enumerate(decisions):
        try:
            text = explain_decision(decision, currencies[decision.request_id],
                                    usage=_LedgerBridge(ledger, decision.request_id))
        except Exception as error:  # noqa: BLE001 - one bad call must not lose the row
            print(f"[explain] {decision.request_id} kept templated text ({type(error).__name__})")
            continue
        rows[index] = replace_explanation(rows[index], text)
        written += 1
    print(f"[explain] {written} of {len(decisions)} explanations written by the model")
    return written


def _resolve_blank_amounts(enabled: bool, ledger: Ledger | None = None) -> int:
    """Fill blank event amounts from cached or freshly extracted image reads."""
    if not enabled:
        return 0
    try:
        from code.extract.images import extract_blank_amounts
        from code.extract.resolve import resolved_amounts
    except ImportError as error:
        print(f"[images] extraction unavailable, blanks stay unresolved: {error}")
        return 0
    try:
        extractions = extract_blank_amounts(ledger=ledger)
    except Exception as error:  # noqa: BLE001 - degrade, never fail the run
        print(f"[images] extraction skipped ({type(error).__name__}: {error})")
        return 0
    filled = apply_resolved_amounts(resolved_amounts(extractions))
    print(f"[images] resolved {filled} blank event amounts from {len(extractions)} images")
    return filled


def run(requests_path: str | Path, output_path: str | Path, limit: int | None = None,
        strategy: str = "median_gap", gap_statistic: str = "mean",
        use_images: bool = True, use_messages: bool = False,
        use_explanations: bool = False, residual_scale: float = 0.0,
        residual_lookback: int = 90) -> None:
    ledger = Ledger()
    images_resolved = _resolve_blank_amounts(use_images, ledger)
    _apply_message_claims(use_messages, ledger)
    requests = load_requests(requests_path)
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        requests = requests.head(limit).copy()

    options = pd.read_csv(ROOT / "dataset" / "request_payment_options.csv")
    rows: list[dict[str, object]] = []
    decisions: list = []
    currencies: dict[str, str] = {}
    contexts: dict[str, object] = {}
    series_by_request = {}

    for request_id in requests["request_id"].astype(str):
        context = build_context(request_id, requests_path)
        history = normalize_event_amounts(context.past_events, context.home_currency, RATES)
        series = detect_series(history, gap_statistic)  # type: ignore[arg-type]
        residual = (
            residual_daily_spend(
                history, series, context.request["request_date"], residual_lookback
            ) * residual_scale
            if residual_scale
            else 0.0
        )
        timeline = build_timeline(
            context, series, context.request["request_date"], strategy, residual  # type: ignore[arg-type]
        )
        minimum_balance = float(context.profile["minimum_balance_to_keep"])
        requested_amount = float(context.request["requested_amount"])
        safe_amount = amount_safe_to_pay(timeline, minimum_balance, requested_amount)
        earliest = earliest_full_payment_date(timeline, minimum_balance, requested_amount)
        decision = choose_plan(context, timeline, series, options, safe_amount, earliest)
        rows.append({
            "request_id": decision.request_id,
            "amount_safe_to_pay": decision.amount_safe_to_pay,
            "affordability_status": decision.affordability_status,
            "recommended_payment_method": decision.recommended_payment_method,
            "payment_plan": decision.payment_plan,
            "earliest_date_for_full_payment": decision.earliest_date_for_full_payment,
            "spending_changes_needed": decision.spending_changes_needed,
            "decision_explanation": decision.decision_explanation,
        })
        decisions.append(decision)
        currencies[request_id] = context.home_currency
        contexts[request_id] = context
        series_by_request[request_id] = series

    _add_explanations(rows, decisions, currencies, ledger, use_explanations)
    validate_rows(rows, requests, contexts, series_by_request, options)
    write_output(rows, output_path)
    write_report(ledger.calls, len(rows), cache_hits={"images": images_resolved})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Buy or Wait? predictions")
    parser.add_argument("--requests", default=str(ROOT / "dataset" / "requests.csv"))
    parser.add_argument("--out", default=str(ROOT / "dataset" / "output.csv"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--strategy", default="median_gap",
                        choices=["median_gap", "monthly_only", "monthly_plus_average"])
    parser.add_argument("--gap-stat", default="mean", choices=["mean", "median"],
                        help="statistic used for a series' inter-arrival gap")
    parser.add_argument("--no-images", action="store_true",
                        help="skip image extraction and leave blank amounts unresolved")
    parser.add_argument("--residual-scale", type=float, default=0.0,
                        help=("weight on undetected variable spending; measured on the "
                              "labelled set, 0 scored 63.3%%, 0.5 scored 64.0%%, "
                              "1.0 scored 51.3%%, so it defaults off"))
    parser.add_argument("--residual-lookback", type=int, default=90)
    parser.add_argument("--messages", action="store_true",
                        help="extract and reconcile message claims (needs an API key)")
    parser.add_argument("--explain", action="store_true",
                        help="write explanations with the model (needs an API key)")
    args = parser.parse_args()
    run(args.requests, args.out, args.limit, args.strategy, args.gap_stat,
        use_images=not args.no_images, use_messages=args.messages,
        use_explanations=args.explain, residual_scale=args.residual_scale,
        residual_lookback=args.residual_lookback)


if __name__ == "__main__":
    main()
