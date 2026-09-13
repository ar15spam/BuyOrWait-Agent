"""Validation for Buy or Wait? output rows."""

from __future__ import annotations

from datetime import date
from typing import Mapping, Sequence

import pandas as pd

from .ledger import Series

OUTPUT_COLUMNS = [
    "request_id", "amount_safe_to_pay", "affordability_status",
    "recommended_payment_method", "payment_plan",
    "earliest_date_for_full_payment", "spending_changes_needed",
    "decision_explanation",
]
STATUS_VALUES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHOD_VALUES = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def validate_rows(
    rows: Sequence[Mapping[str, object]],
    requests: pd.DataFrame,
    contexts: Mapping[str, object],
    series_by_request: Mapping[str, Sequence[Series]],
    payment_options: pd.DataFrame,
) -> None:
    """Assert that every output row obeys the challenge contract."""
    request_ids = [str(value) for value in requests["request_id"]]
    output_ids = [str(row.get("request_id", "")) for row in rows]
    assert len(output_ids) == len(request_ids), "wrong number of output rows"
    assert len(set(output_ids)) == len(output_ids), "duplicate request_id"
    assert set(output_ids) == set(request_ids), "output request_ids do not match input"
    for row in rows:
        request_id = str(row["request_id"])
        request = requests.loc[requests["request_id"] == request_id].iloc[0]
        _validate_row(row, request, contexts[request_id], series_by_request.get(request_id, ()), payment_options)


def _validate_row(row, request: pd.Series, context: object, series: Sequence[Series], payment_options: pd.DataFrame) -> None:
    requested_amount = float(request["requested_amount"])
    safe_amount = float(row["amount_safe_to_pay"])
    assert -1e-8 <= safe_amount <= requested_amount + 1e-8, f"{row['request_id']}: amount_safe_to_pay is out of bounds"
    status = str(row["affordability_status"]).strip()
    method = str(row["recommended_payment_method"]).strip()
    assert status in STATUS_VALUES, f"invalid affordability_status: {status}"
    assert method in METHOD_VALUES, f"invalid payment method: {method}"

    payments = _parse_plan(row["payment_plan"])
    request_date = _as_date(request["request_date"])
    desired_date = _as_date(request["desired_completion_date"])
    if method == "not_recommended":
        assert not payments, "not_recommended must have payment_plan=none"
    elif method == "installments":
        assert _matches_installment_option(str(request["request_id"]), payments, payment_options), f"{row['request_id']}: installment plan is not a supplied option"
    elif method == "partial_payment":
        assert status == "affordable_with_plan"
        assert len(payments) == 2, "partial payment must have two payments"
        assert payments[0][0] == request_date, "partial payment must start today"
        assert abs(sum(amount for _, amount in payments) - requested_amount) <= 0.01
        assert safe_amount > 0 and safe_amount < requested_amount
        assert abs(payments[0][1] - safe_amount) <= 0.01
        assert payments[1][0] <= desired_date
    if status == "affordable_now":
        assert str(row["earliest_date_for_full_payment"]).strip() == request_date.isoformat()
    _validate_changes(row["spending_changes_needed"], series, context)


def _parse_plan(value: object) -> list[tuple[date, float]]:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "none"}:
        return []
    payments: list[tuple[date, float]] = []
    previous: date | None = None
    for item in text.split("|"):
        parts = item.strip().split(":")
        assert len(parts) == 2, f"invalid payment plan entry: {item}"
        raw_date = parts[0].strip()
        when = pd.Timestamp(raw_date).date()
        assert when.isoformat() == raw_date, f"invalid payment date: {raw_date}"
        amount = float(parts[1].strip())
        assert amount >= 0, "payment amount cannot be negative"
        if previous is not None:
            assert when >= previous, "payment_plan is not chronological"
        previous = when
        payments.append((when, amount))
    return payments


def _matches_installment_option(request_id: str, payments: Sequence[tuple[date, float]], options: pd.DataFrame) -> bool:
    request_options = options.loc[(options["request_id"].astype(str) == request_id) & (options["payment_method"].astype(str).str.strip() == "installments")]
    for _, option in request_options.iterrows():
        count = int(option["number_of_payments"])
        first = _as_date(option["first_payment_date"])
        frequency = int(option["payment_frequency_days"])
        amount = float(option["payment_amount"])
        expected = [(first + pd.Timedelta(days=i * frequency), amount) for i in range(count)]
        expected = [(when.date() if isinstance(when, pd.Timestamp) else when, value) for when, value in expected]
        if len(payments) == len(expected) and all(actual_date == expected_date and abs(actual_amount - expected_amount) <= 0.01 for (actual_date, actual_amount), (expected_date, expected_amount) in zip(payments, expected)):
            return True
    return False


def _validate_changes(value: object, series: Sequence[Series], context: object) -> None:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "none"}:
        return
    actions = [part.strip() for part in text.split("|") if part.strip()]
    assert len(actions) <= 3, "at most three spending changes are allowed"
    by_id = {item.last_event_id: item for item in series}
    profile = context.profile
    protected = _pipe_set(profile.get("expense_categories_to_protect"))
    stop_categories = _pipe_set(profile.get("expense_categories_user_is_willing_to_stop"))
    reduce_categories = _pipe_set(profile.get("expense_categories_user_is_willing_to_reduce"))
    seen: set[str] = set()
    for action in actions:
        if action.startswith("stop:"):
            event_id = action.removeprefix("stop:").strip()
            assert event_id and event_id not in seen, "series cannot be changed twice"
            recurrence = by_id.get(event_id)
            assert recurrence is not None, f"unknown spending series: {event_id}"
            assert recurrence.direction == "debit"
            assert recurrence.flexibility in {"stoppable", "reducible_or_stoppable"}
            assert recurrence.category in stop_categories and recurrence.category not in protected
            seen.add(event_id)
            continue
        assert action.startswith("reduce_to:"), f"invalid spending change: {action}"
        parts = action.split(":", 2)
        assert len(parts) == 3 and parts[1].strip()
        event_id = parts[1].strip()
        recurrence = by_id.get(event_id)
        assert event_id not in seen and recurrence is not None
        assert recurrence.direction == "debit"
        assert recurrence.flexibility in {"reducible", "reducible_or_stoppable"}
        assert recurrence.category in reduce_categories and recurrence.category not in protected
        assert recurrence.minimum_allowed_amount is not None
        assert abs(float(parts[2]) - recurrence.minimum_allowed_amount) <= 0.01
        seen.add(event_id)


def _pipe_set(value: object) -> set[str]:
    if value is None:
        return set()
    try:
        if bool(pd.isna(value)):
            return set()
    except (TypeError, ValueError):
        pass
    return {part.strip() for part in str(value).split("|") if part.strip()}


def _as_date(value: object) -> date:
    return pd.Timestamp(value).date()
