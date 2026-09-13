"""Pure safety calculations over a ledger Timeline."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import pandas as pd

from .ledger import Timeline, TimelineEntry


Payment = tuple[date, float]


def replay(
    timeline: Timeline,
    payments: Iterable[Payment],
    changes: Iterable[str],
) -> list[tuple[date, float]]:
    """Replay the timeline and return the balance after each movement date.

    Payment amounts are positive outflows. Changes apply only to projected
    entries whose event_id matches the referenced series' last_event_id.
    Entries on the same date are combined before the balance is recorded.
    """
    movements: dict[date, float] = {timeline.start_date: 0.0}

    for entry in _effective_entries(timeline.entries, changes):
        movements[entry.date] = movements.get(entry.date, 0.0) + entry.delta

    for payment_date, payment_amount in payments:
        when = _as_date(payment_date)
        if timeline.start_date <= when <= timeline.end_date:
            movements[when] = movements.get(when, 0.0) - float(payment_amount)

    balance = timeline.opening_balance
    result: list[tuple[date, float]] = []
    for when in sorted(movements):
        balance += movements[when]
        result.append((when, balance))

    return result


def is_safe(
    timeline: Timeline,
    min_balance: float,
    payments: Iterable[Payment],
    changes: Iterable[str],
) -> bool:
    """Return whether every replayed balance stays at or above the minimum."""
    balances = replay(timeline, payments, changes)
    return bool(balances) and min(balance for _, balance in balances) >= min_balance


def amount_safe_to_pay(
    timeline: Timeline,
    min_balance: float,
    requested_amount: float,
) -> float:
    """Return the maximum safe payment today using the closed-form rule."""
    baseline = replay(timeline, [], [])
    lowest_balance = min(balance for _, balance in baseline)
    safe_amount = max(0.0, min(float(requested_amount), lowest_balance - min_balance))
    return round(safe_amount, 2)


def earliest_full_payment_date(
    timeline: Timeline,
    min_balance: float,
    requested_amount: float,
) -> str:
    """Return the first date a single full payment can safely be made.

    Every calendar day in the window is a candidate, not only days that carry a
    movement: the earliest safe date is often the day after a large debit
    clears. The baseline is replayed once and scanned with prefix and suffix
    minima, so the whole search stays linear in the window length.
    """
    baseline = replay(timeline, [], [])
    if not baseline:
        return ""

    balance_by_date = dict(baseline)
    required_after_payment = min_balance + float(requested_amount)

    day = timeline.start_date
    daily: list[tuple[date, float]] = []
    running = timeline.opening_balance
    while day <= timeline.end_date:
        running = balance_by_date.get(day, running)
        daily.append((day, running))
        day += timedelta(days=1)

    balances = [balance for _, balance in daily]
    suffix_min = list(balances)
    for index in range(len(balances) - 2, -1, -1):
        suffix_min[index] = min(balances[index], suffix_min[index + 1])

    prefix_min_before = float("inf")
    for index, (when, balance) in enumerate(daily):
        if (
            prefix_min_before >= min_balance
            and suffix_min[index] >= required_after_payment
        ):
            return when.isoformat()
        prefix_min_before = min(prefix_min_before, balance)

    return ""


def _effective_entries(
    entries: Iterable[TimelineEntry],
    changes: Iterable[str],
) -> list[TimelineEntry]:
    parsed_changes = _parse_changes(changes)
    effective: list[TimelineEntry] = []

    for entry in entries:
        change = parsed_changes.get(entry.event_id)
        if entry.kind == "projected" and change is not None:
            action, amount = change
            if action == "stop":
                continue
            if action == "reduce_to":
                sign = 1.0 if entry.delta >= 0 else -1.0
                entry = entry._replace(delta=sign * amount)
        effective.append(entry)

    return effective


def _parse_changes(changes: Iterable[str]) -> dict[str, tuple[str, float]]:
    parsed: dict[str, tuple[str, float]] = {}

    for raw_change in changes:
        change = str(raw_change).strip()
        if change.startswith("stop:"):
            event_id = change.removeprefix("stop:").strip()
            if not event_id:
                raise ValueError("stop change requires an event_id")
            parsed[event_id] = ("stop", 0.0)
            continue

        if change.startswith("reduce_to:"):
            parts = change.split(":", maxsplit=2)
            if len(parts) != 3 or not parts[1].strip():
                raise ValueError(f"Invalid reduce_to change: {raw_change}")
            try:
                amount = float(parts[2].strip())
            except ValueError as error:
                raise ValueError(f"Invalid reduce_to amount: {raw_change}") from error
            if amount < 0:
                raise ValueError(f"reduce_to amount cannot be negative: {raw_change}")
            parsed[parts[1].strip()] = ("reduce_to", amount)
            continue

        if change:
            raise ValueError(f"Unknown spending change: {raw_change}")

    return parsed


def _as_date(value: date | pd.Timestamp | str) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()
