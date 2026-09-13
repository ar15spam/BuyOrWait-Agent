from datetime import date

from code.ledger import Timeline, TimelineEntry
from code.safety import amount_safe_to_pay, earliest_full_payment_date, is_safe, replay

def make_timeline() -> Timeline:
    return Timeline(
        opening_balance=1200.0,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 4),
        entries=[
            TimelineEntry(date(2026, 1, 2), -100.0, "rent-1", "projected", "fixed", "rent"),
            TimelineEntry(date(2026, 1, 3), -300.0, "flex-1", "projected", "stoppable", "dining"),
            TimelineEntry(date(2026, 1, 4), 100.0, "income-1", "real", "fixed", "salary"),
        ],
    )


def test_amount_safe_is_bounded() -> None:
    timeline = make_timeline()
    safe = amount_safe_to_pay(timeline, 500.0, 900.0)
    assert 0.0 <= safe <= 900.0
    assert safe == 300.0


def test_full_safe_amount_is_safe_on_request_date() -> None:
    timeline = Timeline(
        opening_balance=2000.0,
        entries=[],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 4),
    )
    safe = amount_safe_to_pay(timeline, 500.0, 1000.0)
    assert safe == 1000.0
    assert earliest_full_payment_date(timeline, 500.0, 1000.0) == "2026-01-01"


def test_stopping_projected_spending_does_not_lower_safe_amount() -> None:
    timeline = make_timeline()
    without_change = min(balance for _, balance in replay(timeline, [], []))
    with_change = min(
        balance
        for _, balance in replay(timeline, [], ["stop:flex-1"])
    )
    assert with_change >= without_change


def test_is_safe_applies_payment_on_its_date() -> None:
    timeline = make_timeline()
    assert is_safe(timeline, 500.0, [(date(2026, 1, 1), 300.0)], [])
    assert not is_safe(timeline, 500.0, [(date(2026, 1, 1), 301.0)], [])
