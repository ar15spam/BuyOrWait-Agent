"""Smoke checks for the ingest and context layer."""

import code.build_context as bc


def test_normalize_future_events_has_signed_amounts():
    context = bc.build_context("request_01")
    events = context.normalize_future_events()
    assert "cash_amount" in events.columns
    credits = events.loc[events["direction"] == "credit", "cash_amount"]
    debits = events.loc[events["direction"] == "debit", "cash_amount"]
    assert (credits >= 0).all()
    assert (debits <= 0).all()


if __name__ == "__main__":
    test_normalize_future_events_has_signed_amounts()
    print("ok")
