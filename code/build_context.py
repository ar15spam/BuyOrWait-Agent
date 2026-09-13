"""Build and inspect the financial context for a request.

This is an intentionally small first stage. It loads the sample data, separates
past from future events, converts event amounts into the user's home currency,
and projects the balance through the known future events.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"


profiles = pd.read_csv(DATASET_DIR / "financial_profiles.csv")
events = pd.read_csv(DATASET_DIR / "financial_events.csv")
samples = pd.read_csv(DATASET_DIR / "sample_requests.csv")
exchange_rates = pd.read_csv(DATASET_DIR / "exchange_rates.csv")

samples["request_date"] = pd.to_datetime(samples["request_date"])
events["event_date"] = pd.to_datetime(events["event_date"])
events["settlement_date"] = pd.to_datetime(events["settlement_date"])
events["cash_date"] = events["settlement_date"].fillna(events["event_date"])
exchange_rates["rate_date"] = pd.to_datetime(exchange_rates["rate_date"])


def build_rate_lookup(
    rate_table: pd.DataFrame,
) -> dict[tuple[pd.Timestamp, str, str], float]:
    """Create an exact lookup: (date, from_currency, to_currency) -> rate."""
    lookup: dict[tuple[pd.Timestamp, str, str], float] = {}

    for row in rate_table.itertuples(index=False):
        key = (
            cast(pd.Timestamp, row.rate_date),
            str(row.from_currency),
            str(row.to_currency),
        )
        lookup[key] = cast(float, row.rate)

    return lookup


RATES = build_rate_lookup(exchange_rates)


@dataclass
class RequestContext:
    home_currency: str
    request: pd.Series
    profile: pd.Series
    past_events: pd.DataFrame
    future_events: pd.DataFrame

    def normalize_future_events(self) -> pd.DataFrame:
        """Return valid future events with home-currency cash amounts."""
        return normalize_future_events(
            self.future_events,
            self.home_currency,
            RATES,
        )


def build_context(request_id: str) -> RequestContext:
    request = get_request(request_id)
    user_id = get_user_id(request)
    profile = get_user_profile(user_id)
    past, future = get_past_and_future_events(
        get_user_events(user_id),
        request["request_date"],
    )

    return RequestContext(
        home_currency=str(profile["home_currency"]),
        request=request,
        profile=profile,
        past_events=past,
        future_events=future,
    )


def get_request(request_id: str) -> pd.Series:
    matches = samples.loc[samples["request_id"] == request_id]
    if matches.empty:
        raise ValueError(f"Could not find request: {request_id}")
    return matches.iloc[0]


def get_user_id(request: pd.Series) -> str:
    return str(request["user_id"])


def get_request_date(request_id: str) -> pd.Timestamp:
    return get_request(request_id)["request_date"]


def get_user_profile(user_id: str) -> pd.Series:
    matches = profiles.loc[profiles["user_id"] == user_id]
    if matches.empty:
        raise ValueError(f"Could not find profile for user: {user_id}")
    return matches.iloc[0]


def get_user_events(user_id: str) -> pd.DataFrame:
    user_events = events.loc[events["user_id"] == user_id].copy()
    if user_events.empty:
        raise ValueError(f"Could not find events for user: {user_id}")
    return user_events


def get_possible_statuses() -> list[str]:
    return sorted(events["status"].dropna().unique().tolist())


def get_past_and_future_events(
    user_events: pd.DataFrame,
    request_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split history and forecast inputs without putting request-day events in both."""
    past = user_events.loc[user_events["cash_date"] < request_date].copy()
    future = user_events.loc[user_events["cash_date"] >= request_date].copy()

    return (
        past.sort_values("cash_date"),
        future.sort_values("cash_date"),
    )


def normalize_event_amounts(
    event_table: pd.DataFrame,
    home_currency: str,
    rate_lookup: dict[tuple[pd.Timestamp, str, str], float],
) -> pd.DataFrame:
    """Add ``amount_home`` while preserving blank amounts as missing."""
    normalized = event_table.copy()
    converted: list = []

    for row in normalized.itertuples(index=False):
        if pd.isna(row.amount):
            converted.append(None)
            continue

        if str(row.currency) == home_currency:
            converted.append(row.amount)
            continue

        rate_date = cast(pd.Timestamp, row.cash_date)
        key = (rate_date, str(row.currency), home_currency)
        if key not in rate_lookup:
            raise KeyError(f"Missing exchange rate for {key}")

        converted.append(cast(float, row.amount) * rate_lookup[key])

    normalized["amount_home"] = converted
    return normalized


def normalize_future_events(
    future_events: pd.DataFrame,
    home_currency: str,
    rate_lookup: dict[tuple[pd.Timestamp, str, str], float],
) -> pd.DataFrame:
    """Filter unusable events and add signed home-currency cash amounts."""
    clean_events = normalize_event_amounts(
        future_events,
        home_currency,
        rate_lookup,
    )

    excluded_statuses = {"cancelled", "failed", "unrealized"}
    valid = (
        ~clean_events["status"].isin(excluded_statuses)
        & clean_events["amount_home"].notna()
        & ~(
            (clean_events["status"] == "pending")
            & (clean_events["direction"] == "credit")
        )
    )

    clean_events = clean_events.loc[valid].copy()
    clean_events["cash_amount"] = clean_events["amount_home"].where(
        clean_events["direction"] == "credit",
        -clean_events["amount_home"],
    )

    return clean_events.sort_values("cash_date")


def project_balances(context: RequestContext) -> pd.DataFrame:
    """Project balance through known future events, before any new purchase."""
    projected = context.normalize_future_events()
    opening_balance = context.profile["current_available_balance"]
    projected["balance_after"] = (
        opening_balance + projected["cash_amount"].cumsum()
    )
    return projected

def is_safe_to_pay(context):
    payment = context.request["requested_amount"]
    minimum_balance = context.profile["minimum_balance_to_keep"]
    opening_balance = context.profile["current_available_balance"]

    events = context.normalize_future_events()

    balance_after_payment = opening_balance - payment

    balances = (
        balance_after_payment
        + events["cash_amount"].cumsum()
    )

    lowest_future_balance = balances.min()

    lowest_balance = min(
        balance_after_payment,
        lowest_future_balance,
    )

    return lowest_balance >= minimum_balance
    