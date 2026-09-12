"""Inspect the first-stage ledger for one request.

Run: ``python3 code/main.py --request request_01``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ingest import load_dataset, rows_for
from ledger import forecast_for_request


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", default="request_01", help="Request ID to inspect")
    args = parser.parse_args()
    data = load_dataset(ROOT / "dataset")
    request = rows_for(data.requests, "request_id", args.request)
    if request.empty:
        request = rows_for(data.sample_requests, "request_id", args.request)
    if request.empty:
        raise SystemExit(f"Unknown request ID: {args.request}")
    row = request.iloc[0]
    profile = rows_for(data.profiles, "user_id", row.user_id).iloc[0]
    forecast = forecast_for_request(data.events, row.user_id, row.request_date)
    print(f"Request: {row.request_id} ({row.request_date}), amount {row.requested_amount:g}")
    print(f"Opening balance: {profile.current_available_balance:g}")
    print(f"Minimum balance: {profile.minimum_balance_to_keep:g}")
    print("\nForecast cash flows:")
    for flow in forecast:
        print(f"{flow.when.isoformat()}  {flow.amount:>12.2f}  {flow.source:<20} {flow.category}")


if __name__ == "__main__":
    main()
