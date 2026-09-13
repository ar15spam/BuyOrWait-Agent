"""Apply validated message claims to financial events with an audit trail."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

import pandas as pd

from .extract.messages import MessageClaim, MessageRejection, MessageResult


@dataclass(frozen=True)
class AuditEntry:
    message_id: str
    kind: str
    target_event_id: str
    outcome: str
    reason: str
    asserted_at: str = ""


@dataclass(frozen=True)
class ReconciliationResult:
    events: pd.DataFrame
    audit: tuple[AuditEntry, ...]


def reconcile_events(
    events: pd.DataFrame,
    messages: Mapping[str, MessageResult | Sequence[MessageClaim]],
) -> ReconciliationResult:
    """Apply claims in precedence order and retain every decision in ``audit``.

    The order is explicit claim first, then the newer assertion, then settled
    over an estimate, then a conservative cash-flow interpretation. Claims that
    cannot safely be applied are rejected rather than silently changing data.
    """
    if "event_id" not in events.columns:
        raise ValueError("events must contain event_id")
    result = events.copy(deep=True)
    audit: list[AuditEntry] = []
    known_ids = {str(value) for value in result["event_id"]}

    claims: list[tuple[str, MessageClaim]] = []
    for message_id, message in messages.items():
        if isinstance(message, MessageResult):
            for rejection in message.rejected:
                audit.append(_rejected_message(rejection))
            claim_items = message.claims
        else:
            claim_items = tuple(message)
        for claim in claim_items:
            claims.append((str(message_id), claim))

    grouped: dict[str, list[tuple[str, MessageClaim]]] = {}
    for message_id, claim in claims:
        if claim.target_event_id not in known_ids:
            audit.append(AuditEntry(message_id, claim.kind, claim.target_event_id, "rejected", "unknown target_event_id", claim.asserted_at))
            continue
        grouped.setdefault(claim.target_event_id, []).append((message_id, claim))

    # One pass to locate every target. The previous per-claim lookup scanned and
    # re-stringified all 25,343 event rows for each of ~160 claims.
    index_by_event_id = {
        str(event_id): row_index
        for row_index, event_id in zip(result.index, result["event_id"])
    }

    for target_event_id, target_claims in grouped.items():
        row_index = index_by_event_id[target_event_id]
        ordered = sorted(target_claims, key=_precedence_key)
        state: dict[str, object] = {
            "cancelled": False,
            "settled": str(result.at[row_index, "status"]) == "settled" if "status" in result else False,
            "amount_changed": False,
            "date_changed": False,
        }
        for message_id, claim in ordered:
            applied, reason = _apply_claim(result, row_index, claim, state)
            audit.append(
                AuditEntry(
                    message_id,
                    claim.kind,
                    claim.target_event_id,
                    "applied" if applied else "rejected",
                    reason,
                    claim.asserted_at,
                )
            )

    return ReconciliationResult(result, tuple(audit))


def _apply_claim(
    events: pd.DataFrame,
    row_index: object,
    claim: MessageClaim,
    state: dict[str, object],
) -> tuple[bool, str]:
    if claim.kind == "none":
        return True, "no event change"
    if state["cancelled"]:
        return False, "target already cancelled or marked duplicate"
    if claim.kind == "cancel":
        events.at[row_index, "status"] = "cancelled"
        state["cancelled"] = True
        return True, "explicit cancellation applied"
    if claim.kind == "duplicate_of":
        events.at[row_index, "status"] = "cancelled"
        state["cancelled"] = True
        return True, "duplicate event excluded from cash flow"
    if claim.kind == "settle":
        if state["settled"]:
            return False, "event was already settled"
        events.at[row_index, "status"] = "settled"
        state["settled"] = True
        return True, "explicit settlement applied"
    if claim.kind == "confirm":
        if str(events.at[row_index, "status"]) == "settled":
            return False, "settled event takes precedence over confirmation"
        events.at[row_index, "status"] = "scheduled"
        return True, "confirmation applied as scheduled cash"
    if claim.kind == "amend_amount":
        if state["amount_changed"]:
            return False, "newer amount amendment takes precedence"
        events.at[row_index, "amount"] = claim.new_amount
        state["amount_changed"] = True
        return True, "amount amendment applied"
    if claim.kind == "delay":
        if state["date_changed"]:
            return False, "newer date amendment takes precedence"
        if "settlement_date" not in events.columns:
            return False, "event list has no settlement_date column"
        events.at[row_index, "settlement_date"] = claim.new_date
        state["date_changed"] = True
        return True, "payment date amendment applied"
    return False, "unsupported claim kind"


def _precedence_key(item: tuple[str, MessageClaim]) -> tuple[int, float, int]:
    message_id, claim = item
    explicit_priority = {
        "cancel": 0,
        "duplicate_of": 0,
        "settle": 0,
        "amend_amount": 0,
        "delay": 0,
        "confirm": 1,
        "none": 2,
    }[claim.kind]
    try:
        asserted = datetime.fromisoformat(claim.asserted_at.replace("Z", "+00:00")).timestamp()
    except ValueError:
        asserted = float("-inf")
    numeric_id = int(message_id.rsplit("_", 1)[-1]) if message_id.rsplit("_", 1)[-1].isdigit() else 0
    return explicit_priority, -asserted, -numeric_id


def _rejected_message(rejection: MessageRejection) -> AuditEntry:
    return AuditEntry(
        rejection.message_id,
        "invalid",
        rejection.raw_claim,
        "rejected",
        rejection.reason,
    )
