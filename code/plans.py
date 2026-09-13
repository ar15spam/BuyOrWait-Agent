"""Pure candidate-plan enumeration, filtering, ranking, and decision output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from itertools import combinations, product
from typing import Protocol, Sequence

import pandas as pd

from .ledger import Series, Timeline
from .safety import earliest_full_payment_date, is_safe, replay


@dataclass(frozen=True)
class PlanCandidate:
    method: str
    payments: tuple[tuple[date, float], ...]
    changes: tuple[str, ...]
    total_amount_paid: float
    payment_option_id: str = ""

    @property
    def first_payment_date(self) -> date:
        return min(when for when, _ in self.payments)

    @property
    def completes_by_desired_completion_date(self) -> bool:
        return self._desired_completion_date is not None

    _desired_completion_date: date | None = None


@dataclass(frozen=True)
class Decision:
    """All output columns plus facts for a later explanation stage."""

    request_id: str
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str
    opening_balance: float
    minimum_balance: float
    binding_constraint_date: str
    next_income_date: str
    desired_completion_date: str = ""


class PlanContextLike(Protocol):
    request: pd.Series
    profile: pd.Series


def choose_plan(
    context: PlanContextLike,
    timeline: Timeline,
    series: Sequence[Series],
    payment_options: pd.DataFrame,
    amount_safe_to_pay: float | None = None,
    earliest_full_payment_date_value: str | None = None,
) -> Decision:
    """Choose the highest-ranked safe plan for one request.

    ``context`` must expose ``request`` and ``profile`` pandas Series. The
    timeline already contains the real and projected cash movements; this
    function only enumerates and evaluates possible request payments.
    """
    request = context.request
    profile = context.profile
    request_id = str(request["request_id"])
    request_date = _as_date(request["request_date"])
    desired_date = _as_date(request["desired_completion_date"])
    requested_amount = float(request["requested_amount"])
    minimum_balance = float(profile["minimum_balance_to_keep"])
    accepted_methods = _pipe_set(profile.get("payment_methods_user_will_consider"))

    if amount_safe_to_pay is None:
        from .safety import amount_safe_to_pay as calculate_safe_amount

        amount_safe_to_pay = calculate_safe_amount(
            timeline,
            minimum_balance,
            requested_amount,
        )
    if earliest_full_payment_date_value is None:
        earliest_full_payment_date_value = earliest_full_payment_date(
            timeline,
            minimum_balance,
            requested_amount,
        )

    candidates = _payment_candidates(
        request=request,
        profile=profile,
        request_date=request_date,
        desired_date=desired_date,
        requested_amount=requested_amount,
        amount_safe_to_pay=amount_safe_to_pay,
        earliest_date=earliest_full_payment_date_value,
        payment_options=payment_options,
        accepted_methods=accepted_methods,
    )
    def survivors_for(change_sets: Sequence[tuple[str, ...]]) -> list[PlanCandidate]:
        found: list[PlanCandidate] = []
        for candidate in candidates:
            for changes in change_sets:
                if not is_safe(timeline, minimum_balance, candidate.payments, changes):
                    continue
                found.append(
                    PlanCandidate(
                        method=candidate.method,
                        payments=candidate.payments,
                        changes=changes,
                        total_amount_paid=candidate.total_amount_paid,
                        payment_option_id=candidate.payment_option_id,
                        _desired_completion_date=(
                            desired_date
                            if candidate.payments[-1][0] <= desired_date
                            else None
                        ),
                    )
                )
        return found

    # The ranking key puts "no spending changes" ahead of every plan that needs
    # them, so if an unchanged plan already completes by the deadline no change
    # set can outrank it and the combinatorial search is dead work.
    survivors = survivors_for([()])
    if not any(candidate.completes_by_desired_completion_date for candidate in survivors):
        survivors.extend(survivors_for(_change_sets(series, profile)[1:]))

    winner = min(survivors, key=lambda candidate: _ranking_key(candidate)) if survivors else None
    return _decision_from_winner(
        context=context,
        timeline=timeline,
        winner=winner,
        amount_safe_to_pay=amount_safe_to_pay,
        earliest_date=earliest_full_payment_date_value,
        requested_amount=requested_amount,
    )


def _payment_candidates(
    *,
    request: pd.Series,
    profile: pd.Series,
    request_date: date,
    desired_date: date,
    requested_amount: float,
    amount_safe_to_pay: float,
    earliest_date: str,
    payment_options: pd.DataFrame,
    accepted_methods: set[str],
) -> list[PlanCandidate]:
    candidates: list[PlanCandidate] = []

    if "full_payment" in accepted_methods:
        candidates.append(
            PlanCandidate(
                method="full_payment",
                payments=((request_date, requested_amount),),
                changes=(),
                total_amount_paid=requested_amount,
            )
        )

    options = payment_options
    if "request_id" in options.columns:
        options = options.loc[options["request_id"] == request["request_id"]]

    max_months = profile.get("max_installment_months")
    for _, option in options.iterrows():
        method = str(option["payment_method"]).strip()
        if method not in accepted_methods:
            continue
        if method == "installments":
            if _is_missing(max_months) or int(option["number_of_payments"]) > int(max_months):
                continue
        payments = _option_payments(option)
        candidates.append(
            PlanCandidate(
                method=method,
                payments=payments,
                changes=(),
                total_amount_paid=float(option["total_payable_amount"]),
                payment_option_id=str(option["payment_option_id"]),
            )
        )

    if (
        _as_bool(request["allows_partial_payment"])
        and "partial_payment" in accepted_methods
        and 0 < amount_safe_to_pay < requested_amount
        and earliest_date
        and _as_date(earliest_date) <= desired_date
    ):
        remainder = round(requested_amount - amount_safe_to_pay, 2)
        candidates.append(
            PlanCandidate(
                method="partial_payment",
                payments=(
                    (request_date, amount_safe_to_pay),
                    (_as_date(earliest_date), remainder),
                ),
                changes=(),
                total_amount_paid=requested_amount,
            )
        )

    if (
        "full_payment" in accepted_methods
        and earliest_date
        and _as_date(earliest_date) > request_date
    ):
        candidates.append(
            PlanCandidate(
                method="wait",
                payments=((_as_date(earliest_date), requested_amount),),
                changes=(),
                total_amount_paid=requested_amount,
            )
        )

    return candidates


def _option_payments(option: pd.Series) -> tuple[tuple[date, float], ...]:
    count = int(option["number_of_payments"])
    first_date = _as_date(option["first_payment_date"])
    frequency = 0 if _is_missing(option["payment_frequency_days"]) else int(option["payment_frequency_days"])
    amount = float(option["payment_amount"])
    return tuple(
        (
            first_date + timedelta(days=index * frequency),
            amount,
        )
        for index in range(count)
    )


def _change_sets(series: Sequence[Series], profile: pd.Series) -> list[tuple[str, ...]]:
    protected = _pipe_set(profile.get("expense_categories_to_protect"))
    stoppable_categories = _pipe_set(profile.get("expense_categories_user_is_willing_to_stop"))
    reducible_categories = _pipe_set(profile.get("expense_categories_user_is_willing_to_reduce"))
    actions_by_series: list[tuple[str, list[str]]] = []

    for recurrence in sorted(series, key=lambda item: item.last_event_id):
        if recurrence.direction != "debit" or recurrence.category in protected:
            continue

        actions: list[str] = []
        if (
            recurrence.flexibility in {"stoppable", "reducible_or_stoppable"}
            and recurrence.category in stoppable_categories
        ):
            actions.append(f"stop:{recurrence.last_event_id}")
        if (
            recurrence.flexibility in {"reducible", "reducible_or_stoppable"}
            and recurrence.category in reducible_categories
            and recurrence.minimum_allowed_amount is not None
        ):
            actions.append(
                f"reduce_to:{recurrence.last_event_id}:"
                f"{_format_amount(recurrence.minimum_allowed_amount)}"
            )
        if actions:
            actions_by_series.append((recurrence.last_event_id, actions))

    change_sets: list[tuple[str, ...]] = [()]
    for size in range(1, min(3, len(actions_by_series)) + 1):
        for chosen in combinations(actions_by_series, size):
            for actions in product(*(item[1] for item in chosen)):
                change_sets.append(tuple(sorted(actions)))
    return change_sets


def _ranking_key(candidate: PlanCandidate) -> tuple[bool, bool, float, date, int, str]:
    return (
        not candidate.completes_by_desired_completion_date,
        bool(candidate.changes),
        candidate.total_amount_paid,
        candidate.first_payment_date,
        len(candidate.payments),
        candidate.payment_option_id,
    )


def _decision_from_winner(
    *,
    context: PlanContextLike,
    timeline: Timeline,
    winner: PlanCandidate | None,
    amount_safe_to_pay: float,
    earliest_date: str,
    requested_amount: float,
) -> Decision:
    request = context.request
    profile = context.profile
    request_id = str(request["request_id"])
    opening_balance = float(profile["current_available_balance"])
    minimum_balance = float(profile["minimum_balance_to_keep"])
    binding_date = _binding_constraint_date(timeline)
    next_income = _next_income_date(timeline)

    if winner is None:
        status = "not_affordable"
        method = "not_recommended"
        plan = "none"
        changes = "none"
        explanation = "No eligible payment plan stays above the minimum balance."
    else:
        method = winner.method
        plan = _format_plan(winner.payments)
        changes = "|".join(winner.changes) if winner.changes else "none"
        if method == "wait":
            status = "affordable_later"
        elif method == "full_payment" and not winner.changes:
            status = "affordable_now"
        else:
            status = "affordable_with_plan"
        explanation = (
            f"Use {method}; projected balance is constrained near {binding_date}."
        )

    return Decision(
        request_id=request_id,
        amount_safe_to_pay=round(amount_safe_to_pay, 2),
        affordability_status=status,
        recommended_payment_method=method,
        payment_plan=plan,
        earliest_date_for_full_payment=earliest_date,
        spending_changes_needed=changes,
        decision_explanation=explanation,
        opening_balance=opening_balance,
        minimum_balance=minimum_balance,
        binding_constraint_date=binding_date,
        next_income_date=next_income,
        desired_completion_date=_as_date(request["desired_completion_date"]).isoformat(),
    )


def _binding_constraint_date(timeline: Timeline) -> str:
    balances = replay(timeline, [], [])
    if not balances:
        return timeline.start_date.isoformat()
    return min(balances, key=lambda item: item[1])[0].isoformat()


def _next_income_date(timeline: Timeline) -> str:
    income = [entry.date for entry in timeline.entries if entry.delta > 0]
    return min(income).isoformat() if income else ""


def _format_plan(payments: Sequence[tuple[date, float]]) -> str:
    return "|".join(
        f"{when.isoformat()}:{_format_amount(amount)}"
        for when, amount in sorted(payments, key=lambda item: item[0])
    )


def _format_amount(amount: float) -> str:
    if abs(amount - round(amount)) < 1e-9:
        return str(int(round(amount)))
    return f"{amount:.2f}"


def _pipe_set(value: object) -> set[str]:
    if _is_missing(value):
        return set()
    return {part.strip() for part in str(value).split("|") if part.strip()}


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _as_date(value: date | pd.Timestamp | str) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
