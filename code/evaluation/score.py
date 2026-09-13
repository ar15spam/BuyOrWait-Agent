"""Score Buy or Wait? predictions against the 25 labeled sample requests.

Command line usage:
    python3 -m code.evaluation.score --pred path/to/predictions.csv [--verbose]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
TRUTH_PATH = REPO_ROOT / "dataset" / "sample_requests.csv"

STATUS_LABELS = [
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
]

METHOD_LABELS = [
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
]

SCORED_COLUMNS = [
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
]


@dataclass(frozen=True)
class Failure:
    request_id: str
    column: str
    predicted: object
    expected: object


@dataclass
class ScoreReport:
    total_rows: int
    missing_rows: int
    column_accuracies: dict[str, float]
    amount_exact_rate: float
    amount_within_5_percent_rate: float
    amount_mape: float | None
    affordability_confusion: pd.DataFrame
    payment_method_confusion: pd.DataFrame
    payment_plan_exact_rate: float
    same_dates_different_amounts: int
    earliest_date_exact_rate: float
    earliest_date_presence_agreement_rate: float
    spending_changes_exact_rate: float
    decision_explanation_mean_length: float
    failures: list[Failure] = field(default_factory=list)

    @property
    def headline_score(self) -> float:
        if not self.column_accuracies:
            return 0.0
        return sum(self.column_accuracies.values()) / len(self.column_accuracies)


def score(
    pred_rows: pd.DataFrame | Iterable[Mapping[str, object]],
    truth_rows: pd.DataFrame | Iterable[Mapping[str, object]],
) -> ScoreReport:
    """Return metrics for predictions joined to truth by ``request_id``."""
    predictions = _as_dataframe(pred_rows)
    truth = _as_dataframe(truth_rows)

    if "request_id" not in truth.columns:
        raise ValueError("Truth rows must contain a request_id column.")

    prediction_by_id = _index_predictions(predictions)
    total = len(truth)
    missing_rows = 0
    failures: list[Failure] = []

    correct = {column: 0 for column in SCORED_COLUMNS}
    amount_within_5_percent = 0
    amount_percentage_errors: list[float] = []
    same_dates_different_amounts = 0
    date_presence_agreements = 0
    explanation_lengths: list[int] = []

    status_confusion = pd.DataFrame(
        0,
        index=pd.Index(STATUS_LABELS, name="expected"),
        columns=pd.Index(STATUS_LABELS, name="predicted"),
        dtype=int,
    )
    method_confusion = pd.DataFrame(
        0,
        index=pd.Index(METHOD_LABELS, name="expected"),
        columns=pd.Index(METHOD_LABELS, name="predicted"),
        dtype=int,
    )

    for truth_row in truth.to_dict(orient="records"):
        request_id = _request_id(truth_row.get("request_id"))
        prediction = prediction_by_id.get(request_id)
        has_prediction = prediction is not None

        if not has_prediction:
            missing_rows += 1
            prediction = {}

        amount_pred = _parse_amount(prediction.get("amount_safe_to_pay"))
        amount_truth = _parse_amount(truth_row.get("amount_safe_to_pay"))
        amount_ok = False
        within_5_percent = False

        if has_prediction and amount_pred is not None and amount_truth is not None:
            difference = abs(amount_pred - amount_truth)
            tolerance = max(0.01, abs(amount_truth) * 0.005)
            amount_ok = difference <= tolerance
            if amount_truth == 0:
                within_5_percent = difference <= 0.01
            else:
                within_5_percent = difference <= abs(amount_truth) * 0.05

        if amount_ok:
            correct["amount_safe_to_pay"] += 1
        else:
            failures.append(
                Failure(
                    request_id,
                    "amount_safe_to_pay",
                    _display_value(prediction, "amount_safe_to_pay", has_prediction),
                    truth_row.get("amount_safe_to_pay"),
                )
            )

        if within_5_percent:
            amount_within_5_percent += 1

        if amount_truth is not None and amount_truth != 0:
            if has_prediction and amount_pred is not None:
                amount_percentage_errors.append(abs(amount_pred - amount_truth) / abs(amount_truth))
            else:
                amount_percentage_errors.append(1.0)

        status_pred = _normalize_string(prediction.get("affordability_status"))
        status_truth = _normalize_string(truth_row.get("affordability_status"))
        status_ok = has_prediction and status_pred == status_truth
        if status_ok:
            correct["affordability_status"] += 1
        else:
            failures.append(
                Failure(
                    request_id,
                    "affordability_status",
                    _display_value(prediction, "affordability_status", has_prediction),
                    truth_row.get("affordability_status"),
                )
            )
        if status_truth in STATUS_LABELS and status_pred in STATUS_LABELS:
            status_confusion.loc[status_truth, status_pred] += 1

        method_pred = _normalize_string(prediction.get("recommended_payment_method"))
        method_truth = _normalize_string(truth_row.get("recommended_payment_method"))
        method_ok = has_prediction and method_pred == method_truth
        if method_ok:
            correct["recommended_payment_method"] += 1
        else:
            failures.append(
                Failure(
                    request_id,
                    "recommended_payment_method",
                    _display_value(prediction, "recommended_payment_method", has_prediction),
                    truth_row.get("recommended_payment_method"),
                )
            )
        if method_truth in METHOD_LABELS and method_pred in METHOD_LABELS:
            method_confusion.loc[method_truth, method_pred] += 1

        plan_pred = _normalize_string(prediction.get("payment_plan"))
        plan_truth = _normalize_string(truth_row.get("payment_plan"))
        plan_ok = has_prediction and plan_pred == plan_truth
        if plan_ok:
            correct["payment_plan"] += 1
        else:
            failures.append(
                Failure(
                    request_id,
                    "payment_plan",
                    _display_value(prediction, "payment_plan", has_prediction),
                    truth_row.get("payment_plan"),
                )
            )
            if has_prediction and _same_dates_different_amounts(plan_pred, plan_truth):
                same_dates_different_amounts += 1

        date_pred = _normalize_string(prediction.get("earliest_date_for_full_payment"))
        date_truth = _normalize_string(truth_row.get("earliest_date_for_full_payment"))
        date_ok = has_prediction and date_pred == date_truth
        if date_ok:
            correct["earliest_date_for_full_payment"] += 1
        else:
            failures.append(
                Failure(
                    request_id,
                    "earliest_date_for_full_payment",
                    _display_value(prediction, "earliest_date_for_full_payment", has_prediction),
                    truth_row.get("earliest_date_for_full_payment"),
                )
            )
        if has_prediction and bool(date_pred) == bool(date_truth):
            date_presence_agreements += 1

        changes_pred = _parse_change_set(prediction.get("spending_changes_needed"))
        changes_truth = _parse_change_set(truth_row.get("spending_changes_needed"))
        changes_ok = has_prediction and changes_pred == changes_truth
        if changes_ok:
            correct["spending_changes_needed"] += 1
        else:
            failures.append(
                Failure(
                    request_id,
                    "spending_changes_needed",
                    _display_value(prediction, "spending_changes_needed", has_prediction),
                    truth_row.get("spending_changes_needed"),
                )
            )

        explanation = (
            _normalize_string(prediction.get("decision_explanation"))
            if has_prediction
            else ""
        )
        explanation_lengths.append(len(explanation))

    denominator = total if total else 1
    accuracies = {
        column: correct[column] / denominator
        for column in SCORED_COLUMNS
    }

    return ScoreReport(
        total_rows=total,
        missing_rows=missing_rows,
        column_accuracies=accuracies,
        amount_exact_rate=accuracies["amount_safe_to_pay"],
        amount_within_5_percent_rate=amount_within_5_percent / denominator,
        amount_mape=(
            sum(amount_percentage_errors) / len(amount_percentage_errors)
            if amount_percentage_errors
            else None
        ),
        affordability_confusion=status_confusion,
        payment_method_confusion=method_confusion,
        payment_plan_exact_rate=accuracies["payment_plan"],
        same_dates_different_amounts=same_dates_different_amounts,
        earliest_date_exact_rate=accuracies["earliest_date_for_full_payment"],
        earliest_date_presence_agreement_rate=date_presence_agreements / denominator,
        spending_changes_exact_rate=accuracies["spending_changes_needed"],
        decision_explanation_mean_length=(
            sum(explanation_lengths) / len(explanation_lengths)
            if explanation_lengths
            else 0.0
        ),
        failures=failures,
    )


def _as_dataframe(
    rows: pd.DataFrame | Iterable[Mapping[str, object]],
) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    return pd.DataFrame(list(rows))


def _index_predictions(predictions: pd.DataFrame) -> dict[str, dict[str, object]]:
    if "request_id" not in predictions.columns:
        return {}

    indexed: dict[str, dict[str, object]] = {}
    for row in predictions.to_dict(orient="records"):
        request_id = _request_id(row.get("request_id"))
        if request_id and request_id not in indexed:
            indexed[request_id] = row
    return indexed


def _request_id(value: object) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if not isinstance(result, (pd.Series, pd.DataFrame)) else False
    except (TypeError, ValueError):
        return False


def _normalize_string(value: object) -> str:
    if _is_missing(value):
        return ""
    text = str(value).strip()
    if text.lower() == "none":
        return ""
    return text


def _parse_amount(value: object) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_change_set(value: object) -> set[str]:
    text = _normalize_string(value)
    if not text:
        return set()
    return {
        token.strip()
        for token in text.split("|")
        if token.strip() and token.strip().lower() != "none"
    }


def _parse_plan(plan: str) -> tuple[list[str], list[float]] | None:
    if not plan:
        return None
    dates: list[str] = []
    amounts: list[float] = []
    for token in plan.split("|"):
        try:
            date_value, amount_value = token.strip().split(":", maxsplit=1)
            dates.append(date_value.strip())
            amounts.append(float(amount_value.strip()))
        except (TypeError, ValueError):
            return None
    return dates, amounts


def _same_dates_different_amounts(predicted: str, expected: str) -> bool:
    pred_plan = _parse_plan(predicted)
    truth_plan = _parse_plan(expected)
    if pred_plan is None or truth_plan is None:
        return False
    pred_dates, pred_amounts = pred_plan
    truth_dates, truth_amounts = truth_plan
    return pred_dates == truth_dates and pred_amounts != truth_amounts


def _display_value(
    prediction: Mapping[str, object],
    column: str,
    has_prediction: bool,
) -> object:
    if not has_prediction:
        return "<missing row>"
    value = prediction.get(column)
    return "<empty>" if _is_missing(value) or str(value).strip() == "" else value


def _percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def _print_report(report: ScoreReport, verbose: bool) -> None:
    print(f"Truth rows: {report.total_rows}")
    print(f"Missing prediction rows: {report.missing_rows}")
    print(
        "amount_safe_to_pay: "
        f"exact={_percentage(report.amount_exact_rate)}, "
        f"within_5%={_percentage(report.amount_within_5_percent_rate)}, "
        f"MAPE={_percentage(report.amount_mape) if report.amount_mape is not None else 'n/a'}"
    )
    print(
        "affordability_status: "
        f"accuracy={_percentage(report.column_accuracies['affordability_status'])}"
    )
    print(report.affordability_confusion.to_string())
    print(
        "recommended_payment_method: "
        f"accuracy={_percentage(report.column_accuracies['recommended_payment_method'])}"
    )
    print(report.payment_method_confusion.to_string())
    print(
        "payment_plan: "
        f"exact={_percentage(report.payment_plan_exact_rate)}, "
        f"same_dates_different_amounts={report.same_dates_different_amounts}"
    )
    print(
        "earliest_date_for_full_payment: "
        f"exact={_percentage(report.earliest_date_exact_rate)}, "
        "empty_nonempty_agreement="
        f"{_percentage(report.earliest_date_presence_agreement_rate)}"
    )
    print(
        "spending_changes_needed: "
        f"set_accuracy={_percentage(report.spending_changes_exact_rate)}"
    )
    print(
        "decision_explanation: "
        f"mean_length={report.decision_explanation_mean_length:.1f}"
    )

    if verbose and report.failures:
        print("\nFailures:")
        _print_failures(report.failures)

    print(f"SCORE: {report.headline_score * 100:.1f}%")


def _print_failures(failures: Sequence[Failure]) -> None:
    widths = (14, 33, 30, 30)
    header = _format_failure_row(
        "request_id",
        "column",
        "predicted",
        "expected",
        widths,
    )
    print(header)
    print("-" * len(header))
    for failure in failures:
        print(
            _format_failure_row(
                failure.request_id,
                failure.column,
                failure.predicted,
                failure.expected,
                widths,
            )
        )


def _format_failure_row(
    request_id: object,
    column: object,
    predicted: object,
    expected: object,
    widths: tuple[int, int, int, int],
) -> str:
    values = (request_id, column, predicted, expected)
    cells = [
        _truncate(str(value), width).ljust(width)
        for value, width in zip(values, widths)
    ]
    return " | ".join(cells)


def _truncate(value: str, width: int) -> str:
    clean = " ".join(value.split())
    if len(clean) <= width:
        return clean
    if width <= 3:
        return clean[:width]
    return clean[: width - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path, help="Prediction CSV to score")
    parser.add_argument("--verbose", action="store_true", help="Show incorrect fields")
    args = parser.parse_args()

    predictions = pd.read_csv(args.pred)
    truth = pd.read_csv(TRUTH_PATH)
    report = score(predictions, truth)
    _print_report(report, args.verbose)


if __name__ == "__main__":
    main()
