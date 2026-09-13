"""Generate grounded explanations after the deterministic plan is final."""

from __future__ import annotations

import hashlib

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .plans import Decision


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USAGE_PATH = REPO_ROOT / ".cache" / "explanations" / "usage.json"
USAGE_REPORT_PATH = REPO_ROOT / "code" / "evaluation" / "usage_report.md"

# These are used only to estimate totals in the report. The report itself does
# not expose rates, credentials, endpoints, or any other configuration.
MODEL_COST_USD_PER_MILLION = {
    "gpt-4o-mini": (0.15, 0.60),
}


@dataclass(frozen=True)
class ExplanationInput:
    request_id: str
    currency: str
    amount_safe_to_pay: float
    status: str
    method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    opening_balance: float
    minimum_balance: float
    binding_constraint_date: str
    next_income_date: str
    desired_completion_date: str = ""

    @classmethod
    def from_decision(cls, decision: Decision, currency: str) -> "ExplanationInput":
        return cls(
            request_id=decision.request_id,
            currency=currency,
            amount_safe_to_pay=decision.amount_safe_to_pay,
            status=decision.affordability_status,
            method=decision.recommended_payment_method,
            payment_plan=decision.payment_plan,
            earliest_date_for_full_payment=decision.earliest_date_for_full_payment,
            spending_changes_needed=decision.spending_changes_needed,
            opening_balance=decision.opening_balance,
            minimum_balance=decision.minimum_balance,
            binding_constraint_date=decision.binding_constraint_date,
            next_income_date=decision.next_income_date,
            desired_completion_date=getattr(decision, "desired_completion_date", ""),
        )


@dataclass(frozen=True)
class ExplanationUsage:
    request_id: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    estimated_cost_usd: float


class ExplanationClient(Protocol):
    def complete(self, prompt: str, request_id: str) -> tuple[str, ExplanationUsage]: ...


class UsageLedger:
    """Persistent running list consumed by ``write_usage_report``."""

    def __init__(self, path: str | Path = DEFAULT_USAGE_PATH) -> None:
        self.path = Path(path)
        self.records: list[ExplanationUsage] = self._load()

    def add(self, record: ExplanationUsage) -> None:
        self.records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(item) for item in self.records], indent=2) + "\n",
            encoding="utf-8",
        )

    def _load(self) -> list[ExplanationUsage]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return [ExplanationUsage(**item) for item in raw]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return []


@dataclass(frozen=True)
class OpenAICompatibleExplanationClient:
    """Small Chat Completions client; the API key is never persisted."""

    api_key: str
    model: str = "gpt-4o-mini"
    provider: str = "openai"
    endpoint: str = "https://api.openai.com/v1/chat/completions"

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleExplanationClient":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for explanations")
        return cls(
            api_key=api_key,
            model=os.environ.get("EXPLANATION_MODEL", "gpt-4o-mini"),
            provider=os.environ.get("EXPLANATION_PROVIDER", "openai"),
            endpoint=os.environ.get(
                "EXPLANATION_API_ENDPOINT",
                "https://api.openai.com/v1/chat/completions",
            ),
        )

    def complete(self, prompt: str, request_id: str) -> tuple[str, ExplanationUsage]:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Write the decision explanation only. Use the supplied plan "
                            "and facts as authoritative. Return one or two concise "
                            "sentences, with no bullets, labels, quotes, or markdown."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"explanation request failed with HTTP {error.code} {error.reason}: "
                               f"{_error_body(error)}") from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"explanation request failed: {type(error).__name__}: {error}") from error

        text = _response_text(payload)
        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        input_rate, output_rate = MODEL_COST_USD_PER_MILLION.get(
            str(payload.get("model", self.model)),
            (0.0, 0.0),
        )
        record = ExplanationUsage(
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=str(payload.get("model", self.model)),
            provider=self.provider,
            estimated_cost_usd=(input_tokens * input_rate + output_tokens * output_rate) / 1_000_000,
        )
        return text, record


EXPLANATION_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "explanations"


def _cache_key(prompt: str) -> str:
    """Key on the prompt, so changed decisions get fresh text and reruns are free."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:32]


def explain_decision(
    decision: Decision,
    currency: str,
    client: ExplanationClient | None = None,
    usage: UsageLedger | None = None,
    use_cache: bool = True,
) -> str:
    """Return one explanation for one final decision, from cache or one call.

    The cache is keyed on the prompt rather than the request_id: a decision that
    changed produces a different prompt and is re-explained, while an unchanged
    decision costs nothing on a rerun. Without this, every rerun of a 250-row
    dataset repaid the full explanation bill.
    """
    facts = ExplanationInput.from_decision(decision, currency)
    prompt = _build_prompt(facts)
    cache_path = EXPLANATION_CACHE_DIR / f"{_cache_key(prompt)}.txt"

    if use_cache and cache_path.is_file():
        cached = cache_path.read_text(encoding="utf-8").strip()
        if cached:
            try:
                _assert_explanation(cached)
                return cached
            except AssertionError:
                pass  # bad cached text: fall through and ask again

    explanation_client = client or OpenAICompatibleExplanationClient.from_environment()
    text, usage_record = explanation_client.complete(prompt, decision.request_id)
    (usage or UsageLedger()).add(usage_record)
    text = trim_to_two_sentences(text)
    _assert_explanation(text)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
    return text


def replace_explanation(
    row: Mapping[str, object],
    explanation: str,
) -> dict[str, object]:
    """Replace only the explanation column and assert all other fields are fixed."""
    _assert_explanation(explanation)
    result = dict(row)
    original = dict(row)
    result["decision_explanation"] = explanation
    for key, value in original.items():
        if key != "decision_explanation":
            assert result.get(key) == value, f"explanation changed column: {key}"
    return result


def write_usage_report(
    records: Sequence[ExplanationUsage],
    output_path: str | Path = USAGE_REPORT_PATH,
    expected_requests: int | None = None,
) -> None:
    """Write aggregate and per-model token/cost totals without secrets."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    calls = len(records)
    input_tokens = sum(item.input_tokens for item in records)
    output_tokens = sum(item.output_tokens for item in records)
    cost = sum(item.estimated_cost_usd for item in records)
    denominator = expected_requests if expected_requests is not None else calls
    average_tokens = (input_tokens + output_tokens) / denominator if denominator else 0.0
    average_cost = cost / denominator if denominator else 0.0

    lines = [
        "# Explanation token usage report",
        "",
        "This report contains aggregate usage only; no keys, credentials, or endpoint configuration are included.",
        "",
        f"Requests in run: {denominator}",
        f"Recorded model calls: {calls}",
        "",
        "| Provider | Model | Calls | Input tokens | Output tokens | Total tokens | Estimated cost (USD) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for provider, model in sorted({(item.provider, item.model) for item in records}):
        group = [item for item in records if item.provider == provider and item.model == model]
        group_input = sum(item.input_tokens for item in group)
        group_output = sum(item.output_tokens for item in group)
        group_cost = sum(item.estimated_cost_usd for item in group)
        lines.append(
            f"| {provider} | {model} | {len(group)} | {group_input} | {group_output} | "
            f"{group_input + group_output} | ${group_cost:.6f} |"
        )
    lines.extend(
        [
            f"| **Overall** | — | **{calls}** | **{input_tokens}** | **{output_tokens}** | "
            f"**{input_tokens + output_tokens}** | **${cost:.6f}** |",
            "",
            f"Average tokens per request: {average_tokens:.2f}",
            f"Average estimated cost per request: ${average_cost:.6f}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_prompt(facts: ExplanationInput) -> str:
    return (
        "Explain this already-decided financial plan in the register of the provided "
        "examples. Do not reconsider or change the plan.\n\n"
        f"request_id: {facts.request_id}\n"
        f"currency: {facts.currency}\n"
        f"status: {facts.status}\n"
        f"method: {facts.method}\n"
        f"payment_plan: {facts.payment_plan}\n"
        f"amount_safe_to_pay: {facts.amount_safe_to_pay:.2f}\n"
        f"earliest_date_for_full_payment: {facts.earliest_date_for_full_payment}\n"
        f"spending_changes_needed: {facts.spending_changes_needed}\n"
        f"opening_balance: {facts.opening_balance:.2f}\n"
        f"minimum_balance: {facts.minimum_balance:.2f}\n"
        f"binding_constraint_date: {facts.binding_constraint_date}\n"
        f"next_income_date: {facts.next_income_date or 'none'}\n"
        f"desired_completion_date: {facts.desired_completion_date or 'none'}\n\n"
        "Rules for the wording:\n"
        "- Any date you name must be one of the dates given above, written as "
        "'5 March 2024'. Never invent or infer a date.\n"
        "- When the status is not_affordable, name desired_completion_date, "
        "never binding_constraint_date.\n"
        "- When a payment is recommended, name the first date in payment_plan.\n"
        "- Always state the minimum_balance that is being protected.\n\n"
        "Examples of the desired register:\n"
        "Pay EUR 620.40 today. This leaves at least EUR 800 available.\n"
        "Use 3 installments of INR 68,432, starting 12 September 2024. "
        "This leaves at least INR 93,000 available.\n"
        "Do not make this payment by 12 January 2026. None of the available options "
        "keeps the ZAR 13,100 minimum protected."
    )


def trim_to_two_sentences(text: str) -> str:
    """Normalise whitespace and keep at most the first two sentences.

    A model that answers in three sentences has still done the work; discarding
    it kept a templated line and paid for the call anyway.
    """
    normalized = " ".join(str(text).split())
    sentences = _sentences(normalized)
    if len(sentences) > 2:
        return " ".join(sentences[:2])
    return normalized


def _assert_explanation(text: str) -> None:
    assert isinstance(text, str), "explanation must be text"
    normalized = " ".join(text.split())
    assert normalized == text.strip() and normalized, "explanation must not be blank"
    assert len(_sentences(normalized)) in {1, 2}, "explanation must be one or two sentences"


def _sentences(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=[.!?])\s+", text) if part]


def _response_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("explanation response has no choices")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError("explanation response has no text content")
    return content


def _error_body(error: Any) -> str:
    """Return a short slice of an HTTP error body to make failures diagnosable."""
    try:
        return error.read().decode("utf-8", "replace")[:300]
    except Exception:  # noqa: BLE001
        return ""
