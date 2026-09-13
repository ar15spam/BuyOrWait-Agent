"""One token-usage ledger and one report across every model the run uses.

The submission requires a single ``evaluation/usage_report.md`` covering every
provider and model, with per-model and overall totals. The perception, message
and explanation stages each recorded usage in their own shape, so this module
is the common sink they all write to.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER_PATH = REPO_ROOT / ".cache" / "usage.json"
USAGE_REPORT_PATH = REPO_ROOT / "code" / "evaluation" / "usage_report.md"

# USD per 1M tokens. Override per model as needed; unknown models cost 0.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "claude-sonnet-4-5": (3.00, 15.00),
}


@dataclass(frozen=True)
class Call:
    stage: str          # "images" | "messages" | "explanations"
    reference: str      # image_id, message_id, or request_id
    provider: str
    model: str
    input_tokens: int
    output_tokens: int


class Ledger:
    """Append-only record of every model call made during a run."""

    def __init__(self, path: str | Path = DEFAULT_LEDGER_PATH) -> None:
        self.path = Path(path)
        self.calls: list[Call] = self._load()

    def add(self, call: Call) -> None:
        self.calls.append(call)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(item) for item in self.calls], indent=2) + "\n",
            encoding="utf-8",
        )

    def _load(self) -> list[Call]:
        if not self.path.exists():
            return []
        try:
            return [Call(**item) for item in json.loads(self.path.read_text(encoding="utf-8"))]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return []


def rates_for(model: str) -> tuple[float, float]:
    """Look up pricing for a model id, tolerating dated suffixes.

    The API echoes a dated id such as "gpt-4o-mini-2024-07-18", which does not
    match a base-name key, and an unpriced model silently reported a cost of
    zero. Longest matching prefix wins so a more specific key still takes
    precedence over a shorter one.
    """
    if model in PRICING:
        return PRICING[model]
    matches = [key for key in PRICING if model.startswith(key)]
    if matches:
        return PRICING[max(matches, key=len)]
    return (0.0, 0.0)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    rate_in, rate_out = rates_for(model)
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000


def deduplicate(calls: list[Call]) -> list[Call]:
    """Keep the most recent call per artifact.

    The ledger persists across runs so the report can cover the cached
    extractions this output depends on, not just the calls a warm rerun happens
    to make. Without deduplication a rerun would add a second explanation call
    for every request and the report would overstate the work by a whole pass.
    """
    latest: dict[tuple[str, str], Call] = {}
    for call in calls:
        latest[(call.stage, call.reference)] = call
    return list(latest.values())


def write_report(
    calls: list[Call],
    requests_in_run: int,
    output_path: str | Path = USAGE_REPORT_PATH,
    cache_hits: dict[str, int] | None = None,
) -> None:
    """Write per-model and overall totals. Contains no keys or endpoints."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    calls = deduplicate(calls)

    by_model: dict[tuple[str, str], list[Call]] = {}
    for call in calls:
        by_model.setdefault((call.provider, call.model), []).append(call)

    total_in = sum(call.input_tokens for call in calls)
    total_out = sum(call.output_tokens for call in calls)
    total_cost = sum(cost_usd(call.model, call.input_tokens, call.output_tokens) for call in calls)
    denominator = requests_in_run or 1

    by_stage = {}
    for call in calls:
        by_stage[call.stage] = by_stage.get(call.stage, 0) + 1
    stage_summary = ", ".join(f"{stage} {count}" for stage, count in sorted(by_stage.items()))

    lines = [
        "# Token usage and cost report",
        "",
        f"Requests in the final run: {requests_in_run}",
        f"Total model calls: {len(calls)}" + (f" ({stage_summary})" if stage_summary else ""),
        "",
        "Counts cover every model call this output depends on, including calls whose",
        "results are served from the on-disk cache on a rerun, counted once per artifact.",
        "Aggregate usage only. No API keys, credentials, or endpoint configuration appear here.",
        "",
        "## Per model",
        "",
        "| Provider | Model | Stage(s) | Calls | Input tokens | Output tokens | Total tokens | Est. cost (USD) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for (provider, model), group in sorted(by_model.items()):
        stages = ", ".join(sorted({item.stage for item in group}))
        group_in = sum(item.input_tokens for item in group)
        group_out = sum(item.output_tokens for item in group)
        group_cost = sum(cost_usd(item.model, item.input_tokens, item.output_tokens) for item in group)
        lines.append(
            f"| {provider} | {model} | {stages} | {len(group)} | {group_in:,} | {group_out:,} "
            f"| {group_in + group_out:,} | {group_cost:.4f} |"
        )
    if not by_model:
        lines.append("| — | — | — | 0 | 0 | 0 | 0 | 0.0000 |")

    lines += [
        "",
        "## Overall",
        "",
        f"- Input tokens: {total_in:,}",
        f"- Output tokens: {total_out:,}",
        f"- Total tokens: {total_in + total_out:,}",
        f"- Average tokens per request: {(total_in + total_out) / denominator:,.1f}",
        f"- Estimated total cost: USD {total_cost:.4f}",
        f"- Estimated cost per request: USD {total_cost / denominator:.6f}",
        "",
    ]
    unpriced = sorted({call.model for call in calls if rates_for(call.model) == (0.0, 0.0)})
    if unpriced:
        lines += ["", "Models with no published rate in PRICING (cost shown as 0): "
                  + ", ".join(unpriced), ""]

    if cache_hits:
        lines += [
            "## Cache",
            "",
            "Extraction is cached per artifact, so a re-run makes no further calls "
            "for artifacts already read. Counts below are cache hits in this run.",
            "",
        ]
        lines += [f"- {stage}: {count}" for stage, count in sorted(cache_hits.items())]
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
