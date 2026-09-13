"""Write validated decision rows in the exact submission format."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .validate import OUTPUT_COLUMNS


def format_amount(amount: float) -> str:
    """Render a money value the way the labelled data does: no trailing ``.0``."""
    if abs(amount - round(amount)) < 1e-9:
        return str(int(round(amount)))
    return f"{amount:.2f}".rstrip("0").rstrip(".")


def write_output(rows: Sequence[Mapping[str, object]], output_path: str | Path) -> None:
    normalised = []
    for row in rows:
        item = dict(row)
        value = item.get("amount_safe_to_pay")
        if isinstance(value, (int, float)):
            item["amount_safe_to_pay"] = format_amount(float(value))
        normalised.append(item)

    frame = pd.DataFrame(normalised, columns=OUTPUT_COLUMNS)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, columns=OUTPUT_COLUMNS)
