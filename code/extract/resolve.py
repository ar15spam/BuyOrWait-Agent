"""Turn validated image extractions into event amounts the ledger can use.

A blank ``amount`` in ``financial_events.csv`` is not zero: the amount lives in
a linked PNG. This module is the only bridge between the perception layer and
the deterministic pipeline, and it applies the same discard-do-not-repair rule
as the extractor — anything that does not line up exactly is left unresolved so
the event stays out of the forecast rather than entering it with a wrong value.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import ImageExtraction

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"


def resolved_amounts(
    extractions: dict[str, ImageExtraction],
    events_path: str | Path = DATASET_DIR / "financial_events.csv",
    images_path: str | Path = DATASET_DIR / "images.csv",
) -> dict[str, float]:
    """Map ``event_id`` to the amount read from its linked image.

    An extraction is dropped when the image is not linked to a blank-amount
    event, or when the currency it reports disagrees with the currency the
    event row already declares — a disagreement means one of the two is wrong,
    and guessing which would put an unsupported number into the forecast.
    """
    events = pd.read_csv(events_path)
    images = pd.read_csv(images_path)

    blank_ids = set(events.loc[events["amount"].isna(), "event_id"].astype(str))
    currency_by_event = dict(
        zip(events["event_id"].astype(str), events["currency"].astype(str))
    )
    event_by_image = dict(
        zip(images["image_id"].astype(str), images["related_event_id"].astype(str))
    )

    resolved: dict[str, float] = {}
    for image_id, extraction in extractions.items():
        event_id = event_by_image.get(image_id, "")
        if event_id not in blank_ids:
            continue
        if extraction.currency.strip().upper() != currency_by_event[event_id].upper():
            continue
        if extraction.amount <= 0:
            continue
        resolved[event_id] = float(extraction.amount)

    return resolved
