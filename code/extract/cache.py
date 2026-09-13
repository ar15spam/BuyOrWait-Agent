"""Disk cache and token-usage ledger for image extraction calls."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .schemas import ImageExtraction


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "images"


@dataclass(frozen=True)
class UsageRecord:
    image_id: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str


class UsageLedger:
    """Append-only call metadata that a usage-report step can consume."""

    def __init__(self, path: str | Path = DEFAULT_CACHE_DIR / "usage.json") -> None:
        self.path = Path(path)
        self.records: list[UsageRecord] = self._load()

    def add(self, record: UsageRecord) -> None:
        self.records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(item) for item in self.records], indent=2) + "\n",
            encoding="utf-8",
        )

    def _load(self) -> list[UsageRecord]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return [UsageRecord(**item) for item in raw]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return []


class ImageCache:
    """Cache raw JSON responses while exposing only validated extractions."""

    def __init__(self, directory: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.directory = Path(directory)

    def path_for(self, image_id: str) -> Path:
        _validate_image_id(image_id)
        return self.directory / f"{image_id}.json"

    def read(self, image_id: str) -> tuple[bool, ImageExtraction | None]:
        """Return ``(cache_hit, extraction)``; invalid cached data stays unresolved."""
        path = self.path_for(image_id)
        if not path.exists():
            return False, None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return True, ImageExtraction.model_validate(payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return True, None

    def write_response(self, image_id: str, response: Any) -> None:
        """Persist the model's raw JSON response, including invalid responses."""
        path = self.path_for(image_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(response, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


def _validate_image_id(image_id: str) -> None:
    if not image_id or Path(image_id).name != image_id or "/" in image_id or "\\" in image_id:
        raise ValueError(f"invalid image_id: {image_id!r}")
