"""Strict schemas for facts transcribed from supporting images."""

from __future__ import annotations

from datetime import date as Date
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictFloat, field_validator


class ImageExtraction(BaseModel):
    """One validated, non-decision fact read from an image."""

    model_config = ConfigDict(extra="forbid", strict=True)

    image_id: str
    document_type: Literal[
        "payroll_letter",
        "statement",
        "bill",
        "receipt",
        "other",
    ]
    amount: StrictFloat
    currency: str
    date: str
    confidence: Literal["high", "medium", "low"]
    evidence: str

    @field_validator("image_id", "currency", "evidence")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("date")
    @classmethod
    def iso_date(cls, value: str) -> str:
        if len(value) != 10 or value[4] != "-" or value[7] != "-":
            raise ValueError("date must use YYYY-MM-DD")
        try:
            Date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("date must use YYYY-MM-DD") from error
        return value

    @field_validator("amount")
    @classmethod
    def finite_amount(cls, value: float) -> float:
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("amount must be finite")
        return value
