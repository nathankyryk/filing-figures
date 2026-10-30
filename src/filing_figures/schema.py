"""The answer format: six fields, each with a value in base units, the scale the filing states,
the period end and the fiscal year. Every part may be null (the figure is not in the text).

`JSON_SCHEMA` is written out by hand (no $ref, every key required, no extra keys) so the same
schema works for vLLM's structured outputs, OpenRouter and the API providers behind it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from filing_figures.labels import FIELD_NAMES

SCALES = ("units", "thousands", "millions", "billions")
SCALE_POWER = {"units": 0, "thousands": 3, "millions": 6, "billions": 9}


class FieldAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float | None
    scale: Literal["units", "thousands", "millions", "billions"] | None
    period_end: str | None
    fiscal_year: int | None


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revenue: FieldAnswer
    net_income: FieldAnswer
    eps_diluted: FieldAnswer
    total_assets: FieldAnswer
    shares_outstanding: FieldAnswer
    operating_cash_flow: FieldAnswer


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


_FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "value": _nullable({"type": "number"}),
        "scale": _nullable({"type": "string", "enum": list(SCALES)}),
        "period_end": _nullable({"type": "string"}),
        "fiscal_year": _nullable({"type": "integer"}),
    },
    "required": ["value", "scale", "period_end", "fiscal_year"],
    "additionalProperties": False,
}

JSON_SCHEMA = {
    "type": "object",
    "properties": {name: _FIELD_SCHEMA for name in FIELD_NAMES},
    "required": list(FIELD_NAMES),
    "additionalProperties": False,
}


def response_format() -> dict:
    """The OpenAI-style `response_format` for constrained decoding."""
    return {"type": "json_schema", "json_schema": {"name": "filing_figures", "strict": True, "schema": JSON_SCHEMA}}
