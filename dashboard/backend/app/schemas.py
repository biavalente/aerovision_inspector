from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .defect_labels import canonical_defect_label


Decision = Literal["approved_visual", "rejected_visual", "escalated_ndt", "recapture_requested"]


class BoundingBoxInput(BaseModel):
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)
    width: float = Field(gt=0, le=100)
    height: float = Field(gt=0, le=100)
    label: str = Field(default="defect", max_length=80)
    confidence: float | None = Field(default=None, ge=0, le=1)
    source: Literal["human", "model"] = "human"

    @field_validator("width")
    @classmethod
    def width_is_valid(cls, value: float, info):
        if info.data.get("x", 0) + value > 100.001:
            raise ValueError("A caixa excede os limites horizontais da imagem")
        return value

    @field_validator("height")
    @classmethod
    def height_is_valid(cls, value: float, info):
        if info.data.get("y", 0) + value > 100.001:
            raise ValueError("A caixa excede os limites verticais da imagem")
        return value

    @field_validator("label")
    @classmethod
    def normalise_label(cls, value: str) -> str:
        """Keep reviewer annotations usable by the later local fine-tuning job."""
        clean = value.strip()
        if not clean:
            return "unknown"
        return canonical_defect_label(clean) or clean.lower()


class ReviewPayload(BaseModel):
    decision: Decision
    reason: str = Field(default="", max_length=1000)
    actor: str = Field(default="reviewer", max_length=120)
    boxes: list[BoundingBoxInput] = Field(default_factory=list)


class BatchDecisionPayload(BaseModel):
    inspection_ids: list[int] = Field(min_length=1, max_length=500)
    decision: Decision
    reason: str = Field(default="", max_length=1000)
    actor: str = Field(default="reviewer", max_length=120)


class CandidatePayload(BaseModel):
    actor: str = Field(default="model_manager", max_length=120)

