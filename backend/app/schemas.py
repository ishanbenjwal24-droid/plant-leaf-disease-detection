"""API response models."""

from typing import Any

from pydantic import BaseModel, Field


class ClassScore(BaseModel):
    class_name: str
    display_name: str
    score: float = Field(description="Uncalibrated model score; not a probability.")


class PredictionResult(BaseModel):
    model: str
    predicted_class: str
    display_name: str
    class_score: float = Field(description="Uncalibrated model score; not a probability.")
    top_classes: list[ClassScore]
    inference_ms: float
    model_version: str | None = None


class HealthResponse(BaseModel):
    status: str
    models: dict[str, bool]


class ClassesResponse(BaseModel):
    available: bool
    classes: list[str]


class CompareResponse(BaseModel):
    ml: PredictionResult
    cnn: PredictionResult
    total_inference_ms: float


class HistoryItem(BaseModel):
    id: int
    created_at: str
    mode: str
    predictions: Any
    latency_ms: float
