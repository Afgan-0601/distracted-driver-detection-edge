"""Pydantic schemas for the detection API."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class RiskDetail(BaseModel):
    class_key: str
    label: str
    confidence: float
    base_risk: float
    sustained_seconds: float
    sustained_multiplier: float
    weighted_risk: float
    composite_risk: float
    risk_level: str
    alert: bool
    is_distracted: bool


class DetectionResponse(BaseModel):
    class_id: int
    class_key: str
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    low_confidence: bool
    all_scores: list[float]
    risk: RiskDetail


class BatchDetectionRequest(BaseModel):
    """For offline batch evaluation via JSON payload."""
    predictions: list[tuple[int, float]] = Field(
        ...,
        description="List of (class_id, confidence) tuples",
    )
    frame_interval_seconds: float = Field(
        0.1,
        ge=0.01,
        description="Simulated time between frames (seconds)",
    )


class TripSummaryResponse(BaseModel):
    total_frames: int
    distracted_frames: int
    distracted_pct: float
    alert_frames: int
    max_composite_risk: float
    mean_composite_risk: float
    time_per_level: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_backend: str
    version: str = "1.0.0"
