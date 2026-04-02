"""
Detection endpoints.

POST /predict          — upload a single driver image, get risk result
POST /predict/batch    — offline batch evaluation from (class_id, confidence) list
POST /session/reset    — reset the sustained-distraction timer
GET  /classes          — list all classes and their base risk scores
"""

from __future__ import annotations

import io

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from PIL import Image

from api.schemas.detection import (
    BatchDetectionRequest,
    DetectionResponse,
    TripSummaryResponse,
)
from src.config import MAX_IMAGE_SIZE_MB
from src.utils.risk_calculator import class_base_risks, trip_summary

router = APIRouter(prefix="/api/v1", tags=["detection"])

# Pipeline is injected at startup — see main.py
_pipeline = None


def set_pipeline(pipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def _get_pipeline():
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline not yet initialised.",
        )
    return _pipeline


@router.post("/predict", response_model=DetectionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Classify a single inside-cabin driver image and return a risk assessment.

    Accepts JPEG / PNG images up to MAX_IMAGE_SIZE_MB.
    The risk calculator is stateful — submit frames sequentially to benefit
    from sustained-distraction tracking.
    """
    _validate_image_upload(file)
    raw_bytes = await file.read()
    _validate_size(raw_bytes)

    pl = _get_pipeline()
    try:
        result = pl.predict_bytes(raw_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference failed: {exc}",
        )
    return result


@router.post("/predict/batch", response_model=TripSummaryResponse)
async def predict_batch(request: BatchDetectionRequest):
    """
    Offline batch evaluation from pre-computed (class_id, confidence) pairs.

    Useful for post-processing recorded trips without re-running the model.
    Returns a trip-level summary including distraction percentages and
    time-bucketed risk distribution.
    """
    pl = _get_pipeline()
    pl.reset_session()

    results = pl.risk_calc.evaluate_batch(
        request.predictions,
        frame_interval_seconds=request.frame_interval_seconds,
    )
    summary = trip_summary(results)
    pl.reset_session()
    return summary


@router.post("/session/reset", status_code=status.HTTP_204_NO_CONTENT)
async def reset_session():
    """
    Reset the sustained-distraction timer.

    Call this when a new driver session begins (engine start, driver change).
    """
    _get_pipeline().reset_session()


@router.get("/classes")
async def list_classes():
    """Return all 10 driver behaviour classes with their base risk scores."""
    return class_base_risks()


# ── Validation helpers ────────────────────────────────────────────────────────

def _validate_image_upload(file: UploadFile) -> None:
    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {file.content_type}. "
                   f"Allowed: {allowed}",
        )


def _validate_size(raw_bytes: bytes) -> None:
    max_bytes = MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image too large. Max size: {MAX_IMAGE_SIZE_MB} MB.",
        )
