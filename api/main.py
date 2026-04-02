"""
FastAPI application entry-point.

Startup sequence
----------------
1. Load the ONNX model (falls back to PyTorch if ONNX not found).
2. Inject the pipeline into the detection router.
3. Mount health + detection routers.

Start locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import detection, health
from src.config import API_HOST, API_PORT, MODELS_DIR, ONNX_MODEL_NAME, PT_MODEL_NAME

# ── Pipeline singleton ────────────────────────────────────────────────────────
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; clean up on shutdown."""
    global pipeline
    from src.pipeline.detection_pipeline import DetectionPipeline

    onnx_path = MODELS_DIR / ONNX_MODEL_NAME
    pt_path   = MODELS_DIR / PT_MODEL_NAME

    if onnx_path.exists():
        pipeline = DetectionPipeline(backend="onnx", model_path=onnx_path)
        print(f"[API] Loaded ONNX model from {onnx_path}")
    elif pt_path.exists():
        pipeline = DetectionPipeline(backend="pytorch", model_path=pt_path)
        print(f"[API] Loaded PyTorch model from {pt_path}")
    else:
        # Allow server to start without a model (useful during development)
        print(
            "[API] WARNING: No model weights found. "
            f"Place model at {onnx_path} or {pt_path}."
        )

    detection.set_pipeline(pipeline)
    yield
    pipeline = None


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Distracted Driver Detection API",
    description=(
        "Inside-cabin driver behaviour classification with real-time "
        "risk assessment. Powered by MobileNetV2 + ONNX Runtime."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(detection.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
