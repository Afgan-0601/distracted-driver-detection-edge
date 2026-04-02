"""Health-check router."""

from fastapi import APIRouter
from api.schemas.detection import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness probe — used by Docker health checks and load balancers."""
    from api.main import pipeline  # lazy import to avoid circular reference
    return HealthResponse(
        status="ok",
        model_backend=pipeline.backend,
    )
