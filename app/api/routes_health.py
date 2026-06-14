"""Health endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_detection_service
from app.services.detection_service import DetectionService

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str
    model_loaded: bool


@router.get("/health", response_model=HealthResponse)
def health(
    service: DetectionService = Depends(get_detection_service),
) -> HealthResponse:
    """Report service status and whether the primary detector is ready."""
    return HealthResponse(status="ok", model_loaded=service.is_loaded)
