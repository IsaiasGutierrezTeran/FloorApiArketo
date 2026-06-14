"""FastAPI dependencies shared by the routers."""

from __future__ import annotations

from fastapi import Request

from app.config import Settings, get_settings
from app.services.detection_service import DetectionService


def get_detection_service(request: Request) -> DetectionService:
    """Return the singleton detection service created during startup."""
    return request.app.state.detection_service


def settings_dependency() -> Settings:
    """Expose cached settings as a dependency (kept thin for overrides in tests)."""
    return get_settings()
