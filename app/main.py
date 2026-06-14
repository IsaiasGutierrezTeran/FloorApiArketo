"""FastAPI application factory: wiring, CORS, error handling and startup.

Run locally with::

    uvicorn app.main:app --reload

or ``python -m app.main`` (uses HOST/PORT from settings). Interactive docs are
served at ``/docs``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import routes_detect, routes_health
from app.config import Settings, get_settings
from app.exceptions import ApiError
from app.schemas.errors import ErrorResponse
from app.services import create_detection_service

logger = logging.getLogger("app")


def _configure_logging(level: str) -> None:
    """Configure structured-ish logging once, at startup."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the detection service once and attach it to the app state."""
    settings = get_settings()
    _configure_logging(settings.log_level)
    logger.info("Starting Floorplan API v%s (detector=%s)", __version__, settings.detector)
    app.state.detection_service = create_detection_service(settings)
    yield
    logger.info("Shutting down Floorplan API")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory (also used by the test suite)."""
    settings = settings or get_settings()

    app = FastAPI(
        title="FloorPlan-to-3D Detection API",
        description=(
            "Detects walls, doors and windows in a 2D floor plan and returns a "
            "normalized JSON ready for 3D extrusion. Part of the 'Plan Risk 3D' "
            "system."
        ),
        version=settings.api_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_error_handlers(app)

    app.include_router(routes_health.router)
    app.include_router(routes_detect.router)
    return app


def _register_error_handlers(app: FastAPI) -> None:
    """Map exceptions to uniform ``{error, detail}`` JSON responses."""

    @app.exception_handler(ApiError)
    async def _api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=exc.error, detail=exc.detail).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="validation_error", detail=str(exc.errors())
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error", detail="An unexpected error occurred."
            ).model_dump(),
        )


app = create_app()


if __name__ == "__main__":
    import uvicorn

    _settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=_settings.host,
        port=_settings.port,
        reload=False,
    )
