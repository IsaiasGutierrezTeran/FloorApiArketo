"""The `/detect` endpoint: image in, normalized 3D floor plan out.

The router only orchestrates I/O and dependency wiring; image handling lives in
`utils.image_io`, detection in the `DetectionService`, and normalization in the
`FloorPlanBuilder`.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.api.deps import get_detection_service, settings_dependency
from app.config import Settings
from app.schemas.detection import FloorPlan3D
from app.schemas.errors import ErrorResponse
from app.services import opencv_params_from_settings
from app.services.detection_service import DetectionService
from app.services.floorplan_builder import BuildOptions, FloorPlanBuilder
from app.services.opencv_detector import OpenCVParams
from app.utils.image_io import load_and_validate_image

logger = logging.getLogger(__name__)

router = APIRouter(tags=["detection"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid image"},
    413: {"model": ErrorResponse, "description": "Image too large"},
    422: {"model": ErrorResponse, "description": "Validation error"},
    500: {"model": ErrorResponse, "description": "Inference error"},
}


@router.post("/detect", response_model=FloorPlan3D, responses=_ERROR_RESPONSES)
async def detect(
    file: UploadFile = File(..., description="Floor plan image (PNG/JPG)."),
    # --- Inference / scaling parameters (override config defaults) ----------
    confidence_threshold: float | None = Query(None, ge=0.0, le=1.0),
    wall_height: float | None = Query(None, gt=0.0),
    default_wall_thickness: float | None = Query(None, gt=0.0),
    pixels_per_meter: float | None = Query(
        None, gt=0.0, description="If omitted, geometry is normalized to 0..1."
    ),
    # --- OpenCV detector parameters (only relevant for the opencv detector) --
    min_wall_length_px: int | None = Query(None, gt=0),
    hough_threshold: int | None = Query(None, gt=0),
    hough_min_line_length: int | None = Query(None, gt=0),
    hough_max_line_gap: int | None = Query(None, ge=0),
    merge_distance_px: float | None = Query(None, ge=0.0),
    service: DetectionService = Depends(get_detection_service),
    settings: Settings = Depends(settings_dependency),
) -> FloorPlan3D:
    """Detect architectural elements and return a normalized 3D floor plan."""
    data = await file.read()
    image, width, height = load_and_validate_image(
        data,
        max_size_mb=settings.max_image_size_mb,
        max_dimension=settings.max_image_dimension,
    )

    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else settings.confidence_threshold
    )
    opencv_overrides = _resolve_opencv_params(
        settings,
        min_wall_length_px=min_wall_length_px,
        hough_threshold=hough_threshold,
        hough_min_line_length=hough_min_line_length,
        hough_max_line_gap=hough_max_line_gap,
        merge_distance_px=merge_distance_px,
    )

    started = time.perf_counter()
    result = service.detect(
        image, confidence_threshold=threshold, opencv_params=opencv_overrides
    )
    # Keep only detections at/above the threshold before normalizing.
    detections = [d for d in result.detections if d.confidence >= threshold]

    options = BuildOptions(
        confidence_threshold=threshold,
        wall_height=wall_height if wall_height is not None else settings.wall_height,
        default_wall_thickness=(
            default_wall_thickness
            if default_wall_thickness is not None
            else settings.default_wall_thickness
        ),
        pixels_per_meter=pixels_per_meter,
        door_height=settings.door_height,
        window_height=settings.window_height,
        window_sill_height=settings.window_sill_height,
    )

    builder = FloorPlanBuilder(api_version=settings.api_version)
    plan = builder.build(
        detections,
        image_width=width,
        image_height=height,
        options=options,
        model_name=result.model_name,
        fallback_used=result.fallback_used,
    )
    plan.meta.processing_ms = int((time.perf_counter() - started) * 1000)

    logger.info(
        "Detected %d walls, %d doors, %d windows (model=%s, fallback=%s)",
        len(plan.walls),
        len(plan.doors),
        len(plan.windows),
        plan.meta.model,
        plan.meta.fallback_used,
    )
    return plan


def _resolve_opencv_params(
    settings: Settings,
    *,
    min_wall_length_px: int | None,
    hough_threshold: int | None,
    hough_min_line_length: int | None,
    hough_max_line_gap: int | None,
    merge_distance_px: float | None,
) -> OpenCVParams | None:
    """Merge per-request OpenCV overrides onto config defaults, or None."""
    overrides = {
        "min_wall_length_px": min_wall_length_px,
        "hough_threshold": hough_threshold,
        "hough_min_line_length": hough_min_line_length,
        "hough_max_line_gap": hough_max_line_gap,
        "merge_distance_px": merge_distance_px,
    }
    if all(value is None for value in overrides.values()):
        return None
    return opencv_params_from_settings(settings).with_overrides(**overrides)
