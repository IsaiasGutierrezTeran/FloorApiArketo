"""Service layer: detectors, the normalization builder and their wiring."""

from __future__ import annotations

import logging

from app.config import Settings
from app.services.detection_service import DetectionService
from app.services.detector_base import DetectorBase
from app.services.maskrcnn_detector import MaskRCNNDetector
from app.services.mock_detector import MockDetector
from app.services.opencv_detector import OpenCVDetector, OpenCVParams

logger = logging.getLogger(__name__)


def opencv_params_from_settings(settings: Settings) -> OpenCVParams:
    """Build OpenCV defaults from configuration."""
    return OpenCVParams(
        min_wall_length_px=settings.opencv_min_wall_length_px,
        hough_threshold=settings.opencv_hough_threshold,
        hough_min_line_length=settings.opencv_hough_min_line_length,
        hough_max_line_gap=settings.opencv_hough_max_line_gap,
        merge_distance_px=settings.opencv_merge_distance_px,
    )


def _build_primary(settings: Settings) -> DetectorBase:
    """Instantiate the configured primary detector."""
    if settings.detector == "opencv":
        return OpenCVDetector(opencv_params_from_settings(settings))

    if settings.detector == "maskrcnn":
        detector = MaskRCNNDetector(
            api_url=settings.legacy_api_url,
            model_name=settings.model_name,
            timeout=settings.legacy_api_timeout,
            default_confidence=settings.legacy_default_confidence,
        )
        # If the legacy model is unreachable and we have no OpenCV safety net,
        # degrade to the mock so the API still serves something usable.
        if not detector.is_loaded and not settings.fallback_to_opencv:
            logger.warning(
                "Mask R-CNN service unavailable and FALLBACK_TO_OPENCV is off; "
                "using MockDetector so the API stays responsive."
            )
            return MockDetector()
        return detector

    return MockDetector()


def create_detection_service(settings: Settings) -> DetectionService:
    """Wire the primary detector and the optional OpenCV fallback together."""
    primary = _build_primary(settings)

    fallback: OpenCVDetector | None = None
    if settings.fallback_to_opencv and not isinstance(primary, OpenCVDetector):
        fallback = OpenCVDetector(opencv_params_from_settings(settings))

    logger.info(
        "Detection service ready: primary=%s (loaded=%s), fallback=%s",
        primary.model_name,
        primary.is_loaded,
        "opencv-classic" if fallback else "none",
    )
    return DetectionService(primary=primary, fallback=fallback)


__all__ = [
    "DetectionService",
    "create_detection_service",
    "opencv_params_from_settings",
]
