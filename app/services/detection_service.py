"""Detection orchestration: primary detector + optional OpenCV fallback.

Keeps the fallback policy out of both the web layer and the individual
detectors. The web layer asks this service for detections and receives, along
with them, which model actually produced them and whether the fallback kicked in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import fmean

from app.exceptions import InferenceError
from app.services.detector_base import WALL, Detection, DetectorBase
from app.services.opencv_detector import OpenCVDetector, OpenCVParams

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of a detection run."""

    detections: list[Detection]
    model_name: str
    fallback_used: bool


class DetectionService:
    """Run the primary detector and fall back to OpenCV when configured."""

    def __init__(
        self,
        *,
        primary: DetectorBase,
        fallback: OpenCVDetector | None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def model_name(self) -> str:
        return self._primary.model_name

    @property
    def is_loaded(self) -> bool:
        return self._primary.is_loaded

    @property
    def fallback_enabled(self) -> bool:
        return self._fallback is not None

    def detect(
        self,
        image,
        *,
        confidence_threshold: float,
        opencv_params: OpenCVParams | None = None,
    ) -> DetectionResult:
        """Detect with the primary detector, applying the fallback policy.

        The fallback (OpenCV) is used when it is enabled and the primary either
        raised, produced no walls, or produced walls whose mean confidence is
        below ``confidence_threshold``.

        Args:
            image: Decoded RGB image array.
            confidence_threshold: Threshold used for the fallback decision.
            opencv_params: Per-request overrides for the OpenCV detector (used
                whether OpenCV is the primary or the fallback).
        """
        primary = self._resolve_primary(opencv_params)

        try:
            detections = primary.detect(image)
            needs_fallback, reason = self._needs_fallback(detections, confidence_threshold)
        except Exception as exc:  # noqa: BLE001 - any detector failure -> fallback
            logger.warning("Primary detector '%s' failed: %s", primary.model_name, exc)
            detections = []
            needs_fallback, reason = True, "primary raised an exception"
            if not self.fallback_enabled:
                raise InferenceError(f"Detector failed: {exc}") from exc

        if needs_fallback and self.fallback_enabled:
            logger.info(
                "Falling back to OpenCV detector (%s).", reason
            )
            fallback = self._resolve_fallback(opencv_params)
            try:
                detections = fallback.detect(image)
            except Exception as exc:  # noqa: BLE001
                raise InferenceError(
                    f"Primary and fallback detectors both failed: {exc}"
                ) from exc
            return DetectionResult(detections, fallback.model_name, fallback_used=True)

        return DetectionResult(detections, primary.model_name, fallback_used=False)

    # ------------------------------------------------------------------ #

    def _resolve_primary(self, opencv_params: OpenCVParams | None) -> DetectorBase:
        """Rebuild the primary with overrides when it is an OpenCV detector."""
        if opencv_params is not None and isinstance(self._primary, OpenCVDetector):
            return OpenCVDetector(opencv_params)
        return self._primary

    def _resolve_fallback(self, opencv_params: OpenCVParams | None) -> OpenCVDetector:
        assert self._fallback is not None  # guarded by fallback_enabled
        if opencv_params is not None:
            return OpenCVDetector(opencv_params)
        return self._fallback

    @staticmethod
    def _needs_fallback(
        detections: list[Detection], confidence_threshold: float
    ) -> tuple[bool, str]:
        """Decide whether the fallback should replace the primary result."""
        walls = [d for d in detections if d.label == WALL]
        if not walls:
            return True, "primary found 0 walls"
        mean_confidence = fmean(d.confidence for d in walls)
        if mean_confidence < confidence_threshold:
            return True, (
                f"mean wall confidence {mean_confidence:.2f} < "
                f"threshold {confidence_threshold:.2f}"
            )
        return False, ""
