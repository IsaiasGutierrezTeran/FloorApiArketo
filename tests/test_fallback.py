"""Tests for the primary -> OpenCV fallback policy in DetectionService."""

from __future__ import annotations

import numpy as np

from app.services.detection_service import DetectionService
from app.services.detector_base import WALL, Detection, DetectorBase
from app.services.opencv_detector import OpenCVDetector


class _EmptyPrimary(DetectorBase):
    """A stand-in primary detector that always returns nothing."""

    @property
    def model_name(self) -> str:
        return "stub-primary"

    @property
    def is_loaded(self) -> bool:
        return True

    def detect(self, image: np.ndarray) -> list[Detection]:
        return []


class _GoodPrimary(DetectorBase):
    """A stand-in primary detector that returns confident walls."""

    @property
    def model_name(self) -> str:
        return "stub-primary"

    @property
    def is_loaded(self) -> bool:
        return True

    def detect(self, image: np.ndarray) -> list[Detection]:
        return [Detection(WALL, (0, 0, 100, 4), 0.95)]


def test_fallback_used_when_primary_finds_no_walls(
    floorplan_array: np.ndarray,
) -> None:
    service = DetectionService(primary=_EmptyPrimary(), fallback=OpenCVDetector())
    result = service.detect(floorplan_array, confidence_threshold=0.5)

    assert result.fallback_used is True
    assert result.model_name == "opencv-classic"
    walls = [d for d in result.detections if d.label == WALL]
    assert len(walls) >= 4


def test_no_fallback_when_primary_succeeds(floorplan_array: np.ndarray) -> None:
    service = DetectionService(primary=_GoodPrimary(), fallback=OpenCVDetector())
    result = service.detect(floorplan_array, confidence_threshold=0.5)

    assert result.fallback_used is False
    assert result.model_name == "stub-primary"


def test_no_fallback_when_disabled(floorplan_array: np.ndarray) -> None:
    # No fallback configured -> empty primary result is returned as-is.
    service = DetectionService(primary=_EmptyPrimary(), fallback=None)
    result = service.detect(floorplan_array, confidence_threshold=0.5)

    assert result.fallback_used is False
    assert result.detections == []
