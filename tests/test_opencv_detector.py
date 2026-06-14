"""Tests for the classical OpenCV wall detector."""

from __future__ import annotations

import numpy as np

from app.schemas.detection import FloorPlan3D
from app.services.detector_base import WALL
from app.services.floorplan_builder import BuildOptions, FloorPlanBuilder
from app.services.opencv_detector import OpenCVDetector


def test_opencv_detects_room_walls(floorplan_array: np.ndarray) -> None:
    detector = OpenCVDetector()
    detections = detector.detect(floorplan_array)

    walls = [d for d in detections if d.label == WALL]
    # A rectangle (4 sides) + 1 divider should yield at least 4 wall segments.
    assert len(walls) >= 4
    # Wall detections carry an explicit pixel segment.
    assert all(d.segment is not None for d in walls)
    assert all(0.0 <= d.confidence <= 1.0 for d in walls)


def test_opencv_output_builds_valid_floorplan(floorplan_array: np.ndarray) -> None:
    detector = OpenCVDetector()
    detections = detector.detect(floorplan_array)

    options = BuildOptions(
        confidence_threshold=0.5,
        wall_height=2.7,
        default_wall_thickness=0.15,
        pixels_per_meter=50.0,
        door_height=2.1,
        window_height=1.1,
        window_sill_height=0.9,
    )
    builder = FloorPlanBuilder(api_version="1.0")
    plan = builder.build(
        detections,
        image_width=floorplan_array.shape[1],
        image_height=floorplan_array.shape[0],
        options=options,
        model_name=detector.model_name,
    )

    # Validate against the same public contract as every other detector.
    FloorPlan3D.model_validate(plan.model_dump())
    assert plan.meta.model == "opencv-classic"
    assert len(plan.walls) >= 4
    # Doors/windows are intentionally not detected by the classical pipeline.
    assert plan.doors == []
    assert plan.windows == []


def test_opencv_no_lines_returns_empty() -> None:
    blank = np.full((300, 300, 3), 255, dtype=np.uint8)
    detector = OpenCVDetector()
    assert detector.detect(blank) == []
