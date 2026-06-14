"""A deterministic, ML-free detector for development and tests.

Returns a simple rectangular room (4 walls) with one door and one window, sized
relative to the input image. This lets the frontend and the normalization
pipeline be built and tested without a GPU, weights or the legacy service.
"""

from __future__ import annotations

import logging

import numpy as np

from app.services.detector_base import DOOR, WALL, WINDOW, Detection, DetectorBase

logger = logging.getLogger(__name__)


class MockDetector(DetectorBase):
    """Generate plausible example detections from the image dimensions."""

    def __init__(self, model_name: str = "mock-detector") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        # The mock is always ready.
        return True

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[0], image.shape[1]

        margin = 0.1
        thickness = max(4.0, 0.02 * min(width, height))
        half_t = thickness / 2.0

        # Room rectangle corners in pixel space (Y down).
        left = width * margin
        right = width * (1.0 - margin)
        top = height * margin
        bottom = height * (1.0 - margin)

        detections: list[Detection] = [
            # Top wall (horizontal).
            Detection(WALL, (left, top - half_t, right, top + half_t), 0.95),
            # Bottom wall (horizontal).
            Detection(WALL, (left, bottom - half_t, right, bottom + half_t), 0.93),
            # Left wall (vertical).
            Detection(WALL, (left - half_t, top, left + half_t, bottom), 0.92),
            # Right wall (vertical).
            Detection(WALL, (right - half_t, top, right + half_t, bottom), 0.90),
        ]

        # Door centered on the bottom wall.
        door_w = 0.10 * (right - left)
        door_cx = (left + right) / 2.0
        detections.append(
            Detection(
                DOOR,
                (door_cx - door_w / 2.0, bottom - thickness, door_cx + door_w / 2.0, bottom + thickness),
                0.88,
            )
        )

        # Window on the top wall, offset to the left.
        win_w = 0.14 * (right - left)
        win_cx = left + 0.30 * (right - left)
        detections.append(
            Detection(
                WINDOW,
                (win_cx - win_w / 2.0, top - thickness, win_cx + win_w / 2.0, top + thickness),
                0.85,
            )
        )

        logger.debug("MockDetector produced %d detections", len(detections))
        return detections
