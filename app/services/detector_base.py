"""Detector abstraction.

The web layer depends only on `DetectorBase`, never on a concrete model. This
lets us swap Mask R-CNN, YOLO, an ONNX runtime or a mock without touching the
endpoint. A detector turns a decoded RGB image into raw pixel-space detections;
turning those into the normalized 3D contract is the builder's job.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# Object classes the system understands.
WALL = "wall"
DOOR = "door"
WINDOW = "window"
VALID_LABELS = frozenset({WALL, DOOR, WINDOW})


@dataclass(frozen=True)
class Detection:
    """A single raw detection in image (pixel) space.

    Detectors that only produce axis-aligned boxes (Mask R-CNN, the mock) fill in
    ``bbox`` and leave ``segment``/``thickness_px`` as ``None``; the builder then
    reduces the box to a center-line. Detectors that already produce true line
    segments (e.g. OpenCV/Hough, which may be diagonal) set ``segment`` directly
    so the diagonal is preserved. Either way the builder owns all normalization.

    Attributes:
        label: One of ``wall`` / ``door`` / ``window``.
        bbox: ``(x1, y1, x2, y2)`` in pixels, origin top-left, Y down.
        confidence: Detection score in ``[0, 1]``.
        segment: Optional explicit center-line ``(x1, y1, x2, y2)`` in pixels for
            walls; takes precedence over ``bbox`` when present.
        thickness_px: Optional measured wall thickness in pixels; when ``None``
            the builder falls back to the configured default thickness.
    """

    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    segment: tuple[float, float, float, float] | None = None
    thickness_px: float | None = None


class DetectorBase(ABC):
    """Interface every concrete detector must implement."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier reported in the response `meta.model`."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the detector is ready to serve requests."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run inference on an RGB image array and return raw detections.

        Args:
            image: Array of shape ``(height, width, 3)``, dtype ``uint8``.

        Returns:
            A list of :class:`Detection`. May be empty.
        """
