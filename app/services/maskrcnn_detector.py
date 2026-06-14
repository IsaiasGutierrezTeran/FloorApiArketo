"""Detector backed by the original FloorPlanTo3D Mask R-CNN model.

That model (Fady Aziz' repo) is pinned to Python 3.6 + TensorFlow 1.15 and
cannot be imported inside this modern (3.10+) process. Instead of vendoring an
incompatible stack, this detector treats the original Flask service as a remote
inference backend: it forwards the image over HTTP and adapts the legacy JSON
(`points` / `classes` / `Width` / `Height` / `averageDoor`) into our uniform
:class:`Detection` list. The normalization to the 3D contract still happens in
:mod:`app.services.floorplan_builder`.

TODO (optional, advanced): replace the HTTP hop by exporting the trained model to
ONNX and running it in-process here with `onnxruntime`, removing the dependency
on the legacy Python 3.6 service.
"""

from __future__ import annotations

import logging
import socket
from urllib.parse import urlparse

import httpx
import numpy as np

from app.exceptions import InferenceError
from app.services.detector_base import VALID_LABELS, Detection, DetectorBase
from app.utils.image_io import encode_png

logger = logging.getLogger(__name__)


class MaskRCNNDetector(DetectorBase):
    """Forward inference to the legacy Mask R-CNN Flask service over HTTP."""

    def __init__(
        self,
        *,
        api_url: str,
        model_name: str,
        timeout: float = 120.0,
        default_confidence: float = 0.99,
    ) -> None:
        self._api_url = api_url
        self._model_name = model_name
        self._timeout = timeout
        self._default_confidence = default_confidence
        self._available = self._check_reachable()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """True when the legacy service was reachable at startup."""
        return self._available

    def _check_reachable(self) -> bool:
        """Best-effort TCP probe of the legacy service host:port."""
        parsed = urlparse(self._api_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=2.0):
                logger.info("Legacy Mask R-CNN service reachable at %s:%s", host, port)
                return True
        except OSError as exc:
            logger.warning(
                "Legacy Mask R-CNN service not reachable at %s:%s (%s). "
                "Detector marked as not loaded.",
                host,
                port,
                exc,
            )
            return False

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Send the image to the legacy service and adapt its response."""
        payload = encode_png(image)
        try:
            response = httpx.post(
                self._api_url,
                files={"image": ("plan.png", payload, "image/png")},
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise InferenceError(
                f"Legacy Mask R-CNN request failed: {exc}"
            ) from exc

        return self._adapt(data)

    def _adapt(self, data: dict) -> list[Detection]:
        """Map the legacy `{points, classes, ...}` payload to detections.

        The legacy model does not return per-detection scores, so every
        detection is assigned ``default_confidence``.
        """
        points = data.get("points", [])
        classes = data.get("classes", [])
        detections: list[Detection] = []
        for point, klass in zip(points, classes):
            label = str(klass.get("name", "")).lower()
            if label not in VALID_LABELS:
                continue
            try:
                bbox = (
                    float(point["x1"]),
                    float(point["y1"]),
                    float(point["x2"]),
                    float(point["y2"]),
                )
            except (KeyError, TypeError, ValueError):
                logger.debug("Skipping malformed legacy point: %r", point)
                continue
            detections.append(
                Detection(label=label, bbox=bbox, confidence=self._default_confidence)
            )

        logger.info("Legacy Mask R-CNN returned %d usable detections", len(detections))
        return detections
