"""Shared pytest fixtures: a test client and synthetic floor-plan images."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """A TestClient that runs the app lifespan (default = MockDetector)."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def plain_png() -> bytes:
    """A plain white PNG; content is irrelevant for the MockDetector."""
    image = Image.new("RGB", (240, 180), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _draw_floorplan(size: int = 500) -> np.ndarray:
    """Create an RGB image of a rectangular room with one internal divider."""
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    black = (0, 0, 0)
    thickness = 4
    margin = 50
    far = size - margin
    # Outer rectangle = 4 walls.
    cv2.rectangle(canvas, (margin, margin), (far, far), black, thickness)
    # Internal vertical divider = 1 wall.
    mid = size // 2
    cv2.line(canvas, (mid, margin), (mid, far), black, thickness)
    return canvas


@pytest.fixture
def floorplan_array() -> np.ndarray:
    """Synthetic floor-plan image as an RGB array (for detector unit tests)."""
    return _draw_floorplan()


@pytest.fixture
def floorplan_png() -> bytes:
    """Synthetic floor-plan image encoded as PNG bytes (for API tests)."""
    array = _draw_floorplan()
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()
