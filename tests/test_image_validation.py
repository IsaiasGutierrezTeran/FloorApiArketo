"""Tests for image validation and error responses."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.exceptions import ImageTooLargeError, InvalidImageError
from app.utils.image_io import load_and_validate_image


def test_detect_rejects_non_image(client: TestClient) -> None:
    response = client.post(
        "/detect", files={"file": ("note.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_image"


def test_detect_requires_file(client: TestClient) -> None:
    response = client.post("/detect")
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_load_image_empty_raises() -> None:
    with pytest.raises(InvalidImageError):
        load_and_validate_image(b"", max_size_mb=10, max_dimension=4096)


def test_load_image_too_large_raises() -> None:
    # ~2 MB of bytes against a 1 MB limit -> 413 domain error.
    blob = b"\x00" * (2 * 1024 * 1024)
    with pytest.raises(ImageTooLargeError):
        load_and_validate_image(blob, max_size_mb=1.0, max_dimension=4096)


def test_load_image_downscales_large_dimensions() -> None:
    import io

    from PIL import Image

    big = Image.new("RGB", (1000, 500), "white")
    buffer = io.BytesIO()
    big.save(buffer, format="PNG")

    array, width, height = load_and_validate_image(
        buffer.getvalue(), max_size_mb=10, max_dimension=400
    )
    assert max(width, height) == 400
    assert array.shape == (height, width, 3)
