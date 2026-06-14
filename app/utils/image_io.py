"""Image decoding, validation and resizing.

Keeps all Pillow/NumPy handling in one place so the web and service layers only
ever see a validated `(rgb_array, width, height)` triple.
"""

from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.exceptions import ImageTooLargeError, InvalidImageError

logger = logging.getLogger(__name__)

_BYTES_PER_MB = 1024 * 1024


def load_and_validate_image(
    data: bytes,
    *,
    max_size_mb: float,
    max_dimension: int,
) -> tuple[np.ndarray, int, int]:
    """Decode raw bytes into an RGB image array, validating size and format.

    Args:
        data: Raw uploaded file bytes.
        max_size_mb: Maximum allowed payload size in megabytes (-> 413).
        max_dimension: Images whose longest side exceeds this are downscaled.

    Returns:
        A tuple of ``(rgb_array, width, height)`` where ``rgb_array`` has shape
        ``(height, width, 3)`` and dtype ``uint8``.

    Raises:
        InvalidImageError: Empty or undecodable input (-> 400).
        ImageTooLargeError: Payload exceeds ``max_size_mb`` (-> 413).
    """
    if not data:
        raise InvalidImageError("Uploaded file is empty.")

    size_mb = len(data) / _BYTES_PER_MB
    if size_mb > max_size_mb:
        raise ImageTooLargeError(
            f"Image is {size_mb:.1f} MB; the maximum allowed is {max_size_mb} MB."
        )

    try:
        # `verify()` checks integrity but leaves the file unusable, so reopen.
        Image.open(io.BytesIO(data)).verify()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("File is not a valid image.") from exc

    width, height = image.size
    longest = max(width, height)
    if longest > max_dimension:
        ratio = max_dimension / float(longest)
        new_size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        logger.info("Downscaling image from %sx%s to %sx%s", width, height, *new_size)
        image = image.resize(new_size, Image.Resampling.BILINEAR)
        width, height = image.size

    return np.asarray(image, dtype=np.uint8), width, height


def encode_png(image: np.ndarray) -> bytes:
    """Encode an RGB image array back to PNG bytes (for forwarding upstream)."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()
