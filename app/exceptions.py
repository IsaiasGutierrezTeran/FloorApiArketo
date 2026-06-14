"""Domain-specific exceptions mapped to HTTP responses in `app.main`.

Each error carries an HTTP status code and a stable machine-readable `error`
code so clients can branch on it without parsing free-text messages.
"""

from __future__ import annotations


class ApiError(Exception):
    """Base class for errors that translate into a JSON HTTP response."""

    status_code: int = 500
    error: str = "internal_error"

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail
        super().__init__(detail or self.error)


class InvalidImageError(ApiError):
    """The uploaded file is missing, empty or not a decodable image."""

    status_code = 400
    error = "invalid_image"


class ImageTooLargeError(ApiError):
    """The uploaded image exceeds the configured byte-size limit."""

    status_code = 413
    error = "image_too_large"


class InferenceError(ApiError):
    """The detector failed while running inference."""

    status_code = 500
    error = "inference_error"
