"""Error response model returned for every non-2xx response."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Uniform error body: `{ "error": <code>, "detail": <message> }`."""

    error: str = Field(..., description="Stable machine-readable error code.")
    detail: str | None = Field(
        default=None, description="Human-readable explanation of what went wrong."
    )
