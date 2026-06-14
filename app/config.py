"""Application configuration loaded from environment variables / `.env`.

All tunables (server, detector selection, inference defaults, limits and CORS)
live here so that no other module reads the environment directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings.

    Values are read (case-insensitively) from environment variables or a local
    `.env` file. See `.env.example` for the full list.
    """

    # `protected_namespaces=()` allows fields such as `model_name` without
    # triggering Pydantic's "model_" namespace warning.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Server -------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # --- Detector selection -------------------------------------------------
    detector: Literal["mock", "maskrcnn", "opencv"] = "mock"
    weights_path: str = "weights/maskrcnn_floorplan.h5"
    model_name: str = "maskrcnn-resnet101"

    # When True, if the primary detector fails / finds 0 walls / has mean wall
    # confidence below `confidence_threshold`, the request is retried with the
    # classical OpenCV detector (and `meta.fallback_used` is set).
    fallback_to_opencv: bool = False

    # --- OpenCV (classical) detector parameters -----------------------------
    opencv_min_wall_length_px: int = 40
    opencv_hough_threshold: int = 80
    opencv_hough_min_line_length: int = 50
    opencv_hough_max_line_gap: int = 10
    opencv_merge_distance_px: float = 10.0

    # --- Legacy Mask R-CNN service ------------------------------------------
    # The real model (Fady Aziz' FloorPlanTo3D-API) runs on Python 3.6 / TF 1.15
    # and cannot be imported in-process here, so `MaskRCNNDetector` reaches it
    # over HTTP instead. Point this at the running Flask service.
    legacy_api_url: str = "http://127.0.0.1:5000/"
    legacy_api_timeout: float = 120.0
    # The legacy model returns no per-detection score, so we assign this one.
    legacy_default_confidence: float = 0.99

    # --- Inference defaults (overridable per request) ----------------------
    confidence_threshold: float = 0.5
    wall_height: float = 2.7
    default_wall_thickness: float = 0.15
    # Heights cannot be measured from a top-down plan, so they come from config.
    door_height: float = 2.1
    window_height: float = 1.1
    window_sill_height: float = 0.9

    # --- Image limits -------------------------------------------------------
    max_image_size_mb: float = 10.0
    max_image_dimension: int = 4096

    # --- CORS ---------------------------------------------------------------
    cors_origins: list[str] = ["*"]

    # --- Meta ---------------------------------------------------------------
    api_version: str = "1.0"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: object) -> object:
        """Accept a comma-separated string for `CORS_ORIGINS` in addition to JSON.

        This lets `.env` use `CORS_ORIGINS=http://localhost:4200,http://localhost:8080`
        instead of requiring a JSON array.
        """
        if isinstance(value, str) and not value.strip().startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Return a cached `Settings` instance (single source of truth)."""
    return Settings()
