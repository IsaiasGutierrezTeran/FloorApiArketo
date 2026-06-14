"""Pydantic v2 models for the normalized 3D floor-plan response.

This is the public contract consumed by the Django backend, Angular, Flutter and
the Three.js viewer. Coordinates use a *world* frame: origin at the bottom-left,
Y pointing up. Units are meters when `pixels_per_meter` is known, otherwise the
geometry is normalized to the 0..1 range (aspect ratio preserved).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Unit(str, Enum):
    """Unit of the geometric coordinates in the response."""

    meters = "meters"
    normalized = "normalized"


class Point2D(BaseModel):
    """A 2D point in world coordinates (Y up)."""

    x: float
    y: float


class ImageInfo(BaseModel):
    """Source image metadata and the unit used for the geometry."""

    width: int = Field(..., description="Image width in pixels (after any resize).")
    height: int = Field(..., description="Image height in pixels (after any resize).")
    unit: Unit = Field(..., description="`meters` if scaled, else `normalized`.")
    pixels_per_meter: float | None = Field(
        default=None, description="Scale used, or null when normalized."
    )


class ScaleInfo(BaseModel):
    """Real-world extrusion hints applied to the geometry."""

    wall_height: float = Field(..., description="Wall height in meters.")
    default_wall_thickness: float = Field(
        ..., description="Fallback wall thickness in meters."
    )


class Wall(BaseModel):
    """A wall as an extrudable segment (length = |end - start|)."""

    id: str
    start: Point2D = Field(..., description="Segment start, world coords.")
    end: Point2D = Field(..., description="Segment end, world coords.")
    thickness: float = Field(..., description="Wall thickness (extrusion width).")
    height: float = Field(..., description="Wall height (extrusion height).")
    confidence: float = Field(..., ge=0.0, le=1.0)


class Door(BaseModel):
    """A door opening attached to a wall."""

    id: str
    wall_id: str | None = Field(
        default=None, description="Nearest wall id, or null if not inferable."
    )
    position: Point2D = Field(..., description="Opening center, world coords.")
    width: float = Field(..., description="Opening width along the wall.")
    height: float = Field(..., description="Door height in meters.")
    confidence: float = Field(..., ge=0.0, le=1.0)


class Window(BaseModel):
    """A window opening attached to a wall."""

    id: str
    wall_id: str | None = Field(
        default=None, description="Nearest wall id, or null if not inferable."
    )
    position: Point2D = Field(..., description="Opening center, world coords.")
    width: float = Field(..., description="Opening width along the wall.")
    height: float = Field(..., description="Window height in meters.")
    sill_height: float = Field(..., description="Sill height from the floor in meters.")
    confidence: float = Field(..., ge=0.0, le=1.0)


class Bounds(BaseModel):
    """Axis-aligned bounding box of all geometry (to center the camera)."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float


class Meta(BaseModel):
    """Processing metadata."""

    # Allow the `model` field name without Pydantic's protected-namespace warning.
    model_config = ConfigDict(protected_namespaces=())

    model: str = Field(..., description="Detector / model actually used.")
    version: str = Field(..., description="API contract version.")
    processing_ms: int = Field(..., description="Server-side processing time in ms.")
    fallback_used: bool = Field(
        default=False,
        description="True when the primary detector was replaced by the fallback.",
    )


class FloorPlan3D(BaseModel):
    """Top-level normalized response, ready for 3D extrusion."""

    image: ImageInfo
    scale: ScaleInfo
    walls: list[Wall]
    doors: list[Door]
    windows: list[Window]
    bounds: Bounds
    meta: Meta
