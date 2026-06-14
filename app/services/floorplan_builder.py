"""Normalize raw detections into the public 3D floor-plan contract.

This is the single place where pixel-space detections become world-space,
client-ready geometry. Rules implemented here:

* Origin at the bottom-left, Y axis pointing up (image Y is flipped).
* If ``pixels_per_meter`` is given, everything is converted to meters; otherwise
  geometry is normalized to 0..1 (divided by the longest side, aspect ratio
  preserved) and ``unit`` becomes ``normalized``.
* Walls are emitted as segments (start/end), not pixel boxes, so clients extrude
  prisms directly.
* Real ``bounds`` are computed from all geometry to center the camera.
* Each opening is attached to its nearest wall (``wall_id``) when walls exist.
* Stable ids: ``w1``, ``w2`` … / ``d1`` … / ``win1`` …

No detector logic lives here, and no normalization lives in the detectors.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.detection import (
    Bounds,
    Door,
    FloorPlan3D,
    ImageInfo,
    Meta,
    Point2D,
    ScaleInfo,
    Unit,
    Wall,
    Window,
)
from app.services.detector_base import DOOR, WALL, WINDOW, Detection
from app.utils.geometry import (
    bbox_center,
    bbox_opening_width,
    bbox_to_segment,
    pixel_to_world,
    point_to_segment_distance,
)

# Floor for converted wall thickness so a wall is never zero-width.
_MIN_THICKNESS = 1e-4


@dataclass(frozen=True)
class BuildOptions:
    """Per-request build parameters (resolved from query params + settings)."""

    confidence_threshold: float
    wall_height: float
    default_wall_thickness: float
    pixels_per_meter: float | None
    door_height: float
    window_height: float
    window_sill_height: float


class FloorPlanBuilder:
    """Convert a list of :class:`Detection` into a :class:`FloorPlan3D`."""

    def __init__(self, api_version: str) -> None:
        self._api_version = api_version

    def build(
        self,
        detections: list[Detection],
        *,
        image_width: int,
        image_height: int,
        options: BuildOptions,
        model_name: str,
        fallback_used: bool = False,
        processing_ms: int = 0,
    ) -> FloorPlan3D:
        """Build the normalized response from raw detections.

        ``detections`` are expected to be already confidence-filtered by the
        caller; this method does not re-filter.
        """
        scale, unit, ppm = self._resolve_scale(image_width, image_height, options)

        # --- Walls ----------------------------------------------------------
        # Stored world segments power both the response and opening->wall linking.
        wall_models: list[Wall] = []
        wall_world_segments: list[tuple[str, float, float, float, float]] = []

        wall_dets = [d for d in detections if d.label == WALL]
        for index, det in enumerate(wall_dets, start=1):
            wall_id = f"w{index}"
            ax_px, ay_px, bx_px, by_px, thickness_px = self._wall_pixels(det)

            ax, ay = pixel_to_world(ax_px, ay_px, image_height, scale)
            bx, by = pixel_to_world(bx_px, by_px, image_height, scale)
            thickness = self._wall_thickness(thickness_px, scale, unit, options)

            wall_models.append(
                Wall(
                    id=wall_id,
                    start=Point2D(x=ax, y=ay),
                    end=Point2D(x=bx, y=by),
                    thickness=thickness,
                    height=options.wall_height,
                    confidence=det.confidence,
                )
            )
            wall_world_segments.append((wall_id, ax, ay, bx, by))

        # --- Doors & windows ------------------------------------------------
        door_models: list[Door] = []
        for index, det in enumerate(
            (d for d in detections if d.label == DOOR), start=1
        ):
            cx_px, cy_px = bbox_center(det.bbox)
            cx, cy = pixel_to_world(cx_px, cy_px, image_height, scale)
            door_models.append(
                Door(
                    id=f"d{index}",
                    wall_id=self._nearest_wall(cx, cy, wall_world_segments),
                    position=Point2D(x=cx, y=cy),
                    width=bbox_opening_width(det.bbox) * scale,
                    height=options.door_height,
                    confidence=det.confidence,
                )
            )

        window_models: list[Window] = []
        for index, det in enumerate(
            (d for d in detections if d.label == WINDOW), start=1
        ):
            cx_px, cy_px = bbox_center(det.bbox)
            cx, cy = pixel_to_world(cx_px, cy_px, image_height, scale)
            window_models.append(
                Window(
                    id=f"win{index}",
                    wall_id=self._nearest_wall(cx, cy, wall_world_segments),
                    position=Point2D(x=cx, y=cy),
                    width=bbox_opening_width(det.bbox) * scale,
                    height=options.window_height,
                    sill_height=options.window_sill_height,
                    confidence=det.confidence,
                )
            )

        bounds = self._compute_bounds(wall_models, door_models, window_models)

        return FloorPlan3D(
            image=ImageInfo(
                width=image_width,
                height=image_height,
                unit=unit,
                pixels_per_meter=ppm,
            ),
            scale=ScaleInfo(
                wall_height=options.wall_height,
                default_wall_thickness=options.default_wall_thickness,
            ),
            walls=wall_models,
            doors=door_models,
            windows=window_models,
            bounds=bounds,
            meta=Meta(
                model=model_name,
                version=self._api_version,
                processing_ms=processing_ms,
                fallback_used=fallback_used,
            ),
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_scale(
        image_width: int, image_height: int, options: BuildOptions
    ) -> tuple[float, Unit, float | None]:
        """Return (pixel->world scale, unit, pixels_per_meter-or-None)."""
        if options.pixels_per_meter and options.pixels_per_meter > 0:
            return 1.0 / options.pixels_per_meter, Unit.meters, options.pixels_per_meter
        longest = float(max(image_width, image_height)) or 1.0
        return 1.0 / longest, Unit.normalized, None

    @staticmethod
    def _wall_pixels(det: Detection) -> tuple[float, float, float, float, float | None]:
        """Center-line endpoints + thickness in pixels for a wall detection.

        Uses an explicit ``segment`` when present (preserves diagonals); otherwise
        reduces the axis-aligned ``bbox`` to a center-line.
        """
        if det.segment is not None:
            ax, ay, bx, by = det.segment
            return ax, ay, bx, by, det.thickness_px
        ax, ay, bx, by, thickness_px = bbox_to_segment(det.bbox)
        return ax, ay, bx, by, thickness_px

    @staticmethod
    def _wall_thickness(
        thickness_px: float | None,
        scale: float,
        unit: Unit,
        options: BuildOptions,
    ) -> float:
        """Convert measured thickness to world units, else use the default."""
        if thickness_px is not None and unit is Unit.meters:
            return max(thickness_px * scale, _MIN_THICKNESS)
        return options.default_wall_thickness

    @staticmethod
    def _nearest_wall(
        x: float,
        y: float,
        wall_segments: list[tuple[str, float, float, float, float]],
    ) -> str | None:
        """Id of the wall whose segment is closest to (x, y), or None."""
        best_id: str | None = None
        best_distance = float("inf")
        for wall_id, ax, ay, bx, by in wall_segments:
            distance = point_to_segment_distance(x, y, ax, ay, bx, by)
            if distance < best_distance:
                best_distance = distance
                best_id = wall_id
        return best_id

    @staticmethod
    def _compute_bounds(
        walls: list[Wall], doors: list[Door], windows: list[Window]
    ) -> Bounds:
        """Axis-aligned bounds over all geometry; zeros when empty."""
        xs: list[float] = []
        ys: list[float] = []
        for wall in walls:
            xs.extend((wall.start.x, wall.end.x))
            ys.extend((wall.start.y, wall.end.y))
        for opening in (*doors, *windows):
            xs.append(opening.position.x)
            ys.append(opening.position.y)
        if not xs:
            return Bounds(min_x=0.0, min_y=0.0, max_x=0.0, max_y=0.0)
        return Bounds(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))
