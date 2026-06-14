"""Unit tests for the normalization builder and geometry rules."""

from __future__ import annotations

from app.schemas.detection import Unit
from app.services.detector_base import DOOR, WALL, Detection
from app.services.floorplan_builder import BuildOptions, FloorPlanBuilder

_OPTIONS_METERS = BuildOptions(
    confidence_threshold=0.5,
    wall_height=2.7,
    default_wall_thickness=0.15,
    pixels_per_meter=10.0,
    door_height=2.1,
    window_height=1.1,
    window_sill_height=0.9,
)


def _build(detections: list[Detection], options: BuildOptions, height: int = 100):
    builder = FloorPlanBuilder(api_version="1.0")
    return builder.build(
        detections,
        image_width=100,
        image_height=height,
        options=options,
        model_name="test",
    )


def test_y_axis_is_flipped_to_world_coordinates() -> None:
    # Two horizontal walls: one near the image top, one near the bottom.
    top = Detection(WALL, (10, 8, 90, 12), 0.9)      # pixel y ~10 (top)
    bottom = Detection(WALL, (10, 88, 90, 92), 0.9)   # pixel y ~90 (bottom)
    plan = _build([top, bottom], _OPTIONS_METERS)

    # In world space (Y up) the top wall must end up higher than the bottom one.
    assert plan.walls[0].start.y > plan.walls[1].start.y
    # ppm=10, height=100 -> top y = (100-10)/10 = 9.0
    assert plan.walls[0].start.y == 9.0
    assert plan.walls[1].start.y == 1.0


def test_explicit_segment_is_preserved_for_diagonals() -> None:
    # A diagonal wall provided as an explicit segment must keep both axes.
    diagonal = Detection(WALL, (0, 0, 50, 50), 0.8, segment=(0.0, 0.0, 50.0, 50.0))
    plan = _build([diagonal], _OPTIONS_METERS)
    wall = plan.walls[0]
    assert wall.start.x != wall.end.x
    assert wall.start.y != wall.end.y


def test_opening_is_linked_to_nearest_wall() -> None:
    wall = Detection(WALL, (0, 48, 100, 52), 0.9)  # horizontal wall mid-height
    door = Detection(DOOR, (48, 47, 58, 57), 0.9)  # centered on the wall
    plan = _build([wall, door], _OPTIONS_METERS)
    assert plan.doors[0].wall_id == plan.walls[0].id


def test_normalized_mode_when_no_scale() -> None:
    options = BuildOptions(
        confidence_threshold=0.5,
        wall_height=2.7,
        default_wall_thickness=0.15,
        pixels_per_meter=None,
        door_height=2.1,
        window_height=1.1,
        window_sill_height=0.9,
    )
    wall = Detection(WALL, (0, 0, 100, 4), 0.9)
    plan = _build([wall], options)
    assert plan.image.unit is Unit.normalized
    assert plan.image.pixels_per_meter is None
    # Normalized coordinates stay within 0..1.
    assert 0.0 <= plan.walls[0].end.x <= 1.0


def test_bounds_cover_all_geometry() -> None:
    walls = [
        Detection(WALL, (0, 0, 100, 4), 0.9),
        Detection(WALL, (0, 0, 4, 80), 0.9),
    ]
    plan = _build(walls, _OPTIONS_METERS)
    assert plan.bounds.min_x <= plan.bounds.max_x
    assert plan.bounds.min_y <= plan.bounds.max_y
    assert plan.bounds.max_x > 0
