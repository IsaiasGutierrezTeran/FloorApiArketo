"""Pure geometry helpers used by the floor-plan builder.

All functions are stateless and operate on plain floats/tuples so they are easy
to unit-test. Pixel coordinates use image space (origin top-left, Y down); world
coordinates use a bottom-left origin with Y up.
"""

from __future__ import annotations

import math

# A bounding box in pixel space: (x1, y1, x2, y2).
BBox = tuple[float, float, float, float]
# A segment in pixel space: (ax, ay, bx, by, thickness).
Segment = tuple[float, float, float, float, float]


def bbox_to_segment(bbox: BBox) -> Segment:
    """Reduce a wall bounding box to a center-line segment plus thickness.

    A wall box is long in one axis and thin in the other. The segment runs along
    the long axis through the box center; the thickness is the short side.
    """
    x1, y1, x2, y2 = bbox
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    if width >= height:  # horizontal wall
        center_y = (y1 + y2) / 2.0
        return (min(x1, x2), center_y, max(x1, x2), center_y, height)
    # vertical wall
    center_x = (x1 + x2) / 2.0
    return (center_x, min(y1, y2), center_x, max(y1, y2), width)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    """Return the (x, y) center of a bounding box in pixel space."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_opening_width(bbox: BBox) -> float:
    """Opening width of a door/window box (its longer side, in pixels)."""
    x1, y1, x2, y2 = bbox
    return max(abs(x2 - x1), abs(y2 - y1))


def pixel_to_world(
    px: float,
    py: float,
    image_height: int,
    scale: float,
) -> tuple[float, float]:
    """Convert pixel coordinates (Y down) to world coordinates (Y up).

    Args:
        px, py: Pixel coordinates.
        image_height: Image height in pixels (used to flip the Y axis).
        scale: Multiplier mapping pixels to world units (1/ppm, or 1/longest_side).
    """
    return (px * scale, (image_height - py) * scale)


def point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Shortest distance from point (px, py) to segment (a, b)."""
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x, proj_y = ax + t * dx, ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)
