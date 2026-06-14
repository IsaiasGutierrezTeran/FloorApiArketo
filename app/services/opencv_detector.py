"""Classical computer-vision wall detector (no deep learning).

A GPU-free, weight-free fallback that extracts wall segments from a clean floor
plan using standard OpenCV operations. Doors and windows are *not* reliably
detectable without learning, so they are returned empty on purpose (we never
fabricate data). All code here is original.

Pipeline (see :meth:`OpenCVDetector.detect`):
    1. grayscale
    2. binarize (Otsu, inverted -> dark walls become foreground)
    3. morphology (open to denoise, close to bridge strokes)
    4. HoughLinesP -> straight segment candidates
    5. merge collinear/near segments and drop short ones -> wall segments
    6. (optional) outer contour is implicit in the merged segments / bounds,
       which the builder computes from the geometry.

The detector only emits crude pixel-space segments; converting them to meters,
flipping Y, computing bounds and assigning ids is the builder's responsibility.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from app.services.detector_base import WALL, Detection, DetectorBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenCVParams:
    """Tunable parameters for the classical pipeline (sane defaults).

    Attributes:
        min_wall_length_px: Discard merged segments shorter than this.
        hough_threshold: HoughLinesP accumulator threshold (votes).
        hough_min_line_length: Minimum length of a raw Hough segment.
        hough_max_line_gap: Max gap to bridge collinear Hough points.
        merge_distance_px: Max perpendicular/longitudinal distance to merge
            two near-collinear segments into one.
    """

    min_wall_length_px: int = 40
    hough_threshold: int = 80
    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10
    merge_distance_px: float = 10.0

    def with_overrides(self, **overrides: float | int | None) -> "OpenCVParams":
        """Return a copy with the non-``None`` overrides applied."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean) if clean else self


# Two segments are considered "same direction" within this angular tolerance.
_ANGLE_TOLERANCE_DEG = 10.0


class OpenCVDetector(DetectorBase):
    """Detect walls as line segments using classical OpenCV techniques."""

    def __init__(self, params: OpenCVParams | None = None) -> None:
        self._params = params or OpenCVParams()

    @property
    def params(self) -> OpenCVParams:
        return self._params

    @property
    def model_name(self) -> str:
        return "opencv-classic"

    @property
    def is_loaded(self) -> bool:
        # No weights to load: always ready.
        return True

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run the classical pipeline and return wall detections."""
        params = self._params

        # 1) Grayscale. Our arrays are RGB (see image_io), so use RGB2GRAY.
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 2) Binarize with Otsu, inverted so dark wall strokes become white (255)
        #    foreground on a black background.
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 3) Morphology: opening removes salt noise, closing reconnects strokes
        #    broken by text/symbols so walls form continuous lines.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 4) Probabilistic Hough transform -> raw straight segments.
        raw = cv2.HoughLinesP(
            cleaned,
            rho=1,
            theta=math.pi / 180.0,
            threshold=params.hough_threshold,
            minLineLength=params.hough_min_line_length,
            maxLineGap=params.hough_max_line_gap,
        )
        if raw is None:
            logger.info("OpenCVDetector: HoughLinesP found no segments")
            return []

        segments = [tuple(map(float, line[0])) for line in raw]

        # 5) Merge near-collinear segments, then drop anything too short to be a
        #    wall.
        merged = _merge_segments(
            segments,
            merge_distance=params.merge_distance_px,
            angle_tolerance_deg=_ANGLE_TOLERANCE_DEG,
        )
        walls = [s for s in merged if _length(s) >= params.min_wall_length_px]

        diagonal = math.hypot(image.shape[1], image.shape[0])
        detections = [
            Detection(
                label=WALL,
                bbox=_segment_bbox(seg),
                confidence=_segment_confidence(seg, diagonal),
                segment=seg,
                # Thickness is not measured from Hough lines -> let the builder
                # apply the configured default wall thickness.
                thickness_px=None,
            )
            for seg in walls
        ]

        # NOTE: doors and windows require learned features; returning them empty
        # is intentional. TODO: optionally detect door arcs / window double-lines
        # with template matching or arc detection if reliable enough.
        logger.info("OpenCVDetector produced %d wall segments", len(detections))
        return detections


# --------------------------------------------------------------------------- #
# Segment helpers (pure functions)                                            #
# --------------------------------------------------------------------------- #

Seg = tuple[float, float, float, float]


def _length(seg: Seg) -> float:
    x1, y1, x2, y2 = seg
    return math.hypot(x2 - x1, y2 - y1)


def _angle_deg(seg: Seg) -> float:
    """Orientation in [0, 180) degrees (direction-agnostic)."""
    x1, y1, x2, y2 = seg
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    return angle


def _angle_diff(a: float, b: float) -> float:
    """Smallest difference between two angles in [0, 180)."""
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def _segment_bbox(seg: Seg) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = seg
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _segment_confidence(seg: Seg, diagonal: float) -> float:
    """Heuristic confidence: longer segments are more likely real walls.

    This is a transparent length heuristic, not a learned probability. It maps a
    segment spanning ~25% of the image diagonal (or more) to ~0.9, and short
    ones to the 0.5 floor.
    """
    if diagonal <= 0:
        return 0.5
    ratio = min(1.0, _length(seg) / (0.25 * diagonal))
    return round(0.5 + 0.4 * ratio, 4)


def _point_to_line_distance(
    px: float, py: float, ax: float, ay: float, ux: float, uy: float
) -> float:
    """Perpendicular distance from (px, py) to the infinite line through (ax, ay)
    with unit direction (ux, uy)."""
    # Cross product magnitude of (p - a) x u, with u already unit-length.
    return abs((px - ax) * uy - (py - ay) * ux)


def _merge_segments(
    segments: list[Seg],
    *,
    merge_distance: float,
    angle_tolerance_deg: float,
) -> list[Seg]:
    """Greedily merge near-collinear, nearby segments into single segments.

    Two segments merge when their orientations match within
    ``angle_tolerance_deg`` and the endpoints of one lie within
    ``merge_distance`` (perpendicular) of the other's supporting line. Each merged
    group is refit to a single segment spanning the projected extent along the
    group's dominant direction.
    """
    n = len(segments)
    used = [False] * n
    result: list[Seg] = []

    for i in range(n):
        if used[i]:
            continue
        group = [segments[i]]
        used[i] = True

        # Iteratively absorb compatible segments until the group stops growing.
        changed = True
        while changed:
            changed = False
            current = _fit_group(group)
            cx1, cy1, cx2, cy2 = current
            clen = _length(current) or 1.0
            ux, uy = (cx2 - cx1) / clen, (cy2 - cy1) / clen
            cur_angle = _angle_deg(current)

            for j in range(n):
                if used[j]:
                    continue
                if _angle_diff(_angle_deg(segments[j]), cur_angle) > angle_tolerance_deg:
                    continue
                jx1, jy1, jx2, jy2 = segments[j]
                d1 = _point_to_line_distance(jx1, jy1, cx1, cy1, ux, uy)
                d2 = _point_to_line_distance(jx2, jy2, cx1, cy1, ux, uy)
                if d1 <= merge_distance and d2 <= merge_distance:
                    group.append(segments[j])
                    used[j] = True
                    changed = True

        result.append(_fit_group(group))

    return result


def _fit_group(group: list[Seg]) -> Seg:
    """Refit a group of segments to one segment along its dominant direction."""
    if len(group) == 1:
        return group[0]

    # Dominant direction = orientation of the longest segment in the group.
    longest = max(group, key=_length)
    lx1, ly1, lx2, ly2 = longest
    length = _length(longest) or 1.0
    ux, uy = (lx2 - lx1) / length, (ly2 - ly1) / length

    # Project every endpoint onto the direction; keep the extreme projections.
    points = [(s[0], s[1]) for s in group] + [(s[2], s[3]) for s in group]
    origin_x, origin_y = points[0]
    projections = [((px - origin_x) * ux + (py - origin_y) * uy) for px, py in points]
    t_min, t_max = min(projections), max(projections)

    start = (origin_x + t_min * ux, origin_y + t_min * uy)
    end = (origin_x + t_max * ux, origin_y + t_max * uy)
    return (start[0], start[1], end[0], end[1])
