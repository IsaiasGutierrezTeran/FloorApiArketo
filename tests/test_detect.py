"""End-to-end tests for /detect using the MockDetector (default config)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.schemas.detection import FloorPlan3D


def test_detect_returns_valid_schema(client: TestClient, plain_png: bytes) -> None:
    response = client.post(
        "/detect", files={"file": ("plan.png", plain_png, "image/png")}
    )
    assert response.status_code == 200

    # The response must validate against the public Pydantic contract.
    plan = FloorPlan3D.model_validate(response.json())

    # MockDetector yields a 4-wall room with one door and one window.
    assert len(plan.walls) == 4
    assert len(plan.doors) == 1
    assert len(plan.windows) == 1
    assert plan.meta.model == "mock-detector"
    assert plan.meta.fallback_used is False
    # Every opening should be linked to one of the walls.
    wall_ids = {wall.id for wall in plan.walls}
    assert plan.doors[0].wall_id in wall_ids
    assert plan.windows[0].wall_id in wall_ids


def test_detect_normalized_without_scale(client: TestClient, plain_png: bytes) -> None:
    response = client.post(
        "/detect", files={"file": ("plan.png", plain_png, "image/png")}
    )
    plan = FloorPlan3D.model_validate(response.json())
    assert plan.image.unit.value == "normalized"
    assert plan.image.pixels_per_meter is None


def test_detect_meters_with_scale(client: TestClient, plain_png: bytes) -> None:
    response = client.post(
        "/detect",
        files={"file": ("plan.png", plain_png, "image/png")},
        params={"pixels_per_meter": 50.0},
    )
    plan = FloorPlan3D.model_validate(response.json())
    assert plan.image.unit.value == "meters"
    assert plan.image.pixels_per_meter == 50.0


def test_detect_confidence_filter(client: TestClient, plain_png: bytes) -> None:
    # With a high threshold only the strongest walls survive and the lower-score
    # door/window are filtered out.
    response = client.post(
        "/detect",
        files={"file": ("plan.png", plain_png, "image/png")},
        params={"confidence_threshold": 0.91},
    )
    plan = FloorPlan3D.model_validate(response.json())
    assert len(plan.walls) == 3  # walls at 0.95 / 0.93 / 0.92 pass, 0.90 dropped
    assert plan.doors == []
    assert plan.windows == []
