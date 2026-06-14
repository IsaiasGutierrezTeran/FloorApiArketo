# FloorPlan-to-3D Detection API

A clean, decoupled **FastAPI** service that turns a 2D architectural floor-plan
image into a **normalized JSON** describing walls, doors and windows — ready for
3D extrusion in a Three.js viewer, Flutter app or any client. It is the
inference service of the larger **Plan Risk 3D** system (Django backend, Angular
frontend, Flutter app).

It modernizes (and reuses) the original
[FloorPlanTo3D-API](https://github.com/fadyazizz/FloorPlanTo3D-API) by Fady Aziz:
the trained Mask R-CNN model still does the heavy lifting, but it is wrapped
behind a clean, typed, pluggable API.

- **Python 3.10+**, FastAPI, Pydantic v2, OpenAPI docs at `/docs`
- Pluggable detectors behind a single `DetectorBase` interface
- World-space, unit-aware output (meters or normalized 0..1)
- CORS, structured logging, typed errors, tests, Docker

---

## Architecture

```
Angular / Flutter / Three.js
        │  multipart image
        ▼
  floorplan-api  (FastAPI · Pydantic v2)
   ├─ utils/image_io      validate · decode · resize
   ├─ services/DetectorBase
   │    ├─ MockDetector        example data (no ML)
   │    ├─ MaskRCNNDetector    HTTP → original Flask model (Py3.6/TF1.15)
   │    └─ OpenCVDetector      classical CV fallback (no ML)
   ├─ services/DetectionService  primary + optional OpenCV fallback
   └─ services/FloorPlanBuilder  → normalized 3D JSON (Y-up, meters, bounds, ids)
```

The web layer depends only on `DetectorBase`, so detectors are swappable without
touching the endpoint. **All** normalization (pixel→world, Y flip, scaling,
ids, bounds, opening↔wall linking) lives in `FloorPlanBuilder` — detectors only
emit raw pixel-space detections.

---

## Detectores disponibles

Selected with the `DETECTOR` environment variable (`mock | maskrcnn | opencv`).

| Detector   | `DETECTOR` | Needs           | Detects            | When to use |
|------------|------------|-----------------|--------------------|-------------|
| **Mock**   | `mock`     | nothing         | example room       | Frontend/dev without GPU, weights or the legacy service. Deterministic. |
| **Mask R-CNN** | `maskrcnn` | the original Flask service running | walls, doors, windows | Real inference. Best quality. Delegates to the trained model over HTTP. |
| **OpenCV** | `opencv`   | nothing         | walls only         | GPU-free backup. Works on clean, moderate-size plans; **does not** detect doors/windows. |

### Mask R-CNN detector

The original model is pinned to Python 3.6 / TensorFlow 1.15 and cannot run
in this 3.10+ process, so `MaskRCNNDetector` calls the original Flask API over
HTTP (`LEGACY_API_URL`) and adapts its `points`/`classes` response. Start that
service separately (see the original repo), then set:

```env
DETECTOR=maskrcnn
LEGACY_API_URL=http://127.0.0.1:5000/
```

> The legacy model returns no per-detection score, so each detection is assigned
> `LEGACY_DEFAULT_CONFIDENCE`.

### OpenCV classical detector (backup)

Pure OpenCV pipeline (no deep learning): grayscale → Otsu threshold → morphology
→ `HoughLinesP` → merge collinear segments → drop short ones. It returns **walls
only**; doors and windows are left empty on purpose (not reliably detectable
without learning — we never fabricate data).

> **Note:** OpenCV is a *backup*. Quality depends on clean plans of moderate
> size; very large images should be resized first (the API already downscales
> anything above `MAX_IMAGE_DIMENSION`).

Tunable via env or per-request query params: `min_wall_length_px`,
`hough_threshold`, `hough_min_line_length`, `hough_max_line_gap`,
`merge_distance_px`.

### Automatic fallback

Set `FALLBACK_TO_OPENCV=true` to make the API retry with the OpenCV detector
when the primary (e.g. `maskrcnn`) **fails**, returns **0 walls**, or returns
walls whose **mean confidence is below `CONFIDENCE_THRESHOLD`**. The response
marks the real source used:

```json
"meta": { "model": "opencv-classic", "fallback_used": true }
```

---

## Setup

Requires Python 3.10+.

```bash
cd floorplan-api
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt        # runtime
pip install -r requirements-dev.txt    # + tests

cp .env.example .env                    # optional; defaults work out of the box
```

## Run

```bash
uvicorn app.main:app --reload          # http://127.0.0.1:8000
# or:  python -m app.main
```

Open the interactive docs at **http://127.0.0.1:8000/docs**.

## Tests

```bash
pytest
```

The suite passes with the **MockDetector** and the **OpenCVDetector** (no GPU,
weights or legacy service required).

---

## API

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

### `POST /detect`

`multipart/form-data` with a `file` field (PNG/JPG). Optional query params:

| Param | Default | Meaning |
|-------|---------|---------|
| `confidence_threshold` | `0.5` | Drop detections below this score. |
| `wall_height` | `2.7` | Wall extrusion height (m). |
| `default_wall_thickness` | `0.15` | Fallback wall thickness (m). |
| `pixels_per_meter` | `null` | If set → meters; if omitted → normalized 0..1. |
| `min_wall_length_px`, `hough_threshold`, `hough_min_line_length`, `hough_max_line_gap`, `merge_distance_px` | config | OpenCV tuning (only used by the OpenCV detector). |

#### Example (curl)

```bash
# Normalized output (no scale):
curl -X POST http://127.0.0.1:8000/detect \
  -F "file=@plan.png"

# Real-world meters, with a higher confidence threshold:
curl -X POST "http://127.0.0.1:8000/detect?pixels_per_meter=50&confidence_threshold=0.6" \
  -F "file=@plan.png"
```

#### Example response

```json
{
  "image": { "width": 1024, "height": 768, "unit": "meters", "pixels_per_meter": 50.0 },
  "scale": { "wall_height": 2.7, "default_wall_thickness": 0.15 },
  "walls": [
    {
      "id": "w1",
      "start": { "x": 0.0, "y": 0.0 },
      "end":   { "x": 5.0, "y": 0.0 },
      "thickness": 0.15,
      "height": 2.7,
      "confidence": 0.92
    }
  ],
  "doors": [
    { "id": "d1", "wall_id": "w1", "position": { "x": 2.5, "y": 0.0 }, "width": 0.9, "height": 2.1, "confidence": 0.88 }
  ],
  "windows": [
    { "id": "win1", "wall_id": "w1", "position": { "x": 4.0, "y": 0.0 }, "width": 1.2, "height": 1.1, "sill_height": 0.9, "confidence": 0.85 }
  ],
  "bounds": { "min_x": 0.0, "min_y": 0.0, "max_x": 5.0, "max_y": 4.0 },
  "meta": { "model": "maskrcnn-resnet101", "version": "1.0", "processing_ms": 320, "fallback_used": false }
}
```

#### Errors

Uniform body `{ "error": <code>, "detail": <message> }`:

| Status | `error` | Cause |
|--------|---------|-------|
| 400 | `invalid_image` | Empty or undecodable file. |
| 413 | `image_too_large` | Payload exceeds `MAX_IMAGE_SIZE_MB`. |
| 422 | `validation_error` | Missing `file` / bad query param. |
| 500 | `inference_error` | Detector failed (and no fallback succeeded). |

---

## Coordinate convention

- Origin at the **bottom-left**, **Y up** (world coords, not image coords).
- With `pixels_per_meter` → **meters**; otherwise geometry is **normalized 0..1**
  (divided by the longest side, aspect ratio preserved) and `unit: "normalized"`.
- Walls are **segments** (`start`/`end`): extrude a prism of `length =
  |end-start|`, `width = thickness`, `height = height`.

---

## Docker

```bash
docker compose up --build      # serves on http://127.0.0.1:8000
```

`./weights` is mounted read-only into the container. Configure the container via
a `.env` file (optional) or `environment:` overrides in `docker-compose.yml`.

---

## Configuration

See [`.env.example`](.env.example) for every variable and its default.
