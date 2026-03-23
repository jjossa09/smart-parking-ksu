# Architecture — KSU Smart Parking

This document explains how `backend2` and the React `frontend` work together end-to-end.

---

## System Overview

```
Video Source (mp4 / RTSP)
        │
        ▼
┌──────────────────────────────────┐
│  scanner/spot_scanner.py         │
│  ─────────────────────────────── │
│  • OpenCV reads frames           │
│  • YOLOv8 detects vehicles       │
│  • ByteTrack assigns stable IDs  │
│  • Polygon overlap = spot taken  │
└───────────┬──────────────────────┘
            │ frame_callback(annotated_frame, statuses, polys, owner_map)
            ▼
┌──────────────────────────────────┐
│  api/main.py  (FastAPI)          │
│  ─────────────────────────────── │
│  • Encodes frame as Base64 JPEG  │
│  • Scales polygons to frame size │
│  • Builds WebSocket JSON payload │
│  • Broadcasts to all /ws clients │
└───────────┬──────────────────────┘
            │ WebSocket JSON
            ▼
┌──────────────────────────────────┐
│  frontend/src/App.jsx  (React)   │
│  ─────────────────────────────── │
│  • Receives WS message           │
│  • Renders live JPEG as <img>    │
│  • Draws SVG polygons on top     │
│  • Displays counts in stat cards │
└──────────────────────────────────┘
```

---

## Component Details

### 1. `scanner/spot_scanner.py`

**Purpose:** The AI engine. Runs entirely in a background thread to avoid blocking the async server.

**Key concepts:**

- **Polygon normalization** — `spots.json` stores raw pixel coordinates drawn on a 640×360 canvas. `_build_raw()` converts these to 0–1 floats. `_scale(w, h)` then multiplies by the live frame's actual dimensions so YOLO's bounding boxes and the polygon regions are in the same coordinate space for overlap testing.
- **YOLOv8** — runs at `imgsz=1280` for high accuracy on small/far cars.  
- **ByteTrack** (`persist=True`) — assigns a stable ID to each car across frames so it doesn't flicker.
- **Permanent IDs (P1, P2, …)** — a second ID layer that survives brief occlusion gaps. A `REMATCH_PX` radius re-links a reappearing car to its last permanent ID rather than assigning a new one.
- **N_CONFIRM=3** — a spot status only changes after 3 consecutive frames agree, eliminating transient false positives.
- **Row guard** — if a row is flagged as having more taken spots than detected cars, excess `taken` flags are cleared.
- **`frame_callback`** — once per frame the scanner calls back into `main.py` with the annotated frame, the `statuses` dict, `_polys` (pixel-space polygons), and `owner_map` (car → spot). This is the **adapter interface** between the scanner and the API.

### 2. `api/main.py`

**Purpose:** The FastAPI adapter layer. Translates between the scanner's internal model and the frontend's WebSocket contract.

**Key functions:**

| Function | Role |
|---|---|
| `convert_repo_spots_to_scanner_spots()` | Reads `spots.json`, converts raw pixel coords to Spot objects with normalized 0–1 `pts` |
| `set_latest_frame()` | Called by the scanner every frame — encodes JPEG, scales polygons back to pixel space, broadcasts JSON via WebSocket |
| `ConnectionManager` | Manages the list of active WebSocket clients and fan-out broadcasts |
| `/stream/start` | Loads polygons, instantiates `SpotScanner`, starts it as an async task |
| `/ws` | WebSocket endpoint — the React app connects here on load |
| `/feed` | MJPEG endpoint — provides annotated video to any direct browser tab |

**The adapter pattern:**  
`main.py` owns the API contract. The scanner only handles CV/AI logic. Neither layer needs to know the internals of the other — they communicate purely through the `frame_callback` signature:

```python
frame_callback(
    annotated_frame: np.ndarray,   # BGR frame with overlays drawn by scanner
    statuses: dict[str, str],      # {"1": "open", "2": "taken", ...}
    polys: dict[str, np.ndarray],  # normalized 0-1 polygon coords per spot
    owner_map: dict[str, str],     # {"P1": "4", ...} car → spot
)
```

**Polygon coordinate flow:**

```
spots.json  (pixels, 640×360 canvas)
    │ ÷ 640, ÷ 360
    ▼
Spot.pts  (normalized 0.0–1.0)
    │  scanner._build_raw()  →  scanner._scale(fw, fh)
    ▼
_polys  (pixels, actual video frame resolution — used by YOLO overlap test)
    │  × [frame_resized.w, frame_resized.h]
    ▼
payload["spots"][n]["polygon"]  (pixels, matches JPEG image dimensions)
    │  SVG viewBox = "0 0 imgDim.w imgDim.h"
    ▼
<polygon points="..."> on frontend
```

### 3. `frontend/src/App.jsx`

**Purpose:** The live dashboard UI.

- Connects to `ws://localhost:8000/ws` on startup.
- On each WebSocket `update` message it updates 3 state values: `lotData` (counts + spot list), `imgDim` (image natural dimensions from `handleImageLoad`).
- Renders a `<img>` tag with the Base64 JPEG as `src`.
- Overlays an absolutely-positioned `<svg>` with the same `width`/`height` and a `viewBox` matching the JPEG's natural resolution so polygon coordinates map pixel-for-pixel.
- Each `<polygon>` is colored green (available) or red (occupied) via CSS class.

---

## Data File Reference

| File | Created by | Purpose |
|------|-----------|---------|
| `scripts/annotation/spots.json` | Human annotations | Source of truth for parking spot polygons |
| `backend2/data/polygons.json` | `main.py` at startup | Normalized copy (0–1) used by the scanner |
| `backend2/data/reference.jpg` | Scanner on first frame | Reference image so annotations match the video |
| `backend2/data/status.json` | `main.py` per frame | Disk cache of latest spot statuses |
| `backend2/yolov8n.pt` | Auto-downloaded | YOLO model weights |
