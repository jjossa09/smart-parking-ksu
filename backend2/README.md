# KSU Smart Parking — Backend2

YOLOv8 + ByteTrack real-time parking occupancy service.  
Streams annotated video and live spot status to the React frontend over WebSockets.

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10 or 3.11 |
| pip | any recent |
| Git | any |

---

## Setup (first time only)

```powershell
# 1. Enter the backend2 directory
cd backend2

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
.\venv\Scripts\activate

# 4. Install all dependencies
pip install -r requirements.txt
```

> **Note:** The first run will download the YOLOv8 nano model (`yolov8n.pt`, ~6 MB) automatically.

---

## Running the Server

```powershell
# From the backend2 directory with venv active:
python run.py
```

The server starts at **http://localhost:8000** and automatically begins processing the demo video at startup.

To stop the server press `Ctrl + C`.

---

## Running the Frontend

```powershell
# From the frontend directory (separate terminal):
cd frontend
npm install   # first time only
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Key Files

```
backend2/
├── run.py                  # Entry point — starts uvicorn
├── requirements.txt        # Python dependencies
├── api/
│   └── main.py             # FastAPI app, WebSocket, adapter logic
├── scanner/
│   └── spot_scanner.py     # YOLOv8 + ByteTrack occupancy engine
├── data/
│   ├── polygons.json       # (generated) normalized polygon coords
│   ├── reference.jpg       # (generated) first video frame for editor
│   └── status.json         # (generated) latest spot status cache
└── yolov8n.pt              # YOLO model weights (auto-downloaded)
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server health check |
| GET | `/feed` | MJPEG live stream with polygon overlays |
| POST | `/stream/start` | Start scanner on a given video URL |
| POST | `/stream/stop` | Stop the scanner |
| GET | `/polygons` | Get current polygon definitions |
| PUT | `/polygons` | Update polygons without restarting |
| WS | `/ws` | WebSocket — pushes `update` JSON to the frontend |

### WebSocket payload example

```json
{
  "type": "update",
  "lotName": "KSU Main Campus",
  "totalSpots": 45,
  "availableSpots": 29,
  "occupiedSpots": 16,
  "frameImage": "data:image/jpeg;base64,/9j/...",
  "spots": [
    { "id": 1, "label": "1", "isOccupied": false, "polygon": [[319,130],[330,149],[286,184],[276,161]] },
    { "id": 2, "label": "2", "isOccupied": true,  "polygon": [[309,111],[318,129],[276,158],[269,138]] }
  ]
}
```

---

## Changing the Video Source

Edit the `AUTO_START_URL` constant in `api/main.py`:

```python
AUTO_START_URL = r"C:\path\to\your\video.mp4"
# or a live RTSP stream:
AUTO_START_URL = "rtsp://your-camera-ip/stream"
```

Or POST to `/stream/start` at runtime:

```http
POST http://localhost:8000/stream/start
Content-Type: application/json

{ "url": "rtsp://your-camera-ip/stream" }
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the `venv` |
| Black video / no feed | Check that the video or RTSP path is valid |
| Polygons misaligned | Confirm `spots.json` was annotated at 640×360 resolution |
| `lap` spam in logs | Run `pip install lapx` inside the `venv` to enable ByteTrack |
| Port already in use | Kill existing Python processes: `taskkill /IM python.exe /F` |
