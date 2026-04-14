# Smart Parking KSU

Real-time parking lot availability detection using YOLOv8. Built by **JESDAF** for the Bank of America hackathon, Kennesaw State University.

Open a browser, pick a video, watch parking spots light up green (empty) or red (occupied) in real time. Scan a QR code on your phone and it works there too.

## Repo layout

- **`v1/`** — Phase 1 work (SVM + Random Forest approach, archived). Not used anymore, kept for reference.
- **`v2/`** — Phase 2 work, active. This is what you run. **All instructions below assume you're working in `v2/`.**

## Quickstart (first time setup)

You need **Python 3.10 or newer** and **Git**. That's it. Everything else installs automatically.

```powershell
# 1. Clone the repo
git clone https://github.com/jjossa09/smart-parking-ksu.git
cd smart-parking-ksu\v2

# 2. Create a virtual environment (isolates our dependencies)
python -m venv .venv-yolo

# 3. Activate it (Windows PowerShell)
.venv-yolo\Scripts\activate

# 4. Install everything
pip install -r backend\requirements.txt

# 5. Start the server
uvicorn backend.app:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser. You should see the Smart Parking KSU menu. Click Select Video → Detect Spots → watch it run.

**On Mac/Linux**, step 3 is `source .venv-yolo/bin/activate`. Everything else is identical.

## Daily use (after first setup)

```powershell
cd smart-parking-ksu\v2
.venv-yolo\Scripts\activate
uvicorn backend.app:app --reload --port 8000
```

Three commands. Memorize these.

## Adding a new video to the demo

1. Drop the video file into `v2\ml\yolo\videos\` (e.g. `mylot.mp4`)
2. Annotate the parking spots once:
```powershell
   python -m ml.yolo.annotate_spots --video ml\yolo\videos\mylot.mp4
```
   Click 4 corners per spot, press `n` after each spot, press `s` when done to save.
3. Reload the browser. The new video appears in the Select Video dropdown automatically.

Full KSU integration walkthrough is in [`v2/docs/ksu_data_integration.md`](v2/docs/ksu_data_integration.md).

## How it works (30 seconds)

Four stages, running on every processed frame:

1. **YOLO detects cars** in the frame (bounding boxes + confidence scores)
2. **Spot polygons** are loaded from a JSON file you annotate once per camera
3. **Assignment** — each car claims its single best-matching spot via IoU overlap
4. **Smoothing** — a state only flips after 5 consecutive frames agree, so the UI doesn't flicker

The backend runs all of this in a background thread and exposes JSON at `/parking-status`. The frontend polls once per second and updates the live view.

For the full technical breakdown: [`v2/docs/architecture.md`](v2/docs/architecture.md).

## Troubleshooting

- **`uvicorn` command not found** → the venv isn't activated. Run `.venv-yolo\Scripts\activate` first.
- **Browser shows raw JSON instead of the UI** → you're hitting an old cached version. Hard refresh with `Ctrl+Shift+R`, or try incognito mode.
- **"No videos available" in the dropdown** → a video needs both a file in `v2/ml/yolo/videos/` AND a matching `spots_*.json` in `v2/ml/yolo/spots/`. Run `annotate_spots.py` to create the spots file.
- **Server crashes with a Python error** → copy the full traceback and send it to whoever's maintaining the code. Most likely a missing dependency — `pip install -r backend\requirements.txt` again.

## For teammates extending the project

- **Change model** (e.g. after a new fine-tune): edit one line in `v2/ml/yolo/config.py` → `MODEL_WEIGHTS = ...`
- **Change detection thresholds**: same file, all commented inline
- **Add UI screens**: edit `v2/frontend/index.html`, `style.css`, `app.js`. Vanilla JS, no build step.
- **Add API endpoints**: edit `v2/backend/app.py`. FastAPI, hot-reloads on save.

## Project docs

All technical docs live in [`v2/docs/`](v2/docs/):

- **`architecture.md`** — the three-swappable-pieces design story
- **`ksu_data_integration.md`** — the runbook for when real KSU footage arrives
- **`model_evaluation.md`** — why we shipped COCO and not the fine-tuned PKLot model

## Team

JESDAF — Kennesaw State University
