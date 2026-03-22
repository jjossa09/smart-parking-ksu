# System Architecture & Documentation

This document explains the core differences between the two backend approaches, what each file does, and how they seamlessly interact with the React frontend to power the dashboard.

---

## 1. The Tale of Two Backends

### Original `backend` (Image Classification)
- **The Logic**: It uses a **Random Forest Classifier** (`predict_spot.py`). For every single frame of the video, it uses OpenCV to physically cut out 45 tiny mini-images (crops) based on the coordinates in `spots.json`. It flattens these tiny images into pixel arrays and feeds them to the ML model.
- **Why it struggled**: The model was trained rigidly on 150x150 rectangular patches of cars. The parking lot video has complex, skewed, trapezoidal polygons. When we masked the backgrounds of these polygons in black, the model saw shapes it had never been trained on and confidently guessed "0" (Empty) for occupied cars.

### New `backend2` (YOLOv8 Object Tracking)
- **The Logic**: It uses a state-of-the-art **Neural Network (YOLOv8 Small)**. Instead of slicing the frame 45 times, it looks at the entire frame *once*. It draws bounding boxes around anything that looks like a car, truck, or bus. 
- **Tracking (ByteTrack)**: It assigns a unique ID to every car (e.g., Car #205). As the car moves, the ID stays the same. 
- **Spot Assignment**: It calculates the mathematical center of the car's bounding box. If that center is hovering inside one of your 45 `spots.json` polygons, it marks the spot as `Taken`. If a car leaves, the tracker waits 4 seconds (`RELEASE_SECONDS`) before marking it empty, completely eliminating flickering!

---

## 2. File & Directory Breakdown

### The Artificial Intelligence (`backend2/scanner/spot_scanner.py`)
This is the heavy lifter. 
1. It opens the `demo-parking.mp4` video.
2. It mathematically scales your `spots.json` coordinates (drawn on a 640x360 frame) to make sure they align perfectly with the raw video pixels.
3. Every frame, it runs the `ultralytics` YOLO model.
4. It compares the `(x, y)` location of every tracked vehicle against the 45 polygons using `cv2.pointPolygonTest` (a native C++ geometry function).
5. It spits out a `statuses` dictionary (e.g., `{"1": "taken", "2": "open", "3": "taken"}`).

### The Web Server (`backend2/api/main.py`)
This is the bridge between Python and Javascript.
1. It runs a `FastAPI` instance.
2. It hosts a `WebSocket` endpoint at `ws://localhost:8000/ws`.
3. In `set_latest_frame()`, it grabs the raw image from the AI, deeply compresses it to Base64 to save bandwidth, bundles it into a payload alongside the `statuses` dictionaries, and blasts it to the React frontend 20 times a second.

### The Source of Truth (`scripts/annotation/spots.json`)
The absolute master geometric configuration. `backend2` reaches over and reads this exact file natively. If you use your `annotate_spots.py` script to draw a 46th polygon, `backend2` reads it on boot and instantly tracks it without any code changes.

### The UI Dashboard (`frontend/`)
- **`src/App.jsx`**: The React entry point.
  - It connects to the `ws://localhost:8000/ws` WebSocket.
  - For every message it receives computationally, it updates its React *State*.
  - It renders the Base64 background natively as a standard `<img>`.
  - **The SVG Magic**: Instead of python burning text onto the video, React uses an invisible `<svg>` ViewBox stacked directly over the image. It uses `.map()` to loop through the 45 coordinates sent by the backend and natively draws 45 separate HTML `<polygon>` elements perfectly tracking the cars.
- **`src/App.css`**: It uses CSS ternary logic so that if `isOccupied` is true, the SVG polygon glows Translucent Red, otherwise Translucent Green. It also powers the hover tooltips seamlessly.
