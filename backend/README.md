# Smart Parking System - Backend

This is the Python-based AI Backend for the Smart Parking System. It utilizes computer vision to analyze frames (simulating a camera feed) and broadcast the real-time availability of parking spots.

## Technology Stack
- **Python**: Core programming language.
- **OpenCV (`cv2`)**: Used for all image processing, pixel counting, and computer vision tasks.
- **FastAPI**: A modern, high-performance web framework used to create the API server.
- **WebSockets**: For maintaining a continuous, bi-directional tunnel to the React Frontend, pushing updates instantly.
- **Uvicorn**: The ASGI web server implementation that runs the FastAPI application.

## Structure
- `detector.py`: The brain of the operation. This class maps out theoretical parking spot coordinates and implements the OpenCV algorithm to detect cars by converting frames to grayscale, applying blurring, adaptive thresholding, and finally counting non-zero edges inside the defined boundaries.
- `server.py`: The networking layer. It initializes the FastAPI app, opens an endpoint at `/ws`, tracks connected browser clients, and runs a background `asyncio` loop that continually queries `detector.py` and pushes the updated JSON to anyone listening.

## How it Works (The Process)
1. **Spot Coordinates**: The `detector.py` defines regions of interest (x, y, width, height) representing spots (e.g., A1-A10, B1-B10) mapped over the parking lot image scale.
2. **OpenCV Pipeline**: 
   When `process_frame` is called (currently mocked due to lack of a physical camera for the prototype):
   - The image is converted to completely flat grayscale.
   - It is aggressively blurred to remove visual noise (like shadows or debris).
   - An adaptive threshold turns it into a stark black canvas with pure white lines outlining all distinct objects (like the glossy edges and windows of a car).
   - It iterates through each "Region of Interest" (parking spot).
   - If a spot is full of white edges (pixel count > threshold), it implies a car's structure is interfering with the flat pavement. `isOccupied = True`.
3. **Broadcasting**: `server.py` awaits the result of this layout. Once verified, it wraps the entire lot's status array into a JSON object and fires it through the WebSocket to the React app.

## Running Locally

1. Create a virtual environment and step into it:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # Windows
   source venv/bin/activate  # Mac/Linux
   ```
2. Install the necessary dependencies:
   ```bash
   pip install fastapi uvicorn websockets opencv-python numpy
   ```
3. Boot up the real-time server:
   ```bash
   python server.py
   ```
4. Leave this terminal open. Switch to the `frontend` instructions to boot the dashboard.
