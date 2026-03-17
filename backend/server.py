import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from detector import ParkingDetector

# ==========================================
# 1. SERVER CONFIGURATION & SETUP
# ==========================================
# Configure basic logging so we can see what the server is doing in the terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the main FastAPI application instance. This is the core of our backend.
app = FastAPI(title="Smart Parking KSU API")

# Cross-Origin Resource Sharing (CORS) Configuration.
# This is crucial! It allows our React frontend (running on a different port like 5173/5174) 
# to talk to this Python backend without being blocked by browser security policies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For the hackathon, we allow all ports. In production, this would be the exact frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. WEBSOCKET CONNECTION MANAGER
# ==========================================
class ConnectionManager:
    """
    Manages all active WebSocket connections. If multiple users (or judges) load the 
    React dashboard at the same time, this class keeps track of all their connections 
    so the server can broadcast the parking data to everyone simultaneously.
    """
    def __init__(self):
        # A list to store the active tunnels to connected browsers
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        # Accept the incoming connection request from a new browser tab
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        # Remove the browser from the list if the user closes the tab
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        # Send a single JSON message to every single browser currently looking at the dashboard
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")

# Instantiate the manager so it is ready to handle traffic
manager = ConnectionManager()

# ==========================================
# 3. API ENDPOINTS
# ==========================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    The endpoint the React frontend connects to (e.g., ws://localhost:8000/ws).
    Unlike standard HTTP which closes instantly, this function stays "open" in an infinite loop,
    creating a persistent, real-time tunnel between the browser and the AI.
    """
    await manager.connect(websocket)
    try:
        # Keep the connection alive infinitely
        while True:
            # We don't inherently expect the browser to send us data back for this project,
            # but we must 'listen' to keep the WebSocket protocol satisfied.
            data = await websocket.receive_text()
            logger.info(f"Received message from client: {data}")
    except WebSocketDisconnect:
        # If the browser throws an error or closes, handle it gracefully
        manager.disconnect(websocket)

# ==========================================
# 4. BACKGROUND AI PROCESS
# ==========================================
async def broadcast_parking_status():
    """
    The main infinite loop that drives the entire application. 
    It runs in the background independently of any web requests. It constantly asks the 
    OpenCV computer vision script what it sees, and then pushes that data to the frontend.
    """
    # Wait a few seconds for the FastAPI server to fully boot before starting heavy AI tasks
    await asyncio.sleep(2)
    
    logger.info("Initializing OpenCV Engine...")
    # Instantiate our custom Computer Vision class which reads the video feed
    detector = ParkingDetector()
    
    logger.info("Starting parking detection loop...")
    
    # Infinite loop: Read Video -> Analyze Spots -> Send to React -> Repeat
    while True:
        try:
            # Step A: Ask OpenCV for the latest frame's array of parking spots tracking
            current_spots = detector.get_current_spots()
            
            # Step B: If anyone is looking at the dashboard, send them the data
            if len(manager.active_connections) > 0:
                # We package the Python dictionary into a JSON string format that React understands
                await manager.broadcast(json.dumps({
                    "type": "update",
                    "spots": current_spots
                }))
            
            # Step C: Pause for 1 second. We don't need to analyze 60 frames a second for parked cars.
            # Processing 1 frame a second saves massive amounts of CPU power.
            await asyncio.sleep(1.0)
            
        except Exception as e:
            logger.error(f"Error in detection loop: {e}")
            await asyncio.sleep(5)  # If it crashes, wait 5 seconds and try again (Backoff)

# ==========================================
# 5. SERVER LAUNCH
# ==========================================
@app.on_event("startup")
async def startup_event():
    """
    When `python server.py` is run, FastAPI calls this function right as it boots.
    We tell it to spawn our background AI loop asynchronously so it doesn't freeze the web server.
    """
    asyncio.create_task(broadcast_parking_status())

# Standard Python boilerplate to execute the server if run directly from the terminal
if __name__ == "__main__":
    import uvicorn
    # uvicorn is the high-performance ASGI server that runs FastAPI applications
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
