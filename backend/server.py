import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from detector import ParkingDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Parking KSU API")

# Allow the React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Keep connection open
        while True:
            # We don't really expect messages from the client right now
            data = await websocket.receive_text()
            logger.info(f"Received message from client: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def broadcast_parking_status():
    """Background task to continually process computer vision and send updates."""
    # Give the server a moment to start before beginning detection loop
    await asyncio.sleep(2)
    
    logger.info("Initializing OpenCV Engine...")
    detector = ParkingDetector()
    
    logger.info("Starting parking detection loop...")
    while True:
        try:
            # Ask the OpenCV detector for the current state of all spots
            current_spots = detector.get_current_spots()
            
            # Broadcast the state to all connected React clients
            if len(manager.active_connections) > 0:
                await manager.broadcast(json.dumps({
                    "type": "update",
                    "spots": current_spots
                }))
            
            # Send updates every second
            await asyncio.sleep(1.0)
            
        except Exception as e:
            logger.error(f"Error in detection loop: {e}")
            await asyncio.sleep(5)  # Backoff on error

@app.on_event("startup")
async def startup_event():
    # Start the background broadcasting task when the FastAPI server starts
    asyncio.create_task(broadcast_parking_status())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
