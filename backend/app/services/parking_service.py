import os
import json
import asyncio
import logging
from datetime import datetime, timezone
import cv2

# Import the inference service
from .inference_service import InferenceService

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
ROOT_DIR = os.path.dirname(BACKEND_DIR)

SPOTS_PATH = os.path.join(ROOT_DIR, "scripts", "annotation", "spots.json")
MODEL_PATH = os.path.join(ROOT_DIR, "ml", "random_forest_parking_model.pkl")
VIDEO_PATH = os.path.join(ROOT_DIR, "scripts", "demo", "demo-parking.mp4")

import base64

class ParkingService:
    def __init__(self, connection_manager):
        self.manager = connection_manager
        
        # Load the manually annotated spots geometry
        self.spots_data = []
        if os.path.exists(SPOTS_PATH):
            with open(SPOTS_PATH, "r") as f:
                self.spots_data = json.load(f)
            logger.info(f"Loaded {len(self.spots_data)} spots from {SPOTS_PATH}")
        else:
            logger.error(f"Could not find spots.json at {SPOTS_PATH}")

        # Initialize ML inference engine
        try:
            self.inference = InferenceService(MODEL_PATH)
        except Exception as e:
            logger.error(f"Failed to load ML Model: {e}")
            self.inference = None

    async def detection_loop(self):
        """Background task running continuously to analyze the frame."""
        logger.info("Starting ML detection loop...")
        await asyncio.sleep(2)  # brief startup delay
        
        # Initialize the global video capture stream
        if not os.path.exists(VIDEO_PATH):
            logger.error(f"Waiting for video at {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)

        while True:
            try:
                # If no clients are connected, wait gracefully
                if not self.manager.active_connections:
                    await asyncio.sleep(2.0)
                    continue

                if not self.inference or not self.spots_data:
                    logger.warning("Missing inference model or spots.json")
                    await asyncio.sleep(5.0)
                    continue

                # Read current frame from Video
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video file reached. Looping back to the beginning...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # small wait before retrieving the next frame
                    await asyncio.sleep(0.1)
                    continue

                # Run model to predict the occupancy of all spots in the lot
                predictions = self.inference.predict_occupancy(frame, self.spots_data)

                total_spots = len(predictions)
                occupied_spots = sum(1 for p in predictions if p["isOccupied"])
                available_spots = total_spots - occupied_spots
                
                # Compress, scale, and encode the exact video frame as a Base64 JPEG to send to frontend
                height, width = frame.shape[:2]
                max_width = 1280
                if width > max_width:
                    scale = max_width / width
                    frame = cv2.resize(frame, (max_width, int(height * scale)))
                    
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                base64_img = base64.b64encode(buffer).decode('utf-8')
                frame_uri = f"data:image/jpeg;base64,{base64_img}"

                # Format payload for frontend exactly as designed
                payload = {
                    "type": "update",
                    "lotName": "Live Video Feed",
                    "frameImage": frame_uri, 
                    "totalSpots": total_spots,
                    "availableSpots": available_spots,
                    "occupiedSpots": occupied_spots,
                    "spots": [
                        {
                            "id": p["id"],
                            "label": p["id"],
                            "isOccupied": p["isOccupied"],
                            "polygon": p["polygon"]
                        } for p in predictions
                    ],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

                # Broadcast to connected React clients in perfectly synchronized real-time!
                await self.manager.broadcast(json.dumps(payload))
                
                # Throttled to ~10 FPS for a smooth video UI without crashing WebSockets
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in detection loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)
