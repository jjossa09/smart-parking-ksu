import cv2
import numpy as np
import logging
import json
import os
import asyncio

logger = logging.getLogger(__name__)

class ParkingDetector:
    def __init__(self, video_path="carPark.mp4", mask_path="mask.json"):
        self.video_path = video_path
        self.mask_path = mask_path
        self.cap = cv2.VideoCapture(self.video_path)
        
        # We process fewer frames to save CPU, e.g., only every 30th frame (1 per sec)
        self.frame_skip = 15
        self.frame_count = 0
        
        # Width and height of the boxes we defined in create_mask.py
        self.box_width = 100
        self.box_height = 150
        
        # Load the mask definitions
        self.spot_definitions = []
        if os.path.exists(self.mask_path):
            with open(self.mask_path, 'r') as f:
                self.spot_definitions = json.load(f)
            logger.info(f"Loaded {len(self.spot_definitions)} spots from mask.")
        else:
            logger.warning("No mask.json found! Using fallback locations.")
            # Fallback to the hardcoded coordinates from earlier if mask isn't built yet
            for i in range(10):
                self.spot_definitions.append({"id": f"A{i+1}", "coords": (50 + (i * 110), 50, 100, 150)})
                self.spot_definitions.append({"id": f"B{i+1}", "coords": (50 + (i * 110), 250, 100, 150)})

        # Initialize the state dictionary for the React frontend
        self.spots_state = {spot["id"]: {"id": spot["id"], "isOccupied": False} for spot in self.spot_definitions}
        
    def process_frame(self):
        """Reads the next frame, processes it, and updates spot occupancy."""
        success, frame = self.cap.read()
        
        # If video ends, loop it for the demonstration
        if not success:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.cap.read()
            if not success:
                logger.error("Could not read video file.")
                return list(self.spots_state.values())

        # 1. Grayscale & Blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        
        # 2. Adaptive Thresholding: creates harsh black/white outlines of cars
        img_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 25, 16)
        
        # 3. Clean up noise and thicken edges
        img_median = cv2.medianBlur(img_thresh, 5)
        kernel = np.ones((3, 3), np.uint8)
        img_dilate = cv2.dilate(img_median, kernel, iterations=1)

        # 4. Check each spot
        for spot in self.spot_definitions:
            x, y, w, h = spot["coords"]
            
            # Crop the highly contrasted image to just this one bounding box
            spot_crop = img_dilate[y:y+h, x:x+w]
            
            # Count the number of pure white pixels (edges/reflections)
            count = cv2.countNonZero(spot_crop)
            
            # Threshold: In our generic synthetic video, an empty gray block has ~0 edges.
            # A drawn "car" with a windshield has hundreds of edges.
            # In real life, 900 might be a good starting point.
            if count < 500:
                self.spots_state[spot["id"]]["isOccupied"] = False
            else:
                self.spots_state[spot["id"]]["isOccupied"] = True

        return list(self.spots_state.values())

    def get_current_spots(self):
        """Called by the API server to get the latest state array."""
        # Instead of dummy simulation, we actually run the OpenCV frame analysis
        # Only analyze every Nth frame to save massive CPU load for WebSockets
        self.frame_count += 1
        if self.frame_count >= self.frame_skip:
            self.frame_count = 0
            return self.process_frame()
            
        # If we skipped processing this specific millisecond, return the cached state
        return list(self.spots_state.values())

    def release(self):
        """Clean up video feed resources."""
        if hasattr(self, 'cap'):
            self.cap.release()
