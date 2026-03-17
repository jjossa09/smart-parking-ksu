
import cv2
import numpy as np
import logging
import json
import os
import asyncio

logger = logging.getLogger(__name__)

# ==========================================
# OPEN CV AI PARKING DETECTION ENGINE
# ==========================================
class ParkingDetector:
    """
    This class handles the core Artificial Intelligence of the project using OpenCV.
    Rather than using a heavy, slow deep-learning model (like YOLO), this uses classical 
    Computer Vision math (Thresholding, Blurring, Dilating) to detect cars in real-time.
    This approach is wildly faster, requires less compute, and runs seamlessly on web servers.
    """
    def __init__(self, video_path='carPark.mp4', mask_path='mask.json'):
        self.video_path = video_path
        self.mask_path = mask_path
        
        # Load the predefined parking spot regions (the bounding boxes drawn in the mask tool)
        self.spot_definitions = []
        if os.path.exists(self.mask_path):
            with open(self.mask_path, 'r') as f:
                self.spot_definitions = json.load(f)
            logger.info(f"Loaded {len(self.spot_definitions)} spots from mask.")
        
        # Fallback to hardcoded coordinates if the user hasn't created a mask.json yet
        # These map perfectly to the React frontend UI coordinates by default.
        if len(self.spot_definitions) == 0:
            logger.warning("No mask.json found! Using fallback locations.")
            # Lot A (Left side of the screen)
            for i in range(10):
                self.spot_definitions.append({"id": f"A{i+1}", "coords": (50 + (i * 110), 50, 100, 150)})
            # Lot B (Right side of the screen)
            for i in range(10):
                self.spot_definitions.append({"id": f"B{i+1}", "coords": (50 + (i * 110), 250, 100, 150)})

        # Initialize the state dictionary that will be sent to the React frontend
        self.spots_state = {spot['id']: False for spot in self.spot_definitions}
        
        # Open the simulated CCTV video stream feed
        self.cap = cv2.VideoCapture(self.video_path)
        
        # Performance optimization: We only process every Nth frame, skipping the rest.
        # This allows 60fps video to process with the CPU load of a 5fps video.
        self.frame_skip = 10
        self.frame_count = 0

    def process_frame(self):
        """
        Grabs the next frame from the CCTV feed and runs it through the detection pipeline.
        Returns a dictionary mapping Spot ID -> Boolean (True if occupied, False if empty).
        """
        # Read the next image frame from the video stream
        success, img = self.cap.read()
        
        # If the video ends during the hackathon demo, loop it back to the beginning seamlessly
        if not success or self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, img = self.cap.read()
            if not success:
                logger.error("Failed to read video stream even after looping.")
                return self.spots_state
        
        # Step 1: Grayscale Conversion
        # Colors aren't needed to find shapes. Grayscale reduces data size by 3x (RGB -> BW)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Gaussian Blur
        # Smooths the image to remove 'noise' (leaves, shadows, artifacts) that might look like edges
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        
        # Step 3: Adaptive Thresholding
        # This is the magic. It converts the image into pure black and white pixels based on local contrast.
        # This highlights the edges/contours of cars dramatically against the flat asphalt.
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 25, 16)
        
        # Step 4: Median Blur
        # Removes isolated "salt and pepper" noise pixels that survived the thresholding
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        
        # Step 5: Dilation
        # Thickens the remaining white pixels (the edges of the cars). This makes them easier to count.
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

        # Step 6: Count pixels in each parking spot
        for spot in self.spot_definitions:
            spot_id = spot['id']
            x, y, w, h = spot['coords']
            
            # Crop the heavily processed, mostly black-and-white image to just this one parking spot
            imgCrop = imgDilate[y:y+h, x:x+w]
            
            # Count how many white pixels (car edges) exist inside this cropped box
            count = cv2.countNonZero(imgCrop)
            
            # The Threshold Logic:
            # If there are fewer than 900 white edge pixels, the spot is empty (just flat gray asphalt).
            # If there are more than 900 white edge pixels, a complex shape (a car) is occupying it.
            # *Note: 900 is tuned specifically for the camera angle in `carPark.mp4`.
            if count < 900:
                self.spots_state[spot_id] = False # Available
            else:
                self.spots_state[spot_id] = True  # Occupied

        return self.spots_state

    def get_current_spots(self):
        """
        This is the public method called by `server.py`'s background loop every 1 second.
        It forcefully processes frames to stay synced with time, then returns the React-friendly API payload.
        """
        # Read frames until we hit our frame_skip threshold (which simulates real-time passing)
        for _ in range(self.frame_skip):
            self.process_frame()
            
        # Transform our internal dictionary state into the Array format the React frontend expects
        formatted_spots = []
        for spot_id, is_occupied in self.spots_state.items():
            formatted_spots.append({
                "id": spot_id,
                "isOccupied": is_occupied
            })
            
        return formatted_spots

    def release(self):
        """Clean up video feed resources."""
        if hasattr(self, 'cap'):
            self.cap.release()

