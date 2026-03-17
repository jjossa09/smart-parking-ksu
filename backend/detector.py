import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ParkingDetector:
    def __init__(self):
        # In a real scenario, these coordinates come from a configuration tool or JSON file
        # For our React app, we defined 20 spots (A1-A10, B1-B10). We'll map them here.
        # Format: {"id": "spot_name", "coords": (x, y, w, h)}
        self.spot_definitions = [
            {"id": "A1", "coords": (50, 50, 100, 150)},
            {"id": "A2", "coords": (160, 50, 100, 150)},
            {"id": "A3", "coords": (270, 50, 100, 150)},
            {"id": "A4", "coords": (380, 50, 100, 150)},
            {"id": "A5", "coords": (490, 50, 100, 150)},
            {"id": "A6", "coords": (600, 50, 100, 150)},
            {"id": "A7", "coords": (710, 50, 100, 150)},
            {"id": "A8", "coords": (820, 50, 100, 150)},
            {"id": "A9", "coords": (930, 50, 100, 150)},
            {"id": "A10", "coords": (1040, 50, 100, 150)},
            
            {"id": "B1", "coords": (50, 250, 100, 150)},
            {"id": "B2", "coords": (160, 250, 100, 150)},
            {"id": "B3", "coords": (270, 250, 100, 150)},
            {"id": "B4", "coords": (380, 250, 100, 150)},
            {"id": "B5", "coords": (490, 250, 100, 150)},
            {"id": "B6", "coords": (600, 250, 100, 150)},
            {"id": "B7", "coords": (710, 250, 100, 150)},
            {"id": "B8", "coords": (820, 250, 100, 150)},
            {"id": "B9", "coords": (930, 250, 100, 150)},
            {"id": "B10", "coords": (1040, 250, 100, 150)}
        ]

        # The image/video we are analyzing. Since we don't have a real feed yet for the hackathon,
        # we will simulate it by reading a dummy image if we need to, but for now we'll just mock 
        # the status toggling like we did in React, but from the backend. 
        # When you get a real camera feed or video, we replace `simulate_detection` with `process_frame`.
        self.spots_state = {spot["id"]: {"id": spot["id"], "isOccupied": False} for spot in self.spot_definitions}
        
    def process_frame(self, frame):
        """
        This is the actual OpenCV logic you requested an explanation for.
        It takes a video frame, converts it, and checks each defined parking spot.
        """
        # 1. Convert to Grayscale (removes color noise, focuses on intensity)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Add Gaussian Blur (smooths the image, removes small random noise)
        blur = cv2.GaussianBlur(gray, (3, 3), 1)
        
        # 3. Apply Adaptive Threshold (turns image into pure black & white edges)
        # This highlights the sharp edges of cars vs the flat surface of empty pavement
        img_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 25, 16)
        
        # 4. Optional: Median blur to remove salt-and-pepper noise after thresholding
        img_median = cv2.medianBlur(img_thresh, 5)
        
        # 5. Dilation (thickens the white edges slightly making them easier to count)
        kernel = np.ones((3, 3), np.uint8)
        img_dilate = cv2.dilate(img_median, kernel, iterations=1)

        # 6. Analyze each defined spot
        for spot in self.spot_definitions:
            x, y, w, h = spot["coords"]
            
            # Crop the processed image to just this one parking spot
            spot_crop = img_dilate[y:y+h, x:x+w]
            
            # Count the non-zero (white) pixels inside this cropped area
            count = cv2.countNonZero(spot_crop)
            
            # The Magic Number: If there are many edges (white pixels), there is a car.
            # If there are few edges, it's empty pavement.
            # Note: 900 is an example threshold. You must calibrate this based on your camera angle!
            if count < 900:
                self.spots_state[spot["id"]]["isOccupied"] = False
            else:
                self.spots_state[spot["id"]]["isOccupied"] = True

        return self.get_current_spots()

    def get_current_spots(self):
        """Returns the array of spots formatted for the React frontend."""
        # For now, we simulate the detection since we lack a real OpenCV video source input.
        # Pick a random spot to toggle
        import random
        random_spot = random.choice(self.spot_definitions)["id"]
        current_state = self.spots_state[random_spot]["isOccupied"]
        self.spots_state[random_spot]["isOccupied"] = not current_state
        return list(self.spots_state.values())
