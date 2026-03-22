import os
import sys
import json
import cv2
import numpy as np
import logging

# Set up paths to reach the ML directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
ML_DIR = os.path.join(ROOT_DIR, "ml")

# Add ML directory to path so we can import predict_spot
if ML_DIR not in sys.path:
    sys.path.append(ML_DIR)

from predict_spot import load_model, predict_spot

logger = logging.getLogger(__name__)

class InferenceService:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        self.model = load_model(model_path)
        logger.info(f"Loaded parking ML model from {model_path}")

    def crop_polygon_region(self, frame, polygon_points):
        """Extract a bounding box around the polygon and mask it out."""
        pts = np.array(polygon_points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        crop = frame[y:y+h, x:x+w].copy()
        
        # We removed the artificial black polygon mask because it vastly reduces 
        # ML accuracy by inserting false feature data.
        return crop, pts

    def predict_occupancy(self, frame, spots_data):
        predictions = []
        for spot in spots_data:
            spot_id = spot["id"]
            polygon_points = spot["points"]
            try:
                masked_crop, _ = self.crop_polygon_region(frame, polygon_points)
                # predict_spot returns 1 for occupied, 0 for empty
                pred = predict_spot(masked_crop, self.model)
                is_occupied = (pred == 1)
            except Exception as e:
                logger.error(f"Error predicting spot {spot_id}: {e}")
                is_occupied = False
            
            predictions.append({
                "id": str(spot_id),
                "isOccupied": is_occupied,
                "polygon": polygon_points
            })
        return predictions
