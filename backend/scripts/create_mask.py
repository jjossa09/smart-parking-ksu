import cv2
import json
import logging
import os

logging.basicConfig(level=logging.INFO)

# Configuration
IMAGE_PATH = '../carParkImg.png'
MASK_FILE = '../mask.json'
BOX_WIDTH, BOX_HEIGHT = 100, 150

positions = []

def mouse_click(events, x, y, flags, params):
    global positions

    # Left click: Add a parking spot (Top-left corner)
    if events == cv2.EVENT_LBUTTONDOWN:
        # Calculate top-left based on click being roughly center
        # or just treat click as top-left (easier for precision). 
        # We'll treat click as top-left here.
        spot = {
            "id": f"Spot_{len(positions)+1}",
            "coords": (x, y, BOX_WIDTH, BOX_HEIGHT)
        }
        positions.append(spot)
        logging.info(f"Added {spot['id']} at ({x}, {y})")

    # Right click: Delete the nearest spot
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(positions):
            px, py, pw, ph = pos["coords"]
            # Check if click is inside the box
            if px < x < px + pw and py < y < py + ph:
                removed = positions.pop(i)
                logging.info(f"Removed {removed['id']}")
                break

# Load existing mask if it exists
if os.path.exists(MASK_FILE):
    with open(MASK_FILE, 'r') as f:
        try:
            positions = json.load(f)
            logging.info(f"Loaded {len(positions)} spots from mask.json")
        except:
            positions = []

while True:
    # Read fresh image every loop to redraw cleanly
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        logging.error(f"Could not read {IMAGE_PATH}. Please ensure it exists.")
        break
        
    for pos in positions:
        x, y, w, h = pos["coords"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    cv2.imshow("KSU Smart Parking - Mask Editor", img)
    cv2.setMouseCallback("KSU Smart Parking - Mask Editor", mouse_click)
    
    # Wait 1ms for key press
    key = cv2.waitKey(1)
    
    # Press 'q' or 'ESC' to save and quit
    if key == ord('q') or key == 27:
        with open(MASK_FILE, 'w') as f:
            json.dump(positions, f, indent=4)
        logging.info(f"Saved {len(positions)} spots to {MASK_FILE} and exiting.")
        break

cv2.destroyAllWindows()
