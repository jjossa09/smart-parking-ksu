import os
import sys
import cv2
import json
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(SCRIPTS_DIR)

from config import FRAME_PATH, SPOTS_PATH

DISPLAY_SCALE = 1.7
POINT_RADIUS = 2
LINE_THICKNESS = 1
POLYGON_THICKNESS = 2
TEXT_SCALE = 0.45
TEXT_THICKNESS = 1
WINDOW_NAME = "Annotate Parking Spots"

image = cv2.imread(FRAME_PATH)
if image is None:
    raise FileNotFoundError(
        f"Could not load image at: {FRAME_PATH}\n"
        f"Make sure the frame exists first."
    )

base_image = image.copy()
spots = []
current_points = []

mode_input = input("Choose annotation mode ('2' for rectangle, '4' for polygon): ").strip()
if mode_input not in ["2", "4"]:
    print("Invalid choice. Defaulting to 4-point mode.")
    mode_input = "4"

POINTS_PER_SPOT = int(mode_input)


def normalize_rectangle_to_4_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    return [
        [x_min, y_min],  # top-left
        [x_max, y_min],  # top-right
        [x_max, y_max],  # bottom-right
        [x_min, y_max]   # bottom-left
    ]


def redraw():
    global image
    image = base_image.copy()

    # Draw saved spots
    for spot in spots:
        pts = np.array(spot["points"], dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=POLYGON_THICKNESS)

        x, y = pts[0]
        cv2.putText(
            image,
            str(spot["id"]),
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            (0, 0, 255),
            TEXT_THICKNESS
        )

    # Draw current unfinished shape
    if len(current_points) > 0:
        for pt in current_points:
            cv2.circle(image, tuple(pt), POINT_RADIUS, (0, 255, 0), -1)

        if len(current_points) > 1:
            pts = np.array(current_points, dtype=np.int32)
            cv2.polylines(image, [pts], isClosed=False, color=(0, 255, 255), thickness=LINE_THICKNESS)


def mouse_callback(event, x, y, flags, param):
    global current_points, spots

    if event == cv2.EVENT_LBUTTONDOWN:
        real_x = int(x / DISPLAY_SCALE)
        real_y = int(y / DISPLAY_SCALE)

        current_points.append([real_x, real_y])
        print(f"Clicked display: ({x}, {y}) -> real: ({real_x}, {real_y})")
        redraw()

        if len(current_points) == POINTS_PER_SPOT:
            if POINTS_PER_SPOT == 2:
                final_points = normalize_rectangle_to_4_points(current_points[0], current_points[1])
            else:
                final_points = current_points.copy()

            spots.append({
                "id": len(spots) + 1,
                "points": final_points
            })

            print(f"Spot {len(spots)} saved: {final_points}")
            current_points = []
            redraw()


cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

print("Instructions:")
if POINTS_PER_SPOT == 2:
    print("- Click 2 corners for each spot (top-left and bottom-right recommended)")
else:
    print("- Click 4 corners for each spot")
    print("- Recommended order: top-left, top-right, bottom-right, bottom-left")
print("- Press 's' to save")
print("- Press 'u' to undo last saved spot")
print("- Press 'c' to clear current unfinished spot")
print("- Press 'r' to reset all spots")
print("- Press 'q' to quit")

redraw()

while True:
    display_image = cv2.resize(
        image,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_LINEAR
    )

    cv2.imshow(WINDOW_NAME, display_image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        with open(SPOTS_PATH, "w") as f:
            json.dump(spots, f, indent=2)
        print(f"Saved {len(spots)} spots to {SPOTS_PATH}")

    elif key == ord("u"):
        if spots:
            removed = spots.pop()
            print(f"Removed spot {removed['id']}")
            redraw()

    elif key == ord("c"):
        current_points = []
        print("Cleared current unfinished spot.")
        redraw()

    elif key == ord("r"):
        spots = []
        current_points = []
        print("Reset all spots.")
        redraw()

    elif key == ord("q"):
        print("Quitting.")
        break

cv2.destroyAllWindows()