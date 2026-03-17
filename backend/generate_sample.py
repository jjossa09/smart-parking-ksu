import cv2
import numpy as np

# Create a blank black image (simulate empty asphalt)
width, height = 1200, 800
img = np.zeros((height, width, 3), np.uint8)
img[:] = (50, 50, 50)  # Dark gray pavement

# Draw "painted lines" for 20 parking spots
# Lot A: Top row
for i in range(10):
    x = 50 + (i * 110)
    cv2.rectangle(img, (x, 50), (x+100, 200), (255, 255, 255), 2)
    cv2.putText(img, f"A{i+1}", (x+30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Lot B: Bottom row
for i in range(10):
    x = 50 + (i * 110)
    cv2.rectangle(img, (x, 250), (x+100, 400), (255, 255, 255), 2)
    cv2.putText(img, f"B{i+1}", (x+30, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Save the base image to draw the mask on
cv2.imwrite("carParkImg.png", img)

# Now, generate a 10-second video at 30fps where cars randomly appear and disappear
out = cv2.VideoWriter('carPark.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (width, height))

import random
for frame_num in range(60): # 30 seconds at 2 fps
    frame = img.copy()
    
    # Randomly draw "cars" (bright colored rectangles with some details to create edges)
    for row_y, prefix in [(50, 'A'), (250, 'B')]:
        for i in range(10):
            if random.random() > 0.5: # 50% chance a car is here this frame
                x = 50 + (i * 110)
                # Draw car body
                car_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                cv2.rectangle(frame, (x+10, row_y+10), (x+90, row_y+140), car_color, -1)
                # Draw windshield (creates strong edges for OpenCV)
                cv2.rectangle(frame, (x+20, row_y+30), (x+80, row_y+60), (200, 250, 255), -1)
                # Draw rear window
                cv2.rectangle(frame, (x+20, row_y+100), (x+80, row_y+120), (200, 250, 255), -1)

    out.write(frame)

out.release()
print("Successfully generated carParkImg.png and carPark.mp4")
