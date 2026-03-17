import cv2
import numpy as np

# Create a simulated "KSU Campus Map" dark aesthetic background
width, height = 1200, 800
img = np.zeros((height, width, 3), np.uint8)

# Base campus ground color
img[:] = (30, 35, 40)

# Draw some "buildings" (lighter gray blocks)
cv2.rectangle(img, (100, 100), (300, 300), (45, 50, 60), -1)
cv2.putText(img, "Science Lab", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

cv2.rectangle(img, (900, 500), (1100, 750), (45, 50, 60), -1)
cv2.putText(img, "Student Ctr", (920, 625), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

# Draw the "Parking Deck" (where our spots go)
cv2.rectangle(img, (150, 400), (950, 750), (20, 22, 25), -1)
cv2.rectangle(img, (150, 400), (950, 750), (100, 100, 100), 2)

# Draw painted parking lines for our 20 spots
for i in range(10):
    x_base = 240 + (i * 72)
    # Top Row (Lot A)
    cv2.rectangle(img, (x_base, 520), (x_base+40, 600), (255, 255, 255), 1)
    # Bottom Row (Lot B)
    cv2.rectangle(img, (x_base, 620), (x_base+40, 700), (255, 255, 255), 1)

cv2.imwrite(r"C:\Users\MatambaPC\Documents\hackathon1\frontend\src\assets\ksu_map.png", img)
print("Successfully generated placeholder KSU Map")
