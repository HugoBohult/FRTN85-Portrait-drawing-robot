import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("Edward.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Optional: blur to smooth intensity variations
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Edge detection for structure
edges = cv2.Canny(blur, 80, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Blank canvas
canvas = np.ones_like(gray) * 255

# Scribble parameters
max_scribbles = 2   # max scribbles per pixel
scribble_length = 4 # max length of each stroke

height, width = gray.shape

# Fill entire image based on intensity
for y in range(height):
    for x in range(width):
        intensity = blur[y, x]
        threshold = 30  # pixels brighter than this get no scribbles
        density = int(max_scribbles * (1 - intensity / 255))
        if intensity > threshold:
            density = 0
        for _ in range(density):
            dx = random.randint(-scribble_length, scribble_length)
            dy = random.randint(-scribble_length, scribble_length)
            # make sure lines stay inside image bounds
            x2 = np.clip(x+dx, 0, width-1)
            y2 = np.clip(y+dy, 0, height-1)
            cv2.line(canvas, (x, y), (x2, y2), 0, 1)

# Optionally, reinforce contours with additional scribbles
for contour in contours:
    for point in contour:
        x, y = point[0]
        for _ in range(2):  # extra scribbles along edges
            dx = random.randint(-scribble_length, scribble_length)
            dy = random.randint(-scribble_length, scribble_length)
            x2 = np.clip(x+dx, 0, width-1)
            y2 = np.clip(y+dy, 0, height-1)
            cv2.line(canvas, (x, y), (x2, y2), 0, 1)

# Show results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Scribble Art")
plt.imshow(canvas, cmap='gray')
plt.axis('off')
plt.show()
