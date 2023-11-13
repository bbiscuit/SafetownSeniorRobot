import cv2
import numpy as np

# Load an image in BGR format
cam = cv2.VideoCapture(0)

ret, bgr_image = cam.read()

# Convert BGR image to HSV
hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the color you want to detect in HSV
lower_bound = np.array([20, 100, 100])  # Lower bound for HSV (yellow)
upper_bound = np.array([30, 255, 255])  # Upper bound for HSV (yellow)

# Create a binary mask to isolate the desired color
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through the detected contours and fit lines
id = 0
for contour in contours:
    if len(contour) >= 2:
        area = cv2.contourArea(contour)
        if area > 100:
            cv2.drawContours(bgr_image, contours, id, (255, 0, 0))
        print(cv2.contourArea(contour))
    id += 1

# Display the result with lines connecting color regions
cv2.imwrite('test.jpg', bgr_image)