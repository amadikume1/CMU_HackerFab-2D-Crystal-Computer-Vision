from helper import Shannon_Entropy, Entropy_Mask, Threshold_Pass, Threshold, save_stage, extract_shape_features
import cv2
import numpy as np
import os, sqlite3, json
from datetime import datetime


## Entropy Mask Pipeline

base_rgb_image = cv2.imread("images/test_plate7.png") #Base color
base_rgb_image = cv2.resize(base_rgb_image, (640, 480))
base_rgb_image = cv2.GaussianBlur(base_rgb_image, (3,3), 0)

HSV_image = cv2.cvtColor(base_rgb_image, cv2.COLOR_BGR2HSV) #HSV
# grayscale_image = cv2.cvtColor(HSV_image, cv2.COLOR_BGR2GRAY)  #Grayscale
grayscale_image = cv2.cvtColor(base_rgb_image, cv2.COLOR_BGR2GRAY)

#cv2.imshow("Entropy_Filterd_mask", grayscale_image)
save_stage(grayscale_image, "outputs", "plate7", "gray")
#cv2.waitKey(0)
#cv2.destroyAllWindows()


## Edge Detection Pipeline

edge_detection = cv2.Canny(grayscale_image, 5, 12, apertureSize=3)
edge_detection = cv2.dilate(edge_detection, np.ones((2,2), np.uint8), iterations=1)

original_filled = np.zeros_like(edge_detection)
Detected_Regions = cv2.findContours(edge_detection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(original_filled, Detected_Regions[0], -1, 255, -1)
Entropy_Filterd_mask = Entropy_Mask(edge_detection, Detected_Regions, grayscale_image)

scale_factor = 0.8
height, width = grayscale_image.shape[:2]
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)

# Resize images
gray_resized = cv2.resize(grayscale_image, (new_width, new_height))
edge_resized = cv2.resize(edge_detection, (new_width, new_height))
filled_resized = cv2.resize(original_filled, (new_width, new_height))
entropy_resized = cv2.resize(Entropy_Filterd_mask, (new_width, new_height))

#cv2.imshow("Gray", gray_resized)
#cv2.imshow("edge", edge_resized)
save_stage(edge_detection, "outputs", "plate7", "edges")
#cv2.imshow("Filled", filled_resized)
save_stage(filled_resized, "outputs", "plate7", "filled")
#cv2.imshow("Entropy_Filterd_mask", entropy_resized)
save_stage(Entropy_Filterd_mask, "outputs", "plate7", "entropy_mask")
#cv2.waitKey(0)
#cv2.destroyAllWindows()


## Thresholding Pipeline

Background = cv2.imread("images/Background7.png")
Background = cv2.resize(Background, (base_rgb_image.shape[1], base_rgb_image.shape[0]))

gaussian_filtered = cv2.GaussianBlur(Background, (5, 5), 0)
mean_filtered = cv2.blur(gaussian_filtered, (5, 5))
HSV_Background = cv2.cvtColor(mean_filtered, cv2.COLOR_BGR2HSV)

x = len(list(HSV_image))
y = len(list(HSV_image)[0])
x_b = len(list(Background))
y_B = len(list(Background)[0])

HSV_image = cv2.GaussianBlur(HSV_image, (3, 3), 0)
Threshold_mask = Threshold(HSV_Background, HSV_image)
kernel = np.ones((3, 3), np.uint8)
opened = cv2.morphologyEx(Threshold_mask, cv2.MORPH_OPEN, kernel, iterations=1)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

test_res = cv2.bitwise_and(closed, Entropy_Filterd_mask)

# Resize images
resized = cv2.resize(closed, (new_width, new_height))
resized2 = cv2.resize(test_res, (new_width, new_height))

#cv2.imshow("Threshold_mask", resized)
#cv2.imshow("Threshold_mask", resized2)
save_stage(Threshold_mask, "outputs", "plate7", "threshold_mask")
#cv2.waitKey(0)
#cv2.destroyAllWindows()
    
"""
input: already thresholded image
output: optical microscope images, positions of the XY stage and objective lens, and shape features in SQL database form
"""

# Extract shape features from the final mask
shape_features = extract_shape_features(test_res)
for f in shape_features:              # f is one dict
    for feature, value in f.items():  # iterate its key:value pairs
        print(feature, value)
    print("----")

# Save results to SQLite database
conn = sqlite3.connect('results.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS results
                (timestamp TEXT, area REAL, perimeter REAL, circularity REAL, aspect_ratio REAL,
                centroid_x REAL, centroid_y REAL, bbox_x INTEGER, bbox_y INTEGER, bbox_w INTEGER, bbox_h INTEGER)
''')

for feature in shape_features:
    c.execute('''INSERT INTO results (timestamp, area, perimeter, circularity, aspect_ratio,
                centroid_x, centroid_y, bbox_x, bbox_y, bbox_w, bbox_h)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                    datetime.utcnow().isoformat(),
                    feature.get("area"),
                    feature.get("perimeter"),
                    feature.get("circularity"),
                    feature.get("aspect_ratio"),
                    feature.get("centroid_x"),
                    feature.get("centroid_y"),
                    feature.get("bbox_x"),
                    feature.get("bbox_y"),
                    feature.get("bbox_w"),
                    feature.get("bbox_h"),
                ))

conn.commit()