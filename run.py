from helper import Shannon_Entropy, Entropy_Mask, Threshold_Pass, Threshold, save_stage, extract_shape_features
import cv2
import numpy as np
import os, sqlite3, json
from datetime import datetime


"""Entropy Mask Pipeline"""

base_rgb_image = cv2.imread("Input_images/test_plate7.png") #Base color
base_rgb_image = cv2.resize(base_rgb_image, (640, 480))
base_rgb_image = cv2.GaussianBlur(base_rgb_image, (3,3), 0)

HSV_image = cv2.cvtColor(base_rgb_image, cv2.COLOR_BGR2HSV) #HSV
grayscale_image = cv2.cvtColor(base_rgb_image, cv2.COLOR_BGR2GRAY)

save_stage(grayscale_image, "Output_images", "plate7", "gray")

"""Edge Detection Pipeline"""

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

save_stage(edge_detection, "Output_images", "plate7", "edges")
save_stage(filled_resized, "Output_images", "plate7", "filled")
save_stage(Entropy_Filterd_mask, "Output_images", "plate7", "entropy_mask")

"""Thresholding Pipeline"""

Background = cv2.imread("Input_images/Background7.png")
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

save_stage(Threshold_mask, "Output_images", "plate7", "threshold_mask")

"""Extract shape features from the final mask"""

shape_features = extract_shape_features(test_res)

# Connect to existing substrate database
conn = sqlite3.connect("Database/substrate_database")
c = conn.cursor()

# Ensure the substrate table exists (matches your database.py)
c.execute('''CREATE TABLE IF NOT EXISTS substrate (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               Wafer_ID TEXT,
               Material TEXT,
               Shape TEXT,
               Size_Width REAL,
               Size_Height REAL,
               Color TEXT,
               Position_X REAL,
               Position_Y REAL
            )''')

# Insert each feature into the substrate table
for feature in shape_features:
    c.execute('''INSERT INTO substrate 
                    (Wafer_ID, Material, Shape, Size_Width, Size_Height, Color, Position_X, Position_Y)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
                     feature.get("Wafer_ID", ""),
                     feature.get("Material", ""),
                     feature.get("Shape", ""),
                     feature.get("Size_Width", 0.0),
                     feature.get("Size_Height", 0.0),
                     feature.get("Color", ""),
                     feature.get("Position_X", 0.0),
                     feature.get("Position_Y", 0.0),
                 ))

conn.commit()
conn.close()