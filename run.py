"""
Entropy-Based Substrate Identification Pipeline
------------------------------------------------
This script performs preprocessing, entropy-based segmentation, thresholding,
and shape feature extraction for optical microscopy images of 2D materials.

"""

# --- Imports ---
from helper import (
    Shannon_Entropy, Entropy_Mask, Threshold_Pass,
    Threshold, save_stage, extract_shape_features
)
import cv2
import numpy as np
import os, sqlite3, json
from datetime import datetime


"""=============================
1. ENTROPY MASK PIPELINE
=============================="""

# Load the base microscope image and resize for standardization
base_rgb_image = cv2.imread("Input_images/test_plate7.png")
base_rgb_image = cv2.resize(base_rgb_image, (640, 480))

# Apply light Gaussian blur to reduce sensor noise before processing
base_rgb_image = cv2.GaussianBlur(base_rgb_image, (3, 3), 0)

# Convert RGB to HSV for color-based filtering (hue, saturation, value)
HSV_image = cv2.cvtColor(base_rgb_image, cv2.COLOR_BGR2HSV)

# Convert to grayscale for entropy and edge operations
grayscale_image = cv2.cvtColor(base_rgb_image, cv2.COLOR_BGR2GRAY)

# Save grayscale stage for visualization/debugging
save_stage(grayscale_image, "Output_images", "plate7", "gray")


"""=============================
2. EDGE DETECTION PIPELINE
=============================="""

# Detect edges using Canny with low thresholds for fine edge sensitivity
edge_detection = cv2.Canny(grayscale_image, 5, 12, apertureSize=3)

# Dilate thin edges slightly to improve contour connectivity
edge_detection = cv2.dilate(edge_detection, np.ones((2, 2), np.uint8), iterations=1)

# Initialize blank mask for filled region drawing
original_filled = np.zeros_like(edge_detection)

# Find outermost contours in the edge map
Detected_Regions = cv2.findContours(edge_detection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Fill all detected contours into a binary mask
cv2.drawContours(original_filled, Detected_Regions[0], -1, 255, -1)

# Compute entropy-based mask to retain textured regions of interest
Entropy_Filterd_mask = Entropy_Mask(edge_detection, Detected_Regions, grayscale_image)

# Downscale images for faster visualization and preview
scale_factor = 0.8
height, width = grayscale_image.shape[:2]
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)

# Resize intermediate outputs
gray_resized = cv2.resize(grayscale_image, (new_width, new_height))
edge_resized = cv2.resize(edge_detection, (new_width, new_height))
filled_resized = cv2.resize(original_filled, (new_width, new_height))
entropy_resized = cv2.resize(Entropy_Filterd_mask, (new_width, new_height))

# Save intermediate edge/entropy results
save_stage(edge_detection, "Output_images", "plate7", "edges")
save_stage(filled_resized, "Output_images", "plate7", "filled")
save_stage(Entropy_Filterd_mask, "Output_images", "plate7", "entropy_mask")


"""=============================
3. COLOR THRESHOLDING PIPELINE
=============================="""

# Load background reference image for same substrate
Background = cv2.imread("Input_images/Background7.png")
Background = cv2.resize(Background, (base_rgb_image.shape[1], base_rgb_image.shape[0]))

# Apply Gaussian and mean filtering to smooth background lighting variations
gaussian_filtered = cv2.GaussianBlur(Background, (5, 5), 0)
mean_filtered = cv2.blur(gaussian_filtered, (5, 5))

# Convert background image to HSV for comparison
HSV_Background = cv2.cvtColor(mean_filtered, cv2.COLOR_BGR2HSV)

# Apply additional blur to the main image for smoother threshold transitions
HSV_image = cv2.GaussianBlur(HSV_image, (3, 3), 0)

# Generate binary mask highlighting color differences between background and sample
Threshold_mask = Threshold(HSV_Background, HSV_image)

# Apply morphological operations to remove small noise and fill small holes
kernel = np.ones((3, 3), np.uint8)
opened = cv2.morphologyEx(Threshold_mask, cv2.MORPH_OPEN, kernel, iterations=1)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

# Combine color threshold mask with entropy-based mask for refined segmentation
test_res = cv2.bitwise_and(closed, Entropy_Filterd_mask)

# Resize results for quick visualization
resized = cv2.resize(closed, (new_width, new_height))
resized2 = cv2.resize(test_res, (new_width, new_height))

# Save the thresholding output
save_stage(Threshold_mask, "Output_images", "plate7", "threshold_mask")


"""=============================
4. SHAPE FEATURE EXTRACTION & DATABASE ENTRY
=============================="""

# Extract geometric and intensity-based shape features from the final binary mask
shape_features = extract_shape_features(test_res, base_rgb_image)

# Connect to local SQLite database (creates if not existing)
conn = sqlite3.connect("Database/substrate_database")
c = conn.cursor()

# Ensure the table schema exists for storing substrate info
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

# Insert each detected region’s features into the database
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

# Commit changes and close database connection
conn.commit()
conn.close()
