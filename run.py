"""
Entropy-Based Substrate Identification Pipeline
------------------------------------------------
Batch-processing version:
Performs preprocessing, entropy-based segmentation, thresholding,
and saves intermediate results for multiple optical microscopy images.
Then loads the saved masks for shape feature extraction & database entry.
"""

# --- Imports ---
from helper import (
    Shannon_Entropy, Entropy_Mask, Threshold_Pass,
    Threshold, save_stage, extract_shape_features, extract_shape_features_new
)
import cv2
import numpy as np
import os, sqlite3
from datetime import datetime


"""=============================
1–3. ENTROPY + EDGE + THRESHOLD PIPELINE (BATCH)
=============================="""

def process_batch(input_dir, background_dir, output_dir):
    """
    Process a batch of sample images with corresponding backgrounds.
    Each processed stage is saved to disk using `save_stage`.

    Args:
        input_dir (str): Directory containing input sample images.
        background_dir (str): Directory containing background reference images.
        output_dir (str): Directory to store all intermediate results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # List all sample images
    sample_files = [f for f in os.listdir(input_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for sample_file in sample_files:
        prefix = os.path.splitext(sample_file)[0]
        sample_path = os.path.join(input_dir, sample_file)

        # Try to find matching background (e.g., Background7.png for test_plate7.png)
        bg_name = f"Background{prefix[-1]}.png" if f"Background{prefix[-1]}.png" in os.listdir(background_dir) else None
        if bg_name is None:
            print(f"[Warning] No matching background found for {sample_file}. Skipping.")
            continue
        background_path = os.path.join(background_dir, bg_name)

        print(f"[Processing] {sample_file} with {bg_name}")

        # --- 1. Load & preprocess ---
        base_rgb_image = cv2.imread(sample_path)
        base_rgb_image = cv2.resize(base_rgb_image, (640, 480))
        base_rgb_image = cv2.GaussianBlur(base_rgb_image, (3, 3), 0)

        HSV_image = cv2.cvtColor(base_rgb_image, cv2.COLOR_BGR2HSV)
        grayscale_image = cv2.cvtColor(base_rgb_image, cv2.COLOR_BGR2GRAY)
        save_stage(grayscale_image, output_dir, prefix, "gray")

        # --- 2. Edge detection ---
        edge_detection = cv2.Canny(grayscale_image, 5, 12, apertureSize=3)
        edge_detection = cv2.dilate(edge_detection, np.ones((2, 2), np.uint8), iterations=1)
        original_filled = np.zeros_like(edge_detection)

        Detected_Regions = cv2.findContours(edge_detection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(original_filled, Detected_Regions[0], -1, 255, -1)

        Entropy_Filterd_mask = Entropy_Mask(edge_detection, Detected_Regions, grayscale_image)
        save_stage(edge_detection, output_dir, prefix, "edges")
        save_stage(original_filled, output_dir, prefix, "filled")
        save_stage(Entropy_Filterd_mask, output_dir, prefix, "entropy_mask")

        # --- 3. Color thresholding ---
        Background = cv2.imread(background_path)
        Background = cv2.resize(Background, (base_rgb_image.shape[1], base_rgb_image.shape[0]))

        gaussian_filtered = cv2.GaussianBlur(Background, (5, 5), 0)
        mean_filtered = cv2.blur(gaussian_filtered, (5, 5))
        HSV_Background = cv2.cvtColor(mean_filtered, cv2.COLOR_BGR2HSV)

        HSV_image = cv2.GaussianBlur(HSV_image, (3, 3), 0)
        Threshold_mask = Threshold(HSV_Background, HSV_image)

        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(Threshold_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

        combined_mask = cv2.bitwise_and(closed, Entropy_Filterd_mask)

        # Save all intermediate stages
        save_stage(Threshold_mask, output_dir, prefix, "threshold_mask")
        save_stage(combined_mask, output_dir, prefix, "combined_mask")

        print(f"[Saved] Results for {prefix} to {output_dir}")


"""=============================
4. SHAPE FEATURE EXTRACTION & DATABASE ENTRY
=============================="""

def extract_features_from_saved(output_dir, db_path="Database/substrate_sobel_new.db"):
    """
    Loads previously saved combined masks and extracts geometric features.
    Stores all feature data into SQLite database.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Ensure table schema exists
    c.execute('''CREATE TABLE IF NOT EXISTS substrate (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   Wafer_ID TEXT,
                   Material TEXT,
                   Shape TEXT,
                   Size_Width REAL,
                   Size_Height REAL,
                   Color TEXT,
                   Position_X REAL,
                   Position_Y REAL,
                   Polygon TEXT
                )''')

    # Iterate through saved combined masks
    for file in os.listdir(output_dir):
        if not file.endswith("__combined_mask.png"):
            continue

        prefix = file.replace("__combined_mask.png", "")
        mask_path = os.path.join(output_dir, file)
        img_path = os.path.join("Input_images", f"{prefix}.png")

        if not os.path.exists(img_path):
            print(f"[Warning] Sample image not found for {prefix}, skipping.")
            continue

        combined_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        base_rgb_image = cv2.imread(img_path)

        # Extract and insert features
        shape_features = extract_shape_features_new(combined_mask, base_rgb_image, image_name=prefix)
        for feature in shape_features:
            c.execute('''INSERT INTO substrate 
                            (Wafer_ID, Material, Shape, Size_Width, Size_Height, Color, Position_X, Position_Y, Polygon)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                             feature.get("Wafer_ID", ""),
                             feature.get("Material", ""),
                             feature.get("Shape", ""),
                             feature.get("Size_Width", 0.0),
                             feature.get("Size_Height", 0.0),
                             feature.get("Color", ""),
                             feature.get("Position_X", 0.0),
                             feature.get("Position_Y", 0.0),
                             feature.get("Polygon", "")
                         ))

        print(f"[Database] Added features for {prefix}")

    conn.commit()
    conn.close()
    print(f"[Done] All features stored in {db_path}")


def extract_features_from_model(output_dir="Sobel_vis_results_new",
                                db_path="Database/substrate_database_model.db"):
    """
    Loads binary/original image pairs from sobel_vis_results,
    extracts geometric features, and stores them into SQLite database.
    """

    # Ensure database directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS substrate")
    # Ensure table schema exists
    c.execute('''CREATE TABLE IF NOT EXISTS substrate (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   Wafer_ID TEXT,
                   Material TEXT,
                   Shape TEXT,
                   Size_Width REAL,
                   Size_Height REAL,
                   Color TEXT,
                   Position_X REAL,
                   Position_Y REAL,
                   Polygon TEXT
                )''')

    # Iterate through binary masks
    for file in os.listdir(output_dir):
        if not file.endswith("_binary.png"):
            continue

        prefix = file.replace("_binary.png", "")
        binary_path = os.path.join(output_dir, file)
        #original_path = os.path.join(output_dir, f"{prefix}_original.png")
        sharpened_path = os.path.join(output_dir, f"{prefix}_sharpened.png")

        if not os.path.exists(sharpened_path):
            print(f"[Warning] No sharpened image found for {prefix}, skipping.")
            continue

        # Read binary + color images
        binary_image = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)
        color_image = cv2.imread(sharpened_path)

        if binary_image is None or color_image is None:
            print(f"[Warning] Failed to load images for {prefix}, skipping.")
            continue

        # Extract geometric & color features
        shape_features = extract_shape_features(binary_image, color_image, "Sobel_vis_results_new", image_name=prefix)

        # Insert each feature into database
        # Insert each feature into database
        for feature in shape_features:
            c.execute('''INSERT INTO substrate 
                            (Wafer_ID, Material, Shape, Size_Width, Size_Height, Color, Position_X, Position_Y, Polygon)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                            feature.get("Wafer_ID", ""),
                            feature.get("Material", ""),
                            feature.get("Shape", ""),
                            feature.get("Size_Width", 0.0),
                            feature.get("Size_Height", 0.0),
                            feature.get("Color", ""),
                            feature.get("Position_X", 0.0),
                            feature.get("Position_Y", 0.0),
                            feature.get("Polygon", "")
                        ))
        print(f"[Database] Added features for {prefix}")

    conn.commit()
    conn.close()
    print(f"[Done] All features stored in {db_path}")

"""=============================
MAIN EXECUTION
=============================="""

if __name__ == "__main__":
    
    process_batch(
        input_dir="Sobel_inputs",
        background_dir="Input_images/Background_images",
        output_dir="Sobel_outputs"
    )
    

    extract_features_from_saved(
        output_dir="Sobel_outputs",
        db_path="Database/substrate_sobel_new.db"
    )
    '''
    extract_features_from_model(
        output_dir="Sobel_vis_results_new",
        db_path="Database/substrate_database_model.db"
    )
    '''