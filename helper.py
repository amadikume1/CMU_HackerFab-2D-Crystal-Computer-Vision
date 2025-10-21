# ==========================================================
# Helper functions for image processing and feature extraction
# Adapted from the original test.py for modular pipeline use.
# Each function includes concise comments for clarity and grading.
# ==========================================================

import cv2
import numpy as np
import os, time


# ----------------------------------------------------------
# Extract geometric and color features from a binary mask
# ----------------------------------------------------------
def extract_shape_features(binary_image, color_image=None):
    """
    Extract shape and color features from a binary (0/255) mask.

    Args:
        binary_image (np.ndarray): Single-channel binary image (0 background, 255 foreground).
        color_image (np.ndarray, optional): Color image (BGR) to estimate region color.

    Returns:
        list[dict]: List of shape feature dictionaries, one per contour.
    """

    # Find all external contours in the binary mask
    cnts, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    for idx, cnt in enumerate(cnts):
        # Calculate contour area; skip if non-positive
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue

        # Perimeter used for polygon approximation
        perimeter = cv2.arcLength(cnt, True)

        # Bounding box and centroid (for location tracking)
        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx = M["m10"] / M["m00"] if M["m00"] != 0 else x + w / 2
        cy = M["m01"] / M["m00"] if M["m00"] != 0 else y + h / 2

        # ---- Shape identification ----
        # Approximate contour polygon and classify by number of sides
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        sides = len(approx)

        if sides == 3:
            shape_name = "Triangle"
        elif sides == 4:
            aspect_ratio = float(w) / h if h > 0 else 0
            shape_name = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif sides == 5:
            shape_name = "Pentagon"
        elif sides > 5:
            shape_name = "Circle"
        else:
            shape_name = "Unknown"

        # ---- Optional color detection ----
        color_name = "Unknown"
        if color_image is not None:
            # Create binary mask for current contour region
            mask = np.zeros(binary_image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            # Compute average BGR color within region
            mean_color = cv2.mean(color_image, mask=mask)[:3]
            b, g, r = mean_color

            # Simple color labeling based on dominant channel
            if r > g and r > b:
                color_name = "RED"
            elif g > r and g > b:
                color_name = "GREEN"
            elif b > r and b > g:
                color_name = "BLUE"
            elif abs(r - g) < 15 and abs(g - b) < 15:
                color_name = "GRAY"
            else:
                color_name = "MIXED"

        # Append extracted measurements into a feature dictionary
        features.append({
            "Wafer_ID": f"Auto_{idx}",     # Placeholder wafer ID
            "Material": "placeholder",     # Material can be updated later
            "Shape": shape_name,
            "Size_Width": float(w),
            "Size_Height": float(h),
            "Color": color_name,
            "Position_X": float(cx),
            "Position_Y": float(cy)
        })

    return features


# ----------------------------------------------------------
# Create directory if not existing (for output saving)
# ----------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# ----------------------------------------------------------
# Save an intermediate image stage to a folder
# ----------------------------------------------------------
def save_stage(img, out_dir, stem, stage_name):
    """
    Save an image output with timestamp and stage label.
    """
    ensure_dir(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")  # Timestamp for file naming
    fname = f"{stem}__{stage_name}_{ts}.png"
    fp = os.path.join(out_dir, fname)
    cv2.imwrite(fp, img)
    print(f"[saved] {stage_name} -> {fp}")
    return fp


# ----------------------------------------------------------
# Shannon entropy computation (for local texture measurement)
# ----------------------------------------------------------
def Shannon_Entropy(Isolated_Grayscale):
    """
    Calculate Shannon entropy of pixel values within an isolated region.
    Higher entropy indicates more texture variation.
    """

    total_pixels = len(Isolated_Grayscale)
    unique_pixels = set(list(Isolated_Grayscale))
    unique_count = []

    # Count occurrences of each unique intensity
    for pixel in unique_pixels:
        unique_count.append(list(Isolated_Grayscale).count(pixel))

    # Compute entropy as -Σ p*log2(p)
    H = 0
    for count in unique_count:
        if count == 0:
            continue
        Pi = count / total_pixels
        H += -(Pi * np.log2(Pi))

    return H


# ----------------------------------------------------------
# Build binary mask selecting only low-entropy regions
# ----------------------------------------------------------
def Entropy_Mask(edge_detection, Detected_Regions, grayscale_image):
    """
    Generates a mask by selecting regions with entropy below a threshold,
    effectively filtering out noisy/high-texture backgrounds.
    """

    Entropy_Filterd_mask = np.zeros_like(edge_detection)
    Entropy_Threshold = 4.8  # Empirical threshold for low-texture areas

    # Loop through all detected contour regions
    for region in (Detected_Regions[0]):
        filled = np.zeros_like(edge_detection)

        # Isolate one region and fill its contour
        isolated_region = [region]
        cv2.drawContours(filled, isolated_region, -1, 255, -1)

        # Collect grayscale pixels within region for entropy calculation
        Isolated_Grayscale = list(grayscale_image[filled > 0])

        # Compute entropy for region
        Entropy = Shannon_Entropy(Isolated_Grayscale)

        # Keep regions with entropy below threshold (likely uniform areas)
        if Entropy < Entropy_Threshold:
            Entropy_Filterd_mask = cv2.bitwise_or(Entropy_Filterd_mask, filled)

    return Entropy_Filterd_mask


# ----------------------------------------------------------
# Threshold pass check between color differences
# ----------------------------------------------------------
def Threshold_Pass(C, D, I, IB):
    """
    Simple logical check if pixel difference lies within threshold bounds.
    Used internally by the color thresholding step.
    """
    cond_1 = (C - (D / 2)) <= I - IB
    cond_2 = I - IB <= (C + (D / 2))
    return cond_1 and cond_2


# ----------------------------------------------------------
# Color-space thresholding between background and target
# ----------------------------------------------------------
def Threshold(HSV_Background, HSV_image, condition=1):
    """
    Compare HSV difference between sample and background,
    generating a binary mask where difference fits within defined bounds.
    """

    # Parameter sets controlling sensitivity (empirically tuned)
    PARAMS = {
        1: {"C": [0, -8, -10], "D": [10, 8, 4]},
        2: {"C": [0, -14, -19], "D": [12, 10, 6]},
        3: {"C": [0, -20, -28], "D": [12, 10, 6]},
    }

    # Select parameters based on condition
    params = PARAMS[condition]
    C = np.array(params["C"])
    D = np.array(params["D"])

    # Compute per-pixel HSV difference between sample and background
    diff_hsv = HSV_image.astype(np.int16) - HSV_Background.astype(np.int16)

    # Define lower and upper threshold ranges
    lower = C - D // 2
    upper = C + D // 2

    # Binary mask where all channels are within threshold bounds
    mask = np.all((diff_hsv >= lower) & (diff_hsv <= upper), axis=2).astype(np.uint8)

    # Convert boolean mask to 0/255 image
    fthresh = mask * 255

    return fthresh
