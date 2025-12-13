# ==========================================================
# Helper functions for image processing and feature extraction
# Adapted from the original test.py for modular pipeline use.
# Each function includes concise comments for clarity and grading.
# ==========================================================

import cv2
import numpy as np
import json
import os, time


# ----------------------------------------------------------
# Extract geometric and color features from a binary mask
# ----------------------------------------------------------
def extract_shape_features_old(binary_img, color_img, wafer_id, image_name=""):
    features = []

    h, w = binary_img.shape

    # ===============================================================
    # 1. CREATE A VERY LOOSE BINARY MASK (detect ANY non-black edge)
    # ===============================================================
    loose_bin = (binary_img > 100).astype(np.uint8) * 255

    # Close gaps so contours are continuous
    loose_bin = cv2.dilate(loose_bin, np.ones((3,3), np.uint8), 2)
    loose_bin = cv2.medianBlur(loose_bin, 2)

    # ===============================================================
    # 2. FIND ALL OUTER CONTOURS
    # ===============================================================
    contours, _ = cv2.findContours(loose_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print(f"[Warning] No contours detected at all for {image_name}")
        save_stage(loose_bin, "Sobel_vis_results_new", image_name, "no_contours")
        return []

    # ===============================================================
    # 3. REMOVE SUBSTRATE BORDER (TOUCHING IMAGE EDGE)
    # ===============================================================
    internal = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, closed=False)

        # Skip tiny noise
        if area < 200 or peri < 80:
            continue

        xs = cnt[:,0,0]
        ys = cnt[:,0,1]

        # Count how many contour points touch border
        border_touch = (
            (xs <= 1).sum() +
            (xs >= w-2).sum() +
            (ys <= 1).sum() +
            (ys >= h-2).sum()
        )

        # Compute fraction touching border
        frac_border = border_touch / len(cnt)

        # Skip if MOST of the contour is on the border (substrate frame)
        if frac_border > 0.30:        # << relaxed threshold (was: ANY)
            continue

        # This contour is valid graphene
        internal.append(cnt)

    if len(internal) == 0:
        print(f"[Warning] No internal graphene contours for {image_name}")
        save_stage(loose_bin, "Sobel_vis_results_new", image_name, "no_internal")
        return []


    # ===============================================================
    # 4. SELECT THE LARGEST INTERNAL SHAPE
    # ===============================================================
    largest = max(internal, key=cv2.contourArea)

    # ===============================================================
    # 5. FILL IT
    # ===============================================================
    filled = np.zeros_like(binary_img)
    cv2.drawContours(filled, [largest], -1, 255, -1)

    save_stage(filled, "Sobel_vis_results_new", image_name, "filled_largest")

    # ===============================================================
    # 6. MIN-AREA RECTANGLE (4 CORNERS)
    # ===============================================================
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    polygon_str = json.dumps(box.tolist())

    # ===============================================================
    # 7. GEOMETRIC + COLOR FEATURES
    # ===============================================================
    x, y, w0, h0 = cv2.boundingRect(largest)
    M = cv2.moments(largest)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    masked_color = cv2.mean(color_img, mask=filled)

    feature = {
        "Wafer_ID": wafer_id,
        "Material": "Graphene",
        "Shape": "Polygon",
        "Size_Width": float(w0),
        "Size_Height": float(h0),
        "Color": f"({masked_color[0]:.1f},{masked_color[1]:.1f},{masked_color[2]:.1f})",
        "Position_X": float(cx),
        "Position_Y": float(cy),
        "Polygon": polygon_str
    }

    features.append(feature)
    return features

import cv2
import numpy as np
import json

def extract_shape_features(binary_img, color_img, wafer_id, image_name=""):
    features = []
    h, w = binary_img.shape

    # ===============================================================
    # 1. PREP: Convert to LAB and cluster colors
    # ===============================================================
    lab = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
    Z = lab.reshape((-1, 3)).astype(np.float32)

    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(center)
    labels = label.reshape((h, w))
    L_vals = centers[:, 0]

    # ===============================================================
    # 2. DETECT BACKGROUND (largest cluster)
    # ===============================================================
    counts = np.bincount(labels.flatten(), minlength=K)
    background_cluster = np.argmax(counts)
    background_L = L_vals[background_cluster]

    # ===============================================================
    # 3. REMOVE BRIGHT CLUSTERS
    # ===============================================================
    BRIGHT_THRESHOLD = 10
    bright_clusters = [i for i in range(K) if L_vals[i] > background_L + BRIGHT_THRESHOLD]

    # ===============================================================
    # 4. GRAPHENE CLUSTERS
    # ===============================================================
    graphene_clusters = [i for i in range(K) if i != background_cluster and i not in bright_clusters]

    if len(graphene_clusters) == 0:
        graphene_clusters = [int(np.argmin([L_vals[i] for i in range(K) if i != background_cluster]))]

    # ===============================================================
    # 5. BUILD GRAPHENE MASK
    # ===============================================================
    graphene_mask = np.zeros((h, w), dtype=np.uint8)
    for gc in graphene_clusters:
        graphene_mask[labels == gc] = 255

    # ===============================================================
    # 6. REMOVE NOISE WITHOUT MERGING SHAPES
    # ===============================================================
    # small kernel → preserves thin flakes
    graphene_mask = cv2.medianBlur(graphene_mask, 3)

    # remove tiny components manually
    num_labels, cc_mask = cv2.connectedComponents(graphene_mask)
    cleaned = np.zeros_like(graphene_mask)

    for comp in range(1, num_labels):
        region = (cc_mask == comp).astype(np.uint8)
        area = region.sum()

        if area > 600:    # keep small flakes but remove dust
            cleaned[cc_mask == comp] = 255

    graphene_mask = cleaned

    # *** IMPORTANT FIX ***
    # Replace aggressive close/open with LIGHT smoothing
    graphene_mask = cv2.GaussianBlur(graphene_mask, (5,5), 0)

    save_stage(graphene_mask, "color_seg_kmeans", image_name, "graphene_mask")

    # ===============================================================
    # 7. FIND GRAPHENE CONTOURS
    # ===============================================================
    contours, _ = cv2.findContours(graphene_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 1000]

    if len(contours) == 0:
        print(f"[Warning] No graphene detected for {image_name}")
        save_stage(graphene_mask, "color_seg_kmeans", image_name, "no_graphene")
        return []

    # ===============================================================
    # 8. SELECT LARGEST CONTOUR
    # ===============================================================
    largest = max(contours, key=cv2.contourArea)

    # ===============================================================
    # 9. FILL SHAPE
    # ===============================================================
    filled = np.zeros_like(graphene_mask)
    cv2.drawContours(filled, [largest], -1, 255, -1)
    save_stage(filled, "color_seg_kmeans", image_name, "filled_largest")

    # ===============================================================
    # 10. POLYGON + FEATURES
    # ===============================================================
    epsilon = 0.01 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    num_sides = len(approx)

    rect = cv2.minAreaRect(largest)
    box = np.int32(cv2.boxPoints(rect))
    polygon_str = json.dumps(box.tolist())

    x, y, w0, h0 = cv2.boundingRect(largest)
    M = cv2.moments(largest)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    masked_color = cv2.mean(color_img, mask=filled)

    feature = {
        "Wafer_ID": wafer_id,
        "Material": "Graphene",
        "Shape": f"{num_sides}-side polygon",
        "Size_Width": float(w0),
        "Size_Height": float(h0),
        "Color": f"({masked_color[0]:.1f},{masked_color[1]:.1f},{masked_color[2]:.1f})",
        "Position_X": float(cx),
        "Position_Y": float(cy),
        "Polygon": polygon_str
    }

    features.append(feature)
    return features

def extract_shape_features_new(binary_image, color_image=None, image_name=""):
    """
    Extract shape and color features from a binary (0/255) mask.

    Args:
        binary_image (np.ndarray): Single-channel binary image (0 background, 255 foreground).
        color_image (np.ndarray, optional): Color image (BGR) to estimate region color.

    Returns:
        list[dict]: List of shape feature dictionaries, one per contour.
    """

    cnts, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    for idx, cnt in enumerate(cnts):

        area = cv2.contourArea(cnt)
        if area <= 0:
            continue

        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx = M["m10"] / M["m00"] if M["m00"] != 0 else x + w / 2
        cy = M["m01"] / M["m00"] if M["m00"] != 0 else y + h / 2

        # ---- Shape identification ----
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
            shape_name = f"{sides}-Polygon"
        else:
            shape_name = "Unknown"

        # ---- Optional color detection ----
        color_name = "Unknown"
        if color_image is not None:
            if color_image.shape[:2] != binary_image.shape[:2]:
                color_image = cv2.resize(color_image, (binary_image.shape[1], binary_image.shape[0]))
            mask = np.zeros(binary_image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_color = cv2.mean(color_image, mask=mask)[:3]
            b, g, r = mean_color

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

        # ---- Polygon points ----
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)       # 4 corner points
        box = np.int32(box)
        polygon_str = json.dumps(box.tolist())  # save as JSON string

        # draw box for visualization (optional)
        cv2.drawContours(color_image, [box], 0, (0,255,0), 2)
        save_stage(binary_image, "shape_features_sobel", f"{image_name}", f"binary_{idx+1}")
        save_stage(color_image, "shape_features_sobel", f"{image_name}", f"boxed_{idx+1}")

        # ---- Append everything ----
        features.append({
            "Wafer_ID": f"{image_name}_shape{idx+1}",
            "Material": "Graphene",
            "Shape": shape_name,
            "Size_Width": float(w),
            "Size_Height": float(h),
            "Color": color_name,
            "Position_X": float(cx),
            "Position_Y": float(cy),
            "Polygon": polygon_str
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
    fname = f"{stem}__{stage_name}.png"
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
