# Helper functions extracted from original test.py
import cv2
import numpy as np
import os, time

def extract_shape_features(binary_image, gray_image=None):
    """
    Extract shape features from a binary (0/255) mask.
    
    Args:
        binary_image (np.ndarray): Single-channel binary image (0 background, 255 foreground).
        gray_image (np.ndarray, optional): Grayscale image for entropy calculation.
    
    Returns:
        list[dict]: list of shape feature dictionaries, one per contour.
    """
    # Find contours of thresholded regions
    cnts, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    for idx, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        perimeter = cv2.arcLength(cnt, True)

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

        M = cv2.moments(cnt)
        cx = M["m10"] / M["m00"] if M["m00"] != 0 else x + w / 2
        cy = M["m01"] / M["m00"] if M["m00"] != 0 else y + h / 2

        # Optional: entropy of region
        entropy = None
        if gray_image is not None:
            roi = gray_image[y:y+h, x:x+w]
            if roi.size > 0:
                vals, counts = np.unique(roi, return_counts=True)
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log2(probs))

        features.append({
            "idx": idx,
            "area": float(area),
            "perimeter": float(perimeter),
            "aspect_ratio": float(aspect),
            "solidity": float(solidity),
            "circularity": float(circularity),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "bbox_x": int(x),
            "bbox_y": int(y),
            "bbox_w": int(w),
            "bbox_h": int(h),
            "entropy": float(entropy) if entropy is not None else None,
        })

    return features

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def save_stage(img, out_dir, stem, stage_name):
    """Save an image for a named stage; returns the file path."""
    ensure_dir(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{stem}__{stage_name}_{ts}.png"
    fp = os.path.join(out_dir, fname)
    cv2.imwrite(fp, img)
    print(f"[saved] {stage_name} -> {fp}")
    return fp

def Shannon_Entropy(Isolated_Grayscale):

        #Get Shannon entropy

    total_pixels = len(Isolated_Grayscale)

    unique_pixels = set(list(Isolated_Grayscale))

    unique_count = []

    #Find occurences
    for pixel in unique_pixels:

        unique_count.append(list(Isolated_Grayscale).count(pixel))

    #Summation

    H = 0
    for count in unique_count:

        if (count == 0):
            continue

        Pi = count / total_pixels

        H += -(Pi * np.log2(Pi))

    return H

def Entropy_Mask(edge_detection, Detected_Regions, grayscale_image):

    Entropy_Filterd_mask = np.zeros_like(edge_detection)

    Entropy_Threshold = 4.8

    for region in (Detected_Regions[0]):

        filled = np.zeros_like(edge_detection)

    
        isolated_region = [region]

        

        cv2.drawContours(filled, isolated_region, -1, 255, -1)

        Isolated_Grayscale = list(grayscale_image[filled > 0]) #only take grayscale pixel data if it is in that region

        
        Entropy = Shannon_Entropy(Isolated_Grayscale)

        if Entropy < Entropy_Threshold:
            Entropy_Filterd_mask = cv2.bitwise_or(Entropy_Filterd_mask, filled)

    return Entropy_Filterd_mask

def Threshold_Pass(C, D, I, IB):

    cond_1 = (C - (D/2)) <= I -IB
    cond_2 = I - IB <= (C + (D/2)) 
    return cond_1 and cond_2

def Threshold(HSV_Background, HSV_image, condition=1):
    
   
    PARAMS = {
        1: {"C": [0, -8, -10], "D": [10, 8, 4]},
        2: {"C": [0, -14, -19], "D": [12, 10, 6]},
        3: {"C": [0, -20, -28], "D": [12, 10, 6]},
    }

    params = PARAMS[condition]
    C = np.array(params["C"])
    D = np.array(params["D"])

   
    diff_hsv = HSV_image.astype(np.int16) - HSV_Background.astype(np.int16)


    lower = C - D // 2
    upper = C + D // 2

    
    mask = np.all((diff_hsv >= lower) & (diff_hsv <= upper), axis=2).astype(np.uint8)

   
    fthresh = mask * 255

    # passing_pixels = np.count_nonzero(mask)
    # total_pixels = mask.shape[0] * mask.shape[1]
    # print(f"Number of pixels passing thresholding: {passing_pixels} out of {total_pixels}")

    return fthresh
