# Helper functions extracted from original test.py
import cv2
import numpy as np

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
