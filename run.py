from helper import Shannon_Entropy, Entropy_Mask, Threshold_Pass, Threshold
import cv2
import numpy as np


## Entropy Mask Pipeline

base_rgb_image = cv2.imread("images/test_plate7.png") #Base color
base_rgb_image = cv2.resize(base_rgb_image, (640, 480))
base_rgb_image = cv2.GaussianBlur(base_rgb_image, (3,3), 0)

HSV_image = cv2.cvtColor(base_rgb_image, cv2.COLOR_BGR2HSV) #HSV
# grayscale_image = cv2.cvtColor(HSV_image, cv2.COLOR_BGR2GRAY)  #Grayscale
grayscale_image = cv2.cvtColor(base_rgb_image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Entropy_Filterd_mask", grayscale_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


## Edge Detection Pipeline

edge_detection =  cv2.Canny(grayscale_image, 5, 12, apertureSize=3)
# sobel_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
# sobel_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)
# sobel_edges = np.uint8(np.sqrt(sobel_x**2 + sobel_y**2) > 50) * 255
# edge_detection = cv2.bitwise_or(edge_detection, sobel_edges)
# Apply closing
# kernel_sizes = [1, 1, 2, 2]
# for size in kernel_sizes:
#     edge_detection = cv2.morphologyEx(edge_detection, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)))
edge_detection = cv2.dilate(edge_detection, np.ones((2,2), np.uint8), iterations=1)
#edge_detection = cv2.morphologyEx(edge_detection, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), iterations=2)

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

cv2.imshow("Gray", gray_resized)
cv2.imshow("edge", edge_resized)
cv2.imshow("Filled", filled_resized)
cv2.imshow("Entropy_Filterd_mask", entropy_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


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

cv2.imshow("Threshold_mask", resized)
cv2.imshow("Threshold_mask", resized2)
cv2.waitKey(0)
cv2.destroyAllWindows()
    




