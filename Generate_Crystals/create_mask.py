import numpy as np
import re
import cv2
import random
import matplotlib.pyplot as plt

with open("cjlyhoo9awjqq0714crlf57db-2F0ddf2d28-48da-a0ec-aa56-261f47efbaa9-dl_0000000041_0000000414_0000020443_0000013940_jpg.rf.71a953885a3e2c5607cd0237b06e452b.txt", "r") as f:
    text_data = f.read()


block = re.split(r'([ \n]+)', text_data)

mono = []
few = []

temp = []
in_mono = False
in_few = False

for i in block:
        
    if i in ['1', '0', '1\n']:

        if in_mono:
            mono.append(temp)
        if in_few:
            few.append(temp)

        if i == '1' or i == '1\n':
           
            in_mono = True
            in_few = False
            

        if i == '0':
        
            in_mono = False
            in_few = True
    

        temp = []

        continue
    if (i == ' ' or i == '\n'):
        continue

    temp.append(float(i))

if in_mono:
    mono.append(temp)
if in_few:
    few.append(temp)




mono_polygon_arrays = []
for poly in mono:
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    mono_polygon_arrays.append(pts)



def create_mask(H, W):
    mask = np.zeros((H, W), dtype=np.uint8)

    for i in range(20):
        
        poly_mask = np.zeros((H, W), dtype=np.uint8)
        poly = random.choice(mono_polygon_arrays)  # Same shape both times

        poly_px = (poly * [W, H]).astype(np.float32)

        min_x, min_y = poly_px.min(axis=0)
        max_x, max_y = poly_px.max(axis=0)

        min_shift_x = -min_x
        max_shift_x = W - max_x
        min_shift_y = -min_y
        max_shift_y = H - max_y

        shift_x = random.uniform(min_shift_x, max_shift_x)
        shift_y = random.uniform(min_shift_y, max_shift_y)

        poly_shifted = poly_px + [shift_x, shift_y]

        pts = poly_shifted.astype(np.int32).reshape((-1, 1, 2))

        cv2.fillPoly(poly_mask, [pts], color=1)
        
        mask = mask + poly_mask

    return mask
    
mask = create_mask(512, 512)


print(f"Mask stats: Min={mask.min()}, Max={mask.max()}, Unique values={np.unique(mask)}")

# Simple grayscale visualization (darker = more layers)
plt.figure(figsize=(8, 8))
plt.imshow(mask, cmap='gray_r')
plt.title(f"Grayscale Mask (Darker = More Layers)\nMax: {mask.max()} layers")
plt.colorbar(label='Layer Count')
plt.axis('off')
plt.show()
