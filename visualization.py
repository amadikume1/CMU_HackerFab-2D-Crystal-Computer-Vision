#!/usr/bin/env python3
"""
Visualize Sobel edge enhancement on a single image.

Usage:
    python visualize_sobel_edges.py
"""

import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import save_image
import os

# ------------------------------------------------------------
# === Sobel Kernels ===
Sx = th.tensor([[1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]], dtype=th.float32).view(1, 1, 3, 3)
Sy = th.tensor([[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]], dtype=th.float32).view(1, 1, 3, 3)

def sobel(img):
    """Apply Sobel filter per channel."""
    B, C, H, W = img.shape
    Sx_c = Sx.to(img.device).repeat(C, 1, 1, 1)
    Sy_c = Sy.to(img.device).repeat(C, 1, 1, 1)
    dx = F.conv2d(img, Sx_c, padding=1, groups=C)
    dy = F.conv2d(img, Sy_c, padding=1, groups=C)
    return dx, dy


def gaussian_blur(img, sigma_px=1.0):
    """Depthwise separable Gaussian blur, multi-channel."""
    B, C, H, W = img.shape
    radius = max(1, int(3 * sigma_px))
    xs = th.arange(-radius, radius + 1, device=img.device, dtype=img.dtype)
    k1d = th.exp(-0.5 * (xs / sigma_px) ** 2)
    k1d = (k1d / k1d.sum()).view(1, 1, -1)

    kx = k1d.view(1, 1, 1, -1).repeat(C, 1, 1, 1)  # horizontal
    ky = k1d.view(1, 1, -1, 1).repeat(C, 1, 1, 1)  # vertical

    out = F.conv2d(img, kx, padding=(0, radius), groups=C)
    out = F.conv2d(out, ky, padding=(radius, 0), groups=C)
    return out


# ------------------------------------------------------------
# === Batch Processing Function ===

def process_directory(input_dir):
    save_dir = "sobel_vis_results_new"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Processing directory: {input_dir}")
    print(f"Saving outputs to: {save_dir}")

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(input_dir, fname)
        print(f"\n[Processing] {fname}")

        # Load image
        img = read_image(img_path).float() / 255.0  # [C,H,W]
        img = img.unsqueeze(0)  # [1,C,H,W]

        # --- Sobel edge computation ---
        dx, dy = sobel(img)
        mag = (dx**2 + dy**2).sqrt().mean(1, keepdim=True)

        # Normalize to 0–1 for visibility
        mag = mag / (mag.max() + 1e-8)

        # GOOD binary mask for downstream feature extraction:
        # convert Sobel magnitude to strong edge image
        mag_norm = (mag[0,0] * 255).byte()  # uint8
        mag_np = mag_norm.cpu().numpy()

        # threshold ANY weak edge
        import cv2
        _, binary = cv2.threshold(mag_np, 5, 255, cv2.THRESH_BINARY)
        binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 2)

        # --- Sharpen image ---
        sharp_factor = 20
        sharpened = (img + sharp_factor * mag).clamp(0, 1)

        prefix = os.path.splitext(fname)[0]

        # --- Save outputs ---
        save_image(img, os.path.join(save_dir, f"{prefix}_original.png"))
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_binary.png"), binary)
        save_image(sharpened, os.path.join(save_dir, f"{prefix}_sharpened.png"))

        print(f"[Saved] {prefix}_original.png")
        print(f"[Saved] {prefix}_binary.png")
        print(f"[Saved] {prefix}_sharpened.png")

    print("\n[Done] Sobel visualization for directory complete.")


# ------------------------------------------------------------
# RUN
if __name__ == "__main__":
    input_dir = "Input_images"
    process_directory(input_dir)
