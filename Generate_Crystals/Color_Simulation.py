"""
simulate_reflectance_TMM.py
------------------------------------------------------------
Author: Amadi Ume
Last Updated: [insert date]

Purpose:
    Simulate the optical color of a graphene-on-SiO₂/Si substrate
    using the Transfer Matrix Method (TMM). This reproduces
    the MaskTerial thin-film color model, combining physical
    interference simulation with Gaussian-fitted illumination
    and camera sensitivity curves.

Use case:
    - Validate substrate color for a given SiO₂ thickness
    - Evaluate apparent graphene contrast
    - Generate physically accurate synthetic data

Dependencies:
    numpy, matplotlib, opencv (for mask generation)
    create_mask.py (provides spatial layer mask)
    data/rgb_gauss_fit.json
    data/spectrum_gauss_fit.json
------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from create_mask import *
import cv2


# Gaussian-based Illumination and Camera Modeling(From Paper)

def gauss(x, mu, sigma):
    """Single Gaussian function."""
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def gauss_fit_5(
    x,
    pi1, mu1, sigma1,
    pi2, mu2, sigma2,
    pi3=0, mu3=1, sigma3=1,
    pi4=0, mu4=1, sigma4=1,
    pi5=0, mu5=1, sigma5=1,
    c=0,
):
    """Five-component Gaussian mixture — identical to MaskTerial."""
    return (
        pi1 * gauss(x, mu1, sigma1)
        + pi2 * gauss(x, mu2, sigma2)
        + pi3 * gauss(x, mu3, sigma3)
        + pi4 * gauss(x, mu4, sigma4)
        + pi5 * gauss(x, mu5, sigma5)
        + c
    )

def load_camera_activation(path, wavelengths):
    """Load RGB camera response (from rgb_gauss_fit.json)."""
    with open(path, "r") as f:
        rgb_params = json.load(f)
    r = gauss_fit_5(wavelengths, **rgb_params["r"])
    g = gauss_fit_5(wavelengths, **rgb_params["g"])
    b = gauss_fit_5(wavelengths, **rgb_params["b"])
    return np.stack((r, g, b), axis=0)  # shape (3, W): R,G,B

def load_spectrum(path, wavelengths):
    """Load illumination spectrum (from spectrum_gauss_fit.json)."""
    with open(path, "r") as f:
        spectrum_params = json.load(f)
    return gauss_fit_5(wavelengths, **spectrum_params)



# Core Optical Physics: Transfer Matrix Method (TMM) 

def calculate_reflectance_TMM(M_total, n_incident=1.0, n_substrate=3.88 + 0.02j):
    """Compute reflectance R = |r|² from total characteristic matrix."""
    M11, M12 = M_total[0, 0], M_total[0, 1]
    M21, M22 = M_total[1, 0], M_total[1, 1]

    numerator = (n_incident * M11 + n_incident * n_substrate * M12
                 - M21 - n_substrate * M22)
    denominator = (n_incident * M11 + n_incident * n_substrate * M12
                   + M21 + n_substrate * M22)
    r = numerator / denominator
    return np.abs(r) ** 2


def create_Matrix(wavelength, ni, ti):
    """Construct characteristic matrix for one thin-film layer."""
    # Convert wavelength to meters for unit consistency
    w_m = wavelength * 1e-9
    phase = (2 * np.pi * ni * ti) / w_m
    cos = np.cos(phase)
    sin = np.sin(phase)
    return np.array([[cos, (1j * sin) / ni],
                     [1j * ni * sin, cos]])


def TMM_reflectance(wavelength, ref_index, thickness,
                    is_material, is_background, g_layers):
    """Compute wavelength-dependent reflectance for multilayer stack."""
    n_air, n_si, n_sio2, n_g = ref_index
    t_air, t_si, t_sio2, t_g = thickness
    t_g *= g_layers
    wavelength = np.atleast_1d(wavelength)
    R_values = []

    if is_material:
        for w in wavelength:
            M_g = create_Matrix(w, n_g, t_g)
            M_sio2 = create_Matrix(w, n_sio2, t_sio2)
            M_total = np.dot(M_g, M_sio2)
            R_values.append(calculate_reflectance_TMM(M_total, n_air, n_si))
    elif is_background:
        for w in wavelength:
            M_sio2 = create_Matrix(w, n_sio2, t_sio2)
            R_values.append(calculate_reflectance_TMM(M_sio2, n_air, n_si))

    return np.array(R_values)


# Simulation Parameters 

wavelength = np.linspace(380, 780, 401)  # nm range
ref_index = [1.0, 3.88 + 0.02j, 1.46, 2.6 + 1.3j]  # realistic indices
thickness = [np.inf, np.inf, 90e-9, 0.34e-9]      # 285 nm oxide


# Reflectance Spectrum to Pixel Color 

def pixel_reflectance(pixel_value, ref_index, thickness):
    """Return reflectance spectrum depending on pixel mask."""
    if pixel_value != 0:
        return TMM_reflectance(wavelength, ref_index, thickness, True, False, pixel_value)
    else:
        return TMM_reflectance(wavelength, ref_index, thickness, False, True, pixel_value)

# Load Illumination & Camera Spectra (from JSON)

illumination_path = "data/spectrum_gauss_fit.json"
camera_path = "data/rgb_gauss_fit.json"

Illumination_spectrum = load_spectrum(illumination_path, wavelength)
RGB_channel = load_camera_activation(camera_path, wavelength)

# Normalize each component
Illumination_spectrum /= np.max(Illumination_spectrum)
RGB_channel /= np.max(RGB_channel, axis=1, keepdims=True)



# Integrate Reflectance × Spectra → RGB 

def color_per_pixel(pixel_value, ref_index, thickness,
                    Illumination_spectrum, RGB_channel, wavelength):
    """Integrate R(λ) × S(λ) × C_c(λ) dλ for each RGB channel."""
    Ref = pixel_reflectance(pixel_value, ref_index, thickness)
    RGB = np.trapz(Ref * Illumination_spectrum * RGB_channel, wavelength, axis=1)
    return RGB


# === Compute and Display SIO2 Substrate Color 

background_rgb = color_per_pixel(
    0, ref_index, thickness, Illumination_spectrum, RGB_channel, wavelength
)
background_rgb = np.array(background_rgb, dtype=float)
background_rgb /= np.max(background_rgb)

# --- Auto White Balance (Gray-world assumption) ---
scene_gain = np.trapz(Illumination_spectrum * RGB_channel, wavelength, axis=1)
wb_auto = 1.0 / np.maximum(scene_gain, 1e-12)
wb_auto /= wb_auto.max()
background_rgb *= wb_auto
background_rgb = np.clip(background_rgb / np.max(background_rgb), 0, 1)

# --- Gamma Correction ---
background_rgb_gamma = background_rgb ** (1 / 2.2)

print("Simulated SiO₂ substrate RGB (after white balance):", background_rgb_gamma)

# --- Display ---
plt.imshow([[background_rgb_gamma]])
plt.axis("off")
plt.title("Simulated SiO₂ Background Color")
plt.show()
