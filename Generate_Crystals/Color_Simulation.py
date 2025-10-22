import numpy as np
import matplotlib.pyplot as plt
from T import *
import cv2 
def calculate_reflectance_TMM(M_total, n_incident=1.0, n_substrate=3.5+0.01j):
    
 
    # Extract matrix elements
    M11, M12 = M_total[0, 0], M_total[0, 1]
    M21, M22 = M_total[1, 0], M_total[1, 1]
    
    # Calculate reflection coefficient with boundary conditions
    numerator = (n_incident * M11 + n_incident * n_substrate * M12 - M21 - n_substrate * M22)
    denominator = (n_incident * M11 + n_incident * n_substrate * M12 + M21 + n_substrate * M22)
    
    r = numerator / denominator
    
    # Reflectance is |r|²
    R = np.abs(r)**2
    
    return R
    
    
def create_Matrix(wavelength, ni, ti):
  
    internals = (2 * np.pi * ni * ti) / wavelength
    

    cos = np.cos(internals)
    sin = np.sin(internals)
    
    # Build the 2x2 characteristic matrix
    Mi = np.array([
        [cos,  (1j * sin) / ni],
        [1j * ni * sin,  cos]
    ])
    
    return Mi
    
def TMM_reflectance(wavelength, ref_index, thickness, is_material, is_background, g_layers):
    ni_air, ni_silicon, ni_SIO2, ni_material = ref_index
    ti_air, ti_silicon, ti_SIO2, ti_material = thickness
    ti_material *= g_layers

    wavelength = np.atleast_1d(wavelength)
    R_values = []  # IMPORTANT: Store all wavelengths
    
    if is_material:
        for w in wavelength:
            M_material = create_Matrix(w, ni_material, ti_material)
            M_SIO2 = create_Matrix(w, ni_SIO2, ti_SIO2)
            M_total = np.dot(M_material, M_SIO2)  # Graphene first, then SiO2
            R = calculate_reflectance_TMM(M_total, ni_air, ni_silicon)
            R_values.append(R)
        return np.array(R_values)

    elif is_background:
        for w in wavelength:
            M_SIO2 = create_Matrix(w, ni_SIO2, ti_SIO2)
            R = calculate_reflectance_TMM(M_SIO2, ni_air, ni_silicon)
            R_values.append(R)
        return np.array(R_values)
    
wavelength = np.linspace(380, 780, 401)

ref_index = [1.0, 3.5+0.01j, 1.46, 3.0+1.3j]  # [air, Si, SiO2, graphene]
thickness = [np.inf, np.inf, 285e-9, 0.34e-9]   # [air, Si, SiO2, graphene]

R = TMM_reflectance(wavelength, ref_index, thickness, True, False, 1)

def pixel_reflectance(pixel_value, ref_index, thickness):
    """Calculate reflectance spectrum for a pixel - pass ALL wavelengths at once"""
    if pixel_value != 0:
        return TMM_reflectance(wavelength, ref_index, thickness, True, False, pixel_value)
    else:
        return TMM_reflectance(wavelength, ref_index, thickness, False, True, pixel_value)


H, W = 512, 512

mask = create_mask(H, W)


Illumination_array_mu = np.array([446.34, 448.29, 530.7, 577.1])
Illumination_array_pi = np.array([7.71, 20.09, 13.22, 69.22])
Illumination_array_sigma = np.array([6.98, 15.01, 22.45, 49.95])
Illumination_array_c = np.array([-0.0])


# --- Blue channel ---
RGB_array_b_mu = np.array([453.64, 482.32, 605.0])
RGB_array_b_pi = np.array([61.7, 4.81, -19.39])
RGB_array_b_sigma = np.array([41.69, 19.84, 64.21])
RGB_array_b_c = np.array([0.14])

# --- Green channel ---
RGB_array_g_mu = np.array([390.29, 488.26, 516.14, 569.29, 653.64])
RGB_array_g_pi = np.array([108.74, 13.56, 39.37, 87.48, 15.61])
RGB_array_g_sigma = np.array([134.16, 14.93, 24.27, 37.22, 22.32])
RGB_array_g_c = np.array([-0.28])

# --- Red channel ---
RGB_array_r_mu = np.array([367.84, 525.26, 588.49, 609.13, 646.75])
RGB_array_r_pi = np.array([-7.42, -1.8, 13.52, 21.11, 48.58])
RGB_array_r_sigma = np.array([-31.83, -13.53, 11.78, 16.38, 25.12])
RGB_array_r_c = np.array([0.03])


Illumination_spectrum = 0
Red_channel = 0
Green_channel = 0
Blue_channel = 0

def Gaussian(mean, standard_dev, amplitude, wavelength):
    fraction = 1 / (abs(standard_dev) * np.sqrt(2 * np.pi))
    exp_term = np.exp(-((wavelength - mean)**2) / (2 * standard_dev**2))  # No 1e9 conversion!
    return amplitude * fraction * exp_term


for i in range(len(Illumination_array_mu)):
    Illumination_spectrum += Gaussian(Illumination_array_mu[i], Illumination_array_sigma[i], Illumination_array_pi[i], wavelength)

for i in range(len(RGB_array_r_mu)):
    Red_channel += Gaussian(RGB_array_r_mu[i], RGB_array_r_sigma[i], RGB_array_r_pi[i], wavelength)

for i in range(len(RGB_array_g_mu)):
    Green_channel += Gaussian(RGB_array_g_mu[i], RGB_array_g_sigma[i], RGB_array_g_pi[i], wavelength)

for i in range(len(RGB_array_b_mu)):
    Blue_channel += Gaussian(RGB_array_b_mu[i], RGB_array_b_sigma[i], RGB_array_b_pi[i], wavelength)


Illumination_spectrum += Illumination_array_c[0]
Red_channel += RGB_array_r_c[0]
Blue_channel += RGB_array_b_c[0]
Green_channel += RGB_array_g_c[0]

RGB_channel = [Red_channel,  Blue_channel, Green_channel]



def color_per_pixel(pixel_value, ref_index, thickness, Illumination_spectrum, RGB_channel, wavelength):
    # Get reflectance spectrum (should be shape (401,))
    Ref = pixel_reflectance(pixel_value, ref_index, thickness)
    
    # Use SUM instead of trapz, matching the paper
    Red = np.sum((Ref * Illumination_spectrum) * RGB_channel[0])
    Green = np.sum((Ref * Illumination_spectrum) * RGB_channel[1])
    Blue = np.sum((Ref * Illumination_spectrum) * RGB_channel[2])
    
    return [Red, Green, Blue]


mask = create_mask(512, 512)



