Overview

    This project simulates the optical color of thin-film stacks, specifically graphene on SiO₂/Si substrates, using the Transfer Matrix Method (TMM).
    It reproduces interference-based color appearance seen under optical microscopes — the same principle used in the MaskTerial paper and related thin-film color synthesis works.
    
    Each simulation computes the wavelength-dependent reflectance spectrum, then integrates it with camera response and illumination profiles (fit from experimental data) to yield an RGB value that matches what a real camera would capture.


Features

    Physically accurate TMM optical model for multilayer thin films
    
    Gaussian mixture fits for camera spectral sensitivity and illumination spectra, matching MaskTerial
    
    Configurable oxide thickness, graphene layer count, and white balance correction

  Physical Background

    The simulation uses the Transfer Matrix Method, which models the propagation of light through stratified media.
    <img width="294" height="171" alt="image" src="https://github.com/user-attachments/assets/67ef8ff5-1c37-4d2c-b2a7-f4e53a639990" />

    The reflectance is integrated with an illumination spectrum and RGB camera response functions
  
    <img width="413" height="77" alt="image" src="https://github.com/user-attachments/assets/bef2ec52-c68f-4cf9-8334-141c89f23401" />

  Input Files
    Refractive Index Files (data/refractive_indices/*.txt)
    
    Text files with n and k values per wavelength:

  JSON Files

    rgb_gauss_fit.json: Gaussian mixture fits for R,G,B camera response curves
    
    spectrum_gauss_fit.json: Gaussian mixture fit for the illumination profile (e.g., white LED or halogen lamp)

  Dependencies

      NumPy	Array math and complex arithmetic
      Matplotlib	Reflectance and color visualization
      OpenCV (cv2)	Mask or image operations
      JSON	Load camera and illumination fits

  Note

    actiuvly testing and improving upon algorithm, using the Paper's own code as a refrence
