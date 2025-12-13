# CMU_HackerFab-2D-Crystal-Computer-Vision

This repository contains a computer vision pipeline for automatic identification, enhancement, and characterization of 2D crystal flakes from optical microscopy images.
It serves as the perception and data-extraction module of the CMU Hacker Fab 2D Crystal Stacking Robot, converting raw microscope images into structured, machine-readable flake data for downstream CAD design and robotic assembly.
The pipeline emphasizes robust image preprocessing, interpretable feature extraction, and reproducible data storage, rather than black-box detection alone.

---

## Installation and Usage

### Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/amadikume1/CMU_HackerFab-2D-Crystal-Computer-Vision.git
cd CMU_HackerFab-2D-Crystal-Computer-Vision

# Create and activate environment
conda create -n crystalcv python=3.10
conda activate crystalcv

# Install requirements
pip install -r requirements.txt
```

---

### Usage

#### Baseline + Feature Extraction Pipeline
Run the main script with an input image:
```bash
python run.py
```

This performs:
- image preprocessing and enhancement
- mask generation and contour extraction
- clustering-based noise filtering
- feature extraction and database export

Outputs:
- The **denoised graphene images** will be saved in the `outputs/` directory.  
- Sobel edge visualizations will be saved in `Sobel_vis_results/`.
- Color segmentation + K-means visualizations will be saved in `Color_seg_kmeans/`.
- Extracted **shape and texture features** will be saved in `Database/substrate_database`.  
- To generate a searchable database of substrate features:
  ```bash
  python database_test.py
  ```

---

#### Diffusion-Based Denoising Module
This project integrates an unsupervised **score-matching diffusion model** to enhance image quality before feature extraction and detection.  
The implementation is adapted from [TheodoreChiYu/unsupervised_denoising_score_function](https://github.com/TheodoreChiYu/unsupervised_denoising_score_function).

**Workflow Overview (Christina’s Pipeline):**
1. Store raw microscopy data in `files/data/`  
2. Split dataset → 24 train / 3 val / 3 test (total 30 images)  
3. Choose appropriate noise model and adjust config (`configs/default.yaml`)  
4. Apply the diffusion model for denoising and save results in `results/denoise/`  

Example:
```bash
python train.py --config configs/default.yaml
python test.py --input data/sample_images/ --model checkpoints/model.pth
```

The denoising stage reduces Poisson-Gaussian microscope noise while preserving fine flake boundaries, enabling more stable contour extraction and improved downstream feature quality.

---

## Implementation & System Requirements

### Implementation Details
- Written in **Python**, using **PyTorch** for deep learning and diffusion training  
- Uses **OpenCV** for image processing tasks (color/entropy thresholding)  
- Employs **scikit-learn** for clustering and analysis  
- Includes modular scripts for training, evaluation, visualization, and database operations  
- Handles microscope images in standard `.png` and `.jpg` formats  

### System Requirements
- **Python:** 3.8 or higher  
- **PyTorch:** 1.12 or higher (with CUDA support for GPU acceleration)  
- **OpenCV:** 4.5 or higher  
- **scikit-image, scikit-learn, numpy, matplotlib**  
- **PyYAML, tqdm, pandas**  
- A **CUDA-capable GPU** is strongly recommended for diffusion model training

---

## Project Workflow

```
Raw Images (data/)
        ↓
Diffusion-Based Denoising
        ↓
Edge Enhancement (Sobel)
        ↓
Mask Generation
        ↓
Contour Extraction
        ↓
K-Means Clustering (Noise vs Flakes)
        ↓
Feature Extraction (Geometry + Appearance)
        ↓
SQLite / CSV Database
```

---

## Contributors
- **Xiaoqi Wu** — Diffusion denoising, integration, testing, and analysis  
- **Amadi Ume** — Graphene dataset generation and YOLOv8 detection  
- **Hacker Fab Team @ Carnegie Mellon University**

---

## Citation
If you use this repository or its diffusion-based denoising adaptation, please cite:

> Theodore Yu, *“Unsupervised Denoising via Score Function Matching”*, 2021.  
> [GitHub Repository](https://github.com/TheodoreChiYu/unsupervised_denoising_score_function)

---

## requirements.txt
Place the following dependencies in your `requirements.txt` file:

```
torch>=1.12.0
torchvision>=0.13.0
torchaudio
numpy>=1.23.0
scipy>=1.10.0
matplotlib>=3.7.0
opencv-python>=4.5.0
scikit-image>=0.21.0
scikit-learn>=1.2.0
pillow>=9.0.0
tqdm>=4.65.0
pyyaml>=6.0
h5py>=3.8.0
pandas>=1.5.0
sqlite3-binary
```
