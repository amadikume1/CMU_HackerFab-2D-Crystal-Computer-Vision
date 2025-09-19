# CMU_HackerFab-2D-Crystal-Computer-Vision
 A computer vision system for automatic identification, characterization and storage of 2D crystals in microscopy images. This tool enables analysis of crystal samples by detecting, classifying, and measuring crystal properties from optical microscopy data.

# Installation and usage
### Installation
Clone the repository and install dependencies:
`git clone https://github.com/amadikume1/CMU_HackerFab-2D-Crystal-Computer-Vision.git`
`cd CMU_HackerFab-2D-Crystal-Computer-Vision`
`# Create and activate environment`
`conda create -n crystalcv python=3.10`
`conda activate crystalcv`
`# Install requirements`
`pip install -r requirements.txt`

### Usage
Run the main script with an input image:
`python run.py`
Then, the denoised graphene images will be saved in the `outputs/` directory and the extracted shape features will be saved in `Database/substrate_database`. To obtain the database of substrate features, run:
`python database_test.py`
The database will be saved as `substrate.csv` in the `Database/` directory.

### (In the future) Neural Network Training
To train and test the neural network, use the training script:
`python train.py --config configs/default.yaml`
`python test.py --input images/sample_images/ --model checkpoints/model.pth`
Options can be listed with `--help`.

### Implementation & System Requirements
Implementation:
- Written in Python, using PyTorch for deep learning.
- Utilizes OpenCV for image processing tasks.
- Employs scikit-learn for clustering algorithms.
- Includes modular code: training, evaluation, visualization, and helper scripts.
- Dataset handling for microscope images in standard formats (.png/.jpg).
System Requirements:
- Python 3.8 or higher.
- PyTorch 1.7 or higher with CUDA support for GPU acceleration.
- OpenCV 4.5 or higher.
- numpy, scikit-learn, matplotlib libraries.
- A CUDA-capable GPU is recommended for training the neural network.

