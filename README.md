
# Plant Disease Classification with GAN, DINOv2, HGWOS, and MoE

This repository implements a comprehensive methodology for **plant disease classification** using a combination of advanced techniques:  
- **GAN-based augmentation** (Identity-Enhanced Generative Adversarial Network - IE-GAN)  
- **Feature extraction using DINOv2-ViT-S** (self-supervised Vision Transformer)  
- **Hybrid Grey Wolf–Whale Optimization Strategy (HGWOS)** for feature selection  
- **Mixture of Experts (MoE)** for robust classification  

The framework is designed to handle imbalanced datasets, improve accuracy in disease detection, and provide interpretability.

---

## Table of Contents
- [Overview](#overview)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
  - [Step 1: Train IE-GAN](#step-1-train-ie-gan)
  - [Step 2: Generate Synthetic Data](#step-2-generate-synthetic-data)
  - [Step 3: Feature Extraction](#step-3-feature-extraction)
  - [Step 4: HGWOS Feature Selection](#step-4-hgwos-feature-selection)
  - [Step 5: Train MoE Classifier](#step-5-train-moe-classifier)
  - [Running the Full Pipeline](#running-the-full-pipeline)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)

---

## Overview

This project aims to classify plant diseases using deep learning and optimization algorithms. The methodology consists of:

1. **IE-GAN** is used for data augmentation to generate synthetic plant images, especially for underrepresented disease classes.
2. **DINOv2-ViT-S** is used to extract deep feature representations from plant images.
3. **HGWOS (Hybrid Grey Wolf–Whale Optimization Strategy)** is applied for feature selection, improving classifier performance by selecting optimal features.
4. **Mixture of Experts (MoE)** is used for final classification, utilizing multiple expert networks combined by a gating network.

The overall framework improves the robustness of disease classification while handling class imbalances and optimizing computational efficiency.

---

## Technologies

This project uses the following technologies:
- **PyTorch** for deep learning model development and training
- **TorchVision** for image transformations and pre-trained models
- **NumPy** for numerical operations
- **Matplotlib** for plotting and visualizing results
- **Scikit-learn** for machine learning models and metrics

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/plant-disease-classification.git
   cd plant-disease-classification
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the appropriate versions of the following libraries:
   - Python 3.7+
   - PyTorch (>=2.0.0)
   - TorchVision (>=0.15.0)
   - CUDA (if using GPU)

---

## Usage

### Step 1: Train IE-GAN

Train the **Identity-Enhanced GAN (IE-GAN)** to augment the dataset. This step generates synthetic images that help balance underrepresented plant disease classes.

```bash
python scripts/train_ie_gan.py
```

This script will train the **Generator** and **Discriminator**, saving the model weights to `outputs/checkpoints/ie_gan.pt`.

---

### Step 2: Generate Synthetic Data

Generate synthetic images using the trained **IE-GAN** model. This is useful for augmenting your training dataset.

```bash
python scripts/generate_synthetic.py
```

Synthetic images will be saved to `outputs/synthetic/`.

---

### Step 3: Feature Extraction

Use the **DINOv2-ViT-S** model to extract features from your plant images. This step generates a 384-dimensional feature vector for each image using the pretrained model.

```bash
python scripts/extract_dinov2_features.py
```

The extracted features will be saved to `outputs/features/`.

---

### Step 4: HGWOS Feature Selection

Apply the **Hybrid Grey Wolf–Whale Optimization Strategy (HGWOS)** to select the optimal subset of features. This step uses optimization algorithms to reduce feature dimensionality and improve classification accuracy.

```bash
python scripts/run_hgwos_feature_selection.py
```

This will save the selected feature mask to `outputs/features/best_mask.npy`.

---

### Step 5: Train MoE Classifier

Train the **Mixture of Experts (MoE)** classifier using the selected features. This classifier combines multiple experts for improved performance and robustness.

```bash
python scripts/train_moe.py
```

The trained model will be saved to `outputs/checkpoints/moe.pt`.

---

### Running the Full Pipeline

To run the entire pipeline from start to finish, you can execute the following command, which will sequentially train the IE-GAN, generate synthetic data, extract features, perform feature selection, and train the MoE classifier.

```bash
python main_pipeline.py
```

This will automatically call all the necessary scripts and save intermediate results in the `outputs` folder.

---

## Project Structure

```
plant_pipeline/
│
├── config.py                # Configuration file with hyperparameters
├── data.py                  # Data loading and preprocessing
├── ie_gan.py                # IE-GAN model implementation
├── dino_features.py         # DINOv2 feature extraction script
├── hgwos.py                 # HGWOS optimization algorithm
├── moe.py                   # MoE classifier implementation
├── train_moe.py             # MoE training script
├── run_pipeline.py          # Run full pipeline
│
├── scripts/                 # Script directory
│   ├── train_ie_gan.py      # IE-GAN training
│   ├── generate_synthetic.py # Synthetic image generation
│   ├── extract_dinov2_features.py # Feature extraction
│   ├── run_hgwos_feature_selection.py # Feature selection via HGWOS
│   └── train_moe.py         # MoE classifier training
│
└── requirements.txt         # Project dependencies
```

---

## Results

The framework has been tested on the following datasets:
- **PlantDoc**
- **PlantVillage**
- **FieldPlant**


## Contributing

Feel free to fork this repository and submit pull requests. If you have any questions or suggestions for improvements, please open an issue or contact me directly.


