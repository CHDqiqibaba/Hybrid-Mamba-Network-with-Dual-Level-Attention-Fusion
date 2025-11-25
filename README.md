# Hybrid-Mamba Network for Multimodal COPD Diagnosis

This repository contains the official implementation of:

**A Hybrid-Mamba Network with Dual Level Attention Fusion for Multimodal COPD Diagnosis**

The proposed framework integrates CT imaging and clinical data using a Hybrid-Mamba backbone, achieving accurate, efficient, and robust COPD classification.

---

## 🚀 Highlights

- Hybrid architecture combining convolution, Mamba state-space modeling, and attention
- Dual-Level Attention Fusion for integrating CT and clinical variables
- Computationally efficient (≈97.5 ms per patient)
- Strong diagnostic performance and robustness under perturbations
- Supports interpretability through CAM visualization

---

## 🧠 Network Overview

### 1. CT Feature Extraction
- Mamba-powered encoder with:
  - Multi-Scale Squeeze-and-Excitation Block (MSEB)
  - Hybrid-DWConv-AAS Block
  - Axial-Attention Block

### 2. Clinical Feature Encoding
- Random forest feature selection  
- Standardization and fully connected embedding

### 3. Multimodal Fusion
- Dual-Level Attention Fusion Block  
- Cascaded self-attention + cross-attention  
- Adaptive weighting of imaging and clinical sources

### 4. Interpretability
- Class activation maps (CAM) for regional visualization

---

## 📁 Repository Structure

```
Hybrid-Mamba-COPD/
│
├── data/
│   ├── ct/                    # CT slices
│   ├── clinical.csv           # Clinical variables
│   └── splits/                # Train/val/test split indices
│
├── models/
│   ├── mamba_backbone.py      # Hybrid Mamba backbone
│   ├── mseb.py                # Multi-scale SE block
│   ├── dwconv_aas.py          # Hybrid-DWConv-AAS block
│   ├── fusion_module.py       # Cross-modal fusion
│   └── classifier.py          # Final classifier
│
├── utils/
│   ├── metrics.py             # AUC, precision, recall, F1
│   ├── dataset.py             # Dataset loader
│   └── cam.py                 # CAM visualization
│
├── train.py                   # Training pipeline
├── test.py                    # Evaluation script
├── predict_single.py          # Single case inference
│
├── requirements.txt           # Package dependencies
└── README.md
```

## 📦 Installation
```bash
conda create -n mamba-copd python=3.10
conda activate mamba-copd
pip install -r requirements.txt

## 📦 Dependencies include:
torch>=2.1
numpy
scikit-learn
opencv-python
matplotlib
pydicom


## 📂 Dataset Description

### CT Data
- 30 uniform slices per subject  
- Lung windowing and segmentation  
- Standard preprocessing  

### Clinical Data
- Demographic information  
- Pulmonary function  
- Blood gas and laboratory indicators  
- Random Forest feature selection  


## 🧪 Experimental Setup
- Train / Validation / Test split: **70% / 10% / 20%**
- Within train + validation:
  - **5-fold cross-test**
- Early stopping enabled
- 30 slices used as CT input

## 🧪 Training
python train.py \
    --data ./data \
    --batch 16 \
    --lr 1e-4 \
    --folds 5

# 🧪 Evaluation
python test.py --data ./data --weights checkpoint.pth
