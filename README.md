# Hybrid-Mamba-Network-with-Dual-Level-Attention-Fusion
This repository contains the official implementation of:A Hybrid-Mamba Network with Dual Level Attention Fusion for Multimodal COPD Diagnosis

This work proposes a Hybrid-Mamba architecture that integrates CT imaging and clinical data, achieving high diagnostic accuracy, robustness, and inference efficiency for COPD classification.

Highlights

First application of Mamba-based modeling in multimodal COPD diagnosis

Hybrid backbone combining convolution, Mamba state-space modeling, and attention

Dual-Level Attention Fusion for adaptive integration of CT and clinical data

Efficient inference (≈97.5 ms per patient) with strong robustness

Supports capacitated feature extraction, multimodal fusion, and interpretability (CAM)

Network Overview

The framework consists of:

1️⃣ CT Feature Extraction

Hybrid Mamba backbone with four stages:

Multi-Scale Squeeze-and-Excitation Block (MSEB)

Hybrid-DWConv-AAS Block

Axial-Attention Block

Captures both local textures & long-range dependencies

2️⃣ Clinical Feature Extraction

Random Forest feature selection

Standardization and FC encoding

3️⃣ Multimodal Fusion

Dual-Level Attention Fusion Block:

Slice-level self-attention aggregation

Cross-modal fusion with gated weighting

4️⃣ CAM Visualization

Generates interpretable heatmaps to highlight pathological CT regions

Repository Structure
Hybrid-Mamba-COPD/
│
├── data/
│   ├── ct/                      # Preprocessed CT slices
│   ├── clinical.csv             # Clinical feature table
│   └── splits/                  # Train/val/test index files
│
├── models/
│   ├── mamba_backbone.py        # Hybrid Mamba + axial attention backbone
│   ├── mseb.py                  # Multi-scale SE block
│   ├── dwconv_aas.py            # Hybrid-DWConv-AAS block
│   ├── fusion_module.py         # Dual-level attention fusion
│   └── classifier.py
│
├── train.py                     # Training entry
├── test.py                      # Evaluation entry
├── predict_single.py            # Inference on individual patients
│
├── utils/
│   ├── metrics.py               # AUC, precision, recall, F1
│   ├── dataset.py               # CT + clinical dataset loader
│   └── cam.py                   # CAM generation
│
├── requirements.txt
└── README.md

Installation

conda create -n mamba-copd python=3.10
conda activate mamba-copd
pip install -r requirements.txt

training
python train.py \
    --data ./data \
    --batch 16 \
    --lr 1e-4 \
    --folds 5

Default Settings

Train/Val/Test split: 7 : 1 : 2
5-fold cross-validation within train+val
Early stopping enabled

Testing
python test.py --data ./data --weights checkpoint.pth
