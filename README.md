
# Exploring Sample Quality for Face Presentation Attack Detection (PAD)

## Project Overview

This repository contains the official implementation of the paper:

**"The Impact of Sample Quality Stratification on Generalization in Face Presentation Attack Detection"**
*Eva María Benito Sanz, Biometrics Systems*

We introduce a quality-aware protocol for training Face Presentation Attack Detection (PAD) models, which improves generalization to unseen domains and attacks by leveraging a custom sample quality metric (QPAD) and stratified training.

## Motivation

PAD systems often suffer from poor generalization across domains due to sample quality variation. This project addresses this issue by:

* Defining a new PAD-specific quality metric (QPAD)
* Stratifying data into Low (LQ), Medium (MQ), and High (HQ) quality tiers
* Evaluating  cross-dataset.
* Enhancing performance using causal learning (CF-PAD)

## Repository Structure

```
BIOMETRICDTU2025/
├── CF-PAd/                             # Main implementation: training, evaluation, datasets, metrics
│   ├── checkpoints/                   # Saved model checkpoints
│   ├── confusion_matrices/           # Confusion matrix plots
│   ├── datasets/                     # Custom PyTorch dataset loading logic
│   ├── logs/                         # Training and evaluation logs
│   ├── metrics/                      # Implementation of evaluation metrics 
│   ├── quality_split_images_casia_train/     # CASIA dataset split into quality tiers (image folders)
│   ├── quality_splits_LCC_traaining/         # LCC dataset splits into quality tiers (text file lists)
│   ├── results/                      # Output: result CSVs, misclassified images, tables
│   ├── count.py                      # Counts samples per quality category
│   ├── dataset.py                    # Dataset loading logic and preprocessing
│   ├── eval.py                       # Evaluation pipeline
│   ├── eval_only_runner.py           # Evaluation runner without training
│   ├── folder_high_quality_best_model.pth    # Example pre-trained model checkpoint
│   ├── model.py                      # PAD model
│   ├── script-runner.py              # Main script for cross-domain training and evaluation
│   ├── train.py                      # Training script
│   └── utils.py                      # General utility functions
├── confusion_matrices/               # Global plots and matrix visualizations
├── img_lists_LCC/                    # List of LCC images by quality
├── logs/                             # Global logs
├── Magface_v2/                       # QPAD feature extraction and quality scoring (MagFace-based)
│   ├── models/                       # MagFace model loading and configs
│   ├── preprocessing_v1/            # Image preprocessing (normalization, resizing)
│   ├── quality_split_images copy/         # Temporary or backup CASIA splits
│   ├── quality_splits_LCC_development/    # Experimental LCC quality splits (development versions)
│   ├── extract_quality_scores.py          # Main QPAD computation script (generic)
│   ├── extract_quality_scores_casia.py    # QPAD computation for CASIA
│   ├── fake.py                       # Likely test/mocking functionality
│   ├── gen_feat_casia.py             # MagFace feature extraction for CASIA
│   ├── gen_feat_lcc.py               # MagFace feature extraction for LCC
│   ├── gen_img_list_lcc.py           # Generate LCC img.list from folder structure
│   ├── img_lcc_fix.py                # Patch or clean file paths/labels in LCC
│   ├── network_inf.py                # Model inference wrapper using MagFace
│   └── utils.py                      # Helpers for quality scoring, feature handling
├── scripts/                          # Additional scripts
├── .gitattributes
├── .gitignore
└── README.md                         # Project documentation

```

## Getting Started

### 1. Do some preprocessing and generate img.list wit scripts.

### 2.1 Generation of embeddings casia

```bash
python Magface_v2/gen_feat_casia.py
```
### 2.2 Generation of embeddings lcc

```bash
python Magface_v2/gen_feat_lcc.py
```

### 3.1 Stratify dataset Casia

```bash
python Magface_v2\extract_quality_scores_casia.py
```
### 3.2 Stratify dataset LCC

```bash
python Magface_v2\extract_quality_scores.py
```

### 4. Train and evaluate models in both directions

```bash
python CF-PAd/script-runner.py --direction both
```

### 5. Evaluate models in both directions

```bash
python CF-PAd/script-runner.py --direction both

```

## Author

Eva María Benito Sanz
MSc Computer Science and Engineering – Technical University of Denmark (DTU)
[s243313@dtu.dk](mailto:s243313@dtu.dk)

## Acknowledgements

This work is part of my Project for Biometrics Systems Course in DTU. It represents original research on improving generalization in biometric PAD systems through sample quality stratification and causal modeling.
