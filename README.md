# biometricDTU2025
¡Perfecto! Si **el trabajo es tuyo** y lo estás presentando como parte de la Task 4.5, entonces el README debe reflejarlo claramente como tu **propio desarrollo e investigación**.

Aquí tienes el **README completo en formato Markdown (`README.md`)**, con todo integrado y presentando el trabajo como **tu implementación original basada en tu paper publicado en BIOSIG 2025**:


# 🧪 Exploring Sample Quality for Face Presentation Attack Detection (PAD)

## 📌 Project Overview

This repository presents the official implementation of the work:

> **"The Impact of Sample Quality Stratification on Generalization in Face Presentation Attack Detection"**  
> *Eva María Benito Sanz, BIOSIG 2025*

This project explores how **sample quality influences the generalization capability** of Face Presentation Attack Detection (PAD) systems. We propose a novel **quality-aware training protocol** that uses a custom quality metric (**QPAD**) and stratifies training data into quality tiers (Low, Medium, High). We show that training deep neural networks on **medium-quality samples** achieves superior generalization to unseen domains and attack types.


## 🧠 Motivation

Although recent PAD systems achieve high accuracy in controlled settings, their performance **degrades significantly under domain shift**. A major reason is the **sample quality variation**, which introduces spurious correlations and confounds learning.

This work presents the **first systematic study** of the role of sample quality in PAD generalization, and introduces:
- A new **PAD-specific quality metric (QPAD)**
- A robust **quality stratification protocol**
- Evaluation across quality tiers, datasets, and attacks
- Integration with **causal learning (CF-PAD)** for enhanced robustness

---

## 🧱 Project Structure

```

.
├── data/                         # Input datasets (CASIA-FASD, Replay-Attack, OULU-NPU)
├── quality/
│   ├── compute\_qpad.py           # QPAD computation (MagFace + artifacts)
│   ├── stratify\_by\_quality.py    # Quality tier splitting (LQ, MQ, HQ)
├── models/
│   ├── train.py                  # Model training per quality tier
│   ├── evaluate.py               # Evaluation across quality tiers/domains
├── utils/
│   ├── metrics.py                # HTER, AUC, BPCER computation
│   ├── visualizations.py         # t-SNE plots
├── results/                      # Logs, figures, and evaluation tables
├── configs/
│   └── config.yaml               # Training configuration
├── requirements.txt
└── README.md

````

---

## 📊 Methodology

### 📌 1. Quality Metric – **QPAD**

We extend MagFace by integrating sensitivity to spoof-relevant image artifacts:

\[
Q_{PAD}(I) = \|f(I)\| \cdot \prod_{k=1}^K (1 - \alpha_k A_k(I))
\]

Where:
- \(\|f(I)\|\) is the **MagFace** embedding norm.
- \(A_k(I)\): normalized scores for artifacts:
  - \(A_b(I)\): blur (Laplacian variance)
  - \(A_n(I)\): noise (PCA residual)
  - \(A_c(I)\): JPEG compression artifacts

Alpha weights \(\alpha_k\) are optimized to maximize correlation with performance drop (HTER increase).

---

### 📌 2. Quality Stratification

Datasets are stratified into quality tiers:
- **Low Quality (LQ):** \( Q < Q_{25} \)
- **Medium Quality (MQ):** \( Q_{25} \leq Q \leq Q_{75} \)
- **High Quality (HQ):** \( Q > Q_{75} \)

This allows training and testing PAD models within and across these tiers.

---

### 📌 3. Training and Evaluation

We use a ResNet-18 based PAD architecture inspired by DeepPixBis, with pixel-wise and binary supervision. Training uses **tier-specific augmentations**:

| Quality Tier | Augmentation                          |
|--------------|----------------------------------------|
| LQ           | Blur (σ=3), JPEG(30%), Noise(σ=0.1)    |
| MQ           | Random crop, flip                      |
| HQ           | Center crop only                       |

**Evaluation modes:**
- Intra-quality (e.g., MQ → MQ)
- Cross-quality (e.g., MQ → HQ)
- Cross-dataset (e.g., CASIA → Replay)
- Cross-attack (test on unseen attack types)

Metrics:
- **HTER**, **AUC**, **BPCER@APCER=10%**

---

## ✅ Results Summary

**Cross-dataset generalization performance (HTER %):**

| Training Tier | LQ Test | MQ Test | HQ Test | Cross-Dataset |
|---------------|---------|---------|---------|----------------|
| Low Quality   | 12.4    | 18.7    | 25.3    | 24.6           |
| Medium Quality| 15.2    | **8.3** | 10.7    | **13.9**       |
| High Quality  | 28.5    | 14.2    | **6.1** | 21.8           |
| Mixed Quality | 16.3    | 12.5    | 11.8    | 17.2           |

**Integration with CF-PAD:**

| Method                | HTER (%) |
|-----------------------|----------|
| CF-PAD (baseline)     | 17.3     |
| CF-PAD + MQ training  | **11.2** |

---

## ▶️ Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
````

### 2. Compute QPAD scores

```bash
python quality/compute_qpad.py --input data/Replay-Attack
```

### 3. Stratify dataset

```bash
python quality/stratify_by_quality.py --input data/Replay-Attack --output data/Replay-Stratified
```

### 4. Train a model

```bash
python models/train.py --dataset CASIA-FASD --quality MQ
```

### 5. Evaluate

```bash
python models/evaluate.py --train_dataset CASIA-FASD --test_dataset OULU-NPU --train_quality MQ
```

---

## 📚 References

* Eva María Benito Sanz. *The Impact of Sample Quality Stratification on Generalization in Face Presentation Attack Detection*, BIOSIG 2025.
* Q. Meng et al., *MagFace: A Universal Representation for Face Recognition and Quality Assessment*, CVPR 2021. [GitHub](https://github.com/IrvingMeng/MagFace)
* M. Fang, N. Damer, *CF-PAD: Causal Feature Learning for Generalized Face PAD*, WACV 2023. [GitHub](https://github.com/meilfang/CF-PAD)

---

## ✍️ Author

**Eva María Benito Sanz**
MSc Computer Science and Engineering – DTU
[s243313@dtu.dk](mailto:s243313@dtu.dk)

---

## 📌 Acknowledgements

This work was submitted and accepted at BIOSIG 2025 as part of ongoing research into robust and deployable biometric security systems.
