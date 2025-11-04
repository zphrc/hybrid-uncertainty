# Hybrid Uncertainty Evaluation Tool

A **lightweight hybrid measurement tool** for evaluating **aleatoric and epistemic uncertainty** in image classification tasks.  
This repository implements the thesis *“A Hybrid Measurement Tool for Evaluating Uncertainty in Image Classification”* (Dejito & Roca, 2025).

---

## Overview

Modern lightweight CNNs often make overconfident predictions on ambiguous or unfamiliar data.  
This tool combines two post-hoc uncertainty metrics into a single **Hybrid Danger Score**:

- **Softmax Entropy** → measures *aleatoric uncertainty* (data ambiguity).  
- **Input Gradient Sensitivity** → measures *epistemic uncertainty* (model ignorance).

By averaging their normalized values, the tool improves uncertainty ranking, calibration, and selective prediction performance.

---

## Features

- Train lightweight CNNs (ShuffleNetV2, MobileNetV3-Small, EfficientNetV2-S)  
- Compute entropy, gradient sensitivity, and hybrid danger scores  
- Evaluate uncertainty metrics (AUROC, ECE, AvUC, ARC)  
- Visualize calibration and accuracy–rejection curves  
- Works with MNIST, Fashion-MNIST, and CIFAR-10  
- Apple Silicon / CUDA compatible (MPS & GPU supported)

---

## Project Structure

```bash
hybrid-uncertainty/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ Makefile
├─ .gitignore
├─ config/
│  ├─ base.yaml
│  ├─ datasets.yaml
│  └─ models.yaml
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ results/
├─ notebooks/
│  └─ exploration.ipynb
├─ src/
│  ├─ utils/
│  ├─ data/
│  ├─ models/
│  ├─ train/
│  ├─ uncertainty/
│  ├─ eval/
│  └─ cli/
└─ app/
   ├─ backend/
   └─ frontend/
```

> Note: The `data/` directory is excluded from version control and is automatically created when you run the commands below.

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/hybrid-uncertainty.git
cd hybrid-uncertainty
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## Reproducibility Instructions

The following sequence reproduces the MNIST + ShuffleNetV2 baseline results (AUROC ≈ 0.877, ECE ≈ 0.007):

### 1. Train lightweight CNN

```bash
hu-train --dataset mnist --model shufflenet_v2_0_5
```

### 2. Compute uncertainty metrics

```bash
hu-score --dataset mnist --model shufflenet_v2_0_5
```

### 3. Evaluate and visualize results

```bash
hu-evaluate --dataset mnist --model shufflenet_v2_0_5
```

Expected output files:

```bash
data/results/checkpoints/mnist_shufflenet_v2_0_5.pt
data/results/scores_mnist_shufflenet_v2_0_5.csv
data/results/metrics_mnist_shufflenet_v2_0_5.txt
data/results/reliability_mnist_shufflenet_v2_0_5.png
data/results/arc_mnist_shufflenet_v2_0_5.png
```

## Metrics

| Metric | Description | Goal |
|---------|--------------|------|
| **AUROC** | Measures how well uncertainty ranks incorrect predictions | Higher = better |
| **ECE** | Expected Calibration Error (confidence vs accuracy) | Lower = better |
| **AvUC Loss** | Accuracy vs Uncertainty Calibration | Lower = better |
| **ARC** | Accuracy–Rejection Curve (retained accuracy vs rejection rate) | Higher area = better |

---

## Example Results (MNIST + ShuffleNetV2)

| Metric | Result |
|--------|--------:|
| Base Accuracy | 92.55 % |
| AUROC | 0.877 |
| ECE | 0.0073 |
| AvUC | 0.207 |
| Retained Accuracy @30% Reject | 98.4 % |

Hybrid danger score improves reliability and selective accuracy compared to entropy-only or gradient-only metrics.

---

## Data Handling

- **Datasets** are automatically downloaded to `data/raw/`.  
- **Checkpoints** and results are saved to `data/results/`.  
- **Preprocessing** (tensor conversion & normalization) is handled dynamically in `src/data/transforms.py`.  

---

## Authors

- **Christine Ann P. Dejito**  
- **Zophia Maureen A. Roca**  
University of San Carlos — BS Computer Science, 2025

---

## License

MIT License © 2025 Dejito & Roca  
See [LICENSE](LICENSE) for details.

