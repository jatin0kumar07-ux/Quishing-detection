# Quishing Detection

A research framework for detecting **QR code-based phishing attacks (Quishing)** by analyzing raw QR code pixel patterns — without extracting or following embedded URLs.

---

## Overview

Traditional phishing defenses rely on URL analysis, which requires resolving the QR code's payload and may expose users to malicious content. This project proposes a **content-agnostic** approach: classifying QR codes as phishing or benign purely from their visual structure using machine learning.

The pipeline covers the full research workflow:

1. **URL collection** — balanced, leakage-safe dataset from PhishStorm + optional live feeds
2. **QR image generation** — three realism tiers across multiple renderers
3. **Model training & evaluation** — XGBoost (pixel vectors) vs. CNN backbones (ResNet-18, etc.)
4. **Adversarial robustness** — measuring detection AUC as QR codes are progressively corrupted
5. **Explainability** — Grad-CAM visualizations showing where the CNN attends
6. **Figure generation** — all paper figures reproduced from one script

---

## Key finding

 After building a confound-controlled, length-matched benchmark, I found that a **URL-length artifact** — benign URLs are short, phishing URLs long — was inflating reported payload-free detection accuracy by **up to 0.24 AUC**. Once the classes are length-matched, genuine signal is modest on clean codes, weak on styled codes, and collapses to near-chance on realistically captured codes. A tamper-robustness probe further shows that *random* module corruption destroys a QR code's decodability long before it evades detection — leaving optimised, error-correction-aware evasion as the key open problem.


---

## Repository Structure

```
Quishing-detection/
├── collect_urls.py          # Collect & split URLs from PhishStorm / OpenPhish / Majestic
├── collect_urls_matched.py  # Length-matched URL collection to remove the URL-length confound
├── build_dataset.py         # Render QR images across three realism tiers
├── train_eval.py            # Train XGBoost & CNN models; evaluate per tier
├── multiseed_eval.py        # Multi-seed evaluation with bootstrap confidence intervals
├── adversarial_probe.py     # Tamper-robustness curve: AUC vs. corruption level
├── grad_cam.py              # Grad-CAM attention visualizations for ResNet-18
├── make_figures.py          # Reproduce all paper figures
├── robustness_curve.csv     # Pre-computed adversarial robustness results
└── urlset.csv               # PhishStorm URL dataset (download separately)
```

---

## Dataset Tiers

| Tier | Description |
|------|-------------|
| `synthetic_clean` | Fixed ECC=L, fixed module size, black & white — replicates the original paper's setting |
| `rendered` | Random ECC, module size, colours, optional logo overlay |
| `captured` | `rendered` + perspective distortion, blur, brightness jitter, JPEG compression, occlusion |

---

## Installation

```bash
pip install numpy pandas pillow opencv-python scikit-learn xgboost torch timm segno tldextract scipy matplotlib
# Optional: QR decodability check during adversarial probing
pip install pyzbar
```

---

## Quick Start

### Step 1 — Collect URLs

Download PhishStorm's `urlset.csv` from [Aalto University](https://research.aalto.fi/en/datasets/phishstorm-phishing-legitimate-url-dataset/), then run:

```bash
# Standard collection (50/50 phishing/benign, leakage-safe domain split)
python collect_urls.py --phishstorm-csv urlset.csv --target-per-class 8000 --use-openphish

# Or: length-matched collection to remove URL-length as a confounding feature
python collect_urls_matched.py --phishstorm-csv urlset.csv --target-per-class 8000
```

Output: `data/urls/{train,val,test}.csv`

### Step 2 — Build the QR Image Dataset

```bash
python build_dataset.py \
  --urls-dir data/urls \
  --out-dir data \
  --tiers synthetic_clean rendered captured

# Quick smoke-test with 50 URLs per split
python build_dataset.py --urls-dir data/urls --out-dir data --limit 50
```

Output: `data/images/` + `data/metadata.csv`

### Step 3 — Train & Evaluate

```bash
# XGBoost (pixel vectors) vs ResNet-18
python train_eval.py --train-meta data/metadata.csv --model resnet18 --epochs 10

# Cross-generator test (train on segno, test on qrcode-rendered images)
python train_eval.py \
  --train-meta data/metadata.csv \
  --test-meta data_xgen/metadata.csv \
  --model resnet18

# Multi-seed evaluation with 95% bootstrap confidence intervals
python multiseed_eval.py --train-meta data/metadata.csv --seeds 0 1 2 --epochs 15
```

### Step 4 — Adversarial Robustness

```bash
# Corrupt QR codes by inverting random module-blocks; plot AUC vs. corruption fraction
python adversarial_probe.py --tier rendered --epochs 8 --grid 21
```

Output: `robustness_curve.csv`

### Step 5 — Grad-CAM Visualizations

```bash
python grad_cam.py --tier rendered --epochs 8 --n-show 8 --out grad_cam.png
```

Output: `grad_cam.png` — side-by-side (input | Grad-CAM heatmap) for phishing and benign codes

### Step 6 — Generate Paper Figures

```bash
python make_figures.py --meta data/metadata.csv --curve robustness_curve.csv
```

Produces: `figure1_tiers.png`, `figure2_results.png`, `figure3_confound.png`, `figure4_robustness.png`

---

## Key Results

| Tier | XGBoost AUC | ResNet-18 AUC |
|------|-------------|---------------|
| `synthetic_clean` | 0.755 | 0.688 |
| `rendered` | 0.683 | 0.577 |
| `captured` | 0.536 | 0.523 |

> Results shown after removing the URL-length confound via length-matched sampling. XGBoost consistently outperforms the CNN, but both models degrade substantially as realism increases — highlighting the difficulty of real-world QR-centric detection.

---

## Responsible Use

Phishing URLs in this project are stored as plain text solely for training a security detector.  
**Never open, click, scan, or resolve them. Work in an isolated environment.**

---

## License

This project is intended for research purposes. Please review any licenses associated with the PhishStorm dataset and OpenPhish feed before redistributing.

