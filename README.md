# Team aimoment at PAN2026: Content-Agnostic Contrastive Stylometry for Multi-Author Writing Style Analysis

Official implementation of the **ContraStyle** system submitted to the
[PAN 2026 Multi-Author Writing Style Analysis](https://pan.webis.de/clef26/pan26-web/style-change-detection.html)
shared task at CLEF 2026.

> **Paper:** *Team aimoment at PAN2026: Content-Agnostic Contrastive Stylometry for Multi-Author Writing Style Analysis<img width="1536" height="1024" alt="file_00000000be687243bbb9af3a6161aa71" src="https://github.com/user-attachments/assets/f09ef18a-6c47-4668-a9b5-dbc5d2175537" />
*
> Momen Ibrahim — Alexandria University
> CLEF 2026 Working Notes *(link will be added upon publication)*

**Competition result:** F1-macro of **1.000 / 0.741 / 0.773** (easy / medium / hard).

---

## Overview

Given a multi-author document, ContraStyle predicts for each consecutive sentence pair whether a writing-style change occurs (1) or not (0).

The system is built around one principle: **content-agnostic style learning** — every component is designed to suppress topical shortcuts and amplify pure stylistic signals. This is critical for the *hard* and *medium* difficulty levels where documents share the same topic.

### Architecture

```
Input document (N sentences)
         │
         ▼
   Windowed pair extraction
         │
         ├──────────────────────┬──────────────────────┐
         ▼                      ▼                      ▼
  DeBERTa-v3              LightGBM               SSPC (BiLSTM)
  (local pair level)    (surface stylometrics)  (document sequence)
  CE + SCL + R-Drop       108 topic-agnostic     Full-document SBERT
  + CACO loss             features               trajectory
         │                      │                      │
         └──────────────┬────────────────────────────-─┘
                        ▼
             Weighted ensemble blend
             + per-difficulty threshold calibration
                        ▼
             {"changes": [0, 1, 0, ...]}
```

### Three complementary views

| Component | Level | Content-agnostic by design |
|---|---|---|
| **DeBERTa-v3** + CACO | Local pair | CACO uses same-document hard negatives to prevent topic leakage |
| **LightGBM** | Surface stylometrics | Function-word rates, punctuation patterns, word-length distributions — all topic-independent |
| **SSPC BiLSTM** | Document sequence | Models the style trajectory across the full document via frozen SBERT embeddings |

### Key training techniques

- **CACO** (Content-Agnostic Contrastive Objective): InfoNCE loss with in-batch same-document hard negatives
- **SCL** (Supervised Contrastive Loss): Pulls same-class style embeddings together
- **R-Drop**: Bidirectional KL divergence between two dropout-augmented forward passes
- **Focal Loss** (γ=2.0): Applied to medium/hard to address class imbalance
- **FGM** (Fast Gradient Method, ε=0.5): Adversarial word-embedding perturbation
- **EMA** (Exponential Moving Average, decay=0.9995): Shadow weights for evaluation and saving
- **LLRD** (Layer-wise LR Decay, factor=0.9): Gentler updates for lower encoder layers

---

## Data

Download from Zenodo and structure as follows:

```
DATA/
├── 19068843/mawsa26-pan-zenodo/   # PAN 2026
├── 14891299/pan25-multi-author-analysis/   # PAN 2025
├── 10677876/pan24-multi-author-analysis/   # PAN 2024
├── 7729178/pan23-multi-author-analysis/release/  # PAN 2023
└── 6334245/pan22/                 # PAN 2022
```

The `DATA/` directory should sit one level above this repo. Update `src/config.py` if your layout differs.

---

## Installation

```bash
pip install -r requirements.txt
```

GPU with ≥16 GB VRAM required for DeBERTa and SSPC training.

---

## Usage

### 1. Prepare data

Merges all years, deduplicates, and saves to `data_prepared/`:

```bash
python prepare_data.py
```

### 2. Train

```bash
# Train all models for all difficulty levels
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python train.py

# Train only hard difficulty (e.g. resume after interruption)
HF_HUB_OFFLINE=1 python train.py --difficulties hard --force-transformer --force-lgbm --force-sspc

# Skip individual components
python train.py --skip-lgbm --skip-sspc   # DeBERTa only
python train.py --skip-transformer        # LightGBM + SSPC only
```

Key flags:

| Flag | Description |
|---|---|
| `--difficulties easy medium hard` | Subset of difficulties to train |
| `--force-transformer` | Retrain DeBERTa even if checkpoint exists |
| `--force-lgbm` | Retrain LightGBM |
| `--force-sspc` | Retrain SSPC |
| `--warm-start` | Initialise DeBERTa from existing checkpoint, train fresh |
| `--no-scl` | Disable Supervised Contrastive Loss |
| `--no-rdrop` | Disable R-Drop |
| `--no-fgm` | Disable FGM adversarial training |
| `--no-focal` | Disable Focal Loss |

### 3. Predict

```bash
python predict.py -i /path/to/input -o /path/to/output
```

Input may contain `easy/`, `medium/`, `hard/` subdirectories with `problem-*.txt` files (standard PAN/TIRA layout).

### 4. Evaluate

```bash
python evaluate.py --predictions /path/to/output --ground-truth /path/to/data
```

### 5. Monitor training

```bash
# Interactive Plotly dashboard
python dashboard.py

# Live mode (auto-refresh while training)
python dashboard.py --live
```

---

## Pre-trained Models

Pre-trained model weights will be released on Hugging Face Hub after the paper is published.
*(Link will be added here)*

---

## Repository Structure

```
├── src/
│   ├── config.py             # All hyperparameters and paths
│   ├── data.py               # Data loading and pair construction
│   ├── transformer_model.py  # DeBERTa classifier, CACO, FGM, EMA, LLRD
│   ├── features.py           # Stylometric feature extraction (108 features)
│   ├── classical_models.py   # LightGBM wrapper
│   ├── sspc_model.py         # BiLSTM sequential profiler
│   ├── ensemble.py           # Blending, isotonic calibration, threshold search
│   └── training_logger.py    # Structured logging (JSONL + human-readable)
├── train.py                  # Training entry point
├── predict.py                # Inference entry point (TIRA-compatible)
├── evaluate.py               # Evaluation script
├── prepare_data.py           # Multi-year data merging and deduplication
├── analyze_data.py           # Dataset statistics and analysis
├── dashboard.py              # Plotly training dashboard
├── Dockerfile                # TIRA/Docker submission
├── requirements.txt          # Training dependencies
└── requirements-inference.txt  # Inference-only dependencies
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ibrahim:2026,
  title     = {ContraStyle: Content-Agnostic Contrastive Stylometry for
               Multi-Author Writing Style Analysis},
  author    = {Ibrahim, Momen},
  booktitle = {Working Notes of CLEF 2026},
  year      = {2026}
}
```

---

## License

This code is released under the MIT License.
