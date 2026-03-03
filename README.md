# Anatomy-Aware Deep Learning for CT-Based Neuroprognostication

## Overview

This repository contains the code for the **Anatomy-Aware Deep Learning (AADL)** neuroprognostication model: a 3D DenseNet-121 with atlas-guided attention gates for CT-based brain outcome prediction.

The model uses a **3D DenseNet-121** backbone with **Anatomy-Aware Attention Gates** that leverage atlas-derived ROI masks to guide the network's attention to clinically relevant brain structures (caudate, putamen, posterior limb of internal capsule, corpus callosum).

**Key features:**
- **3D DenseNet-121** backbone with dense feature reuse
- **Atlas-Guided Attention Gates** at two feature scales (after dense blocks 3 and 4)
- **Self-contained preprocessing** — no external tools (BET, FSL) needed; brain extraction is fully self-contained in Python using SimpleITK (HU thresholding + morphological operations)
- **ANTsPy-based SyN registration** — atlas-to-patient registration using ANTsPy SyNRA (no external ANTs binaries required)
- **GradCAM++ explainability** maps for combined, backbone, and atlas-attention branches

## Repository Structure

```
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── preprocessing.py       # Skull stripping + ANTsPy SyN registration + ROI mask generation
├── dataset.py             # PyTorch Dataset + MONAI transforms
├── model.py               # 3D DenseNet-121 with Atlas Attention Gates
├── train.py               # Training script
├── evaluate.py            # Evaluation + attention map visualizations
├── explainability.py      # GradCAM++ saliency maps
└── atlases/
    └── README.md          # Instructions for required atlas files
```

## Installation

```bash
pip install -r requirements.txt
```

> **Note**: `antspyx` may require separate installation depending on your platform. See [ANTsPy installation](https://github.com/ANTsX/ANTsPy) for details.

## Data Preparation

### Expected Directory Structure

```
data/
├── raw/                        # Raw skull-stripped NIfTI CTs
│   ├── patient_001/
│   │   └── brain.nii.gz
│   └── ...
├── processed/                  # Output of preprocessing.py
│   ├── patient_001/
│   │   ├── brain.nii.gz       # Cropped, HU-clipped brain
│   │   └── roi_mask.nii.gz    # Union of 4 ROI masks
│   └── ...
└── split/                      # Train/val/test splits for training
    ├── train/
    │   ├── good/
    │   │   └── patient_XXX/
    │   │       ├── brain.nii.gz
    │   │       └── roi_mask.nii.gz
    │   └── poor/
    ├── val/
    └── test/
```

Each patient folder in `split/` must contain:
- `brain.nii.gz` — skull-stripped CT (HU clipped to [0, 80])
- `roi_mask.nii.gz` — binary atlas ROI mask

## Pipeline Steps

### 1. Preprocessing

```bash
python preprocessing.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --atlas_dir atlases/ \
    --num_workers 4 \
    --qc
```

Performs skull stripping (self-contained, no FSL/BET needed), ANTsPy SyNRA registration to MNI152, atlas-guided ROI extraction, cropping, and HU clipping. See `atlases/README.md` for required atlas files.

### 2. Training

```bash
python train.py \
    --data_dir data/split \
    --output_dir outputs
```

### 3. Evaluation

```bash
python evaluate.py \
    --model_path outputs/run_xxx/best_model.pth \
    --data_dir data/split
```

### 4. Explainability

```bash
python explainability.py \
    --model_path outputs/run_xxx/best_model.pth \
    --data_dir data/split
```

## Atlases

The preprocessing pipeline requires the following atlas files placed in the `atlases/` directory:

| File | Description |
|------|-------------|
| `MNI152_T1_1mm_brain.nii.gz` | MNI152 T1 brain template (1mm) |
| `HarvardOxford-sub-maxprob-thr25-1mm.nii.gz` | Harvard-Oxford subcortical atlas |
| `JHU-ICBM-labels-1mm.nii.gz` | JHU white matter tract labels |
| `HarvardOxford-Subcortical.xml` | Harvard-Oxford label definitions |
| `JHU-labels.xml` | JHU label definitions |

These files are available from [FSL](https://fsl.fmrib.ox.ac.uk/). See [`atlases/README.md`](atlases/README.md) for details.


## Citation

If you use this code, please cite:

```
Anonymized
```

