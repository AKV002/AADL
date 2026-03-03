"""
Dataset utilities for CT-based neuroprognostication.

Provides:
  - get_data_list()       — discover patients from split/good|poor directory layout
  - get_transforms()      — MONAI transform pipeline (with optional augmentation)
  - compute_class_weights() — inverse-frequency class weights
  - create_dataloader()   — convenience wrapper
  - seed_worker()         — DataLoader worker seed function for reproducibility
"""

import random
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
)


def get_data_list(data_dir, split):
    """
    Discover patients from the directory layout::

        <data_dir>/<split>/good/<patient_id>/brain.nii.gz
        <data_dir>/<split>/poor/<patient_id>/brain.nii.gz

    Returns a list of dicts with keys: image, atlas, label, patient_id.
    """
    data_list = []
    for class_name, label in [("good", 0), ("poor", 1)]:
        class_dir = Path(data_dir) / split / class_name
        if class_dir.exists():
            for patient_dir in sorted(class_dir.iterdir()):
                ct_path = patient_dir / "brain.nii.gz"
                atlas_path = patient_dir / "roi_mask.nii.gz"
                if ct_path.exists() and atlas_path.exists():
                    data_list.append(
                        {
                            "image": str(ct_path),
                            "atlas": str(atlas_path),
                            "label": label,
                            "patient_id": patient_dir.name,
                        }
                    )
    n_good = sum(1 for d in data_list if d["label"] == 0)
    print(
        f"[{split.upper()}] {len(data_list)} samples "
        f"(Good: {n_good}, Poor: {len(data_list) - n_good})"
    )
    return data_list


def get_transforms(target_shape, augment=False):
    """
    Build a MONAI Compose transform pipeline.

    Args:
        target_shape: (D, H, W) or (H, W, D) spatial size for Resized.
        augment:      If True, add random augmentations (train split).

    Returns:
        monai.transforms.Compose
    """
    keys = ["image", "atlas"]
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, -1), mode=("bilinear", "nearest")),
        # pixdim=(1.0, 1.0, -1): resample x and y to 1 mm isotropic; -1 preserves original z-spacing.
        Resized(keys=keys, spatial_size=target_shape, mode=("trilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=80, b_min=0, b_max=1, clip=True),
    ]
    if augment:
        transforms += [
            RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandGaussianNoised(keys=["image"], prob=0.3, std=0.01),
        ]
    transforms.append(EnsureTyped(keys=keys + ["label"]))
    return Compose(transforms)


def compute_class_weights(data_list, device):
    """
    Compute inverse-frequency class weights for weighted cross-entropy.

    Args:
        data_list: list of sample dicts (each with a 'label' key).
        device:    torch.device to place the weights tensor on.

    Returns:
        torch.Tensor of shape (2,).
    """
    counts = [sum(1 for d in data_list if d["label"] == i) for i in [0, 1]]
    weights = torch.tensor([len(data_list) / (2 * c) for c in counts]).to(device)
    print(f"Class weights: Good={weights[0]:.4f}, Poor={weights[1]:.4f}")
    return weights


def seed_worker(worker_id):
    """DataLoader worker initializer for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
    data_dir,
    split,
    target_shape,
    batch_size,
    augment=False,
    seed=42,
    num_workers=4,
):
    """
    Convenience function to build a DataLoader for a given split.

    Args:
        data_dir:     Root data directory (contains train/val/test sub-dirs).
        split:        One of 'train', 'val', 'test'.
        target_shape: Spatial size passed to get_transforms().
        batch_size:   DataLoader batch size.
        augment:      Whether to apply random augmentations.
        seed:         Random seed for the DataLoader generator.
        num_workers:  Number of DataLoader worker processes.

    Returns:
        (DataLoader, list) — the loader and the underlying data_list.
    """
    data_list = get_data_list(data_dir, split)
    transforms = get_transforms(target_shape, augment=augment)
    dataset = Dataset(data=data_list, transform=transforms)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
    )
    return loader, data_list
