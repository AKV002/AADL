"""
Preprocessing pipeline for CT-based neuroprognostication.

Self-contained skull stripping + ANTsPy SyN atlas registration + ROI extraction.
No external tools (BET, FSL, etc.) required.

Steps per patient:
  1. Load raw NIfTI CT
  2. Skull strip (HU threshold + morphological ops via SimpleITK)
  3. Register skull-stripped brain to MNI152 using ANTsPy SyNRA
  4. Inverse-warp Harvard-Oxford and JHU atlases to patient native space
  5. Extract ROI masks (caudate, putamen, PLIC, corpus callosum) using XML-parsed labels
  6. Erode ROI masks, union into single binary mask
  7. Crop brain + ROI mask to bounding box (square XY, padded Z)
  8. Clip HU to [0, 80], zero outside brain mask
  9. Save brain.nii.gz and roi_mask.nii.gz
  10. (Optional) Save QC PNG

Usage:
    python preprocessing.py \\
        --input_dir data/raw \\
        --output_dir data/processed \\
        --atlas_dir atlases/ \\
        --num_workers 4 \\
        --qc
"""

import argparse
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion


# ---------------------------------------------------------------------------
# Default skull-stripping configuration
# ---------------------------------------------------------------------------


class SkullStripper:
    """Self-contained skull stripping — no external tools needed."""

    DEFAULT_CONFIG = {
        "smooth_sigma": 1.0,
        "window_min": 0,
        "window_max": 100,
        "threshold_min": 0,
        "threshold_max": 80,
        "erosion_radius": 3,
        "dilation_radius": 3,
        "closing_radius": 2,
        "final_hu_max": 100,
        "final_hu_min": -10,
    }

    def __init__(self, config=None):
        self.config = config or self.DEFAULT_CONFIG
        self.brain_mask = None
        self.brain_image = None
        self.stats = {}

    def extract_brain(self, image):
        cfg = self.config
        spacing = image.GetSpacing()
        erosion_voxels = [max(1, int(cfg["erosion_radius"] / s)) for s in spacing]
        dilation_voxels = [max(1, int(cfg["dilation_radius"] / s)) for s in spacing]
        closing_voxels = [max(1, int(cfg["closing_radius"] / s)) for s in spacing]

        image_float = sitk.Cast(image, sitk.sitkFloat32)
        if cfg["smooth_sigma"] > 0:
            smoothed = sitk.SmoothingRecursiveGaussian(image_float, cfg["smooth_sigma"])
        else:
            smoothed = image_float
        windowed = sitk.Clamp(smoothed, sitk.sitkFloat32, cfg["window_min"], cfg["window_max"])
        initial_mask = sitk.BinaryThreshold(windowed, cfg["threshold_min"], cfg["threshold_max"], 1, 0)
        initial_mask = sitk.Cast(initial_mask, sitk.sitkUInt8)
        head_mask = sitk.BinaryThreshold(image_float, -200, 3000, 1, 0)
        head_mask = sitk.Cast(head_mask, sitk.sitkUInt8)
        head_mask = sitk.BinaryFillhole(head_mask)
        head_mask = self._keep_largest(head_mask)
        constrained = sitk.And(initial_mask, head_mask)
        eroded = self._slice_erosion(constrained, erosion_voxels)
        largest = self._keep_largest(eroded)
        dilated = sitk.BinaryDilate(largest, dilation_voxels, sitk.sitkBall)
        filled = sitk.BinaryFillhole(dilated)
        hu_mask = sitk.BinaryThreshold(image_float, cfg["final_hu_min"], cfg["final_hu_max"], 1, 0)
        hu_mask = sitk.Cast(hu_mask, sitk.sitkUInt8)
        hu_filtered = sitk.And(filled, hu_mask)
        final = sitk.BinaryMorphologicalClosing(hu_filtered, closing_voxels, sitk.sitkBall)
        final = sitk.BinaryFillhole(final)
        final = self._keep_largest(final)
        final = sitk.And(final, head_mask)
        self.brain_mask = final
        stats_filter = sitk.StatisticsImageFilter()
        stats_filter.Execute(image)
        self.brain_image = sitk.Mask(image, final, outsideValue=stats_filter.GetMinimum())
        self._calc_stats(image)
        return self.brain_image, self.brain_mask

    def _slice_erosion(self, mask, erosion_voxels):
        mask_np = sitk.GetArrayFromImage(mask)
        result_np = np.zeros_like(mask_np)
        erosion_2d = max(1, (erosion_voxels[0] + erosion_voxels[1]) // 2)
        for z in range(mask_np.shape[0]):
            if np.sum(mask_np[z]) < 100:
                continue
            slice_sitk = sitk.GetImageFromArray(mask_np[z].astype(np.uint8))
            eroded = sitk.BinaryErode(slice_sitk, [erosion_2d, erosion_2d], sitk.sitkBall)
            labeled = sitk.ConnectedComponent(eroded)
            relabeled = sitk.RelabelComponent(labeled, sortByObjectSize=True)
            largest = sitk.BinaryThreshold(relabeled, 1, 1, 1, 0)
            result_np[z] = sitk.GetArrayFromImage(largest)
        result = sitk.GetImageFromArray(result_np.astype(np.uint8))
        result.CopyInformation(mask)
        return result

    def _keep_largest(self, mask):
        labeled = sitk.ConnectedComponent(mask)
        relabeled = sitk.RelabelComponent(labeled, sortByObjectSize=True)
        largest = sitk.BinaryThreshold(relabeled, 1, 1, 1, 0)
        return sitk.Cast(largest, sitk.sitkUInt8)

    def _calc_stats(self, original):
        spacing = original.GetSpacing()
        voxel_vol_ml = np.prod(spacing) / 1000
        stats_filter = sitk.StatisticsImageFilter()
        stats_filter.Execute(sitk.Cast(self.brain_mask, sitk.sitkFloat32))
        brain_voxels = int(stats_filter.GetSum())
        label_stats = sitk.LabelStatisticsImageFilter()
        label_stats.Execute(original, self.brain_mask)
        if label_stats.HasLabel(1):
            self.stats = {
                "brain_voxels": brain_voxels,
                "brain_volume_ml": brain_voxels * voxel_vol_ml,
                "mean_hu": label_stats.GetMean(1),
                "std_hu": label_stats.GetSigma(1),
                "median_hu": label_stats.GetMedian(1),
                "min_hu": label_stats.GetMinimum(1),
                "max_hu": label_stats.GetMaximum(1),
            }
        else:
            self.stats = {
                "brain_voxels": 0,
                "brain_volume_ml": 0,
                "mean_hu": 0,
                "std_hu": 0,
                "median_hu": 0,
                "min_hu": 0,
                "max_hu": 0,
            }


# ---------------------------------------------------------------------------
# Atlas XML parsing
# ---------------------------------------------------------------------------
def parse_atlas_xml(xml_path):
    """Parse FSL atlas XML. Returns dict: label_name -> index."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = {}
    for label in root.findall(".//data/label"):
        index = int(label.get("index"))
        name = label.text.strip()
        labels[name] = index
    return labels


# ---------------------------------------------------------------------------
# SyN registration + ROI extraction
# ---------------------------------------------------------------------------
def register_and_extract_rois(brain_sitk, atlas_dir):
    """
    SyNRA registration to MNI152 + inverse warp atlases to native space.

    Required files in atlas_dir:
    - MNI152_T1_1mm_brain.nii.gz
    - HarvardOxford-sub-maxprob-thr25-1mm.nii.gz
    - JHU-ICBM-labels-1mm.nii.gz
    - HarvardOxford-Subcortical.xml
    - JHU-labels.xml
    """
    import ants

    atlas_dir = Path(atlas_dir)
    mni_path = atlas_dir / "MNI152_T1_1mm_brain.nii.gz"
    ho_path = atlas_dir / "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"
    jhu_path = atlas_dir / "JHU-ICBM-labels-1mm.nii.gz"
    ho_xml = atlas_dir / "HarvardOxford-Subcortical.xml"
    jhu_xml = atlas_dir / "JHU-labels.xml"

    for p in [mni_path, ho_path, jhu_path, ho_xml, jhu_xml]:
        if not p.exists():
            raise FileNotFoundError(f"Required atlas file not found: {p}")

    # Window brain to [0, 80] for registration; preserve spatial metadata
    brain_np = sitk.GetArrayFromImage(brain_sitk).transpose(2, 1, 0).astype(np.float32)
    brain_reg = np.clip(brain_np, 0, 80).astype(np.float32)
    spacing = brain_sitk.GetSpacing()
    origin = brain_sitk.GetOrigin()
    direction = np.array(brain_sitk.GetDirection()).reshape(3, 3)
    brain_ants = ants.from_numpy(brain_reg, spacing=spacing, origin=origin, direction=direction)
    mni_template = ants.image_read(str(mni_path))

    reg = ants.registration(
        fixed=mni_template,
        moving=brain_ants,
        type_of_transform="SyNRA",
        syn_metric="mattes",
        syn_sampling=32,
        reg_iterations=(200, 200, 100, 50),
        aff_metric="mattes",
    )

    ho_ants = ants.image_read(str(ho_path))
    ho_native = ants.apply_transforms(
        fixed=brain_ants,
        moving=ho_ants,
        transformlist=reg["invtransforms"],
        interpolator="nearestNeighbor",
        whichtoinvert=[True, False],
    )

    jhu_ants = ants.image_read(str(jhu_path))
    jhu_native = ants.apply_transforms(
        fixed=brain_ants,
        moving=jhu_ants,
        transformlist=reg["invtransforms"],
        interpolator="nearestNeighbor",
        whichtoinvert=[True, False],
    )

    ho_np = ho_native.numpy()
    jhu_np = jhu_native.numpy()

    # Parse XML label indices
    ho_labels = parse_atlas_xml(str(ho_xml))
    jhu_labels = parse_atlas_xml(str(jhu_xml))

    # Harvard-Oxford: voxel_value = xml_index + 1 (FSL HO atlas uses 1-based indexing)
    def ho_mask(names):
        m = np.zeros_like(ho_np, dtype=bool)
        for name in names:
            if name in ho_labels:
                m |= (ho_np == ho_labels[name] + 1)
        return m

    # JHU: voxel_value = xml_index (JHU atlas uses 0-based indexing, no offset)
    def jhu_mask(names):
        m = np.zeros_like(jhu_np, dtype=bool)
        for name in names:
            if name in jhu_labels:
                m |= (jhu_np == jhu_labels[name])
        return m

    caudate = ho_mask(["Left Caudate", "Right Caudate"])
    putamen = ho_mask(["Left Putamen", "Right Putamen"])
    plic = jhu_mask([
        "Posterior limb of internal capsule R",
        "Posterior limb of internal capsule L",
    ])
    corpus_callosum = jhu_mask([
        "Genu of corpus callosum",
        "Splenium of corpus callosum",
    ])

    roi_masks = {"caudate": caudate, "putamen": putamen, "plic": plic, "corpus_callosum": corpus_callosum}
    union = np.zeros_like(ho_np, dtype=np.float32)
    for name, mask in roi_masks.items():
        if mask.any():
            eroded = binary_erosion(mask, iterations=1)
            if not eroded.any():
                eroded = mask
            union = np.maximum(union, eroded.astype(np.float32))

    return union


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------
def find_brain_bbox(mask, pad_z=2):
    """Return (x0, x1, y0, y1, z0, z1) bounding box with square XY and padded Z."""
    coords = np.where(mask > 0)
    x0, x1 = coords[0].min(), coords[0].max()
    y0, y1 = coords[1].min(), coords[1].max()
    z0, z1 = coords[2].min(), coords[2].max()
    z0 = max(0, z0 - pad_z)
    z1 = min(mask.shape[2] - 1, z1 + pad_z)
    sq = int(max(x1 - x0, y1 - y0) * 1.05)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0s = max(0, cx - sq // 2)
    x1s = min(mask.shape[0], x0s + sq)
    if x1s == mask.shape[0]:
        x0s = max(0, x1s - sq)
    y0s = max(0, cy - sq // 2)
    y1s = min(mask.shape[1], y0s + sq)
    if y1s == mask.shape[1]:
        y0s = max(0, y1s - sq)
    return x0s, x1s, y0s, y1s, z0, z1 + 1


# ---------------------------------------------------------------------------
# QC helper
# ---------------------------------------------------------------------------
def save_qc_png(brain_np, mask_np, out_path):
    """Save a mid-slice QC image (brain + mask overlay)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_mid = brain_np.shape[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, sl, title in zip(
        axes,
        [brain_np[:, :, z_mid], mask_np[:, :, z_mid], brain_np[:, :, z_mid] * mask_np[:, :, z_mid]],
        ["Brain CT", "Mask", "Masked CT"],
    ):
        ax.imshow(sl.T, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-patient processing
# ---------------------------------------------------------------------------
def process_patient(patient_dir, output_dir, atlas_dir, generate_qc, pad_z=2):
    """
    Process a single patient directory.

    Expects: <patient_dir>/brain.nii.gz  (raw NIfTI CT, skull stripping applied in this pipeline)
    Produces: <output_dir>/<patient_id>/brain.nii.gz
              <output_dir>/<patient_id>/roi_mask.nii.gz  (if atlas_dir provided)
              <output_dir>/<patient_id>/qc.png           (if generate_qc)
    """
    patient_dir = Path(patient_dir)
    ct_path = patient_dir / "brain.nii.gz"
    if not ct_path.exists():
        return f"SKIP {patient_dir.name}: brain.nii.gz not found"

    out_patient = Path(output_dir) / patient_dir.name
    out_patient.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    sitk_image = sitk.ReadImage(str(ct_path))

    # --- Skull strip ---
    stripper = SkullStripper()
    brain_sitk, mask_sitk = stripper.extract_brain(sitk_image)

    # --- Convert to numpy (SimpleITK order: z, y, x → transpose to x, y, z) ---
    brain_np = sitk.GetArrayFromImage(brain_sitk).transpose(2, 1, 0).astype(np.float32)
    mask_np = sitk.GetArrayFromImage(mask_sitk).transpose(2, 1, 0).astype(np.float32)

    if mask_np.sum() == 0:
        return f"WARN {patient_dir.name}: empty brain mask after skull stripping"

    # --- Preserve original affine from nibabel for saving ---
    nib_orig = nib.load(str(ct_path))
    affine = nib_orig.affine

    # --- ROI mask via ANTsPy SyN registration ---
    roi_mask = None
    if atlas_dir is not None:
        try:
            roi_mask = register_and_extract_rois(brain_sitk, atlas_dir)
        except Exception as e:
            print(f"WARN {patient_dir.name}: ROI extraction failed — {e}")

    # --- Bounding-box crop (square XY, padded Z) ---
    x0, x1, y0, y1, z0, z1 = find_brain_bbox(mask_np, pad_z=pad_z)
    brain_cropped = brain_np[x0:x1, y0:y1, z0:z1]
    mask_cropped = mask_np[x0:x1, y0:y1, z0:z1]

    # --- HU clip [0, 80], zero outside mask ---
    brain_clipped = np.clip(brain_cropped, 0, 80) * mask_cropped

    nib.save(nib.Nifti1Image(brain_clipped, affine), str(out_patient / "brain.nii.gz"))

    if roi_mask is not None:
        roi_cropped = roi_mask[x0:x1, y0:y1, z0:z1]
        nib.save(nib.Nifti1Image(roi_cropped, affine), str(out_patient / "roi_mask.nii.gz"))

    # --- QC ---
    if generate_qc:
        save_qc_png(brain_clipped, mask_cropped, str(out_patient / "qc.png"))

    return f"OK  {patient_dir.name}: brain_voxels={stripper.stats.get('brain_voxels', 0)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Self-contained CT preprocessing: skull stripping + ANTsPy SyN registration + ROI mask generation."
    )
    parser.add_argument("--input_dir", required=True, help="Directory with per-patient NIfTI CT folders")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed files")
    parser.add_argument("--atlas_dir", default=None, help="Directory with atlas NIfTI and XML files for ROI mask generation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--qc", action="store_true", help="Generate QC PNG images")
    parser.add_argument("--pad_z", type=int, default=2, help="Z-padding for bounding box crop (default: 2)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    patient_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    print(f"Found {len(patient_dirs)} patient directories in {input_dir}")

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_patient, p, args.output_dir, args.atlas_dir, args.qc, args.pad_z): p
                for p in patient_dirs
            }
            for future in as_completed(futures):
                print(future.result())
    else:
        for p in patient_dirs:
            print(process_patient(p, args.output_dir, args.atlas_dir, args.qc, args.pad_z))


if __name__ == "__main__":
    main()
