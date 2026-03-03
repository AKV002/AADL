# Atlas Files

Place the following atlas files in this directory before running preprocessing.

## Required Files

| File | Description |
|------|-------------|
| `MNI152_T1_1mm_brain.nii.gz` | MNI152 T1 brain template (1mm) |
| `HarvardOxford-sub-maxprob-thr25-1mm.nii.gz` | Harvard-Oxford subcortical atlas |
| `JHU-ICBM-labels-1mm.nii.gz` | JHU white matter tract labels |
| `HarvardOxford-Subcortical.xml` | Harvard-Oxford label definitions |
| `JHU-labels.xml` | JHU label definitions |

## Source

These atlases are available from:
- FSL (FMRIB Software Library): https://fsl.fmrib.ox.ac.uk/
- Standard neuroimaging repositories

## ROIs Used

The preprocessing pipeline extracts four ROIs:
- **Caudate** (Left + Right) — from Harvard-Oxford
- **Putamen** (Left + Right) — from Harvard-Oxford
- **PLIC** (Left + Right posterior limb of internal capsule) — from JHU
- **Corpus Callosum** (Genu + Splenium) — from JHU

