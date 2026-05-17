# AGENTS.md — TFM: Lung Nodule Detection & Classification (LIDC-IDRI)

## Project Overview

Master's thesis project for detecting and classifying lung nodules from CT scans using the LIDC-IDRI dataset. The pipeline has three stages:

1. **Data generation** — Extract full CT volumes and full-size consensus masks from DICOM data using `pylidc`.
2. **Segmentation** — U-Net 3D to detect and segment nodules from complete CT volumes.
3. **Classification** — CNN to predict malignancy and other annotation features from segmented nodules.

## Repository Structure

```
TFM/
├── process_all_masks.py       # Stage 1: full CT volumes + full-size masks per patient
├── preprocessing_pipeline.py  # Stage 2: windowing, normalization, split by patient
├── train_unet3d_baseline.py   # Stage 3: Dataset (patch extraction), U-Net 3D, training loop
├── visualize_preprocessed_data.ipynb # Visualize full CT volumes and masks
├── visualize_data.ipynb       # Visualize raw data
├── process_data.ipynb         # Exploratory notebook (single patient)
├── process_data_test.ipynb    # Exploratory notebook with visualization
├── output/                    # Generated data (gitignored)
│   ├── CT/                    #   Full CT volumes per patient (.npy)
│   ├── masks/                 #   Full-size masks per patient (.npy)
│   ├── metadata.csv           #   Annotation features per nodule
│   └── preprocessed/          #   Windowed + normalized volumes
│       ├── train/             #     CT/, masks/, metadata.csv
│       ├── val/               #     CT/, masks/, metadata.csv
│       └── test/              #     CT/, masks/, metadata.csv
├── checkpoints/               #   Trained model checkpoints
│   └── unet3d_baseline/
│       ├── best.pt
│       ├── last.pt
│       ├── history.csv
│       └── summary.json
├── .devcontainer/
│   └── devcontainer.json
├── AGENTS.md
├── SKILLS.md
├── QA.md                      # Histórico de preguntas y respuestas
└── .gitignore
```

## Dataset

- **LIDC-IDRI**: 1018 CT scans with expert annotations from up to 4 radiologists.
- DICOM files are stored locally in `/lidc_idri` (gitignored).
- Access is via the `pylidc` Python library which reads a local SQLite database built from the XML annotations.
- The `pylidc` config file (`~/.pylidcrc`) must point to the DICOM directory:
  ```ini
  [dicom]
  path = /path/to/lidc_idri
  ```

## Key Libraries

| Library | Purpose |
|---------|---------|
| `pylidc` | Query LIDC-IDRI annotations, cluster nodules, compute consensus masks |
| `pydicom` | Read DICOM files |
| `SimpleITK` | Image processing and resampling |
| `numpy` | Array operations |
| `scipy` | Morphological operations |
| `scikit-image` | Image processing utilities |
| `matplotlib` | Visualization |
| `tqdm` | Progress bars |
| `torch` | Deep learning framework (U-Net 3D, DataLoader) |
| `scikit-learn` | Train/val/test split (GroupShuffleSplit) |

## Conventions

### Compatibility Shims

`pylidc` uses deprecated numpy/configparser APIs. Always include these before importing pylidc:

```python
import numpy as np
np.int = int

import configparser
configparser.SafeConfigParser = configparser.ConfigParser
```

### Naming

- CT volumes: `{patient_id}.npy` (e.g., `LIDC-IDRI-0078.npy`) — one file per patient
- Masks: same name, full-size binary mask with all nodules marked
- Masks are saved as `uint8` (0/1 binary).
- CT volumes are saved as raw HU values (float32).

### Consensus Masks

- Generated using `pylidc.utils.consensus(annotations, clevel=0.5)`.
- `clevel=0.5` means a voxel is included if ≥50% of radiologists marked it.
- All nodules are included regardless of annotation count; `num_annotations` in the CSV allows filtering later.
- Each nodule mask is inserted into a full-size mask (same shape as CT volume) at its bounding box position.

### Preprocessing Pipeline

Applied by `preprocessing_pipeline.py` before model training:

1. **Windowing HU** — Clip to [-1000, 600] to focus on lung tissue and nodules while preserving calcification info.
2. **Normalization** — Scale to [0, 1] after windowing.
3. **Train/val/test split** — 70/15/15 split grouped by patient to prevent data leakage.

Patch extraction and augmentation happen at training time in the DataLoader (dataset.py):

4. **Patch extraction** — Random 64×64×64 patches from full volumes. 50% centered on a nodule, 50% random (balanced sampling).
5. **Data augmentation** — Random flips and 90° rotations (applied at training time only).

### Annotation Features

Each nodule has these averaged features (1–5 scale unless noted):

| Feature | Description |
|---------|-------------|
| `subtlety` | How subtle the nodule appearance is |
| `internalStructure` | Internal structure (1=soft tissue, 2=fluid, 3=fat, 4=air) |
| `calcification` | Calcification pattern (1–6) |
| `sphericity` | How spherical the nodule is |
| `margin` | How well-defined the margin is |
| `lobulation` | Degree of lobulation |
| `spiculation` | Degree of spiculation |
| `texture` | Texture (1=non-solid/GGO, 5=solid) |
| `malignancy` | Malignancy likelihood (1=benign, 5=malignant) |

## Running the Pipeline

### Prerequisites

```bash
pip install pylidc pydicom SimpleITK matplotlib numpy scipy scikit-image tqdm torch scikit-learn
```

### Stage 1: Generate Full Volumes and Masks

```bash
python process_all_masks.py --output_dir output --clevel 0.5
```

This produces:
- `output/CT/{patient_id}.npy` — full CT volume per patient
- `output/masks/{patient_id}.npy` — full-size binary mask (all nodules)
- `output/metadata.csv` — annotation features per nodule

### Stage 2: Preprocess for Training

```bash
python preprocessing_pipeline.py --output_dir output
```

This produces `output/preprocessed/{train,val,test}/` with windowed and normalized full volumes.

### Stage 3: Train U-Net 3D

```bash
python train_unet3d_baseline.py --output_dir output --epochs 40 --batch_size 2 --eval_test
```

This trains the U-Net 3D and saves:
- `checkpoints/unet3d_baseline/best.pt` — best checkpoint (by validation Dice)
- `checkpoints/unet3d_baseline/last.pt` — last checkpoint
- `checkpoints/unet3d_baseline/history.csv` — loss and Dice curves per epoch
- `checkpoints/unet3d_baseline/summary.json` — final metrics

### U-Net 3D Architecture (`train_unet3d_baseline.py`)

- **Encoder**: 1 → 16 → 32 → 64 → 128 channels, MaxPool3d between levels
- **Decoder**: 128 → 64 → 32 → 16 channels, ConvTranspose3d + skip connections
- **Loss**: Dice Loss + BCE (weighted 0.5 each) for stable training
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Dataset**: `FullVolumeNoduleDataset` — loads full volumes, extracts 64³ patches on the fly
  - 50% patches centred on a nodule bounding box (positive mining)
  - 50% patches at random positions (background diversity)
- **Augmentation**: random flips + 90° rotations applied at training time only
- **Checkpoints**: best (by val Dice) + last saved to `checkpoints/unet3d_baseline/`

## Future Work

- [ ] U-Net 2D segmentation model (slice-by-slice)
- [ ] CNN classification model for malignancy prediction
- [ ] Training curves visualization notebook
