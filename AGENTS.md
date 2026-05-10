# AGENTS.md — TFM: Lung Nodule Detection & Classification (LIDC-IDRI)

## Project Overview

Master's thesis project for detecting and classifying lung nodules from CT scans using the LIDC-IDRI dataset. The pipeline has three stages:

1. **Data preprocessing** — Extract consensus masks and CT patches from DICOM data using `pylidc`.
2. **Segmentation** — U-Net models (2D and 3D) to segment nodules from CT volumes.
3. **Classification** — CNN to predict malignancy and other annotation features from segmented nodules.

## Repository Structure

```
TFM/
├── process_all_masks.py       # Stage 1: generates masks for all patients/nodules
├── preprocessing_pipeline.py  # Stage 2: windowing, normalization, padding, split
├── visualize_nodules.ipynb    # Visualize processed nodules and metadata
├── process_data.ipynb         # Exploratory notebook (single patient)
├── process_data_test.ipynb    # Exploratory notebook with visualization
├── output/                    # Generated data (gitignored)
│   ├── CT/                    #   Raw CT patches as .npy files
│   ├── masks/                 #   Consensus masks as .npy files
│   ├── metadata.csv           #   Annotation features per nodule
│   └── preprocessed/          #   Ready-to-train data
│       ├── train/             #     CT/, masks/, metadata.csv
│       ├── val/               #     CT/, masks/, metadata.csv
│       └── test/              #     CT/, masks/, metadata.csv
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

- Output files: `{patient_id}_nod{nodule_index}.npy` (e.g., `LIDC-IDRI-0078_nod0.npy`)
- Masks are saved as `uint8` (0/1 binary).
- CT patches are saved as raw HU values (float).

### Consensus Masks

- Generated using `pylidc.utils.consensus(annotations, clevel=0.5)`.
- `clevel=0.5` means a voxel is included if ≥50% of radiologists marked it.
- All nodules are included regardless of annotation count; `num_annotations` in the CSV allows filtering later.

### Preprocessing Pipeline

Applied by `preprocessing_pipeline.py` before model training:

1. **Windowing HU** — Clip to [-1000, 600] to focus on lung tissue and nodules while preserving calcification info.
2. **Normalization** — Scale to [0, 1] after windowing.
3. **Padding/Crop** — Pad with zeros (air) or center-crop to a fixed size (default 64×64×64).
4. **Train/val/test split** — 70/15/15 split grouped by patient to prevent data leakage.
5. **Data augmentation** — Random flips and 90° rotations (applied at training time, not during preprocessing).

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
pip install pylidc pydicom SimpleITK matplotlib numpy scipy scikit-image tqdm
```

### Stage 1: Generate All Masks

```bash
python process_all_masks.py --output_dir output --clevel 0.5
```

This produces:
- `output/CT/{patient}_nod{i}.npy` — CT patch for each nodule
- `output/masks/{patient}_nod{i}.npy` — binary consensus mask
- `output/metadata.csv` — annotation features for classification

### Stage 2: Preprocess for Training

```bash
python preprocessing_pipeline.py --output_dir output --target_size 64
```

This produces `output/preprocessed/{train,val,test}/` with windowed, normalized, and padded data ready for the U-Net.

## Future Work

- [ ] U-Net 3D segmentation model
- [ ] U-Net 2D segmentation model (slice-by-slice)
- [ ] CNN classification model for malignancy prediction
- [ ] Data augmentation pipeline
- [ ] Train/validation/test split strategy
