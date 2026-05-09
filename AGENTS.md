# AGENTS.md — TFM: Lung Nodule Detection & Classification (LIDC-IDRI)

## Project Overview

Master's thesis project for detecting and classifying lung nodules from CT scans using the LIDC-IDRI dataset. The pipeline has three stages:

1. **Data preprocessing** — Extract consensus masks and CT patches from DICOM data using `pylidc`.
2. **Segmentation** — U-Net models (2D and 3D) to segment nodules from CT volumes.
3. **Classification** — CNN to predict malignancy and other annotation features from segmented nodules.

## Repository Structure

```
TFM/
├── process_all_masks.py       # Batch script: generates masks for all patients/nodules
├── process_data.ipynb         # Exploratory notebook (single patient)
├── process_data_test.ipynb    # Exploratory notebook with visualization
├── output/                    # Generated data (gitignored)
│   ├── CT/                    #   CT patches as .npy files
│   ├── masks/                 #   Consensus masks as .npy files
│   └── metadata.csv           #   Annotation features per nodule
├── .devcontainer/
│   └── devcontainer.json
├── AGENTS.md
├── SKILLS.md
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
- Only nodules with ≥3 annotations are processed (configurable via `--min_annotations`).

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

### Generate All Masks

```bash
python process_all_masks.py --output_dir output --clevel 0.5
```

This produces:
- `output/CT/{patient}_nod{i}.npy` — CT patch for each nodule
- `output/masks/{patient}_nod{i}.npy` — binary consensus mask
- `output/metadata.csv` — annotation features for classification

## Future Work

- [ ] U-Net 3D segmentation model
- [ ] U-Net 2D segmentation model (slice-by-slice)
- [ ] CNN classification model for malignancy prediction
- [ ] Data augmentation pipeline
- [ ] Train/validation/test split strategy
