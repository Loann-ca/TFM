# SKILLS.md — Reusable Workflows for TFM

## 1. Generate Full CT Volumes and Masks

**When to use:** You need to (re)generate full CT volumes, full-size masks, and metadata for the LIDC-IDRI dataset.

**Steps:**
1. Ensure `~/.pylidcrc` points to the DICOM directory.
2. Run:
   ```bash
   python src/preprocessing/process_all_masks.py --output_dir output --clevel 0.5
   ```
3. Output: `output/CT/{patient_id}.npy` (full CT), `output/masks/{patient_id}.npy` (full-size mask with all nodules), `output/metadata.csv`.

**Parameters:**
- `--clevel` — Consensus level (default 0.5). Lower values include more voxels, higher values are stricter.
- `--output_dir` — Where to write results (default `output`).

**Note:** Each patient produces one CT file and one mask file. The mask has the same shape as the CT with all nodules marked.

---

## 2. Explore a Single Patient / Nodule

**When to use:** You want to visually inspect a specific patient's CT and nodule masks.

**Steps:**
1. Open `process_data_test.ipynb`.
2. Change the scan query to target a specific patient:
   ```python
   scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == "LIDC-IDRI-0078").first()
   ```
3. Change `nodules[0]` to `nodules[N]` to select a different nodule.
4. Run all cells to see CT slices, masks, and overlays.

---

## 3. Run Preprocessing Pipeline

**When to use:** After generating full volumes with `process_all_masks.py`, before training the U-Net.

**Steps:**
```bash
python src/preprocessing/preprocessing_pipeline.py --output_dir output
```

**What it does:**
1. Splits data into train/val/test (70/15/15) grouped by patient
2. Applies HU windowing [-1000, 600] and normalizes to [0, 1]
3. Saves full preprocessed volumes in `output/preprocessed/{train,val,test}/`

**Parameters:**
- `--hu_min` / `--hu_max` — Windowing range (default -1000 / 600).

**Patch extraction and augmentation** happen at training time in the DataLoader (`dataset.py`), not during preprocessing.

---

## 4. Load Preprocessed Data

**When to use:** You want to load a full preprocessed volume for inspection or custom processing.

**Steps:**
```python
import numpy as np
import pandas as pd
import os

split_dir = "output/preprocessed/train"
meta = pd.read_csv(os.path.join(split_dir, "metadata.csv"))
patient_id = meta.patient_id.unique()[0]

ct = np.load(os.path.join(split_dir, "CT", f"{patient_id}.npy"))
mask = np.load(os.path.join(split_dir, "masks", f"{patient_id}.npy"))

print(f"CT shape: {ct.shape}, Mask voxels: {mask.sum()}")
```

---

## 5. Add pylidc Compatibility Shims

**When to use:** Any new Python file or notebook that imports `pylidc`.

**Pattern:**
```python
import numpy as np
np.int = int

import configparser
configparser.SafeConfigParser = configparser.ConfigParser

import pylidc as pl
from pylidc.utils import consensus
```

These shims fix deprecation errors in `pylidc` with numpy ≥1.24 and Python ≥3.12.

---

## 6. Configure pylidc Database Path

**When to use:** Setting up a new environment or machine.

**Steps:**
1. Create `~/.pylidcrc`:
   ```ini
   [dicom]
   path = /absolute/path/to/lidc_idri
   ```
2. Verify:
   ```python
   import pylidc as pl
   scans = pl.query(pl.Scan).all()
   print(f"Found {len(scans)} scans")
   ```

---

## 7. Train U-Net 3D

**When to use:** After preprocessing, to train the segmentation model.

**Steps:**
```bash
python src/segmentation/unet3d_segmentation/train_unet3d_baseline.py --output_dir output --epochs 50 --batch_size 4 --lr 1e-3
```

**What it does:**
1. Loads preprocessed data via DataLoader with augmentation on train
2. Trains U-Net 3D with Dice+BCE loss
3. Saves best model (by val Dice) to `output/models/unet3d_best.pth`
4. Early stopping with patience=10

**Parameters:**
- `--epochs` — Max training epochs (default 50)
- `--batch_size` — Batch size (default 4). Reduce to 2 if GPU runs out of memory.
- `--lr` — Learning rate (default 1e-3)
- `--num_workers` — DataLoader workers (default 2)

**Load a trained model:**
```python
import torch
from unet3d import UNet3D

model = UNet3D()
checkpoint = torch.load("output/models/unet3d_best.pth", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

---

## 8. Log Questions & Answers

**When to use:** Every time the user asks a conceptual or technical question about the project.

**Steps:**
1. After answering the question, append the Q&A to `QA.md`.
2. Use the next sequential number as heading.
3. Format:
   ```markdown
   ## N. <Question>

   **Respuesta:** <Answer>
   ```
4. Keep answers concise but complete. Include code snippets if relevant.

**File:** `QA.md` in the project root.

---

## 9. Visualize CT + Mask Overlay

**When to use:** Quick visual check of a nodule segmentation.

**Pattern:**
```python
import matplotlib.pyplot as plt

slice_idx = ct_patch.shape[2] // 2

plt.figure(figsize=(8, 8))
plt.imshow(ct_patch[:, :, slice_idx], cmap="gray")
plt.imshow(mask[:, :, slice_idx], alpha=0.4, cmap="Reds")
plt.title("CT + Mask overlay")
plt.axis("off")
plt.show()
```
