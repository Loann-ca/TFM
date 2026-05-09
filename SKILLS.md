# SKILLS.md — Reusable Workflows for TFM

## 1. Generate Consensus Masks for All Patients

**When to use:** You need to (re)generate masks, CT patches, and metadata for the full LIDC-IDRI dataset.

**Steps:**
1. Ensure `~/.pylidcrc` points to the DICOM directory.
2. Run:
   ```bash
   python process_all_masks.py --output_dir output --clevel 0.5
   ```
3. Output appears in `output/CT/`, `output/masks/`, and `output/metadata.csv`.

**Parameters:**
- `--clevel` — Consensus level (default 0.5). Lower values include more voxels, higher values are stricter.
- `--min_annotations` — Minimum radiologist annotations required (default 3).
- `--output_dir` — Where to write results (default `output`).

**Recovery:** The CSV is written incrementally. If the script crashes, delete the partial CSV and re-run — `.npy` files are overwritten safely.

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

## 3. Load Preprocessed Data for Model Training

**When to use:** You have generated the output and want to load it for training a segmentation or classification model.

**Steps:**
```python
import numpy as np
import pandas as pd
import os

output_dir = "output"
meta = pd.read_csv(os.path.join(output_dir, "metadata.csv"))

# Load a single nodule
row = meta.iloc[0]
ct = np.load(os.path.join(output_dir, "CT", f"{row.patient_id}_nod{row.nodule_idx}.npy"))
mask = np.load(os.path.join(output_dir, "masks", f"{row.patient_id}_nod{row.nodule_idx}.npy"))

# For classification: use row.malignancy, row.spiculation, etc.
print(f"Malignancy: {row.malignancy}, Shape: {ct.shape}")
```

---

## 4. Add pylidc Compatibility Shims

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

## 5. Configure pylidc Database Path

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

## 6. Visualize CT + Mask Overlay

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
