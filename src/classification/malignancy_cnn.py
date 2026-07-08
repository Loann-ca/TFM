"""
Malignancy classification from segmented lung nodules.

This script supports two workflows:
1) Train a CNN to classify malignancy score (1..5) from detected nodules.
2) Predict malignancy for nodules from a CT + segmentation mask pair.

The detector source can be U-Net predictions or ground-truth masks.
Each connected component in the mask is treated as one candidate nodule.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CLASS_NAMES = ["1", "2", "3", "4", "5"]


@dataclass
class NoduleSample:
    ct_path: str
    center: Tuple[int, int, int]
    bbox: Tuple[int, int, int, int, int, int]
    component_voxels: int
    label_value: float | None
    patient_id: str


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_bbox_range(text: str) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*(-?\d+)\s*-\s*(-?\d+)\s*", str(text))
    if m is None:
        raise ValueError(f"Invalid bbox range: {text}")
    a, b = int(m.group(1)), int(m.group(2))
    if a > b:
        a, b = b, a
    return a, b


def metadata_bbox_xyz(row: pd.Series) -> Tuple[int, int, int, int, int, int]:
    x0, x1 = parse_bbox_range(str(row["bbox_x"]))
    y0, y1 = parse_bbox_range(str(row["bbox_y"]))
    z0, z1 = parse_bbox_range(str(row["bbox_z"]))
    return x0, x1, y0, y1, z0, z1


def bbox_iou3d(a: Tuple[int, int, int, int, int, int], b: Tuple[int, int, int, int, int, int]) -> float:
    ax0, ax1, ay0, ay1, az0, az1 = a
    bx0, bx1, by0, by1, bz0, bz1 = b

    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    iz0, iz1 = max(az0, bz0), min(az1, bz1)

    if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
        return 0.0

    inter = (ix1 - ix0 + 1) * (iy1 - iy0 + 1) * (iz1 - iz0 + 1)
    va = (ax1 - ax0 + 1) * (ay1 - ay0 + 1) * (az1 - az0 + 1)
    vb = (bx1 - bx0 + 1) * (by1 - by0 + 1) * (bz1 - bz0 + 1)
    union = va + vb - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def center_of_bbox(b: Tuple[int, int, int, int, int, int]) -> Tuple[float, float, float]:
    x0, x1, y0, y1, z0, z1 = b
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1))


def label_component(
    component_bbox: Tuple[int, int, int, int, int, int],
    meta_rows: Sequence[pd.Series],
    center_xyz: Tuple[int, int, int],
) -> float | None:
    if not meta_rows:
        return None

    best_iou = -1.0
    best_row = None
    for row in meta_rows:
        gt_bbox = metadata_bbox_xyz(row)
        iou = bbox_iou3d(component_bbox, gt_bbox)
        if iou > best_iou:
            best_iou = iou
            best_row = row

    if best_row is None or best_iou <= 0.0:
        cx, cy, cz = center_xyz
        best_dist = float("inf")
        best_row = None
        for row in meta_rows:
            gx, gy, gz = center_of_bbox(metadata_bbox_xyz(row))
            d2 = (cx - gx) ** 2 + (cy - gy) ** 2 + (cz - gz) ** 2
            if d2 < best_dist:
                best_dist = d2
                best_row = row

    if best_row is None:
        return None

    malignancy = float(best_row["malignancy"])
    return float(np.clip(malignancy, 1.0, 5.0))


def extract_2p5d_patch(volume: np.ndarray, center: Tuple[int, int, int], patch_size: int) -> np.ndarray:
    if patch_size <= 0 or patch_size % 2 != 0:
        raise ValueError("patch_size must be a positive even integer")

    cx, cy, cz = center
    h, w, d = volume.shape
    half = patch_size // 2

    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    z0, z1 = cz - half, cz + half

    pad_x0 = max(0, -x0)
    pad_y0 = max(0, -y0)
    pad_z0 = max(0, -z0)
    pad_x1 = max(0, x1 - h)
    pad_y1 = max(0, y1 - w)
    pad_z1 = max(0, z1 - d)

    if any(v > 0 for v in [pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1]):
        volume = np.pad(
            volume,
            ((pad_x0, pad_x1), (pad_y0, pad_y1), (pad_z0, pad_z1)),
            mode="constant",
            constant_values=0.0,
        )
        cx += pad_x0
        cy += pad_y0
        cz += pad_z0
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half
        z0, z1 = cz - half, cz + half

    axial = volume[x0:x1, y0:y1, cz]
    coronal = volume[x0:x1, cy, z0:z1]
    sagittal = volume[cx, y0:y1, z0:z1]

    patch = np.stack([axial, coronal, sagittal], axis=0).astype(np.float32)
    return patch


def components_from_mask(mask: np.ndarray, min_voxels: int) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int, int, int], int]]:
    labeled, n = ndimage.label(mask > 0)
    objects = ndimage.find_objects(labeled)

    out: List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int, int, int], int]] = []
    for label_idx, slc in enumerate(objects, start=1):
        if slc is None:
            continue

        comp = labeled[slc] == label_idx
        voxels = int(comp.sum())
        if voxels < min_voxels:
            continue

        x0, x1 = int(slc[0].start), int(slc[0].stop) - 1
        y0, y1 = int(slc[1].start), int(slc[1].stop) - 1
        z0, z1 = int(slc[2].start), int(slc[2].stop) - 1
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        cz = (z0 + z1) // 2

        out.append(((cx, cy, cz), (x0, x1, y0, y1, z0, z1), voxels))

    return out


def load_split_samples(
    split_dir: str,
    mask_dir: str | None,
    patch_size: int,
    min_voxels: int,
    require_labels: bool,
) -> List[NoduleSample]:
    ct_dir = os.path.join(split_dir, "CT")
    meta_path = os.path.join(split_dir, "metadata.csv")

    if mask_dir is None:
        mask_dir = os.path.join(split_dir, "masks")

    if not os.path.isdir(ct_dir):
        raise FileNotFoundError(f"CT dir not found: {ct_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask dir not found: {mask_dir}")
    if require_labels and not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.csv not found: {meta_path}")

    if require_labels:
        meta = pd.read_csv(meta_path)
        grouped: Dict[str, List[pd.Series]] = {}
        for patient_id, group in meta.groupby("patient_id"):
            grouped[str(patient_id)] = [row for _, row in group.iterrows()]
    else:
        grouped = {}

    samples: List[NoduleSample] = []
    ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith(".npy")])

    for f in ct_files:
        patient_id = os.path.splitext(f)[0]
        ct_path = os.path.join(ct_dir, f)
        mask_path = os.path.join(mask_dir, f)
        if not os.path.exists(mask_path):
            continue

        ct = np.load(ct_path, mmap_mode="r")
        mask = np.load(mask_path)
        if ct.shape != mask.shape:
            continue

        h, w, d = ct.shape
        half = patch_size // 2
        components = components_from_mask(mask, min_voxels=min_voxels)
        meta_rows = grouped.get(patient_id, [])

        for center, bbox, voxels in components:
            cx, cy, cz = center
            cx = int(np.clip(cx, half, max(half, h - half - 1)))
            cy = int(np.clip(cy, half, max(half, w - half - 1)))
            cz = int(np.clip(cz, 0, max(0, d - 1)))

            label_idx = label_component(bbox, meta_rows, (cx, cy, cz)) if require_labels else None
            if require_labels and label_idx is None:
                continue

            samples.append(
                NoduleSample(
                    ct_path=ct_path,
                    center=(cx, cy, cz),
                    bbox=bbox,
                    component_voxels=voxels,
                    label_value=label_idx,
                    patient_id=patient_id,
                )
            )

    return samples


class MalignancyNoduleDataset(Dataset):
    def __init__(self, samples: List[NoduleSample], patch_size: int, augment: bool, seed: int) -> None:
        if len(samples) == 0:
            raise RuntimeError("No samples found for dataset.")
        self.samples = samples
        self.patch_size = patch_size
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        ct = np.load(s.ct_path)
        patch = extract_2p5d_patch(ct, center=s.center, patch_size=self.patch_size)

        if self.augment:
            rng = np.random.RandomState(self.seed * 100_000 + idx)
            # Geometric augmentation
            if rng.rand() < 0.5:
                patch = np.flip(patch, axis=1).copy()
            if rng.rand() < 0.5:
                patch = np.flip(patch, axis=2).copy()
            k = int(rng.randint(0, 4))
            if k > 0:
                patch = np.rot90(patch, k=k, axes=(1, 2)).copy()
            # Intensity augmentation (brightness + contrast jitter)
            brightness = rng.uniform(-0.1, 0.1)
            contrast = rng.uniform(0.9, 1.1)
            patch = np.clip(patch * contrast + brightness, 0.0, 1.0).astype(np.float32)
            # Gaussian noise
            if rng.rand() < 0.3:
                patch = np.clip(patch + rng.normal(0, 0.02, patch.shape).astype(np.float32), 0.0, 1.0)

        out: Dict[str, torch.Tensor] = {
            "image": torch.from_numpy(patch),
        }
        if s.label_value is not None:
            out["label"] = torch.tensor(s.label_value, dtype=torch.float32)
        out["center"] = torch.tensor(s.center, dtype=torch.int32)
        return out


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock2D(nn.Module):
    """Pre-activation residual block (He et al. 2016)."""
    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MalignancyCNN(nn.Module):
    """
    Deeper 2.5D CNN for nodule malignancy classification.

    Architecture inspired by NoduleX (Causey 2018) and DeepLung:
    - Stem conv → 4 residual stages with downsampling → GAP → dropout head.
    - Supports regression (task='regression') and binary (task='binary').
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        task: str = "regression",
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        assert task in ("regression", "binary"), f"Unknown task: {task}"
        self.task = task
        c = base_channels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # 4 residual stages, each doubles channels and halves spatial
        self.stage1 = nn.Sequential(ResBlock2D(c), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(
            nn.Conv2d(c, c * 2, 1, bias=False),
            ResBlock2D(c * 2), nn.MaxPool2d(2)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, 1, bias=False),
            ResBlock2D(c * 4), nn.MaxPool2d(2)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(c * 4, c * 8, 1, bias=False),
            ResBlock2D(c * 8),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        out_dim = 1  # regression or binary (logit)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(c * 8, c * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(c * 4, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.head(x).squeeze(1)


# Keep alias for backward compatibility with saved checkpoints
SmallMalignancyCNN = MalignancyCNN


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class FocalBCELoss(nn.Module):
    """
    Focal loss for binary classification (Lin et al. 2017).
    Reduces weight of easy examples, focuses on hard ones.
    """
    def __init__(self, gamma: float = 2.0, pos_weight: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets.float())
        p_t = torch.exp(-bce_loss)
        focal = (1.0 - p_t) ** self.gamma * bce_loss
        return focal.mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    criterion: nn.Module,
    task: str = "regression",
) -> Dict[str, float]:
    """
    Run one epoch. Returns a dict of metrics.
    task: 'regression' or 'binary'
    """
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0
    total_abs_error = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[float] = []  # for AUC in binary mode

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True).float()

            logits = model(x)
            loss = criterion(logits, y)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            total_loss += float(loss.item()) * y.size(0)
            total += int(y.numel())

            if task == "binary":
                probs = torch.sigmoid(logits).detach().cpu()
                preds_bin = (probs >= 0.5).long()
                targets_bin = y.detach().cpu().long()
                total_correct += int((preds_bin == targets_bin).sum().item())
                total_abs_error += float(torch.abs(probs - y.cpu()).sum().item())
                all_preds.extend(preds_bin.tolist())
                all_targets.extend(targets_bin.tolist())
                all_probs.extend(probs.tolist())
            else:
                rounded_preds = torch.clamp(torch.round(logits.detach()), 1, 5).cpu()
                rounded_y = torch.clamp(torch.round(y.detach()), 1, 5).cpu().long()
                total_correct += int((rounded_preds.long() == rounded_y).sum().item())
                total_abs_error += float(torch.abs(logits.detach().cpu() - y.detach().cpu()).sum().item())
                all_preds.extend(rounded_preds.long().tolist())
                all_targets.extend(rounded_y.tolist())

    mean_loss = total_loss / max(total, 1)
    mae = float(total_abs_error / max(total, 1))
    acc = float(total_correct / max(total, 1))

    metrics: Dict[str, float] = {"loss": mean_loss, "mae": mae, "acc": acc}

    if total > 0:
        if task == "binary":
            metrics["macro_f1"] = float(f1_score(all_targets, all_preds, average="macro", labels=[0, 1], zero_division=0))
            metrics["balanced_acc"] = float(balanced_accuracy_score(all_targets, all_preds))
            if len(set(all_targets)) > 1:
                metrics["auc"] = float(roc_auc_score(all_targets, all_probs))
            else:
                metrics["auc"] = float("nan")
        else:
            metrics["macro_f1"] = float(f1_score(all_targets, all_preds, average="macro", labels=[1, 2, 3, 4, 5], zero_division=0))
            metrics["balanced_acc"] = float(balanced_accuracy_score(all_targets, all_preds))
            metrics["auc"] = float("nan")
    else:
        metrics["macro_f1"] = 0.0
        metrics["balanced_acc"] = 0.0
        metrics["auc"] = float("nan")

    return metrics


def infer_components(
    model: nn.Module,
    device: torch.device,
    ct_path: str,
    mask_path: str,
    patch_size: int,
    min_voxels: int,
) -> List[Dict[str, object]]:
    ct = np.load(ct_path)
    mask = np.load(mask_path)
    if ct.shape != mask.shape:
        raise ValueError(f"CT/mask shape mismatch: {ct.shape} vs {mask.shape}")

    components = components_from_mask(mask, min_voxels=min_voxels)
    model.eval()

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for idx, (center, bbox, voxels) in enumerate(components, start=1):
            patch = extract_2p5d_patch(ct, center=center, patch_size=patch_size)
            x = torch.from_numpy(patch).unsqueeze(0).to(device)
            pred_score = float(model(x).item())
            pred_round = int(np.clip(np.rint(pred_score), 1, 5))

            row: Dict[str, object] = {
                "nodule_id": idx,
                "center_x": int(center[0]),
                "center_y": int(center[1]),
                "center_z": int(center[2]),
                "bbox_x": f"{bbox[0]}-{bbox[1]}",
                "bbox_y": f"{bbox[2]}-{bbox[3]}",
                "bbox_z": f"{bbox[4]}-{bbox[5]}",
                "voxels": int(voxels),
                "pred_malignancy": pred_score,
                "pred_malignancy_rounded": pred_round,
            }
            rows.append(row)

    return rows


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN malignancy classification for U-Net detected nodules")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Train a malignancy CNN")
    train.add_argument("--output_dir", type=str, default="output", help="Root output directory")
    train.add_argument("--save_dir", type=str, default="checkpoints/malignancy_cnn", help="Directory for run artifacts")
    train.add_argument("--run_name", type=str, default=None, help="Optional run name")
    train.add_argument("--train_mask_dir", type=str, default=None, help="Mask dir for train split (default: preprocessed/train/masks)")
    train.add_argument("--val_mask_dir", type=str, default=None, help="Mask dir for val split (default: preprocessed/val/masks)")
    train.add_argument("--test_mask_dir", type=str, default=None, help="Mask dir for test split (default: preprocessed/test/masks)")
    train.add_argument("--epochs", type=int, default=150)
    train.add_argument("--batch_size", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight_decay", type=float, default=1e-5)
    train.add_argument("--patch_size", type=int, default=64)
    train.add_argument("--min_voxels", type=int, default=20, help="Minimum component voxels to keep")
    train.add_argument("--base_channels", type=int, default=32)
    train.add_argument("--num_workers", type=int, default=0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    train.add_argument("--eval_test", action="store_true")
    train.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    train.add_argument(
        "--task",
        type=str,
        default="regression",
        choices=["regression", "binary"],
        help="'regression': predict malignancy 1-5 (MSE). 'binary': benign(1-2)=0 vs malignant(4-5)=1, skip 3.",
    )
    train.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Gamma for FocalBCELoss in binary mode (0 = standard BCE)",
    )
    train.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout probability in the classifier head",
    )

    pred = sub.add_parser("predict", help="Predict malignancy for detected nodules")
    pred.add_argument("--input_ct", type=str, required=True, help="Input CT .npy (H, W, D)")
    pred.add_argument("--input_mask", type=str, required=True, help="Input predicted mask .npy (H, W, D)")
    pred.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (best.pt)")
    pred.add_argument("--output_csv", type=str, default=None, help="Output CSV path")
    pred.add_argument("--patch_size", type=int, default=64)
    pred.add_argument("--min_voxels", type=int, default=20)
    pred.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    return parser.parse_args()


def default_run_name() -> str:
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def train_main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)

    pre_root = os.path.join(args.output_dir, "preprocessed")
    train_split = os.path.join(pre_root, "train")
    val_split = os.path.join(pre_root, "val")
    test_split = os.path.join(pre_root, "test")

    task = args.task

    train_samples_raw = load_split_samples(
        split_dir=train_split,
        mask_dir=args.train_mask_dir,
        patch_size=args.patch_size,
        min_voxels=args.min_voxels,
        require_labels=True,
    )
    val_samples_raw = load_split_samples(
        split_dir=val_split,
        mask_dir=args.val_mask_dir,
        patch_size=args.patch_size,
        min_voxels=args.min_voxels,
        require_labels=True,
    )

    # Binary mode: remap labels and exclude ambiguous (malignancy = 3)
    def remap_binary(samples: List[NoduleSample]) -> List[NoduleSample]:
        out = []
        for s in samples:
            if s.label_value is None:
                continue
            if s.label_value <= 2.5:      # 1 or 2 → benign
                out.append(NoduleSample(**{**s.__dict__, "label_value": 0.0}))
            elif s.label_value >= 3.5:   # 4 or 5 → malignant
                out.append(NoduleSample(**{**s.__dict__, "label_value": 1.0}))
            # malignancy ≈ 3 → excluded (ambiguous)
        return out

    if task == "binary":
        train_samples = remap_binary(train_samples_raw)
        val_samples   = remap_binary(val_samples_raw)
    else:
        train_samples = train_samples_raw
        val_samples   = val_samples_raw

    ensure_dir(args.save_dir)
    run_dir = os.path.join(args.save_dir, args.run_name.strip() if args.run_name else default_run_name())
    ensure_dir(run_dir)

    train_ds = MalignancyNoduleDataset(train_samples, patch_size=args.patch_size, augment=True, seed=args.seed)
    val_ds = MalignancyNoduleDataset(val_samples, patch_size=args.patch_size, augment=False, seed=args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    labels_np = np.array([int(np.rint(s.label_value)) for s in train_samples], dtype=np.int64)

    if task == "binary":
        class_counts = np.bincount(labels_np, minlength=2)
        # pos_weight = n_neg / n_pos for binary BCE
        n_neg = float(class_counts[0]) if len(class_counts) > 0 else 1.0
        n_pos = float(class_counts[1]) if len(class_counts) > 1 else 1.0
        pos_weight = n_neg / max(n_pos, 1.0)
        criterion: nn.Module = FocalBCELoss(gamma=args.focal_gamma, pos_weight=pos_weight)
    else:
        class_counts = np.bincount(labels_np, minlength=5)
        # Weighted MSE: higher weight to rare classes
        sample_weights = np.ones(len(train_samples), dtype=np.float32)
        class_weights_raw = np.where(
            class_counts > 0, class_counts.sum() / np.maximum(class_counts, 1.0), 0.0
        ).astype(np.float32)
        class_weights_raw = class_weights_raw / np.maximum(class_weights_raw.sum(), 1e-6) * len(class_weights_raw)
        for i, s in enumerate(train_samples):
            cls_idx = int(np.rint(s.label_value)) - 1
            if 0 <= cls_idx < len(class_weights_raw):
                sample_weights[i] = class_weights_raw[cls_idx]
        class_counts = class_counts  # reuse for logging
        criterion = nn.MSELoss()

    model = MalignancyCNN(
        in_channels=3,
        base_channels=args.base_channels,
        task=task,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",      # Queremos minimizar val_loss
        factor=0.5,      # Reduce el LR a la mitad
        patience=5,      # Espera 5 épocas sin mejorar
        min_lr=1e-6,
    )
    best_val_acc = -1.0
    best_val_macro_f1 = -1.0
    best_val_balanced_acc = -1.0
    best_val_auc = -1.0
    best_val_loss = float("inf")

    epochs_without_improvement = 0
    best_epoch = 0

    history: List[Dict[str, float]] = []

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Task: {task}")
    print(f"Train nodules: {len(train_ds)}")
    print(f"Val nodules: {len(val_ds)}")
    print(f"Rounded class counts train: {class_counts.tolist()}")

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, criterion, task=task)
        va = run_epoch(model, val_loader, None, device, criterion, task=task)

        scheduler.step(va["loss"])

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        auc_str = f" val_auc={va['auc']:.4f}" if not (isinstance(va['auc'], float) and np.isnan(va['auc'])) else ""
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"tr_loss={tr['loss']:.4f} tr_mae={tr['mae']:.4f} tr_f1={tr['macro_f1']:.4f} tr_bacc={tr['balanced_acc']:.4f} | "
            f"va_loss={va['loss']:.4f} va_mae={va['mae']:.4f} va_f1={va['macro_f1']:.4f} va_bacc={va['balanced_acc']:.4f}{auc_str}"
        )

        last_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "args": vars(args),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(last_ckpt, os.path.join(run_dir, "last.pt"))

        # Primary metric: AUC for binary, balanced_acc for regression
        va_primary = va["auc"] if task == "binary" and not np.isnan(va["auc"]) else va["balanced_acc"]
        best_primary = best_val_auc if task == "binary" else best_val_balanced_acc
        improved = (
            va_primary > best_primary
            or (va_primary == best_primary and va["macro_f1"] > best_val_macro_f1)
            or (va_primary == best_primary and va["macro_f1"] == best_val_macro_f1 and va["loss"] < best_val_loss)
        )

        if improved:
            best_val_acc = va["acc"]
            best_val_macro_f1 = va["macro_f1"]
            best_val_balanced_acc = va["balanced_acc"]
            best_val_auc = va["auc"]
            best_val_loss = va["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0

            best_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_auc": best_val_auc,
                "best_val_loss": best_val_loss,
                "args": vars(args),
            }
            torch.save(best_ckpt, os.path.join(run_dir, "best.pt"))
            print(f"✓ Best checkpoint saved (val_bacc={best_val_balanced_acc:.4f}, val_auc={best_val_auc:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement ({epochs_without_improvement}/{args.patience})")

        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)

        if epochs_without_improvement >= args.patience:
            print("\nEarly stopping triggered.")
            print(f"Best epoch: {best_epoch}")
            print(f"Best val balanced_acc: {best_val_balanced_acc:.4f}  val_auc: {best_val_auc:.4f}")
            break

    summary: Dict[str, object] = {
        "run_id": os.path.basename(run_dir),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "best_val_acc": best_val_acc,
        "best_val_auc": best_val_auc,
        "train_nodules": len(train_ds),
        "val_nodules": len(val_ds),
        "class_counts_train": class_counts.tolist(),
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_balanced_acc": best_val_balanced_acc,
        "hyperparameters": vars(args),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }

    if args.eval_test:
        test_samples_raw = load_split_samples(
            split_dir=test_split,
            mask_dir=args.test_mask_dir,
            patch_size=args.patch_size,
            min_voxels=args.min_voxels,
            require_labels=True,
        )
        test_samples = remap_binary(test_samples_raw) if task == "binary" else test_samples_raw
        test_ds = MalignancyNoduleDataset(test_samples, patch_size=args.patch_size, augment=False, seed=args.seed)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        te = run_epoch(model, test_loader, None, device, criterion, task=task)
        for k, v in te.items():
            summary[f"test_{k}"] = v
        summary["test_nodules"] = len(test_ds)
        te_str = "  ".join(f"{k}={v:.4f}" for k, v in te.items() if not (isinstance(v, float) and np.isnan(v)))
        print(f"Test | {te_str}")

    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    registry = os.path.join(args.save_dir, "training_runs_history.jsonl")
    with open(registry, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "run_id": os.path.basename(run_dir),
                    "run_dir": run_dir,
                    "task": task,
                    "best_val_acc": best_val_acc,
                    "best_val_auc": best_val_auc,
            )
            + "\n"
        )

    print("Training finished.")
    print(f"Best val macro-F1: {best_val_macro_f1:.4f}")
    print(f"Artifacts saved in: {run_dir}")


def predict_main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    base_channels = int(ckpt_args.get("base_channels", 32))
    patch_size = int(ckpt_args.get("patch_size", args.patch_size))
    task = str(ckpt_args.get("task", "regression"))
    dropout = float(ckpt_args.get("dropout", 0.4))

    model = MalignancyCNN(
        in_channels=3, base_channels=base_channels, task=task, dropout=dropout
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    if args.output_csv is None:
        patient = os.path.splitext(os.path.basename(args.input_ct))[0]
        args.output_csv = os.path.join("output", "classification", f"{patient}_malignancy.csv")

    rows = infer_components(
        model=model,
        device=device,
        ct_path=args.input_ct,
        mask_path=args.input_mask,
        patch_size=patch_size,
        min_voxels=args.min_voxels,
    )

    ensure_dir(os.path.dirname(os.path.abspath(args.output_csv)))
    fieldnames = [
        "nodule_id",
        "center_x",
        "center_y",
        "center_z",
        "bbox_x",
        "bbox_y",
        "bbox_z",
        "voxels",
        "pred_malignancy",
        "pred_malignancy_rounded",
    ]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Device: {device}")
    print(f"Detected nodules: {len(rows)}")
    print(f"Saved: {args.output_csv}")


def main() -> None:
    args = parse_args()
    if args.cmd == "train":
        train_main(args)
    elif args.cmd == "predict":
        predict_main(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
