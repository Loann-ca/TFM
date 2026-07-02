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
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CLASS_NAMES = ["1", "2", "3", "4", "5"]


@dataclass
class NoduleSample:
    ct_path: str
    center: Tuple[int, int, int]
    bbox: Tuple[int, int, int, int, int, int]
    component_voxels: int
    label_idx: int | None
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
) -> int | None:
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
    cls = int(np.clip(np.rint(malignancy), 1, 5))
    return cls - 1


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
                    label_idx=label_idx,
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
            if rng.rand() < 0.5:
                patch = np.flip(patch, axis=1).copy()
            if rng.rand() < 0.5:
                patch = np.flip(patch, axis=2).copy()
            k = int(rng.randint(0, 4))
            if k > 0:
                patch = np.rot90(patch, k=k, axes=(1, 2)).copy()

        out: Dict[str, torch.Tensor] = {
            "image": torch.from_numpy(patch),
        }
        if s.label_idx is not None:
            out["label"] = torch.tensor(s.label_idx, dtype=torch.long)
        out["center"] = torch.tensor(s.center, dtype=torch.int32)
        return out


class SmallMalignancyCNN(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, num_classes: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == y).sum().item())
            total += int(y.numel())
            total_loss += float(loss.item()) * y.size(0)

    mean_loss = total_loss / max(total, 1)
    acc = float(total_correct / max(total, 1))
    return mean_loss, acc


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
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))

            row: Dict[str, object] = {
                "nodule_id": idx,
                "center_x": int(center[0]),
                "center_y": int(center[1]),
                "center_z": int(center[2]),
                "bbox_x": f"{bbox[0]}-{bbox[1]}",
                "bbox_y": f"{bbox[2]}-{bbox[3]}",
                "bbox_z": f"{bbox[4]}-{bbox[5]}",
                "voxels": int(voxels),
                "pred_malignancy": int(pred_idx + 1),
            }
            for i, cls in enumerate(CLASS_NAMES):
                row[f"prob_{cls}"] = float(probs[i])
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
    train.add_argument("--epochs", type=int, default=40)
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

    train_samples = load_split_samples(
        split_dir=train_split,
        mask_dir=args.train_mask_dir,
        patch_size=args.patch_size,
        min_voxels=args.min_voxels,
        require_labels=True,
    )
    val_samples = load_split_samples(
        split_dir=val_split,
        mask_dir=args.val_mask_dir,
        patch_size=args.patch_size,
        min_voxels=args.min_voxels,
        require_labels=True,
    )

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

    labels_np = np.array([int(s.label_idx) for s in train_samples], dtype=np.int64)
    class_counts = np.bincount(labels_np, minlength=5)
    class_weights = np.where(class_counts > 0, class_counts.sum() / np.maximum(class_counts, 1), 0.0).astype(np.float32)
    class_weights = class_weights / np.maximum(class_weights.sum(), 1e-6) * len(class_weights)

    model = SmallMalignancyCNN(in_channels=3, base_channels=args.base_channels, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    history: List[Dict[str, float]] = []

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Train nodules: {len(train_ds)}")
    print(f"Val nodules: {len(val_ds)}")
    print(f"Class counts train: {class_counts.tolist()}")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, device, criterion)
        va_loss, va_acc = run_epoch(model, val_loader, None, device, criterion)

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )

        last_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_weights": class_weights.tolist(),
            "args": vars(args),
        }
        torch.save(last_ckpt, os.path.join(run_dir, "last.pt"))

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "class_weights": class_weights.tolist(),
                "args": vars(args),
            }
            torch.save(best_ckpt, os.path.join(run_dir, "best.pt"))

        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)

    summary: Dict[str, object] = {
        "run_id": os.path.basename(run_dir),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_val_acc": best_val_acc,
        "train_nodules": len(train_ds),
        "val_nodules": len(val_ds),
        "class_counts_train": class_counts.tolist(),
        "hyperparameters": vars(args),
    }

    if args.eval_test:
        test_samples = load_split_samples(
            split_dir=test_split,
            mask_dir=args.test_mask_dir,
            patch_size=args.patch_size,
            min_voxels=args.min_voxels,
            require_labels=True,
        )
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
        te_loss, te_acc = run_epoch(model, test_loader, None, device, criterion)
        summary["test_loss"] = te_loss
        summary["test_acc"] = te_acc
        summary["test_nodules"] = len(test_ds)
        print(f"Test | loss={te_loss:.4f} acc={te_acc:.4f}")

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
                    "best_val_acc": best_val_acc,
                    "summary_path": os.path.join(run_dir, "summary.json"),
                    "history_path": os.path.join(run_dir, "history.csv"),
                    "best_checkpoint": os.path.join(run_dir, "best.pt"),
                    "last_checkpoint": os.path.join(run_dir, "last.pt"),
                }
            )
            + "\n"
        )

    print("Training finished.")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Artifacts saved in: {run_dir}")


def predict_main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    base_channels = int(ckpt_args.get("base_channels", 32))
    patch_size = int(ckpt_args.get("patch_size", args.patch_size))

    model = SmallMalignancyCNN(in_channels=3, base_channels=base_channels, num_classes=5).to(device)
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
        "prob_1",
        "prob_2",
        "prob_3",
        "prob_4",
        "prob_5",
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
