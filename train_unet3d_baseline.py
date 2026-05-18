"""
Baseline training script for 3D U-Net on preprocessed LIDC-IDRI nodules.

Expected data layout:
    output/
      preprocessed/
        train/{CT,masks,metadata.csv}
        val/{CT,masks,metadata.csv}
        test/{CT,masks,metadata.csv}

Usage:
    python train_unet3d_baseline.py --output_dir output --epochs 50 --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Descriptor mínimo de una muestra de entrenamiento (patch 3D).
@dataclass
class PatchItem:
    ct_path: str
    mask_path: str
    patient_id: str
    center: Tuple[int, int, int]


def _parse_bbox(bbox_str: str) -> Tuple[int, int]:
    """Parse bbox range strings like '12-34' or '-2-18'."""
    m = re.fullmatch(r"\s*(-?\d+)\s*-\s*(-?\d+)\s*", str(bbox_str))
    if m is None:
        raise ValueError(f"Invalid bbox range format: {bbox_str}")
    return int(m.group(1)), int(m.group(2))


def _bbox_center(bbox_x: str, bbox_y: str, bbox_z: str) -> Tuple[int, int, int]:
    # Convierte los rangos de bbox a su centro en coordenadas del volumen global.
    x0, x1 = _parse_bbox(bbox_x)
    y0, y1 = _parse_bbox(bbox_y)
    z0, z1 = _parse_bbox(bbox_z)
    return ((x0 + x1) // 2, (y0 + y1) // 2, (z0 + z1) // 2)


def set_seed(seed: int) -> None:
    # Fija semillas para reproducibilidad en Python, NumPy y PyTorch.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_augment_3d(ct: np.ndarray, mask: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """Apply simple geometry-preserving augmentation to a CT/mask pair."""
    for axis in range(3):
        if rng.rand() < 0.5:
            ct = np.flip(ct, axis=axis)
            mask = np.flip(mask, axis=axis)

    k = rng.randint(0, 4)
    if k > 0:
        # Rotation in x/y plane keeps depth unchanged.
        ct = np.rot90(ct, k=k, axes=(0, 1))
        mask = np.rot90(mask, k=k, axes=(0, 1))

    return ct.copy(), mask.copy()


class FullVolumeNoduleDataset(Dataset):
    """Dataset que carga volúmenes CT completos por paciente y extrae patches al vuelo.

    Por cada paciente genera ``patches_per_patient`` muestras:
      - La mitad centradas en un nódulo (positive mining).
      - La mitad en posiciones aleatorias (diversidad de fondo).

    Todos los patches tienen tamaño fijo ``patch_size³``.
    """

    def __init__(
        self,
        split_dir: str,
        patch_size: int = 64,
        patches_per_patient: int = 8,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        # Validación básica de argumentos para evitar errores silenciosos.
        if patch_size <= 0 or patch_size % 2 != 0:
            raise ValueError("patch_size must be a positive even integer.")
        if patches_per_patient <= 0:
            raise ValueError("patches_per_patient must be > 0.")

        self.patch_size = patch_size
        self.augment = augment
        self.rng = np.random.RandomState(seed)

        meta_path = os.path.join(split_dir, "metadata.csv")
        ct_dir = os.path.join(split_dir, "CT")
        mask_dir = os.path.join(split_dir, "masks")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"metadata.csv not found: {meta_path}")
        if not os.path.isdir(ct_dir):
            raise FileNotFoundError(f"CT directory not found: {ct_dir}")
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"masks directory not found: {mask_dir}")

        meta = pd.read_csv(meta_path)
        self.items: List[PatchItem] = []
        half = patch_size // 2
        rng = np.random.RandomState(seed)

        # Se agrupa por paciente para cargar CT/mask full-size una sola vez por ID.
        for patient_id, group in meta.groupby("patient_id"):
            ct_path = os.path.join(ct_dir, f"{patient_id}.npy")
            mask_path = os.path.join(mask_dir, f"{patient_id}.npy")
            if not (os.path.exists(ct_path) and os.path.exists(mask_path)):
                continue

            # Peek at shape without loading the full array.
            vol = np.load(ct_path, mmap_mode="r")
            sh = vol.shape  # (H, W, D)

            max_x = sh[0] - half - 1
            max_y = sh[1] - half - 1
            max_z = sh[2] - half - 1
            if max_x < half or max_y < half or max_z < half:
                continue  # Volume too small for one patch.

            # Centros positivos: uno por nódulo a partir de su bbox.
            nodule_centers: List[Tuple[int, int, int]] = []
            for _, row in group.iterrows():
                cx, cy, cz = _bbox_center(
                    str(row.bbox_x), str(row.bbox_y), str(row.bbox_z)
                )
                # Clip para garantizar que el patch completo cae dentro del volumen.
                cx = int(np.clip(cx, half, max_x))
                cy = int(np.clip(cy, half, max_y))
                cz = int(np.clip(cz, half, max_z))
                nodule_centers.append((cx, cy, cz))

            # Balance simple: mitad patches positivos, mitad de fondo aleatorio.
            n_nodule = max(1, patches_per_patient // 2)
            n_random = patches_per_patient - n_nodule

            for i in range(n_nodule):
                center = nodule_centers[i % len(nodule_centers)]
                self.items.append(PatchItem(ct_path, mask_path, str(patient_id), center))

            for _ in range(n_random):
                cx = int(rng.randint(half, max_x + 1))
                cy = int(rng.randint(half, max_y + 1))
                cz = int(rng.randint(half, max_z + 1))
                self.items.append(PatchItem(ct_path, mask_path, str(patient_id), (cx, cy, cz)))

        if len(self.items) == 0:
            raise RuntimeError(f"No valid samples found in split: {split_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        half = self.patch_size // 2
        cx, cy, cz = item.center

        # Memory-map so only the required pages are read from disk.
        ct_full = np.load(item.ct_path, mmap_mode="r")
        mask_full = np.load(item.mask_path, mmap_mode="r")

        ct = np.array(
            ct_full[cx - half:cx + half, cy - half:cy + half, cz - half:cz + half],
            dtype=np.float32,
        )
        mask = np.array(
            mask_full[cx - half:cx + half, cy - half:cy + half, cz - half:cz + half],
            dtype=np.float32,
        )
        # Asegura máscara binaria aunque el origen tenga otro dtype/escala.
        mask = (mask > 0.5).astype(np.float32)

        if self.augment:
            ct, mask = maybe_augment_3d(ct, mask, self.rng)

        # (H, W, D) → (D, H, W) → add channel dim.
        ct = np.transpose(ct, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        return {
            "image": torch.from_numpy(ct).unsqueeze(0),   # [1, D, H, W]
            "mask": torch.from_numpy(mask).unsqueeze(0),  # [1, D, H, W]
        }


class DoubleConv3D(nn.Module):
    # Bloque base de U-Net: Conv3d + BN + ReLU repetido 2 veces.
    # Mantiene la resolución espacial (padding=1) y refina representación local.
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()

        # =========================
        # Encoder (contracción)
        # =========================
        # Cada nivel aplica DoubleConv para extraer características más ricas.
        # Tras cada nivel se aplica MaxPool3d(2): divide D/H/W entre 2.

        # Nivel 1: entrada (CT con 1 canal) -> base_channels.
        self.enc1 = DoubleConv3D(in_channels, base_channels)
        # Nivel 2: duplica canales para capturar patrones más complejos.
        self.enc2 = DoubleConv3D(base_channels, base_channels * 2)
        # Nivel 3: más contexto semántico con más canales.
        self.enc3 = DoubleConv3D(base_channels * 2, base_channels * 4)
        # Bottleneck: representación más comprimida y de alto nivel.
        self.bottleneck = DoubleConv3D(base_channels * 4, base_channels * 8)

        # Operación de downsampling entre niveles del encoder.
        self.pool = nn.MaxPool3d(kernel_size=2)

        # =========================
        # Decoder (expansión)
        # =========================
        # ConvTranspose3d duplica resolución y reduce canales.
        # Luego se concatena con la feature map homóloga del encoder (skip connection)
        # para recuperar detalle espacial fino perdido en el pooling.

        # Subida 1: bottleneck -> resolución de e3.
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        # Tras concatenar [up3, e3], canales = 4C + 4C = 8C.
        self.dec3 = DoubleConv3D(base_channels * 8, base_channels * 4)

        # Subida 2: resolución de e2.
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        # Tras concatenar [up2, e2], canales = 2C + 2C = 4C.
        self.dec2 = DoubleConv3D(base_channels * 4, base_channels * 2)

        # Subida 3: resolución de e1.
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        # Tras concatenar [up1, e1], canales = C + C = 2C.
        self.dec1 = DoubleConv3D(base_channels * 2, base_channels)

        # Capa de salida 1x1x1: proyecta a 1 canal de logits (segmentación binaria).
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # -------------------------
        # 1) Ruta de bajada (encoder)
        # -------------------------
        # e1 conserva más detalle espacial (alta resolución).
        e1 = self.enc1(x)
        # pool(e1) reduce resolución y permite ampliar campo receptivo.
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        # b contiene la representación más abstracta del volumen.
        b = self.bottleneck(self.pool(e3))

        # -------------------------
        # 2) Ruta de subida (decoder)
        # -------------------------
        # Up-conv recupera resolución; concat con skip e3 para recuperar detalle.
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        # Repite el mismo patrón en el siguiente nivel.
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        # Último nivel, cerca de la resolución original de entrada.
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # -------------------------
        # 3) Proyección final
        # -------------------------
        # Devuelve logits (sin sigmoid). La sigmoid se aplica en la loss/métricas.
        return self.out_conv(d1)


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Dice sobre probabilidades; se combina con BCE para entrenamiento estable.
    probs = torch.sigmoid(logits)
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    # Métrica Dice en binario (umbral 0.5) para reportar calidad de segmentación.
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds = preds.contiguous().view(preds.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    inter = (preds * targets).sum(dim=1)
    denom = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return float(dice.mean().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    bce_weight: float,
    dice_weight: float,
) -> Tuple[float, float]:
    # Si hay optimizador => train; si no => evaluación sin gradientes.
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    bce = nn.BCEWithLogitsLoss()

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)

            logits = model(x)
            loss_bce = bce(logits, y)
            loss_dice = dice_loss_from_logits(logits, y)
            # Loss total ponderada.
            loss = bce_weight * loss_bce + dice_weight * loss_dice

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_dice = dice_score_from_logits(logits, y)
            total_loss += float(loss.item())
            total_dice += batch_dice
            n_batches += 1

    mean_loss = total_loss / max(n_batches, 1)
    mean_dice = total_dice / max(n_batches, 1)
    return mean_loss, mean_dice


def evaluate_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    bce_weight: float,
    dice_weight: float,
) -> Dict[str, float]:
    # Reusa run_epoch en modo evaluación para el split de test.
    test_loss, test_dice = run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        device=device,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
    )
    return {
        "test_loss": test_loss,
        "test_dice": test_dice,
    }


def parse_args() -> argparse.Namespace:
    # Hiperparámetros y rutas configurables por CLI.
    parser = argparse.ArgumentParser(description="Baseline training for 3D U-Net lung nodule segmentation")

    parser.add_argument("--output_dir", type=str, default="output", help="Root output directory")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--patch_size", type=int, default=64, help="Patch side length in voxels (default: 64)")
    parser.add_argument("--patches_per_patient", type=int, default=8, help="Patches generated per patient per epoch (default: 8)")
    parser.add_argument("--base_channels", type=int, default=16, help="U-Net base channels")
    parser.add_argument("--bce_weight", type=float, default=0.5, help="Weight for BCE loss")
    parser.add_argument("--dice_weight", type=float, default=0.5, help="Weight for Dice loss")

    parser.add_argument("--save_dir", type=str, default="checkpoints/unet3d_baseline", help="Directory for checkpoints and logs")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint in save_dir")
    parser.add_argument("--resume_path", type=str, default=None, help="Optional checkpoint path to resume from")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    parser.add_argument("--eval_test", action="store_true", help="Evaluate on test split after training")

    return parser.parse_args()


def main() -> None:
    # 1) Configuración general.
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    preprocessed_root = os.path.join(args.output_dir, "preprocessed")
    train_dir = os.path.join(preprocessed_root, "train")
    val_dir = os.path.join(preprocessed_root, "val")
    test_dir = os.path.join(preprocessed_root, "test")

    os.makedirs(args.save_dir, exist_ok=True)

    # 2) Dataset y DataLoader para train/val.
    train_ds = FullVolumeNoduleDataset(
        train_dir,
        patch_size=args.patch_size,
        patches_per_patient=args.patches_per_patient,
        augment=True,
        seed=args.seed,
    )
    val_ds = FullVolumeNoduleDataset(
        val_dir,
        patch_size=args.patch_size,
        patches_per_patient=args.patches_per_patient,
        augment=False,
        seed=args.seed,
    )

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

    # 3) Modelo y optimizador.
    model = UNet3D(in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, float]] = []
    best_val_dice = -1.0
    best_epoch = -1
    start_epoch = 1

    # Reanudación opcional desde checkpoint y restauración de métricas previas.
    history_path = os.path.join(args.save_dir, "history.csv")
    if os.path.exists(history_path):
        prev_hist_df = pd.read_csv(history_path)
        if not prev_hist_df.empty:
            history = prev_hist_df.to_dict(orient="records")

    resume_path = args.resume_path
    if args.resume and resume_path is None:
        resume_path = os.path.join(args.save_dir, "last.pt")

    if resume_path is not None:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_dice = float(ckpt.get("best_val_dice", best_val_dice))

        if len(history) > 0:
            best_idx = max(range(len(history)), key=lambda i: float(history[i].get("val_dice", -1.0)))
            best_epoch = int(history[best_idx].get("epoch", -1))

        print(f"Resumed from: {resume_path}")
        print(f"Restarting at epoch {start_epoch}/{args.epochs}")

    print(f"Device: {device}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    # 4) Bucle principal de entrenamiento.
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_dice = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
        )

        val_loss, val_dice = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_dice": val_dice,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
        )

        # Se actualiza best.pt solo si mejora Dice de validación.
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            checkpoint_best = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "args": vars(args),
            }
            torch.save(checkpoint_best, os.path.join(args.save_dir, "best.pt"))

        checkpoint_last = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_dice": best_val_dice,
            "args": vars(args),
        }
        # Se guarda siempre el último estado para poder reanudar/inspeccionar.
        torch.save(checkpoint_last, os.path.join(args.save_dir, "last.pt"))

        # Persistencia incremental para no perder el histórico en cortes de energía.
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(history_path, index=False)

    # 5) Resumen final y evaluación opcional en test.
    summary = {
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
    }

    if args.eval_test:
        test_ds = FullVolumeNoduleDataset(
            test_dir,
            patch_size=args.patch_size,
            patches_per_patient=args.patches_per_patient,
            augment=False,
            seed=args.seed,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        best_ckpt_path = os.path.join(args.save_dir, "best.pt")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        test_metrics = evaluate_test(
            model=model,
            test_loader=test_loader,
            device=device,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
        )
        summary.update(test_metrics)
        print(f"Test | loss={test_metrics['test_loss']:.4f} dice={test_metrics['test_dice']:.4f}")

    with open(os.path.join(args.save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training finished.")
    print(f"Best val dice: {best_val_dice:.4f} (epoch {best_epoch})")
    print(f"Artifacts saved in: {args.save_dir}")


if __name__ == "__main__":
    main()
