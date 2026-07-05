from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List

import csv

import numpy as np


@dataclass
class CaseResult:
    patient_id: str
    has_probs: bool
    best_threshold: float
    best_dice: float
    dice_at_05: float
    precision_at_best: float
    recall_at_best: float
    pred_vox_at_best: int
    gt_vox: int


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    inter = np.logical_and(pred, gt).sum(dtype=np.int64)
    denom = pred.sum(dtype=np.int64) + gt.sum(dtype=np.int64)
    return float((2.0 * inter) / (denom + eps))


def precision_recall(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    tp = np.logical_and(pred, gt).sum(dtype=np.int64)
    fp = np.logical_and(pred, np.logical_not(gt)).sum(dtype=np.int64)
    fn = np.logical_and(np.logical_not(pred), gt).sum(dtype=np.int64)
    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    return precision, recall


def evaluate_case(pred_dir: str, gt_dir: str, patient_id: str, thresholds: np.ndarray) -> CaseResult | None:
    gt_path = os.path.join(gt_dir, f"{patient_id}.npy")
    pred_mask_path = os.path.join(pred_dir, f"{patient_id}.npy")
    probs_path = os.path.join(pred_dir, f"{patient_id}_probs.npy")

    if not os.path.exists(gt_path):
        return None
    if not os.path.exists(pred_mask_path) and not os.path.exists(probs_path):
        return None

    gt = np.load(gt_path)
    gt_bin = gt > 0
    gt_vox = int(gt_bin.sum())

    if os.path.exists(probs_path):
        probs = np.load(probs_path).astype(np.float32)
        if probs.shape != gt.shape:
            return None

        best_dice = -1.0
        best_t = 0.5
        best_pred = None
        for t in thresholds:
            pred_bin = probs >= t
            d = dice_score(pred_bin, gt_bin)
            if d > best_dice:
                best_dice = d
                best_t = float(t)
                best_pred = pred_bin

        assert best_pred is not None
        dice05 = dice_score(probs >= 0.5, gt_bin)
        precision, recall = precision_recall(best_pred, gt_bin)

        return CaseResult(
            patient_id=patient_id,
            has_probs=True,
            best_threshold=best_t,
            best_dice=float(best_dice),
            dice_at_05=float(dice05),
            precision_at_best=float(precision),
            recall_at_best=float(recall),
            pred_vox_at_best=int(best_pred.sum()),
            gt_vox=gt_vox,
        )

    pred_mask = np.load(pred_mask_path)
    if pred_mask.shape != gt.shape:
        return None
    pred_bin = pred_mask > 0
    d = dice_score(pred_bin, gt_bin)
    precision, recall = precision_recall(pred_bin, gt_bin)

    return CaseResult(
        patient_id=patient_id,
        has_probs=False,
        best_threshold=0.5,
        best_dice=float(d),
        dice_at_05=float(d),
        precision_at_best=float(precision),
        recall_at_best=float(recall),
        pred_vox_at_best=int(pred_bin.sum()),
        gt_vox=gt_vox,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank CT cases by prediction quality against GT masks")
    parser.add_argument("--pred_dir", type=str, default="output/predictions3d", help="Directory with <patient>.npy and optional <patient>_probs.npy")
    parser.add_argument("--gt_dir", type=str, default="output/preprocessed/test/masks", help="Directory with GT masks <patient>.npy")
    parser.add_argument("--top_k", type=int, default=10, help="How many top cases to print")
    parser.add_argument("--th_min", type=float, default=0.05, help="Minimum threshold to sweep")
    parser.add_argument("--th_max", type=float, default=0.95, help="Maximum threshold to sweep")
    parser.add_argument("--th_steps", type=int, default=19, help="Number of threshold values to test")
    parser.add_argument("--csv_out", type=str, default=None, help="Optional path to save the full ranking as CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.pred_dir):
        raise FileNotFoundError(f"Prediction dir not found: {args.pred_dir}")
    if not os.path.isdir(args.gt_dir):
        raise FileNotFoundError(f"GT dir not found: {args.gt_dir}")

    thresholds = np.linspace(args.th_min, args.th_max, args.th_steps, dtype=np.float32)

    patient_ids = set()
    for name in os.listdir(args.pred_dir):
        if not name.endswith(".npy"):
            continue
        if name.endswith("_probs.npy"):
            patient_ids.add(name[: -len("_probs.npy")])
        else:
            patient_ids.add(name[: -len(".npy")])

    results: List[CaseResult] = []
    for pid in sorted(patient_ids):
        out = evaluate_case(args.pred_dir, args.gt_dir, pid, thresholds)
        if out is not None:
            results.append(out)

    if not results:
        print("No comparable cases found.")
        return

    results.sort(key=lambda x: x.best_dice, reverse=True)

    print("patient_id,source,best_dice,best_threshold,dice_at_0.5,precision_best,recall_best,pred_vox_best,gt_vox")
    for r in results[: args.top_k]:
        source = "probs" if r.has_probs else "mask_only"
        print(
            f"{r.patient_id},{source},{r.best_dice:.5f},{r.best_threshold:.2f},"
            f"{r.dice_at_05:.5f},{r.precision_at_best:.5f},{r.recall_at_best:.5f},"
            f"{r.pred_vox_at_best},{r.gt_vox}"
        )

    if args.csv_out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv_out)), exist_ok=True)
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "patient_id",
                "source",
                "best_dice",
                "best_threshold",
                "dice_at_0.5",
                "precision_best",
                "recall_best",
                "pred_vox_best",
                "gt_vox",
            ])
            for r in results:
                writer.writerow([
                    r.patient_id,
                    "probs" if r.has_probs else "mask_only",
                    f"{r.best_dice:.6f}",
                    f"{r.best_threshold:.2f}",
                    f"{r.dice_at_05:.6f}",
                    f"{r.precision_at_best:.6f}",
                    f"{r.recall_at_best:.6f}",
                    r.pred_vox_at_best,
                    r.gt_vox,
                ])
        print(f"Saved CSV ranking to: {args.csv_out}")


if __name__ == "__main__":
    main()
