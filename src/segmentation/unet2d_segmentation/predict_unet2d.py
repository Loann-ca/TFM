"""
Inference script for 2D U-Net segmentation using a trained run checkpoint.

It loads best.pt from a run directory, applies the same HU windowing + normalization
used in preprocessing, predicts slice-by-slice, and saves:
- Binary mask volume (H, W, D) as uint8
- Optional probability volume (H, W, D) as float32

Usage example:
    python src/segmentation/unet2d_segmentation/predict_unet2d.py \
        --input_ct output/CT/LIDC-IDRI-0078.npy \
        --run_dir checkpoints/unet2d_baseline/run_20260615_153012 \
        --output_mask output/predictions2d/LIDC-IDRI-0078_mask.npy \
        --save_probs
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from train_unet2d_baseline import UNet2D


def apply_windowing(ct: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    ct = np.clip(ct, hu_min, hu_max)
    ct = (ct - hu_min) / (hu_max - hu_min)
    return ct.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for 2D U-Net lung nodule segmentation")

    parser.add_argument("--input_ct", type=str, required=True, help="Path to input CT volume .npy (H, W, D)")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing best.pt")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path. Defaults to run_dir/best.pt")

    parser.add_argument("--output_mask", type=str, default=None, help="Output path for predicted mask .npy (default: output/predictions2d/<patient>.npy)")
    parser.add_argument("--save_probs", action="store_true", help="Also save probability volume")
    parser.add_argument("--output_probs", type=str, default=None, help="Optional output path for probabilities .npy")

    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary mask")
    parser.add_argument("--hu_min", type=float, default=-1000.0, help="HU window minimum")
    parser.add_argument("--hu_max", type=float, default=600.0, help="HU window maximum")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint_path = args.checkpoint or os.path.join(args.run_dir, "best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not os.path.exists(args.input_ct):
        raise FileNotFoundError(f"Input CT not found: {args.input_ct}")

    if args.output_mask is None:
        patient_name = os.path.splitext(os.path.basename(args.input_ct))[0]
        args.output_mask = os.path.join("output", "predictions2d", f"{patient_name}.npy")

    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    base_channels = int(ckpt_args.get("base_channels", 32))

    model = UNet2D(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ct = np.load(args.input_ct).astype(np.float32)
    if ct.ndim != 3:
        raise ValueError(f"Expected 3D volume (H, W, D). Got shape: {ct.shape}")

    ct = apply_windowing(ct, hu_min=args.hu_min, hu_max=args.hu_max)

    h, w, d = ct.shape
    prob_volume = np.zeros((h, w, d), dtype=np.float32)

    with torch.no_grad():
        for z in range(d):
            slice_2d = ct[:, :, z]
            x = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            prob_volume[:, :, z] = probs.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    mask = (prob_volume >= args.threshold).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_mask)), exist_ok=True)
    np.save(args.output_mask, mask)

    if args.save_probs:
        output_probs = args.output_probs
        if output_probs is None:
            base, _ = os.path.splitext(args.output_mask)
            output_probs = f"{base}_probs.npy"
        os.makedirs(os.path.dirname(os.path.abspath(output_probs)), exist_ok=True)
        np.save(output_probs, prob_volume)
        print(f"Saved probabilities: {output_probs}")

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input shape: {ct.shape}")
    print(f"Saved mask: {args.output_mask}")


if __name__ == "__main__":
    main()
