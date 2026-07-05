"""
Inference script for 3D U-Net segmentation using a trained run checkpoint.

It loads best.pt from a run directory, applies the same HU windowing + normalization,
runs full-volume 3D inference, and saves:
- Binary mask volume (H, W, D) as uint8
- Optional probability volume (H, W, D) as float32

Usage example:
    python src/segmentation/unet3d_segmentation/predict_unet3d.py \
        --input_ct output/CT/LIDC-IDRI-0078.npy \
        --run_dir checkpoints/unet3d_baseline/run_20260615_153012 \
        --output_mask output/predictions3d/LIDC-IDRI-0078_mask.npy \
        --save_probs
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from train_unet3d_baseline import UNet3D


def apply_windowing(ct: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    ct = np.clip(ct, hu_min, hu_max)
    ct = (ct - hu_min) / (hu_max - hu_min)
    return ct.astype(np.float32)


def pad_dhw_to_multiple(volume_dhw: np.ndarray, multiple: int = 8) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Pad (D, H, W) volume with zeros so each dim is divisible by `multiple`."""
    d, h, w = volume_dhw.shape
    pd = (multiple - (d % multiple)) % multiple
    ph = (multiple - (h % multiple)) % multiple
    pw = (multiple - (w % multiple)) % multiple
    padded = np.pad(volume_dhw, ((0, pd), (0, ph), (0, pw)), mode="constant", constant_values=0.0)
    return padded, (d, h, w)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for 3D U-Net lung nodule segmentation")

    parser.add_argument("--input_ct", type=str, required=True, help="Path to input CT volume .npy (H, W, D)")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory containing best.pt")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path. Defaults to run_dir/best.pt")

    parser.add_argument("--output_mask", type=str, default=None, help="Output path for predicted mask .npy (default: output/predictions3d/<patient>.npy)")
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


def sliding_window_inference(
    volume_dhw: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    patch_size: int = 64,
    overlap: int = 32,
) -> np.ndarray:
    """
    Sliding window inference on a 3D volume using patches.
    
    Args:
        volume_dhw: (D, H, W) volume
        model: UNet3D model in eval mode
        device: torch device
        patch_size: Size of each patch (64x64x64)
        overlap: Overlap between patches (32 = 50% overlap)
    
    Returns:
        probs: (D, H, W) probability volume
    """
    d, h, w = volume_dhw.shape
    stride = patch_size - overlap
    
    # Accumulate probabilities and counts for averaging overlaps
    prob_accum = np.zeros((d, h, w), dtype=np.float32)
    count_accum = np.zeros((d, h, w), dtype=np.float32)
    
    # Iterate over all patch positions
    for start_d in range(0, d, stride):
        end_d = min(start_d + patch_size, d)
        if end_d - start_d < patch_size:
            start_d = max(0, d - patch_size)
            end_d = d
        
        for start_h in range(0, h, stride):
            end_h = min(start_h + patch_size, h)
            if end_h - start_h < patch_size:
                start_h = max(0, h - patch_size)
                end_h = h
            
            for start_w in range(0, w, stride):
                end_w = min(start_w + patch_size, w)
                if end_w - start_w < patch_size:
                    start_w = max(0, w - patch_size)
                    end_w = w
                
                # Extract patch
                patch = volume_dhw[start_d:end_d, start_h:end_h, start_w:end_w]
                
                # Pad to patch_size if needed
                pad_d = patch_size - (end_d - start_d)
                pad_h = patch_size - (end_h - start_h)
                pad_w = patch_size - (end_w - start_w)
                if pad_d > 0 or pad_h > 0 or pad_w > 0:
                    patch = np.pad(patch, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
                
                # Inference on patch
                with torch.no_grad():
                    x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                    logits = model(x)
                    patch_probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
                
                # Accumulate without padding
                patch_probs = patch_probs[:end_d - start_d, :end_h - start_h, :end_w - start_w]
                prob_accum[start_d:end_d, start_h:end_h, start_w:end_w] += patch_probs
                count_accum[start_d:end_d, start_h:end_h, start_w:end_w] += 1.0
    
    # Average overlapping regions
    probs = prob_accum / (count_accum + 1e-8)
    return probs


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
        args.output_mask = os.path.join("output", "predictions3d", f"{patient_name}.npy")

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    base_channels = int(ckpt_args.get("base_channels", 16))

    model = UNet3D(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ct = np.load(args.input_ct).astype(np.float32)
    if ct.ndim != 3:
        raise ValueError(f"Expected 3D volume (H, W, D). Got shape: {ct.shape}")

    ct = apply_windowing(ct, hu_min=args.hu_min, hu_max=args.hu_max)

    # Training uses (D, H, W) for model input, while files are stored as (H, W, D).
    ct_dhw = np.transpose(ct, (2, 0, 1)).astype(np.float32)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input shape: {ct.shape} (H,W,D) -> {ct_dhw.shape} (D,H,W)")
    print("Running sliding window inference with 64³ patches and 50% overlap...")
    
    probs = sliding_window_inference(ct_dhw, model, device, patch_size=64, overlap=32)

    prob_volume = np.transpose(probs, (1, 2, 0))  # Back to (H, W, D)
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

    print(f"Saved mask: {args.output_mask}")


if __name__ == "__main__":
    main()
