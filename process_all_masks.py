"""
Batch mask generation for the LIDC-IDRI dataset.

Iterates over all patients and all nodules in the pylidc database,
generates consensus masks from expert annotations, extracts the
corresponding CT patches, and saves everything as .npy files along
with a CSV of annotation features (malignancy, subtlety, etc.).

Output structure:
    output/
        CT/          — CT volume patches per nodule  (*.npy)
        masks/       — consensus binary masks         (*.npy)
        metadata.csv — one row per nodule with patient_id, nodule index,
                       bounding box, and averaged annotation features

Usage:
    python process_all_masks.py [--output_dir OUTPUT] [--clevel 0.5]
"""

import argparse
import csv
import os
import sys
import warnings

import numpy as np

# Compatibility shims for pylidc with newer numpy / configparser
np.int = int
import configparser
configparser.SafeConfigParser = configparser.ConfigParser

import pylidc as pl
from pylidc.utils import consensus

# Annotation feature columns that pylidc exposes per annotation
FEATURE_NAMES = [
    "subtlety",
    "internalStructure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
    "malignancy",
]


def process_scan(scan, output_ct_dir, output_mask_dir, clevel):
    """Process a single scan: extract all nodules, masks, and features.

    Returns a list of dicts (one per nodule) with metadata.
    """
    patient_id = scan.patient_id
    rows = []

    try:
        nodules = scan.cluster_annotations()
    except Exception as e:
        print(f"  ⚠️  Could not cluster annotations for {patient_id}: {e}")
        return rows

    if len(nodules) == 0:
        return rows

    # Load the CT volume once per patient
    try:
        vol = scan.to_volume()
    except Exception as e:
        print(f"  ⚠️  Could not load volume for {patient_id}: {e}")
        return rows

    for nod_idx, nod in enumerate(nodules):
        try:
            mask, bbox, _ = consensus(nod, clevel=clevel)
        except Exception as e:
            print(f"  ⚠️  Consensus failed for {patient_id} nodule {nod_idx}: {e}")
            continue

        # Extract the CT patch matching the mask bounding box
        ct_patch = vol[bbox[0], bbox[1], bbox[2]]

        if ct_patch.shape != mask.shape:
            print(
                f"  ⚠️  Shape mismatch for {patient_id} nodule {nod_idx}: "
                f"CT {ct_patch.shape} vs mask {mask.shape}. Skipping."
            )
            continue

        # File naming: LIDC-IDRI-0078_nod0.npy
        fname = f"{patient_id}_nod{nod_idx}.npy"
        np.save(os.path.join(output_ct_dir, fname), ct_patch)
        np.save(os.path.join(output_mask_dir, fname), mask.astype(np.uint8))

        # Average annotation features across radiologists (skip None values)
        features = {}
        for feat in FEATURE_NAMES:
            values = [getattr(ann, feat) for ann in nod if getattr(ann, feat) is not None]
            features[feat] = round(np.mean(values), 2) if values else None

        rows.append(
            {
                "patient_id": patient_id,
                "nodule_idx": nod_idx,
                "num_annotations": len(nod),
                "mask_shape": str(mask.shape),
                "bbox_x": f"{bbox[0].start}-{bbox[0].stop}",
                "bbox_y": f"{bbox[1].start}-{bbox[1].stop}",
                "bbox_z": f"{bbox[2].start}-{bbox[2].stop}",
                **features,
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate consensus masks for all LIDC-IDRI nodules."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Root directory for output files (default: output)",
    )
    parser.add_argument(
        "--clevel",
        type=float,
        default=0.5,
        help="Consensus level — fraction of annotators that must agree (default: 0.5)",
    )
    args = parser.parse_args()

    # Create output directories
    output_ct_dir = os.path.join(args.output_dir, "CT")
    output_mask_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(output_ct_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "metadata.csv")

    # Remove previous CSV to avoid duplicates on re-runs
    if os.path.exists(csv_path):
        os.remove(csv_path)
        
    # Query all scans
    scans = pl.query(pl.Scan).all()
    print(f"Found {len(scans)} scans in the database.")

    all_rows = []
    total_nodules = 0

    for i, scan in enumerate(scans):
        print(f"[{i + 1}/{len(scans)}] Processing {scan.patient_id} ...")
        rows = process_scan(scan, output_ct_dir, output_mask_dir, args.clevel)
        all_rows.extend(rows)
        total_nodules += len(rows)

        # Write CSV incrementally so progress is not lost on crash
        if rows:
            file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerows(rows)

    print(f"\nDone. Processed {len(scans)} scans, extracted {total_nodules} nodules.")
    print(f"Output saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
