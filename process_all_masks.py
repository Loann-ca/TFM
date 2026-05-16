"""
Generación de volúmenes CT completos y máscaras full-size para LIDC-IDRI.

Para cada paciente guarda:
  - El TAC completo (512x512xN) como .npy
  - Una máscara del mismo tamaño con TODOS los nódulos marcados (resto a 0)
  - Metadata CSV con features promediadas por nódulo

Output:
    output/
        CT/{patient_id}.npy        — volumen CT completo (float32)
        masks/{patient_id}.npy     — máscara binaria full-size (uint8)
        metadata.csv               — una fila por nódulo con features

Uso:
    python process_all_masks.py --output_dir output --clevel 0.5
"""

import argparse
import csv
import os
import warnings

import numpy as np

# Compatibility shims for pylidc
np.int = int
import configparser
configparser.SafeConfigParser = configparser.ConfigParser

import pylidc as pl
from pylidc.utils import consensus

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
    """Procesa un paciente: guarda CT completo y máscara con todos los nódulos.

    Returns:
        Lista de dicts con metadata por nódulo.
    """
    patient_id = scan.patient_id
    rows = []

    try:
        nodules = scan.cluster_annotations()
    except Exception as e:
        print(f"  Could not cluster annotations for {patient_id}: {e}")
        return rows

    if len(nodules) == 0:
        return rows

    # Cargar volumen CT completo
    try:
        vol = scan.to_volume()
    except Exception as e:
        print(f"  Could not load volume for {patient_id}: {e}")
        return rows

    # Crear máscara vacía del tamaño del volumen completo
    full_mask = np.zeros(vol.shape, dtype=np.uint8)

    for nod_idx, nod in enumerate(nodules):
        try:
            mask, bbox, _ = consensus(nod, clevel=clevel)
        except Exception as e:
            print(f"  Consensus failed for {patient_id} nodule {nod_idx}: {e}")
            continue

        # Recortar bbox para que no se salga del volumen
        slices = []
        mask_slices = []
        skip = False
        for dim in range(3):
            b = bbox[dim]
            start = max(b.start, 0)
            stop = min(b.stop, vol.shape[dim])
            if stop <= start:
                print(f"  BBox fuera de rango para {patient_id} nodule {nod_idx} dim {dim}. Skipping.")
                skip = True
                break
            # Ajustar la máscara si se recortó el bbox
            mask_start = start - b.start
            mask_stop = mask.shape[dim] - (b.stop - stop)
            slices.append(slice(start, stop))
            mask_slices.append(slice(mask_start, mask_stop))

        if skip:
            continue

        mask_cropped = mask[mask_slices[0], mask_slices[1], mask_slices[2]].astype(np.uint8)

        # Insertar la máscara del nódulo en su posición dentro del volumen completo
        full_mask[slices[0], slices[1], slices[2]] = np.maximum(
            full_mask[slices[0], slices[1], slices[2]],
            mask_cropped,
        )

        # Features promediadas por nódulo
        features = {}
        for feat in FEATURE_NAMES:
            values = [getattr(ann, feat) for ann in nod if getattr(ann, feat) is not None]
            features[feat] = round(np.mean(values), 2) if values else None

        rows.append({
            "patient_id": patient_id,
            "nodule_idx": nod_idx,
            "num_annotations": len(nod),
            "vol_shape": str(vol.shape),
            "bbox_x": f"{bbox[0].start}-{bbox[0].stop}",
            "bbox_y": f"{bbox[1].start}-{bbox[1].stop}",
            "bbox_z": f"{bbox[2].start}-{bbox[2].stop}",
            **features,
        })

    # Guardar CT completo y máscara completa (un fichero por paciente)
    np.save(os.path.join(output_ct_dir, f"{patient_id}.npy"), vol.astype(np.float32))
    np.save(os.path.join(output_mask_dir, f"{patient_id}.npy"), full_mask)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Genera volúmenes CT completos y máscaras full-size para LIDC-IDRI."
    )
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--clevel", type=float, default=0.5,
                        help="Nivel de consenso (default: 0.5)")
    args = parser.parse_args()

    output_ct_dir = os.path.join(args.output_dir, "CT")
    output_mask_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(output_ct_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "metadata.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    scans = pl.query(pl.Scan).all()
    print(f"Found {len(scans)} scans in the database.")

    all_rows = []
    total_nodules = 0

    for i, scan in enumerate(scans):
        print(f"[{i + 1}/{len(scans)}] Processing {scan.patient_id} ...")
        rows = process_scan(scan, output_ct_dir, output_mask_dir, args.clevel)
        all_rows.extend(rows)
        total_nodules += len(rows)

        if rows:
            file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerows(rows)

    print(f"\nDone. {len(scans)} scans, {total_nodules} nodules.")
    print(f"Output: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
