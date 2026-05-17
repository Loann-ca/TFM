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
import shutil
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


def _save_npy_atomic(path, array):
    """Save .npy atomically to avoid leaving partial files on write failures."""
    tmp_path = f"{path}.tmp.npy"
    try:
        np.save(tmp_path, array)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _ensure_free_space(output_dir, required_bytes, safety_margin_bytes=200 * 1024 * 1024):
    """Check there is enough free space before writing large arrays.

    A safety margin helps avoid mid-write failures due to filesystem metadata,
    temporary files, and concurrent writes.
    """
    free_bytes = shutil.disk_usage(output_dir).free
    min_required = required_bytes + safety_margin_bytes
    if free_bytes < min_required:
        req_mb = required_bytes / (1024 ** 2)
        free_mb = free_bytes / (1024 ** 2)
        min_mb = min_required / (1024 ** 2)
        raise RuntimeError(
            "Insufficient disk space to save CT + mask. "
            f"Need at least {min_mb:.1f} MB (data={req_mb:.1f} MB + safety margin), "
            f"but only {free_mb:.1f} MB free in '{os.path.abspath(output_dir)}'."
        )


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

        # Insertar la máscara del nódulo en su posición dentro del volumen completo.
        # consensus() añade pad=2 por defecto, lo que puede extender el bbox fuera
        # de los límites del volumen si el nódulo está cerca del borde. Se recorta.
        slices_vol = []
        slices_mask = []
        for bbox_sl, vol_size in zip(bbox, vol.shape):
            s0, s1 = bbox_sl.start, bbox_sl.stop
            m0, m1 = 0, s1 - s0
            if s0 < 0:
                m0 = -s0
                s0 = 0
            if s1 > vol_size:
                m1 -= s1 - vol_size
                s1 = vol_size
            slices_vol.append(slice(s0, s1))
            slices_mask.append(slice(m0, m1))
        slices_vol = tuple(slices_vol)
        slices_mask = tuple(slices_mask)

        full_mask[slices_vol] = np.maximum(
            full_mask[slices_vol],
            mask[slices_mask].astype(np.uint8),
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
    ct_to_save = vol.astype(np.float32)
    required_bytes = ct_to_save.nbytes + full_mask.nbytes
    _ensure_free_space(output_ct_dir, required_bytes)

    ct_path = os.path.join(output_ct_dir, f"{patient_id}.npy")
    mask_path = os.path.join(output_mask_dir, f"{patient_id}.npy")
    try:
        _save_npy_atomic(ct_path, ct_to_save)
        _save_npy_atomic(mask_path, full_mask)
    except OSError as e:
        raise RuntimeError(
            f"Failed writing patient {patient_id} arrays to disk: {e}. "
            "Check free space, write permissions, and disk health."
        ) from e

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Genera volúmenes CT completos y máscaras full-size para LIDC-IDRI."
    )
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--clevel", type=float, default=0.5,
                        help="Nivel de consenso (default: 0.5)")
    parser.add_argument("--start_idx", type=int, default=1,
                        help="Índice 1-based del scan donde empezar (default: 1)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="Índice 1-based del scan donde terminar (incluido)")
    parser.add_argument("--resume", action="store_true",
                        help="Reanuda ejecución: no borra metadata.csv y salta pacientes ya guardados")
    args = parser.parse_args()

    output_ct_dir = os.path.join(args.output_dir, "CT")
    output_mask_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(output_ct_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, "metadata.csv")
    # Solo reinicia metadata en ejecución completa desde el inicio y sin modo resume.
    if os.path.exists(csv_path) and not args.resume and args.start_idx == 1 and args.end_idx is None:
        os.remove(csv_path)

    scans = pl.query(pl.Scan).all()
    total_scans = len(scans)
    print(f"Found {total_scans} scans in the database.")

    start_idx = max(1, args.start_idx)
    end_idx = total_scans if args.end_idx is None else min(args.end_idx, total_scans)
    if start_idx > end_idx:
        raise ValueError(f"Invalid range: start_idx ({start_idx}) > end_idx ({end_idx})")

    scans_to_process = scans[start_idx - 1:end_idx]
    print(f"Processing range: [{start_idx}, {end_idx}] ({len(scans_to_process)} scans)")

    if os.path.exists(csv_path) and (args.resume or start_idx > 1):
        print(f"Keeping existing metadata file: {csv_path}")

    all_rows = []
    total_nodules = 0

    for i, scan in enumerate(scans_to_process, start=start_idx):
        patient_id = scan.patient_id
        ct_path = os.path.join(output_ct_dir, f"{patient_id}.npy")
        mask_path = os.path.join(output_mask_dir, f"{patient_id}.npy")

        if args.resume and os.path.exists(ct_path) and os.path.exists(mask_path):
            print(f"[{i}/{total_scans}] Skipping {patient_id} (already saved)")
            continue

        print(f"[{i}/{total_scans}] Processing {patient_id} ...")
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

    print(f"\nDone. Processed {len(scans_to_process)} scans in selected range, {total_nodules} nodules.")
    print(f"Output: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
