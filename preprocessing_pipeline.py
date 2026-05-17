"""
Pipeline de preprocesamiento para U-Net 3D (detección + segmentación).

Trabaja con volúmenes CT completos por paciente (no patches).
Aplica windowing y normalización a cada volumen, divide por paciente
en train/val/test, y guarda los volúmenes preprocesados.

La extracción de patches se hace al vuelo en el DataLoader (dataset.py).

Pasos:
  1. Split train/val/test por paciente (70/15/15)
  2. Windowing HU [-1000, 600] + normalización a [0, 1]
  3. Guardado de volúmenes preprocesados por split

Uso:
    python preprocessing_pipeline.py --output_dir output
"""

import argparse
import os
import shutil
import warnings
import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ──────────────────────────────────────────────
# Funciones de preprocesamiento
# ──────────────────────────────────────────────

def apply_windowing(ct, hu_min=-1000, hu_max=600):
    """Aplica windowing HU y normaliza a [0, 1].

    Recorta intensidades fuera del rango. Usamos [-1000, 600]
    para preservar información de calcificaciones.
    """
    ct = np.clip(ct, hu_min, hu_max)
    ct = (ct - hu_min) / (hu_max - hu_min)
    return ct.astype(np.float32)


def _check_free_space(required_bytes, path_for_disk):
    """Raises OSError if there is not enough free disk space."""
    free_bytes = shutil.disk_usage(path_for_disk).free
    if free_bytes < required_bytes:
        req_mb = required_bytes / (1024 ** 2)
        free_mb = free_bytes / (1024 ** 2)
        raise OSError(
            f"Insufficient disk space. Required ~{req_mb:.1f} MB, available ~{free_mb:.1f} MB "
            f"on disk containing: {path_for_disk}"
        )


def _safe_save_npy(file_path, arr):
    """Atomically saves an array as .npy to avoid partial/corrupt files."""
    out_dir = os.path.dirname(file_path)
    os.makedirs(out_dir, exist_ok=True)

    # Roughly estimate required bytes: data bytes + .npy header overhead.
    required_bytes = int(arr.nbytes + 1024 * 1024)
    _check_free_space(required_bytes, out_dir)

    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".__tmp_", suffix=".npy", dir=out_dir)
    os.close(tmp_fd)

    try:
        np.save(tmp_path, arr)
        os.replace(tmp_path, file_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


# ──────────────────────────────────────────────
# Split por paciente
# ──────────────────────────────────────────────

def split_by_patient(meta):
    """Divide en train/val/test (70/15/15) agrupando por paciente.

    Usa patient_id como grupo para que todos los nódulos de un
    paciente vayan al mismo split (evita data leakage).
    """
    # Obtener lista única de pacientes
    patients = meta.drop_duplicates(subset=["patient_id"])[["patient_id"]]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(splitter.split(patients, groups=patients["patient_id"]))

    train_patients = set(patients.iloc[train_idx].patient_id)
    temp_patients = patients.iloc[temp_idx]

    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(splitter2.split(temp_patients, groups=temp_patients["patient_id"]))

    val_patients = set(temp_patients.iloc[val_idx].patient_id)
    test_patients = set(temp_patients.iloc[test_idx].patient_id)

    # Verificar que no hay pacientes compartidos
    assert train_patients & val_patients == set()
    assert train_patients & test_patients == set()
    assert val_patients & test_patients == set()

    meta_train = meta[meta.patient_id.isin(train_patients)].reset_index(drop=True)
    meta_val = meta[meta.patient_id.isin(val_patients)].reset_index(drop=True)
    meta_test = meta[meta.patient_id.isin(test_patients)].reset_index(drop=True)

    return meta_train, meta_val, meta_test


# ──────────────────────────────────────────────
# Procesamiento por split
# ──────────────────────────────────────────────

def process_split(meta_split, split_name, output_dir, hu_min, hu_max):
    """Preprocesa y guarda los volúmenes completos de un split."""
    save_dir = os.path.join(output_dir, "preprocessed", split_name)
    os.makedirs(os.path.join(save_dir, "CT"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "masks"), exist_ok=True)

    # Obtener pacientes únicos del split
    patients = meta_split.patient_id.unique()
    processed = 0

    for patient_id in patients:
        ct_path = os.path.join(output_dir, "CT", f"{patient_id}.npy")
        mask_path = os.path.join(output_dir, "masks", f"{patient_id}.npy")

        if not os.path.exists(ct_path):
            continue

        # Cargar volumen completo
        ct = np.load(ct_path).astype(np.float32)
        mask = np.load(mask_path)

        # Aplicar windowing + normalización
        ct = apply_windowing(ct, hu_min, hu_max)

        # Guardar volumen preprocesado con escritura atómica.
        _safe_save_npy(os.path.join(save_dir, "CT", f"{patient_id}.npy"), ct)
        _safe_save_npy(os.path.join(save_dir, "masks", f"{patient_id}.npy"), mask)
        processed += 1

    # Guardar metadata del split
    meta_split.to_csv(os.path.join(save_dir, "metadata.csv"), index=False)
    print(f"  {split_name}: {processed} pacientes, {len(meta_split)} nódulos")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocesamiento de volúmenes CT completos para U-Net 3D"
    )
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--hu_min", type=float, default=-1000)
    parser.add_argument("--hu_max", type=float, default=600)
    args = parser.parse_args()

    # Cargar metadata
    meta = pd.read_csv(os.path.join(args.output_dir, "metadata.csv"))
    meta = meta.drop_duplicates(subset=["patient_id", "nodule_idx"]).reset_index(drop=True)
    print(f"Total nódulos: {len(meta)}")
    print(f"Pacientes únicos: {meta.patient_id.nunique()}")

    # Split por paciente
    meta_train, meta_val, meta_test = split_by_patient(meta)
    print(f"\nSplit:")
    print(f"  Train: {meta_train.patient_id.nunique()} pacientes, {len(meta_train)} nódulos")
    print(f"  Val:   {meta_val.patient_id.nunique()} pacientes, {len(meta_val)} nódulos")
    print(f"  Test:  {meta_test.patient_id.nunique()} pacientes, {len(meta_test)} nódulos")

    # Procesar cada split
    print(f"\nPreprocesando (HU=[{args.hu_min}, {args.hu_max}])...")
    process_split(meta_train, "train", args.output_dir, args.hu_min, args.hu_max)
    process_split(meta_val, "val", args.output_dir, args.hu_min, args.hu_max)
    process_split(meta_test, "test", args.output_dir, args.hu_min, args.hu_max)

    print("\nPreprocesamiento completo.")
    print(f"Datos en: {os.path.join(os.path.abspath(args.output_dir), 'preprocessed')}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
