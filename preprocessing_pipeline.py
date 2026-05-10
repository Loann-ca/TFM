"""
Pipeline de preprocesamiento para U-Net 3D.

Prepara los parches CT y máscaras generados por process_all_masks.py
para entrenar una U-Net 3D de segmentación de nódulos pulmonares.

Pasos:
  1. Split train/val/test por paciente (evita data leakage)
  2. Windowing HU [-1000, 600] + normalización a [0, 1]
  3. Padding/crop a tamaño fijo (64x64x64)
  4. Guardado de datos preprocesados por split

Uso:
    python preprocessing_pipeline.py --output_dir output --target_size 64
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from sklearn.model_selection import GroupShuffleSplit


# ──────────────────────────────────────────────
# Funciones de preprocesamiento
# ──────────────────────────────────────────────

def resample_volume(ct, mask, original_spacing, target_spacing=(1.0, 1.0, 1.0)):
    """Resamplea CT y máscara a spacing isotrópico.

    Args:
        ct: array 3D con valores HU
        mask: array 3D binario
        original_spacing: (sx, sy, sz) en mm
        target_spacing: spacing objetivo en mm

    Returns:
        ct_resampled, mask_resampled
    """
    resize_factor = np.array(original_spacing) / np.array(target_spacing)

    # CT: interpolación cúbica para suavidad
    ct_resampled = zoom(ct, resize_factor, order=3)

    # Máscara: nearest neighbor para mantener valores binarios
    mask_resampled = zoom(mask, resize_factor, order=0)
    mask_resampled = (mask_resampled > 0.5).astype(np.uint8)

    return ct_resampled, mask_resampled


def apply_windowing(ct, hu_min=-1000, hu_max=600):
    """Aplica windowing HU y normaliza a [0, 1].

    Recorta intensidades fuera del rango. Usamos [-1000, 600]
    para preservar información de calcificaciones (+700 se satura
    pero sigue apareciendo como valor máximo).
    """
    ct = np.clip(ct, hu_min, hu_max)
    ct = (ct - hu_min) / (hu_max - hu_min)
    return ct.astype(np.float32)


def pad_or_crop(volume, target_shape=(64, 64, 64)):
    """Ajusta el volumen a target_shape con padding centrado o crop.

    - Si el volumen es menor en un eje: padding con ceros (aire)
    - Si es mayor: crop centrado
    """
    result = np.zeros(target_shape, dtype=volume.dtype)

    starts_src, ends_src = [], []
    starts_dst, ends_dst = [], []

    for i in range(3):
        if volume.shape[i] <= target_shape[i]:
            pad = (target_shape[i] - volume.shape[i]) // 2
            starts_dst.append(pad)
            ends_dst.append(pad + volume.shape[i])
            starts_src.append(0)
            ends_src.append(volume.shape[i])
        else:
            crop = (volume.shape[i] - target_shape[i]) // 2
            starts_dst.append(0)
            ends_dst.append(target_shape[i])
            starts_src.append(crop)
            ends_src.append(crop + target_shape[i])

    result[
        starts_dst[0]:ends_dst[0],
        starts_dst[1]:ends_dst[1],
        starts_dst[2]:ends_dst[2]
    ] = volume[
        starts_src[0]:ends_src[0],
        starts_src[1]:ends_src[1],
        starts_src[2]:ends_src[2]
    ]

    return result


def augment_3d(ct, mask, seed=None):
    """Augmentaciones geométricas para un par CT/máscara.

    - Flip aleatorio en cada eje (50% probabilidad)
    - Rotación 90° aleatoria en plano axial

    Se aplican idénticamente a CT y máscara.
    Solo usar en entrenamiento.
    """
    rng = np.random.RandomState(seed)

    for axis in range(3):
        if rng.random() > 0.5:
            ct = np.flip(ct, axis=axis)
            mask = np.flip(mask, axis=axis)

    k = rng.randint(0, 4)
    if k > 0:
        ct = np.rot90(ct, k=k, axes=(0, 1))
        mask = np.rot90(mask, k=k, axes=(0, 1))

    return ct.copy(), mask.copy()


def preprocess_nodule(ct_path, mask_path, target_shape=(64, 64, 64),
                      hu_min=-1000, hu_max=600):
    """Pipeline completo para un nódulo.

    Orden: Windowing → Normalización [0,1] → Padding/Crop
    (Resampling se haría aquí si se dispone del spacing original)
    """
    ct = np.load(ct_path).astype(np.float32)
    mask = np.load(mask_path)

    ct = apply_windowing(ct, hu_min, hu_max)
    ct = pad_or_crop(ct, target_shape)
    mask = pad_or_crop(mask, target_shape)

    return ct, mask


# ──────────────────────────────────────────────
# Split por paciente
# ──────────────────────────────────────────────

def split_by_patient(meta):
    """Divide en train/val/test (70/15/15) agrupando por paciente."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(splitter.split(meta, groups=meta["patient_id"]))

    meta_train = meta.iloc[train_idx].reset_index(drop=True)
    meta_temp = meta.iloc[temp_idx].reset_index(drop=True)

    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(splitter2.split(meta_temp, groups=meta_temp["patient_id"]))

    meta_val = meta_temp.iloc[val_idx].reset_index(drop=True)
    meta_test = meta_temp.iloc[test_idx].reset_index(drop=True)

    # Verificar que no hay pacientes compartidos
    assert set(meta_train.patient_id) & set(meta_val.patient_id) == set()
    assert set(meta_train.patient_id) & set(meta_test.patient_id) == set()
    assert set(meta_val.patient_id) & set(meta_test.patient_id) == set()

    return meta_train, meta_val, meta_test


# ──────────────────────────────────────────────
# Procesamiento por split
# ──────────────────────────────────────────────

def process_split(meta_split, split_name, output_dir, target_shape, hu_min, hu_max):
    """Preprocesa y guarda un split completo."""
    save_dir = os.path.join(output_dir, "preprocessed", split_name)
    os.makedirs(os.path.join(save_dir, "CT"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "masks"), exist_ok=True)

    processed = 0
    skipped = 0

    for _, row in meta_split.iterrows():
        fname = f"{row.patient_id}_nod{int(row.nodule_idx)}.npy"
        ct_path = os.path.join(output_dir, "CT", fname)
        mask_path = os.path.join(output_dir, "masks", fname)

        if not os.path.exists(ct_path):
            skipped += 1
            continue

        ct, mask = preprocess_nodule(
            ct_path, mask_path,
            target_shape=target_shape,
            hu_min=hu_min, hu_max=hu_max,
        )

        np.save(os.path.join(save_dir, "CT", fname), ct)
        np.save(os.path.join(save_dir, "masks", fname), mask)
        processed += 1

    meta_split.to_csv(os.path.join(save_dir, "metadata.csv"), index=False)
    print(f"  {split_name}: {processed} nódulos guardados, {skipped} omitidos")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocesamiento de datos para U-Net 3D"
    )
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--target_size", type=int, default=64,
                        help="Tamaño objetivo por eje (default: 64)")
    parser.add_argument("--hu_min", type=float, default=-1000)
    parser.add_argument("--hu_max", type=float, default=600)
    args = parser.parse_args()

    target_shape = (args.target_size, args.target_size, args.target_size)

    # Cargar metadata
    meta = meta.drop_duplicates(subset=["patient_id", "nodule_idx"]).reset_index(drop=True)
    print(f"Total nódulos: {len(meta)}")
    print(f"Pacientes únicos: {meta.patient_id.nunique()}")

    # Split
    meta_train, meta_val, meta_test = split_by_patient(meta)
    print(f"\nSplit:")
    print(f"  Train: {len(meta_train)} nódulos ({meta_train.patient_id.nunique()} pacientes)")
    print(f"  Val:   {len(meta_val)} nódulos ({meta_val.patient_id.nunique()} pacientes)")
    print(f"  Test:  {len(meta_test)} nódulos ({meta_test.patient_id.nunique()} pacientes)")

    # Procesar cada split
    print(f"\nPreprocesando (target={target_shape}, HU=[{args.hu_min}, {args.hu_max}])...")
    process_split(meta_train, "train", args.output_dir, target_shape, args.hu_min, args.hu_max)
    process_split(meta_val, "val", args.output_dir, target_shape, args.hu_min, args.hu_max)
    process_split(meta_test, "test", args.output_dir, target_shape, args.hu_min, args.hu_max)

    print("\nPreprocesamiento completo.")
    print(f"Datos guardados en: {os.path.join(os.path.abspath(args.output_dir), 'preprocessed')}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
