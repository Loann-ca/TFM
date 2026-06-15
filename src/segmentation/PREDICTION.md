# Inferencia — Scripts de predicción

Una vez entrenado un modelo, estos scripts cargan el checkpoint guardado y generan una máscara de segmentación sobre un nuevo volumen CT sin necesidad de reentrenar.

## Scripts disponibles

| Script | Modelo | Carpeta |
|---|---|---|
| `unet2d_segmentation/predict_unet2d.py` | UNet 2D (slice-by-slice) | `unet2d_segmentation/` |
| `unet3d_segmentation/predict_unet3d.py` | UNet 3D (volumen completo) | `unet3d_segmentation/` |

---

## Qué hacen

1. **Cargan el checkpoint** `best.pt` del run indicado (o el que se especifique con `--checkpoint`).
2. **Reconstruyen la arquitectura** con los mismos hiperparámetros guardados en el checkpoint.
3. **Aplican el mismo preprocesado** que en entrenamiento: windowing HU [-1000, 600] y normalización a [0, 1].
4. **Ejecutan inferencia** en modo eval (sin gradientes).
5. **Guardan la máscara predicha** como array NumPy (H, W, D) en `uint8` (valores 0 o 1).
6. Opcionalmente guardan también el **volumen de probabilidades** en `float32` con `--save_probs`.

### Diferencia entre 2D y 3D

- **predict_unet2d**: procesa el volumen **corte a corte** (eje Z). Más rápido y con menor uso de VRAM.
- **predict_unet3d**: procesa el **volumen completo** de una sola pasada hacia adelante. Necesita más memoria GPU pero considera contexto volumétrico.

---

## Argumentos

| Argumento | Obligatorio | Default | Descripción |
|---|---|---|---|
| `--input_ct` | Sí | — | Ruta al volumen CT `.npy` de entrada (H, W, D) |
| `--run_dir` | Sí | — | Carpeta del run entrenado (ej. `checkpoints/unet2d_baseline/run_20260615_153012`) |
| `--checkpoint` | No | `run_dir/best.pt` | Ruta alternativa al checkpoint `.pt` |
| `--output_mask` | Sí | — | Ruta de salida para la máscara predicha `.npy` |
| `--save_probs` | No | `false` | Si se activa, guarda también el volumen de probabilidades |
| `--output_probs` | No | `<mask>_probs.npy` | Ruta de salida para las probabilidades |
| `--threshold` | No | `0.5` | Umbral sobre sigmoid para binarizar la máscara |
| `--hu_min` | No | `-1000.0` | Límite inferior del windowing HU |
| `--hu_max` | No | `600.0` | Límite superior del windowing HU |
| `--device` | No | `auto` | Dispositivo: `auto`, `cpu`, o `cuda` |

---

## Ejemplos de uso

### UNet 2D

```powershell
python src/segmentation/unet2d_segmentation/predict_unet2d.py `
  --input_ct output/CT/LIDC-IDRI-0078.npy `
  --run_dir checkpoints/unet2d_baseline/run_20260615_153012 `
  --output_mask output/predictions/LIDC-IDRI-0078_mask2d.npy
```

Con probabilidades y umbral personalizado:

```powershell
python src/segmentation/unet2d_segmentation/predict_unet2d.py `
  --input_ct output/CT/LIDC-IDRI-0078.npy `
  --run_dir checkpoints/unet2d_baseline/run_20260615_153012 `
  --output_mask output/predictions/LIDC-IDRI-0078_mask2d.npy `
  --save_probs `
  --threshold 0.4
```

### UNet 3D

```powershell
python src/segmentation/unet3d_segmentation/predict_unet3d.py `
  --input_ct output/CT/LIDC-IDRI-0078.npy `
  --run_dir checkpoints/unet3d_baseline/run_20260615_153012 `
  --output_mask output/predictions/LIDC-IDRI-0078_mask3d.npy
```

Con probabilidades:

```powershell
python src/segmentation/unet3d_segmentation/predict_unet3d.py `
  --input_ct output/CT/LIDC-IDRI-0078.npy `
  --run_dir checkpoints/unet3d_baseline/run_20260615_153012 `
  --output_mask output/predictions/LIDC-IDRI-0078_mask3d.npy `
  --save_probs
```

---

## Cómo encontrar el mejor run

Antes de predecir, localiza el run con mejor val Dice usando el registro histórico:

```python
import json

registry = "checkpoints/unet2d_baseline/training_runs_history.jsonl"
with open(registry) as f:
    runs = [json.loads(line) for line in f if line.strip()]

best = max(runs, key=lambda r: r["best_result"]["best_val_dice"])
print("Mejor run:", best["run_dir"])
print("Val Dice: ", best["best_result"]["best_val_dice"])
```

Luego pasa `best["run_dir"]` como `--run_dir`.

---

## Formato de salida

- **Máscara** `(H, W, D)` `uint8`: valores 0 (fondo) y 1 (nódulo).
- **Probabilidades** `(H, W, D)` `float32`: valor sigma entre 0.0 y 1.0 por vóxel.

Ambos arrays tienen el mismo sistema de coordenadas que el volumen de entrada.
