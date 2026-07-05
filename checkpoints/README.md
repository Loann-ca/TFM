# checkpoints/

Directorio donde se almacenan todos los modelos entrenados y sus artefactos asociados.

## Estructura

```
checkpoints/
├── unet2d_baseline/                  # Entrenamientos del modelo UNet 2D
│   ├── training_runs_history.jsonl   # Registro global de todos los entrenamientos
│   └── run_YYYYMMDD_HHMMSS/          # Una carpeta por ejecución de entrenamiento
│       ├── best.pt
│       ├── last.pt
│       ├── history.csv
│       └── summary.json
│
└── unet3d_baseline/                  # Entrenamientos del modelo UNet 3D
    ├── training_runs_history.jsonl
    └── run_YYYYMMDD_HHMMSS/
        ├── best.pt
        ├── last.pt
        ├── history.csv
        └── summary.json
```

## Descripción de cada fichero

### `training_runs_history.jsonl`

Registro histórico global append-only. Cada línea es un JSON independiente con una entrada por entrenamiento completado. Contiene:

| Campo | Descripción |
|---|---|
| `run_id` | Nombre de la carpeta del run |
| `run_dir` | Ruta absoluta al directorio del run |
| `execution_date_utc` | Fecha y hora de inicio en UTC |
| `best_result` | Mejor época y mejor val Dice obtenido |
| `full_training_result` | Épocas completadas y métricas de la última época |
| `hyperparameters` | Todos los hiperparámetros usados en ese entrenamiento |
| `summary_path` | Ruta al `summary.json` del run |
| `history_path` | Ruta al `history.csv` del run |
| `best_checkpoint` | Ruta al `best.pt` del run |
| `last_checkpoint` | Ruta al `last.pt` del run |

Útil para comparar resultados entre ejecuciones sin abrir cada carpeta.

---

### `run_YYYYMMDD_HHMMSS/`

Carpeta creada automáticamente al iniciar un entrenamiento. El nombre incluye la fecha y hora UTC de inicio. Se puede sobrescribir con `--run_name`.

---

### `best.pt`

Checkpoint PyTorch del mejor modelo según Dice de validación. Se sobrescribe a lo largo del entrenamiento cada vez que mejora la métrica. Contiene:

```python
{
    "epoch": int,                  # Época en que se obtuvo
    "model_state_dict": ...,       # Pesos del modelo
    "optimizer_state_dict": ...,   # Estado del optimizador
    "best_val_dice": float,        # Mejor Dice de validación
    "args": dict,                  # Hiperparámetros del entrenamiento
}
```

**Este es el fichero que se carga para hacer predicciones** (ver scripts `predict_unet2d.py` / `predict_unet3d.py`).

---

### `last.pt`

Checkpoint PyTorch del estado al final de la última época ejecutada. Mismo formato que `best.pt`. Se usa para **reanudar** un entrenamiento interrumpido con `--resume`.

---

### `history.csv`

Curvas de entrenamiento por época. Se escribe de forma incremental para no perder datos si el proceso se interrumpe. Columnas:

| Columna | Descripción |
|---|---|
| `epoch` | Número de época |
| `train_loss` | Loss en train |
| `train_dice` | Dice en train |
| `val_loss` | Loss en validación |
| `val_dice` | Dice en validación |

---

### `summary.json`

Resumen completo del entrenamiento. Se escribe al finalizar. Contiene:

```json
{
  "run_id": "run_YYYYMMDD_HHMMSS",
  "started_at_utc": "...",
  "finished_at_utc": "...",
  "best_result": {
    "best_epoch": 42,
    "best_val_dice": 0.7831
  },
  "full_training_result": {
    "epochs_completed": 80,
    "final_epoch": 80,
    "final_train_loss": 0.12,
    "final_train_dice": 0.84,
    "final_val_loss": 0.19,
    "final_val_dice": 0.76
  },
  "hyperparameters": { ... }
}
```

## Seleccionar el mejor run para predicción

Para encontrar el run con mejor Dice de validación:

```powershell
# Muestra todos los runs ordenados por best_val_dice
Get-Content checkpoints/unet2d_baseline/training_runs_history.jsonl |
  ForEach-Object { $_ | ConvertFrom-Json } |
  Sort-Object { $_.best_result.best_val_dice } -Descending |
  Select-Object run_id, execution_date_utc, @{n='best_val_dice';e={$_.best_result.best_val_dice}} |
  Format-Table
```

O equivalente en Python:

```python
import json

with open("checkpoints/unet2d_baseline/training_runs_history.jsonl") as f:
    runs = [json.loads(line) for line in f if line.strip()]

best = max(runs, key=lambda r: r["best_result"]["best_val_dice"])
print(best["run_dir"])
print(best["best_result"])
```
