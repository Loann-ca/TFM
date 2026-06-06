# Optuna Hyperparameter Search 2D - Documentación

## Descripción General

`optuna_search_2d.py` realiza una **búsqueda automatizada y optimizada de hiperparámetros** para el modelo U-Net 2D usando Optuna. El objetivo es encontrar la mejor combinación de parámetros que maximice la métrica `val_dice` (validación Dice coefficient) en entrenamientos slice-by-slice de CT.

## Funcionalidad Principal

### 1. **Búsqueda Automática de Hiperparámetros (20 trials)**

Optuna ejecuta **20 entrenamientos independientes** con diferentes combinaciones de parámetros, cada uno completo (100 epochs).

**Parámetros explorados:**

| Parámetro | Rango/Opciones | Descripción |
|-----------|---|---|
| **lr** (learning rate) | [1e-4, 1e-3] continuo | Tasa de aprendizaje del optimizador Adam (búsqueda logarítmica) |
| **base_channels** | [16, 24, 32, 48] discreto | Canales base en primera capa del U-Net (afecta capacidad modelo) |
| **patch_size** | [64, 96, 128, 160] discreto | Tamaño de parche 2D en píxeles (importante para receptive field) |
| **bce_weight** | [0.2, 0.8] continuo | Peso de la pérdida BCE en la loss total |
| **dice_weight** | 1.0 - bce_weight | Peso automático de la pérdida Dice (complemento BCE) |
| **weight_decay** | [1e-6, 1e-4] continuo | Regularización L2 del optimizador (evita overfitting) |
| **lr_patience** | [3, 5, 10] discreto | Paciencia del ReduceLROnPlateau scheduler (epochs sin mejora antes de reducir lr) |

### 2. **Batch Size Adaptativo**

El batch size se ajusta automáticamente según `base_channels` (2D permite batch más grandes que 3D):

```
base_channels ≤ 16  → batch_size = 8
base_channels ≤ 32  → batch_size = 6
base_channels > 32  → batch_size = 4
```

### 3. **Ejecución de Entrenamientos**

Para cada trial, se ejecuta el comando:

```bash
python src/segmentation/unet2d_segmentation/train_unet2d_baseline.py \
    --lr <valor> \
    --base_channels <valor> \
    --patch_size <valor> \
    --batch_size <valor_adaptativo> \
  --save_dir output/optuna_2d/trial_{id}_lr_.../ \
    --epochs 100 \
    --bce_weight <valor> \
    --dice_weight <valor> \
    --weight_decay <valor> \
    --lr_patience <valor> \
    --patience 30
```

**Cada entrenamiento:**
- Entrena por **100 epochs** (o menos si early stopping a paciencia 30)
- Procesa todos los slices 2D de los volúmenes (CT[z] → parche 2D)
- Guarda checkpoints en `output/optuna_2d/trial_{id}_*/`
- Genera `summary.json` con métricas finales

### 4. **Recolección de Resultados**

Después de cada entrenamiento:
1. Lee el archivo `summary.json` de ese trial
2. Extrae el valor `best_val_dice` (mejor Dice en validación)
3. Retorna ese valor a Optuna

### 5. **Base de Datos Persistente (SQLite)**

Usa `output/optuna_2d/optuna_study_2d.db` para almacenar:
- Todos los trials ejecutados
- Parámetros de cada trial (incluyendo patch_size)
- Resultados (val_dice)
- Mejor trial encontrado

**Ventaja:** Si el script se interrumpe, puede reanudarse sin perder datos.

### 6. **Información en Consola**

Para cada trial imprime:

```
[Trial 0] Running 2D U-Net with:
  lr=5.3e-04, channels=32, patch_size=128
  batch=6
  bce_weight=0.45, dice_weight=0.55
  weight_decay=2.5e-05, lr_patience=5
[Trial 0] Best val Dice: 0.7892
```

### 7. **Resultado Final**

Al terminar todos los 20 trials, imprime:

```
============================================================
BEST 2D U-NET RESULT:
============================================================
Best val Dice: 0.8156
Best params: {'lr': 0.0004, 'base_channels': 32, 'patch_size': 128, ...}
Best trial: 12
============================================================
```

## Diferencias: 2D vs 3D

| Aspecto | U-Net 3D | U-Net 2D |
|--------|----------|---------|
| **Script Optuna** | `optuna_search.py` | `optuna_search_2d.py` |
| **BD SQLite** | `output/optuna_3d/optuna_study.db` | `output/optuna_2d/optuna_study_2d.db` |
| **Carpeta results** | `output/optuna_3d/` | `output/optuna_2d/` |
| **Patch** | 64×64×64 voxels (fijo) | 64-160×64-160 píxels (variable) |
| **Batch size** | 2-4 (VRAM limitado) | 4-8 (menos VRAM por slice) |
| **Entrada modelo** | Cubo 3D | Imagen 2D de un slice |
| **Ventaja 2D** | Más contexto 3D | Más rápido, menos VRAM |

## Tiempo Estimado

- **1 trial 2D (~100 epochs, ~512 slices/epoch):** ~20-40 min (más rápido que 3D)
- **20 trials:** ~7-12 horas total
- Puede dejarse ejecutando en background

## Cómo Usar

### Ejecutar búsqueda desde cero
```bash
python src/segmentation/unet2d_segmentation/optuna_search_2d.py
```

### Reanudar búsqueda (si se corta)
```bash
python src/segmentation/unet2d_segmentation/optuna_search_2d.py
```

Optuna detectará automáticamente `output/optuna_2d/optuna_study_2d.db` y continuará desde donde se quedó.

### Ver resultados intermedios
```bash
sqlite3 output/optuna_2d/optuna_study_2d.db
SELECT trial_id, value FROM trials ORDER BY value DESC LIMIT 5;
```

## Ejecutar ambos en Paralelo

Puedes optimizar tiempo corriendo 2D y 3D **simultáneamente** (usan diferentes GPUs o comparten):

```bash
# Terminal 1: U-Net 3D
python src/segmentation/unet3d_segmentation/optuna_search.py

# Terminal 2: U-Net 2D
python src/segmentation/unet2d_segmentation/optuna_search_2d.py
```

Ambos son **independientes** y no interfieren:
- BDs separadas (`output/optuna_3d/optuna_study.db` vs `output/optuna_2d/optuna_study_2d.db`)
- Carpetas de results diferentes (`output/optuna_3d/` vs `output/optuna_2d/`)
- Entrenamientos con diferentes modelos

## Comparativa con Baseline

**Anterior (manual, 1 solo config):**
- 2D: sin optimización específica
- Dice: desconocido

**Ahora (Optuna, 20 trials):**
- 2D: búsqueda de 7 hiperparámetros
- Esperanza: encontrar combinación que supere el 80% Dice del 3D

## Objetivo

Determinar si **U-Net 2D es mejor, peor o comparable** al 3D en el dataset LIDC-IDRI, y encontrar su configuración óptima.
