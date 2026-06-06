# Guia del script train_unet3d_baseline.py

Este documento explica de forma continua como funciona el script `train_unet3d_baseline.py` para segmentacion de nodulos pulmonares con U-Net 3D.

## 1. Objetivo del script

El script entrena una red U-Net 3D para segmentar nodulos en TACs de LIDC-IDRI.

La idea clave es:
- Los datos se guardan por paciente como volumen completo (CT y mascara full-size).
- El entrenamiento no mete el volumen completo a la red de una vez.
- El dataset extrae patches 3D (subvolumenes) al vuelo para entrenar de forma viable en memoria.

## 2. Estructura esperada de datos

El script espera esta estructura:

- output/preprocessed/train/CT/*.npy
- output/preprocessed/train/masks/*.npy
- output/preprocessed/train/metadata.csv
- output/preprocessed/val/CT/*.npy
- output/preprocessed/val/masks/*.npy
- output/preprocessed/val/metadata.csv
- output/preprocessed/test/CT/*.npy
- output/preprocessed/test/masks/*.npy
- output/preprocessed/test/metadata.csv

Cada fichero de CT y mascara corresponde a un paciente (mismo nombre base).

## 3. Flujo general del script

El flujo en `main()` es:

1. Leer argumentos CLI (epocas, batch size, patch size, etc.).
2. Fijar semilla para reproducibilidad.
3. Elegir dispositivo (`cuda` o `cpu`).
4. Crear datasets de train/val.
5. Crear dataloaders de train/val.
6. Construir modelo U-Net 3D.
7. Entrenar por epocas y validar cada epoca.
8. Guardar `last.pt` siempre y `best.pt` si mejora Dice de validacion.
9. Guardar historico (`history.csv`) y resumen (`summary.json`).
10. Opcionalmente evaluar test si se activa `--eval_test`.

## 4. Como se construyen las muestras (Dataset)

La clase `FullVolumeNoduleDataset` hace dos fases:

### 4.1 Fase de planificacion (en __init__)

Aqui no se recortan voxeles todavia.
Se crea una lista `self.items` con metadatos de cada muestra:
- ruta CT
- ruta mascara
- patient_id
- centro del patch `(cx, cy, cz)`

Por paciente, genera `patches_per_patient` muestras:
- aproximadamente la mitad centradas en nodulos (usando el centro del bbox de `metadata.csv`)
- el resto aleatorias (fondo) dentro de limites validos

### 4.2 Fase de materializacion (en __getitem__)

Cuando `DataLoader` pide una muestra:
- se carga CT y mascara con `mmap_mode='r'`
- se recorta un cubo 3D de tamano `patch_size`
- se binariza la mascara
- se aplica augmentacion opcional (flip/rot90)
- se reordenan ejes a formato PyTorch 3D
- se devuelve:
  - `image`: tensor `[1, D, H, W]`
  - `mask`: tensor `[1, D, H, W]`

## 5. Que hace el DataLoader

`DataLoader` no inventa datos; orquesta el muestreo en lotes.

En cada iteracion del bucle `for batch in loader`:
- selecciona indices (barajados en train)
- llama a `__getitem__` para cada indice del batch
- agrupa las muestras en tensores batch

Formato batch que entra al modelo:
- `x`: `[B, C, D, H, W]`
- `B` = batch size
- `C` = canales (1 para CT)

## 6. Arquitectura de la U-Net 3D

La red tiene dos mitades:

### 6.1 Encoder (contraccion)

Bloques `DoubleConv3D` + `MaxPool3d(2)`.

Efecto:
- baja resolucion espacial (D/H/W se reducen)
- sube numero de canales (mas capacidad semantica)

Canales por nivel (con base_channels=16):
- 16 -> 32 -> 64 -> 128 (bottleneck)

### 6.2 Decoder (expansion)

Bloques de subida con `ConvTranspose3d` y concatenacion con skips del encoder.

Efecto:
- recupera resolucion espacial
- reutiliza detalle fino del encoder (skip connections)
- termina en `Conv3d(1x1x1)` para producir logits binarios

## 7. Que hace el forward

El metodo `forward` define la ruta de datos:

1. Baja por encoder y guarda `e1`, `e2`, `e3`.
2. Pasa por bottleneck.
3. Sube por decoder:
   - upsample
   - concat con skip correspondiente
   - refina con convoluciones
4. Devuelve logits de salida (sin sigmoid).

Importante:
- No llamas `forward` directamente.
- PyTorch lo ejecuta cuando haces `model(x)`.

## 8. Loss y metricas

En cada batch se calculan:

- BCEWithLogitsLoss (estabilidad numerica con logits)
- Dice loss (solape de segmentacion)

Loss total:
- `loss = bce_weight * BCE + dice_weight * Dice`

Metrica reportada:
- Dice score con umbral 0.5 sobre sigmoid(logits)

## 9. Entrenamiento por epocas

En cada epoca:

1. `run_epoch(..., optimizer=optimizer)` para train:
   - forward
   - loss
   - backward
   - step de optimizador

2. `run_epoch(..., optimizer=None)` para validacion:
   - sin gradientes

3. Se guarda:
   - `last.pt` siempre
   - `best.pt` si mejora Dice de validacion

Al final:
- `history.csv` con curvas por epoca
- `summary.json` con mejor epoca y mejor Dice (y test opcional)

## 10. Que guarda cada artefacto

En `checkpoints/unet3d_baseline/`:

- `last.pt`: ultimo estado (modelo + optimizador + epoca)
- `best.pt`: mejor modelo segun val Dice
- `history.csv`: train_loss, train_dice, val_loss, val_dice por epoca
- `summary.json`: resumen final

## 11. Parametros clave para ajustar

- `--patch_size`: tamano del patch 3D (memoria vs contexto)
- `--patches_per_patient`: cuantas muestras aporta cada paciente por epoca
- `--batch_size`: patches por iteracion
- `--base_channels`: anchura de la red
- `--bce_weight` y `--dice_weight`: balance de la loss
- `--num_workers`: paralelismo de carga

## 12. Idea mental final

- Fuente de datos: volumen completo por paciente.
- Unidad de entrenamiento: patch 3D.
- Red: encoder (contexto) + decoder (detalle) + skips (precision espacial).
- Objetivo: aprender a predecir mascara binaria de nodulo en cada patch.
