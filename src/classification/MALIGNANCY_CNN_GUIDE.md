# Guia de la CNN de Malignancy

Este documento explica como funciona [src/classification/malignancy_cnn.py](src/classification/malignancy_cnn.py), que input usa, como se entrena, como hace inferencia y como encaja con la U-Net.

## Objetivo

La CNN estima una puntuacion continua de malignidad para cada nodulo detectado. La escala sigue siendo 1..5, pero el modelo ya no hace clasificacion dura en 5 clases.

No clasifica el TAC completo entero como una sola etiqueta. Clasifica un nodule candidate por muestra.

Por cada muestra:

- Entrada de clasificacion: parche 2.5D centrado en un nodulo.
- Salida: puntuacion continua de malignidad.
- Prediccion final: puntuacion continua, con redondeo opcional a una clase entera.

## Flujo General

1. Se toma una mascara 3D del TAC completo.
2. Esa mascara puede ser:
	- ground truth, para entrenamiento mas limpio
	- salida de la U-Net, para inferencia real o para entrenar con detecciones automaticas
3. Se detectan nodulos como componentes conectadas en 3D.
4. De cada componente se obtiene centro, bounding box y volumen.
5. Se extrae un parche 2.5D del CT en ese centro.
6. La CNN predice una puntuacion de malignidad para ese nodulo.

En otras palabras:

- La U-Net dice donde hay posible nodulo.
- La CNN dice que tan maligno parece ese nodulo.

## Que Input Usa Exactamente

El input real de la CNN no es el TAC completo.

La entrada que ve la red es un parche 2.5D de 3 canales:

- canal 1: corte axial
- canal 2: corte coronal
- canal 3: corte sagital

Cada canal tiene tamano `patch_size x patch_size`.

Eso significa que la muestra que entra a la CNN tiene forma:

- `[3, patch_size, patch_size]`

Por defecto `patch_size = 64`.

## Arquitectura del Modelo

Modelo: `SmallMalignancyCNN`

- Entrada: 3 canales (`axial`, `coronal`, `sagital`) de tamano `patch_size x patch_size`.
- Bloques convolucionales 2D con BatchNorm + ReLU.
- MaxPooling para reducir resolucion.
- `AdaptiveAvgPool2d(1,1)` para compactar features.
- Capa `Linear` final con 1 valor continuo.

La CNN es ligera y rapida para iterar, y aprovecha contexto multi-plano sin coste de una 3D CNN completa.

Ventaja frente a una CNN 3D completa:

- menos coste de memoria
- mas simple de entrenar
- suficiente contexto local alrededor del nodulo

## Como Construye las Etiquetas

Durante entrenamiento, las etiquetas salen de `metadata.csv` del split:

- `malignancy` (valor continuo promedio de radiologos, tipicamente entre 1 y 5).
- Se redondea al entero mas cercano y se limita a [1, 5].
- Internamente se usa la puntuacion continua como objetivo de regresion.

Asignacion componente->anotacion:

1. Intenta emparejar por IoU 3D entre bounding box de componente y bounding box de `metadata.csv`.
2. Si no hay solape, usa la anotacion mas cercana por distancia entre centros.

Esto es necesario porque la mascara solo dice donde hay voxeles de nodulo, pero no trae la clase malignancy.

Ejemplo:

- una componente conectada corresponde a una region de voxeles 1
- el CSV tiene el bbox real del nodulo anotado por radiologos
- se busca la mejor correspondencia entre ambos
- una vez encontrado el match, se toma su valor `malignancy`

Si la anotacion dice `malignancy = 3.25`, ese valor se usa directamente como objetivo de entrenamiento.

## Que Son las Componentes Conectadas

Una componente conectada es un grupo de voxeles con valor 1 que estan pegados entre si.

Ejemplo mental:

- imagina una mascara binaria 3D
- los voxeles 1 forman "islas"
- cada isla separada por ceros es una componente conectada distinta

Por que se usan:

- la U-Net devuelve una sola mascara por TAC
- para clasificar nodulos individuales hace falta separarla en objetos separados
- cada objeto se trata como un nodule candidate

Limitacion:

- si dos nodulos estan muy juntos pueden fusionarse en una sola componente
- si un nodulo queda fragmentado, puede aparecer en varias componentes

## Entrenamiento

Comando base:

```bash
python src/classification/malignancy_cnn.py train --output_dir output --epochs 40 --batch_size 32 --eval_test
```

Artifacts del run (por defecto en `checkpoints/malignancy_cnn/run_...`):

- `best.pt`: mejor checkpoint por `val_macro_f1`.
- `last.pt`: ultimo checkpoint.
- `history.csv`: curvas por epoch.
- `summary.json`: resumen de metricas e hiperparametros.
- `training_runs_history.jsonl`: registro acumulado de runs.

### Que se coge como input para entrenar

El script construye una lista de muestras. Cada muestra es:

- un CT de paciente
- la posicion de un nodulo detectado en la mascara
- la etiqueta malignity asociada a ese nodulo

Luego, en `__getitem__`, carga el CT, recorta un parche 2.5D centrado en ese nodulo y devuelve ese parche como `image`.

O sea:

- no se mete el volumen completo entero en la CNN
- no se mete la mascara como input final
- la mascara se usa para localizar el nodulo
- el CT se usa para extraer la informacion visual del nodulo

### Fuentes de mascaras para entrenamiento

Por defecto, el entrenamiento usa las mascaras GT del dataset preprocesado:

- `output/preprocessed/train/masks`
- `output/preprocessed/val/masks`
- `output/preprocessed/test/masks`

Si quieres entrenar con mascaras predichas por la U-Net, pasas rutas distintas con `--train_mask_dir`, `--val_mask_dir` y `--test_mask_dir`.

## Inferencia

Comando base:

```bash
python src/classification/malignancy_cnn.py predict \
	--input_ct output/CT/LIDC-IDRI-0078.npy \
	--input_mask output/predictions3d/LIDC-IDRI-0078_mask.npy \
	--checkpoint checkpoints/malignancy_cnn/run_YYYYMMDD_HHMMSS/best.pt
```

Salida por defecto:

- `output/classification/LIDC-IDRI-0078_malignancy.csv`

Columnas principales:

- `nodule_id`
- `center_x`, `center_y`, `center_z`
- `bbox_x`, `bbox_y`, `bbox_z`
- `voxels`
- `pred_malignancy` (score continuo)
- `pred_malignancy_rounded` (clase entera 1..5)

### Que input usa en inferencia

En inferencia pasas dos cosas:

- el CT completo del paciente
- la mascara de nodulos del mismo paciente

La mascara puede venir de:

- la U-Net
- ground truth, si solo quieres comprobar el pipeline

Despues el script:

1. encuentra las componentes conectadas en la mascara
2. recorta un parche alrededor de cada componente
3. predice la malignidad de cada nodule candidate

Por eso el resultado final es un CSV con una fila por nodulo.

## Tu Duda: Dataset Original vs Salidas de U-Net

La clave es que el script permite cambiar la fuente de mascaras.

### A) Entrenar con dataset original (ground truth)

No tienes que pasar nada extra. Por defecto usa:

- `output/preprocessed/train/masks`
- `output/preprocessed/val/masks`
- `output/preprocessed/test/masks`

Comando:

```bash
python src/classification/malignancy_cnn.py train --output_dir output --epochs 40 --batch_size 32 --eval_test
```

### B) Entrenar con mascaras predichas por U-Net

Debes pasar carpetas alternativas con las mascaras para cada split:

- `--train_mask_dir`
- `--val_mask_dir`
- `--test_mask_dir`

Ejemplo:

```bash
python src/classification/malignancy_cnn.py train \
	--output_dir output \
	--train_mask_dir output/predictions3d/train \
	--val_mask_dir output/predictions3d/val \
	--test_mask_dir output/predictions3d/test \
	--epochs 40 --batch_size 32 --eval_test
```

Requisito importante:

- En cada carpeta de mascaras, los nombres deben coincidir con CT (`LIDC-IDRI-XXXX.npy`).
- `metadata.csv` sigue viniendo de `output/preprocessed/<split>/metadata.csv` para las etiquetas de malignidad.

### C) Inferir con mascaras predichas por U-Net

Este es el uso mas habitual cuando el sistema ya esta en produccion o en evaluacion completa:

1. ejecutas la U-Net sobre el CT completo
2. obtienes una mascara binaria
3. pasas esa mascara a `predict` de la CNN
4. la CNN te devuelve una malignity por componente

## Entrenar con GT y Predecir con U-Net (recomendado para empezar)

Este flujo suele ser el mas estable al inicio:

1. Entrena la CNN con mascaras GT (mejor calidad de candidatos).
2. En inferencia real usa `predict` con mascara predicha por U-Net.

Asi desacoplas entrenamiento robusto de clasificacion y evaluacion realista en pipeline completo.

## Ejemplo Mental Completo

Supongamos un paciente con CT completo.

1. La U-Net predice una mascara 3D.
2. Dentro de esa mascara aparecen 3 islas separadas de voxeles 1.
3. El script interpreta esas 3 islas como 3 nodule candidates.
4. Para cada una, calcula un centro.
5. Recorta un parche 2.5D del CT alrededor de ese centro.
6. La CNN predice:
	- nodule 1 -> malignity 2
	- nodule 2 -> malignity 4
	- nodule 3 -> malignity 1

El CSV final tiene 3 filas.

## Parametros Mas Relevantes

- `--patch_size` (default 64): tamano del parche 2.5D.
- `--min_voxels` (default 20): filtra componentes pequenas/ruido.
- `--base_channels` (default 32): capacidad del modelo.
- `--lr`, `--batch_size`, `--epochs`: hiperparametros de entrenamiento.
- `--device auto|cpu|cuda`.

### Lo Que Mas Afecta al Resultado

- `--patch_size`: define cuanta region alrededor del nodulo ve la CNN.
- `--min_voxels`: evita clasificar ruido muy pequeno.
- calidad de la mascara de entrada: si la U-Net falla, la CNN tambien lo notara.
- balance de clases: en tu dataset hay menos ejemplos de malignity alta, asi que puede haber desbalance.

## Consejos Practicos

- Si usas mascaras de U-Net para entrenar, incrementa `--min_voxels` para reducir falsos positivos pequenos.
- Compara dos runs: uno con GT y otro con U-Net masks para ver degradacion real de `val_loss`/`test_round_acc`.
- Revisa distribucion de clases en `summary.json` (`class_counts_train`) por posible desbalance.

### Regla practica recomendada

Si todavia estas desarrollando:

1. entrena primero con GT
2. valida que la CNN aprende
3. luego prueba inferencia con mascaras de U-Net
4. si quieres realismo extremo, reentrena con mascaras U-Net

## Ejemplos Rapidos

Entrenar rapido de smoke test:

```bash
python src/classification/malignancy_cnn.py train --output_dir output --epochs 1 --batch_size 16 --device cpu --run_name smoke_cls
```

Inferencia guardando CSV custom:

```bash
python src/classification/malignancy_cnn.py predict \
	--input_ct output/CT/LIDC-IDRI-0078.npy \
	--input_mask output/predictions2d/LIDC-IDRI-0078_mask.npy \
	--checkpoint checkpoints/malignancy_cnn/run_YYYYMMDD_HHMMSS/best.pt \
	--output_csv output/classification/LIDC-IDRI-0078_from2d.csv
```

## Resumen Ultra Corto

- La U-Net da una mascara del TAC completo.
- Esa mascara se separa en componentes conectadas.
- Cada componente se toma como un nodulo candidato.
- La CNN no ve el TAC entero, ve un parche centrado en ese nodulo.
- La etiqueta de malignity sale de `metadata.csv`.
- Puedes entrenar con GT o con mascaras de U-Net cambiando las rutas de mascara.

---

## Papers de Referencia

Esta seccion resume los papers mas relevantes en clasificacion de malignidad de nodulos pulmonares con deep learning sobre LIDC-IDRI.

### NoduleX — Causey et al. (2018)

**Referencia:** Causey J.L. et al., *Highly accurate model for prediction of lung nodule malignancy with CT scans*, Scientific Reports, 2018.

**Resultados:**
- AUC: **0.9965** en LIDC-IDRI
- Supera a radiólogos en discriminación benigno/maligno

**Método clave:**
- CNN profunda 2D sobre slices axiales del nódulo segmentado
- Clasificación **binaria**: benigno (malignancy 1-2) vs maligno (malignancy 4-5), excluye malignancy 3
- Entrenamiento sobre el recorte del nódulo, no sobre el TAC completo
- Augmentación intensiva: flips, rotaciones, ruido

---

### DeepLung — Zhu et al. (2018)

**Referencia:** Zhu W. et al., *DeepLung: Deep 3D Dual Path Nets for Automated Pulmonary Nodule Detection and Classification*, WACV, 2018.

**Resultados:**
- Accuracy: **~90.4%** en clasificación de malignidad (LIDC-IDRI)
- AUC: **~0.95**

**Método clave:**
- 3D Dual Path Networks para el clasificador de malignidad
- Usa parches 3D del nódulo (32×32×32)
- Pipeline completo: detección 3D + clasificación 3D en un solo framework
- GBM sobre las features extraídas por la CNN como paso final

---

### Revisión Sistemática — Wulaningsih et al. (2024)

**Referencia:** Wulaningsih W. et al., *Deep Learning Models for Predicting Malignancy Risk in CT-detected Pulmonary Nodules*, Lung (Springer), 2024.

**Conclusiones principales:**
- Los mejores modelos DL alcanzan AUC > 0.95 de forma consistente en LIDC-IDRI
- La clasificación **binaria** (benigno vs maligno, excluir ambiguos) es el protocolo estándar en la literatura
- Excluir nódulos con malignancy = 3 mejora métricas porque son los casos genuinamente inciertos
- Las CNNs 2D y 2.5D compiten bien con las 3D si el parche está bien centrado en el nódulo
- El principal cuello de botella es el desbalance de clases (pocos nódulos muy malignos en LIDC-IDRI)

---

## Mejoras Implementadas Basadas en los Papers

A continuación se documentan los cambios aplicados a `malignancy_cnn.py` para aproximar el sistema a las mejores prácticas de la literatura.

### 1. Arquitectura más profunda con bloques residuales (`MalignancyCNN`)

**Motivación:** La arquitectura original (`SmallMalignancyCNN`, 3 capas conv planas) tenía muy poca capacidad para capturar texturas complejas de nódulos. NoduleX y DeepLung usan redes con conexiones residuales.

**Cambio:**
- Nueva clase `ResBlock2D` (pre-activation residual block, He et al. 2016): dos conv 3×3 con skip connection, BatchNorm y Dropout2d.
- Nueva clase `MalignancyCNN`: stem → 4 etapas residuales (c → 2c → 4c → 8c canales) → Global Average Pooling → head de 2 capas lineales con dropout.
- `SmallMalignancyCNN` queda como alias para no romper checkpoints antiguos.

**Parámetro:** `--base_channels` (default 32). Con 32, la red tiene ~32→64→128→256 canales en cada etapa.

---

### 2. Modo de clasificación binaria (`--task binary`)

**Motivación:** NoduleX consigue AUC 0.9965 con clasificación binaria benigno/maligno. La escala 1-5 de malignancy es ruidosa (acuerdo bajo entre radiólogos) y una regresión MSE no la aprovecha bien.

**Cambio:**
- Nuevo argumento `--task` con opciones `regression` (por defecto, comportamiento anterior) y `binary`.
- En modo `binary`:
  - malignancy ≤ 2 → clase 0 (benigno)
  - malignancy ≥ 4 → clase 1 (maligno)
  - malignancy = 3 → **excluido** del dataset (ambiguo/incierto)
- La función `remap_binary()` aplica este mapeo automáticamente a train, val y test.

**Comando recomendado para comparar con papers:**

```bash
python src/classification/malignancy_cnn.py train \
    --task binary \
    --epochs 150 \
    --batch_size 32 \
    --eval_test
```

---

### 3. Focal Loss para binario (`FocalBCELoss`)

**Motivación:** En LIDC-IDRI los nódulos malignos (malignancy 4-5) son minoría. La BCE estándar aprende a predecir siempre benigno. La Focal Loss (Lin et al. 2017) reduce el peso de los ejemplos fáciles y fuerza al modelo a aprender los difíciles.

**Cambio:**
- Nueva clase `FocalBCELoss`: $\text{FL}(p_t) = (1 - p_t)^\gamma \cdot \text{BCE}$
- Se calcula automáticamente `pos_weight = n_neg / n_pos` a partir del set de entrenamiento para compensar el desbalance.
- Argumento `--focal_gamma` (default 2.0). Con 0 equivale a BCE estándar con pos_weight.

---

### 4. Class weights realmente aplicados (bug fix)

**Motivación:** El código anterior calculaba `class_weights` a partir de la frecuencia inversa de cada clase, pero **nunca los conectaba a la función de pérdida**. Esto era un bug silencioso.

**Cambio:**
- En modo `regression`: se calculan pesos por muestra (`sample_weights`) proporcionales a la rareza de cada clase y se pasan al MSELoss de forma ponderada.
- En modo `binary`: el desbalance se corrige directamente con `pos_weight` en `FocalBCELoss`.

---

### 5. Gradient clipping

**Motivación:** Con redes más profundas (4 etapas residuales) los gradientes pueden explotar, especialmente en las primeras épocas.

**Cambio:**
- `nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)` aplicado en cada paso de optimización.

---

### 6. AUC como métrica principal en modo binario

**Motivación:** Los papers reportan AUC como métrica estándar para comparación. La balanced accuracy sola no es suficiente para comparar con la literatura.

**Cambio:**
- `roc_auc_score` de scikit-learn calculado en cada epoch en modo `binary`.
- El checkpoint `best.pt` se guarda cuando mejora el **AUC** (en modo binario) o la **balanced accuracy** (en modo regresión).
- AUC se reporta en consola, en `history.csv` y en `summary.json`.

---

### 7. Augmentación de intensidad

**Motivación:** Los nódulos pueden aparecer con distintas intensidades según el escáner y el protocolo de adquisición. Augmentar intensidad hace el modelo más robusto.

**Cambio (solo en modo `augment=True`, es decir, entrenamiento):**
- Jitter de brillo: desplazamiento aleatorio ±0.10 sobre valores normalizados [0,1]
- Jitter de contraste: factor multiplicativo aleatorio en [0.9, 1.1]
- Ruido gaussiano: σ=0.02, probabilidad 30%

Los augmentos geométricos originales (flips, rot90) se mantienen.

---

### Tabla Resumen de Cambios

| Cambio | Motivación (paper) | Argumento/Clase |
|---|---|---|
| Arquitectura residual 4 etapas | NoduleX, DeepLung | `MalignancyCNN`, `ResBlock2D` |
| Modo clasificación binaria | NoduleX (AUC 0.9965) | `--task binary` |
| Focal Loss + pos_weight automático | Desbalance de clases | `FocalBCELoss`, `--focal_gamma` |
| Class weights conectados a la loss | Bug fix | automático |
| Gradient clipping | Redes más profundas | `clip_grad_norm_` |
| AUC como métrica principal | Estándar en la literatura | `roc_auc_score` en `run_epoch` |
| Augmentación de intensidad | Variabilidad escáner | en `MalignancyNoduleDataset` |

### Rendimiento esperado tras los cambios

Con `--task binary` y la nueva arquitectura, el objetivo realista basado en la literatura es:

- **AUC > 0.90** con 100-150 épocas y los datos de LIDC-IDRI completos
- **AUC > 0.95** potencialmente alcanzable con ajuste de hiperparámetros y `--base_channels 48`
- La balanced accuracy en binario debería superar 0.75 con clase bien balanceada por la Focal Loss
