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
