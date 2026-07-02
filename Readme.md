# TFM — Lung Nodule Detection & Classification (LIDC-IDRI)

## Dependencias

```bash
pip install pylidc pydicom SimpleITK matplotlib numpy scipy scikit-image pandas tqdm torch scikit-learn
```

## 1) Generar volúmenes CT completos y máscaras

```bash
python src/preprocessing/process_all_masks.py --output_dir output --clevel 0.5
```

Produce:
- `output/CT/{patient_id}.npy` — volumen CT completo por paciente (float32)
- `output/masks/{patient_id}.npy` — máscara binaria full-size con todos los nódulos (uint8)
- `output/metadata.csv` — features anotadas por nódulo

## 2) Preprocesar para entrenamiento

```bash
python src/preprocessing/preprocessing_pipeline.py --output_dir output
```

Produce `output/preprocessed/{train,val,test}/` con volúmenes normalizados.

## 3) Entrenar U-Net 3D baseline

```bash
python src/segmentation/unet3d_segmentation/train_unet3d_baseline.py --output_dir output --epochs 40 --batch_size 2 --eval_test
```

Checkpoints guardados en `checkpoints/unet3d_baseline/`:
- `best.pt` — mejor epoch por val Dice
- `last.pt` — último epoch
- `history.csv` — curvas de loss y Dice
- `summary.json` — métricas finales

## 4) Clasificar malignancy de nódulos detectados (CNN)

Entrenamiento (usa componentes conectados de máscaras por split; por defecto usa `preprocessed/*/masks`):

```bash
python src/classification/malignancy_cnn.py train --output_dir output --epochs 40 --batch_size 32 --eval_test
```

Inferencia sobre nódulos detectados por U-Net:

```bash
python src/classification/malignancy_cnn.py predict \
	--input_ct output/CT/LIDC-IDRI-0078.npy \
	--input_mask output/predictions3d/LIDC-IDRI-0078_mask.npy \
	--checkpoint checkpoints/malignancy_cnn/run_YYYYMMDD_HHMMSS/best.pt
```

Salida de inferencia:
- `output/classification/<patient>_malignancy.csv`
- Una fila por nódulo detectado (componente conectada) con centro, bbox, volumen y probabilidades `prob_1..prob_5`.