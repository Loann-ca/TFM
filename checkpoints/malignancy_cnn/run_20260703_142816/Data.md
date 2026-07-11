# Malignancy CNN Architecture (Final Run)

Run: `run_20260703_142816`  
Patch size: `64`  
Input channels (2.5D): `3` (axial, coronal, sagittal)  
Base channels: `32`  
Head dropout: `0.4`

```mermaid
flowchart TD
	A["Input patch\n3 x 64 x 64"] --> B["Stem\nConv 3x3: 3 to 32, BN, ReLU\n32 x 64 x 64"]
	B --> C["Stage 1\nResBlock2D 32ch, MaxPool2d k=2\n32 x 32 x 32"]
	C --> D["Stage 2\nConv 1x1: 32 to 64, ResBlock2D 64ch, MaxPool2d k=2\n64 x 16 x 16"]
	D --> E["Stage 3\nConv 1x1: 64 to 128, ResBlock2D 128ch, MaxPool2d k=2\n128 x 8 x 8"]
	E --> F["Stage 4\nConv 1x1: 128 to 256, ResBlock2D 256ch\n256 x 8 x 8"]
	F --> G["Global Average Pooling\nAdaptiveAvgPool2d output 1x1\n256 x 1 x 1"]
	G --> H["Head\nFlatten, Dropout 0.4, Linear 256 to 128, ReLU, Dropout 0.2, Linear 128 to 1"]
	H --> I["Output\n1 scalar"]
```

## ResBlock2D

Each residual block applies:

1. BatchNorm2d(C)
2. ReLU
3. Conv2d(C, C, 3x3, padding=1)
4. BatchNorm2d(C)
5. ReLU
6. Dropout2d(p=0.1)
7. Conv2d(C, C, 3x3, padding=1)
8. Residual sum with input

## Notes

- The model is trained in regression mode (MSE) to predict malignancy as a continuous value.
- During inference, output can be rounded and clipped to the range 1..5 for ordinal interpretation.