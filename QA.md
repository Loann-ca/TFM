# Preguntas y Respuestas — TFM

---

## 1. ¿El CT guarda el patch o todo el CT?

**Respuesta:** Guarda solo el patch (el recorte del nódulo), no el CT completo. La línea `ct_patch = vol[bbox[0], bbox[1], bbox[2]]` recorta del volumen completo (512×512×N) solo la región del bounding box del nódulo. El resultado tiene el mismo shape que la máscara (por ejemplo 30×25×12). El volumen completo no tiene sentido guardarlo porque ocupa mucho más disco y para la U-Net y el CNN necesitas pares (patch, máscara) del mismo tamaño.

---

## 2. ¿Lo guarda en 3D?

**Respuesta:** Sí. El `ct_patch` es un array con 3 dimensiones (x, y, z). La máscara también es 3D con el mismo shape. Para la U-Net 2D, se extraerían los slices individuales en el DataLoader al entrenar, no en el preprocesamiento.

---

## 3. ¿Cómo se gestionan los valores None en las features de los radiólogos?

**Respuesta:** En LIDC-IDRI las 9 features son obligatorias en el esquema de anotación, así que no deberían existir valores nulos. Aun así, se añadió protección: se filtran los `None` antes de promediar, y si ningún radiólogo rellenó una feature, se guarda como `None` en el CSV.

```python
values = [getattr(ann, feat) for ann in nod if getattr(ann, feat) is not None]
features[feat] = round(np.mean(values), 2) if values else None
```

---

## 4. ¿Por qué el CSV solo tiene nódulos 0, 1 y 3 para LIDC-IDRI-0078?

**Respuesta:** Inicialmente el script filtraba nódulos con menos de 3 anotaciones. El nódulo 2 fue anotado por menos de 3 radiólogos y se descartaba. Se eliminó ese filtro — ahora se guardan todos los nódulos y el campo `num_annotations` en el CSV permite filtrar después según lo que se necesite en cada experimento.

---

## 5. ¿Qué pasos de preprocesamiento hacer antes de la U-Net?

**Respuesta:**

1. **Windowing HU** — Clip a un rango relevante y normalizar a [0, 1].
2. **Resampling isotrópico** — Resamplear a 1mm×1mm×1mm para que los tamaños sean consistentes entre pacientes.
3. **Padding/crop a tamaño fijo** — La U-Net necesita input de tamaño fijo. Padding con ceros (aire) es preferible a resize para no distorsionar proporciones.
4. **Data augmentation** — Flips, rotaciones 90°. No cambiar brillo/contraste porque los valores HU tienen significado físico.
5. **Split train/val/test** — Partir por paciente, no por nódulo, para evitar data leakage.

---

## 6. ¿Qué es el windowing y por qué se hace?

**Respuesta:** El windowing recorta el rango de intensidades HU para maximizar el contraste en los tejidos de interés. Sin windowing, al normalizar a [0, 1] el nódulo (+30 a +50 HU) ocupa un rango minúsculo porque valores extremos como hueso (+2000) estiran toda la escala. Con windowing a [-1000, 600], se recorta todo lo que pase de 600 y el nódulo pasa a ocupar un rango mucho más visible (0.74-0.75 vs 0.36-0.37 sin windowing).

---

## 7. ¿Qué es un vóxel? ¿Por qué [-1000, 400] si la calcificación es +700?

**Respuesta:** Un vóxel es un píxel en 3D — cada posición en el volumen CT. Su valor en HU representa la densidad del tejido. El rango [-1000, 400] es la ventana pulmonar estándar, pero para este proyecto se amplió a [-1000, 600] porque la calcificación es una de las features a clasificar y conviene preservar esa información. Las calcificaciones se saturan pero siguen apareciendo como valor máximo.

---

## 8. ¿Qué valor de entrada darle a la CNN de clasificación?

**Respuesta:** La máscara binaria sola no sirve. La CNN necesita ver la apariencia del nódulo. La mejor opción es usar **CT + máscara como 2 canales** (input shape: x, y, z, 2). El canal CT aporta la textura, bordes y densidad; el canal máscara le dice a la red exactamente dónde mirar. Esto es lo más usado en la literatura para LIDC-IDRI.

---

## 9. ¿Nos haría falta coger valores como -1000 (aire) para la U-Net?

**Respuesta:** El aire puro (-1000) no aporta información para segmentar nódulos. Pero mantener -1000 como mínimo del windowing tiene una ventaja práctica: el padding con ceros corresponde naturalmente a aire, que es lo que hay fuera del volumen CT. Con un rango más estrecho (ej: [-700, 600]), el padding con ceros correspondería a tejido pulmonar (-700 HU), introduciendo información falsa. La pérdida de contraste por usar el rango más amplio es despreciable en la práctica.

---

## 10. ¿Opción A (rango estrecho + padding explícito) u opción B (mantener -1000)?

**Respuesta:** Opción B (mantener -1000). El padding y los valores de la imagen quedan en la misma escala de forma natural — cero significa aire tanto en el padding como en el CT real. Con la opción A habría que gestionar el padding por separado (usar un valor especial como -1), añadiendo complejidad sin ganancia real. La diferencia de contraste entre ambas opciones (~19% más resolución con rango estrecho) es despreciable porque la red tiene de sobra con float32 para distinguir los tejidos.

---

## 11. ¿Cómo visualizar los datos preprocesados?

**Respuesta:** Se creó `visualize_preprocessed.py` que lee los datos de `output/preprocessed/{train,val,test}/` y ofrece varios modos de visualización:

- **Nódulo individual** (slice central con CT, máscara y overlay):
  ```bash
  python visualize_preprocessed.py --split train --index 0
  ```
- **Todos los slices** de un nódulo:
  ```bash
  python visualize_preprocessed.py --split train --index 0 --all_slices
  ```
- **Filtrar por paciente**:
  ```bash
  python visualize_preprocessed.py --patient LIDC-IDRI-0078
  ```
- **Resumen de los 3 splits** (estadísticas + comparación de 8 nódulos + distribución de malignancy):
  ```bash
  python visualize_preprocessed.py --summary
  ```






