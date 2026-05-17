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

---

## 12. En `train_unet3d_baseline.py`, ¿qué hace `__getitem__` y dónde se utiliza?

**Respuesta:** `__getitem__` construye una muestra individual del dataset: carga CT y máscara, recorta el patch 3D alrededor del centro, binariza máscara, aplica augmentación opcional, reordena ejes y devuelve tensores (`image`, `mask`). No se llama manualmente: lo invoca automáticamente `DataLoader` durante `for batch in loader` dentro de `run_epoch`.

---

## 13. ¿Qué son los patches del "punto 1" y del "punto 3"?

**Respuesta:**

- Punto 1: patches planificados (metadatos). En `self.items` se guardan rutas y centros, pero no los voxels.
- Punto 3: patch real recortado. Ocurre en `__getitem__`, donde se extrae el cubo desde el volumen usando el centro.

---

## 14. ¿Cómo calcula el centro de los patches aleatorios?

**Respuesta:** Se muestrea uniforme dentro de límites válidos para que el patch no se salga del volumen:

- `half = patch_size // 2`
- `cx ~ randint(half, max_x + 1)`
- `cy ~ randint(half, max_y + 1)`
- `cz ~ randint(half, max_z + 1)`

No usa la máscara para evitar nódulos; pueden caer sobre nódulo por casualidad.

---

## 15. ¿Cuál es el tamaño del patch? ¿En la lista solo se guardan rutas y centros?

**Respuesta:** El tamaño es global (`patch_size`), por defecto 64, así que cada patch es `64x64x64`. En `self.items` solo se guardan rutas, `patient_id` y centro `(cx, cy, cz)`. El contenido del patch se recorta al vuelo en `__getitem__`.

---

## 16. ¿Cuándo entran en juego DataLoader y `__getitem__`?

**Respuesta:**

1. Se crea el dataset (`FullVolumeNoduleDataset`) y se planifican muestras.
2. Se crea el `DataLoader`.
3. En entrenamiento/evaluación, al iterar `for batch in loader`, el `DataLoader` llama a `__getitem__` para cada índice del batch.

---

## 17. ¿En un epoch hay varios batches? ¿Cada batch son patches aleatorios?

**Respuesta:** Sí, un epoch contiene muchos batches. Cada batch tiene `batch_size` patches. En este script los centros aleatorios se generan al construir el dataset (no se re-muestrean cada epoch); lo que sí cambia en train es la augmentación aleatoria.

---

## 18. ¿1206 eran patches o batches?

**Respuesta:** En el ejemplo con 603 pacientes, `patches_per_patient=8` y `batch_size=4`:

- Patches por epoch: `603 * 8 = 4824`
- Batches por epoch: `4824 / 4 = 1206`

Por tanto, 1206 son batches.

---

## 19. ¿Qué es un batch?

**Respuesta:** Un batch es un grupo de muestras procesadas juntas en una iteración. Aquí, una muestra es un patch 3D (`image`, `mask`), y `batch_size` indica cuántos patches se usan a la vez para calcular la loss y actualizar pesos.

---

## 20. ¿Por qué extrae patches si se entrena con CT completo?

**Respuesta:** Se parte de volúmenes CT completos por paciente, pero el modelo entrena con subvolúmenes (patches) para que el coste en memoria/cómputo sea viable y para balancear mejor nódulo/fondo.

---

## 21. ¿En predicción se pasa CT completo o patches?

**Respuesta:** Conceptualmente se predice sobre CT completo. En práctica, suele hacerse por ventanas (patches) y luego se reconstruye la máscara full-size, normalmente con ventana deslizante (sliding window) y fusión en zonas solapadas.

---

## 22. ¿El muestreo de este script es sliding window o aleatorio?

**Respuesta:** Es muestreo de patches al vuelo con centros positivos (bbox) + centros aleatorios. No es sliding window sistemático.

---

## 23. ¿Qué es un canal y por qué `DoubleConv3D(in_channels, base_channels)`?

**Respuesta:** Un canal es un mapa de características. `in_channels` es cuántos canales entran al bloque y `base_channels` cuántos salen. En CT se suele entrar con 1 canal (intensidad HU) y salir con más canales para aprender distintos patrones.

---

## 24. ¿Por qué CT en HU tiene 1 canal y por qué salida inicial 16 canales?

**Respuesta:** El CT en HU es una magnitud escalar por voxel, por eso 1 canal. El valor 16 es un hiperparámetro típico de compromiso entre capacidad representacional y consumo de memoria en 3D.

---

## 25. ¿Por qué los canales se multiplican por 2 (2, 4, 8...)?

**Respuesta:** Es una convención de U-Net: al bajar resolución espacial, se aumenta capacidad en canales para mantener información semántica. No es obligatorio; puede ajustarse según memoria y rendimiento.

---

## 26. ¿Qué hace `MaxPool3d`?

**Respuesta:** Hace downsampling 3D tomando máximos locales. Con `kernel_size=2` (y `stride=2` implícito) reduce `D/H/W` a la mitad y mantiene canales. Reduce coste y amplía campo receptivo efectivo.

---

## 27. ¿Qué es la B en `[B, C, D, H, W]`?

**Respuesta:** `B` es el batch size: número de muestras (patches) procesadas simultáneamente.

---

## 28. ¿Qué hace exactamente `forward` y cuándo se usa?

**Respuesta:** `forward` define el recorrido de datos por la red (encoder, bottleneck, decoder, salida). Se ejecuta cada vez que se llama `model(x)`, tanto en train como en validación/test.

---

## 29. ¿Por qué encoder baja resolución y sube canales, y decoder hace lo contrario?

**Respuesta:** Encoder prioriza contexto semántico global (más abstracto, menos resolución). Decoder recupera detalle espacial para segmentar voxel a voxel (más resolución, menos compresión en canales).

---

## 30. ¿Qué significa concatenar skips del encoder para recuperar detalle fino?

**Respuesta:** Al hacer pooling se pierde precisión espacial. Las conexiones skip llevan features de alta resolución del encoder al decoder (`cat([d*, e*])`) para combinar contexto profundo con bordes/localización fina y mejorar la máscara final.






