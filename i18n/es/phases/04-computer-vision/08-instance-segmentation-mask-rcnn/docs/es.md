# Segmentación de instancia  Máscara R-CNN

> Añadir una pequeña rama de máscara a un detector R-CNN más rápido y tienes segmentación de instancia. La parte difícil es RoIAlign, y es más difícil de lo que parece.

**Type:** Build + Learn
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06 (YOLO), Phase 4 Lesson 07 (U-Net)
**Time:** ~75 minutes

## Objetivos de aprendizaje

- Rastrear la arquitectura de la máscara R-CNN de extremo a extremo: columna vertebral, FPN, RPN, RoIAlign, cabeza de caja, cabeza de máscara
- Implementar RoIAlign desde cero y explicar por qué ya no se utiliza RoIPool
- Usa la visión de la antorcha `maskrcnn_resnet50_fpn_v2`modelo pre-entrenado para máscaras de instancia de calidad de producción y lectura correcta de su formato de salida
- La capa de máscara R-CNN de ajuste fino en un pequeño conjunto de datos personalizado mediante la sustitución de la caja y las cabezas de máscara y la congelación de la columna vertebral

## El problema

La segmentación semántica le da una máscara por clase. La segmentación de instancia le da una máscara por objeto, incluso cuando dos objetos comparten una clase. Contar individuos, rastrear los marcos y medir cosas (la caja de límite de cada ladrillo en una pared, cada célula en una imagen del microscopio) todo requiere segmentación de instancia.

Mask R-CNN (He et al., 2017) resolvió esto reformando la segmentación de instancia como detección-más-una-máscara. El diseño fue tan limpio que durante los próximos cinco años casi todos los papeles de segmentación de instancia fueron una variante de Mask R-CNN, y la implementación de torchvision sigue siendo el estándar de producción para conjuntos de datos pequeños a medianos.

El problema difícil de la ingeniería es la muestreo: ¿cómo extraer una región de características de tamaño fijo de una caja de propuestas cuyos rincones no se alinean con los límites de píxeles?

## El concepto

### La arquitectura

```mermaid
flowchart LR
    IMG["Input"] --> BB["ResNet<br/>backbone"]
    BB --> FPN["Feature<br/>Pyramid Network"]
    FPN --> RPN["Region<br/>Proposal<br/>Network"]
    FPN --> RA["RoIAlign"]
    RPN -->|"top-K proposals"| RA
    RA --> BH["Box head<br/>(class + refine)"]
    RA --> MH["Mask head<br/>(14x14 conv)"]
    BH --> NMS["NMS"]
    MH --> NMS
    NMS --> OUT["boxes +<br/>classes + masks"]

    style BB fill:#dbeafe,stroke:#2563eb
    style FPN fill:#fef3c7,stroke:#d97706
    style RPN fill:#fecaca,stroke:#dc2626
    style OUT fill:#dcfce7,stroke:#16a34a
```

Cinco piezas para entender:

1. **Backbone** ResNet-50 o ResNet-101 entrenado en ImageNet. Produce una jerarquía de mapas de características en los pasos 4, 8, 16, 32.
2. **FPN (Feature Pyramid Network)** Conexiones laterales de arriba hacia abajo que dan a cada nivel C canales de características ricas en semántica.
3. **RPN (Region Proposal Network)** una pequeña cabeza de concha que, en cada posición de anclaje, predice "¿hay un objeto aquí?" y "cómo refinar la caja?". Produce ~ 1000 propuestas por imagen.
4. **RoIAlign** muestra un parche de tamaño fijo (por ejemplo, 7x7) de cualquier caja en cualquier nivel de FPN.
5. **Heads** cabezal de caja de dos capas que refina la caja y elige una clase, más una pequeña cabeza de caja que saca una `28x28`mascarilla binaria para cada propuesta.

### ¿Por qué RoIAlign y no RoIPool?

El R-CNN rápido original utilizó RoIPool, que divide una caja de propuestas en una cuadrícula, toma la característica máxima en cada célula y redondea todas las coordenadas a números enteros.

```
RoIPool:
  box (34.7, 51.3, 98.2, 142.9)
  round -> (34, 51, 98, 142)
  split grid -> round each cell boundary
  misalignment accumulates at every step

RoIAlign:
  box (34.7, 51.3, 98.2, 142.9)
  sample at exact float coordinates using bilinear interpolation
  no rounding anywhere
```

RoIAlign aumenta la máscara AP en 3-4 puntos en COCO de forma gratuita.

### El RPN en un párrafo

En cada posición de un mapa de características, coloque cajas de anclaje K de diferentes tamaños y formas. Prevé un puntaje de objetividad para cada ancla y una compensación de regresión para convertir el anclaje en una caja más adecuada. Mantenga las cajas superiores por puntaje, aplique NMS en IoU 0.7, y entregue a los supervivientes a las cabezas. El RPN está entrenado con su propia mini-pérdida  la misma estructura que la pérdida de YOLO de la Lección 6, sólo con dos clases (objeto / ningún objeto).

### La cabeza de la máscara

Para cada propuesta (después de RoIAlign) la cabeza de la máscara es una pequeña FCN: cuatro convases 3x3, una deconv 2x, una final 1x1 conv que produce `num_classes`canales de salida en `28x28`Resolución. Sólo se mantiene el canal correspondiente a la clase prevista; los demás se ignoran. Esto desacopla la predicción de la máscara de la clasificación.

Muestre la máscara de 28x28 al tamaño original de píxeles de la propuesta para producir la máscara binaria final.

### Las pérdidas

Mask R-CNN tiene cuatro pérdidas sumadas:

```
L = L_rpn_cls + L_rpn_box + L_box_cls + L_box_reg + L_mask
```

- `L_rpn_cls`¿ Qué ?`L_rpn_box` objetividad + recuento de regresión de las propuestas de RPN.
- `L_box_cls` entropía cruzada sobre las clases (C+1) (incluyendo el fondo) en el clasificador de la cabeza.
- `L_box_reg` L1 suave en el refinamiento de la caja de la cabeza.
- `L_mask` Entropia binaria cruzada por píxel en la salida de máscara 28x28.

Cada pérdida tiene su propio peso predeterminado; la implementación de torchvision las expone como argumentos de constructor.

### Formatos de salida

`torchvision.models.detection.maskrcnn_resnet50_fpn_v2`devuelve una lista de dicts, uno por imagen:

```
{
    "boxes":  (N, 4) in (x1, y1, x2, y2) pixel coordinates,
    "labels": (N,) class IDs, 0 = background so indices are 1-based,
    "scores": (N,) confidence scores,
    "masks":  (N, 1, H, W) float masks in [0, 1] — threshold at 0.5 for binary,
}
```

La máscara ya tiene resolución completa de imagen.

```figure
cv3-roialign-sampling
```

## Construye el mismo

### Paso 1: Alineación de la roya desde cero

Este es el componente de Mask R-CNN que es más fácil de entender como código que como prosa.

```python
import torch
import torch.nn.functional as F

def roi_align_single(feature, box, output_size=7, spatial_scale=1 / 16.0):
    """
    feature: (C, H, W) single-image feature map
    box: (x1, y1, x2, y2) in original image pixel coordinates
    output_size: side of the output grid (7 for box head, 14 for mask head)
    spatial_scale: reciprocal of the feature map stride
    """
    C, H, W = feature.shape
    x1, y1, x2, y2 = [c * spatial_scale - 0.5 for c in box]
    bin_w = (x2 - x1) / output_size
    bin_h = (y2 - y1) / output_size

    grid_y = torch.linspace(y1 + bin_h / 2, y2 - bin_h / 2, output_size)
    grid_x = torch.linspace(x1 + bin_w / 2, x2 - bin_w / 2, output_size)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")

    gx = 2 * (xx + 0.5) / W - 1
    gy = 2 * (yy + 0.5) / H - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    sampled = F.grid_sample(feature.unsqueeze(0), grid, mode="bilinear",
                            align_corners=False)
    return sampled.squeeze(0)
```

Cada número está en una posición de muestreo bilinear, sin redondeo, sin quantización, sin gradientes caídos.

### Paso 2: Compare con el RoIAlign de torchvision

```python
from torchvision.ops import roi_align

feature = torch.randn(1, 16, 50, 50)
boxes = torch.tensor([[0, 10, 20, 100, 90]], dtype=torch.float32)  # (batch_idx, x1, y1, x2, y2)

ours = roi_align_single(feature[0], boxes[0, 1:].tolist(), output_size=7, spatial_scale=1/4)
theirs = roi_align(feature, boxes, output_size=(7, 7), spatial_scale=1/4, sampling_ratio=1, aligned=True)[0]

print(f"shape ours:   {tuple(ours.shape)}")
print(f"shape theirs: {tuple(theirs.shape)}")
print(f"max|diff|:    {(ours - theirs).abs().max().item():.3e}")
```

Con `sampling_ratio=1`y `aligned=True`, los dos coinciden con dentro `1e-5`¿ Qué ?

### Paso 3: Cargar una máscara pre-entrenada R-CNN

```python
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
model.eval()
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
print(f"classes (including background): {len(model.roi_heads.box_predictor.cls_score.out_features * [0])}")
```

Los parámetros de 46M, 91 clases (COCO). La primera clase (id 0) es el fondo; todo lo que el modelo detecta realmente comienza en id 1.

### Paso 4: ejecutar inferencias

```python
with torch.no_grad():
    x = torch.randn(3, 400, 600)
    predictions = model([x])
p = predictions[0]
print(f"boxes:  {tuple(p['boxes'].shape)}")
print(f"labels: {tuple(p['labels'].shape)}")
print(f"scores: {tuple(p['scores'].shape)}")
print(f"masks:  {tuple(p['masks'].shape)}")
```

El tensor de la máscara es forma .`(N, 1, H, W)`. Umbral de 0,5 para obtener una máscara binaria por objeto:

```python
binary_masks = (p['masks'] > 0.5).squeeze(1)  # (N, H, W) boolean
```

### Paso 5: Cambiar las cabezas para un conteo de clases personalizado

La receta común de ajuste fino: reutilice la columna vertebral, FPN y RPN; reemplace las dos cabezas del clasificador.

```python
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def build_custom_maskrcnn(num_classes):
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

custom = build_custom_maskrcnn(num_classes=5)
print(f"custom cls_score.out_features: {custom.roi_heads.box_predictor.cls_score.out_features}")
```

`num_classes`debe incluir la clase de fondo, por lo que un conjunto de datos con 4 clases de objetos utiliza `num_classes=5`¿ Qué ?

### Paso 6: Congela lo que no necesita entrenamiento

En conjuntos de datos pequeños, congela la columna vertebral y el FPN. Sólo la objetividad RPN + regresión y las dos cabezas aprenden.

```python
def freeze_backbone_and_fpn(model):
    # torchvision Mask R-CNN packs the FPN inside `model.backbone` (as
    # `model.backbone.fpn`), so iterating `model.backbone.parameters()` covers
    # both the ResNet feature layers and the FPN lateral/output convs.
    for p in model.backbone.parameters():
        p.requires_grad = False
    return model

custom = freeze_backbone_and_fpn(custom)
trainable = sum(p.numel() for p in custom.parameters() if p.requires_grad)
print(f"trainable after freeze: {trainable:,}")
```

En conjuntos de datos de 500 imágenes esta es la diferencia entre convergencia y sobreajuste.

## Usalo

El ciclo de entrenamiento completo de Mask R-CNN en torchvision es de 40 líneas y no cambia significativamente entre las tareas  intercambiar conjuntos de datos y ir.

```python
def train_step(model, images, targets, optimizer):
    model.train()
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
    return {k: v.item() for k, v in loss_dict.items()}
```

El `targets`La lista debe tener dicts por imagen con `boxes`¿ Qué ?`labels`, y `masks`(como `(num_instances, H, W)`El modelo devuelve un dictado de cuatro pérdidas durante el entrenamiento y una lista de predicciones durante la evaluación, con teclas en `model.training`¿ Qué ?

El `pycocotools`evaluador produce mAP@IoU=0.5:0.95 tanto para cajas como para máscaras; necesita ambos números para saber si la cabeza de caja o la cabeza de máscara es el cuello de botella.

## Envío

Esta lección produce:

- `outputs/prompt-instance-vs-semantic-router.md` un mensaje que hace tres preguntas y elige instancia vs semántica vs panóptica más el modelo exacto para comenzar.
- `outputs/skill-mask-rcnn-head-swapper.md` una habilidad que genera las 10 líneas de código para intercambiar cabezas en cualquier modelo de detección de torchvision, dado el nuevo `num_classes`¿ Qué ?

## Los ejercicios

1. **(Easy)**Verifique su línea de trabajo con `torchvision.ops.roi_align`En 100 cajas aleatorias. Informar la diferencia absoluta máxima. También ejecuta RoIPool (comportamiento pre-2017) y muestre que se desvía por ~1-2 píxeles de mapa de características en cajas cerca de la frontera.
2. **(Medium)**- No . - ¿ Qué ?`maskrcnn_resnet50_fpn_v2`En un conjunto de datos personalizados de 50 imágenes (de cualquier clase: globos, peces, agujeros, logotipos).
3. **(Hard)**Reemplaza la cabeza de máscara de Mask R-CNN por una que predica a 56x56 en lugar de 28x28. Mide mAP@IoU = 0,75 antes y después. Explique por qué la ganancia (o falta de una) coincide con el límite de precisión / memoria esperado.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Mask R-CNN | "Detection plus masks" | Faster R-CNN + a small FCN head that predicts a 28x28 mask per proposal per class |
| FPN | "Feature pyramid" | Top-down + lateral connections that give every stride level C channels of semantic-rich features |
| RPN | "Region proposer" | A small conv head that produces ~1000 object/no-object proposals per image |
| RoIAlign | "No-rounding crop" | Bilinearly samples a fixed-size feature grid from any float-coordinate box |
| RoIPool | "Pre-2017 crop" | Same purpose as RoIAlign but rounds box coordinates; obsolete |
| Mask AP | "Instance mAP" | Average precision computed with mask IoU instead of box IoU; the COCO instance segmentation metric |
| Binary mask head | "Per-class mask" | Predicts one binary mask per class for each proposal; only the predicted class's channel is kept |
| Background class | "Class 0" | The catch-all "no object" class; indices for real classes start at 1 |

## Leer más

- [Mask R-CNN (He et al., 2017)](https://arxiv.org/abs/1703.06870) el documento; la sección 3 sobre RoIAlign es la lectura crítica
- [FPN: Feature Pyramid Networks (Lin et al., 2017)](https://arxiv.org/abs/1612.03144) el papel FPN; todos los detectores modernos lo usan
- [torchvision Mask R-CNN tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) la referencia para el circuito de ajuste fino
- [Detectron2 model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) Implementaciones de producción con pesas entrenadas para casi todas las variantes de detección y segmentación
