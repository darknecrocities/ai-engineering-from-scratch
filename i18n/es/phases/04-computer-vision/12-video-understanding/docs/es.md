# El video Comprensión  Modelado temporal

> Un video es una secuencia de imágenes más la física que las conecta. Cada modelo de video trata el tiempo como un eje extra (3D conv), una secuencia para atender (transformador), o una función para extraer una vez y un grupo (2D + grupo).

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 4 Lesson 04 (Image Classification)
**Time:** ~45 minutes

## Objetivos de aprendizaje

- Distinguir los tres enfoques principales de modelado de vídeo (2D+pool, 3D conv, transformador espacio-temporal) y predecir sus costos y correcciones de precisión
- Implementar muestreo de marco, agrupación temporal y clasificador de línea de base 2D+pool en PyTorch
- Explica por qué los núcleos 3D "inflados" de I3D transfieren bien de los pesos de ImageNet y qué hace un conv factorizado (2+1)D de manera diferente
- Lea los conjuntos de datos y métricas estándar de reconocimiento de acción: Kinetics-400/600, UCF101, Something-Something V2; precisión superior a 1 en el nivel de clip y video

## El problema

Un video de 30 segundos a 30 fps es 900 imágenes. Ingenuamente, la clasificación de video es una clasificación de imágenes ejecutada 900 veces seguido de algún tipo de agregación. Eso funciona cuando la acción es visible en casi todos los cuadros (deportes, cocina, videos de ejercicio) y falla gravemente cuando la acción se define por el movimiento en sí: "empujando algo de izquierda a derecha" se ve como dos objetos fijos en cada cuadro.

La pregunta central para cada arquitectura de vídeo es: cuándo se modela la estructura temporal y cómo? La respuesta impulsa todo lo demás  costo de cálculo, estrategia de preentrenamiento, si se pueden reutilizar pesos de ImageNet, qué conjuntos de datos el modelo se entrenan.

Esta lección es deliberadamente más corta que las lecciones de imagen estática. La maquinaria central de la imagen ya está en su lugar, y la comprensión de vídeo se trata principalmente de la historia temporal: muestreo, modelado y agregado.

## El concepto

### Las tres familias arquitectónicas

```mermaid
flowchart LR
    V["Video clip<br/>(T frames)"] --> A1["2D + pool<br/>run 2D CNN per frame,<br/>average over time"]
    V --> A2["3D conv<br/>convolve over<br/>T x H x W"]
    V --> A3["Spatio-temporal<br/>transformer<br/>attention over<br/>(t, h, w) tokens"]

    A1 --> C["Logits"]
    A2 --> C
    A3 --> C

    style A1 fill:#dbeafe,stroke:#2563eb
    style A2 fill:#fef3c7,stroke:#d97706
    style A3 fill:#dcfce7,stroke:#16a34a
```

### 2D + piscina

Tomemos una CNN 2D (ResNet, EfficientNet, ViT). ejecutarla de forma independiente en cada marco muestrado. promedio (o máximo-pool, o pool de atención) las incorporaciones por marco.

Los pros:
- ImageNet traslada directamente a los estudiantes.
- Es más simple de implementar.
- Baratos: T marcos * costo de inferencia de una sola imagen.

Las ventajas:
- No puedo modelar movimiento.
- El agrupamiento temporal es invariable según el orden; "puerta abierta" y "puerta cerrada" se ven iguales.

Cuándo utilizar: tareas pesadas en apariencia, transferencia de aprendizaje en pequeños conjuntos de datos de vídeo, líneas de base iniciales.

### Convolturas en 3D

Los núcleos 2D (H, W) se sustituyen por núcleos 3D (T, H, W). La red se envuelve tanto en el espacio como en el tiempo.

Triko I3D: toma un modelo preentrenado de 2D ImageNet, "infla" cada núcleo 2D copiándolo a lo largo de un nuevo eje de tiempo. Un conv 3x3 2D se convierte en un conv 3x3x3 3D. Esto le da al modelo 3D pesos preentrenados fuertes en lugar de entrenar desde cero.

Los pros:
- Modela directamente el movimiento.
- La inflación I3D ofrece aprendizaje gratuito.

Las ventajas:
- T/8 más FLOPs que la contraparte 2D (para el núcleo temporal de 3 apilados 3 veces).
- Los núcleos temporales son pequeños; el movimiento a largo alcance requiere un enfoque de pirámide o doble corriente.

Cuando utilizar: reconocimiento de acción donde el movimiento es la señal (Algo-Algo V2, Kinética con clases de movimiento pesado).

### Transformadores espaciotemporales

Marca el video en una cuadrícula de parches espacio-tiempo y atenta a todos ellos.

Los patrones de atención que importan:
- **Joint** una gran atención sobre (t, h, w). Cuadrado en `T*H*W`Es muy caro.
- **Divided** dos atenciones por bloque: una en el tiempo, otra en el espacio.
- **Factorised** La atención temporal se alterna con la atención espacial a través de los bloques.

Los pros:
- Precisión SOTA en todos los principales indicadores de referencia.
- Transferencias desde transformadores de imagen (ViT) a través de la inflación de parches.
- Soporta videos de largo contexto a través de poca atención.

Las ventajas:
- - Con hambre de computación.
- Requiere una elección cuidadosa de patrones de atención o globos de tiempo de ejecución.

Cuándo utilizar: grandes conjuntos de datos, comprensión de vídeo de alta fidelidad, tareas de vídeo + texto multimodales.

### Muestreo de los cuadros

Un clip de 10 segundos a 30 fps es de 300 cuadros; alimentar a todos los 300 a cualquier modelo es un desperdicio.

- **Uniform sampling** Seleccione los marcos T uniformemente a través del clip.
- **Dense sampling** ventana de marco T contiguo aleatorio. común para convistas 3D porque el movimiento requiere de marcos vecinos.
- **Multi-clip** muestra varias ventanas de marco T del mismo video, clasifica cada una, predicciones promedio en el momento de la prueba.

T es generalmente 8, 16, 32 o 64. T más alto = más señal temporal en más cálculo.

### Evaluación

Dos niveles:
- **Clip-level accuracy** modelo ve un clip de marco T, informa top-k.
- **Video-level accuracy** predicciones medias de nivel de clip en múltiples clips por video; más alto y más estable.

Siempre informe ambos. Un modelo que obtiene un puntaje de 78% clip / 82% de video depende en gran medida del promedio de tiempo de prueba; uno que obtiene un puntaje de 80% / 81% es más robusto por clip.

### Datasets que se reunirán

- **Kinetics-400 / 600 / 700** el conjunto de datos de acción de propósito general. 400k clips; URL de YouTube (muchos ya muertos).
- **Something-Something V2** Acciones definidas por movimiento ("movimiento de X de izquierda a derecha"). No se pueden resolver por 2D + pool.
- **UCF-101**¿ Qué ?**HMDB-51** mayor, menor, aún reportado.
- **AVA** acción *localización* en el espacio y el tiempo; más difícil que la clasificación.

```figure
v4-video-temporal
```

## Construye el mismo

### Paso 1: Muestra de marco

Muestras uniformes y densas que trabajan en una lista de marcos (o un tensor de vídeo).

```python
import numpy as np

def sample_uniform(num_frames_total, T):
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    step = num_frames_total / T
    return [int(i * step) for i in range(T)]


def sample_dense(num_frames_total, T, rng=None):
    rng = rng or np.random.default_rng()
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    start = int(rng.integers(0, num_frames_total - T + 1))
    return list(range(start, start + T))
```

Ambos regresan .`T`Indices que se utilizan para cortar el tensor de vídeo.

### Paso 2: Una línea de base 2D+pool

Ejecutar un ResNet-18 2D sobre cada fotograma, características de la piscina promedio, clasificar.

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FramePool(nn.Module):
    def __init__(self, num_classes=400, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*(list(backbone.children())[:-1]))  # global avg pool kept
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (N, T, 3, H, W)
        N, T = x.shape[:2]
        x = x.view(N * T, *x.shape[2:])
        feats = self.features(x).view(N, T, -1)
        pooled = feats.mean(dim=1)
        return self.head(pooled)

model = FramePool(num_classes=10)
x = torch.randn(2, 8, 3, 224, 224)
print(f"output: {model(x).shape}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
```

Once millones de parámetros, ImageNet pre-entrenado, se ejecuta por marco, promedios, clasifica. Esta línea de base a menudo está dentro de 5-10 puntos de los modelos 3D adecuados en tareas pesadas en apariencia  a veces mejor, porque reutiliza una columna vertebral ImageNet más fuerte.

### Paso 3: Un enchufe 3D inflado de estilo I3D

Convierta una sola convalescencia 2D en una convalescencia 3D repitiendo pesos a lo largo de un nuevo eje de tiempo.

```python
def inflate_2d_to_3d(conv2d, time_kernel=3):
    out_c, in_c, kh, kw = conv2d.weight.shape
    weight_3d = conv2d.weight.data.unsqueeze(2)  # (out, in, 1, kh, kw)
    weight_3d = weight_3d.repeat(1, 1, time_kernel, 1, 1) / time_kernel
    conv3d = nn.Conv3d(in_c, out_c, kernel_size=(time_kernel, kh, kw),
                        padding=(time_kernel // 2, conv2d.padding[0], conv2d.padding[1]),
                        stride=(1, conv2d.stride[0], conv2d.stride[1]),
                        bias=False)
    conv3d.weight.data = weight_3d
    return conv3d

conv2d = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
conv3d = inflate_2d_to_3d(conv2d, time_kernel=3)
print(f"2D weight shape:  {tuple(conv2d.weight.shape)}")
print(f"3D weight shape:  {tuple(conv3d.weight.shape)}")
x = torch.randn(1, 3, 8, 56, 56)
print(f"3D output shape:  {tuple(conv3d(x).shape)}")
```

La división por `time_kernel`mantiene las magnitudes de activación aproximadamente constantes  importantes para no romper las estadísticas de la norma de lote en el primer pase.

### Paso 4: Convolución de la D (2+1)

Dividir una convección 3D en una convección 2D (espacial) y una convección 1D (temporal). El mismo campo receptivo, menos parámetros, mejor precisión en algunos puntos de referencia.

```python
class Conv2Plus1D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        mid_c = (in_c * out_c * kernel_size * kernel_size * kernel_size) \
                // (in_c * kernel_size * kernel_size + out_c * kernel_size)
        self.spatial = nn.Conv3d(in_c, mid_c, kernel_size=(1, kernel_size, kernel_size),
                                 padding=(0, kernel_size // 2, kernel_size // 2), bias=False)
        self.bn = nn.BatchNorm3d(mid_c)
        self.act = nn.ReLU(inplace=True)
        self.temporal = nn.Conv3d(mid_c, out_c, kernel_size=(kernel_size, 1, 1),
                                  padding=(kernel_size // 2, 0, 0), bias=False)

    def forward(self, x):
        return self.temporal(self.act(self.bn(self.spatial(x))))

c = Conv2Plus1D(3, 64)
x = torch.randn(1, 3, 8, 56, 56)
print(f"(2+1)D output: {tuple(c(x).shape)}")
```

Una red completa R(2+1)D es la misma que una ResNet-18 con cada 3x3 conv sustituido por `Conv2Plus1D`¿ Qué ?

## Usalo

Dos bibliotecas cubren el trabajo de vídeo de producción:

- `torchvision.models.video` R(2+1) D, MViT, Swin3D con pesos de cinética preentrenados. La misma API que los modelos de imagen.
- `pytorchvideo`(Meta)  modelo de zoológico, cargadores de datos para Kinética / SSv2 / AVA, transformaciones estándar.

Para los modelos de vídeo en lenguaje de visión (capción de vídeo, evaluación de calidad de vídeo), utilice `transformers`(El artículo`VideoMAE`¿ Qué ?`VideoLLaMA`¿ Qué ?`InternVideo`¿Qué es lo que se hace?

## Envío

Esta lección produce:

- `outputs/prompt-video-architecture-picker.md` un prompt que selecciona el transformador 2D+pool / I3D / (2+1)D / basado en la apariencia frente al movimiento, el tamaño del conjunto de datos y el presupuesto de cálculo.
- `outputs/skill-frame-sampler-auditor.md` una habilidad que inspecciona el muestreo de una tubería de vídeo y señala errores comunes: índice de un solo, muestreo desigual cuando `num_frames < T`, falta de cultivo que preserve los aspectos, etc.

## Los ejercicios

1. **(Easy)**Computa FLOPs (aproximados) para FramePool con T=8 vs. un ResNet 3D de estilo I3D con T=8.
2. **(Medium)**Generar un conjunto de datos de vídeo sintético: bolas aleatorias que se mueven en direcciones aleatorias, etiquetadas por la dirección de movimiento ("izquierda a derecha", "derecha a izquierda", "diagonal hacia arriba"). Entrenar FramePool en él. Muestre que logra una precisión casi casual, demostrando que la apariencia sola es insuficiente para las tareas de movimiento.
3. **(Hard)**Construir un R(2+1) D-18 reemplazando cada Conv2d en un ResNet-18 con `Conv2Plus1D`. Infla los pesos de los primeros conjuntos de un ResNet-18 pre-entrenado por ImageNet.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| 2D + pool | "Per-frame classifier" | Run a 2D CNN on every sampled frame, average-pool features across time, classify |
| 3D convolution | "Spatio-temporal kernel" | Kernel that convolves over (T, H, W); can model motion natively |
| Inflation | "Lift 2D weights to 3D" | Initialise 3D conv weights by repeating a 2D conv's weights along the new time axis, then divide by kernel_T to preserve activation scale |
| (2+1)D | "Factorised conv" | Split 3D into 2D spatial + 1D temporal; fewer parameters, extra non-linearity between |
| Divided attention | "Time then space" | Transformer block with two attentions per layer: one over tokens at the same frame, one over tokens at the same position |
| Clip | "T-frame window" | A sampled subsequence of T frames; the unit a video model consumes |
| Clip vs video accuracy | "Two eval settings" | Clip = one sample per video, video = average across multiple sampled clips |
| Kinetics | "The ImageNet of video" | 400-700 action classes, 300k+ YouTube clips, the standard video pretraining corpus |

## Leer más

- [I3D: Quo Vadis, Action Recognition (Carreira & Zisserman, 2017)](https://arxiv.org/abs/1705.07750) introduce la inflación y el conjunto de datos de la cinética
- [R(2+1)D: A Closer Look at Spatiotemporal Convolutions (Tran et al., 2018)](https://arxiv.org/abs/1711.11248) con factorizado, todavía un fuerte punto de partida
- [TimeSformer: Is Space-Time Attention All You Need? (Bertasius et al., 2021)](https://arxiv.org/abs/2102.05095) el primer transformador de vídeo fuerte
- [VideoMAE (Tong et al., 2022)](https://arxiv.org/abs/2203.12602) autoencoder enmascarado pre-entrenamiento para video; receta dominante actual pre-entrenamiento
