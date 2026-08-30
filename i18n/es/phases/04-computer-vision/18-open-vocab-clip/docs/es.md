# Visión de vocabulario abierto  CLIP

> Entrenar un codificador de imágenes y un codificador de texto juntos para que los pares de coincidencias (imagen, leyenda) aterricen en el mismo punto en un espacio compartido. Ese es todo el truco.

**Type:** Build + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 14 (ViT), Phase 4 Lesson 17 (Self-Supervised)
**Time:** ~45 minutes

## Objetivos de aprendizaje

- Explicar la arquitectura de dos torres del CLIP y el objetivo de formación contrastada
- Utilice un CLIP (o SigLIP) pre-entrenado para la clasificación de tiro cero sin ningún entrenamiento específico de tarea
- Implementar la clasificación de tiro cero desde cero: codificar las instrucciones de clase, calcular la similitud cosina, tomar argmax
- Distinguir entre los modelos de visión CLIP, SigLIP, OpenCLIP y LLaVA/LLaMA  para qué se destina cada uno en 2026

## El problema

Los clasificadores tradicionales son un vocabulario cerrado: un modelo de 1000 clases de ImageNet solo puede predecir 1000 etiquetas.

CLIP (Radford et al., OpenAI 2021) mostró que el entrenamiento en 400M (imagen, leyenda) pares arrancados de la web produce un modelo que puede clasificar en cualquier conjunto de categorías a la inferencia, descrito puramente en lenguaje natural.

Esta capacidad  transferencia de disparos cero  es la razón por la que cada sistema de visión moderno comienza con un punto de control de la familia CLIP. La detección (Grounding DINO, OWL-ViT), la segmentación (CLIPSeg, SAM), la recuperación, la moderación de contenido, VLMs y la generación de texto a imagen se basan en embebedidos conjuntos de estilo CLIP.

## El concepto

### Dos torres

```mermaid
flowchart LR
    IMG["Image"] --> IENC["Image encoder<br/>(ViT-L/14)"] --> IEMB["Image embedding<br/>(1024,)"]
    TXT["Caption"] --> TENC["Text encoder<br/>(transformer)"] --> TEMB["Text embedding<br/>(1024,)"]
    IEMB --> SIM["Cosine similarity"]
    TEMB --> SIM

    style IENC fill:#dbeafe,stroke:#2563eb
    style TENC fill:#fef3c7,stroke:#d97706
    style SIM fill:#dcfce7,stroke:#16a34a
```

Ambos codificadores terminan con una proyección lineal a la misma dimensión de incorporación (512 para CLIP-B/32, 1024 para CLIP-L/14).

### El objetivo

Dado un lote de pares N (imagen, leyenda) construye una matriz de similitud NxN. Entrenar ambos codificadores para que la diagonal (pares coincidentes) tenga una alta similitud y los fuera de diagonales (no coincidentes) tengan una baja similitud.

```
sim_matrix = image_embeddings @ text_embeddings.T / tau

loss_i2t = cross_entropy(sim_matrix,       targets=arange(N))
loss_t2i = cross_entropy(sim_matrix.T,     targets=arange(N))
loss = (loss_i2t + loss_t2i) / 2
```

Simétrico porque tanto la extracción de imagen a texto como de texto a imagen deberían funcionar. `tau`(temperatura) se aprende típicamente como un parámetro escalar, iniciado en 0,07.

### Siglip: una mejor pérdida

SigLIP (Zhai et al., 2023) sustituyó la softmax por sigmoide por pareja:

```
loss = mean over pairs of log(1 + exp(-y_ij * sim_ij))
y_ij = +1 if matching, -1 otherwise
```

La pérdida por pareja elimina la normalización a nivel de lote que requiere CLIP. SigLIP se entraña mejor en pequeños lotes y coincide o supera CLIP en datos iguales.

### Clasificación de tiro cero

Dado un CLIP capacitado:

1. Para cada clase, compón una invitación: "una foto de una {clase}".
2. Encienda todas las instrucciones de clase con el codificador de texto -> `T`forma (C, d).
3. Encifrar la imagen de prueba -> `I`forma (1, d).
4. Similaridad = `I @ T.T`forma (1, C).
5. Argmax -> clase prevista.

Las preguntas de ingeniería de la instantánea. OpenAI publicó 80 plantillas de la instantánea para ImageNet ("una foto de un {}", "una foto borrosa de un {}", "un boceto de un {}", ...).

### En el que se utilicen modelos CLIP en 2026

- **Zero-shot classification** uso directo.
- **Image retrieval** codificar todas las imágenes una vez, incrustar la consulta en la inferencia.
- **Text-conditioned detection** Al aterrizar DINO, OWL-ViT envuelve una torre de texto CLIP alrededor de un detector.
- **Text-conditioned segmentation** CLIPSeg; SAM utiliza entradas de texto a través de CLIP.
- **VLMs** LLaVA, Qwen-VL, InternVL incorporan un codificador de visión de la familia CLIP en un LLM.
- **Text-to-image gen** Difusión estable, condición DALL-E 3 en las incorporaciones de texto CLIP.

Una vez que tienes un espacio de incorporación compartido, cada tarea de visión+idioma se convierte en un cálculo de distancia.

```figure
clip-contrastive
```

## Construye el mismo

### Paso 1: Un pequeño modelo de dos torres

Para esta lección las torres son pequeñas MLP sobre las características pre-extractadas para que la señal de entrenamiento sea visible en la CPU.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    def __init__(self, img_in=128, txt_in=64, emb=64):
        super().__init__()
        self.image_proj = nn.Sequential(nn.Linear(img_in, 128), nn.ReLU(), nn.Linear(128, emb))
        self.text_proj = nn.Sequential(nn.Linear(txt_in, 128), nn.ReLU(), nn.Linear(128, emb))
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # ln(1/0.07)

    def forward(self, img_feats, txt_feats):
        i = F.normalize(self.image_proj(img_feats), dim=-1)
        t = F.normalize(self.text_proj(txt_feats), dim=-1)
        return i, t, self.logit_scale.exp()
```

Dos proyecciones, salida compartida, temperatura aprendida, la misma forma que la API CLIP real.

### Paso 2: pérdida de contraste

```python
def clip_loss(image_emb, text_emb, logit_scale):
    N = image_emb.size(0)
    sim = logit_scale * image_emb @ text_emb.T
    targets = torch.arange(N, device=sim.device)
    l_i = F.cross_entropy(sim, targets)
    l_t = F.cross_entropy(sim.T, targets)
    return (l_i + l_t) / 2
```

Simétrico. Escala de logit_más alta = más afilada softmax = más seguro pero riesgo de inestabilidad.

### Paso 3: Clasificador de disparos cero

```python
@torch.no_grad()
def zero_shot_classify(model, image_feats, class_text_feats, class_names):
    """
    image_feats:      (N, img_in)
    class_text_feats: (C, txt_in)   one averaged embedding per class
    """
    i = F.normalize(model.image_proj(image_feats), dim=-1)
    t = F.normalize(model.text_proj(class_text_feats), dim=-1)
    sim = i @ t.T
    pred = sim.argmax(dim=-1)
    return [class_names[p] for p in pred.tolist()]
```

Una línea por paso. Este es el procedimiento exacto de tiro cero utilizado con un punto de control CLIP de producción.

### Paso 4: Verificación de la salud mental

```python
torch.manual_seed(0)
model = TwoTower()

img = torch.randn(8, 128)
txt = torch.randn(8, 64)
i, t, scale = model(img, txt)
loss = clip_loss(i, t, scale)
print(f"batch size: {i.size(0)}   loss: {loss.item():.3f}")
```

La pérdida debe estar cerca de `log(N) = log(8) = 2.08`para un modelo iniciado al azar  el objetivo de entropía cruzada simétrica cuando aún no se aprende estructura.

## Usalo

OpenCLIP es el estándar de la comunidad en 2026:

```python
import open_clip
import torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image = preprocess(Image.open("dog.jpg")).unsqueeze(0)
text = tokenizer(["a photo of a dog", "a photo of a cat", "a photo of a car"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print(probs)
```

SigLIP es más nuevo, se entrena mejor a pequeñas escalas y se prefiere para nuevos trabajos: `google/siglip-base-patch16-224`Abrazando a las dos naves.

## Envío

Esta lección produce:

- `outputs/prompt-zero-shot-class-picker.md` un prompt que diseña plantillas de clases para CLIP de tiro cero dado una lista de clases y un dominio.
- `outputs/skill-image-text-retriever.md` una habilidad que construye un índice de incorporación de imágenes con cualquier punto de control CLIP, soporta consulta por texto y consulta por imagen.

## Los ejercicios

1. **(Easy)**Utilice un OpenCLIP ViT-B/32 preentrenado y haga una clasificación de tiro cero en CIFAR-10 con el conjunto de instrucciones de 80 plantillas.
2. **(Medium)**Comparar una sola plantilla ("una foto de un {}") con 80 plantillas promediadas de la misma tarea CIFAR-10. Cuantifique la brecha y explique por qué las plantillas ayudan.
3. **(Hard)**Construir un índice de recuperación de imágenes de cero disparos: embebar 1.000 imágenes con CLIP, crear un índice FAISS, consulta con una descripción de lenguaje natural.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Two-tower | "Dual encoder" | Separate image and text encoders ending in a shared-dim projection head |
| Zero-shot | "No task-specific training" | Classify into classes described only by text at inference; no labels touched |
| Temperature / logit_scale | "tau" | Learned scalar that scales the similarity matrix before softmax |
| Prompt template | "A photo of a {}" | Natural-language wrapper around class names; averaging many templates boosts zero-shot accuracy |
| CLIP | "Image+text model" | The 2021 OpenAI model; vocabulary of the field in 2026 |
| SigLIP | "Sigmoid CLIP" | Swaps softmax for per-pair sigmoid; trains better at small batches |
| OpenCLIP | "Open reproduction" | Community-trained CLIP variants on LAION; production default for open-source pipelines |
| VLM | "Vision-language model" | A CLIP-family encoder plus an LLM, trained to answer questions about images |

## Leer más

- [CLIP: Learning Transferable Visual Models from Natural Language Supervision (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [SigLIP: Sigmoid Loss for Language-Image Pre-Training (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343)
- [OpenCLIP](https://github.com/mlfoundations/open_clip) la base de código de la comunidad
- [DINOv2 vs CLIP vs MAE: a features comparison](https://huggingface.co/blog/dinov2) Guía de HF con casos de uso paralelos
