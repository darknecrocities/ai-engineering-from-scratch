# Comprensión de los documentos y los OCR

> OCR es un sistema de tres etapas que detecta las cajas de texto, reconoce los caracteres y luego las coloca.

**Type:** Learn + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06 (Detection), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~45 minutes

## Objetivos de aprendizaje

- Reconocer la línea de OCR clásica (detect -> recognize -> layout) y las alternativas modernas de extremo a extremo (Donut, Qwen-VL-OCR)
- Implementar la pérdida de CTC (Clasificación temporal de conectividad) para la formación de secuencias de secuencias en OCR
- Utilice PaddleOCR o EasyOCR para el análisis de documentos de producción sin formación
- Distinguir entre OCR, análisis de diseño y comprensión de documentos  y elegir la herramienta correcta por tarea

## El problema

Las imágenes llenas de texto están en todas partes: recibos, facturas, documentos de identificación, libros escaneados, formularios, cuadros blancos, pancartas, capturas de pantalla. Extraer datos estructurados de ellos  no sólo los caracteres, sino "esta es la cantidad total"  es uno de los problemas de visión aplicada de mayor valor.

El campo se divide en tres capas de habilidad:

1. **OCR proper**: convertir los píxeles en texto.
2. **Layout parsing**: la salida de OCR de grupo en regiones (título, cuerpo, tabla, encabezado).
3. **Document understanding**: extraer campos estructurados ("factura_total = $42.50") del diseño.

Cada capa tiene enfoques clásicos y modernos, y la brecha entre "Quiero texto de una imagen" y "Necesito la cantidad total de este recibo" es mayor de lo que la mayoría de los equipos se dan cuenta.

## El concepto

### El oleoducto clásico

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **Text detection**produce cuadrilateros por línea o por palabra.
- **Recognition**cultiva cada región a una altura fija, ejecuta una CNN + BiLSTM + CTC para producir una secuencia de caracteres.
- **Layout**reconstruye el orden de lectura (de arriba a abajo, de izquierda a derecha para el latín; diferente para el árabe, japonés).

### CTC en un párrafo

El reconocimiento OCR produce una secuencia de longitud variable a partir de un mapa de características de longitud fija. CTC (Graves et al., 2006) le permite entrenar esto sin alineación a nivel de caracteres. El modelo saca una distribución sobre (vocab + blanco) en cada paso de tiempo; la pérdida CTC marginaliza sobre todas las alineaciones que se reducen al texto objetivo después de fusionar repeticiones y eliminar espacios en blanco.

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

La CTC es la razón por la que CRNN trabajó en 2015 y todavía entrena a la mayoría de los modelos de producción OCR en 2026.

### Modelos modernos de extremo a extremo

- **Donut**(Kim et al., 2022)  un codificador ViT + un decodificador de texto; lee una imagen y emite JSON directamente.
- **TrOCR** Decodificador de transformador ViT + para OCR de nivel de línea.
- **Qwen-VL-OCR / InternVL** modelos completos de lenguaje de visión ajustados para las tareas de OCR; mejor precisión en 2026 en documentos complejos.
- **PaddleOCR** el oleoducto clásico DB + CRNN en un paquete de producción maduro; todavía el caballo de trabajo de código abierto.

Los modelos de extremo a extremo necesitan más datos y cálculo, pero no se debe acumular errores en las tuberías de múltiples etapas.

### Parse de diseño

Para documentos estructurados, ejecuta un detector de diseño (LayoutLMv3, DocLayNet) que etiqueta cada región: Título, párrafo, figura, tabla, nota a pie de página.

Para los formularios, utilizar **Key-Value extraction**Los modelos (Donut para documentos ricos en visualización, LayoutLMv3 para escaneos simples) toman imágenes + texto detectado + posiciones y predicen pares de valores clave estructurados.

### Metricas de evaluación

- **Character Error Rate (CER)** Distancia de referencia Levenshtein / longitud. Más bajo es mejor. Objetivo de producción: < 2% en escáneres limpios.
- **Word Error Rate (WER)** igual en el nivel de las palabras.
- **F1 on structured fields** para las tareas de valor clave; medidas de si `{invoice_total: 42.50}`Parece correcto.
- **Edit distance on JSON** para el análisis de documentos de extremo a extremo; el papel Donut introdujo una distancia de edición de árboles normalizada.

```figure
cv3-ctc-collapse
```

## Construye el mismo

### Paso 1: pérdida de CTC + codificación codiciada

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss`El codificador codicioso es más simple que una búsqueda de haz y por lo general dentro del 1% de CER de ella.

### Paso 2: Reconocedor de CRNN pequeño

CNN + BiLSTM mínimo para la OCR de línea.

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

Entrada de altura fija (la CNN max-pools altura a 1).

### Paso 3: OCR sintético

Generar cuerdas de dígitos blanco-negro para una prueba de humo de extremo a extremo.

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

Un conjunto de datos OCR real añade fuentes, ruido, rotación, borrado y color.

### Paso 4: Esbozo de formación

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

La pérdida debería caer de ~3 a ~0.2 en 200 pasos en estos datos sintéticos triviales.

## Usalo

Tres vías de producción:

- **PaddleOCR** maduro, rápido, multilingüe. Uso en una línea: `paddleocr.PaddleOCR(lang="en").ocr(image_path)`¿ Qué ?
- **EasyOCR** Python nativo, multilingüe, espina dorsal PyTorch.
- **Tesseract** clásico; todavía útil para documentos antiguos escaneados cuando los modelos luchan.

Para el análisis de documentos de extremo a extremo, utilice Donut o un VLM:

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

Para recibos, facturas y formularios con estructura repetible, ajuste bien Donut. Para documentos arbitrarios o OCR con razonamiento, un VLM como Qwen-VL-OCR es el estándar actual.

## Envío

Esta lección produce:

- `outputs/prompt-ocr-stack-picker.md` una solicitud que selecciona Tesseract / PaddleOCR / Donut / VLM-OCR dado tipo de documento, lenguaje y estructura.
- `outputs/skill-ctc-decoder.md` una habilidad para escribir codificadores CTC codificadores desde cero, incluida la normalización de longitud.

## Los ejercicios

1. **(Easy)**Entrenar el TinyCRNN en cadenas numéricas aleatorias de 5 dígitos durante 500 pasos.
2. **(Medium)**Replace la codificación codificada con búsqueda de haces (beam_width=5).
3. **(Hard)**Utilice PaddleOCR en un conjunto de 20 recibos, extraer elementos de línea y calcular F1 contra la verdad de tierra etiquetada a mano para los pares {item_name, price}.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| OCR | "Text from pixels" | Turning image regions into character sequences |
| CTC | "Alignment-free loss" | Loss that trains a sequence model without per-timestep labels; marginalises over alignments |
| CRNN | "Classic OCR model" | Conv feature extractor + BiLSTM + CTC; the 2015 baseline still used in production |
| Donut | "End-to-end OCR" | ViT encoder + text decoder; emits JSON directly from image |
| Layout parsing | "Find regions" | Detect and label Title/Table/Figure/Paragraph regions in a document |
| Reading order | "Text sequence" | Ordering of recognised regions into a sentence; trivial for Latin, non-trivial for mixed layouts |
| CER / WER | "Error rates" | Levenshtein distance / reference length at character or word granularity |
| VLM-OCR | "LLM that reads" | A vision-language model trained or prompted for OCR tasks; current SOTA on complex documents |

## Leer más

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) la arquitectura original de CNN+RNN+CTC
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) el papel original de CTC; lleno de ideas algorítmicas
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664) Transformador de comprensión de documentos libre de OCR
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) la pila de OCR de producción de código abierto
