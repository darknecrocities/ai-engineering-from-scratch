# Construir un transformador desde cero  La piedra de capas

> Trece lecciones, un modelo, sin atajos.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 minutes

## El problema

Has leído todos los artículos, has implementado atención, divisiones de múltiples cabezas, codificación posicional, bloqueos de codificación y decodificación, pérdidas de BERT y GPT, MoE, KV caché. Ahora haz que trabajen juntos en una tarea real.

La piedra angular: entrenar a un pequeño transformador de un solo decodificador de extremo a extremo en una tarea de modelado de lenguaje a nivel de personajes. Lee Shakespeare. Genera un nuevo Shakespeare. Es lo suficientemente pequeño como para entrenar en una computadora portátil en menos de 10 minutos. Es lo suficientemente correcto que intercambiar un conjunto de datos más grande y un entrenamiento más largo te da un LM real.

Este es el "nanoGPT" del curso. No es original  El tutorial de Karpathy de 2023 nanoGPT es la implementación de referencia que cada estudiante escribe al menos una vez. Levantamos la forma y la retolamos alrededor de lo que hemos cubierto.

## El concepto

![Transformer-from-scratch block diagram](../assets/capstone.svg)

La arquitectura, anotado:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### Lo que enviamos

- `GPTConfig` un lugar para configurar todos los hiperparámetros.
- `MultiHeadAttention` Causal, en lote, con vía opcional de estilo Flash (PyTorch's `scaled_dot_product_attention`¿Qué es lo que se hace?
- `SwiGLUFFN` FFN moderno.
- `Block` Pre-norma, atención envuelta residual + FFN.
- `GPT` embebidos, bloques apilados, cabeza LM, generar().
- Bucle de entrenamiento con AdamW, cosino LR, recorte de gradiente.
- Tokenizer de nivel Char en el texto de Shakespeare.

### Lo que no enviamos

- RoPE  implementado conceptualmente en la Lección 04. Aquí usamos embeddings posicionales aprendidos para la simplicidad.
- El caché KV durante la generación  cada paso de generación recalcula la atención sobre el prefijo completo.
- Atención Flash  PyTorch 2.0+ automáticos de envío si las entradas coinciden; usamos `F.scaled_dot_product_attention`¿ Qué ?
- MoE  FFN por bloque.

### Metricas de objetivo

En una computadora portátil Mac M2, un 4 capas, 4 cabezas, d_model=128 GPT entrenado para 2.000 pasos en `tinyshakespeare.txt`¿Qué es esto ?

- La pérdida de entrenamiento converge de ~ 4,2 (a azar) a ~ 1,5 en aproximadamente 6 minutos.
- La muestra de producción parece en forma de Shakespeare: aparecen palabras arcaicas, interrupciones de líneas, nombres propios como "ROMEO:" .
- La pérdida de valor (el 10% final del texto retenido) sigue de cerca la pérdida de formación; no se sobreajusta a este tamaño/orden de presupuesto.

```figure
n5-block-stack
```

## Construye el mismo

Esta lección utiliza PyTorch.`torch`(Construcción de CPU está bien).`code/main.py`El guión se ocupa de:

- Descargar`tinyshakespeare.txt`si falta (o si se lee una copia local).
- Tokenizaje de car de nivel byte.
- Tren / val dividido en 90/10.
- Bucle de entrenamiento con bf16 autocast en el hardware soportado.
- Se concluye la toma de muestras después del entrenamiento.

### Paso 1: datos

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 caracteres únicos, un pequeño vocabulario, un tamaño de 4 bytes, sin BPE, sin drama de tokenizer.

### Paso 2: modelo

¿ Qué ?`code/main.py`El bloque es un libro de texto de la lección 05  pre-norma, RMSNorm, SwiGLU, MHA causal.

### Paso 3: ciclo de entrenamiento

Obtenga un lote aleatorio de ventanas de 256 tokens de longitud hacia adelante, cambio por entropias cruzadas hacia atrás, paso AdamW, registro, repite.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Paso 4: muestra

Si se le da una solicitud, repetidamente, muestra de los logits de arriba, añade y continúa.

### Paso 5: lee la salida

Después de 2.000 pasos:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

No Shakespeare, pero en forma de Shakespeare, una clara victoria por unos 800K parámetros y 6 minutos en una computadora portátil.

## Usalo

Esta piedra es una arquitectura de referencia.

1. **Swap the tokenizer.**Utilice BPE (por ejemplo `tiktoken.get_encoding("cl100k_base")`El tamaño de la vocab aumentó de 65 a 50.000.
2. **Train on a bigger corpus.**Usar`OpenWebText`o `fineweb-edu`Los tokens 10B en un solo A100 tardan alrededor de 24 horas en un GPT de 125M-param.
3. **Add RoPE + KV cache + Flash Attention.**Los ejercicios que se presentan a continuación te guiarán a través de cada uno.

Esto termina como un GPT de parámetro de 125M que genera inglés fluido. No es un modelo fronterizo. Pero el mismo camino de código  sólo más grande  es lo que Karpathy, EleutherAI y el Instituto Allen utilizan para entrenar los puestos de control de investigación en 2026.

## Envío

¿ Qué ?`outputs/skill-transformer-review.md`La habilidad revisa una implementación de transformador desde cero para verificar la corrección en las 13 lecciones anteriores.

## Los ejercicios

1. **Easy.**- ¿ Qué ?`code/main.py`Verifique si la pérdida de validación en la etapa final de su modelo entrenado es inferior a 2.0.`max_steps`¿La pérdida de val sigue mejorando?
2. **Medium.**Reemplazar las incorporaciones posicionales aprendidas con RoPE. Aplique la rotación a Q y K dentro `MultiHeadAttention`La pérdida de val es al menos tan baja.
3. **Medium.**Implemente una caché KV en el bucle de muestreo. Generar 500 tokens con y sin caché. El reloj de pared debería mejorar en 520x en una computadora portátil.
4. **Hard.**Añadir una segunda cabeza al modelo que predice el siguiente token más uno (MTP  Multi-Token Prediction de DeepSeek-V3).
5. **Hard.**Reemplazar el FFN único por bloque con un MoE de 4 expertos. Router + top-2 enrutamiento. Ver cómo cambia la pérdida de val en parámetros activos coincidentes.

## Términos clave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy's tutorial repo" | Minimal decoder-only transformer training code, ~300 LOC; the canonical reference. |
| tinyshakespeare | "The standard toy corpus" | ~1.1 MB of text; every character-LM tutorial since 2015 uses it. |
| Tied embeddings | "Share input/output matrix" | LM head weight = transpose of token embedding matrix; saves parameters, improves quality. |
| bf16 autocast | "Training precision trick" | Run forward/back in bf16, keep optimizer state in fp32; standard since 2021. |
| Gradient clipping | "Stops spikes" | Cap global grad norm at 1.0; prevents training blowups. |
| Cosine LR schedule | "The 2020+ default" | LR ramps up linearly (warmup) then decays cosine-shaped to 10% of peak. |
| MFU | "Model FLOP Utilization" | Achieved FLOPs / theoretical peak; 40% dense, 30% MoE is strong in 2026. |
| Val loss | "Held-out loss" | Cross-entropy on data the model never saw; overfit detector. |

## Leer más

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) la aplicación clásica de notas.
