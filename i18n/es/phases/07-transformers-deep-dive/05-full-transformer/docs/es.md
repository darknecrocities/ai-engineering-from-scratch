# El transformador completo  Encoder + Decodificador

> La atención es la estrella. Todo lo demás  residuos, normalización, alimentación hacia adelante, atención cruzada  es el andamio que te permite apilarlo profundamente.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## El problema

Una capa de atención única es un extractor de características, no un modelo. Un matmul por capa no es suficiente capacidad para el lenguaje. Necesitas profundidad  y breaks de profundidad sin la tubería adecuada.

El documento Vaswani de 2017 empaquetó seis decisiones de diseño que convirtieron una capa de atención en un bloque apilable. Cada transformador desde  codificador-solo (BERT), decodificador-solo (GPT), codificador-decodificador (T5)  hereda el mismo esqueleto.

Esta lección es el esqueleto. Las siguientes lecciones lo especializan  06 para codificadores, 07 para decodificadores, 08 para codificador-decodificador.

## El concepto

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### Las seis piezas

1. **Embedding + positional signal.**Tokens → vectores. Posición inyectada a través de RoPE (moderno) o sinusoidal (clásico).
2. **Self-attention.**Cada posición se atende a la otra.
3. **Feed-forward network (FFN).**MLP de dos capas en función de la posición: `W_2 · activation(W_1 · x)`. Ratio de expansión 4x por defecto.
4. **Residual connection.** `x + sublayer(x)`Sin esto, los gradientes desaparecen más allá de 6 capas.
5. **Layer normalization.** `LayerNorm`o `RMSNorm`Estabiliza el flujo residual.
6. **Cross-attention (decoder only).**Las consultas provienen del decodificador, claves y valores de la salida del codificador.

Observe el flujo de un vector a través de un bloque: la atención se mezcla a través de posiciones, el residual lo lleva hacia adelante, el FFN lo transforma y la norma mantiene la corriente estable.

```figure
transformer-block
```

### Bloque de codificación (utilizado por el codificador BERT, T5)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

El codificador es bidireccional, no se enmascara, todas las posiciones ven todas las posiciones.

### Bloque de decodificación (utilizado por el decodificador GPT, T5)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

El decodificador tiene tres subcapas por bloque. El medio  atención cruzada  es el único lugar donde la información fluye de un codificador a un decodificador. En una arquitectura pura de decodificador solo (GPT), la atención cruzada se omite y solo tienes la autoatención enmascarada + FFN.

### Pre-norma vs post-norma

Papel original: `x + sublayer(LN(x))`- ¿ Qué ?`LN(x + sublayer(x))`. Después de la norma perdió el favor alrededor de 2019  es más difícil entrenar profundamente sin un calentamiento cuidadoso.`LN`*antes de* subcapas) es el 2026 por defecto: Llama, Qwen, GPT-3+, Mistral todos lo usan.

### El bloque modernizado 2026

Vaswani 2017 envió LayerNorm + ReLU. Las pilas modernas reemplazaron a ambas.

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices, total params match) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

RMSNorm elimina la media de centrarse de LayerNorm (una subtracción menos), que ahorra en la computación y es empíricamente al menos tan estable.`Swish(W1 x) ⊙ W3 x`) supera constantemente a la FFN de ReLU/GELU en ~0,5 puntos en los documentos de Llama, PaLM y Qwen.

### Conto de parámetros

Por un bloque con `d_model = d`y la expansión de FFN `r`¿Qué es esto ?

- MHA: `4 · d²`(Proyecciones Q, K, V, O)
- FFN (SwiGLU): `3 · d · (r · d)`¿ Qué es esto ?`3rd²`
- Normas: insignificantes

En el`d = 4096, r = 2.6, layers = 32`(aproximadamente Llama 3 8B), total: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`(más incrustaciones y cabeza) Las coincidencias publicadas cuentan.

## Construye el mismo

### Paso 1: los bloques de construcción

Usando el pequeño`Matrix`clase de la lección 03 (copiada en este archivo para su independencia):

- `layer_norm(x, eps=1e-5)` restar la media, dividir por std.
- `rms_norm(x, eps=1e-6)` dividir por RMS. No hay subtracción media.
- `gelu(x)`y `silu(x) * W3 x`- ¿Qué es eso?
- `ffn_swiglu(x, W1, W2, W3)`¿ Qué ?
- `encoder_block(x, params)`y `decoder_block(x, enc_out, params)`¿ Qué ?

¿ Qué ?`code/main.py`para el cableado completo.

### Paso 2: conectar un codificador de 2 capas y un decodificador de 2 capas

Colocalas en apilamiento, pasa la salida del codificador a cada decodificador, añade una LN final antes de la proyección de salida.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Paso 3: avanzar en un ejemplo de juguete

A través de una fuente de 6 tokens y un objetivo de 5 tokens.`(5, vocab)`No hay entrenamiento. Esta lección es sobre la arquitectura, no la pérdida.

### Paso 4: intercambiar en RMSNorm + SwiGLU

Replace LayerNorm y ReLU-FFN con RMSNorm y SwiGLU. Confirme que las formas siguen coincidiendo. Esta es la modernización 2026 con una sustitución de función.

## Usalo

Las implementaciones de referencia PyTorch/TF: `nn.TransformerEncoderLayer`¿ Qué ?`nn.TransformerDecoderLayer`Pero la mayoría del código de producción 2026 tiene su propio bloque porque:

- La atención flash se llama dentro de la atención, no a través de `nn.MultiheadAttention`¿ Qué ?
- GQA / MLA no están en la referencia de la STDlib.
- RoPE, RMSNorm, SwiGLU no son los valores predeterminados de PyTorch.

HF `transformers`tiene bloques de referencia limpios que debe leer: `modeling_llama.py`Es el bloque canónico 2026 sólo para decodificadores. Es de aproximadamente 500 líneas y vale la pena caminar una vez.

**Encoder vs decoder vs encoder-decoder — when to pick:**

| Need | Pick | Example |
|------|------|---------|
| Classification, embeddings, QA over text | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

El lenguaje solo con decodificador ganó porque tiene una escala más limpia y maneja tanto la comprensión como la generación.

## Envío

¿ Qué ?`outputs/skill-transformer-block-reviewer.md`. La habilidad revisa la implementación de un nuevo bloque de transformador en función de los valores predeterminados de 2026 y señala las piezas faltantes (pre-norma, RoPE, RMSNorm, GQA, FFN ratio de expansión).

## Los ejercicios

1. **Easy.**Cuenta los parámetros en tu bloque de codificación en `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. Valida mediante la implementación del bloque y el uso de `sum(p.numel() for p in block.parameters())`¿ Qué ?
2. **Medium.**Cambiar de post-norma a pre-norma. Iniciar ambos y medir la norma de activación después de 12 capas apiladas en entrada aleatoria.
3. **Hard.**Implementar un codificador-decodificador de 4 capas en una tarea de copia de juguete (copiar `x`El tren 100 pasos. informe pérdida. ¿El cambio en RMSNorm + SwiGLU + RoPE  ¿La pérdida disminuye?

## Términos clave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Block | "One transformer layer" | Stack of norm + attention + norm + FFN, wrapped in residual connections. |
| Residual | "Skip connection" | `x + f(x)` output; enables gradient flow through deep stacks. |
| Pre-norm | "Normalize before, not after" | Modern: `x + sublayer(LN(x))`. Trains deeper without warmup gymnastics. |
| RMSNorm | "LayerNorm without the mean" | Divide by RMS; one less op, same empirical stability. |
| SwiGLU | "The FFN everyone switched to" | `Swish(W1 x) ⊙ W3 x → W2`. Beats ReLU/GELU on LM ppl. |
| Cross-attention | "How the decoder sees the encoder" | MHA with Q from decoder, K/V from encoder outputs. |
| FFN expansion | "How wide the middle MLP is" | Ratio of hidden-size to d_model, usually 4 (LayerNorm) or 2.6 (SwiGLU). |
| Bias-free | "Drop the +b terms" | Modern stacks omit biases in linear layers; slight ppl improvement, smaller model. |

## Leer más

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) especificación original del bloque.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)¿Por qué la pre-norma supera profundamente la post-norma?
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) el papel SwiGLU.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) bloque canónico 2026 solo para decodificadores.
