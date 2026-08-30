# Encriptación de posición  Sinusoidal, RoPE, ALiBi

> La atención es invariable en la permutación. "El gato se sentó en el tapicero" y "el gato en el sat" producen la misma salida sin señal de posición. Tres algoritmos la arreglan  cada uno con una apuesta diferente en lo que significa "posición".

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 minutes

## El problema

La atención a escala de producto de punto es ciega al orden.`softmax(Q K^T / √d) V`Se calcula a partir de similitudes parecidas.`X`No hay nada dentro de la atención que se preocupe por la posición.

Para el lenguaje, código, audio, video  cualquier cosa donde el orden lleva un significado  es fatal.

La solución es inyectar posición en los embeddings de alguna manera.

1. **Absolute sinusoidal**(Vaswani 2017). Añadir `sin/cos`Es simple, sin aprendizaje, extrapola poco más allá de las longitudes entrenadas.
2. **RoPE — Rotary Position Embeddings**(Su 2021). Rotar los vectores Q y K por un ángulo proporcional a la posición. Encodifica * posición relativa* directamente en el producto de puntos. Dominante en 2026.
3. **ALiBi — Attention with Linear Biases**(Presiones 2022). Salta los embebidos por completo; añade una penalización lineal por cabeza a las puntuaciones de atención basadas en la distancia. Excelente extrapolación de longitud.

A partir de 2026, prácticamente todos los modelos abiertos fronterizos usan RoPE: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. Un puñado de modelos de contexto largo usan ALiBi o sus variantes modernas.

## El concepto

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### Sinusoidal absoluto

Precomputa una matriz fija `PE`de forma`(max_len, d_model)`¿Qué es esto ?

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

Entonces ...`X' = X + PE[:N]`Cada dimensión es un sinusoide con una frecuencia diferente. El modelo aprende a leer la posición del patrón de fase.`max_len`: nada le dijo al modelo lo que pasaba en la posición 2048 cuando sólo veía posiciones 02047.

### RoPE

Gira los vectores Q y K (no embeddings). Para un par de dimensiones `(2i, 2i+1)`¿Qué es esto ?

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

Aplicar la misma rotación a las teclas con posición `pos_k`. El producto de puntos `q'_m · k'_n`se convierte en una función de `(m - n)`Sólo.**the attention score depends only on the relative distance**, aunque la rotación estaba bloqueada en posiciones absolutas.

Extender el RoPE: `base`El Llama 3 se extendió de 8K a 128K de esta manera.

### El mismo

Salta el truco de incrustación.

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

¿ Dónde ?`m_h`es una pendiente específica de la cabeza (por ejemplo `1 / 2^(8·h/H)`Los tokens más cercanos se incrementan; los tokens más lejanos se penalizan. No hay costo de tiempo de entrenamiento. El documento muestra que la extrapolación de longitud supera sinusoidal y coincide con el RoPE en su longitud entrenada original.

### Qué elegir en 2026

| Variant | Extrapolation | Training cost | Used by |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | poor | free | original transformer, early BERT |
| Learned absolute | none | tiny | GPT-2, GPT-3 |
| RoPE | good with scaling | free | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | excellent | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | excellent | free | BLOOM, MPT, Baichuan |

RoPE ganó porque atrajo la atención sin cambiar la arquitectura, codifica la posición relativa y su`base`El hiperparámetro da un botón limpio para ajustar el contexto largo.

```figure
rope-explorer
```

## Construye el mismo

### Paso 1: codificación sinusoidal

¿ Qué ?`code/main.py`Un cálculo de 4 líneas:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

Añadir esto a la matriz de incorporación antes de la primera capa de atención.

### Paso 2: RoPE aplicado a Q, K

RoPE opera en el lugar en Q y K. Para cada par de dimmers:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

Crucial: aplicar la misma función a Q en posición `m`y K en posición `n`Su producto de puntos recoge un`cos((m-n)·θ_i)`La atención aprende la posición relativa de forma gratuita.

### Paso 3: Pistas y sesgos de ALiBi

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

Añadir`bias[h]`a la `(seq_len, seq_len)`Matriz de puntaje de atención de la cabeza `h`, luego softmax.

### Paso 4: verificar la propiedad relativa de distancia del RoPE

Escoge dos vectores aleatorios .`a, b`- Gira por el .`(pos_a, pos_b)`Entonces por`(pos_a + k, pos_b + k)`. ambos productos de puntos deben coincidir dentro del error de punto flotante. Esa propiedad es el punto entero de RoPE  es invariante al desvio absoluto, sólo la brecha relativa importa.

## Usalo

PyTorch 2.5+ envía a los servicios públicos de RoPE en `torch.nn.functional`La mayoría de los códigos de producción usan`flash_attn`o `xformers`en el que se aplique RoPE dentro del núcleo de atención.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Long-context tricks in 2026:**

- **NTK-aware interpolation.**Reescalado`base`¿ Qué ?`base * (scale_factor)^(d/(d-2))`cuando se extiende de 4K a 16K+.
- **YaRN.**Interpolación más inteligente que preserva la entropía de la atención en contextos largos.
- **LongRoPE.**El método 2024 de Microsoft que utiliza la búsqueda evolutiva para seleccionar factores de escala por dimensión.
- **Position interpolation + fine-tuning.**Sólo reducir las posiciones por el factor de extensión y ajustar a 15B tokens. Sorprendentemente eficaz.

## Envío

¿ Qué ?`outputs/skill-positional-encoding-picker.md`. La habilidad elige una estrategia de codificación para un nuevo modelo dada la longitud del contexto objetivo, las necesidades de extrapolación y el presupuesto de formación.

## Los ejercicios

1. **Easy.**Traza el senoideal .`PE`matriz como mapa de calor para `max_len=512, d=128`Confirmar el patrón de "las franjas se amplían a medida que crece el índice de dimensiones".
2. **Medium.**Implemente una escalación de RoPE consciente de NTK. Entrenar un pequeño LM en secuencias de longitud 256, luego probar en longitud 1024 con y sin escalación. Medir la perplejidad.
3. **Hard.**Implemente ALiBi y RoPE en el mismo módulo de atención. Entrenar un transformador de 4 capas en una tarea de copia con secuencias de longitud 512. Extrapolar a 2048 en el momento de la prueba. Comparar degradación.

## Términos clave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Positional encoding | "Tells attention about order" | Any signal added to embeddings or attention that encodes position. |
| Sinusoidal | "The original one" | `sin/cos` at geometric frequencies added to embeddings; doesn't extrapolate. |
| RoPE | "Rotary embeddings" | Rotate Q, K by position-dependent angle; dot product encodes relative distance. |
| ALiBi | "Linear bias trick" | Add `-m·\|i-j\|` to attention scores; no embedding needed, great extrapolation. |
| base | "RoPE's knob" | The frequency scaler in RoPE; increase to extend context at inference. |
| NTK-aware | "A RoPE scaling trick" | Rescale `base` so high-frequency dims aren't squeezed when context expands. |
| YaRN | "The fancy one" | Per-dimension interpolation+extrapolation that preserves attention entropy. |
| Extrapolation | "Works beyond trained length" | Can the position scheme serve correct output past `max_len` seen in training? |

## Leer más

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) original sinusoidal.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) Papel de papel de papel rope.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) ALiBi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) el estado de la técnica de escalado RoPE.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) Meta's Llama 2 papel de largo contexto.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) el método de Microsoft utilizado por Phi-3-Long y citado en la sección Use It.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) Implementaciones en el nivel de producción de cada esquema de escalado de RoPE (por defecto, lineal, dinámico, YaRN, LongRoPE, Llama-3).
