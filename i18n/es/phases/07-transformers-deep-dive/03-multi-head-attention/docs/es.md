# Atención de varias cabezas

> Una cabeza de atención aprende una relación a la vez ocho cabezas aprenden ocho cabezas son libres toma más de ellas

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 minutes

## El problema

Una sola cabeza de autoatención calcula una matriz de atención. Esa matriz captura un tipo de relación  generalmente la que minimiza la pérdida en cualquier señal de entrenamiento. Si sus datos tienen acuerdo entre sujeto y verbo, co-referencia, discurso de largo alcance y fragmentos sintácticos enredados juntos, una sola cabeza los desprende en una sola distribución de máxima suave y pierde la mitad de la señal.

La solución del artículo Vaswani de 2017: ejecuta varias funciones de atención en paralelo, cada una con sus propias proyecciones Q, K, V, y concatenar las salidas.`d_model / n_heads`Los parámetros totales permanecen iguales.

La atención multi-cabeza es el predeterminado de cada transformador en los barcos de 2026. El único argumento es sobre *cuántas cabezas* y si las claves y valores comparten proyecciones (Attención de consulta grupada, Atención de consulta múltiple, Atención latente de múltiples cabezas).

## El concepto

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Split.**¿ Qué ?`X`de forma`(N, d_model)`Proyecto a Q, K, V de cada forma `(N, d_model)`- Reconfigurar para`(N, n_heads, d_head)`donde`d_head = d_model / n_heads`. Transponer a`(n_heads, N, d_head)`¿ Qué ?

**Attend in parallel.**ejecutar escalado producto de puntos de atención dentro de cada cabeza.`(N, d_head)`Las cabezas operan en diferentes subespacios de la incorporación y nunca hablan durante el cálculo de la atención.

**Concatenate and project.**Las cabezas de pila vuelven a la`(N, d_model)`y multiplicar por una matriz de salida aprendida `W_o`de forma`(d_model, d_model)`- ¿ Qué ?`W_o`Es donde las cabezas se mezclan.

**Why it works.**Cada cabeza puede especializarse sin competir con los demás para el presupuesto representativo. Los estudios de sondeo de 20192024 muestran diferentes roles de cabeza: cabezas posicionales, cabeza que atiende al token anterior, cabezas de copia, cabezas de entidades nombradas, cabezas de inducción (que son la base del aprendizaje en contexto).

**The 2026 lineage of variations:**

| Variant | Q heads | K/V heads | Used by |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA es el estándar moderno porque reduce la memoria de caché KV en un factor de `N/G`MLA va más allá comprimiendo K/V en un espacio latente, luego proyectando de nuevo en tiempo de cálculo  cuesta FLOPs, ahorra mucho más memoria.

```figure
multihead-split
```

## Construye el mismo

### Paso 1: separar las cabezas de la atención de la cabeza única que ya tenemos

Toma el .`SelfAttention`En el caso de las clases de la segunda clase, el resultado de la prueba es el resultado de la prueba de la segunda clase.`code/main.py`para una implementación numpy; la lógica es:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

Uno se remodela y otro se transponen.`nn.MultiheadAttention`¿ Qué ?

### Paso 2: ejecutar la atención de producto en escala de puntos por cabeza

Cada cabeza recibe su propia rebanada de Q, K, V. La atención se convierte en un matmul en lote:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

En hardware real .`Qh @ Kh.transpose(...)`Es uno .`bmm`La GPU ve un solo parche de forma .`(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`Añadir cabezas es gratis.

### Paso 3: Variante de atención de cuentas agrupadas

Sólo las proyecciones de clave y valor cambian.`n_heads`los grupos; K y V obtendrán`n_kv_heads < n_heads`grupos y se repiten para coincidir:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

En la inferencia esto ahorra memoria porque sólo`n_kv_heads`Las copias viven en el caché KV, no `n_heads`Llama 3 70B utiliza 64 cabezas de consulta con 8 cabezas de KV  un 8x cache contractor.

### Paso 4: sondear lo que cada cabeza aprendió

Ejecutar MHA en una frase corta con 4 cabezas.`(N, N)`Verá diferentes cabezas seleccionar diferentes estructuras incluso con inicialización aleatoria que es en parte señal, en parte simetría de rotación en los subespacios.

## Usalo

En PyTorch, la versión de una línea:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA a partir de PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**How many heads?**Reglas generales de los modelos de producción en 2026:

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head`casi siempre aterriza en 64 o 128. Es la unidad de cuánto una cabeza puede "ver". Baja por debajo de 32 y las cabezas comienzan a luchar contra el factor de escala.`sqrt(d_head)`Si se superan los 256 y se pierde el beneficio de "muchos especialistas pequeños".

## Envío

¿ Qué ?`outputs/skill-mha-configurator.md`. La habilidad recomienda el recuento de cabezas, el recuento de cabezas kv y la estrategia de proyección para un nuevo transformador dado el presupuesto de parámetros, la longitud de la secuencia y el objetivo de despliegue.

## Los ejercicios

1. **Easy.**Tome el MHA de `code/main.py`y el cambio`n_heads`de 1 a 16 con `d_model=64`¿Ayuda más cabezas, plato o daño?
2. **Medium.**Implemente MQA (una cabeza de KV compartida entre todas las cabezas de consulta). Medir cuánto parámetro cuenta caídas frente a MHA completa. Compute cuánto se reduce el tamaño de la caché de KV a la inferencia para N = 2048.
3. **Hard.**Implemente una versión pequeña de Multi-head Latent Attention: comprime K, V a un rango de`r`Se puede almacenar en el caché KV, descomprimir en el tiempo de atención.`r`¿La memoria de caché pasa por debajo de 1/8 de la MHA completa mientras que la calidad permanece dentro de 1 bit de la validación ppl?

## Términos clave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Head | "A single attention circuit" | One Q/K/V projection of dimension `d_head = d_model / n_heads` with its own attention matrix. |
| d_head | "Head dimension" | Per-head hidden width; almost always 64 or 128 in production. |
| Split / combine | "Reshape tricks" | `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose around attention. |
| W_o | "Output projection" | `(d_model, d_model)` matrix applied after concatenating heads; where heads mix. |
| MQA | "One KV head" | Multi-Query Attention: single shared K/V projection. Smallest KV cache, some quality loss. |
| GQA | "The default since Llama 2" | Grouped-Query Attention with `n_kv_heads < n_heads`; repeats to match Q. |
| MLA | "DeepSeek's trick" | Multi-head Latent Attention: K,V compressed to low-rank latent, decompressed at attend time. |
| Induction head | "The circuit behind in-context learning" | A pair of heads that detect previous occurrences and copy what followed them. |

## Leer más

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) la especificación original de múltiples cabezas.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) el documento de la MQA.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) cómo convertir la MHA en GQA después de la formación.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) MLA y por qué supera a MHA/GQA en memoria caché.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) Mecanicistas mira lo que las cabezas realmente hacen.
