# Pre-entrenamiento de un mini GPT (124M Parámetros)

> GPT-2 Small tiene 124 millones de parámetros. Son 12 capas de transformador, 12 cabezas de atención y 768 embebedidos dimensiones. Puedes entrenarlo desde cero en una sola GPU en unas horas. La mayoría de la gente nunca hace esto. Utilizan puntos de control pre-entrenados. Pero si no entrenas uno tú mismo, no entiendes realmente lo que está sucediendo dentro del modelo en el que estás construyendo productos.

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10, Lessons 01-03 (Tokenizers, Building a Tokenizer, Data Pipelines)
**Time:** ~120 minutes

## Objetivos de aprendizaje

- Implementar la arquitectura completa de GPT-2 (124M parámetros) desde cero: embeddings de tokens, embeddings posicionales, bloques de transformadores y la cabeza de modelo de lenguaje
- Entrenar un modelo GPT en un corpus de texto utilizando predicción de tokens siguientes con pérdida de entropía cruzada
- Implementar la generación de texto autoregresista con muestreo de temperatura y filtración top-k/top-p
- Monitorear las curvas de pérdida de formación y validar que el modelo aprenda patrones de lenguaje coherentes

## El problema

Ya sabes lo que es un transformador, has leído los diagramas, puedes recitar "la atención es todo lo que necesitas" y dibujar cajas etiquetadas "Multi-Head Attention" en un tablero blanco.

Nada de eso significa que entiendas lo que sucede cuando un modelo genera texto.

Hay 124.438.272 parámetros en GPT-2 Small (con gravedad de unión). Cada uno de ellos fue configurado ejecutando un ciclo de entrenamiento: pase hacia adelante, pérdida de cálculo, pase hacia atrás, actualización de pesas. Doce bloques de transformador. Doce cabezas de atención por bloque. Un espacio de incorporación de 768 dimensiones. Un vocabulario de 50.257 tokens. Cada vez que el modelo genera un token, los 124 millones de parámetros participan en una cadena de multiplicación de matriz única que toma una secuencia de ID de token y produce una distribución de probabilidad sobre el siguiente token.

Si nunca lo has construido tú mismo, estás trabajando con una caja negra. Puedes usar la API. Puedes ajustar. Pero cuando algo sale mal - cuando el modelo alucina, cuando se repite, cuando se niega a seguir las instrucciones - no tienes un modelo mental para *por qué*.

Esta lección construye GPT-2 Small desde cero. No en PyTorch. En numpy. Cada multiplicación de matriz es visible. Cada gradiente es calculado por su código. Verá exactamente cómo 124 millones de números conspiran para predecir la próxima palabra.

## El concepto

### La arquitectura del GPT

GPT es un modelo de lenguaje autoregresista. "Autoregresista" significa que genera un token a la vez, cada uno condicionado a todos los tokens anteriores.

Aquí está el gráfico completo de cálculo de las identidades de tokens a las probabilidades de tokens siguientes:

1. Se entran los tokens. Forma: (batch_size, seq_len).
2. Identificación de identificación de un vector de 768 dimensiones.
3. Cada posición (0, 1, 2, ...) se hace con un vector de 768 dimensiones.
4. Añadir embeddings de tokens + posiciones de embeddings.
5. Pasa por 12 bloques de transformadores.
6. La normalización de la capa final.
7. Proyección lineal al tamaño del vocabulario.
8. Softmax para obtener probabilidades.

No hay convulsiones, no hay recurrencias, sólo incorporaciones, atención, redes de retroalimentación y normas de capas apiladas 12 veces.

```mermaid
graph TD
    A["Token IDs\n(batch, seq_len)"] --> B["Token Embeddings\n(batch, seq_len, 768)"]
    A --> C["Position Embeddings\n(batch, seq_len, 768)"]
    B --> D["Add"]
    C --> D
    D --> E["Transformer Block 1"]
    E --> F["Transformer Block 2"]
    F --> G["..."]
    G --> H["Transformer Block 12"]
    H --> I["Layer Norm"]
    I --> J["Linear Head\n(768 -> 50257)"]
    J --> K["Softmax\nNext-token probabilities"]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#0f3460,color:#fff
    style C fill:#1a1a2e,stroke:#0f3460,color:#fff
    style D fill:#1a1a2e,stroke:#16213e,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style H fill:#1a1a2e,stroke:#e94560,color:#fff
    style I fill:#1a1a2e,stroke:#16213e,color:#fff
    style J fill:#1a1a2e,stroke:#0f3460,color:#fff
    style K fill:#1a1a2e,stroke:#51cf66,color:#fff
```

### El bloque de transformador

Cada uno de los 12 bloques sigue el mismo patrón. Arquitectura pre-norma (GPT-2 utiliza pre-norma, no post-norma como el transformador original):

1. La capaNorm
2. Atención personal de varias cabezas
3. Conexión residual (agrega entrada hacia atrás)
4. La capaNorm
5. Red de transmisión de datos (MLP)
6. Conexión residual (agrega entrada hacia atrás)

Las conexiones residuales son críticas. Sin ellas, los gradientes desaparecen cuando alcanzan el bloque 1 durante la propagación posterior. Con ellos, los gradientes pueden fluir directamente desde la pérdida a cualquier capa a través del camino "salto".

### Atención: El mecanismo central

La autoatención permite que cada token mire cada token anterior y decida cuánto asistir a cada uno.

Para cada posición de token, calcular tres vectores de la entrada:
- **Query (Q)**"¿Qué estoy buscando?"
- **Key (K)**"¿Qué tengo?"
- **Value (V)**"¿Qué información llevo?"

```
Q = input @ W_q    (768 -> 768)
K = input @ W_k    (768 -> 768)
V = input @ W_v    (768 -> 768)

attention_scores = Q @ K^T / sqrt(d_k)
attention_scores = mask(attention_scores)   # causal mask: -inf for future positions
attention_weights = softmax(attention_scores)
output = attention_weights @ V
```

La máscara causal es lo que hace que el GPT sea autorregresor. La posición 5 puede atender las posiciones 0-5 pero no 6, 7, 8, etc. Esto evita que el modelo "trampa" mirando futuros tokens durante el entrenamiento.

**Multi-head attention**El espacio de 768 dimensiones se divide en 12 cabezas de 64 dimensiones cada una. Cada cabeza aprende un patrón de atención diferente. Una cabeza puede rastrear relaciones sintácticas (acuerdo entre sujeto y verbo).

```mermaid
graph LR
    subgraph MultiHead["Multi-Head Attention (12 heads)"]
        direction TB
        I["Input (768)"] --> S1["Split into 12 heads"]
        S1 --> H1["Head 1\n(64 dims)"]
        S1 --> H2["Head 2\n(64 dims)"]
        S1 --> H3["..."]
        S1 --> H12["Head 12\n(64 dims)"]
        H1 --> C["Concat (768)"]
        H2 --> C
        H3 --> C
        H12 --> C
        C --> O["Output Projection\n(768 -> 768)"]
    end

    subgraph SingleHead["Each Head Computes"]
        direction TB
        Q["Q = X @ W_q"] --> A["scores = Q @ K^T / 8"]
        K["K = X @ W_k"] --> A
        A --> M["Apply causal mask"]
        M --> SM["Softmax"]
        SM --> MUL["weights @ V"]
        V["V = X @ W_v"] --> MUL
    end

    style I fill:#1a1a2e,stroke:#e94560,color:#fff
    style O fill:#1a1a2e,stroke:#e94560,color:#fff
    style Q fill:#1a1a2e,stroke:#0f3460,color:#fff
    style K fill:#1a1a2e,stroke:#0f3460,color:#fff
    style V fill:#1a1a2e,stroke:#0f3460,color:#fff
```

La división por sqrt(d_k) -- sqrt(64) = 8 -- es escalar. Sin ella, los productos de puntos crecen para vectores de alta dimensión, empujando la softmax a regiones donde los gradientes son casi cero. Esta fue una de las ideas clave en el documento original "Attención es todo lo que necesitas".

### KV Cache: Por qué la inferencia es rápida

Durante el entrenamiento, procesas toda la secuencia a la vez. Durante la inferencia, generas un token a la vez. Sin optimización, generar un token N requiere recomputar la atención para todos los tokens anteriores N-1.

KV Cache resuelve esto. Después de calcular K y V para cada token, almacenalos. Al generar el token N + 1, sólo necesita calcular Q para el nuevo token y buscar el caché K y V de todos los tokens anteriores. Esto reduce el costo por token de O(N) a O(1) para el cálculo de K y V. El cálculo de la puntuación de atención sigue siendo O(N) porque se atienden a todas las posiciones anteriores, pero se evitan multiplicidades redundantes de matriz en la entrada.

Para GPT-2 con 12 capas y 12 capas, el caché KV almacena 2 (K + V) x 12 capas x 12 capas x 64 dims = 18.432 valores por token. Para una secuencia de 1024 tokens, es aproximadamente 75 MB en FP32. Para Llama 3 405B con 128 capas, el caché KV para una sola secuencia puede superar los 10 GB. Esta es la razón por la cual la inferencia de contexto largo está limitada a la memoria.

### Preempleo vs decodificación: dos fases de inferencia

Cuando envías una solicitud a un LLM, la inferencia se produce en dos fases distintas.

**Prefill**procesan todo su prompt en paralelo. Todos los tokens son conocidos, por lo que el modelo puede calcular la atención para todas las posiciones simultáneamente. Esta fase está limitada a la computación - la GPU está haciendo multiplicidades de matriz a todo rendimiento. Para un prompt de 1000 tokens en un A100, el preempleo toma aproximadamente 20-50 ms.

**Decode**genera tokens uno a la vez. Cada nuevo token depende de todos los tokens anteriores. Esta fase está ligada a la memoria -- el cuello de botella es la lectura de los pesos del modelo y la caché KV de la memoria de la GPU, no la matemática de la matriz en sí. Los núcleos de computación de la GPU están en su mayoría inactivos esperando las lecturas de memoria. Para GPT-2, cada paso de decodificación toma aproximadamente el mismo tiempo independientemente de cuántos FLOPs requieran las matmuls, porque el ancho de banda de memoria es la restricción.

Esta distinción es importante para los sistemas de producción. Preemplaza escalas de rendimiento con computación GPU (más FLOPS = preemplazo más rápido). Decodifica escalas de rendimiento con ancho de banda de memoria (memoria más rápida = decodificación más rápida). Es por eso que el H100 de NVIDIA se centró en mejoras en ancho de banda de memoria en comparación con el A100 - acelera directamente la generación de tokens.

```mermaid
graph LR
    subgraph Prefill["Phase 1: Prefill"]
        direction TB
        P1["Full prompt\n(all tokens known)"]
        P2["Parallel computation\n(compute-bound)"]
        P3["Builds KV Cache"]
        P1 --> P2 --> P3
    end

    subgraph Decode["Phase 2: Decode"]
        direction TB
        D1["Generate token N"]
        D2["Read KV Cache\n(memory-bound)"]
        D3["Append to KV Cache"]
        D4["Generate token N+1"]
        D1 --> D2 --> D3 --> D4
        D4 -.->|repeat| D1
    end

    Prefill --> Decode

    style P1 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style P2 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style P3 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D4 fill:#1a1a2e,stroke:#e94560,color:#fff
```

### El ciclo de entrenamiento

El entrenamiento de un LLM es predicción de la próxima ficha. Dadas las fichas [0, 1, 2, ..., N-1], predica fichas [1, 2, 3, ..., N]. La función de pérdida es la entropía cruzada entre la distribución de probabilidad prevista del modelo y la ficha siguiente real.

Un paso de entrenamiento:

1. **Forward pass**Recurrir al lote a través de los 12 bloques. Obtener logits (puntos pre-softmax) para cada posición.
2. **Compute loss**: Entropia cruzada entre logits y tokens objetivo (la entrada desplazada por una posición).
3. **Backward pass**: Calcule los gradientes de todos los parámetros 124M mediante la propagación hacia atrás.
4. **Optimizer step**GPT-2 utiliza a Adam para el calentamiento del ritmo de aprendizaje y la descomposición cosina.

El horario de tasa de aprendizaje es más importante de lo que se podría esperar. GPT-2 se calienta de 0 a la tasa de aprendizaje máxima durante los primeros 2.000 pasos, luego se descompone siguiendo una curva cosina. Comenzando con una tasa de aprendizaje alta hace que el modelo diverja. Mantener una tasa alta constante causa oscilación en el entrenamiento posterior. El patrón de calentamiento-descomposición se utiliza por cada LLM principal.

### GPT-2 pequeño: los números

| Component | Shape | Parameters |
|-----------|-------|------------|
| Token embeddings | (50257, 768) | 38,597,376 |
| Position embeddings | (1024, 768) | 786,432 |
| Per-block attention (W_q, W_k, W_v, W_out) | 4 x (768, 768) | 2,359,296 |
| Per-block FFN (up + down) | (768, 3072) + (3072, 768) | 4,718,592 |
| Per-block LayerNorms (2x) | 2 x 768 x 2 | 3,072 |
| Final LayerNorm | 768 x 2 | 1,536 |
| **Total per block** | | **7,080,960** |
| **Total (12 blocks)** | | **85,054,464 + 39,383,808 = 124,438,272** |

La proyección de salida (cabeza de logits) comparte pesos con la matriz de incorporación de tokens. Esto se llama atado de peso - reduce el conteo de parámetros en 38M y mejora el rendimiento porque obliga al modelo a usar el mismo espacio de representación para la entrada y salida.

## Construye el mismo

### Paso 1: Incorporar una capa

Las incorporaciones de tokens mapean cada una de las 50.257 tokens posibles a un vector de 768 dimensiones.

```python
import numpy as np

class Embedding:
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        self.token_embed = np.random.randn(vocab_size, embed_dim) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, embed_dim) * 0.02

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        tok_emb = self.token_embed[token_ids]
        pos_emb = self.pos_embed[:seq_len]
        return tok_emb + pos_emb
```

La desviación estándar de 0.02 para la inicialización proviene del papel GPT-2. demasiado grande y los pasos iniciales hacia adelante producen valores extremos que desestabilizan el entrenamiento. demasiado pequeño y las salidas iniciales son casi idénticas para todas las entradas, haciendo inútiles las señales de gradiente tempranas.

### Paso 2: Autoatención con máscara causal

La máscara causal establece posiciones futuras a infinito negativo antes de softmax, asegurando que cada posición sólo puede atender a sí misma y a posiciones anteriores.

```python
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, -1, -2 if Q.ndim == 4 else 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ V
```

La implementación de softmax restará el máximo antes de exponenciar. Sin esto, exp(large_number) se sobreflage a infinito. Este es un truco de estabilidad numérica que no cambia la salida porque softmax(x - c) = softmax(x) para cualquier constante c.

### Paso 3: Atención por múltiples cabezas

Divide la entrada 768-dimensional en 12 cabezas de 64 dimensiones cada una. Cada cabeza calcula la atención de forma independiente. Concaten los resultados y proyectar de nuevo a 768 dimensiones.

```python
class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_out = np.random.randn(embed_dim, embed_dim) * 0.02

    def forward(self, x, mask=None):
        batch, seq_len, d = x.shape
        Q = (x @ self.W_q).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = (x @ self.W_k).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = (x @ self.W_v).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)
        attn_out = weights @ V

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d)
        return attn_out @ self.W_out
```

La danza de remodelación-transposición-reforma es la parte más confusa de la atención multi-cabeza. Esto es lo que sucede: el tensor (batch, seq_len, 768) se convierte en (batch, seq_len, 12, 64), luego (batch, 12, seq_len, 64). Ahora cada una de las 12 cabezas tiene su propia (seq_len, 64) matriz para dirigir la atención. Después de la atención, invertimos el proceso: (batch, 12, seq_len, 64) se convierte en (batch, seq_len, 12, 64) se convierte en (batch, seq_len, 768).

### Paso 4: Bloqueo de transformador

Un bloque completo de transformador: LayerNorm, atención multi-cabeza con residual, LayerNorm, feedforward con residual.

```python
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class FeedForward:
    def __init__(self, embed_dim, ff_dim):
        self.W1 = np.random.randn(embed_dim, ff_dim) * 0.02
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, embed_dim) * 0.02
        self.b2 = np.zeros(embed_dim)

    def forward(self, x):
        h = x @ self.W1 + self.b1
        h = np.maximum(0, h)  # GELU approximation: ReLU for simplicity
        return h @ self.W2 + self.b2


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_dim):
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)

    def forward(self, x, mask=None):
        x = x + self.attn.forward(self.ln1.forward(x), mask)
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x
```

La red de retroalimentación expande la entrada de 768 dimensiones a 3,072 dimensiones (4x), aplica una no linearidad, luego proyecta de nuevo a 768. Este patrón de contracción de expansión le da al modelo una representación interna "más amplia" para trabajar en cada posición. GPT-2 utiliza la activación GELU, pero usamos ReLU aquí por simplicidad - la diferencia es menor para entender la arquitectura.

### Paso 5: Modelo completo de GPT

Coloque 12 bloques de transformador. Añade la capa de incorporación en la parte delantera y la proyección de salida en la parte posterior.

```python
class MiniGPT:
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12,
                 num_layers=12, max_seq_len=1024, ff_dim=3072):
        self.embedding = Embedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(embed_dim)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)

        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x, mask)
        x = self.ln_f.forward(x)

        logits = x @ self.embedding.token_embed.T
        return logits

    def count_parameters(self):
        total = 0
        total += self.embedding.token_embed.size
        total += self.embedding.pos_embed.size
        for block in self.blocks:
            total += block.attn.W_q.size + block.attn.W_k.size
            total += block.attn.W_v.size + block.attn.W_out.size
            total += block.ffn.W1.size + block.ffn.b1.size
            total += block.ffn.W2.size + block.ffn.b2.size
            total += block.ln1.gamma.size + block.ln1.beta.size
            total += block.ln2.gamma.size + block.ln2.beta.size
        total += self.ln_f.gamma.size + self.ln_f.beta.size
        return total
```

Observe el peso de la unión: `logits = x @ self.embedding.token_embed.T`La proyección de salida reutiliza la matriz de embebido de tokens (transpuesta). Esto no es solo un truco de ahorro de parámetros. Significa que el modelo utiliza el mismo espacio vectorial para entender los tokens (embeddings) y predecirlos (salida).

### Paso 6: Circuito de entrenamiento

Para una carrera de entrenamiento real en parámetros 124M, necesitaría una GPU y PyTorch. Este ciclo de entrenamiento demuestra la mecánica de un modelo pequeño que funciona en pura numpy. Usamos un modelo pequeño (4 capas, 4 cabezas, 128 dims) para hacerlo manejable.

```python
def cross_entropy_loss(logits, targets):
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    max_logits = logits_flat.max(axis=-1, keepdims=True)
    log_softmax = logits_flat - max_logits - np.log(
        np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
    )

    loss = -log_softmax[np.arange(len(targets_flat)), targets_flat].mean()
    return loss


def train_mini_gpt(text, vocab_size=256, embed_dim=128, num_heads=4,
                   num_layers=4, seq_len=64, num_steps=200, lr=3e-4):
    tokens = np.array(list(text.encode("utf-8")[:2048]))
    model = MiniGPT(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_layers, max_seq_len=seq_len, ff_dim=embed_dim * 4
    )

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Training tokens: {len(tokens):,}")
    print(f"Config: {num_layers} layers, {num_heads} heads, {embed_dim} dims")
    print()

    for step in range(num_steps):
        start_idx = np.random.randint(0, max(1, len(tokens) - seq_len - 1))
        batch_tokens = tokens[start_idx:start_idx + seq_len + 1]

        input_ids = batch_tokens[:-1].reshape(1, -1)
        target_ids = batch_tokens[1:].reshape(1, -1)

        logits = model.forward(input_ids)
        loss = cross_entropy_loss(logits, target_ids)

        if step % 20 == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f}")

    return model
```

La pérdida comienza cerca de ln(vocab_size) - para un vocabulario de nivel de byte de 256 tokens, es decir ln(256) = 5.55. Un modelo aleatorio asigna la misma probabilidad a cada token.

En la producción, se utilizaría el optimizador Adam con acumulación de gradientes, calentamiento de la tasa de aprendizaje y recorte de gradientes.

### Paso 7: Generación de textos

La generación utiliza el modelo entrenado para predecir un token a la vez.

```python
def generate(model, prompt_tokens, max_new_tokens=100, temperature=0.8):
    tokens = list(prompt_tokens)
    seq_len = model.embedding.pos_embed.shape[0]

    for _ in range(max_new_tokens):
        context = np.array(tokens[-seq_len:]).reshape(1, -1)
        logits = model.forward(context)
        next_logits = logits[0, -1, :]

        next_logits = next_logits / temperature
        probs = np.exp(next_logits - next_logits.max())
        probs = probs / probs.sum()

        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(next_token)

    return tokens
```

La temperatura controla la aleatoriedad. La temperatura 1.0 utiliza la distribución bruta. La temperatura 0.5 la agudiza (más determinista - el modelo elige sus mejores opciones con más frecuencia). La temperatura 1.5 la aplania (más aleatorias - los tokens de baja probabilidad tienen una mayor oportunidad). La temperatura 0.0 es codificación codificada (siempre elige el token de mayor probabilidad).

El `tokens[-seq_len:]`La ventana es necesaria porque el modelo tiene una longitud máxima de contexto (1024 para GPT-2). Una vez que lo superes, debe soltar los tokens más antiguos. Esta es la "ventana de contexto" de la que todos hablan.

```figure
sampling-decoder
```

## Usalo

### Formación completa y demostración de generación

```python
corpus = """The transformer architecture has revolutionized natural language processing.
Attention mechanisms allow the model to focus on relevant parts of the input.
Self-attention computes relationships between all pairs of positions in a sequence.
Multi-head attention splits the representation into multiple subspaces.
Each attention head can learn different types of relationships.
The feedforward network provides nonlinear transformations at each position.
Residual connections enable gradient flow through deep networks.
Layer normalization stabilizes training by normalizing activations.
Position embeddings give the model information about token ordering.
The causal mask ensures autoregressive generation during training.
Pre-training on large text corpora teaches the model general language understanding.
Fine-tuning adapts the pre-trained model to specific downstream tasks."""

model = train_mini_gpt(corpus, num_steps=200)

prompt = list("The transformer".encode("utf-8"))
output_tokens = generate(model, prompt, max_new_tokens=100, temperature=0.8)
generated_text = bytes(output_tokens).decode("utf-8", errors="replace")
print(f"\nGenerated: {generated_text}")
```

En un corpus pequeño con un modelo pequeño, el texto generado será semicoherente en el mejor de los casos. Aprenderá algunos patrones de nivel de byte del texto de entrenamiento, pero no puede generalizar la forma en que GPT-2 hace con 40 GB de datos de entrenamiento y la arquitectura completa de parámetros 124M. El punto no es la calidad de salida. El punto es que puedes rastrear cada paso: búsqueda de incorporación, cálculo de atención, transformación de retroalimentación, proyección logit, softmax y muestreo. Cada operación es visible.

## Envío

Esta lección produce`outputs/prompt-gpt-architecture-analyzer.md`-- un prompt que analiza las opciones de arquitectura en cualquier modelo de estilo GPT. Le da una tarjeta modelo o un informe técnico y desglosan la asignación de parámetros, el diseño de atención y las decisiones de escala.

## Los ejercicios

1. Modificar el modelo para utilizar 24 capas y 16 cabezas en lugar de 12/12. Cuente los parámetros. ¿Cómo se compara el duplicado de profundidad con el duplicado de ancho (dimensión de incorporación)?

2. Implemente la función de activación GELU (GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))) y reemplace la ReLU en la red de retroalimentación.

3. Añadir una caché KV a la función de generación. Almacenar los tensores K y V para cada capa después del primer paso hacia adelante, y reutilizarlos para los tokens posteriores. Medir la velocidad: generar 200 tokens con y sin la caché y comparar el tiempo del reloj de pared.

4. Implemente el muestreo de top-k (sólo considere los tokens de mayor probabilidad k) y el muestreo de top-p (muestreo de núcleo: considere el conjunto más pequeño de tokens cuya probabilidad acumulada excede p). Compara la calidad de salida a temperatura 0,8 con top-k=50 vs top-p=0,95.

5. Construye un tractor de curva de pérdida de entrenamiento. Entrenar el modelo para 1000 pasos y la pérdida de gráfico vs paso. Identifique las tres fases: descenso inicial rápido (aprendizaje de bytes comunes), fase media más lenta (patrones de byte de aprendizaje) y planalto (overfitting en el corpus pequeño). La forma de esta curva es la misma si usted está entrenando un modelo de 128 dimensiones o GPT-4.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Autoregressive | "It generates one word at a time" | Each output token is conditioned on all previous tokens -- the model predicts P(token_n \| token_0, ..., token_{n-1}) |
| Causal mask | "It can't see the future" | An upper-triangular matrix of -infinity values that prevents attention to future positions during training |
| Multi-head attention | "Multiple attention patterns" | Splitting Q, K, V into parallel heads (e.g., 12 heads of 64 dims each for GPT-2) so each head can learn different relationship types |
| KV Cache | "Caching for speed" | Storing computed Key and Value tensors from previous tokens to avoid redundant computation during autoregressive generation |
| Prefill | "Processing the prompt" | The first inference phase where all prompt tokens are processed in parallel -- compute-bound on GPU FLOPS |
| Decode | "Generating tokens" | The second inference phase where tokens are generated one at a time -- memory-bound on GPU bandwidth |
| Weight tying | "Sharing embeddings" | Using the same matrix for input token embeddings and the output projection head -- saves 38M params in GPT-2 |
| Residual connection | "Skip connection" | Adding the input directly to the output of a sublayer (x + sublayer(x)) -- enables gradient flow in deep networks |
| Layer normalization | "Normalizing activations" | Normalizing across the feature dimension to mean 0 and variance 1, with learnable scale and bias parameters |
| Cross-entropy loss | "How wrong the predictions are" | -log(probability assigned to the correct next token), averaged over all positions -- the standard LLM training objective |

## Leer más

- [Radford et al., 2019 -- "Language Models are Unsupervised Multitask Learners" (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)-- el papel GPT-2 que introdujo la familia de parámetros 124M a 1.5B
- [Vaswani et al., 2017 -- "Attention Is All You Need"](https://arxiv.org/abs/1706.03762)-- el papel original transformador con la atención de producto punto escalado y atención multi-cabeza
- [Llama 3 Technical Report](https://arxiv.org/abs/2407.21783)-- cómo Meta escalaba la arquitectura GPT a 405B parámetros con GPUs 16K
- [Pope et al., 2022 -- "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102)-- el documento que formalizó preempleo vs decodificación y KV cache análisis
