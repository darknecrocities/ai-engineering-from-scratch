# Embedings de Word  Word2Vec desde cero

> Una palabra es la compañía que mantiene, y si se ejerce una red superficial sobre esa idea, la geometría se cae.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## El problema

TF-IDF sabe `dog`y `puppy`No se sabe que significan casi lo mismo.`dog`no puede generalizarse a una revisión sobre `puppy`Puedes revisar esto listando sinónimos, pero eso falla en términos raros, jerga de dominio y en cada lengua que no anticipaste.

¿ Quieres una representación donde ?`dog`y `puppy`Tierra cerca de uno en el espacio.`king - man + woman`tierras cercanas`queen`- Un modelo entrenado en`dog`Transfiere alguna señal a `puppy`- Por gratis.

Word2Vec nos dio ese espacio. Dos capas de red neuronal, trillones de tokens de entrenamiento, publicados en 2013. La arquitectura es casi vergonzosamente simple. Los resultados remodelaron la PNL durante una década.

## El concepto

**Distributional hypothesis**(Primero, 1957): "Conocerás una palabra por la compañía que mantiene". Si dos palabras aparecen en contextos similares, probablemente significan cosas similares.

Word2Vec viene en dos sabores, ambos explotando esa idea.

- **Skip-gram.**Dado una palabra central, predica las palabras circundantes.`cat -> (the, sat, on)`con el tamaño de la ventana 2.
- **CBOW (continuous bag of words).**Dadas las palabras circundantes, predica el centro.`(the, sat, on) -> cat`¿ Qué ?

El Skip-gram es más lento para entrenar pero maneja mejor las palabras raras.

La red tiene una capa oculta sin ninguna no linealidad. La entrada es un vector de una sola calidez sobre el vocabulario. La salida es una suave máxima sobre el vocabulario. Después del entrenamiento, se tira la capa de salida. Los pesos de la capa oculta son los embebidos.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

El truco: la máxima de más de 100 mil palabras es prohibitivamente cara.**negative sampling**Para convertirlo en una tarea de clasificación binaria. Prevé "¿apareció esta palabra de contexto cerca de esta palabra central, sí o no". Muestre un puñado de palabras negativas (no coincidentes) por par de entrenamiento en lugar de calcular softmax sobre todo el vocabulario.

```figure
word-vector-arithmetic
```

## Construye el mismo

### Paso 1: entrenamiento de pares desde un corpus

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

Cada par (centro, contexto) en una ventana es un ejemplo positivo de entrenamiento.

### Paso 2: incrustación de tablas

Dos matrices.`W`es la tabla de inserción de palabras centrales (la que mantiene). `W'`es la tabla de palabras contextuales (a menudo descartadas, a veces mediadas con `W`¿Qué es lo que se hace?

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

El tamaño de la vocab 10k y dim 100 es realista; para la enseñanza, 50 vocab x 16 dim es suficiente para ver la geometría.

### Paso 3: objetivo negativo de muestreo

Para cada par positivo `(center, context)`, muestra `k`Entrenando el modelo para que el producto punto`W[center] · W'[context]`es alto para los positivos y bajo para los negativos.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

La fórmula mágica: pérdida logística en el par positivo (queremos sigmoide cerca de 1) más pérdida logística en pares negativos (queremos sigmoide cerca de 0). Los gradientes fluyen hacia ambas tablas. La derivación completa está en el papel original; pase a través de él una vez con lápiz y papel si desea que se adhiera.

### Paso 4: entrenar en un cuerpo de juguete

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

Después de suficientes épocas en un gran corpus, las palabras que comparten contextos tienen un centro similar. En un corpus de juguete, se ve el efecto débilmente. En miles de millones de tokens, se ve dramáticamente.

### Paso 5: el truco de analogía

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

En vectores de noticias de Google 300d pre-entrenados:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`No porque el modelo sepa lo que es la realeza, porque el vector`(king - man)`captura algo como "real", y añadiéndolo a `woman`tierras cerca de la región de las mujeres reales.

## Usalo

Escribir Word2Vec desde cero es enseñar.`gensim`¿ Qué ?

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

Para el trabajo real, casi nunca entrenas Word2Vec tú mismo.

- **GloVe** El enfoque de factorizamiento de la matriz de cooccurrencia de Stanford. 50d, 100d, 200d, 300d puntos de control. Buena cobertura general. La lección 04 cubre específicamente GloVe.
- **fastText** La extensión Word2Vec de Facebook que incorpora n-gramas de caracteres.
- **Pretrained Word2Vec on Google News** 300d, vocabulario de palabras 3M, publicado 2013. Todavía descargado diariamente.

### Cuando Word2Vec todavía gane en 2026

- Entrenamiento en resúmenes médicos en una hora en una computadora portátil, obtener vectores especializados sin capturas de modelos generales.
- Ingeniería de características de estilo analógico. `gender_vector = mean(man - woman pairs)`...desde otras palabras para obtener un eje neutral de género.
- Interpretabilidad. 100d es lo suficientemente pequeño como para trazar a través de PCA o t-SNE y ver realmente forma de grupos.
- En cualquier lugar la inferencia tiene que ejecutarse en el dispositivo sin GPU.

### Donde Word2Vec falla

La pared de la polisemia.`bank`tiene un vector. `river bank`y `financial bank`Comparte con nosotros.`table`Un clasificador aguas abajo no puede distinguir los sentidos del vector.

Las incorporaciones contextuales (ELMo, BERT, cada transformador desde entonces) resolvieron esto produciendo un vector diferente para cada ocurrencia de la palabra en función del contexto circundante.

El problema de la falta de vocabulario es el otro fracaso.`Zoomer-approved`Si no se trata de datos de formación, no hay retroceso. fastText corrige esto con la composición de las palabras (lección 04).

## Envío

Salvo como`outputs/skill-embedding-probe.md`¿Qué es esto ?

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Los ejercicios

1. **Easy.**Realice el ciclo de entrenamiento en un pequeño corpus (20 frases sobre gatos y perros).`nearest(vocab, W, W[vocab["cat"]])`retorno `dog`En caso contrario, aumenta las épocas o el vocabulario.
2. **Medium.**Añadir submuestras de palabras frecuentes.`10^-5`Se evaluarán los resultados de las pruebas de formación en pares de formación con probabilidad proporcional a su frecuencia.
3. **Hard.**Entrenar un modelo en el corpus de 20 Newsgroups.`he - she`y `doctor - nurse`. Proyecto de palabras de ocupación en ambos ejes. informe cuáles ocupaciones tienen la mayor brecha de sesgo. este es el tipo de investigación de equidad de la sonda que usan los investigadores.

## Términos clave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Word embedding | Word as a vector | A dense, low-dim (typically 100-300) representation learned from context. |
| Skip-gram | Word2Vec trick | Predict context words from center word. Slower than CBOW, better for rare words. |
| Negative sampling | Training shortcut | Replace softmax over full vocab with binary classification against `k` random words. |
| Static embedding | One vector per word | Same vector regardless of context. Fails on polysemy. |
| Contextual embedding | Context-sensitive vector | Different vector for each occurrence based on surrounding words. What transformers produce. |
| OOV | Out of vocabulary | Word not seen in training. Word2Vec cannot produce a vector for these. |

## Leer más

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) el papel de muestreo negativo.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) la derivación más clara de los gradientes, si la matemática del papel original se siente densa.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) configuraciones de formación de producción que funcionen realmente.
