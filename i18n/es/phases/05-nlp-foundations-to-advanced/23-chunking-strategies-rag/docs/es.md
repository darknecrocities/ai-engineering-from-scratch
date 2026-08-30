# Estrategias de descomposición para RAG

> La configuración de fragmentos influye tanto en la calidad de recuperación como en la elección del modelo de incorporación (Vectara NAACL 2025).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## El problema

El usuario pregunta: "¿Cuál es la cláusula de terminación?" El retriever devuelve la página de portada. ¿Por qué? Porque el modelo fue entrenado en 512 tokens y la cláusula de terminación se encuentra en 20 páginas, divididas en una pausa de página, sin palabras clave locales que la vinculen a la consulta.

La solución no es "comprar un mejor modelo de incorporación", sino la solución es la descomposición. ¿Qué tan grande? ¿Cuál es la superposición? ¿Dónde se divide?

Los índices de referencia de febrero de 2026 muestran resultados sorprendentes:

- El estudio de Vectara de 2026: el chunking recursivo de 512 tokens supera el chunking semántico con una precisión del 69% → 54%.
- SPLADE + Mistral-8B sobre cuestiones naturales: la superposición proporcionó cero beneficio medible.
- Clípico de contexto: la calidad de la respuesta cae drásticamente alrededor de 2.500 tokens de contexto.

La respuesta "obvia" (cumplimiento semántico, superposición del 20%, 1000 tokens) a menudo es incorrecta.

## El concepto

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Fixed chunking.**Divide todos los N caracteres o tokens. línea de base más simple. rompe a mitad de la oración. buena compresión, mala coherencia.

**Recursive.**La cadena de LangChain `RecursiveCharacterTextSplitter`Intentaremos dividirnos .`\n\n`Primero, luego.`\n`, entonces`.`Se vuelve a la normalidad para 2026.

**Semantic.**Incorporar cada frase. Computa similitud cosina entre oraciones adyacentes. Divida donde la similitud cae por debajo de un umbral. Preserva coherencia del tema. Más lento; a veces produce pequeños fragmentos de 40 tokens que perjudican la recuperación.

**Sentence.**Se divide en límites de oraciones. Una oración por pieza o una ventana de N oraciones.

**Parent-document.**Guardar pequeños trozos de niños para la recuperación * y * el mayor parental para el contexto. Recuperar por niño; regresar padre. Degradas graciosamente: los trozos de niños malos todavía devuelven a los padres razonables.

**Late chunking (2024).**Embed todo el documento en el nivel de token primero, luego comparte las incorporaciones de token en embebedings de fragmentos. Preserva el contexto de fragmentos cruzados. Funciona con embebedders de contexto largo (BGE-M3, Jina v3).

**Contextual retrieval (Anthropic, 2024).**Prepare cada pieza con un resumen generado por el LLM de su posición en el documento ("Este pieza es la sección 3.2 de las cláusulas de terminación..."). 35-50% de mejoría de recuperación en el propio índice de referencia de Anthropic.

### La regla que supera a todos los valores

Aparezca el tamaño de la pieza con el tipo de consulta:

| Query type | Chunk size |
|------------|-----------|
| Factoid ("what is the CEO's name?") | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

El punto de referencia de NVIDIA para 2026: el fragmento debe ser lo suficientemente grande como para contener la respuesta más el contexto local, lo suficientemente pequeño como para que los retornos de la parte superior del retriever se centren en la respuesta en lugar del ruido del contexto.

```figure
n5-chunk-cuts
```

## Construye el mismo

### Paso 1: despeje fijo y recursivo

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### Paso 2: fragmentación semántica

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

La música`threshold`En su dominio. demasiado alto → fragmentos. demasiado bajo → una pieza gigante.

### Paso 3: Documento de los padres

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

La clave: padres dedupidos. Múltiples hijos pueden mapear al mismo padre; devolverlos todos desperdiciaría el contexto.

### Paso 4: Recuperación contextual (patrón antropológico)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

Indica los fragmentos contextualizados. En el momento de la consulta, la recuperación se beneficia de la señal adicional circundante.

### Paso 5: evaluar

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

Siempre fija un punto de referencia. La "mejor" estrategia para tu cuerpo puede no coincidir con ninguna publicación de blog.

## Las trampas

- **Chunking evaluated only on factoid queries.**Las consultas de múltiples pasos revelan ganadores muy diferentes.
- **Semantic chunking without a minimum size.**Produce 40 fragmentos de tokens que perjudican la recuperación.`min_tokens`¿ Qué ?
- **Overlap as cargo cult.**Los estudios de 2026 encuentran que la superposición a menudo proporciona beneficios cero y duplica el coste del índice.
- **No min/max enforcement.**Los trozos de 5 tokens o 5000 tokens rompen la recuperación.
- **Cross-doc chunking.**Nunca dejes que una pieza se extienda a dos documentos, siempre por pieza y luego se fusionan.

## Usalo

La pila de 2026:

| Situation | Strategy |
|-----------|----------|
| First build, unknown corpus | Recursive, 512 tokens, no overlap |
| Factoid QA | Recursive, 256-512 tokens |
| Analytical / multi-hop | Recursive, 512-1024 tokens + parent-document |
| Heavy cross-reference (contracts, papers) | Late chunking or contextual retrieval |
| Conversational / dialog corpus | Turn-level chunks + speaker metadata |
| Short utterances (tweets, reviews) | One document = one chunk |

Comience con el recursivo 512. Medir recall@5 en un conjunto de eval de 50 consultas.

## Envío

Salvo como`outputs/skill-chunker.md`¿Qué es esto ?

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## Los ejercicios

1. **Easy.**Comparar el número de piezas y la calidad de los límites.
2. **Medium.**Construir un conjunto de evaluaciones de 30 consultas en 5 documentos. Medir recall@5 para el documento recursivo, semántico y padre. ¿Cuál gana? ¿Combina con las publicaciones del blog?
3. **Hard.**Implementar la recuperación contextual. Medir la mejora de MRR en relación con el índice de referencia recursivo.

## Términos clave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Chunk | A piece of a doc | Sub-document unit that gets embedded, indexed, and retrieved. |
| Overlap | Safety margin | N tokens shared between adjacent chunks; often useless in 2026 benchmarks. |
| Semantic chunking | Smart chunking | Split where adjacent-sentence embedding similarity drops. |
| Parent-document | Two-level retrieval | Retrieve small children, return larger parents. |
| Late chunking | Chunk after embedding | Embed full doc at token level, pool into chunk vectors. |
| Contextual retrieval | Anthropic's trick | LLM-generated summary prepended to each chunk before indexing. |
| Context cliff | 2500-token wall | Quality drop observed around 2.5k context tokens in RAG (Jan 2026). |

## Leer más

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) el incumplimiento de la producción.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) La descomposición es tan importante como la incorporación de la elección.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) el papel de trozo tardío.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) 35-50% mejoría en la recuperación con prefijos de contexto generados por el LLM.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) tamaño de pieza por tipo de consulta.
