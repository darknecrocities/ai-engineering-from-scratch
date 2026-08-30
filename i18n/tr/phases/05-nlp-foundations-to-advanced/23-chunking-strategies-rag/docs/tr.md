# RAG için parçalanma stratejileri

> Çükleme yapılandırması, çekim kalitesini embed model seçimi kadar etkiliyor (Vectara NAACL 2025).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## Sorun

Bir RAG sistemine 50 sayfalık bir sözleşme koyulur. Kullanıcı "Küntüleme şartı nedir?" diye sorar. Retriever kapak sayfasını geri verir. Neden? Çünkü model 512 jeton parçacıklarında eğitilmiş ve sona erme şartı 20 sayfa içinde yer alır, bir sayfa kesintisi boyunca bölünmüş, yerel anahtar kelimeler olmadan soruya bağlanır.

Bu çözüm "daha iyi bir yerleştirme modeli satın alın" değil. Bu çözüm parçalanıyor. Ne kadar büyük?

Şubat 2026 referansları şaşırtıcı sonuçlar göstermektedir:

- Vectara'nın 2026 çalışması: Rekürsiv 512-token parçalanması semantik parçalanmayı 69% → 54% doğrulukla yendi.
- SPLADE + Mistral-8B on Natural Questions: örtüşmeler sıfır ölçülebilir fayda sağladı.
- Konekst uçurumu: yanıt kalitesi 2.500 bağlam tokeni etrafında keskin düşüyor.

"Açık" cevap (semantik parçalanma, %20 üst üste geçiş, 1000 token) genellikle yanlışdır. Bu ders altı strateji için sezgisellik oluşturur ve hangisine ulaşmanızı söyler.

## Anlaşım

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Fixed chunking.**N karakterleri ve işaretleri ayırın, baseline en basit, cümle ortasında kırılır, iyi sıkıştırma, kötü tutarlılık.

**Recursive.**LangChain'ın `RecursiveCharacterTextSplitter`- Ayrılmaya çalış .`\n\n`Önce, sonra.`\n`O zaman ...`.`2026'da geriye düştü.

**Semantic.**Her cümleyi yerleştir. Yakın cümleler arasında cosine benzerliği hesaplayın. benzerlik bir eşiğin altında düştüğünde bölün. Konu tutarlılığını korur. Daha yavaş; bazen geri alınmayı inciten küçük 40 jeton fragmanları üretir.

**Sentence.**Sıfır sınırlarına bölün. Bir cümle her parça veya N cümle penceresi.

**Parent-document.*** ve * daha büyük ebeveyn parçası bağlam için saklayın. Çocuklar tarafından alın; ebeveyn geri. Şımarık derecede: kötü çocuk parçaları hala makul ebeveynleri geri.

**Late chunking (2024).**Tüm belgeyi önce token seviyesinde yerleştirin, sonra token yerleşimlerini parça yerleşimlerine birleştirin. Çaplak bağlamı korur. Uzun bağlamlı yerleşimcileri ile çalışır (BGE-M3, Jina v3). Yüksek hesaplama.

**Contextual retrieval (Anthropic, 2024).**Her parçayı belgedeki pozisyonunun LLM tarafından oluşturulan bir özetle hazırlayın ("Bu parçacık sona erme şartlarının 3.2 bölümü...").

### Her defavalü kuralın üstesinden gelmek için

Bölüm boyutunu sorgu tipi ile eşleştir:

| Query type | Chunk size |
|------------|-----------|
| Factoid ("what is the CEO's name?") | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

NVIDIA'nın 2026 referansı. Bölüm cevap artı yerel bağlamı içerecek kadar büyük olmalı, retriever'in üst-K'i geri dönüşü bağlam gürültüsü yerine cevap üzerinde odaklanmak için yeterince küçük olmalı.

```figure
n5-chunk-cuts
```

## Yapın

### Adım 1: sabit ve rekürziv parçalanma

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

### Adım 2: semantik parçalanma

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

Tune `threshold`Çok yüksek, parçalar, çok düşük, dev bir parça.

### Adım 3: Ebeveyn belgesi

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

Ana babalar, çocukların aynı ebeveynle ilgili bir haritayı yapabilmeleri, hepsini geri vermek bağlamı boşa çıkarır.

### Adım 4: bağlamsal geri alım (Antropik model)

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

Arama sırasında, ekstra çevre sinyalleri ile elde edilen faydalar.

### Adım 5: değerlendirme

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

Her zaman referans göster. "En iyi" strateji, blog yazısına eşleşmeyebilir.

## Tuzaklar

- **Chunking evaluated only on factoid queries.**Çoklu hop sorguları çok farklı kazananları ortaya çıkarır. Sorgu tipi stratified eval seti kullanın.
- **Semantic chunking without a minimum size.**40 token fragmanı üretir ve bu da kurtarmayı incitir.`min_tokens`- Evet .
- **Overlap as cargo cult.**2026 çalışmaları, örtüşmenin genellikle sıfır fayda sağladığını ve indeksi maliyetini ikiye katladığını buldu.
- **No min/max enforcement.**5 veya 5000 tokenin parçaları her ikisi de çekimi bozar.
- **Cross-doc chunking.**Bir parça iki belgeyi asla uzatmasın.

## Kullan

2026'da:

| Situation | Strategy |
|-----------|----------|
| First build, unknown corpus | Recursive, 512 tokens, no overlap |
| Factoid QA | Recursive, 256-512 tokens |
| Analytical / multi-hop | Recursive, 512-1024 tokens + parent-document |
| Heavy cross-reference (contracts, papers) | Late chunking or contextual retrieval |
| Conversational / dialog corpus | Turn-level chunks + speaker metadata |
| Short utterances (tweets, reviews) | One document = one chunk |

512. 50 sorgu değerlendirme seti üzerinde hatırlama @ 5 ölçüm.

## Gönder

- Kaydet .`outputs/skill-chunker.md`- ...

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

## Egzersizler

1. **Easy.**Bir 20 sayfalık belgeyi sabit ((512, 0), rekursif ((512, 0), ve rekursif ((512, 100) ile parçalara sayılar ve sınır kalitesi ile karşılaştırın.
2. **Medium.**5 belge üzerinde 30 sorgu değerlendirme seti oluşturun. Rekürsiv, semantik ve ana belge için hatırlama@5 ölçün. Hangisi kazanır? Blog yayınlarına uymuş mu?
3. **Hard.**Konekstel geri alımı uygulayın. MRR'nin başlangıç geri dönüşü ile karşılaştırıldığında iyileşmesini ölçün. Endeks maliyetini (LLM çağrıları) doğruluk artışına karşı rapor edin.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Chunk | A piece of a doc | Sub-document unit that gets embedded, indexed, and retrieved. |
| Overlap | Safety margin | N tokens shared between adjacent chunks; often useless in 2026 benchmarks. |
| Semantic chunking | Smart chunking | Split where adjacent-sentence embedding similarity drops. |
| Parent-document | Two-level retrieval | Retrieve small children, return larger parents. |
| Late chunking | Chunk after embedding | Embed full doc at token level, pool into chunk vectors. |
| Contextual retrieval | Anthropic's trick | LLM-generated summary prepended to each chunk before indexing. |
| Context cliff | 2500-token wall | Quality drop observed around 2.5k context tokens in RAG (Jan 2026). |

## Daha Fazla Okumak

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) üretimde eksiklik.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) Seçimi yerleştirmek kadar önemli.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)- Geç zamanlı kağıt parçalanması.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) LLM'den kaynaklanan bağlam prefişleri ile %35-50% geri alım iyileştirilmesi.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) sorgu türüne göre parça boyutu.
