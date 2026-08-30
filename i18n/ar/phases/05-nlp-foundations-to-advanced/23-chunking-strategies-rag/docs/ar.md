# استراتيجيات التجزئة لـ RAG

> يؤثر تشكيل التقطيع على جودة الاسترداد بقدر ما يؤثر على اختيار نموذج التضمين (Vectara NAACL 2025). الحصول على التقطيع الخطأ ولا توفر لك أي كمية من إعادة التصنيف.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## المشكلة

يمكنك وضع عقد من 50 صفحة في نظام RAG. يسأل المستخدم: "ما هي شفرة الإيقاف؟" يستعيد المستخدم الصفحة الغلافية. لماذا؟ لأن النموذج تم تدريبه على 512 رمزية و شفرة الإيقاف تقع في 20 صفحة، مقسمة على طول فترة الصفحة، دون أي كلمات رئيسية محلية ربطها بالمسألة.

الحل ليس "شراء نموذج أفضل من إضافة" الحل هو التكثيف. كم هو كبير؟ تتداخل؟ أين تقسيم؟ مع السياق المحيط؟

في فبراير 2026 أظهرت النتائج المدهشة:

- دراسة فيكتارا لعام 2026: التجزئة الترددية 512 رمزية تفوق التجزئة التجميلية 69٪ → 54٪ دقة.
- SPLADE + Mistral-8B على الأسئلة الطبيعية: التداخل يقدم صفر فائدة قابلة للقياس.
- صعوداً سياقية: إنخفاض نوعية الاستجابة بشكل حاد حول 2500 رمز من السياق.

الجواب "الوضوح" (القطع النطاقي، التداخل بنسبة 20٪، 1000 رمز) غالباً ما يكون خاطئاً. هذه الدروس تبني الحسّاسة لست استراتيجيات وتخبرك متى تبحث عن أيّ منها.

## المفهوم

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Fixed chunking.**تقسيم كل حرف أو رمز N، أسهل خط أساسي، يقطع في منتصف الجملة، ضغط جيد، منسقة سيئة.

**Recursive.**لاندشين `RecursiveCharacterTextSplitter`حاول أن تفرق`\n\n`أولاً، ثم`\n`، ثم`.`ثم الفضاء، يُرجعُ إلى الوراء نظيفًا، الوضع الراهن لعام 2026.

**Semantic.**تضمين كل جملة. حساب تشابه كوسين بين الجمل المجاورة. تقسيم حيث تنخفض الشبيهة دون عتبة. يحافظ على التماسك الموضوع. أبطأ؛ في بعض الأحيان ينتج شظايا صغيرة 40 رمزية تؤذي الاسترداد.

**Sentence.**تقسيم على حدود الجملة. جملة واحدة لكل جزء أو نافذة من الجمل N. يطابق الجملة التعبرية حتى ~ 5k رموز بتكلفة صغيرة.

**Parent-document.**تخزين قطع صغيرة من الأطفال لاسترداد * و* الجزء الأكبر من الوالدين للسياق. استرداد من قبل الطفل؛ عودة الوالد. تخفيضات gracefully: قطع الطفل السيئة لا يزال يعود الآباء المعقولين.

**Late chunking (2024).**تضمين الوثيقة بأكملها على مستوى الرمز أولاً ، ثم تجمع تضمين الرمز في تضمينات قطع. يحافظ على السياق المتقاطع. يعمل مع تضمينات السياق الطويل (BGE-M3 ، Jina v3). حساب أعلى.

**Contextual retrieval (Anthropic, 2024).**قم بإعداد كل جزء مع ملخص من الموقف الذي تم إنشاؤه من قبل الجامعة في الوثيقة ("هذا الجزء هو القسم 3.2 من فقرات الإنهاء ..."). تحسن استرداد 35-50٪ في مقياس الإنتروبيك الخاص. مكلف للتصفية.

### القاعدة التي تفوز بكل إغراض

تطابق حجم الجزء مع نوع الاستفسار:

| Query type | Chunk size |
|------------|-----------|
| Factoid ("what is the CEO's name?") | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

مقياس نفيديا لعام 2026، يجب أن يكون الجزء كبيرًا بما يكفي لتحتوي على الإجابة بالإضافة إلى السياق المحلي، صغيرًا بما يكفي لتركيز أعلى K من جهاز الاسترداد على الإجابة بدلاً من ضجيج السياق.

```figure
n5-chunk-cuts
```

## بناءها

### الخطوة الأولى: التقطيع الثابت والمرجعي

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

### الخطوة الثانية: التجزئة التفاصلية

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

أجهز`threshold`على مجالكم، عالٍ جداً → شظايا، منخفضة جداً → قطعة ضخمة

### الخطوة الثالثة: وثيقة الوالدين

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

فكرة رئيسية: والداً من المتخلفين. يمكن لأطفال عدة أن يصلوا إلى نفس الوالد؛ وإعادة كل شيء سيكون ضائعاً بالسياق.

### الخطوة الرابعة: الاستعراض السياسي (النمط الأنثروبي)

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

إدراج الكتل الموضعية في وقت البحث، الاستعراض يستفيد من الإشارة المحيطة الإضافية.

### الخطوة 5: تقييم

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

دائماً مقارنة، قد لا تتطابق استراتيجية "أفضل" للكوربوس الخاص بك مع أي من مقالات المدونة.

## الفخاخ

- **Chunking evaluated only on factoid queries.**استفسارات متعددة المكالمات تكشف عن الفائزين المختلفين جداً. استخدم مجموعة تقييم مستطيلة من نوع الاستفسار.
- **Semantic chunking without a minimum size.**ينتج 40 شريحة رمزية تؤذي الاسترداد`min_tokens`. . .
- **Overlap as cargo cult.**وجدت دراسات 2026 التداخل غالبا ما توفر صفر الفائدة وتضاعف تكلفة المؤشر.
- **No min/max enforcement.**قطع من 5 رموز أو 5000 رموز كلاهما يفسد استرداد.
- **Cross-doc chunking.**لا تدع قطعة واحدة تتفوق على وثائقين دائماً قطعة لكل وثيقة ثم تجمع

## استخدمها

"مجموعة 2026"

| Situation | Strategy |
|-----------|----------|
| First build, unknown corpus | Recursive, 512 tokens, no overlap |
| Factoid QA | Recursive, 256-512 tokens |
| Analytical / multi-hop | Recursive, 512-1024 tokens + parent-document |
| Heavy cross-reference (contracts, papers) | Late chunking or contextual retrieval |
| Conversational / dialog corpus | Turn-level chunks + speaker metadata |
| Short utterances (tweets, reviews) | One document = one chunk |

ابدأ بالخلفية 512. قم بعد الاحتجاز @ 5 على مجموعة تقييم 50 استفسار. قم بتحديد من هناك.

## أرسله

إبقوا`outputs/skill-chunker.md`:

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

## التمارين

1. **Easy.**تقسيم واحد من 20 صفحة وثيقة مع ثابتة ((512, 0) ، الترددي ((512, 0) ، والترددي ((512, 100). مقارنة عدد القسط والجودة الحدودي.
2. **Medium.**قم ببناء مجموعة تقييم 30 سؤالًا على 5 وثائق. قم بقياس recall@5 للوثيقة الترددية والمعنوية والوالدة. من الذي يفوز؟ هل يطابق مشاركات المدونة؟
3. **Hard.**تنفيذ الاستعراض السياسي. قياس تحسن MRR على الخط الأساسي التكراري. تقرير تكلفة المؤشر (دعوات LLM) مقابل زيادة الدقة.

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Chunk | A piece of a doc | Sub-document unit that gets embedded, indexed, and retrieved. |
| Overlap | Safety margin | N tokens shared between adjacent chunks; often useless in 2026 benchmarks. |
| Semantic chunking | Smart chunking | Split where adjacent-sentence embedding similarity drops. |
| Parent-document | Two-level retrieval | Retrieve small children, return larger parents. |
| Late chunking | Chunk after embedding | Embed full doc at token level, pool into chunk vectors. |
| Contextual retrieval | Anthropic's trick | LLM-generated summary prepended to each chunk before indexing. |
| Context cliff | 2500-token wall | Quality drop observed around 2.5k context tokens in RAG (Jan 2026). |

## المزيد من القراءة

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) عدم إدارة الإنتاج
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) تقطيع الأمور بقدر ما تضمين الخيار
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)-الورقة المتأخرة
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) تحسين 35-50% في الاسترداد مع المضادات السياقية التي تم إنشاؤها من خلال ماجستير في التدريب.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) حجم الجزء حسب نوع الاستفسار.
