# आरएजी के लिए चकमक करने की रणनीति

> चकनाचूर विन्यास से निकासी की गुणवत्ता को उतना ही प्रभावित होता है जितना कि एम्बेडिंग मॉडल (वेक्टरा NAACL 2025) का चयन। गलत चकनाचूर हो जाओ और कोई भी री रैंकिंग आपको बचाता है।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## समस्या

आप एक 50 पृष्ठों का अनुबंध एक RAG प्रणाली में डालते हैं। उपयोगकर्ता पूछता हैः "समाप्ति खंड क्या है?" रिट्रीवर कवर पृष्ठ वापस करता है। क्यों? क्योंकि मॉडल 512 टोकन टुकड़ों पर प्रशिक्षित किया गया था और समापन खंड 20 पृष्ठों में बैठता है, एक पृष्ठ ब्रेक पर विभाजित, कोई स्थानीय कीवर्ड इसे क्वेरी से जोड़ता है।

फिक्स "एक बेहतर एम्बेडिंग मॉडल खरीदें" नहीं है। फिक्स है. बड़ा कितना है? ओवरलैप? कहाँ विभाजित करने के लिए? आसपास के संदर्भ के साथ?

फरवरी 2026 के बेंचमार्क आश्चर्यजनक परिणाम दिखाते हैंः

- वेक्टरा के 2026 के अध्ययन मेंः पुनरावर्ती 512-टोकन चकनिंग सेमंटिक चकनिंग को 69% → 54% सटीकता से हराया गया।
- प्राकृतिक प्रश्नों पर SPLADE + Mistral-8B: ओवरलैप ने शून्य मापने योग्य लाभ प्रदान किया।
- संदर्भ चट्टानः प्रतिक्रिया की गुणवत्ता में 2,500 संदर्भ टोकन के आसपास तेजी से गिरावट आई है।

"स्पष्ट" उत्तर (सैमंतिक टुकड़ा, 20% ओवरलैप, 1000 टोकन) अक्सर गलत होता है। यह पाठ छह रणनीतियों के लिए अंतर्ज्ञान बनाता है और आपको बताता है कि किसके लिए पहुंचना है।

## अवधारणा

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Fixed chunking.**N वर्णों या टोकन को विभाजित करें सबसे सरल मूल रेखा वाक्य के बीच में टूटता है अच्छा संपीड़न, खराब सुसंगतता।

**Recursive.**लैंगचेन की `RecursiveCharacterTextSplitter`. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .`\n\n`पहले, फिर `\n`, तो `.`2026 डिफ़ॉल्ट रूप से वापस गिरता है।

**Semantic.**प्रत्येक वाक्य को एम्बेड करें. आसन्न वाक्यों के बीच कॉसिन समानता की गणना करें. जहां समानता एक सीमा से नीचे गिरती है, विभाजित करें. विषय सुसंगतता बनाए रखता है। धीमा; कभी-कभी छोटे 40 टोकन टुकड़े उत्पन्न करता है जो पुनर्प्राप्त करने में नुकसान होता है।

**Sentence.**वाक्य सीमाओं पर विभाजित. प्रति टुकड़ा एक वाक्य या N वाक्य की एक खिड़की. लागत का एक अंश पर ~ 5k टोकन तक की अर्थपूर्ण टुकड़े मेल खाता है.

**Parent-document.*** और * संदर्भ के लिए बड़े माता-पिता के टुकड़े को स्टोर करें। बच्चे द्वारा पुनः प्राप्त करें; माता-पिता को वापस करें। सौंदर्यपूर्वक गिरावटः खराब बच्चे के टुकड़े अभी भी उचित माता-पिता को वापस करते हैं।

**Late chunking (2024).**पहले टोकन स्तर पर पूरे दस्तावेज़ को एम्बेड करें, फिर टुकड़े टुकड़े में टोकन एम्बेड को एक साथ रखें। क्रॉस-चट संदर्भ को संरक्षित करता है। लंबे संदर्भ एम्बेडर्स (BGE-M3, Jina v3) के साथ काम करता है। उच्च गणना।

**Contextual retrieval (Anthropic, 2024).**प्रत्येक टुकड़े को दस्तावेज़ में अपनी स्थिति के एक एलएलएम-जनरेट किए गए सारांश के साथ तैयार करें ("यह टुकड़ा समाप्ति खंडों का खंड 3.2 है ...") । मानव जाति के अपने बेंचमार्क में 35-50% सुधार प्राप्त करें। सूचकांक करने के लिए महंगा है।

### नियम जो हर डिफ़ॉल्ट से बेहतर है

क्वेरी प्रकार के लिए टुकड़ा आकार मेल खाता हैः

| Query type | Chunk size |
|------------|-----------|
| Factoid ("what is the CEO's name?") | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

NVIDIA के 2026 बेंचमार्क. टुकड़ा पर्याप्त बड़ा होना चाहिए जवाब और स्थानीय संदर्भ को शामिल करने के लिए, पर्याप्त छोटा है कि रिट्रीवर के शीर्ष-के रिटर्न संदर्भ शोर की बजाय उत्तर पर ध्यान केंद्रित करें।

```figure
n5-chunk-cuts
```

## इसे बनाओ

### चरण 1: स्थिर और पुनरावर्ती टुकड़े टुकड़े

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

### चरण 2: अर्थिक टुकड़ा

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

ट्यूनिंग`threshold`अपने डोमेन पर. बहुत उच्च → टुकड़े. बहुत कम → एक विशाल टुकड़ा.

### चरण 3: माता-पिता का दस्तावेज

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

एक ही माता-पिता के पास कई बच्चे जा सकते हैं, सब कुछ वापस करना संदर्भ को बर्बाद कर देगा।

### चरण 4: संदर्भिक निकासी (मानविकीय पैटर्न)

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

संदर्भित टुकड़ों को सूचकांकित करें। क्वेरी के समय, अतिरिक्त आसपास के संकेत से प्राप्त लाभ।

### चरण 5: मूल्यांकन

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

हमेशा बेंचमार्क करें. आपके कॉर्पस के लिए "सर्वश्रेष्ठ" रणनीति किसी भी ब्लॉग पोस्ट से मेल नहीं खा सकती है।

## फंदे

- **Chunking evaluated only on factoid queries.**मल्टी-हॉप क्वेरी बहुत अलग विजेताओं का पता चलता है। क्वेरी प्रकार-परतबद्ध मूल्यांकन सेट का उपयोग करें।
- **Semantic chunking without a minimum size.**40 टोकन टुकड़े जो पुनर्प्राप्ति को चोट पहुंचाता है।`min_tokens`. .
- **Overlap as cargo cult.**2026 के अध्ययनों में पाया गया है कि ओवरलैप अक्सर शून्य लाभ प्रदान करता है और सूचकांक लागत को दोगुना करता है।
- **No min/max enforcement.**5 टोकन या 5000 टोकन के टुकड़े दोनों निकालने को तोड़ते हैं।
- **Cross-doc chunking.**एक टुकड़ा दो दस्तावेजों को कभी भी नहीं खींचता है।

## इसका प्रयोग करें

2026 स्टैकः

| Situation | Strategy |
|-----------|----------|
| First build, unknown corpus | Recursive, 512 tokens, no overlap |
| Factoid QA | Recursive, 256-512 tokens |
| Analytical / multi-hop | Recursive, 512-1024 tokens + parent-document |
| Heavy cross-reference (contracts, papers) | Late chunking or contextual retrieval |
| Conversational / dialog corpus | Turn-level chunks + speaker metadata |
| Short utterances (tweets, reviews) | One document = one chunk |

पुनरावर्ती 512 से शुरू करें। 50 प्रश्न मूल्यांकन सेट पर याद@5 मापें। वहां से ट्यून करें।

## इसे भेजें

`outputs/skill-chunker.md`:

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

## व्यायाम

1. **Easy.**एक 20 पन्नों के दस्तावेज़ को फिक्स्ड ((512, 0), रिकर्सिव ((512, 0), और रिकर्सिव ((512, 100) के साथ टुकड़ा करें। टुकड़े की गिनती और सीमा गुणवत्ता की तुलना करें।
2. **Medium.**5 दस्तावेजों पर 30 प्रश्नों का मूल्यांकन सेट बनाएं। रिकर्सिव, अर्थिक और मूल-कार्य के लिए recall@5 मापें। कौन जीता? क्या यह ब्लॉग पोस्ट से मेल खाता है?
3. **Hard.**संदर्भ पुनर्प्राप्ति लागू करें। बेसलाइन रिकर्सिव पर एमआरआर में सुधार को मापें। सूचकांक लागत (एलएलएम कॉल) बनाम सटीकता वृद्धि की रिपोर्ट करें।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Chunk | A piece of a doc | Sub-document unit that gets embedded, indexed, and retrieved. |
| Overlap | Safety margin | N tokens shared between adjacent chunks; often useless in 2026 benchmarks. |
| Semantic chunking | Smart chunking | Split where adjacent-sentence embedding similarity drops. |
| Parent-document | Two-level retrieval | Retrieve small children, return larger parents. |
| Late chunking | Chunk after embedding | Embed full doc at token level, pool into chunk vectors. |
| Contextual retrieval | Anthropic's trick | LLM-generated summary prepended to each chunk before indexing. |
| Context cliff | 2500-token wall | Quality drop observed around 2.5k context tokens in RAG (Jan 2026). |

## आगे पढ़ना

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) उत्पादन में चूक।
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) चश्मा करना उतना ही महत्वपूर्ण है जितना कि चयन को शामिल करना।
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) देर से कागज का टुकड़ा।
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) एलएलएम-जनित संदर्भ पूर्वावधानों के साथ 35-50% पुनर्प्राप्ति में सुधार।
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) क्वेरी प्रकार के अनुसार टुकड़ा आकार।
