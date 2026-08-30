# 碎的RAG策略

> 碎配置对检索质量的影响与嵌入模型的选择一样 (Vectara NAACL 2025). 错误碎,没有重排可以挽救您.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## 问题

你把50页的合同放入RAG系统中.用户问:"终止条款是什么?"检索器返回封面页面.为什么?因为模型是用512个代币块训练的,终止条款位于20页面,分开在一个页面间,没有任何本地关键词将其绑定到查询.

解决方案不是"买一个更好的嵌入式模型".解决方案是碎. 它们有多大? 叠加? 它们在哪里分裂?

2026年2月的基准显示出令人惊的结果:

- 韦卡塔拉的2026研究:复发性512代币分断超过语义分断69% → 54%的准确性.
- 关于自然问题的SPLAD+Mistral-8B:重叠提供了零可测量的好处.
- 语境悬崖:响应质量大幅下降,

答案"显而易见" (语义分量,20%重叠,1000个代币) 通常是错误的.

## 概念

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Fixed chunking.**单词的基本线,句子中断,压缩不好,连贯不好.

**Recursive.**长链的`RecursiveCharacterTextSplitter`试着分开.`\n\n`首先,然后`\n`现在`.`现在,我们在2026年,我们在太空中,

**Semantic.**编入每个句子.计算相邻句子之间的相似性. 分开相似性下降于门.保持主题一致性.慢慢;有时产生小的40代币碎片,损害检索.

**Sentence.**按句子边界划分.每句子一个句子或一个N句子的窗口. 匹配到5k代币的语义分数.

**Parent-document.**保存小小的儿童块来检索 *和*较大的父母块来文本. 按孩子检索; 返回父母. 优雅地降级:坏的孩子块仍然返回合理的父母.

**Late chunking (2024).**首先将整个文档嵌入于代币级别,然后将代币嵌入集成到块嵌入. 保存交叉部分文本. 与长文本嵌入器 (BGE-M3,Jina v3) 进行运作. 高计算.

**Contextual retrieval (Anthropic, 2024).**预备每一个部分,并将其在文档中的位置的LLM生成总结 ("本部分是终止条款的3.2节").

### 规则是胜过任何默认的

匹配部分大小与查询类型:

| Query type | Chunk size |
|------------|-----------|
| Factoid ("what is the CEO's name?") | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

部分应该足够大,以包含答案加上本地背景,足够小,使得回收器的顶级K回报集中在答案而不是背景噪音.

```figure
n5-chunk-cuts
```

## 建立它

### 步骤1:固定和递归的碎片化

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

### 步骤2:语义分断

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

调整`threshold`太高,太低,太大.

### 步骤3:父母文件

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

许多孩子可以把一个父母带到一个地方,把所有东西都带回来了.

### 步骤4:文本检索 (人类模式)

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

在查询时,检索从周围的额外信号中获益.

### 步骤5:评估

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

对于你的体验,最好的策略可能不符合任何博客文章.

## 陷

- **Chunking evaluated only on factoid queries.**通过多次查询,可以发现不同的获胜者.
- **Semantic chunking without a minimum size.**产生40个代币碎片,损害检索.`min_tokens`现在,我们要去.
- **Overlap as cargo cult.**根据2026年研究的结果,重叠通常提供零效益,并且增加了指数成本.
- **No min/max enforcement.**五个代币或5000个代币的碎片都能打破检索.
- **Cross-doc chunking.**永远不要让一个文件跨越两个文件,每份文件,然后合并.

## 用它

现在,我们要做什么?

| Situation | Strategy |
|-----------|----------|
| First build, unknown corpus | Recursive, 512 tokens, no overlap |
| Factoid QA | Recursive, 256-512 tokens |
| Analytical / multi-hop | Recursive, 512-1024 tokens + parent-document |
| Heavy cross-reference (contracts, papers) | Late chunking or contextual retrieval |
| Conversational / dialog corpus | Turn-level chunks + speaker metadata |
| Short utterances (tweets, reviews) | One document = one chunk |

开始于复式512. 测量50查询评估设置的回忆@5. 从那里调整.

## 运送它

保存如`outputs/skill-chunker.md`其他:

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

## 运动

1. **Easy.**按固定512,0),递归512,0和递归512,100的20页文件进行分数.
2. **Medium.**建立一个30个查询评估集在5个文档上. 测量回复性,语义和父母文档的回忆@5. 哪个获胜? 它与博客帖子匹配吗?
3. **Hard.**实现文本检索. 测量MRR改善与基线递归. 报告指数成本 (LLM调用) 与准确度增长.

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Chunk | A piece of a doc | Sub-document unit that gets embedded, indexed, and retrieved. |
| Overlap | Safety margin | N tokens shared between adjacent chunks; often useless in 2026 benchmarks. |
| Semantic chunking | Smart chunking | Split where adjacent-sentence embedding similarity drops. |
| Parent-document | Two-level retrieval | Retrieve small children, return larger parents. |
| Late chunking | Chunk after embedding | Embed full doc at token level, pool into chunk vectors. |
| Contextual retrieval | Anthropic's trick | LLM-generated summary prepended to each chunk before indexing. |
| Context cliff | 2500-token wall | Quality drop observed around 2.5k context tokens in RAG (Jan 2026). |

## 进一步阅读

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/)生产违约.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) 碎碎就像嵌入选择一样重要.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)晚期的纸张.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)通过LLM生成的文本前置,恢复率提高了35-50%.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/)按查询类型的部分大小.
