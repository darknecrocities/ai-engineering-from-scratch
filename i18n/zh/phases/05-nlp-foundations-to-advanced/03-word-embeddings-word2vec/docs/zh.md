# 从零开始 Word2Vec 嵌入式

> 对于这个想法,就会有微小的网络,而几何则会掉下来.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## 问题

国际特种基金会知道`dog`其他`puppy`它们几乎意味着同一个东西.`dog`无法将其总体化为关于`puppy`您可以通过列出同义词来记录下这一点, 但这在罕见的术语,域名语和你不预料的每一种语言上都失败了.

你想要一个代表,`dog`其他`puppy`太空中的陆地.`king - man + woman`附近的土地`queen`模型在哪里训练`dog`传输一些信号到`puppy`免费的.

Word2Vec给了我们这个空间. 两个层神经网络,数万亿代币的训练运行,发表于2013年. 架构几乎是令人尬的简单.

## 概念

**Distributional hypothesis**"你会从一个词的朋友中知道" (第一个,1957年).

两种风味,两种利用这个想法.

- **Skip-gram.**给一个中心词,预测周围的词.`cat -> (the, sat, on)`窗口尺寸2
- **CBOW (continuous bag of words).**根据周围的词汇,预测中心.`(the, sat, on) -> cat`现在,我们要去.

跳转语法训练速度较慢,但处理稀有词语更好.

网络有一个隐藏的层,没有线性.输入是词汇上的一个热向量.输出是词汇上的软最大.训练后,你扔掉输出层.隐藏的层重量是嵌入.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

软max超过100万字是非常昂贵的.**negative sampling**预测"这个文本词是否出现在这个中文字附近,是的或是的". 通过每一个训练对的少数负面 (非同发生) 字样,而不是计算整个词汇中的软max.

```figure
word-vector-arithmetic
```

## 建立它

### 步骤1:从一个体内训练对

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

每个窗口中的 (中心,背景) 双是积极的训练例子.

### 步骤 2:嵌入表

两个矩阵.`W`是一个中文字嵌入表 (你保留的表).`W'`文本词表 (通常被丢弃,有时平均为`W`)

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

字母大小10k和色100是现实的;用于教学,50字母×16色足以看到几何.

### 步骤3:负样本目标

对于每一个正数对`(center, context)`样本`k`训练模型,所以点产量`W[center] · W'[context]`对于积极的情况来说,高,对于负面的情况来说,低.

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

魔术公式:正对的物流损失 (想要sigmoid接近1) 加上负对的物流损失 (想要sigmoid接近0). 渐变体向两个表流动.完整的衍生是在原始纸上;如果你想它粘着,用笔和纸一次穿过它.

### 步骤4:在玩具体上训练

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

在一个大型的体积上,经过足够的时代,分享背景的词语具有类似的中心嵌入.在一个玩具体积上,你看到了效果微弱.在数十亿的代币上,你看到了戏剧性.

### 步骤5:比喻技巧

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

在预训练的300d谷歌新闻载体上:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`不是因为模型知道皇室是什么,而是因为向量`(king - man)`像"皇家"这样的东西,`woman`靠近皇室妇女地区的土地.

## 用它

编写Word2Vec从零开始就是教学.`gensim`现在,我们要去.

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

对于真正的工作,你几乎从来没有训练Word2Vec.

- **GloVe**斯坦福的共发生矩阵因数化方法. 50d, 100d, 200d, 300d检查站.
- **fastText**Facebook的 Word2Vec扩展,嵌入了字符n图.通过编写子词来处理词汇库之外的单词. 第04课.
- **Pretrained Word2Vec on Google News** 300d,3M字词库,发表于 2013. 仍然每天下载.

### 在2026年Word2Vec仍然赢得胜利时

- 通过笔记本电脑,在一个小时内训练医学摘要,获得专业的向量,没有一般模型捕获.
- 类似的特征工程.`gender_vector = mean(man - woman pairs)`现在还在公平研究中使用.
- 它们可以通过PCA或t-SNE绘制图,并实际上看到集群的形成.
- 任何地方的推断都必须在设备上运行,没有GPU.

### Word2Vec 失败的地方

聚墙.`bank`只有一个向量.`river bank`其他`financial bank`让我们分享.`table`后游分类器不能区分感官和向量.

基于周围的环境,语境嵌入式 (ELMo,BERT,自此以来的每个变压器) 通过根据周围的环境生成一个不同的向量来解决这一问题.这就是从Word2Vec到BERT的跳跃:从静态到语境.第7阶段涵盖变压器的一半.

其他失败是词汇缺失问题.`Zoomer-approved`如果没有在训练数据中.没有倒退. fastText通过子词组合来解决这一问题 (课04).

## 运送它

保存如`outputs/skill-embedding-probe.md`其他:

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

## 运动

1. **Easy.**经过200个时代,检查 子的子.`nearest(vocab, W, W[vocab["cat"]])`收益`dog`如果没有,则增加时代或词汇库.
2. **Medium.**增加频率字的子样本.`10^-5`测量对稀有词的相似性的影响.
3. **Hard.**根据20个新闻群体的模型进行训练.`he - she`其他`doctor - nurse`报告哪些职业有最大的偏差差. 这种类型的探测器公平性研究人员使用.

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Word embedding | Word as a vector | A dense, low-dim (typically 100-300) representation learned from context. |
| Skip-gram | Word2Vec trick | Predict context words from center word. Slower than CBOW, better for rare words. |
| Negative sampling | Training shortcut | Replace softmax over full vocab with binary classification against `k` random words. |
| Static embedding | One vector per word | Same vector regardless of context. Fails on polysemy. |
| Contextual embedding | Context-sensitive vector | Different vector for each occurrence based on surrounding words. What transformers produce. |
| OOV | Out of vocabulary | Word not seen in training. Word2Vec cannot produce a vector for these. |

## 进一步阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)负样本纸. 简短可读.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738)最清楚的梯度衍生,如果原始论文的数学感觉密集.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html)实际上有效的生产培训设置.
