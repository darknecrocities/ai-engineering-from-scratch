# إضافة الكلمة  Word2Vec من الصفر

> كلمة هي الشركة التي تبقيها، قم بتدريب شبكة سطحية على تلك الفكرة و الهندسة تسقط.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## المشكلة

يعرف (تف-إيدف)`dog`و`puppy`إنها كلمات مختلفة. لا يعرف أنها تعني نفس الشيء تقريبا.`dog`لا يمكن أن تجميع إلى مراجعة حول `puppy`يمكنك أن تكتب على هذا عن طريق إدراج المختلفات، ولكن هذا يفشل في المصطلحات النادرة، الجارغون المجال، وكل لغة لم تتوقعها.

تريد تمثيلاً حيث`dog`و`puppy`الأرض قريبة من بعضها البعض في الفضاء`king - man + woman`أرض قريبة`queen`حيث تم تدريب عارضة`dog`يُنقل بعض الإشارات إلى`puppy`مجاناً

Word2Vec أعطتنا هذا الفضاء. شبكة عصبية طبقتين، تم تشغيل تريليون رمز، نشرت في عام 2013. الهندسة المعمارية بسيطة بشكل محرج تقريبا. النتائج أعادت تشكيل NLP لمدة عقد.

## المفهوم

**Distributional hypothesis**(الأول، 1957): "تعرف كلمة من خلال الشركة التي تحافظ عليها". إذا ظهرت كلمتان في سياقين متشابهين، فمن المحتمل أن تعني أشياء متشابهة.

Word2Vec يأتي في طعمين، كلاهما يستغل هذه الفكرة.

- **Skip-gram.**مع كلمة مركزية، توقع الكلمات المحيطة.`cat -> (the, sat, on)`مع حجم النافذة 2.
- **CBOW (continuous bag of words).**بالنظر إلى الكلمات المحيطة، توقع المركز.`(the, sat, on) -> cat`. . .

إنّ لغة "سكيپ-جرام" بطيئة في التدريب، ولكنها تتعامل مع الكلمات النادرة بشكل أفضل.

الشبكة لديها طبقة واحدة مخفية بدون عدم وجود خطية. المدخل هو متجه واحد حار على المفردات. الخروج هو softmax على المفردات. بعد التدريب، يمكنك إلقاء الطبقة الخروج. الوزن الطبقة الخفية هي التوابع.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

الخدعة: "البالغة المكثفة من 100 ألف كلمة" مكلفة للغاية.**negative sampling**تحويلها إلى مهمة تصنيف ثنائية. توقع "هل ظهرت هذه الكلمة السياقية بالقرب من هذه الكلمة المركزية ، نعم أو لا". قم بعمل عينة من الكلمات السلبية (غير المتواصلة) لكل زوج تدريب بدلاً من حساب softmax على كل المفردات.

```figure
word-vector-arithmetic
```

## بناءها

### الخطوة الأولى: أزواج التدريب من مجموعة

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

كل زوج (مركز، سياق) في نافذة هو مثال إيجابي للتدريب.

### الخطوة الثانية: إضافة الجداول

-ماثريتين`W`هو جدول إضافة الكلمة المركزية (الذي تحمله). `W'`هو جدول الكلمات السياقية (غالبا ما يتم التخلص منها، في بعض الأحيان يتم متوسطها مع `W`)

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

الحجم الكلامي 10k و dim 100 هو واقعي؛ للتدريس، 50 الكلامي x 16 dim يكفي لرؤية الهندسة.

### الخطوة الثالثة: هدف أخذ العينات السلبي

لكل زوج إيجابي`(center, context)`، عينة`k`تعليمي النموذج حتى النقطة المنتج`W[center] · W'[context]`هو مرتفع للإيجابيات و منخفض للإيجابيات

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

الصيغة السحرية: الخسارة اللوجستية على الزوجة الإيجابية (تريد sigmoid بالقرب من 1) بالإضافة إلى الخسارة اللوجستية على الزوجات السلبية (تريد sigmoid بالقرب من 0). تدفق المعدلات إلى الجداولين. المشتق الكامل في الورق الأصلي؛ تمر عبرها مرة واحدة مع قلم ورق إذا كنت تريد أن يلتصق.

### الخطوة الرابعة: تدريب على جسم الألعاب

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

بعد فترات كافية على مجموعة كبيرة، الكلمات التي تشارك السياقات لها تركيبات مركزية مماثلة. على مجموعة لعبة، ترى التأثير ضعيفا. على مليارات الرموز، ترى ذلك بشكل كبير.

### الخطوة 5: خدعة التشابه

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

على المتجهات المهنية المسبقة لـ 300d Google News:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`ليس لأن النموذج يعرف ما هو الملكية لأن المتجه`(king - man)`يحتجز شيئا مثل "ملكي" ، و إضافة ذلك إلى `woman`أراضي بالقرب من منطقة الملكيات

## استخدمها

كتابة Word2Vec من الصفر هو التدريس.`gensim`. . .

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

للعمل الحقيقي، لا تدربين Word2Vec بنفسك تقريباً. تنزيل متجهات متدربة مسبقاً.

- **GloVe** نهج ستانفورد للتعامل مع المصفوفات المشتركة. نقاط التفتيش 50d، 100d، 200d، 300d. تغطية عامة جيدة. الدروس 04 تغطي GloVe بشكل خاص.
- **fastText** Word2Vec التوسيع فيسبوك الذي يدمج حرف n-جرام. يتعامل مع الكلمات خارج المفردات عن طريق إعداد كلمات فرعية. الدروس 04.
- **Pretrained Word2Vec on Google News** 300d، 3M لغة الكلمات، نشرت 2013. لا يزال يتم تنزيلها يوميا.

### عندما Word2Vec لا يزال يفوز في 2026

- التدريب على الموجات الطبية في ساعة على جهاز كمبيوتر محمول، الحصول على المتجهات المتخصصة لا نموذج عام التقاط.
- الهندسة المميزة في نمط التشابه`gender_vector = mean(man - woman pairs)`.استغرقها من كلمات أخرى للحصول على محور محايد بين الجنسين . ما زال يستخدم في أبحاث العدالة
- التفسير. 100d صغير بما فيه الكفاية لتحديد عبر PCA أو t-SNE و في الواقع ترى تشكيل المجموعات.
- أي مكان يجب أن يبدأ الإستنتاج على الجهاز دون GPU.

### عندما يفشل Word2Vec

جدار البوليسميا`bank`لديه متجه واحد`river bank`و`financial bank`شاركها`table`(صفحة بيانات مقابل أثاث) يشاركها. مصنف أسفل نهر لا يمكن التمييز بين الحواس من المتجه.

حلّت التوابع السياقية (ELMo، BERT، كل محول منذ) هذا الأمر عن طريق إنتاج متجه مختلف لكل ظهور لل كلمة بناءً على السياق المحيط. وهذا هو القفز من Word2Vec إلى BERT: من ثابت إلى سياقي. المرحلة 7 تغطي نصف المحول.

مشكلة خارج المفردات هي الفشل الآخر.`Zoomer-approved`إذا لم يكن في بيانات التدريب. لا يوجد عكس. fastText يصلح هذا مع تركيب الكلمات الفرعية (المدرس 04).

## أرسله

إبقوا`outputs/skill-embedding-probe.md`:

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

## التمارين

1. **Easy.**إدارة حلقة التدريب على مجموعة صغيرة (20 جملة عن القطط والكلاب) بعد 200 دورة، التحقق`nearest(vocab, W, W[vocab["cat"]])`العائدات`dog`في أعلى 3، وإلا، قم بتعزيز الفترات أو المفردات.
2. **Medium.**إضافة عينة فرعية من الكلمات المتكررة. الكلمات ذات التردد أعلاه `10^-5`يتم إلقاءها من أزواج التدريب مع احتمال متناسب مع ترددها.
3. **Hard.**قم بتدريب نموذج على مجموعة الأخبار العشرين. احسب محورين للتحيز:`he - she`و`doctor - nurse`. مشروع كلمات المهنة على كلا المحاور. تقرير المهن التي لديها أكبر فجوة التحيز. هذا هو نوع من السؤال العدالة الباحثين يستخدمون.

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Word embedding | Word as a vector | A dense, low-dim (typically 100-300) representation learned from context. |
| Skip-gram | Word2Vec trick | Predict context words from center word. Slower than CBOW, better for rare words. |
| Negative sampling | Training shortcut | Replace softmax over full vocab with binary classification against `k` random words. |
| Static embedding | One vector per word | Same vector regardless of context. Fails on polysemy. |
| Contextual embedding | Context-sensitive vector | Different vector for each occurrence based on surrounding words. What transformers produce. |
| OOV | Out of vocabulary | Word not seen in training. Word2Vec cannot produce a vector for these. |

## المزيد من القراءة

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)ورقة العينة السلبية قصيرة وقراءة
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) أوضح استنتاج من التراجع، إذا كانت الرياضيات في الورقة الأصلية تشعر كثافة.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html)إعدادات تدريب الإنتاج التي تعمل فعلاً
