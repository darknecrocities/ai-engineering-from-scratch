# वर्ड एम्बेडिंग  स्क्रैच से Word2Vec

> एक शब्द एक कंपनी है जो यह रखता है उस विचार पर एक पतला जाल को प्रशिक्षित करें और ज्यामिति गिर जाती है।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## समस्या

टीएफ-आईडीएफ जानता है `dog`और `puppy`यह अलग-अलग शब्द हैं. यह नहीं जानता कि वे लगभग एक ही बात का मतलब है. एक वर्गीकरण पर प्रशिक्षित`dog`के बारे में एक समीक्षा में सामान्य नहीं कर सकते `puppy`आप इसके बारे में समानार्थी शब्द सूचीबद्ध करके कागज बना सकते हैं, लेकिन यह दुर्लभ शब्दों, डोमेन जार्गोन और हर भाषा पर विफल रहता है जिसे आप उम्मीद नहीं करते थे।

आप एक प्रतिनिधित्व चाहते हैं जहां `dog`और `puppy`अंतरिक्ष में एक साथ जमीन।`king - man + woman`निकटवर्ती भूमि`queen`. जहां एक मॉडल पर प्रशिक्षित किया गया था`dog`कुछ संकेत भेजता है `puppy`मुफ्त में।

Word2Vec ने हमें यह स्थान दिया। दो परत न्यूरल नेटवर्क, ट्रिलियन टोकन प्रशिक्षण रन, 2013 में प्रकाशित। वास्तुकला लगभग शर्मनाक रूप से सरल है। परिणामों ने एक दशक के लिए एनएलपी को फिर से आकार दिया।

## अवधारणा

**Distributional hypothesis**(पहला, 1957): "आपको एक शब्द को उस कंपनी से पता चलेगा जो उसे रखता है।" यदि दो शब्द समान संदर्भों में दिखाई देते हैं, तो वे शायद समान बातें समझते हैं।

Word2Vec दो स्वादों में आता है, दोनों उस विचार का शोषण करते हैं।

- **Skip-gram.**एक मध्य शब्द दिया गया है, आसपास के शब्दों की भविष्यवाणी करें।`cat -> (the, sat, on)`खिड़की का आकार 2 के साथ।
- **CBOW (continuous bag of words).**आसपास के शब्दों को देखते हुए, केंद्र की भविष्यवाणी करें।`(the, sat, on) -> cat`. .

स्किप-ग्राम को प्रशिक्षित करने में धीमा है लेकिन दुर्लभ शब्दों को बेहतर ढंग से संभालता है। यह डिफ़ॉल्ट बन गया।

नेटवर्क में एक छिपी हुई परत है जिसमें कोई गैर-रेखीयता नहीं है। इनपुट शब्दावली पर एक गर्म वेक्टर है। आउटपुट शब्दावली पर एक सॉफ्टमैक्स है। प्रशिक्षण के बाद, आप आउटपुट परत को फेंक देते हैं। छिपी हुई परत वजन एम्बेडमेंट हैं।

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

ट्रिकः 100 हजार शब्दों से अधिक सॉफ्टमैक्स बहुत महंगा है। Word2Vec का उपयोग करता है**negative sampling**यह एक द्विआधारी वर्गीकरण कार्य में बदलना है। भविष्यवाणी करें "क्या यह संदर्भ शब्द इस मध्य शब्द के पास दिखाई देता है, हाँ या नहीं। " प्रशिक्षण जोड़े के लिए कुछ नकारात्मक (गैर-सह-प्रभावित) शब्दों का नमूना लें, पूरे शब्दावली पर सॉफ्टमैक्स की गणना करने के बजाय।

```figure
word-vector-arithmetic
```

## इसे बनाओ

### चरण 1: एक कॉर्पस से प्रशिक्षण जोड़े

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

खिड़की में प्रत्येक (केंद्र, संदर्भ) जोड़ी एक सकारात्मक प्रशिक्षण उदाहरण है।

### चरण 2: तालिकाओं को एम्बेड करना

दो मैट्रिक्स।`W`है केंद्र शब्द एम्बेडिंग तालिका (आप रखती है जो एक) । `W'`संदर्भ-शब्द तालिका (अक्सर त्याग दिया, कभी-कभी औसत के साथ `W`) ।

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

छोटी यादृच्छिक शुरुआत। 10k और dim 100 के आकार का शब्द यथार्थवादी है; शिक्षण के लिए, 50 शब्द x 16 dim ज्यामिति देखने के लिए पर्याप्त है।

### चरण 3: नकारात्मक नमूनाकरण लक्ष्य

प्रत्येक सकारात्मक जोड़ी के लिए `(center, context)`, नमूना `k`शब्द संग्रह से यादृच्छिक शब्दों को नकारात्मक के रूप में। मॉडल को प्रशिक्षित करें ताकि अंक उत्पाद`W[center] · W'[context]`सकारात्मक और नकारात्मक के लिए उच्च है।

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

जादुई सूत्रः सकारात्मक जोड़ी पर लॉजिस्टिक हानि (सिग्मोइड के पास 1 चाहते हैं) और नकारात्मक जोड़े पर लॉजिस्टिक हानि (सिग्मोइड के पास 0 चाहते हैं) । ग्रेडिएंट दोनों तालिकाओं में बहते हैं। पूर्ण व्युत्पन्न मूल कागज में है; यदि आप इसे चिपके रहना चाहते हैं तो पेंसिल और कागज के साथ एक बार इसके माध्यम से जाएं।

### चरण 4: खिलौना कॉर्पस पर प्रशिक्षण

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

एक बड़े कॉर्पस पर पर्याप्त समय के बाद, संदर्भ साझा करने वाले शब्दों में समान केंद्र एम्बेडेड होते हैं। एक खिलौना कॉर्पस पर, आप प्रभाव को हल्के ढंग से देखते हैं। अरबों टोकन पर, आप इसे नाटकीय रूप से देखते हैं।

### चरण 5: एनालॉग ट्रिक

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

पूर्व प्रशिक्षित 300d गूगल समाचार वेक्टरों परः

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .`(king - man)`कुछ "शाही" जैसे कैप्चर करता है, और इसे जोड़ने के लिए `woman`शाही-महिला क्षेत्र के पास भूमि।

## इसका प्रयोग करें

Word2Vec को स्क्रैच से लिखना शिक्षण है। उत्पादन एनएलपी का उपयोग करता है `gensim`. .

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

असली काम के लिए, आप शायद ही कभी Word2Vec को स्वयं प्रशिक्षित करते हैं. आप पूर्व-प्रशिक्षित वेक्टर डाउनलोड करते हैं।

- **GloVe**स्टैनफोर्ड के सह-घटना-मैट्रिक्स फैक्टरिज़ेशन दृष्टिकोण. 50d, 100d, 200d, 300d चेकपोइंट. अच्छा सामान्य कवरेज. पाठ 04 ग्लोव को विशेष रूप से कवर करता है।
- **fastText** फेसबुक का Word2Vec एक्सटेंशन जो वर्ण n-ग्राम को एम्बेड करता है। उपशब्दों को रचना करके शब्दावली से बाहर के शब्दों को संभालता है। पाठ 04.
- **Pretrained Word2Vec on Google News** 300d, 3M शब्द शब्दावली, प्रकाशित 2013। अभी भी दैनिक डाउनलोड किया जाता है।

### जब 2026 में Word2Vec अभी भी जीतता है

- एक घंटे में लैपटॉप पर चिकित्सा सार पर प्रशिक्षण, कोई सामान्य मॉडल कैप्चर नहीं विशेष वेक्टर प्राप्त करें।
- एनालॉग शैली की सुविधा इंजीनियरिंग। `gender_vector = mean(man - woman pairs)`. इसे दूसरे शब्दों से घटाकर लिंग-तटस्थ अक्ष प्राप्त करें. अभी भी न्याय अनुसंधान में उपयोग किया जाता है.
- व्याख्यात्मकता. 100d काफी छोटे है पीसीए या टी-एसएनई के माध्यम से ग्राफिंग करने के लिए और वास्तव में क्लस्टर के रूप में देखने के लिए है.
- कहीं भी अनुमान बिना GPU के डिवाइस पर चलाना है. Word2Vec खोज एक पंक्ति लाता है.

### Word2Vec में विफलता

बहुल दीवार।`bank`एक वेक्टर है।`river bank`और `financial bank`इसे साझा करें।`table`एक वर्गीकरणकर्ता नीचे धारा से इंद्रियों को वेक्टर से अलग नहीं कर सकता है।

संदर्भ एम्बेडेड (ELMo, BERT, तब से हर ट्रांसफार्मर) ने आसपास के संदर्भ के आधार पर शब्द के प्रत्येक घटना के लिए एक अलग वेक्टर का उत्पादन करके इसे हल किया। यही है वर्ड2वीसी से BERT की छलांगः स्थैतिक से संदर्भ में। चरण 7 ट्रांसफार्मर आधा को कवर करता है।

शब्द संग्रह से बाहर की समस्या दूसरी विफलता है। Word2Vec ने कभी नहीं देखा है`Zoomer-approved`यदि यह प्रशिक्षण डेटा में नहीं था। कोई पतन नहीं। fastText उपशब्द संरचना (पाठ 04) के साथ इसे ठीक करता है।

## इसे भेजें

`outputs/skill-embedding-probe.md`:

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

## व्यायाम

1. **Easy.**एक छोटे से कॉर्पस पर प्रशिक्षण लूप चलाएं (20 वाक्य बिल्लियों और कुत्तों के बारे में) 200 युगों के बाद, सत्यापित करें `nearest(vocab, W, W[vocab["cat"]])`रिटर्न `dog`यदि नहीं, तो युगों या शब्दावली को बढ़ाएं।
2. **Medium.**आवृत्ति से ऊपर वाले शब्दों का उप-सैंपल जोड़ें `10^-5`दुर्लभ शब्दों की समानता पर प्रभाव मापें।
3. **Hard.**20 न्यूजग्रुप कॉर्पस पर एक मॉडल को प्रशिक्षित करें। दो पूर्वाग्रह अक्षों की गणना करेंः `he - she`और `doctor - nurse`. दोनों अक्षों पर प्रोजेक्ट व्यवसाय शब्द. रिपोर्ट करें कि किन व्यवसायों में सबसे अधिक पूर्वाग्रह अंतर है. यह जांच निष्पक्षता शोधकर्ताओं का उपयोग करने का प्रकार है.

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Word embedding | Word as a vector | A dense, low-dim (typically 100-300) representation learned from context. |
| Skip-gram | Word2Vec trick | Predict context words from center word. Slower than CBOW, better for rare words. |
| Negative sampling | Training shortcut | Replace softmax over full vocab with binary classification against `k` random words. |
| Static embedding | One vector per word | Same vector regardless of context. Fails on polysemy. |
| Contextual embedding | Context-sensitive vector | Different vector for each occurrence based on surrounding words. What transformers produce. |
| OOV | Out of vocabulary | Word not seen in training. Word2Vec cannot produce a vector for these. |

## आगे पढ़ना

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) नकारात्मक नमूना कागज। संक्षिप्त और पठनीय।
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) ग्रेडिएंट का सबसे स्पष्ट व्युत्पन्न, यदि मूल पेपर की गणित घनी लगती है।
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) उत्पादन प्रशिक्षण सेटिंग्स जो वास्तव में काम करती हैं।
