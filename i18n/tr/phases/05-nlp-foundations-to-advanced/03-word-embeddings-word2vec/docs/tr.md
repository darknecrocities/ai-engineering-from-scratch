# Word Embeddings  Word2Vec sıfırdan

> Bir kelime, bir şirket olarak kalır ve bu fikir üzerinde derin bir ağ oluşturur ve geometri düşer.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## Sorun

TF-IDF biliyor .`dog`ve `puppy`Bu, farklı kelimelerdir.`dog`Bu konuda genel bir inceleme yapamazsınız.`puppy`Bu konuyu eşya ifadelerle yazabilirsiniz, ama bu nadir terimlerde, domain jargonunda ve tahmin edemediğiniz her dilde başarısız olur.

Bir temsilcilik istiyorsun .`dog`ve `puppy`Uzayda birbirine yakın bir yere yerleşmek.`king - man + woman`Yakın bir yerde`queen`- Bir modelin eğitim aldığı yer .`dog`Bir sinyal aktarıyor `puppy`- Ücretsiz.

Word2Vec bize bu alanı verdi. İki katlı sinir ağı, trilyonlarca token eğitim çalışması, 2013 yılında yayınlandı. Mimarlık neredeyse utanç verici bir şekilde basit. Sonuçlar bir on yıl boyunca NLP'yi yeniden şekillendirdi.

## Anlaşım

**Distributional hypothesis**Birinci, 1957: "Bir kelimeyi, onun tutduğu arkadaşlıktan anlayacaksın".

Word2Vec iki çeşitlikte gelir. Her ikisi de bu fikri kullanıyor.

- **Skip-gram.**Merkezi kelime verildiğinde, çevresindeki kelimeleri tahmin et.`cat -> (the, sat, on)`Pencere boyutu 2.
- **CBOW (continuous bag of words).**Etrafta sözcükler varken, merkezi tahmin et.`(the, sat, on) -> cat`- Evet .

Skip-gram daha yavaş eğitimlenir ama nadir kelimeleri daha iyi ele alır.

Ağ, doğrusallık olmayan bir gizli katman vardır. Giriş sözlük üzerinde bir sıcak vektördür. Çıktı sözlük üzerinde bir yumuşak maksimum. Eğitimden sonra, çıkış katmanı atarsınız. Gizli katman ağırlıkları yerleşimlerdir.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

Yaptığımız şey: 100 bin kelimeyi aşan softmax çok pahalı.**negative sampling**Bu nedenle, bu konektör kelimeyi bir ikili sınıflandırma görevine dönüştürmek için bir ikili sınıflandırma görevi oluşturmak için kullanılır. "Bu bağlam kelime bu merkez kelime yakınında, evet veya hayır olarak ortaya çıktı mı?" tahmin et.

```figure
word-vector-arithmetic
```

## Yapın

### Adım 1: bir korpustan eğitim çiftleri

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

Bir penceredeki her (merkezi, bağlam) çift olumlu bir eğitim örneğidir.

### Adım 2: Masalar yerleştirme

İki matris.`W`Ortadaki kelimenin yerleştirme masası (sağlam). `W'`Bu, bağlamlı kelime tablosudur (sık sık atılır, bazen ortalama olarak `W`)

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

Küçük rastgele başlangıç. 10k ve dim 100 kelime boyutu gerçekçi; öğretim için, 50 kelime x 16 dim jeometri görmek için yeterlidir.

### Adım 3: negatif örnekleme hedefi

Her olumlu çift için `(center, context)`, örnek`k`Sözcükten geleneksel kelimeler negatif olarak kullanın.`W[center] · W'[context]`Pozitifler için yüksek, negatifler için düşük.

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

Sihirli formül: pozitif çift üzerinde lojistik kaybı (sigmoid yakınında 1) ek olarak negatif çiftlerde lojistik kaybı (sigmoid yakınında) . Gradientler her iki tabloya akıyor. Tam türev orijinal kağıtta; eğer yapışmak istiyorsanız kalem ve kağıt ile bir kez geçin.

### Dördüncü adım: Oyuncak korpusunda eğitim

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

Büyük bir korpus üzerinde yeterince zaman geçtikten sonra, bağlamları paylaşan kelimelerin merkezi benzerliklere sahiptir. Oyuncak korpusunda, etkisini hafifçe görürsünüz.

### 5. Adım: Analogya hilesi

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

Önceden eğitilmiş 300d Google Haber vektörlerinde:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`- Çünkü model, kraliyetten bahsettiğini biliyor.`(king - man)`"krallık" gibi bir şey yakalar ve ekler.`woman`Kraliyet kadınları bölgesinin yakınlarında yer alan topraklar.

## Kullan

Word2Vec'i sıfırdan yazmak öğretimdir.`gensim`- Evet .

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

Gerçek iş için neredeyse hiç Word2Vec'i kendin eğitmiyorsun.

- **GloVe**Stanford'un ortak oluşu matrisi faktörleşme yaklaşımı. 50d, 100d, 200d, 300d kontrol noktaları. İyi genel kapsam. Ders 04 özellikle GloVe'yi kapsar.
- **fastText** Facebook'un n-gram karakterleri içeren Word2Vec uzantısı. Sözcüklük dışındaki kelimeleri alt kelimeleri oluşturarak ele alır. Ders 04.
- **Pretrained Word2Vec on Google News** 300d, 3M kelime sözlüğü, 2013 yılında yayınlandı.

### Word2Vec 2026'da hala kazanırken

- Uzaylı bir alan özel çekim, bir saatte bir dizüstü bilgisayarla tıbbi çekimler üzerinde çalışmak, özel vektörler almak, genel modeller çekimleri olmamak.
- Analogik tarzlı özellik mühendisliği.`gender_vector = mean(man - woman pairs)`- Başka kelimelerden çıkarıp cinsiyet tarafsızlığı elde edelim.
- 100d, PCA veya t-SNE üzerinden çizim yapıp aslında kümeler oluştuğunu görmek için yeterince küçüktür.
- Herhangi bir yerde sonuçlar GPU olmadan cihazda çalıştırılmalıdır. Word2Vec arama tek satırlı bir çekimdir.

### Word2Vec'in başarısız olduğu yerler

- Polysemi duvarı.`bank`Bir vektörü var.`river bank`ve `financial bank`Paylaşın.`table`Bir sınıflandırıcı aşağı akıntıda duyuları vektörden ayırt edemez.

Konekstel yerleşimler (ELMo, BERT, her transformatör o zamandan beri) çevredeki bağlamlara göre kelimenin her bir oluşumu için farklı bir vektör üreterek bunu çözdü. Bu, Word2Vec'ten BERT'e atlama: statikten bağlamlıya.

Sözcük kaynağı dışında olan sorun diğer başarısızlık.`Zoomer-approved`Eğer bu bilgi eğitim verilerinde bulunmamışsa.

## Gönder

- Kaydet .`outputs/skill-embedding-probe.md`- ...

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

## Egzersizler

1. **Easy.**Küçük bir kurpus üzerinde eğitim döngüsünü yürütün (20 kediler ve köpekler hakkında cümle).`nearest(vocab, W, W[vocab["cat"]])`Devamı`dog`Eğer yapmazsanız, dönemleri veya kelime birikimini artırın.
2. **Medium.**Sık sözcüklerin alt örneklerini ekleyin.`10^-5`Bu nedenle, bu değerlerin, sıklıkla orantılı olan olasılıklara göre, eğitim çiftlerinden düşürülmesi gerekir.
3. **Hard.**20 Haber Grupları korpusunda bir model çalıştırın.`he - she`ve `doctor - nurse`Bu, araştırmacıların kullandığı bir araştırma yöntemidir.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Word embedding | Word as a vector | A dense, low-dim (typically 100-300) representation learned from context. |
| Skip-gram | Word2Vec trick | Predict context words from center word. Slower than CBOW, better for rare words. |
| Negative sampling | Training shortcut | Replace softmax over full vocab with binary classification against `k` random words. |
| Static embedding | One vector per word | Same vector regardless of context. Fails on polysemy. |
| Contextual embedding | Context-sensitive vector | Different vector for each occurrence based on surrounding words. What transformers produce. |
| OOV | Out of vocabulary | Word not seen in training. Word2Vec cannot produce a vector for these. |

## Daha Fazla Okumak

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) negatif örnekleme kağıdı. Kısa ve okunur.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) orijinal kağıtın matematiği yoğun hissettiğinde, gradientlerin en net türü.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) Aslında işe yarayan üretim eğitim ayarları.
