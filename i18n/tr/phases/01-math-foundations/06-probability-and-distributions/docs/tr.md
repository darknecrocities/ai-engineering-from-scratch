# Muhtemelenlik ve dağıtımlar

> Muhtemelenlik, AI'nin belirsizlikleri ifade etmek için kullandığı dildir.

**Type:** Learn
**Language:**Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## Öğrenme Hedefleri

- Bernoulli, kategorik, Poisson, üniform ve normal dağılımlar için PMF ve PDF'leri sıfırdan uygulayın
- Beklenen değeri, varyansi hesaplayın ve Gaussians'ın neden egemen olduğunu açıklamak için Merkez Sınır Teoremi kullanın
- Sayısal istikrar hilesini kullanarak softmax ve log-softmax fonksiyonlarını oluştur (maksimum logit çıkar)
- Logitlerden çapraz entropik kaybı hesaplayın ve onu negatif log olasılığı ile bağlayın

## Sorun

Bir sınıflandırıcı çıkışları `[0.03, 0.91, 0.06]`Dil modeli 50.000 adaydan bir sonraki kelimeyi seçer. Bir difüzyon modeli öğrenilen dağılımlardan örnek alarak görüntüler üretir. Bunların hepsi etkin olasılıklardır.

Bir modelin yaptığı her tahmin bir olasılık dağılımıdır. Her kayıp işlevi tahmin edilen dağılımın gerçek olanlardan ne kadar uzak olduğunu ölçer. Her eğitim adımında bir dağılımın diğerine daha çok benzeyebilmesi için parametreler ayarlanır.

## Anlaşım

### Olaylar, Örnek Alanlar ve Muhtemelenlik

Örnek alanı S, tüm olası sonuçların toplamıdır. Bir olay örnek alanının bir alt kümesidir. Muhtemelenlik olayları 0 ile 1 arasındaki rakamlara haritasıyor.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

Üç aksiom tüm olasılıkları tanımlar:
1. P(A) >= 0 herhangi bir olay için A
2. P(S) = 1 (her zaman bir şeyler olur)
3. P(A veya B) = P(A) + P(B) A ve B'nin ikisi de gerçekleşemediği zaman

Diğer her şey (Bayes teoremi, beklentiler, dağılımlar) bu üç kuralı takip eder.

### Şartlı Muhtemelenlik ve Bağımsızlık

P ((A) B) A'nın B'nin gerçekleşmesi olasılığıdır.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

İki olay bağımsızdır. Birini bilmek diğerini anlatmaz.

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

Para atmak bağımsızdır, değiştirilmeden çizmek de değil.

### Muhtemelen Masa Fonksiyonları vs. Muhtemelen Sıklık Fonksiyonları

Diskret rastgele değişkenlerin olasılık kütle fonksiyonu (PMF) vardır. Her sonuç doğrudan okuyabileceğiniz belirli bir olasılığa sahiptir.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

Sürekli rastgele değişkenlerin olasılık yoğunluğu işlevi vardır. Tek bir noktada yoğunluk olasılık değildir.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

Bu ayrım ML'de önemlidir. Sınıflama çıkışları PMF'lerdir (diskret seçimler). VAE gizli alanları PDF'leri (daima) kullanır.

### Genel dağıtımlar

**Bernoulli:**Bir deney, iki sonuç.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical:**Modeller çok sınıflı sınıflandırma (softmax çıkışı).

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform:**- Tüm sonuçlar eşit olasılıkla.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian):**Çan eğri. ortalama (mu) ve varyansa (sigma^2) ile parametrelidir.

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:**Sık rastlanan olayların belirli bir aralıkta sayılması.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### Beklenen Değer ve Çeşitlilik

Beklenen değer, ağırlıklı ortalama sonuçtır.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

Değişiklik ölçümleri ortalama etrafında yayılmış.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

ML'de, beklenen değer kayıp işlevi (verilerin dağılımında ortalama kayıp) olarak görünür.

### Ortak ve Marjinal dağıtımlar

Bir ortak dağılım P ((X, Y) iki rastgele değişkenyi birlikte tanımlar.

Ortak PMF örneği (X = hava durumu, Y = şemsiye):

| | Y=0 (no umbrella) | Y=1 (umbrella) | Marginal P(X) |
|---|---|---|---|
| X=0 (sun) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (rain) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Marginal P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

Sınırsal dağılım diğer değişkenin toplamını içerir:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

Yukarıdaki tabloda sırada ve sütunda toplamlar, kenarlıklardır.

### Normal Değişiklik Neden Her Yerde Görülüyor?

Merkez Sınır Teoremi: birçok bağımsız rastgele değişkenin toplamı (veya ortalaması) orijinal dağılımdan bağımsız olarak normal bir dağılım için birleşti.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

İşte bu yüzden:
- Ölçüm hataları yaklaşık olarak normaldir (çok küçük bağımsız kaynaklar)
- Nöral ağlarda ağırlık başlangıçları normal dağılımları kullanır
- SGD'deki gradient gürültüsü yaklaşık olarak normaldir (çok sayıda örnek gradientinin toplamı)
- Normal dağılım, verilen bir ortalama ve varyansa için en fazla entropi dağılımıdır.

### Kayıt olasılıkları

Çiğ olasılıklar sayısal sorunlara neden olur. Çok küçük olasılıkları bir araya getirmek hızla sıfıra düşer.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

Log olasılıkları bunu düzeltir.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

Kurallar:
- log(a * b) = log(a) + log(b)
- log olasılığı her zaman <= 0'dur (Çünkü 0 < P <= 1)
- Daha negatif = daha az olası
- Çarpıcı entropik kaybı doğru sınıfın negatif log olasılığıdır

### Softmax, olasılık dağılımı olarak

Sinir ağları çiğ puanlar (logits) çıkarır. Softmax onları geçerli bir olasılık dağılımına dönüştürür.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Softmax numarası: Aşırı akışın önlenmesi için, eksponansiyalandırmadan önce maksimum logit'i çıkarın.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax, sayısal istikrar için softmax ve log'u birleştirir. PyTorch bunu içsel olarak çapraz entropi kaybı için kullanır.

### Örnekleme

Örnekleme, bir dağılımdan rastgele değerlerin çekilmesi anlamına gelir.
- Neuronu kimler sıfırlamak için rastgele örnekler bırakın
- Veri artırma örnekleri rastgele dönüşümler
- Dil modelleri tahmin edilen dağılımdan bir sonraki simgeyi örnekler
- Diffüzyon modelleri gürültü örneğini ve ilerleyerek denosiyonunu gösterir

Kezleyici dağılımlardan örnek almak, ters dönüşüm örneği, reddetme örneği veya reparametreleme hilesi (VAE'lerde kullanılır) gibi teknikleri gerektirir.

```figure
gaussian-pdf
```

## Yapın

### Adım 1: Muhtemelenlik Temellikleri

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Adım 2: PMF ve PDF sıfırdan

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Adım 3: Beklenen değer ve değişim

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 4. Adım: Distribüsiyonlardan örnek alınması

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Adım 5: Softmax ve log olasılığı

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Adım 6: Merkez Sınır Teoremi gösterimi

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 7. Adım: Görüntüleme

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Tüm görsellemeler ile birlikte tam uygulamalar `code/probability.py`- Evet .

## Kullan

NumPy ve SciPy ile yukarıda her şey tek satırlı:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

Şimdi kütüphane aramalarının ne yaptığını biliyorsun.

## Egzersizler

1. Eksponansiyel dağılım için ters dönüşüm örneğini uygulayın. 10.000 değer örneği alıp histogramı gerçek PDF ile karşılaştırarak doğrulayın.

2. İki yüklü zar için ortak bir dağıtım masası oluşturun.

3. Logit çıkaran 5 sınıf sınıflandırıcı için çapraz entropik kaybı hesaplayın `[2.0, 0.5, -1.0, 3.0, 0.1]`Doğru sınıf indeks 3 olduğunda, sonra cevapınızı PyTorch'ın `nn.CrossEntropyLoss`- Evet .

4. Log olasılıklarının bir listesini alıp en olası sırayı, toplam log olasılığını ve eşdeğer çiğ olasılığı geri veren bir işlev yazın.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Sample space | "All the possibilities" | The set S of every possible outcome of an experiment |
| PMF | "The probability function" | A function that gives the exact probability of each discrete outcome, summing to 1 |
| PDF | "The probability curve" | A density function for continuous variables. Integrate it over an interval to get probability |
| Conditional probability | "Probability given something" | P(A\|B) = P(A and B) / P(B). The foundation of Bayesian thinking and Bayes' theorem |
| Independence | "They don't affect each other" | P(A and B) = P(A) * P(B). Knowing one event tells you nothing about the other |
| Expected value | "The average" | The probability-weighted sum of all outcomes. The loss function is an expected value |
| Variance | "How spread out" | The expected squared deviation from the mean. High variance = noisy, unstable estimates |
| Normal distribution | "The bell curve" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). Appears everywhere due to the CLT |
| Central Limit Theorem | "Averages become normal" | The mean of many independent samples converges to a normal distribution regardless of the source |
| Joint distribution | "Two variables together" | P(X, Y) describes the probability of every combination of X and Y outcomes |
| Marginal distribution | "Sum out the other variable" | P(X) = sum_y P(X, Y). Recovers one variable's distribution from the joint |
| Log probability | "Log of the probability" | log P(x). Turns products into sums, preventing numerical underflow in long sequences |
| Softmax | "Turn scores into probabilities" | softmax(z_i) = exp(z_i) / sum(exp(z_j)). Maps real-valued logits to a valid probability distribution |
| Cross-entropy | "The loss function" | -sum(p_true * log(p_predicted)). Measures how different two distributions are. Lower is better |
| Logits | "Raw model outputs" | Unnormalized scores before softmax. Named after the logistic function |
| Sampling | "Drawing random values" | Generating values according to a probability distribution. How models generate output |

## Daha Fazla Okumak

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo)- ortalamaların neden normal hale geldiğinin görsel bir kanıtı
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf)- burada ve daha fazlasını kapsayan kısa bir referans
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)- Sayısal istikrar neden önemlidir ve nasıl elde edilebilir
