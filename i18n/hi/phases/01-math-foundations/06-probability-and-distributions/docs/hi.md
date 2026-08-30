# संभावना और वितरण

> संभावना अस्थिरता को व्यक्त करने के लिए एआई का उपयोग करने वाली भाषा है।

**Type:** Learn
**Language:**पायथन
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## सीखने के लक्ष्य

- बर्नौली, श्रेणी, पोइसोन, वर्दी और सामान्य वितरण के लिए पीएमएफ और पीडीएफ को खरोंच से लागू करें
- अपेक्षित मूल्य, भिन्नता की गणना करें और गौसी के प्रभुत्व के कारणों की व्याख्या करने के लिए केंद्रीय सीमा प्रमेय का उपयोग करें
- संख्यात्मक स्थिरता चाल (मैक्स लॉजिट घटाएं) के साथ सॉफ्टमैक्स और लॉग-सॉफ्टमैक्स फ़ंक्शंस बनाएं
- लॉजिट से क्रॉस-एंट्रोपी हानि की गणना करें और इसे नकारात्मक लॉग-संभावना से जोड़ें

## समस्या

एक वर्गीकरण आउटपुट `[0.03, 0.91, 0.06]`एक भाषा मॉडल 50,000 उम्मीदवारों से अगला शब्द चुनता है। एक विसारण मॉडल सीखने वाले वितरणों से नमूना करके छवियां उत्पन्न करता है। ये सभी कार्रवाई में संभावना हैं।

प्रत्येक भविष्यवाणी एक मॉडल बनाता है एक संभावना वितरण है. प्रत्येक हानि समारोह मापता है कि भविष्यवाणी वितरण सही से कितना दूर है. प्रत्येक प्रशिक्षण चरण पैरामीटर समायोजित करता है ताकि एक वितरण एक और जैसा दिखता है। संभावना के बिना, आप एक भी एमएल पेपर नहीं पढ़ सकते हैं, एक भी मॉडल डिबग नहीं कर सकते हैं, या समझ सकते हैं कि आपका प्रशिक्षण हानि एनएएन क्यों है।

## अवधारणा

### घटनाएँ, नमूना स्थान और संभावना

नमूना स्थान S सभी संभव परिणामों का सेट है। एक घटना नमूना स्थान का एक उपसमूह है। संभावना घटनाओं को 0 से 1 के बीच संख्याओं में मानचित्रित करती है।

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

तीन स्वार्थ सभी संभावनाओं को परिभाषित करते हैंः
1. किसी भी घटना के लिए P(A) >= 0
2. P(S) = 1 (कुछ हमेशा होता है)
3. P(A या B) = P(A) + P(B) जब A और B दोनों नहीं हो सकते

बाकी सब कुछ (बेयस का प्रमेय, अपेक्षाएं, वितरण) इन तीन नियमों से निकलता है।

### सशर्त संभावना और स्वतंत्रता

P ((A) B) A की संभावना है कि B हुआ है।

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

दो घटनाएँ स्वतंत्र होती हैं जब एक को जानने से आपको दूसरे के बारे में कुछ नहीं पता होता हैः

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

सिक्के फेंकना स्वतंत्र है, बिना प्रतिस्थापन के कार्ड खींचना नहीं है।

### संभावना द्रव्यमान फ़ंक्शन बनाम संभावना घनत्व फ़ंक्शन

विवश यादृच्छिक चर में एक संभावना द्रव्यमान फ़ंक्शन (PMF) होता है। प्रत्येक परिणाम में एक विशिष्ट संभावना होती है जिसे आप सीधे पढ़ सकते हैं।

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

निरंतर यादृच्छिक चर में एक संभावना घनत्व समारोह (पीडीएफ) होता है। एक एकल बिंदु पर घनत्व एक संभावना नहीं है। संभावना एक अंतराल पर घनत्व को एकीकृत करने से आती है।

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

यह अंतर एमएल में मायने रखता है। वर्गीकरण आउटपुट पीएमएफ (विशिष्ट विकल्प) हैं। वीएई लटेंट स्पेस पीडीएफ (अंतर) का उपयोग करते हैं।

### आम वितरण

**Bernoulli:**एक परीक्षण, दो परिणाम. मॉडल द्विआधारी वर्गीकरण.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical:**एक परीक्षण, k परिणाम। मॉडल बहु-वर्ग वर्गीकरण (सॉफ्टमैक्स आउटपुट) ।

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform:**सभी परिणामों के लिए समान रूप से संभावना.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian):**कर्ण वक्र। औसत (mu) और भिन्नता (sigma^2) द्वारा पैरामेटरीकृत।

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:**एक निश्चित अंतराल में दुर्लभ घटनाओं की गणना।

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### अपेक्षित मूल्य और भिन्नता

अपेक्षित मूल्य वजन औसत परिणाम है।

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

औसत के आसपास फैली भिन्नता उपाय।

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

एमएल में, अपेक्षित मूल्य हानि फ़ंक्शन (डेटा वितरण पर औसत हानि) के रूप में दिखाई देता है। भिन्नता आपको मॉडल स्थिरता के बारे में बताती है। ग्रेडिएंट में उच्च भिन्नता का मतलब शोर प्रशिक्षण है।

### संयुक्त और मार्जिनल वितरण

संयुक्त वितरण P ((X, Y) दो यादृच्छिक चर को एक साथ वर्णित करता है।

संयुक्त PMF उदाहरण (X = मौसम, Y = छाता):

| | Y=0 (no umbrella) | Y=1 (umbrella) | Marginal P(X) |
|---|---|---|---|
| X=0 (sun) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (rain) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Marginal P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

मार्जिनल वितरण दूसरे चर का योग हैः

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

उपरोक्त तालिका में पंक्ति और स्तंभ कुल मार्जिनल हैं।

### क्यों हर जगह सामान्य वितरण दिखाई देता है

केंद्रीय सीमा प्रमेयः कई स्वतंत्र यादृच्छिक चर का योग (या औसत) मूल वितरण के बावजूद एक सामान्य वितरण के लिए अभिसरण करता है।

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

यही कारण है कि:
- माप की त्रुटियां लगभग सामान्य हैं (कई छोटे स्वतंत्र स्रोत)
- न्यूरल नेटवर्क में वजन आरंभिकरण सामान्य वितरण का उपयोग करते हैं
- SGD में ग्रेडिएंट शोर लगभग सामान्य है (कई नमूना ग्रेडिएंट का योग)
- सामान्य वितरण एक दिए गए औसत और भिन्नता के लिए अधिकतम एंट्रॉपी वितरण है

### लॉग संभावनाएं

कच्ची संभावनाएं संख्यात्मक समस्याएं पैदा करती हैं। कई छोटी संभावनाओं को एक साथ गुणा करने से जल्दी से शून्य तक नीचे बहता है।

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

लॉग संभावनाओं इस को ठीक. गुणाओं योग बन जाते हैं.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

नियम:
- log(a * b) = log(a) + log(b)
- लॉग संभावनाएं हमेशा <= 0 (क्योंकि 0 < P <= 1) हैं
- अधिक नकारात्मक = कम संभावना
- क्रॉस-एंट्रोपी हानि सही वर्ग की नकारात्मक लॉग संभावना है

### संभावना वितरण के रूप में सॉफ्टमैक्स

न्यूरल नेटवर्क कच्चे स्कोर (लॉजिट) आउटपुट करते हैं। सॉफ्टमैक्स उन्हें एक वैध संभावना वितरण में परिवर्तित करता है।

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

सॉफ्टमैक्स ट्रिकः अतिप्रवाह को रोकने के लिए एक्सपोनेंशिएटिंग से पहले अधिकतम लॉजिट घटाएं।

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

लॉग-सॉफ्टमैक्स संख्यात्मक स्थिरता के लिए सॉफ्टमैक्स और लॉग को जोड़ता है। पायटॉर्च क्रॉस-एंट्रोपी हानि के लिए इसे आंतरिक रूप से उपयोग करता है।

### नमूनाकरण

नमूनाकरण का अर्थ है किसी वितरण से यादृच्छिक मान खींचना।
- यादृच्छिक नमूने छोड़ दें जो न्यूरॉन्स शून्य बाहर करने के लिए
- डेटा संवर्धन नमूने यादृच्छिक परिवर्तन
- भाषा मॉडल भविष्यवाणी वितरण से अगले टोकन का नमूना
- विसारण मॉडल शोर का नमूना और धीरे-धीरे डीनोइज़ करें

मनमाने वितरण से नमूने निकालने के लिए विपरीत परिवर्तन नमूने निकालने, अस्वीकार नमूने निकालने या पुनः रेमामेटरीकरण चाल (वीएई में उपयोग की जाने वाली) जैसी तकनीकों की आवश्यकता होती है।

```figure
gaussian-pdf
```

## इसे बनाओ

### चरण 1: संभावना मूल बातें

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

### चरण 2: पीएमएफ और पीडीएफ खरोंच से

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

### चरण 3: अपेक्षित मूल्य और भिन्नता

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

### चरण 4: वितरण से नमूना

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

### चरण 5: सॉफ्टमैक्स और लॉग संभावनाएं

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

### चरण 6: केंद्रीय सीमा प्रमेय प्रदर्शन

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### चरण 7: दृश्य

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

सभी दृश्यों के साथ पूर्ण कार्यान्वयन में हैं `code/probability.py`. .

## इसका प्रयोग करें

NumPy और SciPy के साथ, ऊपर सब कुछ एक पंक्ति हैः

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

आप इन को खरोंच से बनाया है. अब आप जानते हैं कि पुस्तकालय कॉल क्या कर रहे हैं.

## व्यायाम

1. एक्सपोनेंशियल वितरण के लिए उल्टा परिवर्तन नमूनाकरण लागू करें। 10,000 मानों का नमूना करके और हिस्टोग्राम की तुलना वास्तविक पीडीएफ से करके सत्यापित करें।

2. दो लोड किए गए डैस के लिए एक संयुक्त वितरण तालिका बनाएं। मार्जिनल वितरण की गणना करें और जांचें कि क्या डैस स्वतंत्र हैं।

3. 5 वर्ग वर्गीकरण के लिए क्रॉस-एंट्रोपी हानि की गणना करें जो लॉजिट आउटपुट करता है `[2.0, 0.5, -1.0, 3.0, 0.1]`जब सही वर्ग सूचकांक 3 है तो PyTorch के साथ अपनी प्रतिक्रिया की पुष्टि करें `nn.CrossEntropyLoss`. .

4. एक फ़ंक्शन लिखें जो लॉग संभावनाओं की सूची लेता है और सबसे अधिक संभावना वाले अनुक्रम, कुल लॉग संभावना और समकक्ष कच्चे संभावना को लौटाता है। इसे 50 शब्दों के वाक्य के साथ परीक्षण करें जहां प्रत्येक शब्द की संभावना 0.01 है।

## प्रमुख शर्तें

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

## आगे पढ़ना

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo)- औसत सामान्य क्यों हो जाते हैं, इसका दृश्य प्रमाण
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf)- यहां और अधिक को कवर करने वाला संक्षिप्त संदर्भ
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)- संख्यात्मक स्थिरता क्यों महत्वपूर्ण है और इसे कैसे प्राप्त किया जाए
