# 概率和分布

> 概率是人工智能用来表达不确定性的语言.

**Type:** Learn
**Language:**字符串
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## 学习目标

- 实现Bernoulli,类型,Poisson,统一和正常分布的PMF和PDF从零开始
- 计算预期值,变量,并使用中央限量定理来解释为什么高西人占主导地位
- 通过数值稳定技巧构建软max和 log-softmax函数 (减去最大logit)
- 计算从logits中交叉缩损失,并将其连接到负 log-概率

## 问题

一个分类器输出`[0.03, 0.91, 0.06]`语言模型从5万个候选人中选出下一个词. 扩散模型通过从学习分布中抽取样本来生成图像. 所有这些都是行动中的概率.

每个模型的预测都是概率分布. 每个损失函数测量预测的分布与真实的分布有多远. 每个训练步骤都调整参数,使一个分布更像另一个.没有概率,你不能读一篇ML论文,调试一个模型,或者理解为什么你的训练损失是NaN.

## 概念

### 事件,样本空间和可能性

样本空间 S 是所有可能的结果的集合.事件是样本空间的子集.概率将事件映射到0到1之间的数字.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

概率的定义有三个定理:
1. 对于任何事件 A, P(A) >= 0
2. 总是发生了一些事情.
3. P(A或B) = P(A) + P(B) A和B既不能发生

其他一切 (贝斯定理,期望,分布) 都来自于这些三个规则.

### 条件可能和独立性

根据B的概率,B的概率是 A 的概率.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

两个事件是独立的,当知道一个对另一个什么都不告诉你:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

钱是独立的,没有替换的卡是没有的.

### 概率质量函数与概率密度函数

微小随机变量具有概率质量函数 (PMF).每个结果都有特定的概率,你可以直接读取.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

连续随机变量具有概率密度函数 (PDF).一个点的密度不是概率.概率来自于在一个间隔中集成密度.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

在ML中,这种区别很重要.分类输出是PMF (微妙选择).VAE隐藏空间使用PDF (连续).

### 常见的分布

**Bernoulli:**一个试验,两个结果.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical:**模型多类分类 (软最大输出).

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform:**随机初始化.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian):**通过平均 (mu) 和变量 (sigma^2) 参数化.

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:**模型事件率.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### 预期价值和变化

预期值是中值结果.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

变量量分布在平均值周围.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

在ML中,预期值显示为损失函数 (数据分布平均损失).变化告诉你模型稳定性.梯度的高变化意味着噪音训练.

### 联合和边际分销

联合分布 P ((X,Y) 描述两个随机变量.

联合PMF的例子 (X = 天气,Y =雨):

| | Y=0 (no umbrella) | Y=1 (umbrella) | Marginal P(X) |
|---|---|---|---|
| X=0 (sun) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (rain) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Marginal P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边际分布总结了其他变量:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

上面表中的行列和列总数是边缘值.

### 为什么常规分布在各处出现

中央限量定理:许多独立随机变量的总和 (或平均值) 融合到正常分布,不管原始分布如何.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

这就是为什么:
- 测量错误大约是正常的 (许多小的独立来源)
- 在神经网络中重量初始化使用正常分布
- 度噪音在SGD中大约正常 (许多样本度的总和)
- 正常分布是给定的平均和变异的最大缩分布

### 记录概率

几率是数值问题,多次乘以几率的数值,

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

乘法变成了加算.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

规则:
- 标签: 标签: 标签: 标签:
- 记录概率总是 <= 0 (因为 0 < P <= 1)
- 更多负面 = 可能性较小
- 交叉缩损失是正确类的负记录概率

### 软max作为概率分布

神经网络输出原始分数 (logits).软max将它们转化为有效的概率分布.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

软max技巧:在指数化之前减去最大的逻辑值,以防止过度流动.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

鱼使用这个内部用于交叉缩损失.

### 采样

采样是从分布中抽取随机值.
- 随机放弃哪些神经元的样本
- 数据增强样本随机转换
- 语言模型从预测分布中采样下一个代币
- 扩散模型采样噪音和逐步毁

从任意分布中采样需要像反转型采样,拒绝采样或重设方法 (用于VAE) 等技术.

```figure
gaussian-pdf
```

## 建立它

### 步骤1:概率基础

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

### 步骤2:从零开始 PMF和 PDF

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

### 步骤3:预期值和差异

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

### 步骤4:从分布中采样

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

### 步骤5:软max和记录概率

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

### 步骤 6:中央限量定理示范

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 步骤7: 视觉化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整的实现,所有可视化都在`code/probability.py`现在,我们要去.

## 用它

对于NumPy和SciPy,上面的一切都是单行:

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

你从头开始了,现在你知道图书馆的电话是什么.

## 运动

1. 执行反转型样本测试,以对指数分布进行测试.通过采样10,000个值并将历史图与真实的PDF进行比较,

2. 构建一个两个负载的子的联合分布表,计算边缘分布,检查子是否独立.

3. 计算出输出 logits的5类分类器的跨缩损失`[2.0, 0.5, -1.0, 3.0, 0.1]`当正确的类是指数3时,然后用 PyTorch 的答案验证`nn.CrossEntropyLoss`现在,我们要去.

4. 写一个函数,以记录概率列表,返回最有可能的序列,总记录概率和相当的原始概率. 用50字的句子测试,每个字的概率为0.01.

## 关键词

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

## 进一步阅读

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo)- 视觉证明平均水平为什么变得正常
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf)- 简单的参考,涵盖了这里的一切,
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)- 数字稳定性为什么重要,如何实现
