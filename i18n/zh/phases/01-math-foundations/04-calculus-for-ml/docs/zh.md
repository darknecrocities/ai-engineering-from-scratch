# 机器学习的计算

> 导体告诉你哪个方向下坡. 这就是神经网络需要学习的.

**Type:** Learn
**Language:**字符串
**Prerequisites:** Phase 1, Lessons 01-03
**Time:** ~60 minutes

## 学习目标

- 计算常见ML函数的数值和分析衍生物 (x^2,sigmoid,跨)
- 实现从零开始降梯度,以减少1D和2D中的损失函数
- 导出线性回归模型的梯度,并通过手动重量更新训练它
- 解释赫西矩阵,泰勒系列近似和它们与优化方法的联系

## 问题

你有一个数百万个重量的神经网络. 每个重量都是一个. 你需要弄清楚哪个方向转移每个,使模型变得略有不错. 计算给你这个方向.

没有计算,训练神经网络意味着尝试随机变化,希望得到最好的. 借助衍生品,你知道每个权重对错误的影响. 你每次都会把每个扣子转向正确的方向.

## 概念

### 导数是什么?

衍生值测量变化速度.对于函数 y = f(x,衍生值 f'(x) 告诉你:如果你推出 x 微小的数量, y 变化多少?

几何学上,衍生品是线在某个点上的斜率.

**f(x) = x^2:**

| x | f(x) | f'(x) (slope) |
|---|------|---------------|
| 0 | 0    | 0 (flat, at the bottom) |
| 1 | 1    | 2 |
| 2 | 4    | 4 (tangent line slope at this point) |
| 3 | 9    | 6 |

在 x=2 时,斜率是 4. 如果把 x 移动到右边, y 增加了4倍左右.在 x=0,斜率是 0.

官方定义:

```
f'(x) = lim   f(x + h) - f(x)
        h->0  -----------------
                     h
```

在代码中,你跳过了极限,只使用一个非常小的h.

### 部分衍生品:一次性变量

实际函数有很多输入.神经网络损失取决于数千个权重. 一个部分衍生物保持除一个变量以外的所有变量是恒定的,然后取出与那个变量相对于的衍生物.

```
f(x, y) = x^2 + 3xy + y^2

df/dx = 2x + 3y     (treat y as a constant)
df/dy = 3x + 2y     (treat x as a constant)
```

如果我推出这只重量,损失会如何改变?

### 梯度:所有部分衍生物的向量

梯度将每个部分衍生物集成成一个向量.对于函数 f ((x, y, z),梯度是:

```
grad f = [ df/dx, df/dy, df/dz ]
```

梯指向最的升方向.

**Contour plot of f(x,y) = x^2 + y^2:**

函数形成一个碗形状,以圆为轮线.最小值为 (0, 0).

| Point | grad f | -grad f (descent direction) |
|-------|--------|----------------------------|
| (1, 1) | [2, 2] (points uphill, away from minimum) | [-2, -2] (points downhill, toward minimum) |
| (0, 0) | [0, 0] (flat, at the minimum) | [0, 0] |

这就是图像中的梯度下降.

### 优化的联系

训练一个神经网络是优化.你有一个损失函数 L ((w1, w2, ..., wn) 测量模型是多么错误.你想尽量减少它.

```
Gradient descent update rule:

  w_new = w_old - learning_rate * dL/dw

For every weight:
  1. Compute the partial derivative of loss with respect to that weight
  2. Subtract a small multiple of it from the weight
  3. Repeat
```

学习速度控制了步骤的尺寸,太大,你超越了,太小,你爬行了.

**Loss landscape (1D slice):**

损失函数 L ((w) 随着w重量的变化,形成一个曲,具有峰值和谷口.

| Feature | Description |
|---------|-------------|
| Global minimum | The lowest point on the entire curve -- the best solution |
| Local minimum | A valley that is lower than its neighbors but not the lowest overall |
| Slope | Gradient descent follows the slope downhill from any starting point |

渐进下降跟随坡坡下坡. 它可能会被局部最小限制限制,但在高维空间 (数百万重量) 中,这很少是实际的问题.

### 数字与分析衍生品

计算衍生值有两种方法.

分析:手动应用计算规则.为 f  x = x^2,衍生式是 f  x = 2x. 正确.快.

计算 f ((x+h) 和 f ((x-h) 为一个小 h,然后使用差异.

```
Numerical (central difference):

f'(x) ~= f(x + h) - f(x - h)
          -----------------------
                  2h

h = 0.0001 works well in practice
```

数学衍生品较慢,但适用于任何函数.分析衍生品是快速的,但需要你衍生公式.神经网络框架采用第三种方法:自动差异化,它将精确衍生品进行机械计算.

### 简单函数的手动衍生品

这些衍生品,你会在ML中看到一次又一次.

```
Function        Derivative       Used in
--------        ----------       -------
f(x) = x^2     f'(x) = 2x      Loss functions (MSE)
f(x) = wx + b  f'(w) = x        Linear layer (gradient w.r.t. weight)
                f'(b) = 1        Linear layer (gradient w.r.t. bias)
                f'(x) = w        Linear layer (gradient w.r.t. input)
f(x) = e^x     f'(x) = e^x     Softmax, attention
f(x) = ln(x)   f'(x) = 1/x     Cross-entropy loss
f(x) = 1/(1+e^-x)  f'(x) = f(x)(1-f(x))   Sigmoid activation
```

对于 f ((x) = x^2:

```
f(x) = x^2    f'(x) = 2x

  x    f(x)   f'(x)   meaning
  -2    4      -4      slope tilts left (decreasing)
  -1    1      -2      slope tilts left (decreasing)
   0    0       0      flat (minimum!)
   1    1       2      slope tilts right (increasing)
   2    4       4      slope tilts right (increasing)
```

对于 f(w) = wx + b 与 x=3, b=1:

```
f(w) = 3w + 1    f'(w) = 3

The derivative with respect to w is just x.
If x is big, a small change in w causes a big change in output.
```

### 链条规则

当函数组合时,链条规则告诉你如何区分.

```
If y = f(g(x)), then dy/dx = f'(g(x)) * g'(x)

Example: y = (3x + 1)^2
  outer: f(u) = u^2       f'(u) = 2u
  inner: g(x) = 3x + 1    g'(x) = 3
  dy/dx = 2(3x + 1) * 3 = 6(3x + 1)
```

神经网络是函数的链接:输入 -> 直线 -> 激活 -> 直线 -> 激活 -> 损失.反扩散是从输出到输入中反复应用的链条规则.这是整个算法.

### 赫西亚矩阵

梯度告诉你斜率,赫西亚式告诉你曲率.

赫西亚式是二级部分衍生物的矩阵.对于函数 f ((x1, x2, ..., xn),赫西亚式的输入 (i, j) 是:

```
H[i][j] = d^2f / (dx_i * dx_j)
```

对于2变量函数 f ((x,y):

```
H = | d^2f/dx^2    d^2f/dxdy |
    | d^2f/dydx    d^2f/dy^2 |
```

**What the Hessian tells you at a critical point (where gradient = 0):**

| Hessian property | Meaning | Example surface |
|-----------------|---------|-----------------|
| Positive definite (all eigenvalues > 0) | Local minimum | Bowl pointing up |
| Negative definite (all eigenvalues < 0) | Local maximum | Bowl pointing down |
| Indefinite (mixed eigenvalues) | Saddle point | Horse saddle shape |

**Example:**f(x,y) = x^2 - y^2 (一个车函数)

```
df/dx = 2x       df/dy = -2y
d^2f/dx^2 = 2    d^2f/dy^2 = -2    d^2f/dxdy = 0

H = | 2   0 |
    | 0  -2 |

Eigenvalues: 2 and -2 (one positive, one negative)
--> Saddle point at (0, 0)
```

比较 f ((x, y) = x^2 + y^2 (一个碗):

```
H = | 2  0 |
    | 0  2 |

Eigenvalues: 2 and 2 (both positive)
--> Local minimum at (0, 0)
```

**Why the Hessian matters in ML:**

牛顿的方法使用赫西式来采取比梯度下降更好的优化步骤.

```
Newton's update:    w_new = w_old - H^(-1) * gradient
Gradient descent:   w_new = w_old - lr * gradient
```

牛顿的方法更快地接近,因为赫西式"再加速度"的梯度 - - 方向得到更小的步骤,平方向得到更大的步骤.

对于一个具有N参数的神经网络,赫西亚式是N xN.一个拥有100万参数的模型需要一个1万亿参数的矩阵.

| Method | What it uses | Cost | Convergence |
|--------|-------------|------|-------------|
| Gradient descent | First derivatives only | O(N) per step | Slow (linear) |
| Newton's method | Full Hessian | O(N^3) per step | Fast (quadratic) |
| L-BFGS | Approximate Hessian from gradient history | O(N) per step | Medium (superlinear) |
| Adam | Per-parameter adaptive rates (diagonal Hessian approx) | O(N) per step | Medium |
| Natural gradient | Fisher information matrix (statistical Hessian) | O(N^2) per step | Fast |

在实践中,亚当是深度学习的默认优化器.它通过追踪每参数的运行平均和梯度变化,便宜地接近二级信息.

### 泰勒系列近似

任何平滑函数可以通过多项式在本地进行近似:

```
f(x + h) = f(x) + f'(x)*h + (1/2)*f''(x)*h^2 + (1/6)*f'''(x)*h^3 + ...
```

接近的方法越好,但只有在 x 点附近.

**Why Taylor series matter for ML:**

- **First-order Taylor = gradient descent.**当你使用 f(x + h) ~ f(x) + f'(x) *h,你正在做一个线性近似.渐进下降将这个线性模型最小化,选择h = -lr * f'(x.

- **Second-order Taylor = Newton's method.**使用 f(x + h) ~ f(x) + f'(x) *h + (1/2) *f'(x) *h^2,你得到一个方形模型.最小化它会得到 h = -f'(x) /f'(x) - 牛顿的步骤.

- **Loss function design.**它们的Taylor扩展是很好的. 这不是意外. 流的损失使得优化可以预测.

```
Approximation order    What it captures    Optimization method
-------------------    -----------------   -------------------
0th order (constant)   Just the value      Random search
1st order (linear)     Slope               Gradient descent
2nd order (quadratic)  Curvature           Newton's method
Higher orders          Finer structure     Rarely used in ML
```

关键见解:所有基于梯度的优化实际上是将损失函数在本地接近,

### 在ML中的整体

导数告诉你变化率.整体计算积累 - - 曲线下的区域.

在ML中,你很少手动计算整体,

**Probability.**对于密度p(x的连续随机变量:
```
P(a < X < b) = integral from a to b of p(x) dx
```
在a和b之间的概率密度曲线下的区域是该范围的降落概率.

**Expected value.**根据概率权重的平均结果:
```
E[f(X)] = integral of f(x) * p(x) dx
```
预期的数据分布损失是不可或缺的.

**KL divergence.**测量两种分布的不同程度:
```
KL(p || q) = integral of p(x) * log(p(x) / q(x)) dx
```
在 VAEs,知识蒸和贝叶斯推理中使用.

**Normalization constants.**在贝叶斯推理中:
```
p(w | data) = p(data | w) * p(w) / integral of p(data | w) * p(w) dw
```
变量值是所有可能参数值的整体. 它通常是难以解决的,这就是为什么我们使用MCMC和变量推理等近似.

| Integral concept | Where it appears in ML |
|-----------------|----------------------|
| Area under curve | Probability from density functions |
| Expected value | Loss functions, risk minimization |
| KL divergence | VAEs, policy optimization, distillation |
| Normalization | Bayesian posteriors, softmax denominator |
| Marginal likelihood | Model comparison, evidence lower bound (ELBO) |

### 在计算图中多变量链条规则

链条规则不仅适用于线路中的规模函数.在神经网络中,变量扩展和合并.以下是衍生品通过简单的前进传递流动的方式:

```mermaid
graph LR
    x["x (input)"] -->|"*w"| z1["z1 = w*x"]
    z1 -->|"+b"| z2["z2 = w*x + b"]
    z2 -->|"sigmoid"| a["a = sigmoid(z2)"]
    a -->|"loss fn"| L["L = -(y*log(a) + (1-y)*log(1-a))"]
```

后行计算右到左的梯度:

```mermaid
graph RL
    dL["dL/dL = 1"] -->|"dL/da"| da["dL/da = -y/a + (1-y)/(1-a)"]
    da -->|"da/dz2 = a(1-a)"| dz2["dL/dz2 = dL/da * a(1-a)"]
    dz2 -->|"dz2/dw = x"| dw["dL/dw = dL/dz2 * x"]
    dz2 -->|"dz2/db = 1"| db["dL/db = dL/dz2 * 1"]
```

每个箭头乘以本地衍生值.任何参数的梯度是从损失到参数的路径沿线的所有本地衍生值的产量.当路径分支和合并时,你将贡献的数量 (多变链规则).

这就是反向传播:通过计算图系统地应用的链条规则,从输出到输入.

### 雅可比矩阵

当函数将向量映射到向量 (如神经网络层),其衍生物是矩阵. 雅可比安包含每个输出和每个输入的每个部分衍生物.

对于f:R^n ->R^m,雅可比亚J是一个m x n矩阵:

| | x1 | x2 | ... | xn |
|---|---|---|---|---|
| f1 | df1/dx1 | df1/dx2 | ... | df1/dxn |
| f2 | df2/dx1 | df2/dx2 | ... | df2/dxn |
| ... | ... | ... | ... | ... |
| fm | dfm/dx1 | dfm/dx2 | ... | dfm/dxn |

对于神经网络,你不会手动计算Jacobians. PyTorch处理它. 但知道它存在,有助于你理解后延伸的形状:如果一个层映射R^n到R^m,它的Jacobian是m x n.梯度通过这个矩阵的转移流向后.

### 为什么这对神经网络很重要

任何神经网络中的重量都得到一个梯度.梯度告诉你如何调整重量以减少损失.

```mermaid
graph LR
    subgraph Forward["Forward Pass"]
        I["input"] --> W1["W1"] --> R["relu"] --> W2["W2"] --> S["softmax"] --> L["loss"]
    end
```

```mermaid
graph RL
    subgraph Backward["Backward Pass"]
        dL["dL/dloss"] --> dW2["dL/dW2"] --> d2["..."] --> dW1["dL/dW1"]
    end
```

每次重量更新:
- `W1 = W1 - lr * dL/dW1`
- `W2 = W2 - lr * dL/dW2`

进步计算了预测和损失. 倒退的通过计算了损失的梯度与每一个重量. 然后每一个重量都会下坡一步. 重复数百万步. 这就是深度学习.

```figure
derivative-tangent
```

## 建立它

### 步骤1:从零开始的数值衍生

```python
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

for x in [-2, -1, 0, 1, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"x={x:2d}  f'(x) numerical={numerical:.6f}  analytical={analytical:.1f}")
```

数字衍生式与分析的一个相匹配,

### 步骤2:部分衍生品和梯度

```python
def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient

def f_multi(point):
    x, y = point
    return x**2 + 3*x*y + y**2

grad = numerical_gradient(f_multi, [1.0, 2.0])
print(f"Numerical gradient at (1,2): {[f'{g:.4f}' for g in grad]}")
print(f"Analytical gradient at (1,2): [2*1+3*2, 3*1+2*2] = [{2*1+3*2}, {3*1+2*2}]")
```

### 步骤3: 渐进下降,以找到最小的 f ((x) = x^2

```python
x = 5.0
lr = 0.1
for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    print(f"step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
```

从x=5开始,每个步骤都接近x=0 (最小).

### 步骤4: 2D函数上的渐进下降

```python
def f_2d(point):
    x, y = point
    return x**2 + y**2

point = [4.0, 3.0]
lr = 0.1
for step in range(30):
    grad = numerical_gradient(f_2d, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    loss = f_2d(point)
    if step % 5 == 0 or step == 29:
        print(f"step {step:2d}  point=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")
```

### 步骤5:数值和分析衍生品的比较

```python
import math

test_functions = [
    ("x^2",      lambda x: x**2,          lambda x: 2*x),
    ("x^3",      lambda x: x**3,          lambda x: 3*x**2),
    ("sin(x)",   lambda x: math.sin(x),   lambda x: math.cos(x)),
    ("e^x",      lambda x: math.exp(x),   lambda x: math.exp(x)),
    ("1/x",      lambda x: 1/x,           lambda x: -1/x**2),
]

x = 2.0
print(f"{'Function':<12} {'Numerical':>12} {'Analytical':>12} {'Error':>12}")
print("-" * 50)
for name, f, df in test_functions:
    num = numerical_derivative(f, x)
    ana = df(x)
    err = abs(num - ana)
    print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")
```

### 步骤 6: 数字计算赫西语

```python
def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[fxx, fxy], [fxy, fyy]]

def saddle(x, y):
    return x ** 2 - y ** 2

def bowl(x, y):
    return x ** 2 + y ** 2

H_saddle = hessian_2d(saddle, 0.0, 0.0)
H_bowl = hessian_2d(bowl, 0.0, 0.0)
print(f"Saddle Hessian: {H_saddle}")  # [[2, 0], [0, -2]] -- mixed signs
print(f"Bowl Hessian:   {H_bowl}")    # [[2, 0], [0, 2]]  -- both positive
```

座函数的Hessian有2和 -2的本值 (混合符号,确认座点). 碗有2和2的本值 (两者都是正值,确认最小值).

### 步骤7:泰勒近似在行动中

```python
import math

def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result

x0 = 0.0
for h in [0.1, 0.5, 1.0, 2.0]:
    true_val = math.sin(h)
    t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=1)
    t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=2)
    print(f"h={h:.1f}  sin(h)={true_val:.4f}  order1={t1:.4f}  order2={t2:.4f}")
```

接近x0=0, sin(x) ~ x (第一级泰勒).对小h来说,近似非常好,但对大h来说,分解.这就是为什么梯度下降在小学习率下最好工作的原因 - - 每一步都假设线性近似是准确的.

### 步骤8:为什么这对神经网络很重要

```python
import random

random.seed(42)

w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.01

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

for epoch in range(200):
    total_loss = 0
    dw = 0
    db = 0
    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y
        total_loss += error ** 2
        dw += 2 * error * x
        db += 2 * error
    dw /= len(xs)
    db /= len(xs)
    total_loss /= len(xs)
    w -= lr * dw
    b -= lr * db
    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={total_loss:.6f}")

print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
print(f"Actual:  y = 2x + 1")
```

每个基于梯度的训练循环都遵循这个模式:预测,计算损失,计算梯度,更新权重.

## 用它

通过NumPy,相同的操作更快,更简洁:

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

w, b = np.random.randn(), np.random.randn()
lr = 0.01

for epoch in range(200):
    pred = w * x + b
    error = pred - y
    loss = np.mean(error ** 2)
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)
    w -= lr * dw
    b -= lr * db

print(f"Learned: y = {w:.2f}x + {b:.2f}")
```

光器自动化了光计算,但更新循环是相同的.

## 运动

1. 实施`numerical_second_derivative(f, x)`使用`numerical_derivative`检查到x^3的第二个衍生值在x=2是12.
2. 使用梯度下降,找到最小的f ((x,y) = (x - 3) ^2 + (y + 1) ^2.从 (0, 0) 开始.答案应该接近 (3, - 1).
3. 增加动力在梯度下降循环:保持一个速度向量,积累过去梯度.比较与和没有动力的趋同速度在f ((x) = x^4 - 3x^2.

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Derivative | "The slope" | The rate of change of a function at a point. Tells you how much the output changes per unit change in input. |
| Partial derivative | "Derivative of one variable" | The derivative with respect to one variable while all others are held constant. |
| Gradient | "Direction of steepest ascent" | A vector of all partial derivatives. Points in the direction that increases the function fastest. |
| Gradient descent | "Go downhill" | Subtract the gradient (times a learning rate) from the parameters to reduce the loss. The core of neural network training. |
| Learning rate | "Step size" | A scalar that controls how big each gradient descent step is. Too large: diverge. Too small: converge slowly. |
| Chain rule | "Multiply the derivatives" | The rule for differentiating composed functions: df/dx = df/dg * dg/dx. The mathematical basis of backpropagation. |
| Jacobian | "Matrix of derivatives" | When a function maps vectors to vectors, the Jacobian is the matrix of all partial derivatives of outputs with respect to inputs. |
| Numerical derivative | "Finite differences" | Approximating a derivative by evaluating the function at two nearby points and computing the slope between them. |
| Backpropagation | "Reverse-mode autodiff" | Computing gradients layer by layer from output to input using the chain rule. How neural networks learn. |
| Hessian | "Matrix of second derivatives" | The matrix of all second-order partial derivatives. Describes the curvature of a function. Positive definite Hessian at a critical point means local minimum. |
| Taylor series | "Polynomial approximation" | Approximating a function near a point using its derivatives: f(x+h) ~ f(x) + f'(x)h + (1/2)f''(x)h^2 + ... The basis for understanding why gradient descent and Newton's method work. |
| Integral | "Area under the curve" | The accumulation of a quantity over a range. In ML, integrals define probabilities, expected values, and KL divergence. |

## 进一步阅读

- [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus)- 导体,整体和链条规则的视觉直觉
- [Stanford CS231n: Backpropagation](https://cs231n.github.io/optimization-2/)- 如何通过神经网络层流动的梯度
