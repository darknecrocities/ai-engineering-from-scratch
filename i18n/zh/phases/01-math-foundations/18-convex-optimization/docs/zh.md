# 曲的优化

> 曲的问题有一个谷,神经网络有数百万.

**Type:** Build
**Language:**字符串
**Prerequisites:** Phase 1, Lessons 04 (Calculus for ML), 08 (Optimization)
**Time:** ~90 minutes

## 学习目标

- 测试一个函数是否曲,使用定义,第二衍生和赫西标准
- 运用牛顿的方法,并将其方形融合与梯度下降进行比较
- 使用拉格兰奇乘法解决限制优化问题,并解释KKT条件
- 解释为什么神经网络损失景观不凸,但SGD仍然找到好的解决方案

## 问题

课8教你梯度下降,动力和亚当.这些优化者在任何表面上都会下山. 但它们没有保证.在非形的景观上,梯度下降可能会降落在一个糟糕的地方最低值,卡在车点上,或者永远振荡.你无论如何都使用它,因为神经网络是非形的,没有替代方案.

但是机器学习中的许多问题都是曲的.线性回归,物流回归,SVM,LASSO,脊坡回归.对于这些问题来说,存在更强大的东西:优化与数学保障.曲的问题完全有一个谷.任何走下坡的算法都将达到全球最低水平.不需要重新启动.没有学习率的时间表.没有祈祷.

了解曲性有三个作用.首先,它告诉你当你的问题是容易 (曲) 时与硬 (非曲) 时.第二,它给你提供了比如牛顿曲问题的方法的更快工具.第三,它解释了整个 ML 中出现的概念:规律化作为一个限制,SVM 中的二元性,以及深度学习为什么尽管违反了曲性给你的每一个好特性,但它仍然有效.

## 概念

### 曲的集合

集合 S 是曲的,如果在 S 中的任何两个点,它们之间的线段也完全位于 S 中.

| Convex sets | Not convex |
|---|---|
| **Rectangle**: any two points inside can be connected by a line segment that stays inside | **Star/crescent shape**: a line between two interior points can pass outside the set |
| **Triangle**: same property holds for all interior points | **Donut/annulus**: the hole means some line segments leave the set |
| The line segment between any two points stays within the set | The line segment between some pairs of points exits the set |

形式测试:对于任何S中的x,y点和任何[0,1]中的t点,点 tx + (1-t)y也在S中.

曲集合的例子:
- 一条线,一片平面,全部是R^n
- 球 (圆,球,超球)
- 半空间: {x: a^T x <= b}
- 任何数量的曲集合的交叉点

无的集合的例子:
- 甜甜圈 (菜)
- 两个分离的圆的结合
- 任何装有""或"洞"的套件

### 曲函数

如果其域是一个曲的集合,则函数 f 凸,并且对于其域内的任何两个点 x,y 和 [0, 1] 中的任何t:

```
f(tx + (1-t)y) <= t*f(x) + (1-t)*f(y)
```

几何:图表上的任何两个点之间的线段位于图表上或图表上.

| Property | Convex function | Non-convex function |
|---|---|---|
| **Line segment test** | The line between any two points on the graph lies **above or on** the curve | The line between some points on the graph dips **below** the curve |
| **Shape** | Single bowl/valley curving upward | Multiple peaks and valleys with mixed curvature |
| **Local minima** | Every local minimum is the global minimum | Multiple local minima may exist at different heights |

常见的曲函数:
-  () = x^2 (抛物线)
- 没有任何其他方法可以做到.
-  (x) = e^x (指数式)
-  () =最大 () () ()
- 对于 x > 0 (负号)
- 任何线性函数 f ((x) = a^T x + b (形和形)

### 曲度测试

实践测试,从最简单到最严格.

**Test 1: Second derivative test (1D).**如果 f'(x) >=0为所有 x,则 f是凸.

- 形的形.
- 对于x < 0,不曲.
- 形的形.

**Test 2: Hessian test (multivariate).**如果Hessian矩阵H(x) 为所有x的正半定义,则f是凸.Hessian是第二部分衍生物的矩阵.

**Test 3: Definition test.**直接检查不等式 f(tx + (1-t) y) <= t*f(x) + (1-t) *f(y). 对于衍生值难以计算的函数来说有用.

### 为什么曲性很重要

曲优化的核心定理:

**For a convex function, every local minimum is a global minimum.**

任何下坡路都会导致相同的答案.算法保证将趋于最佳解决方案.

```mermaid
graph LR
    subgraph "Convex: ONE answer"
        direction TB
        C1["Loss surface has a single valley"] --> C2["Gradient descent ALWAYS finds the global minimum"]
    end
    subgraph "Non-convex: MANY traps"
        direction TB
        N1["Loss surface has multiple valleys and peaks"] --> N2["Gradient descent may get stuck in a local minimum"]
        N2 --> N3["Global minimum might be missed"]
    end
```

后果:
- 没有必要随机重新启动
- 没有需要复杂的学习时间表
- 融合证明是可能的 (速度取决于函数属性)
- 解决方案是独一无二的 (高达平面区域)

### 在 ML 中,形与非形

| Problem | Convex? | Why |
|---------|---------|-----|
| Linear regression (MSE) | Yes | Loss is quadratic in weights |
| Logistic regression | Yes | Log-loss is convex in weights |
| SVM (hinge loss) | Yes | Maximum of linear functions |
| LASSO (L1 regression) | Yes | Sum of convex functions is convex |
| Ridge regression (L2) | Yes | Quadratic + quadratic = convex |
| Neural network (any loss) | No | Nonlinear activations create non-convex landscape |
| k-means clustering | No | Discrete assignment step |
| Matrix factorization | No | Product of unknowns |

随着线性输出的线性模型是曲的.当你添加隐藏的层,不线性激活,曲的断裂.

### 赫西亚矩阵

函数f:R^n -> R的Hessian H是第二部分衍生物的n x n矩阵.

```
H[i][j] = d^2 f / (dx_i dx_j)
```

对于 f ((x, y) = x^2 + 3xy + y^2:

```
df/dx = 2x + 3y       d^2f/dx^2 = 2      d^2f/dxdy = 3
df/dy = 3x + 2y       d^2f/dydx = 3      d^2f/dy^2 = 2

H = [ 2  3 ]
    [ 3  2 ]
```

赫西亚人告诉你关于曲线:
- 所有正值的自值值:函数在每个方向上方曲线 (在那个点上曲线)
- 所有负值的自值:各方向向下曲线 (形,局部最大)
- 混合标志:座点 (在某些方向上,在其他方向下)
- 零自值:在该方向平 (退化)

为了曲性,赫西亚必须在任何地方都是正半确的 (所有本值 >= 0),而不仅仅是在一个点.

### 牛顿的方法

渐进式下降使用第一级信息 (梯度).牛顿的方法使用第二级信息 (赫西式).它适应当前点的方形近似,直接跳到那个方形的最小.

```
Update rule:
  x_new = x - H^(-1) * gradient

Compare to gradient descent:
  x_new = x - lr * gradient
```

牛顿的方法将规模学习速度取代为反向赫西式. 这自动调整了基于本地曲线的步骤大小和方向.

```mermaid
graph TD
    subgraph "Gradient Descent"
        GD1["Start"] --> GD2["Step 1"]
        GD2 --> GD3["Step 2"]
        GD3 --> GD4["..."]
        GD4 --> GD5["Step ~500: Converged"]
        GD_note["Follows gradient blindly — many small steps"]
    end
    subgraph "Newton's Method"
        NM1["Start"] --> NM2["Step 1"]
        NM2 --> NM3["..."]
        NM3 --> NM4["Step ~5: Converged"]
        NM_note["Uses curvature for optimal steps"]
    end
```

优势:
- 接近最小的四方相近 (每一步的错误方形)
- 没有调节的学习速度
- 规模变异性 (不管你如何参数问题,都能工作)

缺点:
- 计算Hessian成本 O ^2) 存储和 O ^3) 倒换
- 对于一个数百万重量的神经网络,即10^12的输入和10^18的操作
- 对于深度学习来说不实用

### 限制优化

无限制优化:将 f ((x) 最小化在所有x 上.
限制优化:尽量减少 f ((x) 受到限制.

实际问题有限制.你想尽量降低成本,但预算有限.你想尽量减少错误,但模型的复杂性有限.

```mermaid
graph LR
    subgraph "Unconstrained"
        U1["Loss function"] --> U2["Free minimum: lowest point of the loss surface"]
    end
    subgraph "Constrained"
        C1["Loss function"] --> C2["Constrained minimum: lowest point within the feasible region"]
        C3["Constraint boundary limits the search space"]
    end
```

### 缩乘法

拉格兰奇乘法将一个限制的问题转化为一个不受限制的问题.

问题:将 f  x 减至 g  x = 0 值.

解决方案:引入一个新的变量 (拉格兰奇乘法 lambda) 并解决无限制的问题:

```
L(x, lambda) = f(x) + lambda * g(x)
```

在溶液中,L的梯度为零:

```
dL/dx = df/dx + lambda * dg/dx = 0
dL/dlambda = g(x) = 0
```

几何直觉:在限制最小时,f的梯度必须与限制g的梯度平行.如果它们不平行,你可以沿着限制表面移动,进一步减少f.

```mermaid
graph LR
    A["Contours of f(x,y): concentric ellipses"] --- S["Solution point"]
    B["Constraint curve g(x,y) = 0"] --- S
    S --- C["At the solution, gradient of f is parallel to gradient of g"]
```

举个例子:最小化f(x,y) =x^2 + y^2 归属于x + y = 1.

```
L = x^2 + y^2 + lambda(x + y - 1)

dL/dx = 2x + lambda = 0  =>  x = -lambda/2
dL/dy = 2y + lambda = 0  =>  y = -lambda/2
dL/dlambda = x + y - 1 = 0

From first two: x = y
Substituting: 2x = 1, so x = y = 0.5, lambda = -1
```

线 x + y = 1 的最接近原点是 (0.5,0.5).

### 卡卡特条件

卡鲁什-库恩-图克条件将拉格兰奇乘法扩展到不平等的限制.

问题:以 g_i(x) <= 0 为 i = 1, ..., m 减小 f  x.

 KKT条件 (为了最佳效果而必要):

```
1. Stationarity:    df/dx + sum(lambda_i * dg_i/dx) = 0
2. Primal feasibility:  g_i(x) <= 0  for all i
3. Dual feasibility:    lambda_i >= 0  for all i
4. Complementary slackness:  lambda_i * g_i(x) = 0  for all i
```

补充性宽松性是关键的见解:限制是活跃的 (g_i = 0,解决方案位于边界) 或乘法是零 (限制并不重要).一个不影响解决方案的限制是 lambda = 0.

支持向量是限制活动的数据点 (lambda > 0).所有其他数据点都有 lambda = 0,不会影响决策边界.

### 规范化作为限制优化

它们不是任意的俩,而是隐藏的限制优化问题.

**L2 regularization (Ridge):**

```
minimize  Loss(w)  subject to  ||w||^2 <= t

Equivalent unconstrained form:
minimize  Loss(w) + lambda * ||w||^2
```

限制的不变在不变^2 <= t定义一个球 (圆在2D,球3D).解决方案是输入轮首先触摸这个球.

**L1 regularization (LASSO):**

```
minimize  Loss(w)  subject to  ||w||_1 <= t

Equivalent unconstrained form:
minimize  Loss(w) + lambda * ||w||_1
```

限制的定义是钻石 (转形2D方形).

| Property | L2 constraint (circle) | L1 constraint (diamond) |
|---|---|---|
| **Constraint shape** | Circle (sphere in higher dims) | Diamond (rotated square in 2D) |
| **Where loss contour touches** | Smooth boundary — any point on the circle | Corner — aligned with an axis |
| **Solution behavior** | Weights are small but nonzero | Some weights are exactly zero (sparse) |
| **Result** | Weight shrinkage | Feature selection |

这解释了L1为什么生产稀疏的模型 (特征选择),而L2只缩小重量.钻石的角落与轴相一致.损失轮更有可能触及角落,设置一个或多个重量完全为零.

### 两性

每个限制优化问题 (原始) 都有一个伴侣问题 (二元).对于曲的问题,原始和二元具有相同的最佳值.这是强大的二元性.

拉格兰基双重函数:

```
Primal: minimize f(x) subject to g(x) <= 0
Lagrangian: L(x, lambda) = f(x) + lambda * g(x)
Dual function: d(lambda) = min_x L(x, lambda)
Dual problem: maximize d(lambda) subject to lambda >= 0
```

为什么二元性很重要:
- 双重问题有时比原始问题更容易解决
- 解决SVM的形式是双重的,问题取决于数据点之间的点产品 (实现内核技巧)
- 双为原始最佳的下限提供,用于检查溶液质量

对于SVM,具体:

```
Primal: find w, b that maximize the margin 2/||w|| subject to
        y_i(w^T x_i + b) >= 1 for all i

Dual:   maximize sum(alpha_i) - 0.5 * sum_ij(alpha_i * alpha_j * y_i * y_j * x_i^T x_j)
        subject to alpha_i >= 0 and sum(alpha_i * y_i) = 0

The dual only involves dot products x_i^T x_j.
Replace x_i^T x_j with K(x_i, x_j) to get the kernel trick.
```

### 尽管没有曲性,但深度学习为什么有效

由于神经网络损失功能非常不曲.根据每种经典措施,优化它们都应该失败.然而,静态梯度下降可靠地找到好解决方案.

**Most local minima are good enough.**在高维度空间中,随机关键点 (梯度为零) 绝大多数是车点,而不是本地最小值.存在的少数本地最小值往往接近全球最小值.当参数空间有数百万维度时,陷入一个可怕的本地最小值极不可能.

**Saddle points, not local minima, are the real obstacle.**在一个具有 n 参数的函数中,点具有正面和负面曲线方向的混合.对于高维度的随机关键点,所有 n 个体值的正面 (本地最小) 概率大约为 2 ^ - n.几乎所有关键点都是点.SGD 的噪音有助于逃脱它们.

**Overparameterization smooths the landscape.**网络比训练示例更多的参数,更平滑,更连接的损失表面.更广泛的网络具有较少的恶性本地最小值.这与直觉相反,但经验一致.

**Loss landscape structure:**

| Property | Low-dimensional space | High-dimensional space |
|---|---|---|
| **Landscape** | Many isolated peaks and valleys | Smoothly connected valleys |
| **Minima** | Many isolated local minima | Few bad local minima; most are near-optimal |
| **Navigation** | Hard to find global minimum | Many paths lead to good solutions |
| **Critical points** | Mix of local minima and saddle points | Overwhelmingly saddle points, not local minima |

**Stochastic noise acts as implicit regularization.**微批次 SGD 增加噪音,防止落入的最小.的最小超适应;平面最小一般化.噪音偏差优化损失景观的平面区域.

### 实际上使用的二级方法

对于大型模型来说,纯牛顿的方法是不实用的.

**L-BFGS (Limited-memory BFGS):**通过使用最后的 m 梯度差异,接近逆赫西安.需要O(mn) 记忆而不是O(n^2).对高达 ~ 10,000 个参数的问题很好.用于经典ML (物流回归,CRF) 但不是深度学习.

**Natural gradient:**根据Fisher的数据测量,Fisher的数据测量对数据测量进行了测量,并将数据测量进行测量.

**Hessian-free optimization:**使用结合梯度来解决Hx = g,而没有形成H.只需要Hessian向量产品,通过自动分化可以在O ((n) 时间计算.

**Diagonal approximations:**亚当的第二个时刻是赫西亚人的斜角近似.AdaHessian通过哈辛森估计器使用实际的赫西亚斜角元素扩展这一点.

| Method | Memory | Per-step cost | When to use |
|--------|--------|--------------|-------------|
| Gradient descent | O(n) | O(n) | Baseline, large models |
| Newton's method | O(n^2) | O(n^3) | Small convex problems |
| L-BFGS | O(mn) | O(mn) | Medium convex problems |
| Adam | O(n) | O(n) | Deep learning default |
| K-FAC | O(n) | O(n) per layer | Research, large-batch training |

```figure
convex-vs-nonconvex
```

## 建立它

### 步骤1:曲度检查器

通过采样点和检查定义来实验测试曲率的函数.

```python
import random
import math

def check_convexity(f, dim, bounds=(-5, 5), samples=1000):
    violations = 0
    for _ in range(samples):
        x = [random.uniform(*bounds) for _ in range(dim)]
        y = [random.uniform(*bounds) for _ in range(dim)]
        t = random.uniform(0, 1)
        mid = [t * xi + (1 - t) * yi for xi, yi in zip(x, y)]
        lhs = f(mid)
        rhs = t * f(x) + (1 - t) * f(y)
        if lhs > rhs + 1e-10:
            violations += 1
    return violations == 0, violations
```

### 步骤2:牛顿的2D方法

通过明确的赫西式运用牛顿的方法进行比较.

```python
def newtons_method(f, grad_f, hessian_f, x0, steps=50, tol=1e-12):
    x = list(x0)
    history = [x[:]]
    for _ in range(steps):
        g = grad_f(x)
        H = hessian_f(x)
        det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
        if abs(det) < 1e-15:
            break
        H_inv = [
            [H[1][1] / det, -H[0][1] / det],
            [-H[1][0] / det, H[0][0] / det],
        ]
        dx = [
            H_inv[0][0] * g[0] + H_inv[0][1] * g[1],
            H_inv[1][0] * g[0] + H_inv[1][1] * g[1],
        ]
        x = [x[0] - dx[0], x[1] - dx[1]]
        history.append(x[:])
        if sum(gi ** 2 for gi in g) < tol:
            break
    return history
```

### 步骤3:拉格兰奇乘法解决器

通过拉格兰基基梯度下降来解决限制优化.

```python
def lagrange_solve(f_grad, g_val, g_grad, x0, lr=0.01,
                   lr_lambda=0.01, steps=5000):
    x = list(x0)
    lam = 0.0
    history = []
    for _ in range(steps):
        fg = f_grad(x)
        gv = g_val(x)
        gg = g_grad(x)
        x = [
            xi - lr * (fgi + lam * ggi)
            for xi, fgi, ggi in zip(x, fg, gg)
        ]
        lam = lam + lr_lambda * gv
        history.append((x[:], lam, gv))
    return history
```

### 步骤4:比较第一级与第二级

运行梯度下降和牛顿的方法,用相同的平方函数计算到接近的步骤.

```python
def quadratic(x):
    return 5 * x[0] ** 2 + x[1] ** 2

def quadratic_grad(x):
    return [10 * x[0], 2 * x[1]]

def quadratic_hessian(x):
    return [[10, 0], [0, 2]]
```

牛顿的方法将在1步 (这对方位数来说是确切的)  konverge.渐进下降将需要数百步,因为赫西亚的本值差异于5倍,从而产生长长的谷.

## 用它

在选择ML模型和解决器时,曲性分析直接适用于.

对于曲的问题 (物流回归,SVM,LASSO):
- 使用专用解决器 (liblinear, CVXPY, scipy.optimize.minimize with method='L-BFGS-B')
- 期待一个独特的全球解决方案
- 第二阶段的方法是实用的和快速的

对于非形问题 (神经网络):
- 使用第一级方法 (SGD,Adam)
- 接受解决方案取决于初始化和随机性
- 使用过度参数化,噪音和学习率时间表作为隐含的规范化
- 没有必要浪费时间寻找全球最低限度.

```python
from scipy.optimize import minimize

result = minimize(
    fun=lambda w: sum((y - X @ w) ** 2) + 0.1 * sum(w ** 2),
    x0=np.zeros(d),
    method='L-BFGS-B',
    jac=lambda w: -2 * X.T @ (y - X @ w) + 0.2 * w,
)
```

对于SVM,双重配方允许使用内核技巧:

```python
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)
print(f"Support vectors: {svm.n_support_}")
```

## 运动

1. **Convexity gallery.**使用检查器测试这些曲性函数: f(x) = x^4, f(x) = sin(x), f(x,y) = x^2 + y^2, f(x,y) = x*y, f(x) = max(x, 0).解释为什么每个结果都有意义.

2. **Newton vs gradient descent race.**运行两个方法从起点 (10,10) 运行 f ((x,y) = 50*x^2 + y^2 . 每个方法需要多少步骤才能达到损失 < 1e-10?当条件数 (最大至最小的赫西安自值的比例) 增加时,梯度下降发生什么?

3. **Lagrange multiplier geometry.**尽量减少f ((x,y) = (x-3)^2 + (y-3)^2以 x + 2y = 4为主. 通过检查f的梯度是否与溶液上的g的梯度平行,验证解决方案.

4. **Regularization constraint.**实现L1限制优化:最小化 (x-3)^2 + (y-2)^2 归属于 ┃x ┃ + ┃y ┃ <= 1. 显示解决方案有一个坐标等于零 (钻石限制的差).

5. **Hessian eigenvalue analysis.**计算Rosenbrock函数的Hessian在 (1,1) 和 (-1,1).计算两个点的自值.自值告诉你关于最小与远离的曲线?

## 关键词

| Term | What it means |
|------|---------------|
| Convex set | A set where the line segment between any two points in the set stays inside the set |
| Convex function | A function where the line between any two points on its graph lies above or on the graph. Equivalently, Hessian is positive semidefinite everywhere |
| Local minimum | A point lower than all nearby points. For convex functions, every local minimum is the global minimum |
| Global minimum | The lowest point of a function over its entire domain |
| Hessian matrix | The matrix of all second partial derivatives. Encodes curvature information |
| Positive semidefinite | A matrix whose eigenvalues are all non-negative. The multidimensional analogue of "second derivative >= 0" |
| Condition number | Ratio of largest to smallest eigenvalue of the Hessian. High condition number means elongated valleys and slow gradient descent |
| Newton's method | Second-order optimizer that uses the inverse Hessian to determine step direction and size. Quadratic convergence near the minimum |
| Lagrange multiplier | A variable introduced to convert a constrained optimization problem into an unconstrained one |
| KKT conditions | Necessary conditions for optimality with inequality constraints. Generalize Lagrange multipliers |
| Complementary slackness | At the solution, either a constraint is active or its multiplier is zero. Never both nonzero |
| Duality | Every constrained problem has a companion dual problem. For convex problems, both have the same optimal value |
| Strong duality | Primal and dual optimal values are equal. Holds for convex problems satisfying Slater's condition |
| L-BFGS | Approximate second-order method that stores the last m gradient differences instead of the full Hessian |
| Saddle point | A point where the gradient is zero but it is a minimum in some directions and a maximum in others |
| Overparameterization | Using more parameters than training examples. Smooths the loss landscape and reduces bad local minima |

## 进一步阅读

- [Boyd & Vandenberghe: Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)-标准教科书,可在线免费使用
- [Bottou, Curtis, Nocedal: Optimization Methods for Large-Scale Machine Learning (2018)](https://arxiv.org/abs/1606.04838)- 桥梁曲优化理论和深度学习实践
- [Choromanska et al.: The Loss Surfaces of Multilayer Networks (2015)](https://arxiv.org/abs/1412.0233)- 为什么不凸的神经网络的景观并不像看起来那么糟糕
- [Nocedal & Wright: Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5)- 牛顿方法,L-BFGS的综合参考,以及限制优化
