# 规范和距离

> 你的距离函数定义了"类似"的意思.

**Type:** Build
**Language:**字符串
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90 minutes

## 学习目标

- 实现L1,L2,kosine,Mahalanobis,Jaccard,并从零编辑距离函数
- 选择给定的 ML 任务的适当距离指标,并解释替代方案为什么失败
- 连接L1和L2标准到LASSO和Ridge规范化及其几何限制区域
- 展示相同数据集如何在不同的指标下产生不同的近邻

## 问题

你有两个向量. 也许它们是文字嵌入式. 也许它们是用户配置文件. 也许它们是像素阵列. 你需要知道:它们是多么接近?

答案完全取决于你选择的距离函数.两个数据点可以在一个指标下是最接近的邻居,在另一个指标下是遥远的隔离.你的KN分类器,推引擎,向量数据库,集群算法,损失函数 - - 所有这些都取决于这个选择.

没有通用最佳距离.L2用于空间数据.宇宙相似性占据了NLP的主导地位.杰卡德处理集合.编辑距离处理字符串.马哈拉诺比斯计算了相关性.瓦斯斯坦移动了概率质量.每个编码了不同的假设关于"相似"的意思.

这一课将从零开始构建每个主要距离函数, 显示每个工具是什么时候正确的工具, 并展示相同的数据如何产生完全不同的近邻,

## 概念

### 标准:测量向量大小

标准测量向量的"大小".两个向量的每一个距离函数都可以作为它们的差异的标准写成: d(a, b) =a - b) 时时.

### 标准 (曼哈顿距离)

标准L1总结了所有组件的绝对值.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它被称为曼哈顿距离,因为它测量你在城市网格上走多远,

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

什么时候使用L1:
- 高维度稀疏数据 (文本特征,单热编码)
- 当你想要强度到异常值 (一个巨大的差异不主导)
- 特性选择问题 (L1规律化促进稀疏性)

连接到L1调整:1加到你的损失函数 (Lasso) 处罚绝对重值的总和.这将小重量推到完全零,执行自动特征选择.L1处罚在重量空间中创造了钻石形状的限制区域,钻石的角落位于某些重量为零的轴上.

连接到损失函数:平均绝对错误 (MAE) 是预测和目标之间的平均L1距离.它将所有错误线性地处罚,使其与MSE相比强到异常.

### L2标准 (尤克利德距离)

标准是直线距离,正方根是正方体组件的总数.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这就是你在几何课中学到的距离.

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

什么时候使用L2:
- 低至中等维度连续数据
- 当特征尺度可比较时
- 物理距离 (空间数据,传感器读数)
- 像素级的图像相似性

连接到L2调整化 (Ridge):添加到不2^w 输出__2的函数将大重量处罚.就像L1,它不会把重量推到零.它将所有重量比例缩小到零.L2的惩罚创造了圆形的限制区域,因此轴上没有角落.重量变得小,但很少是完全零.

连接到损失函数:平均平方错误 (MSE) 是 L2 距离的平均平方.

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### 标准:一般家庭

L1和L2是Lp标准的特殊案例:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同值的p产生不同形状的"单体球" (所有点在距离原点1的集合):

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### 无限度标准 (切比什夫距离)

随着p接近无限,Lp标准将趋于最大绝对组件.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两个点之间的距离由它们最不同的地方决定.

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

什么时候使用L-无限:
- 如果任何一个维度的最差偏差是重要的
- 棋牌板 (棋牌中的国王在L无限的运动:一个朝任何方向的步骤成本1)
- 制造容量 (每个尺寸都必须符合规范)

### 科西因相似性和科西因距离

两向量之间的角度,不考虑它们的大小.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

它从 -1 (相反方向) 到 +1 (相同方向).垂直向量具有 0 的共数相似性.

位距离将其转换为距离:位距离 = 1 -位相似性. 这从0 (相同的方向) 到2 (相反的方向).

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么Cosine主导NLP和嵌入式:在文本中,文档长度不应该影响相似性. 关于猫的文件是比其他关于猫的文件长两倍的文件仍然应该是"相似的". 两个文件的词分布相同,但长度不同,指向相同方向,得到了1.0的相似性.

什么时候使用可西因相似性:
- 文本相似性 (TF-IDF向量,词嵌入,句子嵌入)
- 任何域,其中大小是噪音,方向是信号
- 推系统 (用户偏好向量)
- 嵌入搜索 (向量数据库几乎总是使用共数或点数值)

### 点产品相似性与可西因相似性

两个向量的点乘法是:

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

两种大小均可正常化时,当两个向量都已经单元均可正常化时 (大小=1),点数和点数相似性是相同的.

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

当它们不同时:点产品包含大小信息.一个大小的向量获得更高的点产品分数.在某些检索系统中,这很重要,你希望"受欢迎"项目排名更高.大小作为隐含的质量或重要性信号.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

在实践中:
- 需要纯方向相似时使用可西因相似
- 使用点数量,当大小带有意义的信息时
- 许多向量数据库 (Pinecone,Weaviate,Qdrant) 让你在它们之间选择
- 如果你的嵌入式是L2正常化的,选择不重要

### 马哈拉诺比斯距离

圆距离对待所有维度均等,但如果你的特征相对或有不同的尺度,L2会产生误导性结果.

马哈拉诺比距离对数据的共变结构负责.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

在此,S是数据的共变矩阵.

直观:马哈拉诺比斯距离首先调解和正常化数据 (白化),然后计算了转换空间中的L2距离.如果S是身份矩阵 (不调整,单元变异特征),马哈拉诺比斯距离将降低到尤克利德距离.

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

什么时候使用Mahalanobis距离:
- 异常检测 (与平均值的马哈拉诺比斯距离较大的点是异常的)
- 特性有不同的尺度和相关性时的分类
- 当你有足够的数据来估计可靠的covariance矩阵
- 制造业质量控制 (多变过程监测)

### 卡德 (对集) 的相似性

杰卡德的相似度测量重叠了两个组.

```
J(A, B) = |A intersect B| / |A union B|
```

距离从0 (没有重叠) 到1 (相同的集合).

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

什么时候使用 Jaccard:
- 标签,类别或特征的组进行比较
- 基于词语存在 (而不是频率) 的文件相似性
- 接近重复检测 (Jaccard的MinHash近似)
- 进行二进制特征向量比较 (存在/缺席数据)
- 评估细分模型 (欧盟交叉 = 雅卡德)

### 修改距离 (莱文施泰恩距离)

编辑距离计算一个字符串转换到另一个字符串所需的单字符操作的最小数量.

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

通过动态编程计算.填写一个矩阵,输入 (i, j) 是字符串 A 的第一个 i 字符和字符串 B 的第一个 j 字符之间的编辑距离.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

编辑距离的使用时间:
- 检查和纠正拼音
- 基因序列配列 (有权重操作)
- 模糊的连串匹配
- 混杂的文本数据的排版

### KL差距 (不是距离,但用作一个)

 KL 分差是如何不同于另一个概率分布的测量. 这在第09课中涵盖,但它属于这个讨论,因为人们使用它作为一个"距离",尽管它不是一个.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键属性:KL差距不是对称.

```
D_KL(P || Q) != D_KL(Q || P)
```

这意味着它不符合距离指标的基本要求. 它也不满足三角形不平等. 它是差距,而不是距离.

前进 KL (D_KL(P  Q)) 是"寻找意义":Q试图涵盖P的所有模式.
逆 KL (D_KL(Q 时 P)) 是"模式寻找":Q 专注于P的单个模式.

当你看到KL分离时:
- 向前的向前的向前的向
- 知识蒸 (学生试图匹配教师的分布)
- 根据该规定,在"C"中,C"的标题是"C"的标题.
- 政策梯度方法 (限制政策更新)

### 瓦斯斯坦距离 (地球移动距离)

瓦斯斯特恩的距离测量了转换一个概率分布到另一个所需的最小"工作".想象它是:如果一个分布是泥土堆,另一个是洞,你需要移动多少泥土,到底要走多远?

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对于1D分布,它简化为累积分布函数的绝对差异的整体:

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

瓦斯斯特林为什么重要:
- 它是一个真实的指标 (对称,满足三角形不平等)
- 它提供梯度,即使分布不重叠 (KL分离到无限)
- 这种特性使得它成为瓦斯斯坦GAN (WGAN) 的核心,解决了原始GAN的训练不稳定性

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

什么时候使用Wasserstein:
- 培训 (WGAN,WGAN-GP)
- 无法重叠的分布进行比较
- 优质交通问题
- 图像检索 (比较颜色的图形图)

### 为什么不同任务需要不同的距离

| Task | Best distance | Why |
|------|--------------|-----|
| Text similarity | Cosine | Magnitude is noise, direction is meaning |
| Image pixel comparison | L2 | Spatial relationships matter, features are comparable scale |
| Sparse high-dim features | L1 | Robust, does not amplify rare large differences |
| Set overlap (tags, categories) | Jaccard | Data is naturally set-valued, not vectorial |
| String matching | Edit distance | Operations map to human editing intuition |
| Outlier detection | Mahalanobis | Accounts for feature correlations and scales |
| Comparing distributions | KL divergence | Measures information lost by using Q instead of P |
| GAN training | Wasserstein | Provides gradients even when distributions do not overlap |
| Embeddings (vector DB) | Cosine or dot product | Embeddings are trained to encode meaning in direction |
| Recommendation | Dot product | Magnitude can encode popularity or confidence |
| DNA sequences | Weighted edit distance | Substitution costs vary by nucleotide pair |
| Manufacturing QC | L-infinity | Worst-case deviation in any dimension matters |

### 连接到损失功能

损失函数是对预测与目标的距离函数.

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 与规范化的联系

规律化增加了对重量的标准处罚.

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

为什么L1产生稀疏性,但L2没有:在2D权重空间中描绘制约束区域.L1是一个钻石,L2是一个圆.损失函数的轮 (圆) 在角落上最有可能触摸钻石,其中一个权重是零.它们在平滑点触摸圆,其中两个权重都是非零的.

### 寻找最接近的邻居

每个距离函数都意味着一个最近邻居搜索问题:给出查询点,在数据集中找到最接近的点.

对于大数据集,这太慢. 对于大数据集,这太慢.

接近近邻 (ANN) 算法以小的精度进行交易,

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

它们是现代向量数据库中的主导算法.它构建了一个多层图表,每个节点与其近邻近的近距离连接.搜索从顶层开始 (sparse,长跳) 降至下层 (密集,短跳).

```figure
norm-unit-balls
```

## 建立它

### 步骤1:所有标准和距离函数

看到`code/distances.py`每个函数都是从零开始构建的,只使用基本的Python数学.

### 步骤2:相同的数据,不同的距离,不同的邻居

演示在`distances.py`创建数据集,选择查询点,并显示最近邻居如何根据距离度量变化.在L1下"最近"的点可能不是L2或Cosine下最接近.

### 步骤3: 嵌入类似性搜索

该代码包括一个模拟嵌入式类似性搜索,该代码使用kosine相似性与L2距离的查询找到最相似的"文件",表明排名可能有所不同.

## 用它

最常见的实用用途:在向量数据库中找到类似的项目.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你打电话时`model.encode(text)`然后搜索一个向量数据库,这是在罩杯下发生的事情.嵌入模型将文本映射到向量中.向量数据库计算了查询向量和每个存储的向量之间的共数相似性 (或点产量),使用ANN算法避免检查它们所有.

## 运动

1. 计算 (1, 2, 3) 和 (4, 0, 6) 之间的 L1, L2 和 L-无限距离. 检查 L-inf <= L2 <= L1 对于任何一对点都适用. 证明为什么这个顺序是保证的.

2. 创建两个向量,其中的相似性是很高 (> 0.9) 但L2距离是很大 (> 10). 几何解释发生了什么. 然后创建两个向量,其中的相似性是很小 (< 0.3) 但L2距离是很小 (< 0.5).

3. 执行一个函数,将数据集和查询点取回L1,L2,Cosine和Mahalanobis距离下最接近的邻居.找到四个不同意哪个点.

4. 通过使用CDF方法手动计算[0.5,0.5,0.0,0,0]和[0,0,0,0,5,0.5]之间的瓦斯斯坦距离.然后计算[0.25,0.25,0.25,0.25]和[0,0,0,0.5,0.5].哪个更大,为什么?

5. 实现 MinHash 实现近似Jaccard相似性.生成100个随机集合,计算所有对的精确Jaccard,并使用 50,100和200个哈希函数与 MinHash近似相比较.绘制近似错误.

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Norm | "Size of a vector" | A function that maps a vector to a non-negative scalar, satisfying triangle inequality, absolute homogeneity, and zero only for the zero vector |
| L1 norm | "Manhattan distance" | Sum of absolute component values. Produces sparsity in optimization. Robust to outliers |
| L2 norm | "Euclidean distance" | Square root of sum of squared components. The straight-line distance in Euclidean space |
| Lp norm | "Generalized norm" | The p-th root of the sum of p-th powers of absolute components. L1 and L2 are special cases |
| L-infinity norm | "Max norm" or "Chebyshev distance" | The maximum absolute component value. The limit of Lp as p approaches infinity |
| Cosine similarity | "Angle between vectors" | Dot product normalized by both magnitudes. Ranges from -1 to +1. Ignores vector length |
| Cosine distance | "1 minus cosine similarity" | Converts cosine similarity to a distance. Ranges from 0 to 2 |
| Dot product | "Unnormalized cosine" | Sum of component-wise products. Equals cosine similarity times both magnitudes |
| Mahalanobis distance | "Correlation-aware distance" | L2 distance in a space that has been whitened (decorrelated and normalized) using the data covariance matrix |
| Jaccard similarity | "Set overlap" | Size of intersection divided by size of union. For sets, not vectors |
| Edit distance | "Levenshtein distance" | Minimum insertions, deletions, and substitutions to transform one string into another |
| KL divergence | "Distance between distributions" | Not a true distance (not symmetric). Measures extra bits from using Q to encode P |
| Wasserstein distance | "Earth mover's distance" | Minimum work to transport mass from one distribution to another. A true metric |
| Approximate nearest neighbor | "ANN search" | Algorithms (HNSW, LSH, IVF) that find approximately closest points much faster than exact search |
| HNSW | "The vector DB algorithm" | Hierarchical Navigable Small World graph. Multi-layer graph for fast approximate nearest neighbor search |
| L1 regularization | "Lasso" | Adding the L1 norm of weights to the loss. Drives weights to zero (sparsity) |
| L2 regularization | "Ridge" or "weight decay" | Adding the squared L2 norm of weights to the loss. Shrinks weights toward zero without sparsity |
| Elastic Net | "L1 + L2" | Combines L1 and L2 regularization. Handles correlated feature groups better than either alone |

## 进一步阅读

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)- 测量数据库用于数亿次的ANN搜索
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875)- 报纸介绍了地球移动器的距离与GAN
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876)- 基础的ANN算法
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781)- Word2Vec,其中的嵌入式的默认变得是 cosine 类似性
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html)- 距离指标和邻居算法的实用指南
