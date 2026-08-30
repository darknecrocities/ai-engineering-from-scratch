# 损失功能

> 网络做出了预测. 基本的真相说相反. 错误多大? 这个数字是损失. 选择错误的损失函数,你的模型完全优化了错误的东西.

**Type:** Build
**Languages:** Python
**Prerequisites:** Lesson 03.04 (Activation Functions)
**Time:** ~75 minutes

## 学习目标

- 从零开始实施MSE,二进制交叉,分类交叉和对比损失 (InfoNCE)
- 解释为什么MSE未能进行分类,通过显示"预测0.5对所有"失败模式
- 涂抹标签滑滑度对跨体,并描述它如何防止过度自信的预测
- 选择回归,二进制分类,多类分类,并嵌入学习任务的正确损失函数

## 问题

通过减少MSE的模型,我们可以预测0.5的损失.

损失函数是你模型真正优化的唯一东西. 没有准确性. 没有F1的成绩. 不是你向你的经理报告的任何标准. 优化器取损失函数的梯度,调整权重,使该数量变得更小. 如果损失函数不捕捉到你关心的东西,模型会找到最便宜的数学方法来满足它,

这是一个具体的例子. 你有双重分类任务. 两个课程,50/50分. 你用MSE作为你的损失. 模型预测每次输入为0.5. 平均MSE为0.25,这是最少的可能, 这种模型没有任何歧视性,但它技术上将你的损失功能降至最低. 转向交叉缩,同样的模型被迫推向0或1,因为 -log(0.5) =0.693是一个可怕的损失,而 -log(0.99) =0.01 奖励自信正确的预测. 损失函数的选择是学习模型和测量模型之间的区别.

变得更糟.在自我监督学习中,你甚至没有标签.对比性损失完全定义了学习信号:什么值得相似,什么值得不同,以及模型应该如何将它们推离开.

## 概念

### 平均平方错误 (MSE)

预测和目标之间的二次差异,平均所有样本.

```
MSE = (1/n) * sum((y_pred - y_true)^2)
```

为什么正方位是重要的:它将大错误处罚成四次. 2 的错误的成本是 4 倍 1. 10 的错误是 100 倍. 这使得MSE 对异常值敏感 - 一个非常错误的预测占据了损失.

实际数字:如果您的模型预测住房价格,并且在$10,000 on most houses but off by $作为一个豪宅的200万,MSE将积极尝试修复那个豪宅,

对于预测的MSE梯度是:

```
dMSE/dy_pred = (2/n) * (y_pred - y_true)
```

错误的线性.更大的错误会变得更大的梯度.这是回归的特征 (大错误需要大修正) 和分类的错误 (你想以指数而不是线性地惩罚自信错误的答案).

### 交叉缩损失

根据信息理论,它测量了预测概率分布和真实分布之间的差异.

**Binary Cross-Entropy (BCE):**

```
BCE = -(y * log(p) + (1 - y) * log(1 - p))
```

在此,y是真实标签 (0或1) 和p是预测概率.

为什么 -log(p) 效果:当真实标签为1并且你预测p =0.99,损失是 -log(0.99) =0.01.当你预测p =0.01,损失是 -log(0.01) =4.6.这460x差异是为什么交叉能效果.它残酷地惩罚自信错误的预测,同时几乎惩罚自信正确的预测.

梯度讲述了同样的故事:

```
dBCE/dp = -(y/p) + (1-y)/(1-p)
```

当 y = 1 和 p 接近零时,梯度是 -1/p,接近负无限.模型得到了一个巨大的信号来纠正错误.当 p 接近 1,梯度是小的.已经正确,没有什么可以纠正.

**Categorical Cross-Entropy:**

对于多类分类,具有单个加密目标.

```
CCE = -sum(y_i * log(p_i))
```

如果有10个类,正确的类得到0.1的概率 (随机猜测),则损失为 -log(0.1) = 2.3. 如果正确的类得到0.9的概率,则损失为 -log(0.9) = 0.105.模型学会集中概率质量在正确的答案.

### 为什么MSE未能进行分类

```mermaid
graph TD
    subgraph "MSE on Classification"
        P1["Predict 0.5 for class 1<br/>MSE = 0.25"]
        P2["Predict 0.9 for class 1<br/>MSE = 0.01"]
        P3["Predict 0.1 for class 1<br/>MSE = 0.81"]
    end
    subgraph "Cross-Entropy on Classification"
        C1["Predict 0.5 for class 1<br/>CE = 0.693"]
        C2["Predict 0.9 for class 1<br/>CE = 0.105"]
        C3["Predict 0.1 for class 1<br/>CE = 2.303"]
    end
    P3 -->|"MSE gradient<br/>flattens near<br/>saturation"| Slow["Slow correction"]
    C3 -->|"CE gradient<br/>explodes near<br/>wrong answer"| Fast["Fast correction"]
```

由于sigmoid和度,MSE梯度平坦化 (由于sigmoid和度). 交叉缩梯度补偿了这一点 - - 记录取消了sigmoid的平坦区域,给出强大的梯度,正是最需要的地方.

### 标签滑滑

标准的单热标签说:"这是100%的3级,其他一切都是0%".这是一个强有力的说法.

```
smooth_label = (1 - alpha) * one_hot + alpha / num_classes
```

对于阿尔法=0.1和10类:而不是 [0, 0, 1, 0, ...],目标变成 [0.01, 0.01, 0.91, 0.01, ...].模型目标是0.91而不是1.0.

为什么这有所效果:试图通过软max出口精确的模型需要将logits推到无限.这导致过度自信,损害了通用化,并使模型变得脆弱的分布转移.标签滑板将目标限制在0.9 (含alpha=0.1),保持logits在合理的范围内.GPT和大多数现代模型使用标签滑板或其相当.

### 显著损失

没有标签,没有类,只是输入的对,问题是:它们相似还是不同的?

**SimCLR-style contrastive loss (NT-Xent / InfoNCE):**

像是一个图像. 创建两个增长的视图 (剪切,旋转,色彩). 这些是"正对" - - 他们应该有相似的嵌入. 批量中的每一个图像都形成"负对" - 他们应该有不同的嵌入.

```
L = -log(exp(sim(z_i, z_j) / tau) / sum(exp(sim(z_i, z_k) / tau)))
```

在 sim() 是相似的, z_i 和 z_j 是正对,总数是所有负数,tau (温度) 控制分布的度.低温 = 硬的负数 = 更加激进的分离.

实际数量:批量大小256意味着每对正数的255负.温度tau=0.07 (SimCLR默认).损失看起来像一个软最大比相似性 - 它希望正数对的相似性在所有256个选项中是最高的.

**Triplet Loss:**

采用三个输入:,正 (相同类),负 (不同的类).

```
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

边缘 (通常是0.2-1.0) 强制使正面和负面距离之间的最小差距.如果负面已经足够远,损失是零 - 没有梯度,没有更新.这使得训练效率高,但需要仔细的三角形挖掘 (选择靠近的硬负面).

### 焦点损失

对于不平衡的数据集.标准的交叉缩对待所有正确分类的例子均等.焦点损失减重简单的例子:

```
FL = -alpha * (1 - p_t)^gamma * log(p_t)
```

在p_t是真类的预测概率,而gamma控制着聚焦.在gamma=0时,这是标准的交叉.在gamma=2时 (默认):

- 简单的例子 (p_t = 0.9):重量 = (0.1) ^ 2 = 0.01.
- 硬实例 (p_t = 0.1):重量 = (0.9) ^2 = 0.81. 完全梯度信号.

焦点损失是由林等人推出的,用于对象检测,其中99%的候选区域是背景 (轻微负面).没有焦点损失,模型会沉浸在轻松的背景示例中,从来没有学会检测对象.通过它,模型将其能力集中在重要的事实中.

### 损失函数决策树

```mermaid
flowchart TD
    Start["What is your task?"] --> Reg{"Regression?"}
    Start --> Cls{"Classification?"}
    Start --> Emb{"Learning embeddings?"}

    Reg -->|"Yes"| Outliers{"Outlier sensitive?"}
    Outliers -->|"Yes, penalize outliers"| MSE["Use MSE"]
    Outliers -->|"No, robust to outliers"| MAE["Use MAE / Huber"]

    Cls -->|"Binary"| BCE["Use Binary CE"]
    Cls -->|"Multi-class"| CCE["Use Categorical CE"]
    Cls -->|"Imbalanced"| FL["Use Focal Loss"]
    CCE -->|"Overconfident?"| LS["Add Label Smoothing"]

    Emb -->|"Paired data"| CL["Use Contrastive Loss"]
    Emb -->|"Triplets available"| TL["Use Triplet Loss"]
    Emb -->|"Large batch self-supervised"| NCE["Use InfoNCE"]
```

### 失景

```mermaid
graph LR
    subgraph "Loss Surface Shape"
        MSE_S["MSE<br/>Smooth parabola<br/>Single minimum<br/>Easy to optimize"]
        CE_S["Cross-Entropy<br/>Steep near wrong answers<br/>Flat near correct answers<br/>Strong gradients where needed"]
        CL_S["Contrastive<br/>Many local minima<br/>Depends on batch composition<br/>Temperature controls sharpness"]
    end
    MSE_S -->|"Best for"| Reg2["Regression"]
    CE_S -->|"Best for"| Cls2["Classification"]
    CL_S -->|"Best for"| Emb2["Representation learning"]
```

```figure
cross-entropy-loss
```

## 建立它

### 第一步:MSE及其分数

```python
def mse(predictions, targets):
    n = len(predictions)
    total = 0.0
    for p, t in zip(predictions, targets):
        total += (p - t) ** 2
    return total / n

def mse_gradient(predictions, targets):
    n = len(predictions)
    grads = []
    for p, t in zip(predictions, targets):
        grads.append(2.0 * (p - t) / n)
    return grads
```

### 步骤2:二进制交叉

如果模型预测正确的例子为0, log(0) =负无限. 剪切阻止这一点.

```python
import math

def binary_cross_entropy(predictions, targets, eps=1e-15):
    n = len(predictions)
    total = 0.0
    for p, t in zip(predictions, targets):
        p_clipped = max(eps, min(1 - eps, p))
        total += -(t * math.log(p_clipped) + (1 - t) * math.log(1 - p_clipped))
    return total / n

def bce_gradient(predictions, targets, eps=1e-15):
    grads = []
    for p, t in zip(predictions, targets):
        p_clipped = max(eps, min(1 - eps, p))
        grads.append(-(t / p_clipped) + (1 - t) / (1 - p_clipped))
    return grads
```

### 步骤3: 软max 的类型交叉透

软max将原始的数据转换为概率,然后我们计算了与一个热点目标的交叉位.

```python
def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]

def categorical_cross_entropy(logits, target_index, eps=1e-15):
    probs = softmax(logits)
    p = max(eps, probs[target_index])
    return -math.log(p)

def cce_gradient(logits, target_index):
    probs = softmax(logits)
    grads = list(probs)
    grads[target_index] -= 1.0
    return grads
```

软max + 交叉缩的梯度简化得很好:它只是 (预测概率 - 1) 对真实类,和 (预测概率) 对所有其他类.

### 步骤4: 标签滑滑

```python
def label_smoothed_cce(logits, target_index, num_classes, alpha=0.1, eps=1e-15):
    probs = softmax(logits)
    loss = 0.0
    for i in range(num_classes):
        if i == target_index:
            smooth_target = 1.0 - alpha + alpha / num_classes
        else:
            smooth_target = alpha / num_classes
        p = max(eps, probs[i])
        loss += -smooth_target * math.log(p)
    return loss
```

### 步骤5:对比损失 (简化信息NCE)

```python
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)

def contrastive_loss(anchor, positive, negatives, temperature=0.07):
    sim_pos = cosine_similarity(anchor, positive) / temperature
    sim_negs = [cosine_similarity(anchor, neg) / temperature for neg in negatives]

    max_sim = max(sim_pos, max(sim_negs)) if sim_negs else sim_pos
    exp_pos = math.exp(sim_pos - max_sim)
    exp_negs = [math.exp(s - max_sim) for s in sim_negs]
    total_exp = exp_pos + sum(exp_negs)

    return -math.log(max(1e-15, exp_pos / total_exp))
```

### 阶段6:MSE与分类交叉透

训练从04课 (循环数据集) 训练相同的网络,使用两个损失函数.

```python
import random

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class LossComparisonNetwork:
    def __init__(self, loss_type="bce", hidden_size=8, lr=0.1):
        random.seed(0)
        self.loss_type = loss_type
        self.lr = lr
        self.hidden_size = hidden_size

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        if self.loss_type == "mse":
            d_loss = 2.0 * (self.out - target)
        else:
            eps = 1e-15
            p = max(eps, min(1 - eps, self.out))
            d_loss = -(target / p) + (1 - target) / (1 - p)

        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            self.w2[i] -= self.lr * d_out * self.h[i]
            for j in range(2):
                self.w1[i][j] -= self.lr * d_h * self.x[j]
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def compute_loss(self, pred, target):
        if self.loss_type == "mse":
            return (pred - target) ** 2
        else:
            eps = 1e-15
            p = max(eps, min(1 - eps, pred))
            return -(target * math.log(p) + (1 - target) * math.log(1 - p))

    def train(self, data, epochs=200):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                self.backward(y)
                total_loss += self.compute_loss(pred, y)
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append((avg_loss, accuracy))
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses
```

## 用它

PyTorch提供了所有标准损失函数,

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

predictions = torch.tensor([0.9, 0.1, 0.7], requires_grad=True)
targets = torch.tensor([1.0, 0.0, 1.0])

mse_loss = F.mse_loss(predictions, targets)
bce_loss = F.binary_cross_entropy(predictions, targets)

logits = torch.randn(4, 10)
labels = torch.tensor([3, 7, 1, 9])
ce_loss = F.cross_entropy(logits, labels)
ce_smooth = F.cross_entropy(logits, labels, label_smoothing=0.1)
```

使用`F.cross_entropy`(没有)`F.nll_loss`通过将软max 单独应用,然后取日志变得不稳定,你在减小大指数时失去精度.

对于对比性学习,大多数团队使用自定义实现或库,如`lightly`或`pytorch-metric-learning`核心循环总是相同的:计算对式相似性,创造出对正和负的软最大,反向扩散.

## 运送它

这一课产生了:
- `outputs/prompt-loss-function-selector.md`-- 选择正确的损失函数的可重复使用提示
- `outputs/prompt-loss-debugger.md`-- 诊断提示,当你的损失曲线看起来错误时

## 运动

1. 运行Huber损失 (柔顺L1损失),这是小错误的MSE和大错误的MAE. 训练一个预测y = sin(x的回归网络,使用MSE与Huber当5%的训练目标随机增加噪音 (异常). 进行最终测试错误的比较.

2. 加入二进制分类训练循环中的焦点损失. 创建一个不平衡的数据集 (90%类0,10%类1). 在200个时代后的少数类回忆中,对比标准 BCE 与焦点损失 (gamma=2) .

3. 实现半硬负挖矿的三分数损失.为5类生成2D嵌入数据.对于每个,找到最硬的负,比正面还远 (半硬).将收缩与随机三分数选择进行比较.

4. 运行MSE与跨缩比较,但在训练期间跟踪每个层的梯度大小.按时段绘制平均梯度规范.检查模型最不确定时代的跨缩产生更大的梯度.

5. 实现KL分离损失并验证将KL ((真实的意思预测) 降至最低时,当真正的分布是单热时,会产生与交叉缩相同的梯度.然后尝试软目标 (如知识蒸) 试验,其中"真实的"分布来自教师模型的软max输出.

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Loss function | "How wrong the model is" | A differentiable function mapping predictions and targets to a scalar that the optimizer minimizes |
| MSE | "Average squared error" | Mean of squared differences between predictions and targets; penalizes large errors quadratically |
| Cross-entropy | "The classification loss" | Measures divergence between predicted probability distribution and true distribution using -log(p) |
| Binary cross-entropy | "BCE" | Cross-entropy for two classes: -(y*log(p) + (1-y)*log(1-p)) |
| Label smoothing | "Softening the targets" | Replacing hard 0/1 targets with soft values (e.g., 0.1/0.9) to prevent overconfidence and improve generalization |
| Contrastive loss | "Pull together, push apart" | A loss that learns representations by making similar pairs close and dissimilar pairs far in embedding space |
| InfoNCE | "The CLIP/SimCLR loss" | Normalized temperature-scaled cross-entropy over similarity scores; treats contrastive learning as classification |
| Focal loss | "The imbalanced data fix" | Cross-entropy weighted by (1-p_t)^gamma to down-weight easy examples and focus on hard ones |
| Triplet loss | "Anchor-positive-negative" | Pushes anchor closer to positive than negative by at least a margin in embedding space |
| Temperature | "Sharpness knob" | A scalar divisor on logits/similarities that controls how peaked the resulting distribution is; lower = sharper |

## 进一步阅读

- 林等人",密集物体检测的焦点损失" (2017) -- 引入对物体检测的极端类分类失衡处理的焦点损失 (RetinaNet)
- 陈等人",视觉表示的对比性学习的简单框架" (SimCLR, 2020) - 定义了与NT-Xent损失的现代对比性学习管道
- 谢吉迪等人",重新思考初始架构" (2016) -- 作为规范化技术引入了标签平滑,现在是大多数大型模型的标准
- 希顿等人",神经网络中的知识蒸" (2015) -- 使用软目标和KL分离的知识蒸,这是模型压缩的基础
