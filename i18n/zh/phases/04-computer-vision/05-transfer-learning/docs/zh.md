# 转移学习和调整

> 其他人花了百万个GPU小时教网络边缘,纹理和物体部分是什么样子.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 4 Lesson 04 (Image Classification)
**Time:** ~75 minutes

## 学习目标

- 根据数据集大小,域距离和计算预算,区分特征提取和细调,选择正确的特征
- 装载预训练的脊柱,取代其分类器头,并只将头部运行到一个工作基线,在20行以下
- 逐步解具有歧视性学习率的层次,所以早期的通用功能比较晚期的任务特定的更新更小
- 诊断出三个常见的故障:在未结块上,特征偏移从过高的LR,在微小的数据集上,BN统计数据崩,以及灾难性遗忘

## 问题

训练一个ResNet-50在图像网上花费了2000个GPU小时.很少有团队有这笔预算,他们运送的每一个任务.几乎每一个团队实际上运送的是一个预训练的脊柱,一个新的头脑训练在几百或几千个任务特定图像.

这不是快捷途径. 任何由 ImageNet训练的CNN的第一个块都能学习边缘和Gabor类似的过器. 接下来的几块学习了简单的纹理和动作. 中间块学习对象部分. 最后的块学习了类似于1000个图像网类别的组合. 由于自然界的边缘和纹理词汇有限,因此,该层次的第一90%几乎没有变化, 剩下的10%是你实际训练的.

转移权有三个错误等待你:破坏预训练的功能,学习率过高,通过过度结信息模型,让BatchNorm的运行统计数据向其他网络从未学习的微小数据集漂移.

## 概念

### 功能提取与细调

两种模式,根据你对预先训练的功能有多信任以及你有多少数据来选择.

```mermaid
flowchart TB
    subgraph FE["Feature extraction — backbone frozen"]
        FE1["Pretrained backbone<br/>(no gradient)"] --> FE2["New head<br/>(trained)"]
    end
    subgraph FT["Fine-tuning — end-to-end"]
        FT1["Pretrained backbone<br/>(tiny LR)"] --> FT2["New head<br/>(normal LR)"]
    end

    style FE1 fill:#e5e7eb,stroke:#6b7280
    style FE2 fill:#dcfce7,stroke:#16a34a
    style FT1 fill:#fef3c7,stroke:#d97706
    style FT2 fill:#dcfce7,stroke:#16a34a
```

基本规则:

| Dataset size | Domain distance | Recipe |
|--------------|-----------------|--------|
| < 1k images | close to ImageNet | Freeze backbone, train head only |
| 1k-10k | close | Freeze first 2-3 stages, fine-tune the rest |
| 10k-100k | any | Fine-tune end-to-end with discriminative LR |
| 100k+ | far | Fine-tune everything; consider training from scratch if domain is far enough |

医疗CT扫描,空卫星图像和显微镜是遥远的领域.

### 冰的作用是什么原因?

图像网的功能, CNN 发现, 他们专注于自然图像的统计:边缘在特定的方向,纹理,对比模式,形状原始. 这些统计数据几乎在每个视觉领域都稳定, 这就是为什么在ImageNet上训练的模型,并通过CIFAR-10进行零射击评估,只使用新的线性头 (没有细节调整脊柱) 达到80%以上的精度. 头脑正在学习哪些已经学习的特征适用于这个任务.

### 歧视性学习率

早期层应比晚层慢训练,早期层应编码你想保存的通用特性,晚层则编码你需要经常移动的任务特定结构.

```
Typical recipe:

  stage 0 (stem + first group): lr = base_lr / 100    (mostly fixed)
  stage 1:                       lr = base_lr / 10
  stage 2:                       lr = base_lr / 3
  stage 3 (last backbone group): lr = base_lr
  head:                          lr = base_lr  (or slightly higher)
```

在 PyTorch 中,这是一个向优化器传递的参数组列表. 一个模型,五个学习速度,零额外代码.

### 批量规则问题

接的BN层`running_mean`其他`running_var`如果您的任务有不同的像素分布,不同的照明,不同的传感器,不同的颜色空间,这些缓冲器是错误的.

1. **Fine-tune with BN in train mode.**让BN更新其运行统计数据以及其他一切. 任务数据集是中型 (>=5k例) 的情况下,默认选择.
2. **Freeze BN in eval mode.**保持图像网统计数据,并仅训练重量.当你的数据集足够小,
3. **Replace BN with GroupNorm.**它们用于检测和细分背骨,其中每个GPU的批量尺寸很小.

错误的默默,将精度提高到5-15%.

### 头部设计

每个火视觉背骨都会发出一个默认的头,你取代:

```
backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)          # ResNet
backbone.classifier[1] = nn.Linear(..., num_classes)                    # EfficientNet, MobileNet
backbone.heads.head = nn.Linear(..., num_classes)                       # torchvision ViT
```

对于小数据集,通常只需要一个线性层.添加一个隐藏层 (线性 -> ReLU -> 放弃 -> 线性) 在任务分布远离脊柱的训练分布时有助.

### 层级 LR衰变

现代细调 (BEiT,DINOv2,ViT-B细调) 中使用的歧视性LR的更平滑版本.

```
lr_layer_k = base_lr * decay^(L - k)
```

化块的化量为0.75个,变压器块的L值为12个,`0.75^11 ≈ 0.04x`对于变压器的细节调节而言,

### 评估什么

转移学习运行需要两个数字,你不会在零零运行上追踪:

- **Pretrained-only accuracy**头部的精度,脊椎结.这是你的地板.
- **Fine-tuned accuracy** 完整训练后的模型.

如果微调不如预训练,你会有学习率或BN错误.

```figure
transfer-learning
```

## 建立它

### 步骤1:装载预训练的脊椎骨,检查它

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print(backbone)
print()
print("classifier head:", backbone.fc)
print("feature dim:", backbone.fc.in_features)
```

`ResNet18`具有四个阶段 (`layer1..layer4`) 加上一个干和一个`fc`每个火视觉分类的脊柱都有类似的结构.

### 冷所有东西,取代头部

```python
def make_feature_extractor(num_classes=10):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model = make_feature_extractor(num_classes=10)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"trainable: {trainable:>10,}")
print(f"frozen:    {frozen:>10,}")
```

只有`model.fc`脊柱是一个冷的特征提取器.

### 步骤3: 歧视性细调

建立一个阶段特定学习率的参数组的实用程序.

```python
def discriminative_param_groups(model, base_lr=1e-3, decay=0.3):
    stages = [
        ["conv1", "bn1"],
        ["layer1"],
        ["layer2"],
        ["layer3"],
        ["layer4"],
        ["fc"],
    ]
    groups = []
    for i, names in enumerate(stages):
        lr = base_lr * (decay ** (len(stages) - 1 - i))
        params = [p for n, p in model.named_parameters()
                  if any(n.startswith(k) for k in names)]
        if params:
            groups.append({"params": params, "lr": lr, "name": "_".join(names)})
    return groups

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 10)
for p in model.parameters():
    p.requires_grad = True

groups = discriminative_param_groups(model)
for g in groups:
    print(f"{g['name']:>10s}  lr={g['lr']:.2e}  params={sum(p.numel() for p in g['params']):>8,}")
```

`decay=0.3`意思是每一段火车的速度为下一个火车的30%`fc`得到了`base_lr`现在`layer4`得到了`0.3 * base_lr`现在`conv1`得到了`0.3^5 * base_lr ≈ 0.00243 * base_lr`极端的声音,经验上,它是有效的.

### 步骤4:批量规范处理

帮助BN结运行统计数据,而不会结其体重.

```python
def freeze_bn_stats(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
    return model
```

打电话后就打电话`model.train()`在每一个时代的开始.`model.train()`转换到训练模式,这只会转换BN层.

### 步骤5:最小的端到端细调循环

```python
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

def fine_tune(model, train_loader, val_loader, device, epochs=5, base_lr=1e-3, freeze_bn=False):
    model = model.to(device)
    groups = discriminative_param_groups(model, base_lr=base_lr)
    optimizer = SGD(groups, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        if freeze_bn:
            freeze_bn_stats(model)
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            tr_total += x.size(0)
            tr_correct += (logits.argmax(-1) == y).sum().item()
        scheduler.step()

        model.eval()
        va_total, va_correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(-1)
                va_total += x.size(0)
                va_correct += (pred == y).sum().item()
        print(f"epoch {epoch}  train {tr_loss/tr_total:.3f}/{tr_correct/tr_total:.3f}  "
              f"val {va_correct/va_total:.3f}")
    return model
```

五个时代,上述CIFAR-10的配方需要`ResNet18-IMAGENET1K_V1`只有头部就会达到86%的水平,而没有碰到脊椎.

### 步骤6:逐步解

时间表从结束到开始,每时段的一个阶段都会解.

```python
def progressive_unfreeze_schedule(model):
    stages = ["layer4", "layer3", "layer2", "layer1"]
    yielded = set()

    def start():
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    def unfreeze(epoch):
        if epoch < len(stages):
            name = stages[epoch]
            yielded.add(name)
            for n, p in model.named_parameters():
                if n.startswith(name):
                    p.requires_grad = True
            return name
        return None

    return start, unfreeze
```

电话`start()`在第一时代之前,`unfreeze(epoch)`任何一个时代的开始,每当训练可行的参数组变化时,重新构建优化器,否则,冷的参数仍然保留了混的缓存时刻.

## 用它

对于大多数真正的任务,`torchvision.models`超过3行,就足够了.当你遇到库默认无法解决的问题时,上面的更重的机器很重要.

```python
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, num_classes)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

其他两项生产级违约:

- `timm`船舶的视力背骨是预训练的~800个,具有一致的API (`timm.create_model("resnet50", pretrained=True, num_classes=10)`对于任何超越火动物园的细节,
- 对于变压器,`transformers.AutoModelForImageClassification.from_pretrained(name, num_labels=N)`给你 ViT / BEiT / DeiT 与文字模型相同的加载语义.

## 运送它

这一课产生了:

- `outputs/prompt-fine-tune-planner.md`一个提示,根据数据集大小,域距离和计算预算,选择功能提取与进步对结尾到结尾的细节调整.
- `outputs/skill-freeze-inspector.md`一个技能,在PyTorch模型中,报告哪些参数可以训练,哪些BatchNorm层在评估模式下,以及优化器是否实际上正在提供训练可用的参数.

## 运动

1. **(Easy)**列车`ResNet18`报告两种准确性,并列.解释哪个空隙告诉你功能转移良好,哪个告诉你它们没有.
2. **(Medium)**设置故意引入一个bug`base_lr = 1e-1`炼损失爆炸,然后通过应用炼损失恢复.`discriminative_param_groups`记录每一个阶段开始分离的 LR.
3. **(Hard)**拿一个医学成像数据集 (例如CheXpert-small,PatchCamelyon或HAM10000) 并比较三个模式: (a) ImageNet预训练的结脊椎+线性头; (b) ImageNet预训练的细调端到端; (c) 划分训练. 报告每个数据集的准确性和计算成本. 在哪个数据集尺寸上划分训练变得竞争力?

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Feature extraction | "Freeze and train head" | Backbone parameters frozen, only the new classifier head receives gradient |
| Fine-tuning | "Retrain end-to-end" | All parameters trainable, usually with much smaller LR than scratch training |
| Discriminative LR | "Smaller LR for early layers" | Optimizer parameter groups where early-stage LR is a fraction of late-stage LR |
| Layer-wise LR decay | "Smooth LR gradient" | Per-layer LR multiplied by decay^(L - k); common in transformer fine-tunes |
| Catastrophic forgetting | "The model lost ImageNet" | A too-high LR overwrites pretrained features before the new task signal is learnt |
| BN statistics drift | "Running mean is wrong" | BatchNorm running_mean/var computed on a different distribution than the current task, silently hurting accuracy |
| Linear probe | "Frozen backbone + linear head" | Evaluation of pretrained features — accuracy of the best linear classifier on top of the frozen representation |
| Catastrophic collapse | "Everything predicts one class" | Happens when fine-tuning with an LR high enough to destroy features before gradients from the head can stabilise |

## 进一步阅读

- [How transferable are features in deep neural networks? (Yosinski et al., 2014)](https://arxiv.org/abs/1411.1792)量化了跨层的特征可转移性的论文
- [Universal Language Model Fine-tuning (ULMFiT, Howard & Ruder, 2018)](https://arxiv.org/abs/1801.06146)原始的歧视性LR/渐进式解凍配方;想法直接转移到视觉
- [timm documentation](https://huggingface.co/docs/timm)现代视觉背骨的参考和它们所训练的精确细调默认
- [A Simple Framework for Linear-Probe Evaluation (Kornblith et al., 2019)](https://arxiv.org/abs/1805.08974)为什么线性探测精度很重要以及如何正确报告
