# 自我监督视觉  SimCLR,DINO,MAE

> 标签是监督视觉的瓶. 自主监督预训练消除了它们:从100万个未标记的图像中学习视觉特征,

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 04 (Image Classification), Phase 4 Lesson 14 (ViT)
**Time:** ~75 minutes

## 学习目标

- 追踪三个主要的自我监督家庭:对比性 (SimCLR),教师-学生 (DINO),面具重建 (MAE) ,并说明每个家庭都优化什么
- 实施InfoNCE损失从零开始,并解释为什么512批成功,但32批失败
- 解释为什么MAE的75%隐藏比不任意,以及它与BERT的15%对文本的不同
- 使用DINOv2或MAE ImageNet检查站进行线性探测和零射击检索

## 问题

监督图像网有1300万个标记图像,这些图像的标记成本估计为1000万美元.医疗和工业数据集较小,标记成本更高.每个视觉团队都问:我们可以预训练低成本的无标记数据吗?

现代的自主监督 ViT 在LAION或JFT上训练,在调整时达到或超过监督的ImageNet精度.它还更好地转移到下游任务 (检测,细分,深度) 比监督的预训练.DINOv2 (Meta, 2023) 和MAE (Meta, 2022) 是可转移视觉功能的当前生产默认.

概念转变是,借口任务 模型训练要做的事情 不必是下游任务. 重要的是,它迫使模型学习有用的特性. 预测灰色图像的颜色,旋转图像,并要求模型分类旋转,掩盖补丁并重建它们都成功了. 对于此,三种方法是对比性学习,教师与学生蒸,

## 概念

### 三个家庭

```mermaid
flowchart LR
    A["Contrastive<br/>SimCLR, MoCo, CLIP"] --> AT["positive pairs<br/>(same image, 2 augs)<br/>pulled together,<br/>negatives pushed apart"]
    B["Teacher-student<br/>DINO, BYOL, iBOT"] --> BT["student predicts<br/>teacher's output;<br/>teacher is EMA of student"]
    C["Masked reconstruction<br/>MAE, BEiT, SimMIM"] --> CT["mask 75% of patches;<br/>reconstruct pixel or<br/>token targets"]

    style A fill:#dbeafe,stroke:#2563eb
    style B fill:#fef3c7,stroke:#d97706
    style C fill:#dcfce7,stroke:#16a34a
```

### 对比学习 (SimCLR)

通过同一个编码器加上投影头,减少说"这两个嵌入应该接近"和"这个嵌入应该远离每个其他图像的嵌入".

```
Loss for positive pair (z_i, z_j) among 2N views per batch:

   L_ij = -log( exp(sim(z_i, z_j) / tau) / sum_k in batch \ {i} exp(sim(z_i, z_k) / tau) )

sim = cosine similarity
tau = temperature (0.1 standard)
```

这就是InfoNCE损失.它需要每一个正数的许多负数,所以批量大小很重要. SimCLR需要512-8192.

### 教师-学生 (DINO)

学生和老师:两个网络具有相同的架构.老师是学生的体重的指数动平均值 (EMA).两个都看到图像的增强视图.学生的输出训练以匹配老师的没有明确的负面.

```
loss = CE( student_output(view_1),  teacher_output(view_2) )
     + CE( student_output(view_2),  teacher_output(view_1) )

teacher_weights = m * teacher_weights + (1 - m) * student_weights   (m ≈ 0.996)
```

为什么"预测常数"不崩:教师的输出是集中 (减减每维度平均值) 和磨损 (分为小温度).中心化防止一个维度主导;磨损防止输出崩到均.

根据DINOv2的扩展,DINOv2在142万个策划图像上进行了扩展.

### 面具重建 (MAE)

掩盖VIT输入的 75%的补丁.通过编码器传输只可见的25%.一个小的解码器在掩盖位置接收编码器的输出加上面具代币,并被训练重建掩盖补丁的像素.

```
Encoder:  visible 25% of patches -> features
Decoder:  features + mask tokens at masked positions -> reconstructed pixels
Loss:     MSE between reconstructed and original pixels on masked patches only
```

让MAE工作的关键设计选择:

- **75% mask ratio**高.强迫编码器学习语义特性;重建25%将是几乎无关紧要的 (邻近的像素是相对的,以至于CNN可以钉它).
- **Asymmetric encoder/decoder**大型ViT编码器只能看到可见的补丁;一个小型的解码器 (8层,512层) 处理重建.比天真BEiT快3倍的预训练.
- **Pixel-space reconstruction target**比BEiT的标记目标更简单,并且在ViT上更有效.

训练前,放弃解码器.

### 为什么75%而不是15%

密码是15%,MAE是75%. 信息密度是区别.

- 预测15%的代币仍然很难,因为每个隐藏的位置都有许多可行的完成.
- 图像补丁具有低值.一个未掩盖的邻居通常几乎准确地确定了掩盖补丁的像素.

75%是足够高的,以至于简单的空间外分不能解决任务;编码器必须代表图像内容.

### 线性探测评估

经过自我监督的预训,标准评估是**linear probe**通过将编码器结,将单个线性分类器放在图像网标签上.

- 升级率: 低于50%
- 子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子
- 果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果果
- 酸 (酸) 含量:

线性探测器是特征质量的纯度衡量;细调通常增加2-5个点,但也会产生头部重训效果.

```figure
data-augmentation
```

## 建立它

### 步骤1:双视图增强管道

```python
import torch
import torchvision.transforms as T

two_view_train = lambda: T.Compose([
    T.RandomResizedCrop(96, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
])


class TwoViewDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
        self.aug = two_view_train()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, _ = self.base[i]
        v1 = self.aug(img)
        v2 = self.aug(img)
        return v1, v2
```

每个__getitem__返回相同图像的两个增长视图;没有需要标签.

### 步骤2:InfoNCE损失

```python
import torch.nn.functional as F

def info_nce(z1, z2, tau=0.1):
    """
    z1, z2: (N, D) L2-normalised embeddings of paired views
    """
    N, D = z1.shape
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = z @ z.T / tau              # (2N, 2N)

    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)
```

在调用之前,将L2嵌入正常化. `tau=0.1`低的值使损失更为明显,需要更多的负值.

### 步骤3: 智力检查 InfoNCE

```python
z1 = F.normalize(torch.randn(16, 32), dim=-1)
z2 = z1.clone()
loss_same = info_nce(z1, z2, tau=0.1).item()
z2_random = F.normalize(torch.randn(16, 32), dim=-1)
loss_random = info_nce(z1, z2_random, tau=0.1).item()
print(f"InfoNCE with identical pairs:  {loss_same:.3f}")
print(f"InfoNCE with random pairs:     {loss_random:.3f}")
```

偶数对应输出低损失 (对于大型批量和冷温度接近0).随机对应应输出 log(2N-1) = ~log(31) = ~3.4 与16对批量.

### 步骤4:MAE风格的罩

```python
def random_mask_indices(num_patches, mask_ratio=0.75, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_keep = int(num_patches * (1 - mask_ratio))
    perm = torch.randperm(num_patches, generator=g)
    visible = perm[:n_keep]
    masked = perm[n_keep:]
    return visible.sort().values, masked.sort().values


num_patches = 196
visible, masked = random_mask_indices(num_patches, mask_ratio=0.75)
print(f"visible: {len(visible)} / {num_patches}")
print(f"masked:  {len(masked)} / {num_patches}")
```

实际的MAE实现将这些进行批量并保持每样品面具.

## 用它

诺维2是2026年的生产标准:

```python
import torch
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")
model.eval()

# Per-image embeddings for zero-shot retrieval
with torch.no_grad():
    inputs = processor(images=[pil_image], return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0]  # CLS token
```

结果的768dim嵌入式是现代图像检索,密集通信和零射击传输管道的脊柱.下游任务的细节调整很少需要超过线性头部.

对于图像文本嵌入,SigLIP或OpenCLIP是相当的;对于MAE风格的细调,`timm`通过每一个检查点.

## 运送它

这一课产生了:

- `outputs/prompt-ssl-pretraining-picker.md`一个提示,根据数据集的大小,计算和下游任务,选择 SimCLR / MAE / DINOv2.
- `outputs/skill-linear-probe-runner.md`写出任何结编码器+标记数据集的线性探测评估的技能.

## 运动

1. **(Easy)**检查如果您降低了适配嵌入式温度时的 InfoNCE 损失降低了,并且如果您降低了随机嵌入式温度时会增加.`tau in [0.05, 0.1, 0.2, 0.5]`损失和损失
2. **(Medium)**通过DINO式的中心缓冲器, 显示学生在几个时代内会崩到一个恒定向量.
3. **(Hard)**训练MAE在CIFAR-100上使用TinyUNet从10课作为脊柱.报告线性探测器的准确性在10,50和200个时代.显示MAE训练的线性探测器在同一1000图片子组上击败了从零开始监督的线性探测器.

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Self-supervised | "Label-free" | A pretext task that produces useful representations from unlabelled data |
| Pretext task | "The fake task" | The objective used during SSL (reconstruct patches, match views); discarded after pretraining |
| Linear probe | "Frozen encoder + linear head" | Standard SSL evaluation: train only a linear classifier on top of frozen features |
| InfoNCE | "Contrastive loss" | softmax over cosine similarities; positive pair is the target class, all others are negatives |
| EMA teacher | "Moving-average teacher" | Teacher whose weights are an exponential moving average of the student's; used by BYOL, MoCo, DINO |
| Mask ratio | "% of patches hidden" | Fraction of patches masked during MAE; 75% for vision, 15% for text |
| Representation collapse | "Constant output" | SSL failure where the encoder outputs a constant vector for all inputs; prevented by centring, sharpening, or negatives |
| DINOv2 | "Production SSL backbone" | Meta's 2023 self-supervised ViT; strongest general-purpose image features in 2026 |

## 进一步阅读

- [SimCLR (Chen et al., 2020)](https://arxiv.org/abs/2002.05709)对比性学习参考
- [DINO (Caron et al., 2021)](https://arxiv.org/abs/2104.14294)               
- [MAE (He et al., 2022)](https://arxiv.org/abs/2111.06377)面具自动编码器预训练VIT
- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193)扩大自主监督的ViT到生产特征
