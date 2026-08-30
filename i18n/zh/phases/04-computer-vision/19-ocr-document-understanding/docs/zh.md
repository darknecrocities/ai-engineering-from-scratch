# 欧CR和文件理解

> 任何现代的OCR系统都会重新排序这些阶段或将它们合并.

**Type:** Learn + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06 (Detection), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~45 minutes

## 学习目标

- 追踪经典的OCR管道 (检测 ->识别 ->布局) 和现代端到端替代方案 (Donut,Qwen-VL-OCR)
- 实现连接器时间分类 (CTC) 损失对序列到序列OCR培训
- 使用PaddleOCR或EasyOCR进行无需培训的生产文件分析
- 区分OCR,布局分析和文档理解,并选择每项任务的正确工具

## 问题

图像中包含文本的图像遍布各处:收据,账单,身份证,扫描书,表格,白板,标志,截图.从中提取结构化数据不仅是字符,而且"这是总数"是应用视觉中最具价值的一个问题.

专业分为三个技能层次:

1. **OCR proper**转换像素成文字.
2. **Layout parsing**: 集团 OCR输出按区域 (标题,体格,表格,标题).
3. **Document understanding**:从布局中提取结构化字段 ("发票_总额 = 42.50美元").

每层都有经典和现代的方法, "我想要图像中的文字"和"我需要这个收件的总数"之间的差距比大多数团队所意识到的更大.

## 概念

### 经典管道

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **Text detection**产生每行或每字四边形.
- **Recognition**运行一个CNN+BiLSTM+CTC,以产生一个字符序列.
- **Layout**修复阅读顺序 (拉丁语的左到右,阿拉伯语,日本语的不同).

### 单段中CTC

基于CTC (Graves等,2006) 的功能,您可以在没有字符级调整的情况下训练.模型在每一步都输出了分布 (语音+空白);CTC损失将所有在重复和删除空白后减少到目标文本的调整进行边缘化.

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

根据CTC的规定,CRNN在2015年工作,并仍在2026年培训大多数生产OCR模型.

### 现代端到端型号

- **Donut** ViT编码器 + 文本解码器; 读取图像并直接发射JSON. 没有文本探测器,没有布局模块.
- **TrOCR** ViT+变压器解码器用于线级OCR.
- **Qwen-VL-OCR / InternVL**完整的视觉语言模型,为OCR任务进行精细调节;在2026年,对复杂文件提供最佳准确性.
- **PaddleOCR**经典的DB+CRNN管道在成熟的生产包中;仍然是开源工作马.

端到端模型需要更多的数据和计算,但避免多阶段管道的错误积累.

### 布局分析

对于结构文件,运行一个布局检测器 (LayoutLMv3, DocLayNet) 将每个区域标记为:标题,段落,图形,表格,脚注.读取顺序将成为"通过布局顺序区域进行反复,连接".

关于表格的使用**Key-Value extraction**模型 (可视化丰富的文件,LayoutLMv3可用于简单扫描).它们采集图像+检测的文本+位置,并预测结构化的键值对.

### 评估指标

- **Character Error Rate (CER)** 莱文施泰因距离/参考长度.较低更好.生产目标:在清洁扫描中< 2%.
- **Word Error Rate (WER)**字面上也是如此.
- **F1 on structured fields**关键价值任务;`{invoice_total: 42.50}`似乎是正确的.
- **Edit distance on JSON**用于端到端的文件分析;唐草纸引入了标准化的树编辑距离.

```figure
cv3-ctc-collapse
```

## 建立它

### 步骤1:CTC损失+贪的解码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss`利的解码器比光束搜索更简单,通常在 1% 的CER内.

### 步骤2:小的CRNN识别器

对于线路OCR的最低CNN+BiLSTM.

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

固定高度输入 (CNN最大积分高度为 1).宽度是CTC的时间尺寸.

### 步骤3:合成OCR

产生黑色到白色的数字字符串,

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

实际的OCR数据集添加字体,噪音,旋转,模糊和颜色.上面的管道是相同的.

### 步骤4:培训草图

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这次微不足道的合成数据上,损失应该从3到0.2以上降低.

## 用它

生产的三个途径:

- **PaddleOCR**成熟,快速,多语言. 一行使用: `paddleocr.PaddleOCR(lang="en").ocr(image_path)`现在,我们要去.
- **EasyOCR** 字符串原生,多语言,PyTorch脊柱.
- **Tesseract**经典;在模型难以完成时,仍然适用于旧扫描文件.

对于端到端文件分析,使用Donut或VLM:

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对于可重复结构的收据,发票和表格,请调整Donut.对于任意文件或有理由的OCR,如Qwen-VL-OCR这样的VLM是当前默认的.

## 运送它

这一课产生了:

- `outputs/prompt-ocr-stack-picker.md`一个提示,选择 Tesseract / PaddleOCR / Donut / VLM-OCR 给定的文档类型,语言和结构.
- `outputs/skill-ctc-decoder.md`从零开始编写贪和光束搜索的CTC解码器的技能,包括长度规范化.

## 运动

1. **(Easy)**训练TinyCRNN在500步的5位数字随机数字串上.
2. **(Medium)**报道 CER 德尔塔. 在哪些输入中,beam 搜索获胜?
3. **(Hard)**根据"项目名称,价格"对,使用PaddleOCR在20个收据,提取线条项的集合上,并计算F1与手标记的地面真相.

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| OCR | "Text from pixels" | Turning image regions into character sequences |
| CTC | "Alignment-free loss" | Loss that trains a sequence model without per-timestep labels; marginalises over alignments |
| CRNN | "Classic OCR model" | Conv feature extractor + BiLSTM + CTC; the 2015 baseline still used in production |
| Donut | "End-to-end OCR" | ViT encoder + text decoder; emits JSON directly from image |
| Layout parsing | "Find regions" | Detect and label Title/Table/Figure/Paragraph regions in a document |
| Reading order | "Text sequence" | Ordering of recognised regions into a sentence; trivial for Latin, non-trivial for mixed layouts |
| CER / WER | "Error rates" | Levenshtein distance / reference length at character or word granularity |
| VLM-OCR | "LLM that reads" | A vision-language model trained or prompted for OCR tasks; current SOTA on complex documents |

## 进一步阅读

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717)原始的CNN+RNN+CTC架构
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)原始的CTC纸;密集包装了算法的想法
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664)无OCR文件理解变压器
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)开源生产OCR堆
