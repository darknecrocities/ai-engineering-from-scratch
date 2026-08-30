# 位置编码 鼻状,ROPE,ALiBi

> 关注是变量不变的. "猫坐在床上"和"猫坐在床上"产生相同的输出,没有位置信号.三个算法每一个通过不同的投注来解决它.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 minutes

## 问题

关注点产品的规模是顺序盲目的.`softmax(Q K^T / √d) V`通过对式相似性计算.`X`没有什么在注意力中关心位置.

对于语言,代码,音频,视频,任何有序的东西都意味着,

解决方案是以某种方式注入位置.

1. **Absolute sinusoidal**加入 `sin/cos`简单,无需学习,不适合超越训练的长度.
2. **RoPE — Rotary Position Embeddings**旋转Q和K向量以与位置相对的角度. 编码直在点数中 *相对*位置. 2026年占主导地位.
3. **ALiBi — Attention with Linear Biases**根据距离,对注意力分数添加一个每头线性罚款. 极好的长度抽出.

截至2026年,基本上每个边境开放模型都使用RoPE:Llama 2/3/4,Qwen 2/3,Mistral,Mixtral,DeepSeek-V3,Kimi.少数长文本模型使用ALiBi或其现代变体.绝对突形是历史性的.

## 概念

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### 绝对的阴影状

预先计算一个固定矩阵`PE`形状`(max_len, d_model)`其他:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

那么`X' = X + PE[:N]`模型学会从相模式中读取位置. 失败超出了`max_len`没有什么告诉模型在2048位置发生什么,当它只看到位置02047时.

### 子

转换Q和K向量 (不是嵌入式).`(2i, 2i+1)`其他:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

按位置的键进行相同的旋转`pos_k`点的产品`q'_m · k'_n`成为一个函数`(m - n)`只有一个人.**the attention score depends only on the relative distance**虽然旋转是绝对位置的.

扩展ROPE: `base`通过这种方式,Llama 3从8K到128K的环境扩展.

### 鱼

忽略嵌入技巧. 偏见的注意力直接得分:

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

在哪里?`m_h`是一个特定的头斜率 (例如 `1 / 2^(8·h/H)`报纸显示,长度抽出比较像座形状,与RoPE在原始训练长度上相匹配.

### 2026年要选择什么

| Variant | Extrapolation | Training cost | Used by |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | poor | free | original transformer, early BERT |
| Learned absolute | none | tiny | GPT-2, GPT-3 |
| RoPE | good with scaling | free | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | excellent | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | excellent | free | BLOOM, MPT, Baichuan |

由于它没有改变建筑,它编码相对位置,它引起了人们的注意.`base`超参数为长文本细调提供了清洁的按.

```figure
rope-explorer
```

## 建立它

### 步骤1:突状编码

看到`code/main.py`计算四行:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一层注意力之前,将此添加到嵌入矩阵中.

### 步骤2:对Q,K应用的RoPE

罗佩在Q和K上在现场运行.

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键:在位置上将相同的函数应用到Q`m`,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,.`n`他们的点产品接收了一个`cos((m-n)·θ_i)`关注可以免费学习相对位置.

### 步骤3:ALiBi斜率和偏差

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

加入`bias[h]`对于`(seq_len, seq_len)`注意力分数矩阵`h`接着是软max.

### 步骤4:检查RoPE的相对距离特性

选择两个随机向量`a, b`旋转`(pos_a, pos_b)`然后,通过`(pos_a + k, pos_b + k)`两种点产品必须在浮点误差内匹配.该属性是RoPE的整点,它与绝对的偏移不变,只有相对差距才有意义.

## 用它

托尔奇2.5+ 运输了RoPE公用品`torch.nn.functional`生产代码的大部分使用`flash_attn`或`xformers`在注意内核内应用RoPE.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Long-context tricks in 2026:**

- **NTK-aware interpolation.**重新扩展`base`为了`base * (scale_factor)^(d/(d-2))`在4K到16K+的时间内.
- **YaRN.**智能的插射,可以在长度的环境中保持注意力缩.
- **LongRoPE.**微软的2024方法,使用进化搜索来选择每维度尺度的因素.
- **Position interpolation + fine-tuning.**只是缩小位置,扩展因素, 调整15B代币.

## 运送它

看到`outputs/skill-positional-encoding-picker.md`技能选择一个新模型编码策略,考虑到目标背景长度,外分需求和培训预算.

## 运动

1. **Easy.**绘制一个突形图`PE`作为热图的矩阵`max_len=512, d=128`确认"随着尺寸指数的增长,条纹变得更宽".
2. **Medium.**运行NTK知性ROPE扩展. 训练一个小的LM在长度256的序列,然后测试在长度1024的与没有扩展. 测量困难.
3. **Hard.**运用 ALiBi 和 RoPE 在同一注意力模块中. 训练一个4层变压器在长度512的序列复制任务上. 在测试时,将其推移到2048年. 进行降解比较.

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Positional encoding | "Tells attention about order" | Any signal added to embeddings or attention that encodes position. |
| Sinusoidal | "The original one" | `sin/cos` at geometric frequencies added to embeddings; doesn't extrapolate. |
| RoPE | "Rotary embeddings" | Rotate Q, K by position-dependent angle; dot product encodes relative distance. |
| ALiBi | "Linear bias trick" | Add `-m·\|i-j\|` to attention scores; no embedding needed, great extrapolation. |
| base | "RoPE's knob" | The frequency scaler in RoPE; increase to extend context at inference. |
| NTK-aware | "A RoPE scaling trick" | Rescale `base` so high-frequency dims aren't squeezed when context expands. |
| YaRN | "The fancy one" | Per-dimension interpolation+extrapolation that preserves attention entropy. |
| Extrapolation | "Works beyond trained length" | Can the position scheme serve correct output past `max_len` seen in training? |

## 进一步阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762)原始的鼻状.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)    纸
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)  
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)最新的ROPE扩展.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)Meta的Llama2长文本论文.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) Phi-3-Long使用的微软方法,并引用在使用它部分.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py)每一个RoPE扩展方案的生产级实施 (默认,线性,动态,YaRN,LongRoPE,Llama-3).
