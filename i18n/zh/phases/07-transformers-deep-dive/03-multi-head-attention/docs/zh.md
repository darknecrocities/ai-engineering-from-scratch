# 多头注意力

> 一个注意力头一次学会一个关系. 八个头学习八个. 头是自由的. 拿更多的.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 minutes

## 问题

一个自我注意力头计算一个注意力矩阵.那个矩阵捕捉到一种关系,通常是减少任何训练信号的损失.如果你的数据有主体verb协议,共参考,长距离的演讲和语法分断,都在一起,一个头将它们成一个软最大分布,失去了一半的信号.

2017年瓦斯瓦尼论文的修正:并行运行几个注意力函数,每个都具有自己的Q,K,V投影,并连接输出.每个头部都在更小的子空间中运行.`d_model / n_heads`总参数保持相同,表达功率上升.

单个论点是关于*多少头,以及键和值是否共享投影 (组列查询注意力,多查询注意力,多头潜伏注意力).

## 概念

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Split.**接下来`X`形状`(N, d_model)`项目到Q,K,V,每个形状`(N, d_model)`改装到`(N, n_heads, d_head)`在哪里`d_head = d_model / n_heads`转移到`(n_heads, N, d_head)`现在,我们要去.

**Attend in parallel.**运行一个级别的点产品注意力在每个头脑.`(N, d_head)`头部在嵌入器的不同子空间上运行,并且在注意力计算过程中永远不会说话.

**Concatenate and project.**堆头回去`(N, d_model)`并且乘以学习的输出矩阵`W_o`形状`(d_model, d_model)`现在,我们要去.`W_o`这就是头脑的混合.

**Why it works.**每个头都可以专业化,而不与其他对象预算竞争.2019年2024年的探测研究显示了不同的头部角色:位置头,参加前一个代币的头,复制头,命名实体头,诱导头 (这是内文学习的基础).

**The 2026 lineage of variations:**

| Variant | Q heads | K/V heads | Used by |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

由于它削减了KV缓存存储量`N/G`通过将K/V压缩到隐藏空间,然后在计算时间上投影回来,

```figure
multihead-split
```

## 建立它

### 步骤1:从我们已经有的单头关注中分开头

拿起`SelfAttention`子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子`code/main.py`对于一个无数的实现,逻辑是:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一个重塑,一个转换,没有循环. 这正是PyTorch在做的事情.`nn.MultiheadAttention`现在,我们要去.

### 步骤2:按标点进行产品关注

每个头都得到了自己的Q,K,V.

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真正的硬件上`Qh @ Kh.transpose(...)`是一个`bmm` GPU 看到一个单批的形状.`(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`增加头子是免费的.

### 步骤3: 组列查询注意力变体

只有关键和值预测变化.`n_heads`组; K 和 V 得到`n_kv_heads < n_heads`组,并重复一致:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

根据推论,这节省了记忆力,因为只有`n_kv_heads`没有在KV缓存中存活的副本`n_heads`拉马370B使用64个查询头,8个KV头,一个8倍缓存缩小器.

### 步骤4:检查每个头脑学到的东西

按一个短句子,用4个头来运行MHA.`(N, N)`随机初始化,也就是部分信号,部分旋转对称性.

## 用它

在 PyTorch 中,单行版本:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

根据 PyTorch 2.5+ 的 GQA:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**How many heads?**2026年生产模型的指纹规则:

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head`几乎总是降落在64或128位.这是一个头能"看到"多少的单位.`sqrt(d_head)`您将失去"许多小专家"的福利.

## 运送它

看到`outputs/skill-mha-configurator.md`技能建议对新变压器进行头数, kv头数和投影策略,以设置参数预算,序列长度和部署目标.

## 运动

1. **Easy.**取出MHA的`code/main.py`改变`n_heads`从1到16`d_model=64`更多头脑有助于,高,或伤害?
2. **Medium.**实现MQA (所有查询头都共享一个KV头).测量参数数量多少下降与全MHA.计算在推断时KV缓存尺寸缩小多少为N=2048.
3. **Hard.**执行多头潜伏注意的小版本:压缩K,V到一个级别`r`隐藏在KV缓存中,在注意时解压.`r`缓存内存在满满MHA的1/8以下,而质量在验证后保持在1位内?

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Head | "A single attention circuit" | One Q/K/V projection of dimension `d_head = d_model / n_heads` with its own attention matrix. |
| d_head | "Head dimension" | Per-head hidden width; almost always 64 or 128 in production. |
| Split / combine | "Reshape tricks" | `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose around attention. |
| W_o | "Output projection" | `(d_model, d_model)` matrix applied after concatenating heads; where heads mix. |
| MQA | "One KV head" | Multi-Query Attention: single shared K/V projection. Smallest KV cache, some quality loss. |
| GQA | "The default since Llama 2" | Grouped-Query Attention with `n_kv_heads < n_heads`; repeats to match Q. |
| MLA | "DeepSeek's trick" | Multi-head Latent Attention: K,V compressed to low-rank latent, decompressed at attend time. |
| Induction head | "The circuit behind in-context learning" | A pair of heads that detect previous occurrences and copy what followed them. |

## 进一步阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762)原始多头型规格.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)MQA论文
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)如何在培训后将MHA转换为GQA.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) MLA 和为什么它超过MHA/GQA在缓存内存.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)机械化看看头部实际上做什么.
