# 完整变压器 编码器+解码器

> 其他一切,剩余物,正常化,向前传递,交叉注意,

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## 问题

单一注意力层是一个特征提取器,而不是模型.每层一个不够语言容量.你需要没有正确的管道.

2017年瓦斯瓦尼论文包装了六项设计决定,将一个注意层转化为可堆叠的块.自编码器 (BERT),解码器 (GPT),编码器-解码器 (T5) 以来,每个变压器都继承了相同的骨架.2026年,这些块已经被精炼 (RMSNorm,SwiGLU,预规范,RoPE),但骨架是相同的.

接下来的课程将其专业化为编码器,07为解码器,08为编码器-解码器.

## 概念

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### 六个小块

1. **Embedding + positional signal.**标志 → 矢量. 通过 RoPE (现代) 或突形 (经典) 注射的位置.
2. **Self-attention.**每个位置都在互相关联,隐藏在解码器中.
3. **Feed-forward network (FFN).**位置的两层MLP: `W_2 · activation(W_1 · x)`预设扩展率为4×.
4. **Residual connection.** `x + sublayer(x)`没有它,梯度消失了6层.
5. **Layer normalization.** `LayerNorm`或`RMSNorm`稳定了剩余的流量.
6. **Cross-attention (decoder only).**查询来自解码器,密钥和值来自编码器输出.

观察一个向量通过一个块流动:注意力在各个位置之间混合,残余物将其运行向前,FFN将其转化,并且规范保持流动稳定.

```figure
transformer-block
```

### 编码器块 (BERT,T5编码器使用)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

编码器是双向的,没有掩盖,所有位置都能看到所有位置.

### 解码器块 (GPT,T5解码器使用)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

解码器每块有三个子层.中间的  交叉注意力是信息从编码器流向解码器的唯一地方.在纯解码器架构 (GPT) 中,交叉注意力被遗漏,你只掩盖了自我注意力 + FFN.

### 标准前与标准后

原始纸:`x + sublayer(LN(x))`其他`LN(x + sublayer(x))`在没有仔细的加热的情况下,更难进行深度训练.`LN`之前的子层) 是2026年默认的:Llama,Qwen,GPT-3+,Mistral都使用它.

### 2026年现代化区块

瓦斯瓦尼2017年出货LayerNorm + ReLU.现代堆取代了这两者.

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices, total params match) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

格鲁 (SwiGLU) 则可以通过格鲁 (SwiGLU) 进行格鲁 (SwiGLU) 进行格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (SwiGLU) 格鲁 (S) 格鲁) 格鲁 (Sw) 格鲁 (Sw) 格鲁) 格鲁 (Sw) 格鲁 (Sw) 格鲁 (Sw) 格鲁) 格鲁 (Sw)`Swish(W1 x) ⊙ W3 x`) 持续超过Llama,PaLM和Qwen论文中的ReLU/GELU FFN的0.5分点.

### 参数数量

为了一个街区`d_model = d`及FFN扩张`r`其他:

- 鱼类`4 · d²`预测量
- 转移: 转移:`3 · d · (r · d)`≈ ≈`3rd²`
- 标准: 无关可视

在`d = 4096, r = 2.6, layers = 32`(大约是Llama 3 8B),总数: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`发行匹配数量.

## 建立它

### 步骤1:建筑块

通过使用小小的`Matrix`课03 (为了独立,复制到本文件):

- `layer_norm(x, eps=1e-5)`减去平均值,分为 std.
- `rms_norm(x, eps=1e-6)` 分为RMS. 没有中减.
- `gelu(x)`其他`silu(x) * W3 x`现在我们要去做什么?
- `ffn_swiglu(x, W1, W2, W3)`现在,我们要去.
- `encoder_block(x, params)`其他`decoder_block(x, enc_out, params)`现在,我们要去.

看到`code/main.py`为了完整的电线.

### 步骤2:将2层编码器和2层解码器连接

输出出码器输入每个解码器交叉注意力. 在输出投影之前添加最后的LN.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 步骤3:在玩具示例上前行

输出源源和目标源源为5个代币.`(5, vocab)`没有培训,这堂课是关于建筑,不是损失.

### 步骤 4: RMSNorm + SwiGLU 中交换

通过RMSNorm和SwiGLU取代LayerNorm和ReLU-FFN. 确认形状仍然匹配.这是2026年现代化,一个函数的替代.

## 用它

 PyTorch/TF 参考实施方案: `nn.TransformerEncoderLayer`现在`nn.TransformerDecoderLayer`但大部分2026生产代码都用了自己的块,因为:

- 闪光注意力是通过内部注意力而不是通过`nn.MultiheadAttention`现在,我们要去.
- 没有任何关于GQA/MLA的参考.
- ,RMSNorm,SwiGLU不是PyTorch默认的.

`transformers`您应该阅读的清洁参考区块: `modeling_llama.py`只有2026个解码器的区块. 它大约500行,值得一遍走过.

**Encoder vs decoder vs encoder-decoder — when to pick:**

| Need | Pick | Example |
|------|------|---------|
| Classification, embeddings, QA over text | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

仅使用解码器才能获胜,因为它规模最清洁,处理理解和生成. 输入中具有清晰的"源序列"身份 (翻译,语音识别,结构化任务) 时,解码器仍然是最好的.

## 运送它

看到`outputs/skill-transformer-block-reviewer.md`技能检查了对2026年默认的变压器块的新实施,并标记了缺失的零件 (前标准,RoPE,RMSNorm,GQA,FFN扩展比).

## 运动

1. **Easy.**计算您的编码_区块中的参数`d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`通过实现区块和使用`sum(p.numel() for p in block.parameters())`现在,我们要去.
2. **Medium.**切换从后规范到前规范. 启动两者,并在随机输入中测量12层堆叠后的激活规范. 后规范的激活应爆炸; 前规则应保持局限.
3. **Hard.**实现4层编码解码器在玩具复制任务 (复制 `x`换 RMSNorm + SwiGLU + RoPE  损失下降吗?

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Block | "One transformer layer" | Stack of norm + attention + norm + FFN, wrapped in residual connections. |
| Residual | "Skip connection" | `x + f(x)` output; enables gradient flow through deep stacks. |
| Pre-norm | "Normalize before, not after" | Modern: `x + sublayer(LN(x))`. Trains deeper without warmup gymnastics. |
| RMSNorm | "LayerNorm without the mean" | Divide by RMS; one less op, same empirical stability. |
| SwiGLU | "The FFN everyone switched to" | `Swish(W1 x) ⊙ W3 x → W2`. Beats ReLU/GELU on LM ppl. |
| Cross-attention | "How the decoder sees the encoder" | MHA with Q from decoder, K/V from encoder outputs. |
| FFN expansion | "How wide the middle MLP is" | Ratio of hidden-size to d_model, usually 4 (LayerNorm) or 2.6 (SwiGLU). |
| Bias-free | "Drop the +b terms" | Modern stacks omit biases in linear layers; slight ppl improvement, smaller model. |

## 进一步阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762)原始的块规格.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)为什么前规则比后规则更深.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)        
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)SwiGLU的报纸.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)可信 2026 单独使用解码器的区块.
