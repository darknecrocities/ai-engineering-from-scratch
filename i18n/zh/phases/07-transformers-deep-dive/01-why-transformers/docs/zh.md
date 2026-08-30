# 为什么变革器  问题与RNN

> 转变器一次处理所有代币.这一次建筑投注改变了深度学习的每一个扩展曲线,2017年后.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 minutes

## 问题

在2017年之前,地球上每一个最先进的序列模型都是一个反复的神经网络.LST和GRU在半十年内获得了像网相当的翻译基准.它们是唯一的工具.

它们有三个致命的缺点. 序列计算意味着你不能沿时间轴平行化:`t+1`需要隐藏状态的代币`t`一个1024代币的序列意味着1 024个串行步骤在一个GPU上,可以每周期完成1,000,000个浮点操作.训练墙钟时间以线性方式与平行设计的硬件上的序列长度进行扩展.

消失的梯度意味着50代币的信息已经被压缩到50个非线性.关闭的复发单位 (LSTM,GRU) 缓和了压缩,但从来没有消除过它.长距离的依赖性"我去年夏天在飞机上读到的书..."经常失败.

固定的宽度隐藏状态意味着编码器在解码器看到任何东西之前将整个源序列挤入一个单个向量.源源是否是5个代币或500个,不管是什么,瓶是相同的形状.

2017年"注意力是你需要的"论文提出了一些根本的建议:完全放弃复发.让每个位置并行地关注其他位置.

结果在2026年之前占据所有模式的主导地位.语言 (GPT-5,Claude 4,Llama 4),视觉 (ViT,DINOv2,SAM 3),音频 (声),生物学 (AlphaFold 3),机器人 (RT-2).相同的区块,不同的输入.

## 概念

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Recurrence as a bottleneck.**电脑计算器`h_t = f(h_{t-1}, x_t)`每一步都取决于前一步.`h_5`在之前`h_4`在现代GPU上,有10,000多个并行芯,

**Attention as a broadcast.**自我注意力计算`output_i = sum_j(a_ij * v_j)`对于每一个对`(i, j)`整个N×N注意力矩阵都填充了一个批量的. 没有一步取决于另一个. GPU 很喜欢它.

**The speedup is not a constant.**它们的区别是`O(N)`系列深度和`O(1)`在实践中,变压器在N=512的匹配硬件上每时训练510倍快,并且随着序列长度的增加,间隙会扩大,直到你达到`O(N²)`记忆注意力墙 (后者被Flash Attention修复了见12课).

**What transformers cost.**关注记忆规模如`O(N²)`对于2K文本来说,很好.对于128K文本来说,你需要滑窗,ROPE外分,闪光注意力,或线性注意力变体.`O(N)`转换器将时间换取记忆,然后通过平行性获取时间.

**The inductive bias shift.**变压器认为没有什么每个对都是关注的候选人.这就是为什么变压器需要更多的数据来训练好,但一旦有了更大的规模.辛奇拉 (2022) 正式化了这一点:给出足够的代币,变压器总是击败一个相同参数数数的RNN.

```figure
rnn-vs-parallel
```

## 建立它

我们数量模拟核心瓶,让你感觉到笔记本电脑上的空隙.

### 步骤1:测量序列深度

看到`code/main.py`我们构建两个函数.一个编码一个序列作为一个连接链 (连续,像RNN一样).一个编码它作为一个平行减小 (像广播,像注意力).同样的数学,不同的依赖图.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

我们在连续上都能计时到10万个元素.RNN版本是O(N) 和单个CPU管道.即使在纯Python中,注意力式的减小也超过了1000,因为Python的`sum()`执行C语言,并且每步无解释器的代价.

### 步骤2:计算理论操作

两个算法都会增加N. 区别是 *依赖深度*:在下一个开始之前,必须进行多次操作. RNN深度 = N. 注意深度 = log(N) 通过树缩小,或1通过并行扫描.深度,而不是操作数量,决定了GPU时间.

### 步骤3:长序列的经验规模化

我们打印了一个时间表,使得O(N) 差距可见.在2026 Mac笔记本电脑上,1000个元素以下的序列太快以测量.100,000的序列显示了清洁的线性扫描.将其量化为16,384个代币变压器和12层LSTM等级,你会看到为什么训练墙钟在2016年是阻碍者.

## 用它

在2026年,还可以选择什么时候:

| Situation | Pick |
|-----------|------|
| Streaming inference, one token at a time, constant memory | RNN or state-space model (Mamba, RWKV) |
| Very long sequences (>1M tokens) where attention memory explodes | Linear attention, Mamba 2, Hyena |
| Edge device with no matmul accelerator | Depthwise-separable RNN still wins on FLOPs/watt |
| Anything else (training, batched inference, context up to 128K) | Transformer |

像Mamba这样的国家空间模型 (SSM) 基本上是具有结构化参数化的RNN,`O(N)`通过选择性扫描,他们恢复了变压器质量的90%通过更好的长文本扩展. 2026年,大多数边境实验室都将混合型SSM+变压器模型 (例如Jamba,Samba) 训练.

## 运送它

看到`outputs/skill-architecture-picker.md`由于长度,吞吐量和训练预算限制,技能选择一个新序列问题架构. 它应该始终拒绝推纯粹的RNN在训练运行超过1B代币的情况下,而不说明交易.

## 运动

1. **Easy.**接下来`rnn_style`其他`code/main.py`测量重复. 随着隐藏状态的维度,连续上层多少长?
2. **Medium.**通过纯 Python 实现平行前总数 (Hillis-Steele 扫描). 验证它产生与1024长度的连续扫描相同的数值输出.
3. **Hard.**按GPU上将注意力式降低调整到PyTorch. 时间同时扫描序列长度从64到65,536. 绘制并解释曲线形状.

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Recurrence | "RNNs are sequential" | Computation where step `t` depends on step `t-1`, forcing serial execution along the time axis. |
| Serial depth | "How deep the graph is" | Longest chain of dependent ops; bounds wall-clock even on infinite hardware. |
| Attention | "Let tokens look at each other" | Weighted sum `sum_j a_ij v_j` where `a_ij` comes from a similarity score between positions i and j. |
| Context window | "How much the model sees" | Number of positions an attention layer can take as input; quadratic memory cost scales here. |
| Inductive bias | "Assumptions baked into the architecture" | Prior about what the data looks like; CNNs assume translation invariance, RNNs assume recency. |
| State-space model | "RNN with algebra behind it" | Recurrence parameterized for parallel training via structured state-space matrices. |
| Quadratic bottleneck | "Why context costs so much" | Attention memory = `O(N²)` in sequence length; Flash Attention hides the constants, not the scaling. |

## 进一步阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762)这篇论文杀死了主流NLP的复发.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)在一个RNN上着注意力.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)原始的LSTM纸,为了记录.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)现代回复式答案变压器.
