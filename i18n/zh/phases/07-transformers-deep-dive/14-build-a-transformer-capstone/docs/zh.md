# 创建一个变压器从零开始  石头

> 十三课,一个模型,没有快捷方式.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 minutes

## 问题

你已经读过每篇论文,你已经实现了注意力,多头分区,位置编码,编码和解码区块,BERT和GPT损失,MoE,KV缓存.现在让它们一起完成一个真正的任务.

终点:训练一个小的单独解码器变压器端到端进行角色级语言建模任务.它读出莎士比亚.它生成了新的莎士比亚.它足够小,以在10分钟内在笔记本电脑上训练.它足够正确,更大的数据集和更长的训练交换给你一个真正的LM.

卡帕蒂的2023年纳米GPT教程是每个学生至少写一次的参考实现.我们把形状抬起,重新整理了我们所覆盖的内容.

## 概念

![Transformer-from-scratch block diagram](../assets/capstone.svg)

建筑,注释:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 我们运送的东西

- `GPTConfig`一个配置所有超参数的地方.
- `MultiHeadAttention`因果性,批量,可选的闪光式路径 (PyTorch的) `scaled_dot_product_attention`)
- `SwiGLUFFN`现代的FFN.
- `Block`预规,残留包装注意力+FFN.
- `GPT`嵌入式,堆积式块,LM头,生成().
- 训练循环与亚当W,数 LR,梯度剪切.
- 对于莎士比亚的文字.

### 我们不送什么东西

- 在课程04中概念上实现了 RoPE.在这里我们使用学习的位置嵌入式来简单化.
- 随着生成的过程中,每个生成步骤重新计算注意力.慢慢,但更简单.练习要求你添加一个KV缓存.
-  PyTorch 2.0+ 自动发送,如果输入相匹配,我们使用`F.scaled_dot_product_attention`现在,我们要去.
- 单个FFN每块.你在第11课中看过MoE.

### 目标指标

在MacM2笔记本电脑上,一个四层,四头,d_model=128GPT训练了2000步`tinyshakespeare.txt`其他:

- 训练损失约6分钟内从4.2 (随机) 降至1.5分钟.
- 采样产品看起来像莎士比亚:古老的词,线条断裂,像"ROMEO:"这样的名字出现.
- 值损失 (最后10%的文本被保留) 密切跟踪训练损失;在这个规模/预算上没有过度适应.

```figure
n5-block-stack
```

## 建立它

这堂课使用PyTorch.`torch`现在,我们可以在这个地方做什么?`code/main.py`剧本处理:

- 下载`tinyshakespeare.txt`如果没有 (或阅读本地副本).
- 字节级卡标记器.
- 列车/车间分为90/10.
- 训练循环,在支持的硬件上自动播放 bf16.
- 训练结束后,

### 步骤1:数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

只有65个字符,有很小的词汇,可以用4字节的词汇,没有BPE,没有标记器戏剧.

### 步骤2:模型

看到`code/main.py`区块是课05 预规,RMSNorm,SwiGLU,因果MHA的教科书.

### 步骤3:训练循环

随机取长度-256个标志窗户,向前,转变为一个,反向,亚当W步骤,记录,重复.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 步骤4:样本

给出提示,反复转发,从顶部p登录中取样,添加,然后继续.

### 步骤5:读取输出

在2000步后:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

没有莎士比亚,但莎士比亚形状, 赢得了800万个参数和6分钟的笔记本电脑.

## 用它

这块顶石是参考架构,有三个扩展,

1. **Swap the tokenizer.**使用BPE (例如:`tiktoken.get_encoding("cl100k_base")`字母尺寸从65升至5万. 模型容量需要扩大,以补偿.
2. **Train on a bigger corpus.**使用`OpenWebText`或`fineweb-edu`单个A100上的10B代币需要24小时才能实现125M的GPT.
3. **Add RoPE + KV cache + Flash Attention.**下面的练习将你通过每一个.

这最终成为一个125M参数GPT,产生流利的英语.不是一个边界模型.但同样的代码路径只是更大是卡帕蒂,埃勒艾伊和艾伦研究所在2026年用来训练研究检查站.

## 运送它

看到`outputs/skill-transformer-review.md`技能检查了所有13个上课的变压器从零开始实施的正确性.

## 运动

1. **Easy.**跑步`code/main.py`检查您训练有素的模型最后一步验证损失低于2.0. 改变`max_steps`                                                                                                                                                                                                                                                              
2. **Medium.**取代学习的位置嵌入式用RoPE. 应用转换到Q和K内部`MultiHeadAttention`列车和验证的值损失至少同样低.
3. **Medium.**通过测试,在测试循环中实现KV缓存. 生成500个代币,无论是没有缓存. 笔记本电脑的墙钟应该提高520x.
4. **Hard.**加入第二个头到模型中,预测下一个加一个代币 (MTP 从DeepSeek-V3的多代币预测).
5. **Hard.**换取每块单个FFN用4个专家MoE.路由器+顶-2路由器.看看在匹配的活跃参数时的值损失变化.

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy's tutorial repo" | Minimal decoder-only transformer training code, ~300 LOC; the canonical reference. |
| tinyshakespeare | "The standard toy corpus" | ~1.1 MB of text; every character-LM tutorial since 2015 uses it. |
| Tied embeddings | "Share input/output matrix" | LM head weight = transpose of token embedding matrix; saves parameters, improves quality. |
| bf16 autocast | "Training precision trick" | Run forward/back in bf16, keep optimizer state in fp32; standard since 2021. |
| Gradient clipping | "Stops spikes" | Cap global grad norm at 1.0; prevents training blowups. |
| Cosine LR schedule | "The 2020+ default" | LR ramps up linearly (warmup) then decays cosine-shaped to 10% of peak. |
| MFU | "Model FLOP Utilization" | Achieved FLOPs / theoretical peak; 40% dense, 30% MoE is strong in 2026. |
| Val loss | "Held-out loss" | Cross-entropy on data the model never saw; overfit detector. |

## 进一步阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/)经典的注释实施.
