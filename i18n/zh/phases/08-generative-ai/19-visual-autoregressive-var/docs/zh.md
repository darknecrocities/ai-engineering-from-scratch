# 视觉自动降低模型 (VAR):下一个规模预测

> 扩散模型以时间的反复样本 (指标步骤). VAR样本以尺度的反复样本 预测1x1代币,然后2x2,然后4x4,直到最终分辨率,每个尺度都与之前的定制. 2024 论文显示VAR与图像生成的GPT式扩展法相匹配,并以相同的计算预算击败DiT.这堂课构建了核心机制.

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 minutes

## 问题

由于它可以预测的规模,所以自动降级的代代码占据了语言建模的主导地位:更多的计算,更多的参数,较低的困惑,更好的输出.在2024年前,图像生成有两个主要的AR尝试:PixelCNN (像素-比-像素) 和DALL-E 1 / Parti / MuseGAN (VQ-VAE代码上的代码-比-代码).

两者都患上了生成顺序问题.像素和代币都在2D格格格中排列,但AR模型必须在1D拉斯特顺序中访问它们.早期角像素不知道图像最终会变成什么.生成质量比GPT-on-text更差,并从未达到匹配计算时的扩散模型质量.

维亚解决了生成顺序问题,通过改变正在生成的东西.而不是在空间中预测图像代币一个接一个,维亚预测一个整整整的图像在增加的分辨率.步骤1:预测一个1x1代币 (整体图像"总结").步骤2:预测一个2x2代币格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格格

每个尺度都在自己的尺度内照顾所有以前的尺度 (因果而言是"尺度顺序") 和平行.

## 概念

### 视频显示器

需要一个**multi-scale discrete tokenizer**对于图像x,它产生了一系列逐渐高分辨率的代币网:

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个z_k 使用相同的代码簿 (典型尺寸4096-16384). 每个尺度的代码化不独立.

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是一个**residual VQ**解码器取出所有规模嵌入式的总和,生成图像.

许多规模的VQ代码符号器一次训练 (如VQGAN) 然后冷.所有生成工作都由上部的自动降低模型完成.

### 下一个规模预测

生成模型是一个变压器,它可以从所有以前的尺度中看到代币,并预测下一个尺度的代币.

输入序列结构:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置嵌入式将在尺度内编码规模指数和空间位置. 注意是因果的规模顺序:在尺度 k,位置 (i, j) 可以关注所有在尺度 1..k的代币和在尺度 k本身的代币,在任何在尺度内顺序使用的之前 (VAR使用固定位置注意,没有在尺度内因果关系,在尺度内所有的位置都是平行预测的).

训练损失:在每个级别 k,预测给所有前级代币的代币 z_k. 离散的VQ代码上的交叉缩损失.除"序列"之外,与GPT相同的结构现在是规模结构化的.

### 世代

在推断时:
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

对于K=10尺度,生成是10个变压器前进传递.每个传递均以平行方式产生其整个尺度.对于256x256图像,这大约是10个传递与DiT的28-50相比.

### 为什么下一个规模胜过下一个代币

结构性三大胜利:
1. **Coarse-to-fine aligns with natural image statistics.**人类视觉感知和图像数据集都表现出依赖规模的规律性:低频结构是稳定的和可预测的;高频细节是依赖于低频内容的.下一个规模预测利用这一点.
2. **Parallel generation within scale.**与GPT式代币AR不同,VAR在一个步骤中生成所有代币.有效生成长度是日志规模而不是线性.
3. **No generation order bias.**在规模k的代币看到所有规模k-1;没有"左"或"上"偏见,迫使早期代币在晚上文本之前承诺.

### 规模定位法

等 显示VAR在 ImageNet 上遵循FID的权力法缩小曲线,就像GPT对困难一样. 双倍参数或计算可靠地减半错误. 这也是第一个像生成模型, 像语言模型一样清洁地表现出这种扩展行为. 结果是,VAR尺度预测是可以从计算中预测的,而不是每一个建筑的经验猜测.

### 传播与传播的关系

维AR和扩散都具有相同的数据压缩故事:两者都将生成问题分解为一系列更容易的子问题.

- 扩散:逐渐增加噪音,学习撤销一步.
- 逐渐增加分辨率,学习预测下一个尺度.

它们通过问题是不同的轴.两者都产生可处理的条件分布.经验上,VAR在推断上更快 (更少的通过,所有都在尺度内平行) 并匹配或超过了DIT在类条件的ImageNet上.文本条件的VAR (VARclip,HART) 是一个活跃的研究方向.

```figure
gx-var-next-scale
```

## 建立它

在`code/main.py`你会:
1. 建造一个小的**multi-scale VQ tokenizer**合成"图像"数据 (2D高斯环).
2. 列车**VAR-style transformer**预测代币的下一个规模.
3. 通过4次 (4个尺度) 调用变压器并解码样本.
4. 检查是否在规模上进行训练,使在规模内产生平行.

目的是看到规模结构化的注意力面具和平行在规模内生成实际工作.

## 运送它

这一课产生了`outputs/skill-var-tokenizer-designer.md`设计多尺度代币的技能:尺度数量,尺度比率,代码书尺寸,残余共享,解码架构.

## 运动

1. **Scale count ablation.**训练VAR使用4,6,8,10级. 测量重建质量与自行退行过关数量.更多的尺度 =更细的残留物 =更好的质量,但更多的过关.

2. **Codebook size.**训练代码符号器,用代码书尺寸5124096,16384更大的代码书可以更好地重建,但更难预测.

3. **Parallel-within-scale check.**对于训练有素的VAR,明确测量注意力模式.在尺度k内,模型是否关注跨尺度位置,但不是内部尺度?验证面具的实现.

4. **VAR vs DiT scaling.**对于相同的ImageNet类条件任务,训练VAR和DiT在匹配的参数预算 (例如33M, 130M, 458M). 剧情FID与计算.VAR应在每个尺寸中拉出DiT前面小规模复制论文的结果.

5. **Text conditioning.**扩展VAR以将文本嵌入 (CLIP合并) 作为通过 adaLN 进行额外的调节输入.这是HART的配方.FID在文本一致的采样中有多好?

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| VAR | "Visual AutoRegressive" | Image generation by next-scale prediction over a pyramid of VQ token grids |
| Next-scale prediction | "Predict coarser, then finer" | The model predicts tokens at increasing resolution scales, conditioning on all previous scales |
| Multi-scale VQ tokenizer | "Residual VQ" | VQ-VAE that produces K token grids of increasing resolution, with decoder summing all scales |
| Scale k | "Pyramid level k" | One of K resolution levels, from 1x1 at k=1 up to (H/p)x(W/p) at k=K |
| Parallel-within-scale | "One forward per scale" | All tokens at scale k are predicted in one transformer pass, not autoregressively |
| Causal-across-scales | "Scale-ordered attention" | Token at scale k can attend to all of scales 1..k but not scales k+1..K |
| Residual VQ | "Additive tokenization" | Each scale's tokens encode the residual left by lower scales; decoder sums all scale embeddings |
| VAR scaling law | "Image GPT scaling" | FID follows a predictable power law in compute, like language models' perplexity |
| HART | "Hybrid VAR + text" | Text-conditional VAR variant combining MaskGIT-style iterative decoding with VAR's scale structure |
| Scale position embedding | "(scale, row, col) triple" | Positional encoding carries both the scale index and spatial coordinates within the scale |

## 进一步阅读

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905)VAR文件,法典参考
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748)  DiT,扩散比较基线
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841)VQGAN,代币家属VAR的多尺度代币器扩展
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937)VQ-VAE,是分离图像标记的基础
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812)文本条件 VAR
