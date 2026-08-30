# 机器和机器的变量

> 简单的自动编码器压缩,然后重建.它记住.它不会生成. 添加一个技巧强加代码看起来高斯式,你得到一个样本器.`z = μ + σ·ε`于是,每一个2026年使用的隐形传播和流量相匹配图像模型都会在输入时有VAE.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 07 (CNNs), Phase 8 · 01 (Taxonomy)
**Time:** ~75 minutes

## 问题

压缩一个784像素的MNIST数字到16个数字代码,然后重建.一个简单的自动编码器将重建MSE,但代码空间是一个乱的混乱.在代码空间中选择一个随机点,解码它,你会得到噪音.它没有样本器.它是一个装饰的压缩模型.

实际上你想要的是: (a) 代码空间是一个清洁的,平稳的分布,你可以从一个同位性高斯人样本`N(0, I)`编码器和解码器仍然压缩得很好. 三个目标,一个架构,一个损失.

通过训练编码器输出 * 分布 *`q(z|x) = N(μ(x), σ(x)²)`拉出了分布到前方`N(0, I)`通过 KL 罚款,然后采样`z`其他`q(z|x)`在解码之前.在推断时,放下编码器,样本`z ~ N(0, I)`卡洛特的惩罚是迫使代码空间结构化.

2026年,VAE很少独立运输,它们因质量而被排名出了,但它们是每个隐藏式扩散模型 (SD 1/2/XL/3,Flux,AudioCraft) 的最佳编码器.学习VAE,你将学习你使用的每个图像管道的无形第一层.

## 概念

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**Autoencoder.** `z = encoder(x)`现在`x̂ = decoder(z)`损失 = `||x - x̂||²`代码空间是不结构化的.

**VAE encoder.**输出两个向量:`μ(x)`其他`log σ²(x)`这些定义了`q(z|x) = N(μ, diag(σ²))`现在,我们要去.

**Reparameterization trick.**采样`q(z|x)`检测量: 检测量:`z = μ + σ·ε`在哪里`ε ~ N(0, I)`现在`z`是一个定性函数`(μ, σ)`梯度流通通过 `μ`其他`σ`现在,我们要去.

**Loss.**证据下层结合 (ELBO),两个术语:

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建推动了`x̂`走向`x`克莱拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉拉`q(z|x)`它们交换. 小 β (<1) = 较敏的样本,代码空间少于高斯. 大 β (>1) = 更清洁的代码空间,模糊的样本. β-VAE (Higgins 2017) 使这个按成为著名的,并启动了解脱研究.

**Sampling.**在推断时:抽出`z ~ N(0, I)`通过解码器进行前进. 一个前进传输,没有反复采样,比如扩散.

```figure
vae-latent-grid
```

## 建立它

`code/main.py`输入是从8维的2组件高斯混合物中获取的8维合成数据.编码器和解码器是单个隐藏层MLP.我们实现了tanh激活,前传,损失和手写后传.不是生产教学.

### 步骤1:向前编码器

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

`log σ²`没有`σ`因此网络输出不受限制 (s 的软加值是陷  梯度在 σ ≈ 0 时死亡).

### 步骤2:重组和解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 步骤3:ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

由于两个分布都是高斯式的,所以它不能数字化整合.人们仍然在2026年运输代码,蒙特卡洛估计KL速度将会慢得3倍.

### 步骤4:生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型,五行.

## 陷

- **Posterior collapse.**卡通通通用驱动器`q(z|x) → N(0, I)`如此积极的`z`没有关于`x`修复: β-取消 (开始 β=0, 向 1), 释放位,或在不活跃的尺寸上跳过 KL.
- **Blurry samples.**盖斯解码器概率意味着MSE重建,这是Bays最佳的L2 (平均值) 一个可信数字的平均值是模糊的数字. 修正:分离解码器 (VQ-VAE,NVAE),或仅用VAE作为编码器和堆扩散在隐藏 (这是稳定扩散所做的).
- **β too large, too early.**看到后部崩. 开始在 β≈0.01 和道.
- **Latent dim too small.**16D为MNIST工作,256-D为ImageNet 2562,2048-D为ImageNet 10242.稳定扩散的VAE压缩为512×512×3 →64×64×4 (32x空间面积下样数,32x频道).

## 用它

2026 年的VAE堆:

| Situation | Pick |
|-----------|------|
| Image-latent encoder for diffusion | Stable Diffusion VAE (`sd-vae-ft-ema`) or Flux VAE |
| Audio-latent encoder | Encodec (Meta), SoundStream, or DAC (Descript) |
| Video latents | Sora's spatiotemporal patches, Latte VAE, WAN VAE |
| Disentangled representation learning | β-VAE, FactorVAE, TCVAE |
| Discrete latents (for transformer modelling) | VQ-VAE, RVQ (ResidualVQ) |
| Continuous latents for generation | Plain VAE, then condition a flow/diffusion model in that latent space |

隐形传播模型是一个VAE,其中一个分布模型在编码器和解码器之间存在.VAE执行粗压,扩散模型执行重量起重.视频 (VAE +视频扩散diT) 和音频 (Encodec + MusicGen变压器) 的模式相同.

## 运送它

保存`outputs/skill-vae-trainer.md`现在,我们要去.

技能:数据集的配置文件 + 隐形dim目标 + 下游使用 (重建,采样或隐形diffusion输入) 和输出:建筑选择 (平面/β/VQ/RVQ), β时间表,隐形dim,解码概率 (Gaussian vs 类型),评估计划 (Recon MSE, KL per dim, Fréchet 距离之间的距离`q(z|x)`其他`N(0, I)`)

## 运动

1. **Easy.**改变`β`在`code/main.py`为了`0.01`现在`0.1`现在`1.0`现在`5.0`记录最后的重建MSE和KL. 哪个 β是最适合你的合成数据?
2. **Medium.**替换高斯解码器概率为伯诺利概率 (跨进力损失).对同样的合成数据的二进制版本进行样本质量比较.
3. **Hard.**延长时间`code/main.py`换成小型VQ-VAE:将连续的`z`通过在 K=32 条目编程册中查找最近邻居. 进行重建MSE的比较,并报告使用的代码册条目数量 (代码册崩是真实的).

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Autoencoder | Encode-decode network | `x → z → x̂`, learn MSE. Not generative. |
| VAE | AE with a sampler | Encoder outputs a distribution, KL penalty shapes code space. |
| ELBO | Evidence lower bound | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`; tight when `q = p(z\|x)`. |
| Reparameterization | `z = μ + σ·ε` | Rewrites stochastic node as deterministic + pure noise. Enables backprop through sampling. |
| Prior | `p(z)` | Target distribution for the latent, typically `N(0, I)`. |
| Posterior collapse | "KL term wins" | Encoder ignores `x`, outputs the prior; decoder must hallucinate. |
| β-VAE | Tunable KL weight | `loss = recon + β·KL`. Higher β = more disentangled but blurrier. |
| VQ-VAE | Discrete latent | Replace continuous `z` with nearest codebook vector; enables transformer modelling. |

## 产品注:VAE是扩散服务器中最热路径

在稳定扩散/流动/SD3管道中,VAE 需要每次调用两次,一次编码 (如果做 img2img / inpainting) 和一次解码.在10242时,解码器传递通常是整个管道中最大的激活记忆峰值,因为它提升了`128×128×16`隐藏的回归`1024×1024×3`两种实际后果:

- **Slice or tile the decode.** `diffusers`暴露`pipe.vae.enable_slicing()`其他`pipe.vae.enable_tiling()`件交易小件`O(tile²)`记忆而不是`O(H·W)`对于10242+的消费者GPU.
- **bf16 decoder, fp32 numerics for the final resize.**在10242+SDXL船上,SD 1.x VAE在fp32中发布,在投到fp16时,它地产生了NaNs.`madebyollin/sdxl-vae-fp16-fix`总是偏爱fp16-fix变体或使用bf16.

## 进一步阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)VAE论文.
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)分离 β-VAE.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) VQ-VAE
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898)最新的图像.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)稳定扩散;VAE作为编码器.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) Encodec,音频VAE标准.
