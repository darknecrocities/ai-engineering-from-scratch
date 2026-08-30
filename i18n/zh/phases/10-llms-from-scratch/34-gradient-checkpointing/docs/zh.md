# 渐进检查和激活重计算

> 后置保持每次中间激活.在70B参数和128K背景,每级为3TB的激活.检查点交易FLOPs的内存:重新计算而不是保存.问题是哪些段落要放下,答案不是"所有".

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## 问题

培训变压器存储每个层次的每个操作的输入,分为后向:注意力输入,Q/K/V投影,软max输出,FFN输入,标准输出和残留流.`d`序列长度`L`批量`B`现在,这是为了`12 * B * L * d`的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,的,,的,,的,,,的,,,,,的,,,,,,的,,,,,的,,,

为了`d=8192, L=8192, B=1`在BF16中,这相当于800MB/层. 64-层模型是51GB的激活,然后再乘以微量级,然后再添加注意力-软max中间体 (`L^2`在你考虑之前,

双面账单:BF16重量加上优化器状态可能适合80GB,但激活会推你过去.渐进检查点 (也称为激活重新计算) 是标准的解决方案.放弃大多数激活;在向后重复前进以获得它们.成本:额外的FLOP. 益处:记忆量减少了检查点段与总层的比例.

通过"智能选择"的Korthikanti等. 您可以节省5倍的内存,而FP8的上费用不到5%. 并且,FP8的,FSDP的脱载,和专家平行的MOE,这真的很重要:您无法负担任何内存或浪费的计算.

## 概念

### 那些落后的人真正需要什么

`output = layer(input)`后退的愿望`grad_input`其他`grad_params`为了计算它们,需要:

- `input`(计算`grad_params = input.T @ grad_output`对于线性层)
- 某些激活衍生品中间体 (ReLU/GELU/softmax的衍生品取决于激活值)

进口通行器将这些自动存储在自动格式图中.`tensor.retain_grad()`每个需要输入的操作都保留了参考.

### 无明的全面检查

分开网络`N`在前进过程中,只存储每个段子的 *输入*.当后退需要中间件时,再运行段子的前进通过来实现它们,然后区分.

举个例子:32层变压器分为32个分段,每个分别为1层.

- 存储器:32层输入 (小) 与32 * (每个层的激活量) (巨大).
- 额外计算:每段额外的前进1个,即前进FLOP总数增加了 ~33% (由于向后是前进2倍,完整步骤将成为1 + 1 + 2 = 4个单位而不是1 + 2 = 3).

这就是陈等人2016年的原始食谱:每一个检查点`sqrt(L)`对于L=64,那就是8个检查点.

### 选择性检查 (科尔蒂坎蒂 2022)

没有所有激活成本相同.`B*L*L*heads`随着序列长度的 * 方形化 * 增长.`B*L*4d`长序列中,软max 主导.

选择性检查点将廉价的存储激活 (线性投影,残余) 保持,只会重新计算昂贵的激活 (注意).

梅加特龙-核心将其作为"选择性"激活重计算.

### 放电

替代式:在前后和后后后之间将CPU RAM激活.需要PCIe带宽;在空置带宽超过重建成本时是有益的.混合策略是常见的:检查某些层,卸载其他层.

FSDP2 作为一流的选择,在 GPU 处于内存瓶状态时, 卸载光线闪耀, 但 CPU-GPU 转移则有空间.

### 计算成本模型

逐步的FLOP,每次都会进行无明的检查.`k`层`L`其他:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

通过选择性检查点,你只重新计算注意力内核,而不是整个层:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 存储记忆模型

每层激活量: `A`为了`L`层,总激活内存: `L * A`现在,我们要去.

完整检查点 (区块大小1): 仅存储`L * input_volume`,我知道.`L * 1/10 A`对于标准变压器).`9 * L * A * 1/10`现在,我们要去.

每次都会有检查点`k`层:存储`L/k * A`另外`k-1`在活跃部分内的层值.

在`k = sqrt(L)`存储和重新计算成本`sqrt(L)`对均成本层进行最佳交易.

### 什么时候不能到检查点

- 管道的最深层层已经在飞行中,他们必须要完成.
- 如果它们占据了舞台的计算 (在变压器中罕见) 的第一和最后层.
-  Flash 已经重新计算了软max速度,因此额外的层级检查点点增加了很少.

### 实施模式

1. **Function wrapper:**包裹一个段子`torch.utils.checkpoint.checkpoint(fn, input)`只有Pytorch商店`input`现在,我们可以重新计算一切.

2. **Decorator-based:**标签层作为可检查的;培训员在配置时决定哪些段子被包装.

3. **Manual explicit recompute:**写回后的传票,叫做习惯.`recompute_forward`转换的输入

包装是标准的语法.

### 与TP/PP/FP8相互作用

- **Tensor parallel:**检查点输入必须在重新计算上收集或重新分配;应承担通信成本.
- **Pipeline parallel:**典型的模式是检查每个管道阶段的前进点,以便反向序列的微洗手机可以重复使用激活内存.
- **FP8 recompute:**在重新计算过程中更新的 amax 历史必须与原始前进的,或FP8 尺度漂移相匹配.大多数框架快速拍摄尺度.

```figure
activation-recompute
```

## 建立它

### 第一个步骤:玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 第二步: 需要所有活动的天真回落

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 步骤3:检查点-每一个k 记忆

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 步骤4:成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 步骤5:记忆估计器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 步骤 6:最佳细分尺寸

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 七步:选择性检查站决定

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 用它

- **torch.utils.checkpoint**其他`from torch.utils.checkpoint import checkpoint` PyTorch 中的可行包装. 包裹一个函数; 仅存储输入,重新计算后退.
- **Megatron-Core activation recomputation**支持`selective`现在`full`其他`block`标准在2024+边境培训.
- **FSDP2 offload**其他`module.to_empty(device="cpu")`随着`offload_policy`在FSDP2中,切断激活到CPU而不是重新计算.
- **DeepSpeed ZeRO-Offload**: 优化状态和激活的CPU脱载,补充检查点.

## 运送它

这一课产生了`outputs/prompt-activation-recompute-policy.md`一个提示,将您的模型配置 (层,隐藏, seq,批量) 和可用的GPU内存取,并发出每层重计算政策 (无 / 选择性 / 完全 / 卸载).

## 运动

1. 检查正确性.`model_forward`其他`model_backward`(完全激活)`model_forward_checkpointed`其他`model_backward_checkpointed`参数梯度必须与机器精度相同.

2. 扫描部分大小`k`从1到`L`图片,记忆,找到曲线的膝盖.

3. 执行选择性检查点:存储注意力模块输入,但不是其中间件. 测量FLOP上层与全层检查点为32层模型的seq=8192.

4. 添加脱载.将段输入保存到模拟的"CPU缓冲器" (单独列表).以字节/时间测量"PCIe带宽",并找到脱载和重新计算之间的分离点.

5. 根据标准,一个真正的 PyTorch 变压器,有或没有`torch.utils.checkpoint`测量记忆 (通过`torch.cuda.max_memory_allocated`) 和步骤时间.

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Gradient checkpointing | "Save memory by redoing forward" | Store segment inputs only; recompute intermediates during backward to get gradient-support tensors |
| Activation recomputation | "Same as checkpointing" | The HPC-flavored name for the same technique |
| Segment size (k) | "How many layers per checkpoint" | Number of layers whose intermediates are dropped and rematerialized together |
| Selective checkpointing | "Korthikanti's trick" | Recompute only expensive-to-store activations (attention softmax); keep cheap ones |
| Full checkpointing | "The naive version" | Recompute every layer's intermediates in every segment |
| Block checkpointing | "Coarse-grained" | Checkpoint whole transformer blocks; largest granularity |
| FLOP overhead | "The compute tax" | Extra FLOPs per step = (recompute FLOPs) / (fwd + bwd FLOPs); 33% naive, 5% selective |
| Activation offload | "Ship to CPU" | Move activations to CPU RAM across forward->backward; alternative to recompute |
| sqrt-L rule | "The classical optimum" | For uniform-cost layers, optimal checkpoint spacing is sqrt(L) layers |
| Attention-softmax volume | "The O(L^2) problem" | L^2 * heads * batch floats; dominates activation memory at long contexts |

## 进一步阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174)根据原始文件,
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198)-- 选择性激活再计算和正式成本分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645)--通过反向模式重现物质化来替代常设存储方法
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840)-- 激活放电量
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html)--标准的API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html)--选择性,完整和区块模式
