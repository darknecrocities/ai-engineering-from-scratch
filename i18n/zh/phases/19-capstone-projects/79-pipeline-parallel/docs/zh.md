# 管道平行和泡分析

> 数平行式分开矩阵乘以各行.管道平行式分开模型跨行,每个行各分别分开一个阶段.微分组通过管道流动.开始和结束的空时间是泡;最小化它是整个工艺.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 Track C lessons 42-49
**Time:** ~90 min

## 学习目标

- 分开一个连续模型成N阶段,并模拟N行间的前进管道.
- 使用GPipe时间表 (仅前面填充,然后倒退) 通过管道进行M微洗,并计算泡分数.
- 根据Megatron-LM和PipeDream所使用的1F1B时间表进行泡比较.
- 防御阶段分配:每阶段的等式计算比每阶段的等式参数数数更重要.

## 问题

只有在fp16中的70B参数模型需要140GB的参数. 没有消费者GPU可以控制它. 泽罗-3在各层中分断参数,但仍然需要每个层来收集每一步的全部层次,每层支付 log ((N) 跳. 管道平行采用不同的路径:将模型切成N阶段,并将每个阶段设置为一个阶段. 阶段1的前进在0级完成,并将激活子交给1级;阶段1运行2级,手交给2级等等. 倒流向后,倒流向后. 记忆线性下降,因为每个级别只包含一个阶段;计算是序列的,这是泡问题.

泡是管道开始时的置时间 (等待第一批微洗到最后阶段),以及最后 (等待最后一批微洗回流). 对于M微洗和N阶段,每个阶段的泡分数为 (N-1)/(M+N-1). 在M=8,N=4是27%. 在M=64时,N=4是4.5%. 泡会缩小,当你每步都有很多微型洗器,这意味着每微型洗器的批量量很小,

## 概念

```mermaid
flowchart LR
  R0[rank 0: stage 0 / layer 0] --> R1[rank 1: stage 1 / layer 1]
  R1 --> R2[rank 2: stage 2 / layer 2]
  R2 --> R3[rank 3: stage 3 / loss]
  R3 -.backward.-> R2
  R2 -.backward.-> R1
  R1 -.backward.-> R0
```

### 轮时间表

在向后开始之前,将M微洗手机填充到前方,然后向后排水. 任何微型的激活必须保持到它向后,所以记忆以线性方式增长与M. 前进需要M+N-1周期,后退需要另一个M+N-1周期. 每阶段有用的工作为2M周期;每阶段泡为2 ((N-1) 周期. 泡分数是 (N-1)/(M+N-1) 每个向前和向后都需要一个时间单位. 选择M比N大得多,就隐藏了泡.

### 时间表1F1B

间歇:一旦微型的前行达到最后阶段,就开始向后流,然后让它回流. 每个阶段,时间表交换一次向前,一次向后. 泡仍然是N-1,但激活记忆由管道深度限制,而不是微量. 生产管道使用1F1B (Megatron, PipeDream). 课程首先将GPipe执行,因为它更简单,

### 为什么每个阶段的等式计算是重要的

如果阶段0需要50ms,阶段1需要100ms,每个周期都被关在阶段1.其他阶段每周期停滞50ms等待阶段1释放.等等参数数数量是错误的轴:变压器的计算由注意力加上每层MLP主导,嵌入层具有许多参数,但计算很少.阶段分配应该等同于每个阶段的FLOP,而不是每个阶段的重量.

### 微分批量对批量

管道运行B尺寸M微.有效批量大小为M*B.管道步骤结束时的梯度是结合M*B示例的梯度.泡分数取决于M;优化器看到M*B.调整M意味着交易泡 (低于M高) 与每微存储器 (GPipe的高于M高的更高的激活存储).

```figure
cd-pipeline-bubble
```

## 建立它

`code/main.py`执行:

- `PipelineStage`子`nn.Module`具有一个阶段的参数和暴露`forward(activation)`现在,我们要去.
- `Pipeline(stages, num_microbatches)`通过模拟每阶段的墙钟,调整GPipe的时间表.
- `bubble_fraction(num_stages, num_microbatches)`:封闭式 (N-1) /(M+N-1).
- 通过4个阶段的演示, 打印每微分量的痕迹和测量的泡分数.

运行它:

```bash
python3 code/main.py
```

输出:一个阶段按微批次的甘特图表和与封闭形式预测相比的泡百分比.

## 野生生产模式

两种模式使管道硬化,

**Activation checkpointing pairs with pipeline.**通过GPipe飞行中的M微分钟,激活内存为M乘以1微分钟.激活检查点重新计算前进时间,对内存进行交易.

**Stage balance is measured, not assumed.**制作团队运行一个配置文件通过,测量目标硬件的实际每层计算 (FLOP和墙钟),然后按此测量进行分区.`--num-layers-per-stage`标志接受列表,以便在各阶段的成本不同时允许不均的层计数.

**Send-recv schedule must avoid deadlock.**管道,每个阶段都在收到电线上的门之前发送.标准的修正是交互:平排阶段先发送,然后回复,奇排阶段先回复,然后发送.课程时间表明确排列,所以模式可见.

## 用它

生产模式:

- **Megatron-LM.**参考管道平行尺度. 使用1F1B,支持子+管道+数据平行组合.
- **DeepSpeed Pipeline.**与 ZeRO 集成; ZeRO-1 + 管道是最大的开放模型的常见组合.
- **PyTorch Pipe.**管道包装, 基于`torch.distributed.pipeline.sync.Pipe`现在,我们要去.

## 运送它

第80课节将每个阶段参数的分片存储在分片检查点. 第81课节将DDP + ZeRO +管道组合到端到端演示中 (精神上;演示保持管道模拟运行时间).

## 运动

1. 执行1F1B,验证泡分数与GPipe相匹配,但激活内存是有限的.
2. 在更深层次的模型上,按测量的墙钟进行分析,并重新平衡阶段的实时阶段.
3. 通过管道微分组加上梯度积累,检查梯度等于相当的全批次向前梯度.
4. 通过激活检查点对配管道进行测量,测量内存下降与计算成本.
5. 结合管道与DDP (每个管道排名在数据平行组中复制) 并通过2D时间表进行推理.

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Pipeline | "Model parallel along depth" | One stage per rank, activations flow stage to stage |
| Bubble | "Pipeline idle time" | (N-1) steps at start + end where some stages have no work |
| Microbatch | "Slice of the batch" | One forward/backward unit; bubble shrinks as M grows |
| GPipe | "Fill then drain" | All M forwards before any backward; high activation memory |
| 1F1B | "Interleaved schedule" | One forward one backward per stage; bounded activation memory |

## 进一步阅读

- [Huang et al, GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [Narayanan et al, PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)
- [Megatron-LM pipeline parallel docs](https://github.com/NVIDIA/Megatron-LM)
- 第19阶段 第76课 - 时间表使用的发送/回收原始函数
- 阶段19课程78 - ZeRO与管道直角,通常是结合的
