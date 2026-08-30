# 石39课:指导调整,由监督调整调整

> 预训练的基模型可以延长一个序列,但不能遵循指令. 监督的细节调整是最小的改变, 问题是,你只想把损失计算在响应上,而不是指令上. 这一课构建了Alpaca式SFT循环,具有自定义的集函数,以掩盖指令令标记.`ignore_index=-100`通过精确匹配,对待了200个指令-响应对,

**Type:** Build
**Languages:** Python (torch, numpy)
**Prerequisites:** Phase 19 lessons 30-37 (NLP LLM track: tokenizer, embedding table, attention block, transformer body, pre-training loop, checkpointing, generation, perplexity)
**Time:** ~90 minutes

## 学习目标

- 格式化对应指示数据,以单个因果序列,并使用明确的边界代码.
- 建立一个掩盖指令代币的集合函数,以便交叉值只计算响应代币.
- 训练一个小的变体体在SFT目标下,看评估指标的移动.
- 实现贪和温度样本生成,尊重响应开始界限.
- 计算生成的完成量上的确切匹配.

## 问题

根据下一个代币预测训练的基模型不知道命令是什么.`"What is the capital of France?"`模型有语言,但没有格式合同.

任何培训例都会成为一个单一的序列,有三个区域:

```text
<INST> What is the capital of France? <RESP> The capital of France is Paris.
```

边界代币是训练时间保留的特殊代币.`<RESP>`基本模型的下一个标志目标仍然适用;它只是在一个体积上训练,每个例子都有这个形状.

但是有一个问题.如果你把整个序列输入到尼拉交叉缩损失中,你就训练模型来预测指示符号.

## 概念

```mermaid
flowchart LR
  Pair[instruction + response] --> Tmpl[apply template<br/>INST + RESP tokens]
  Tmpl --> Tokens[token ids]
  Tokens --> Mask[loss mask<br/>-100 on instruction]
  Mask --> Model[transformer body + LM head]
  Model --> CE[cross-entropy<br/>ignore_index=-100]
  CE --> Step[backward + optimiser step]
```

`ignore_index`是一个特征`torch.nn.functional.cross_entropy`任何目标位置等于`ignore_index`光的定制是:`-100`结合函数构建两个子,例如: `input_ids`其他类型的类型`labels`(本文的副本)`input_ids`通过 `-100`)

模型在前进传递过程中看到整个序列;注意力可以关注指示. 损失只会计算响应代币. 这正是你想要的:指示条件,预测响应.

## 数据

通过确定性生成200个指示响应对`main.py`它们涵盖六种任务类型:

- 实事单次投射 (X项资本)
- 算术
- 清单提取
- 一句总结
- 代码 (打印,排序)
- 定义

每个任务都有一个模板的指示和确定性反应.这是故意简单的.精确匹配很脆弱,课程使用一个固定,正确答案是一个特定字符串.真正的SFT数据集需要模糊的指标;原则是相同的.

测试集包括所有六种任务类型,以便每类别可报告精确匹配.

## 标记和接

标记器是字节级别的,有三个保留的特点:

- `INST_ID = 256`标志着教学区的开始.
- `RESP_ID = 257`标志着指令和响应之间的界限.
- `PAD_ID = 258`:适用于变长批量.

序列是`[INST] inst_bytes [RESP] resp_bytes [PAD]*`聚合函数:

1. 标志着每一个例子.
2. 入每一个分组中的例子,
3. 建筑物`labels``input_ids`转移到一个 (因果性LM目标),其中:
   - 指示区域被取代为 `-100`现在,我们要去.
   - 填充区域被取代为`-100`现在,我们要去.
   - 其他`RESP_ID`边界位置本身被取代为 `-100`(你不训练模型来预测边界标志;它预测下面的情况).

```mermaid
flowchart TD
  Batch[(examples)] --> Tok[encode + insert specials]
  Tok --> Pad[pad to longest]
  Pad --> Shift[shift labels by one]
  Shift --> Mask[set -100 on<br/>inst / pad / boundary]
  Mask --> Out[(input_ids, labels)]
```

转变是标准的因果技巧:位置`i`其他`input_ids`预测位置`i+1`现在,`labels[i] = input_ids[i+1]`面具在转移后应用于右位置.

## 培训

```mermaid
flowchart LR
  DL[Train loader<br/>200 pairs] --> Fwd[forward]
  Fwd --> Logits[B x T x V]
  Logits --> Loss[CE with -100 mask]
  Loss --> Bwd[backward]
  Bwd --> Opt[Adam optimiser]
  Opt --> Body[(updated body)]
```

循环是标准的 PyTorch SFT循环.亚当,学习速度在3e-4到1e-3,这个装置上有十到二十个时代,没有安排器.模型足够小 (96,2块隐藏,最大长度64) 训练到两个分钟内在CPU上融合.

每五个时代,循环在持久的集合上进行了微小的评估通过,并打印出精确匹配.观看精确匹配从0.0在时代一到15时代的0.85是课程的回报:你可以看到模型同时学习格式和答案.

## 世代

在评估时,模型得到了指令前.`[INST] inst_bytes [RESP]`并且生成代币,直到:

- 序列达到`max_len`其他
- 模型发出特殊的停止数:连续两次句子结束字节 (`.`现在`!`现在`?`)

课程中,有贪的解码加上可选的温度样本器.精确匹配使用贪,因为温度会使得测量量量量是固态的.实际系统通常采样,然后模糊地判断;那管道是课程 41.

## 精确匹配的评估

准确匹配是最严格的文本指标.预测响应字符串是正常化的 (小字母,条纹白空,崩双空) 和与参考响应相比,是正常化的.指标是1或0例如.总数是平均值.

实际的SFT管道与代币级 F1 (课 41) 和评审模型补充了精确匹配.精确匹配仍然有用,因为它是明确的;如果它说0.7,正是70%的测试说明产生了字符的黄金响应字符.

```figure
cc-sft-loss-mask
```

## 你会建造什么

实施是一个`main.py`另外还有一些检查.

1. `InstructionTokenizer`编码命令前置或一个完整的对.
2. `make_dataset`产生的200个对在六个任务类型,一个固定的种子.
3. `SFTDataset`收益`(input_ids, labels)`面具已经准备好了.
4. `sft_collate`动态填充,构建批量度,集成`-100`在指令和位上.
5. `TinyGPT`:变压器机体加上绑定或解锁的LM头.
6. `train_sft`通过"SFT循环"进行一次性评估.
7. `generate`:因果解码从一个前,贪或采样,停止的法.
8. `exact_match`标准化字符串比较,回报率浮动`[0, 1]`现在,我们要去.
9. `run_demo`分析,按类别打印分类,成功的结果是零的.

## 为什么面具很重要

没有面具,损失将指令令作为目标. 模型学会预测指令. 这是一个不同的目标,以两种方式产生了更糟糕的模式. 首先,模型容量是浪费的, 其次,响应损失在梯度总数中较小,因为指令令比大多数批量响应令牌更多;优化器对你关心的部分的有效学习率低于你预期的. 面具不是抛光,而是目标.

## 实现目标

- 增加学习速度升温,然后出现骨质衰退.
- 补充每代币损失记录,并绘制损失曲线在训练中.`<RESP>`后期由实际答案代币主导.
- 扩展到BLEU-1或 chrF. 精确匹配低估了产生相同答案的表达式模型.
- 添加一个多轮格式化的聊天模板,并使用包括后续的设置训练.

实现给你提供格式合约,面具和循环. 从基模型到指令追随器的客观变化是一个集函数.
