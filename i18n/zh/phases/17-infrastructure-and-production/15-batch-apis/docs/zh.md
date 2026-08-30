# 批量API 50%折扣作为行业标准

> 每个主要供应商都会发送一个非同步的批量API, 开放AI,人类,谷歌和大多数推断平台 (火灾工程批次,联合批次) 都实施了相同的模式. 随着快速缓存的堆积批量和一夜之间管道降至同步未存储成本的10% 规则非常简单:如果它不是互动的, 内容生成管道,文件分类,数据提取,报告生成,批量标签,目录标签 任何容忍24小时延迟的东西都是剩下的钱,直到它转移到批量. 2026年生产模式是将每一个新的LLM工作负载分为三个行程:互动 (与缓存同步),半互动 (与倒退无步排队),批量 (过夜,缓存输入堆). 工作负载假装是互动性的,但容忍数分钟的延迟浪费最多.

**Type:** Learn
**Languages:** Python (stdlib, toy batch-vs-sync cost simulator)
**Prerequisites:** Phase 17 · 14 (Prompt & Semantic Caching)
**Time:** ~45 minutes

## 学习目标

- 举个供应商批量API (OpenAI,Anthropic,Google) 和共同的50%折扣+24小时回报保证.
- 计算一夜间分类工作负载上堆叠批量+缓存输入的成本,并与同步未存储的基线进行比较.
- 按分组分一个工作负载成交互式/半交互式/批量,并证明行径是正确的.
- 举个两个陷:部分互动性 (用户预计更快于24h) 和输出方案漂移 (按供应商不同批量文件格式).

## 问题

你的团队每晚发送一个报告生成管道. 50,000 份文件,总结每份文件,集结总结,编写执行简报.同步运行需要4个小时,每晚2000美元.

按一下,您可以在系统提示器上进行即时缓存 (共享所有50k通话). 按一下,账单下降到180美元/夜.

批量是LLM成本工具包中最便宜的杆,没有人抽出.原因主要是组织性:团队认为"实时"当SLA实际上是"早上".

## 概念

### 三个批次的API

**OpenAI Batch API**预约24小时的转换 (通常在实践中是2-8小时). 50%的输入和输出代币折扣. `/v1/batches`预存可入的输入也得到了预存输入的定价.

**Anthropic Message Batches**随时运行,50%折扣,支持`cache_control`缓存写字是明确的,读音在批量内自动发生.

**Google Vertex AI Batch Prediction**双子座的50%折扣. 集成到 Vertex 管道.

### 语义:异步,不是缓慢

典型的P50是2-6小时.提供商在高峰窗口之外安排你的批量,当GPU库存未充分使用时.

### 存储存储

通过同一个4K代码系统提示,

- 交时未存储:50000 × ($input × 4000 + $输出 × 200) 完全速率.
- 同步缓存:系统提示在第一次写完后被缓存;剩下的49999得到10倍便宜的输入.
- 存储存储:上述所有内容加上 50% 的阅读和写作折扣.

堆:批量+缓存 = 交代未缓存账单的10%左右.任何一夜运行且具有共享系统提示的工作负载都应该使用此.

### 工作负载分类

**Interactive**用户等待响应.TTFT重要.同步调用,即时缓存.不能批量.

**Semi-interactive**用户提交任务,在几分钟内检查回来. 随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随时随的随时随时随时随时随时随时随时随的随时随时随时随时随时随的随时随时随时随时随时随的随时随时随的随时随时随时随的随时随时随时随的随时随时随的随时随时随的随时随时随的随时随的随时随的随时随的随时随时随的随的随的随的随时随的.

**Batch**用户预计"早上"或"下一个小时"的结果. 内容管道,规模分类,离线分析.

常见错误:将所有东西归类为互动,因为管道是生产.生产不是延迟规范.

### 部分互动性陷

一些功能看起来是互动性的,但可以容忍5到10分钟. 例如:每晚的客户健康报告,按"刷新"按.用户点击刷新;等10分钟是可以的.团队将其作为同步的运输.50次同时刷新成本是批量和通过电子邮件交付成本的10倍.

如果答案是"他们不会注意到",那么请按一下.

### 输出方案陷

批量文件格式因供应商而异:

- 开放AI:JSONL,每行一个请求.
- 人类:JSONL,每行一个消息;嵌入式响应格式.
- 垂直:BigQuery表或GCS前,并使用TFRecord.

通过提供商中写"一批客户端"意味着每个提供商的适配器代码.广告多提供商批量 (Portkey, LiteLLM某些层次) 的门户仍然薄包装原始格式.

### 你应该记住的数字

- 供应商之间批量折扣:输入+输出均为50%.
- 转换SLA:24小时保证,2-6小时典型的P50.
- 堆叠的批量+缓存输入: -10%的同步未缓存成本.
- 工作负载分类规则:如果24小时延迟是可接受的,总是批量.

```figure
batch-lane-triage
```

## 用它

`code/main.py`计算50万份文件工作负载的同步,同步+缓存,批量和批量+缓存成本. 报告以$和百分比节省.

## 运送它

这一课产生了`outputs/skill-batch-triager.md`鉴于工作负载特征,分为互动/半/批量,并估计节省.

## 运动

1. 跑步`code/main.py`对于一个100kdoc管道,有3K代币系统提示和500代币输出,计算全堆积 (批量+缓存) 的节省与同步基线.
2. 选择你熟悉的真实产品中的三个功能,
3. 报告需要3个小时的时间,这是一个错误的批量排序,还是一个合法的互动?
4. 您的批量API返回SLA是24h,但P99是20h. 如何向用户传达这个?
5. 计算平衡:在哪个共享前长度下,批量+缓存比在您自己的预留GPU上一夜运行更便宜?

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Batch API | "async discount" | 50% off with 24h turnaround |
| JSONL | "batch format" | One JSON request per line; OpenAI/Anthropic standard |
| Message Batches | "Anthropic batch" | Anthropic's batch API product name |
| Batch prediction | "Vertex batch" | Vertex AI's batch API product |
| Turnaround SLA | "24h promise" | Guarantee, not typical; typical is 2-6h |
| Workload triage | "interactivity decision" | Interactive / semi / batch routing decision |
| Output schema | "response format" | Per-provider JSONL layout; not portable |
| Stacked discount | "batch + cache" | ~10% of uncached sync bill when both apply |

## 进一步阅读

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) JSONL格式和`/v1/batches`它们是什么意思?
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing)批量格式和`cache_control`互动.
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini)双子座的批量语义.
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
