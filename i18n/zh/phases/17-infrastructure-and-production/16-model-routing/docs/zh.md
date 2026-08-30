# 模式路由作为成本降低的原始

> 动态经纪人评估每一个请求 (任务类型,代币长度,嵌入式相似性,信心), 另外也叫做模特. 生产案例研究显示,在美国/英国/欧盟部署中,ISO-quality成本降低20-60%,在高量SaaS上提高30%的路由效率,将会成为每年六位数的节省. 2026年背景下,LLM推断价格每年下降了~ 10倍$20/M to ~$截至2026年底,每月0.40M. 基本上,服务更好于堆 (阶段17 · 04-09),而不是硬件. 路由是如何将价格下跌转换为利率,而没有产品回归. 失败模式是廉价模型漂移:路线将40%推向较弱的模型,质量在推理任务上下降3-5%,四分之一都没有人注意到. 通过在线质量指标来测试门路线,而不仅仅是离线评估集.

**Type:** Learn
**Languages:** Python (stdlib, toy cascading router simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 19 (AI Gateways)
**Time:** ~60 minutes

## 学习目标

- 解释模型级:低成本,首先是信任检查,
- 列出四个路由信号 (任务分类,快速长度,与已知硬组的类似性,自信自从第一次通过).
- 计算预期的混合成本在目标路由分区和质量损失耐受性时.
- 举个漂移监测指标 (在线质量门) 捕捉到廉价模型的爬虫.

## 问题

你的服务成本为GPT5月80万美元.你的分析显示,70%的查询很简单: "巴黎时间是多少?" "重写这个句子".一个海库类模型以成本的3%完全处理这些. 30%需要GPT-5的推理,编码,数学,多步计划.

如果您将70%的路由向廉价和30%的高价,您的账单将在相同的产品质量下降65%.这是路由.

## 概念

### 路由信号4个

1. **Task classification**简单/复杂/代码/数学/聊天.可以是基于规则的分类器,一个小的LLM (海库类为0.25美元/M),或嵌入标签的桶.输出:路线 =便宜 /平衡 /边界.

2. **Prompt length**提示>4K代币通常需要边界来保持一致性.提示<500代币通常不需要.

3. **Embedding similarity to known-hard set**查询接近已知硬桶 (CAS > 0.88) 则直接升级到边境.

4. **Self-confidence from first-pass**通过接,可通过接方式进行接,可通过接方式进行接,可通过接方式进行接.

### 三种模式

**Pre-route**(前面分类器): ~5-10ms延迟加上;总体来说最快.

**Cascade**低信任率: ~1.2倍的中延期 (廉价运行加上验证), ~2倍的升级.

**Ensemble route**(以样品为平行,以价格便宜和边境运行,以奖励模型为选择):最高质量,成本最高;仅用于关键的A/B.

### 实施

通过AI网关 (阶段17 · 19) 暴露路由.`router`通过"重建"的方法,我们可以使用"重建"的方法,并设置后退和成本路由.Portkey有保卫+路由.Kong AI Gateway有基于插件的路由.OpenRouter的模型市场暴露了推 API.

开源:路线LLM (LMSYS),非钻石 (商业),快速.

### 2026年价格曲线

| Model class | Late 2022 | 2026 | Change |
|-------------|-----------|------|--------|
| GPT-4-level quality | ~$20/M | ~$0.40/M | 50x cheaper |
| Frontier (GPT-5, Claude 4) | — | ~$3-10/M | new tier |

通过路由,您可以在应用层捕获这些收益,而不是等待所有用户迁移到廉价层.

### 漂移是真正的风险

路由器没有注意到,因为它的分类器是基于Q1数据训练.质量沉默下降.没有人抱怨足够大声.你在竞争对手的基准中发现你输了.

通过在线质量指标的门线路:

- 用户指公上/指公下,每条路线.
- 通过每条路线的抽样 (5%) 进行自动的LLM评判.
- 升级率:如果台上升率超过30%,便宜型号将被过度调整.
- 拒绝率每条路线.

### 你应该记住的数字

- 2026 路由节省量:20-60%的案例研究.
- 国际法学学学报价格下降2022-2026:每年总额约10倍
- 基质质质量测试 (GPT-4) 水平2022年至2026年: ~$20/M → ~$子,子,子,子
- 缩延迟影响:平均值约1.2倍,升级约2倍 (约10%的流量).

```figure
model-cascade-router
```

## 用它

`code/main.py`报告混合成本,质量损失和升级率.

## 运送它

这一课产生了`outputs/skill-router-plan.md`根据工作量和质量预算,选择路由模式和信号.

## 运动

1. 跑步`code/main.py`在哪个精度层上,落击中前路线?
2. 您的用户群体是30%的企业 (复杂查询),70%的免费层次 (简单). 设计路由分区. 网络的测量量是什么?
3. 运输线程的质量降低了2%,但节省了40%.
4. 通过OpenAI/人类API的记录检查进行信任检查.
5. 在六个月内,升级率从8%升至22%.

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Model routing | "cost broker" | Dynamic choice of model per request |
| Model cascade | "cheap-first escalate" | Run cheap, fall through to frontier on low confidence |
| Pre-route | "classify first" | Classifier up front; no re-run |
| Ensemble route | "parallel pick" | Run multiple, reward-model picks best |
| Escalation rate | "uprouted %" | Fraction of cascade requests that escalated |
| RouteLLM | "LMSYS router" | OSS router library |
| Not Diamond | "commercial router" | SaaS model-routing product |
| Drift | "cheap creep" | Distribution shift without router noticing |
| Online quality gate | "live check" | Automated LLM-judge sampling live traffic |

## 进一步阅读

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/)多个模型门口,具有路由原始.
