# 士师的观察性堆选择

> 2026年可观测性市场分为两类. 开发平台 (LangSmith,Langfuse,Comet Opik) 通过评估,即时管理,会议重播来组装监测. 网关/仪器工具 (Helicone, SigNoz, OpenLLMetry,Phoenix) 专注于远程测量. 兰格斯是MIT授权的核心,具有强大的OSS平衡 (50K事件/月免费云). 尼克斯是基于Elastic License 2.0的OpenTelemetry原产品,非常适用于漂移/RAG可视化,而不是持续的生产后端. 亚里兹AX使用零副本的冰berg/Parquet集成,声称100倍比单可观测性便宜. 长史密斯为长链/长图带领, $ 39 / 用户 / 月, 仅在企业中自主托管. 机是代理的,15-30分钟的设置,100万个月的空调,但在代理的痕迹上, 常见的生产模式:Gateway (Helicone/Portkey) + eval平台 (Phoenix/TruLens) 附加在OpenTelemetry上.

**Type:** Learn
**Languages:** Python (stdlib, toy trace-sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## 学习目标

- 区分开发平台 (组合:评估+提示+会议) 与门户/远程仪器 (仅跟踪+指标).
- 绘制六个主要工具 (Langfuse,LangSmith,Phoenix,Ariz AX,Helicone,Opik) 根据其许可,定价和甜点使用情况.
- 解释OpenTelemetry粘合图案,允许您将门户工具与单独的评估平台结合起来.
- 命名2026年的成本差异 (Arize AX的零复制方法与单一摄入量) 并说明大约100倍乘法.

## 问题

你发送了LLM功能.它运行.你没有可见的即时失败,工具循环,延迟回归,成本高峰,或即时缓存的击率.你谷歌"LLM可观测性"并得到八种工具,所有声称它们都在三个不同的价格点解决相同的问题.

斯回答"我的RAG管道漂移吗?"子回答"哪个应用程序正在燃烧代币?"斯回答"我可以自主主主办整个东西吗?"不同的工具,不同的观众.

选择涉及四个轴:堆 (长链?原始 SDK?多供应商?),许可证容忍度 (仅为MIT?弹性 OK?商业罚款?),预算 (免费层次? $100/mo? $需要一个好主机,

## 概念

### 两类

**Development platforms**通过测试,提示管理,数据集版本化,会议重播,你运行实验,看哪个提示工作,数据集回归,对旧获胜者进行新的提示.

**Gateway/telemetry tools**仪器推断调用 快速,响应,代币,延迟,模型,成本.直升机,SigNoz,OpenLLMetry, Phoenix.最小化.可以通过OpenTelemetry与单独的评估工具结合.

### 长          

- 通过Docker进行自主托管.
- 云免费级别:每月50万次活动. 支付:每月29美元.
- 基本的数据,即时管理,追踪,数据集,所有四个开发平台的功能.
- 您需要LangSmith类功能,但必须自主托管或保留OSS许可.

###  (Arize) 电力测量第一,开放电力测量本土

- 弹性许可证2.0;自主主机.
- 非常擅长RAG和漂移视觉化, 嵌入空间散射图片是第一级的.
- 没有作为持续的生产后端设计,主要是开发时间可观测性.
- 热点:RAG管道开发,漂移调试,配对生产的单独门口.

###        

- 通过冰山/帕克特进行零复制数据湖集成.
- 据称,它比单形可观测性 (达达多格类) 便宜100倍.
- 需要具有LLC特异性的仪表板,而无需Datadog定价.

### 首先是兰格史密斯 兰格链/兰格图

- 商业价格为39美元/月,只能在企业上自主托管.
- 长链和长图堆的最佳. 如果你没有上任何一个,它不那么有吸引力.
- 团队承诺加入"长链",愿意付出.

### 基代理基的可行最低值

- 通过换取你的15-30分钟设置`OPENAI_API_BASE`给了直升机代理人.
- 美国麻省理工学院授权; 每月100万免费,每月20美元.
- 包含过失,缓存,利率限制 也作为一个门户.
- 经纪人/多步骤的痕迹的深度较小.
- 快速启动,单堆应用程序,需要一个门口+一个可观测的应用程序.

###  OSS开发平台

-  Apache 2.0 完全是OSS.
- 类似于星遗产的兰格斯.
- 点: ML 团队已经在彗星上,想要在同一面板上可以观察LLM.

### 开放Telemetry-第一完整APM

- 通过OpenTelemetry来处理一般APM加上LLM.
- 服务和LLM调用之间可观的统一性.

### 接:OpenTelemetry + GenAI语义公约

开通电气在2025年底发布了GenAI语义公约 (`gen_ai.system`现在`gen_ai.request.model`现在`gen_ai.usage.input_tokens`产品的生产模式出现:

1. 通过每次LLM电话,将Genai会议发送到 OTel.
2. 往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往往
3. 对于回归的双舰到评估平台 (Phoenix/Langfuse).
4. 通过Arize AX或DuckDB进行长期分析的数据湖 (Iceberg) 档案.

### 陷:在错误层上使用仪器

通过使用您的代理框架中的工具 (例如添加LangSmith痕迹) 将您与该框架相结合.在HTTP/OpenAI-SDK层 (通过OpenLLMetry或您的网关) 中的工具是可移植的.

### 样本你不能保留一切

在每天超过100万次请求时,完整的追踪成本高于LLM调用. 根据规则的样本:100%错误,100%高成本,5%成功. 总是保持总量;长尾保持原料.

### 你应该记住的数字

- 免费云:每月50万次活动.
- 长史密斯: 39美元/月.
- 免费升机:每月100万.
- 亚里兹AX声称:比单型尺价格便宜100倍.
- 开通电信GenAI公约:2025航运,2026年广泛采用.

```figure
i4-otel-glue
```

## 用它

`code/main.py`模拟了1M的追踪日间的保留策略 (100%摄入量,样本取,样本取+错误). 报告存储成本和每个数据下丢失的数据.

## 运送它

这一课产生了`outputs/skill-observability-stack.md`根据堆,规模,预算,许可证姿势,选择工具.

## 运动

1. 你的兰格链团队希望OSS自主托管可观测性.
2. 随着5M的每天跟踪,Datadog的价格为15万美元,
3. 设计一个OpenTelemetry GenAI属性,设置您的组织的指导方针应在每次LLM调用时强制执行.
4. 争辩说城是否足够生产.
5. 机是20ms代理上线费用.在P99TTFT300ms,这是可接受的吗?如果SLA是100ms?

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OpenLLMetry | "OTel for LLMs" | Open-source OpenTelemetry instrumentation for LLMs |
| GenAI conventions | "OTel attributes" | Standard OTel attribute names for LLM calls |
| LangSmith | "LangChain observability" | Commercial platform bundled with LangChain ecosystem |
| Langfuse | "OSS LangSmith" | MIT OSS with similar feature set |
| Phoenix | "Arize dev tool" | OpenTelemetry-native dev/eval platform |
| Arize AX | "scale observability" | Commercial zero-copy Iceberg/Parquet observability |
| Helicone | "proxy observability" | HTTP proxy collecting LLM telemetry + gateway features |
| Opik | "Comet LLM" | Apache 2.0 OSS dev platform from Comet |
| Session replay | "trace rerun" | Replay a full agent session with tool calls |
| Eval | "offline test" | Running candidate model/prompt over labeled dataset |

## 进一步阅读

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
