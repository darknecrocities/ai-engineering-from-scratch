#                   

> 兰格斯是开放的. 亚里兹·尼克斯发表了2026年GenAI Semconv映射. 机和BrainTrust都增加了每用户成本的倍率. 特拉塞洛普的OpenLLMetry成为了实际的SDK仪器. 制作形式是ClickHouse为痕迹,Postgres为元数据,Next.js为UI,以及一个小规模的评估工作 (DeepEval,RAGAS,LLM-judge) 通过样本的痕迹. 建立一个自主托管,从至少四个SDK家庭中摄入,并在不到五分钟内证明了接入后退.

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:**子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子子
**Time:** 25 hours

## 问题

每个在2026年运行生产流量的人工智能团队都会在模型旁边保持可观测平面. 成本归因. 发现幻觉. 水监测. 入监狱的信号. 机器的仪表板. 信息泄露警报. 开源引用 兰格斯,尼克斯,OpenLLMetry 作为摄入方案融合在OpenTelemetry GenAI语义公约. 现在可以使用一个SDK和运输兼容的跨度工具 OpenAI,Anthropic,Google,LangChain,LlamaIndex和vLLM.

测量:在有意注入回归 (一个提示开始产生 PII) 时,仪表板抓住它并在不到五分钟内发出警报.

## 概念

输入是OTLP HTTP. SDK生成了GenAI-semconv跨度: `gen_ai.system`现在`gen_ai.request.model`现在`gen_ai.usage.input_tokens`现在`gen_ai.response.id`现在`llm.prompts`现在`llm.completions`对于列分析,ClickHouse的地址;对元数据 (用户,会议,应用程序) 的地址是Postgres.

根据Eval的数据,Eval 测量了数据的数据,并将其运行在数据库中.Eval 测量了数据库中的数据库.Eval 测量了数据库中的数据库.Eval 测量了数据库中的数据库.Eval 测量了数据库中的数据库.Eval 测量了数据库中的数据库.Eval 测量了数据库中的数据库.Eval 测量了数据库中的数据库.Eval 测量了数据库的数据库.Eval 测量了数据库的数据库.Eval 测量了数据库的数据库.Eval 测量数据库的数据库的数据库.Eval 测量数据库的数据库的数据库.Eval 测量数据库的数据库的数据库的数据库.Eval 测量数据库的数据库的数据库的数据库的数据库.Eval 测量数据库的数据库的数据库的数据库的数据库的数据库的数据库的数据库,数据库的数据库的数据库的数据库,数据库的数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库,数据库

漂移检测时间内嵌入空间分布 (PSI或KL差异在快速嵌入) 加上评估分数趋势.警报输送Prometheus Alertmanager然后Slack / PagerDuty.用户界面是Next.js 15与Recharts.

## 建筑

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 堆

- 吞:OpenTelemetry SDK+GenAI语义公约;OTLP HTTP运输
- 收藏器:OpenTelemetry收藏器,配备尾样处理器 (用于成本控制)
- 存储:ClickHouse为跨度,Postgres为元数据,S3为原始事件档案
- 值:深度值,RAGAS 0.2,Ariz Phoenix评价器包,定制的LLM法官
- 漂移:PSI/KL每周的集成快速嵌入 (句子转换器)
- 警告: 预告警报管理器 ->  Slack / PagerDuty
- 接口:下一个.js 15 应用程序路由器 + 调用器 + 服务器操作
- 支持的 SDK:OpenAI,人类,谷歌GenAI,兰格链,LlamaIndex,vLLM

```figure
ce-otel-drift
```

## 建立它

1. **Collector config.**带有OTLP HTTP接收器,一个尾标记器,保持100%的错误痕迹和10%的成功,以及出口到ClickHouse和S3.

2. **ClickHouse schema.**表`spans`具有反射GENAI semconv的列: `gen_ai_system`现在`gen_ai_request_model`现在`input_tokens`现在`output_tokens`现在`latency_ms`现在`prompt_hash`现在`trace_id`现在`parent_span_id`添加使用者_id 和app_id 的二级索引.

3. **SDK coverage test.**使用OpenLLMetry自动仪器编写一个小客户端应用程序,使用每个SDK (OpenAI,Anthropic,Google,LangChain,LlamaIndex,vLLM).检查每个产生的可行的GenAI跨度,登陆ClickHouse.

4. **Eval jobs.**预定工作会读取最后15分钟的样本痕迹,并运行DeepEval忠诚度,毒性和答案相关性.输出是与母体痕迹相关的评估范围.

5. **Custom LLM-judge.**答案给出,请调用警卫的法学士,以评分PII泄漏的可能性.

6. **Drift detection.**星期工作计算了本周的集成快速嵌入和下来的4周基线之间的PSI. 如果PSI超过门,请警报.

7. **Dashboard.**下一页:概述 (跨度/秒,成本/用户,p95延迟),追踪 (搜索+布),评估 (忠诚度趋势,毒性),漂移 (随着时间的推移),警报.

8. **Alerting chain.**报警管理器向Slack进行警告,而对于关键违规则则则向PagerDuty进行访问.

9. **Regression probe.**注入 bug:评估的聊天机器人开始泄漏假SSN1%的时间.测量MTTR:从 bug部署到 Slack警报.

## 用它

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 运送它

`outputs/skill-llm-observability.md`根据LLM应用程序,仪表板将其痕迹吸收,运行评估,漂移警报,并在Next.js中显示成本/用户分类.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Trace-schema coverage | Number of SDK families producing canonical GenAI spans (target: 6+) |
| 20 | Eval correctness | DeepEval / RAGAS scores vs hand-labeled set |
| 20 | Dashboard UX | MTTR on injected regression (under 5 minutes target) |
| 20 | Cost / scale | Sustained ingest at 1k spans/sec without backlog |
| 15 | Alerting + drift detection | Prometheus/Alertmanager chain exercised end to end |
| **100** | | |

## 运动

1. 添加自定义仪器用于Haystack框架. 验证可нони卡通范围登陆ClickHouse与信徒`gen_ai.*`它们的属性.

2. 换取城评测器的深度评测率, 测量两个评测引擎之间的分数漂移.

3. 提高漂移探测器:计算每个应用程序ID的PSI,而不是全球.

4. 添加一个"用户影响"页面:每用户成本和每用户失败率,

5. 建立一个尾巴样本政策,以保持100%的毒性超过0.5的痕迹,再加上10%的分层样本.

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM attributes" | 2025 OpenTelemetry spec for LLM span attributes (system, model, tokens) |
| Tail sampling | "Post-trace sample" | Collector decides to keep or drop a trace after it completes (can peek errors) |
| PSI | "Population stability index" | Drift metric comparing two distributions; > 0.2 typically signals meaningful drift |
| LLM-judge | "Eval as model" | An LLM scoring another LLM's output on a rubric (faithfulness, toxicity, PII) |
| Tail-sampling policy | "Keep-rule" | Rule that decides which traces to persist vs drop; errored + sample-rate |
| Eval span | "Linked eval trace" | Child span carrying an eval score linked to the original LLM call span |
| Cost per user | "Unit economics" | Dollar cost attributed to a user_id over a window; key product metric |

## 进一步阅读

- [Langfuse](https://github.com/langfuse/langfuse) 参考开放核心可观测平台
- [Arize Phoenix](https://github.com/Arize-ai/phoenix)具有强大的漂移支持的替代参考
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry)自动仪器 SDK 家庭
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)摄入方案
- [Helicone](https://www.helicone.ai)可替代托管可观测性
- [Braintrust](https://www.braintrust.dev)替代的第一评估平台
- [ClickHouse documentation](https://clickhouse.com/docs) 专跨度存储
- [DeepEval](https://github.com/confident-ai/deepeval)评价者图书馆
