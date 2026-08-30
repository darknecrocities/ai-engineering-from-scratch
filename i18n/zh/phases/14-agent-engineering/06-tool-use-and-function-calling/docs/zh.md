# 工具使用和功能调用

> 工具former (Schick等同,2023) 开始自主监督工具注释.伯克利函数调用领袖板 V4 (Patil等同,2025) 设定了2026年条:40%的代理,30%的多转,10%的现场,10%的非现场,10%的幻觉.单转是解决的.记忆,动态决策和长视线工具链不是.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 minutes

## 学习目标

- 解释Toolformer的自我监督训练信号:只在执行后节点损失时保留工具注释.
- 列出BFCL V4的五个评估类别,并列出每个评估类别的标准.
- 实现一个使用方案验证,引证强制和执行沙盒的 stdlib 工具登记.
- 诊断2026年出现的三个问题:长视野工具链,动态决策和记忆.

## 问题

早期工具使用被问到:模型能预测一个正确的函数调用吗?现代工具使用问:模型链工具可以跨越40步,具有内存,有部分可观测性,从工具故障中恢复,而不产生幻觉的工具?

工具创始人建立了基线:模型可以学习在自主监督下何时调用工具.BFCL V4定义了2026年评估目标.

## 概念

### 工具former (Schick等人,NeurIPS 2023)

想法:让模型用候选人 API 调用来注释自己的预训练库.对于每个候选人,执行它.只要包括工具结果,就会减少下一个代币的损失. 选的库.

工具包括:计算器,质量检测系统,搜索引擎,翻译器,日历.自监测信号纯粹是关于工具是否有助于预测文本没有人类标签.

规模结果:工具使用出现规模.较小的模型受工具注释的影响;较大的模型获益.这就是为什么2026年边界模型具有强大的工具使用,而大多数7B模型需要明确的工具使用细节调节才能靠谱.

### 伯克利函数调用排名表 V4 (帕蒂尔等人,ICML 2025)

根据"2026年"的实际评估,

- **Agentic (40%)** 完全代理轨迹:记忆,多轮,动态决策.
- **Multi-Turn (30%)**与工具链进行互动对话.
- **Live (10%)**用户提交的真实提示 (更硬的分布).
- **Non-Live (10%)**合成试验案例.
- **Hallucination (10%)**检测不应调用任何工具时.

V3引入了基于状态的评估:在工具序列之后,检查API的实际状态 (例如"文件是否创建?") 而不是与工具调用的AST匹配. V4增加了网页搜索,内存和格式敏感性类别.

关键2026发现:单转函数调用几乎解决了.失败集中在记忆中 (跨转中运载文本),动态决策 (基于先前结果选择工具),长视线链 (20多步后漂移) 和幻觉检测 (拒绝调用当没有工具合适时).

### 工具方案

每个提供商都有一个方案.

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

人类用途 `input_schema`直接使用.`function.parameters`两个都接受JSON方案.描述承载量,模型阅读它们来选择正确的工具.不良的工具描述是错误的工具失败的第一根原因.

### 证据验证

没有工具的呼叫.

1. **Type coercion.**模型可能会返回一个字符串 "5",该字符串表示 int. 如果不含糊,强迫;如果不,拒绝.
2. **Enum validation.**如果该方案说`status in {"open", "closed"}`及模型排放量`"in_progress"`拒绝使用描述错误.
3. **Required fields.**错失的需要字段 -> 立即回到模型,而不是崩.
4. **Format validation.**通过混凝土仪验证日期,电子邮件,URL,而不是regex.

每次验证失败都应返回结构性观察,以便模型能够再次尝试正确的形状.

### 并行工具调用

现代提供商支持平行工具调用,

1. 模型发出3个工具调用,具有不同的`tool_use_id`其他
2. 运行时间执行它们 (如果独立,则并行).
3. 每个结果都回归为`tool_result`相关的块`tool_use_id`现在,我们要去.

工程规则:把相关性ID视为承载,交换它们,你得到错误工具到错误结果的路由.

### 沙盒

简短版本:每个工具都应该指定阅读/写面,网络访问,时间限,内存盖. 概括 `run_shell(cmd)`是红旗;具体`git_status()`这更安全.

```figure
tool-routing
```

## 建立它

`code/main.py`执行生产形工具登记册:

- 通过JSON Schema的子集验证器 (仅在stdlib).
- 工具注册,包含描述,输入方案,时间限和执行器.
- 证据强制和证实.
- 配合工具发送与相关身份证.
- 错误观测作为结构字符串.

运行它:

```
python3 code/main.py
```

痕迹显示一个小代理在一次轮回召唤三个工具, 一次故意错误的呼叫被拒绝,

## 用它

每个提供商都有自己的工具方案.如果需要多提供商,请使用翻译层 (OpenAI Agents SDK,Vercel AI SDK,LangChain工具适配器).BFCL是参考基准.

## 运送它

`outputs/skill-tool-registry.md`包含描述质量检查 (每个工具的描述是否告诉模型何时使用它?).

## 运动

1. 添加一个"无操作"工具,允许模型明确拒绝使用任何其他工具.
2. 强制性是什么原因?
3. 增加每种工具的时间和断路器 (在连续3次故障后拒绝60年的工具). 这改变了模型恢复方式的情况.
4. 阅读BFCL V4描述. 选择一个类别 (例如"多转") 并通过代理运行10个示例提示. 报告通过率.
5. 让我们把SDLB验证器转移到Pydantic或Zod.

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Function calling | "Tool use" | Structured-output tool invocation with validated schema |
| Toolformer | "Self-supervised tool annotation" | Schick 2023 — keep tool calls whose results reduce next-token loss |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 benchmark: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination |
| Tool schema | "Function signature for the model" | name, description, JSON Schema of arguments |
| tool_use_id | "Correlation ID" | Ties a tool call to its result; essential for parallel dispatch |
| Hallucination detection | "Know when not to call" | V4 category: refuse to call when no tool fits |
| Argument coercion | "String-to-int repair" | Narrow fixes for predictable schema-mismatch; reject if ambiguous |
| Sandboxing | "Tool execution boundary" | Per-tool read/write surface, network, timeout, memory cap |

## 进一步阅读

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761)自主监督工具注释
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html)2026年评估基准
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) Claude Agent SDK中的生产工具方案
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)功能工具类型和Guardails
