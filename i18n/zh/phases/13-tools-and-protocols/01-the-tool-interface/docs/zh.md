# 工具界面 代理人需要结构化I/O

> 语言模型产生代币.一个程序采取行动.这两者之间的差距是工具界面:一个合同,允许模型请求操作,主机执行它.每2026年,堆 函数都需要OpenAI,Anthropic和Gemini;MCP的 `tools/call`是同一四步循环的不同的编码.这个课程命名循环,显示运行最小的机器.

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11 (LLM completion APIs)
**Time:** ~45 minutes

## 学习目标

- 解释为什么只能生成文本的法学士不能单独对现实世界采取行动.
- 绘制四步工具调用循环 (描述 → 决定 → 执行 → 观察) 并命名每个步骤的所有者.
- 写一个工具描述为三个部分:名称,JSON Schema输入和确定性执行函数.
- 区分纯净的工具和副作用,并说明为什么分断对安全性有重要意义.

## 问题

士师在下一个代币上发出概率分布.这是整个输出表面.如果你问聊天模型"现在孟加拉的天气是什么",它可以写出一个可信的句子,但它不能拨打一个天气API.句子可能是偶然的或三个天的陈旧.

工具界面的目的是关闭这一差距. 的主机程序 你的代理运行时间,Clode Desktop,ChatGPT,Cursor,或一个定制脚本 广告一个可调用工具的列表. 模型在决定需要采取行动时,发出一个结构化有效载荷,命名工具及其论点. 接待者分析了这些工具, 运行了工具, 循环持续到模型决定不再需要电话.

作为OpenAI的"功能"参数,该合同的第一版本于2023年6月发行.`tool_use`双胞胎添加了`functionDeclarations`几个月后. 每个提供商现在都会暴露相同的形状:一个JSON-Schema-typed工具列表,一个JSON-payload工具调用.模型文本协议 (2024年11月) 将合同概括,因此每个模型都会有一个工具注册表.A2A (2026年4月,v1.0) 为代理到代理委托层构成相同的原始.

其他所有在13阶段的东西都是一个精简.

## 概念

### 步骤一:描述

机器主将每个工具声明为三个字段.

- **Name.**机器可读的稳定识别器.`get_weather`没有"天气问题".
- **Description.**简单一段自然语言简单. "当用户询问特定城市的当前状况时,请使用.不要用于历史数据".
- **Input schema.**描述工具的参数的JSON Schema对象 (2020-12年草案).

现代提供商将这些声明串行到系统提示中,使用特定供应商的模板,因此您作为调用者只处理结构化表格.

### 第二步:决定

鉴于用户的信息和可用的工具,模型选择了三个行为之一.

1. **Answer directly**没有工具调用.
2. **Call one or more tools.**发射结构化调用对象.`parallel_tool_calls: true`(默认在OpenAI和双子座,选择在人类) 模型可以发出一次多次电话.
3. **Refuse.**严格模式结构化输出可以产生一个打字的`refusal`封锁而不是打电话.

工具调用用量有三个稳定的字段:一个调用`id`作为一个工具`name`并且一个JSON`arguments`作为一个对象,这个ID存在,所以主机可以将后来的结果与特定的呼叫相关联,这在并行呼叫出现时是重要的.

### 第三步:执行

接待者接收了调用,验证了对声明的方案的参数,并运行执行器. 无效的参数意味着模型幻觉了一个字段或使用错误的类型 很常见的失败模式在弱型模型上. 制作主机在不有效的参数上做三个事情之一:快速失败并将错误传递到模型中,使用限制式解析器修复JSON,或在提示中包含的验证错误中重新尝试模型.

执行器本身是普通代码. Python,TypeScript,一个 shell命令,数据库查询.它产生一个结果,通常是字符串,但可以是任何JSON值或结构化内容块 (文本,图像或资源引用在MCP).结果必须是串化.

### 第四步:观察

主机将工具结果添加到对话中 (如一个 `tool`配合的角色信息`id`) 并且重新调用模型.模型现在将工具输出放在文本中,可以产生最终答案或请求更多呼叫. 这持续到模型停止发出呼叫或主机达到反复数量的安全限制.

### 信任分开了

工具有两种味道,

- **Pure.**只有阅读,确定性,没有副作用.`get_weather`现在`search_docs`现在`get_current_time`值得推测.
- **Consequential.**变化状态,花钱,触摸用户数据.`send_email`现在`delete_file`现在`execute_trade`必须有门.

根据Meta的2026年代理安全"二项规则",一个转换可以结合最多两项:不值得信赖的输入,敏感数据,后续行动.工具界面是通过拒绝呼叫,要求用户确认或升级范围来执行该规则的.查看第13 · 15阶段,全安全章和第14 · 09阶段,查看代理级许可政策.

### 循环的存在

| Context | Who describes | Who decides | Who executes |
|---------|---------------|-------------|--------------|
| Single-turn function calling (OpenAI/Anthropic/Gemini) | App developer | LLM | App developer |
| MCP | MCP server | LLM via MCP client | MCP server |
| A2A | Agent Card publisher | Calling agent | Called agent |
| Web browser (function-calling agent) | Browser extension / WebMCP | LLM | Browser runtime |

列名字改变,结构不会.

### 现在,我们需要一个模型来发射JSON.

"请模型在JSON中回答"是前函数调用模式.在边界模型上,它失败了5到15%的时间,在较小模型上则失败了更多.失败模式包括缺失的支架,后行逗号,幻觉字段和错误类型.然后你需要一个JSON修复通行,重新尝试或限制的解码器.

由于三个原因,本地函数调用更好. 首先,提供商将模型端到端训练在准确的呼叫形状上, 第二,调用用电话的有效载荷位于其自己的协议插槽,而不是在自由文本中,因此工具调用永远不会泄露到用户可见的回复中. 第三,提供商强制执行方案的遵守限制式解码 (OpenAI的严格模式,Anthropic的严格模式).`tool_use`双子座的`responseSchema`) 产量保证验证.

阶段13·02将三个供应商API与其相结合.

### 电路断电器

循环在模型停止发出电话或主机达到最大的转折数时结束.生产主机设置在5到20转之间.此外,你几乎肯定处于模型无法退出的循环中.克劳德代码默认设置为20;OpenAI助理为10;Cursor的代理模式为25.

替代方案是"每六个月就出现一个无限循环"作为"代理人在一夜之间花费400美元的API通话"的后期测试.

阶段14·12涵盖了错误恢复和自我修复的深度;阶段17涵盖了生产率限制.

### 从这里开始,第13阶段

- 课程02至05将提供商级工具调用表面进行抛光.
- 课程06至14将循环整合成MCP.
- 课程15-18保护循环对抗敌对服务器,对抗用户和未认证的远程作者表面.
- 课程19-22将模式扩展到代理人间合作,可观察性,路由和包装.
- 课23将使用每一个原始的整个生态系统.

现在,我们需要在这个过程中做出一些改变.

```figure
tp-tool-loop
```

## 用它

`code/main.py`运行四步循环,没有LLM.一个假的"决策者"函数通过在用户消息上匹配模式来模拟模型;执行器,方案验证器和观察步骤的带是真实的.运行它以查看可打印的中间状态的完整请求/响应编程,然后在后一堂课中替换假决策者以任何真实提供商.

什么要看:

- 工具登记库每工具包含三个字段:名称,描述,方案和执行器引用.
- 验证器是简单的JSON方案子集 (类型,要求,enum,min/max) 仅用 stdlib 写的.
- 生产代理需要这种电路切断器.

## 运送它

这一课产生了`outputs/skill-tool-interface-reviewer.md`鉴于工具定义草案 (名称 + 描述 + 方案 + 执行程序概述),技能审计它是否适合循环:是名称机器稳定,是描述是完整的使用简介,是方案使用JSON Schema 2020-12正确,是纯对后果分类明确.

## 运动

1. 添加第四个工具`code/main.py`呼叫`get_stock_price(ticker)`写下"用户通过ticker请求当前股价时使用.不要用于历史价格或市场总结."运行带并确认新工具的假决策者路线查询.

2. 打破方案验证器,打电话给谁`arguments`执行前确认主机拒绝了该字段.然后通过一个额外的未知的字段进行调用. 决定:主机应该拒绝或忽视? 用安全参数证明您的选择.

3. 加入一个 子,`consequential: true`随着选择后果工具,将标志到需要的注册表输入,并改变循环,打印一个"会与用户确认"行.这是每个生产主机所需的确认门的形状.

4. 在纸上绘制四步循环,上面填写供应商列表,为您最喜欢的客户端 (Claude Desktop,Cursor,ChatGPT或自定义堆).

5. 阅读OpenAI的函数调用指南从上到下. 确定在请求中包含的一个字段,但不是如图所示的四步循环. 解释它添加什么,以及为什么它比必要更方便.

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Tool | "A thing the model can call" | A triple of name + JSON-Schema-typed input + executor function |
| Function calling | "Native tool use" | Provider-level API support for emitting structured tool calls instead of prose |
| Tool call | "The model's request to act" | A JSON payload with `id`, `name`, `arguments` emitted by the model |
| Tool result | "What the tool returned" | The executor's output, wrapped in a `tool` role message with matching id |
| Parallel tool calls | "Many calls at once" | Multiple call objects in one model turn, independent and orderable by id |
| Strict mode | "Guaranteed JSON" | Constrained decoding that forces the model's output to validate against the declared schema |
| Pure tool | "Read-only tool" | No side effects; safe to re-run |
| Consequential tool | "Action tool" | Mutates external state; requires gate, audit, or user confirmation |
| Four-step loop | "The tool-call cycle" | describe → decide → execute → observe |
| Host | "Agent runtime" | The program that holds the tool registry, calls the model, and runs the executor |

## 进一步阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) OpenAI 式工具声明和呼叫形状的常规参考
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)克劳德的`tool_use`现在,`tool_result`区块格式
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) `functionDeclarations`在双子座中,并行调用语义
- [Model Context Protocol — Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28)工具界面的当前无状态,提供商-无知的通用化
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes)每个现代工具API都会使用的方案方言
