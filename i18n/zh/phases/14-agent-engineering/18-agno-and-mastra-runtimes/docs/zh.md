# 快速实例化和类型工作流程

> 生产代理运行时间优化了原型框架忽略的内容:实例化成本,打字工作流表面和服务准备的后端. 2026 配对:Agno (Python) 旨在实现微秒代理实例化和无状态FastAPI后端.Mastra 将代理,工具,工作流程,统一模型路由和复合存储放在Vercel AI SDK基板上.

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## 学习目标

- 确定阿格诺的性能目标,以及它们什么时候重要.
- 命名Mastra的三个原始物件 代理,工具,工作流程 和支持的服务器适配器.
- 解释为什么无状态的FastAPI后端是AgnO生产路径的建议.
- 选择给定的堆 (Python-first vs TypeScript-first).

## 问题

拉格格拉夫,自动生成,CrewAI是框架重的.想要"只需要代理循环,快速,在我的运行时间"的团队可以使用Agno (Python) 或Mastra (TypeScript).这两者都以原始的速度和更紧密的适应环境堆来交易一些框架所有的原始.

## 概念

### 果

- 之前是Python运行时间,
- "没有图形,链条,或复杂的模式,
- 根据其文件的性能目标: ~ 2μs 代理实例化, ~ 3.75 KiB 存储量每代理, ~ 23 个模型提供商.
- 制作路径:无状态的会话缩写 FastAPI 后端. 每个请求都启动一个新的代理;会话状态在DB中.
- 产生的多模 (文字,图像,音频,视频,文件) 和代理RAG.

速度目标在每秒有数千个短暂的代理时重要 (聊天风扇,评估管道),而当一个代理运行10分钟时,它们更不重要.

### 马斯特拉

- 基于Vercel AI SDK的TypeScript.
- 三个原始:**Agents**现在**Tools**子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子,子**Workflows**现在,我们要去.
- 统一型路由器  3,300+ 型号在 94 个供应商中 (2026 年 3 月).
- 复合存储:内存,工作流程,可观测到不同的后台;ClickHouse建议以实现规模观测.
-  Apache 2.0 版本`ee/`源可用企业许可证下载的目录.
- 服务器适配器用于Express,Hono,Fastify,Koa;第一级Next.js和Astro集成.
- 导航Mastra Studio (本地主机:4111) 进行调试.
- 据悉,GitHub的数据量超过22万,每周每小时下载量超过300万,

### 定位

他们都不想成为兰格拉夫.

- **Language fit.**首先为Python团队提供AgnO,而TypeScript则为Mastra.
- **Runtime ergonomics.**亚格诺=近零的通用费用;马斯特拉=与Vercel生态系统集成.
- **Observability.**两者都与Langfuse/Phoenix/Opik (课 24) 结合起来,但Mastra Studio是第一方.

### 选出每一个

- **Agno**Python后端,许多短暂的代理,强大的性能要求,FastAPI商店.
- **Mastra** 类型脚本后台,下一个.js / Vercel部署,统一的多提供商模型路由,Zod类型工具.
- **LangGraph**长期状态和明确的图形推理比原始速度更重要.
- **OpenAI / Claude Agent SDK**当你想要提供商的生产形状时 (课程1617).

### 在这个模式出现错误的地方

- **Perf-for-perf's-sake.**选择Agno因为2μs听起来很好,当工作负载是每次要求一个缓慢的代理调用.
- **Ecosystem lock-in.**马斯特拉的Vercel味道整合是Vercel的加倍,其他地方是负的.
- **Enterprise license confusion.**马斯特拉的`ee/`如果您打算叉,请阅读许可证.

```figure
wb-runtime-spawn
```

## 建立它

没有单个代码文物可以对这两个框架进行正义.`code/main.py`对于一款横边玩具:至少执行两次 (一次Agnō形,一次Mastra形) 的"运行代理,输出流,持续会议"流程.

运行它:

```
python3 code/main.py
```

两种结构不同但功能相等的痕迹.

## 用它

- **Agno** Python后端需要速度和FastAPI形状.
- **Mastra** 类型Script后台与许多提供商和工作流原始.
- 两艘船都能使用第一方可观测,

## 运送它

`outputs/skill-runtime-picker.md`根据堆,延迟预算和运营形状,选择Agno,Mastra,LangGraph或提供商SDK.

## 运动

1. 读出阿格诺的文件,将Sdlib ReAct循环 (课1) 转移到阿格诺.
2. 阅读Mastra的文件.将相同的循环移植到Mastra.工具打字 (Zod vs.什么都没有) 发生了什么变化?
3. 测量代理实时延迟. 亚格诺的2μs对你的工作负载有什么关系?
4. 如果您在Python中运行CrewAI,如果您搬到Agno,会有什么问题?
5. 阅读马斯特拉的书`ee/`什么限制会影响一个开源叉子?

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agno | "Fast Python agents" | Stateless session-scoped agent runtime |
| Mastra | "TypeScript agents on Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "Multi-provider access" | Single client for 3,300+ models across 94 providers |
| Composite storage | "Multiple backends" | Memory/workflows/observability each to a different store |
| Mastra Studio | "Local debugger" | localhost:4111 UI for introspecting agents |
| Source-available | "Not OSS" | License permits source reading but restricts commercial use |

## 进一步阅读

- [Agno Agent Framework docs](https://www.agno.com/agent-framework)绩效目标,FastAPI集成
- [Mastra docs](https://mastra.ai/docs)原始设备,服务器适配器,路由器模型
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)国家图的替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/)Mastra集成所引用的可观性比较
