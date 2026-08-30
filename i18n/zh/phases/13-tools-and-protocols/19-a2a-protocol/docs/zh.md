#  代理人至代理人协议

> 现在,我们需要一个代理.  A2A (Agent2Agent) 是一个允许不同框架构建的不透明代理合作的开放协议. 谷歌于2025年4月发布,并于2025年6月捐赠给Linux基金会,并在2026年4月获得了150多个支持者,包括AWS,Cisco,微软,Salesforce,SAP和ServiceNow. 它吸收了IBM的ACP,并增加了AP2支付延长. 这一课讲述了特工卡,任务生命周期,

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## 学习目标

- 区分代理到工具 (MCP) 与代理到代理 (A2A) 使用情况.
- 在 发行代理卡`/.well-known/agent.json`具有技能和终端数据.
- 查看任务生命周期 (提交 → 工作 → 输入要求 → 完成 / 失败 / 取消 / 拒绝).
- 使用部分 (文字,文件,数据) 和文物的消息作为输出.

## 问题

客户服务代理需要将报告编写委托给专业的作家代理.

- 定制RESTAPI,但每次配对都是一次性的.
- 需要两个代理运行相同的框架.
- 没有合适:MCP是用来调用工具,而不是两个代理合作,同时保持每个代理的不透明的内部推理.

A2A填补了空白.它模拟了一个代理向另一个代理发送任务的交互,使用生命周期,消息和文物.所谓的代理内部状态保持不透明.

 A2A 是"让跨框架的代理人相互交谈"协议. 它不取代MCP;这两种协议是互补的.

## 概念

### 代理卡

每个符合A2A的代理人都会在`/.well-known/agent.json`其他:

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现是基于URL的:拿到卡片,学习A2A终端点的URL,列出技能.

### 签署的代理卡 (AP2)

发布者用JWT签署自己的卡;消费者验证.防止伪造.

### 任务生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

客户开始使用`tasks/send`调用代理通过州进行过渡;客户通过SSE或民意调查订阅状态更新.

### 信息和部分

信息包含一个或多个部分:

- `text` 简单内容.
- `file`使用mimeType的64基块.
- `data`输入JSON有效载荷 (为所调用的代理进行结构化输入).

举个例子:

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### 艺术品

输出是艺术品,而不是原始字符串.

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

艺术品可以作为块流传.

### 两项运输义务

1. **JSON-RPC over HTTP.** `/a2a`终端点,请求的POST, 流媒体的SSE.
2. **gRPC.**对于gRPC原生企业环境.

两个结合都具有相同的逻辑信息形状.

### 保持空位

设计原理:调用代理的内部状态不透明.调用者看到任务状态和文物.调用代理的思想链,其工具调用,其子代理委托都看不见.这与MCP不同,工具调用是透明的.

理由:A2A允许竞争对手在不透露内部信息的情况下协作.A2A可以是"调用这个客户服务代理"而不需要调用者学习该代理如何实现服务.

### 时间线

- **2025-04-09.**谷歌宣布A2A.
- **2025-06-23.**捐给Linux基金会.
- **2025-08.**吸收IBM的ACP.
- **2025-09.**扩展AP2 (代理支付) 船舶.
- **2026-04.**版本 1.0 发布了150多个支持组织.

### 与MCP的关系

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | JWT-signed Agent Cards (AP2) |
| Transport | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

许多生产系统都使用MCP,用于工具层,A2A用于协作层.

```figure
a2a-task-lifecycle
```

## 用它

`code/main.py`通过A2A的最小化,研究代理发布卡,编写代理获得A2A的卡.`tasks/send`通过工作 → input_required → working → 完成的转换,并返回文本文物.所有 stdlib; 使用内存运输以关注消息形状.

什么要看:

- 机器人卡的JSON形状.
- 任务 ID 分配和状态过渡.
- 混合型零件的消息.
- 需要输入的分支在任务中.
- 工艺品在完成后返回.

## 运送它

这一课产生了`outputs/skill-a2a-agent-spec.md`由于一个新的代理,该技能应该被其他代理调用, 产生的代理卡JSON,技能方案,和终点蓝图.

## 运动

1. 跑步`code/main.py`追踪任务的整个生命周期,包括调用代理要求澄清的输入暂停.

2. 加入一个签名的代理卡,用HMAC签名卡的正规JSON,写一个验证器,确认它在一个突变的卡上失败.

3. 执行任务流:编写代理通过SSE发射三个增量文物块,调用者积累它们.

4. 设计一个 A2A 代理,将一个 MCP 服务器包裹起来.将每个 MCP 工具映射到一个 A2A 技能.注意交易.

5. 阅读A2A v1.0公告并确定截至2026年4月,没有任何框架实施的唯一功能. (提示:它涉及多跳任务委托).

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | "Agent-to-Agent protocol" | Open protocol for opaque agent collaboration |
| Agent Card | "`.well-known/agent.json`" | Published metadata describing an agent's skills and endpoint |
| Skill | "A callable unit" | A named operation the agent supports (analog to MCP tool) |
| Task | "Unit of delegation" | A work item with a lifecycle and final artifact |
| Message | "Task input" | Carries Parts (text, file, data) |
| Part | "Typed chunk" | `text` / `file` / `data` element of a message |
| Artifact | "Task output" | Named, typed output returned on completion |
| AP2 | "Agent Payments Protocol" | Signed Agent Cards extension for trust and payments |
| Opacity | "Black-box collaboration" | Called agent's internals are hidden from caller |
| Input-required | "Task pause" | Lifecycle state when the agent needs more info |

## 进一步阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/)可нони A2A规范
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A)参考实施和SDK
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) 2025年6月 管理转让
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade)路线图和合作伙伴势头
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) v1.0 发布说明和后退型紧指导
