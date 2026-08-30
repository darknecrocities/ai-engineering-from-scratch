# 为什么多代理?

> 智能举动不是一个更大的代理,而是更多的代理.

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## 学习目标

- 确定单剂的上限 (文本溢出,混合专业知识,连续瓶) 并解释分为多个代理是什么是正确的举动
- 进行对比 (管道,平行风扇,监督,层次) 调整模式,并选择对特定任务结构的正确模式
- 设计一个多代理系统,具有明确的角色界限,共享状态和通信合同
- 分析多代理复杂性 (延迟,成本,调试难度) 与单代理简单性的折衷

## 问题

在14期中,你建立了一个单个代理.它可以读取文件,运行命令,调用API,并考虑结果.然后你将它指向一个真正的代码库:200个文件,三个语言,依赖基础设施的测试,以及在编写代码之前需要研究外部API.

代理人窒息.不是因为LLM是愚蠢的,而是因为任务超过了一个代理循环可以处理的. 文本窗口充满文件内容. 代理人忘记了40次工具通话之前读到的内容. 他试图同时成为研究人员,编码者和评论员,并且做了这三件事都很差.

这就是单机的天花板.每当任务需要:

- **More context than fits in one window**- 阅读50份文件,超过200万个代币
- **Different expertise at different stages**- 研究需要与代码生成不同的激励
- **Work that can happen in parallel**- - - 既然可以同时读到,为什么要连续读三个文件?

## 概念

### 单机机顶

一个代理是一个循环,一个文本窗口,一个系统提示.

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

三个东西会破裂:

1. **Context saturation**到了30轮,代理已经消耗了15万个文件内容,命令输出和先前推理的代币.

2. **Role confusion**系统提示: "你是一个研究人员,编码者,审查者和测试者",产生一个半研究,半编码的代理,

3. **Sequential bottleneck**经理读取文件A,然后文件B,然后文件C.三次连续LLM电话,三次连续工具执行.

### 多代理解决方案

给每一个代理一个工作,一个背景窗口,一个系统提示调整到这个工作:

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

每个代理都有:
- 专注系统提示 ("你是代码审查员.你的唯一工作是找到错误. ")
- 自己的背景窗口 (不受其他代理人的工作污染)
- 清晰的输出/输入合同 (收到研究说明,输出代码)

### 实际的系统

**Claude Code subagents**- 当克劳德·科德生出一个子女时`Task`孩子的工作是集中的,他回报总结.

**Devin**编程器将工作分成步骤.编程器编写代码.浏览器研究文档.每个都有不同的背景.

**Multi-agent coding teams (SWE-bench)**单代理系统的得分较低. 单代理系统的得分较低.

**ChatGPT Deep Research**- 同时生成多个搜索代理,每个搜索不同的角度,然后合成结果.

### 频谱

多代理不是二进制,而是频谱:

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**Single agent**一个循环,一个提示,适合简单的任务.

**Subagents**孩子们会回报,这就是克劳德·科德所做的.

**Pipeline**对于阶段化工作流程来说,研究 -> 代码 -> 审查 -> 测试.

**Team**经纪人与共享信息公交行,每个人都有角色,一个管弦乐队协调员,当需要不同的技能同时时很好.

**Swarm**没有固定管弦,代理从队列中接收工作,适合高吞吐量并行任务.

### 四种多代理模式

#### 模式1:管道

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

每个代理都会转换数据,传递给人. 简单的推理. 一个阶段的失败阻了其余的阶段.

#### 模式2: 风/风

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

通过平行代理进行分工,然后将结果合并.

#### 模式3:乐团主持人

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

智能管家决定要做什么,委托工作者,并合成结果.

#### 模式4: 同龄人群

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

没有中央调整器,代理人相互沟通,决策来自互动,更难调试,但可以达到许多代理人.

### 什么时候不要使用多剂

复杂性增加了多代理. 代理之间的每一个消息都是潜在的失败点. 调试从"读一场对话"到"追踪五个代理之间的消息".

**Stay single-agent when:**
- 任务适合一个文本窗口 (工作数据的约100k代币以下)
- 你不需要不同的系统提示,
- 顺序执行足够快
- 任务足够简单,把它分为额外的成本

**The complexity cost:**
- 每个代理界限都是一个损失压缩步骤:代理A的全部文本总结为B代理的信息
- 协调逻辑 (谁做什么,什么时候,什么顺序) 是其自身的错误来源
- 延迟增加:N代理意味着N连续LLM调用最小,如果他们需要回来和回来
- 成本乘以:每个代理独立燃烧代币

基本规则:如果一个任务需要不到20个工具调用,并且可以容纳100万个代币,请保持单代理.

```figure
swarm-messages
```

## 建立它

### 第一个步骤: 过度负载的单身代理人

现在,一个单个代理试图做一切. 它有一个巨大的系统提示和一个文本窗口,

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方法的问题:
- 根据研究的过程,它包含了研究说明和代码和先前的推理.
- 系统提示是通用的,不能调整每个阶段.
- 没有什么是平行的.

### 第二步:专业代理人

现在分开,每个代理都能做一个工作:

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专家都有一个专注的提示. 每个人都得到一个清洁的背景窗口,

### 第三步:通过信息协调

给专家们传递明确的信息:

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个代理只收到向其发送的信息,没有环境污染.研究人员阅读的50万份文档,从来没有进入审查者的环境.

### 步骤4:比较

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多代理版本使用更多的总代币 (三代理,三个独立的LLM调用),但每个代理的背景保持清洁.由于系统提示是专业化的,每个阶段的质量都会提高.

## 用它

通过此课程,我们可以重新使用一个提示,`outputs/prompt-multi-agent-decision.md`现在,我们要去.

## 运动

1. 添加第四位专家:一个"测试者"代理,从编码器那里接收代码,并从审查者那里审查反,然后写测试
2. 修改管道,以便审查者可以向编码器发送反,以便进行修改循环 (最大2次)
3. 将序列管道转换为风扇:并行运行研究人员和"要求分析器"代理,然后将其输出结合起来,然后转向编码器

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Swarm | "A hive mind of AI agents" | A set of peer agents with shared state and no fixed leader. Behavior emerges from local interactions. |
| Orchestrator | "The boss agent" | An agent whose tools include spawning and managing other agents. It plans and delegates but may not do the actual work. |
| Coordinator | "The traffic cop" | A non-agent component (often just code, not an LLM) that routes messages between agents based on rules. |
| Consensus | "The agents agree" | A protocol where multiple agents must reach agreement before proceeding. Used when conflicting outputs need resolution. |
| Emergent behavior | "The agents figured it out themselves" | System-level patterns that arise from agent interactions but were not explicitly programmed. Can be useful or harmful. |
| Fan-out / fan-in | "Map-reduce for agents" | Splitting a task across parallel agents (fan-out), then combining their results (fan-in). |
| Message passing | "Agents talk to each other" | The communication mechanism between agents: structured data sent from one agent to another, replacing shared context windows. |

## 进一步阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977)- 调查多代理模式
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155)- 微软的多代理对话框架
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code)- 克劳德·科德如何委托任务
- [CrewAI documentation](https://docs.crewai.com/)-基于角色的多代理框架
