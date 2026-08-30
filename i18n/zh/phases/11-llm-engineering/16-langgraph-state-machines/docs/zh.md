# 代理国机器 图表,节点,检查站

> 通过手写的 ReAct 循环是`while True`作为一个明确的图表,写的循环是你可以检查点,打断,分支,时间旅行. 代理没有改变. 环绕它有.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 minutes

## 问题

运输一个调用函数的代理.它工作了三次,然后发生了一些问题:模型尝试一个返回500的工具,用户在任务中改变了想法,或者代理决定退还一个订单,没有人签署.`while True:`没有子. 你不能暂停它,你不能转它,你不能分分成"如果模型选择了另一种工具".

接下来的步骤是显而易见的. 代理已经是一个状态机 系统提示加上消息历史加上等待工具调用加上下一步行动. 让状态机明确:节点为"模型认为","工具运行","人批准",和边缘为它们之间的条件过渡. 一旦图表明确,该带将获得四件事免费:检查点 (节省步骤之间的状态),中断 (人类的暂停),流 (流代币和中间事件),以及时间旅行 (回返以前的状态并尝试不同的分支).

这种抽象的参考实现是LangGraph.它不是一个代理框架,在LangChain意义上 ("这里有一个代理执行者,好运").它是一个图表运行时间,具有一流状态,一流的持久性和一流的中断.代理循环是你绘制的东西,而不是你手写的东西.

## 概念

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

`StateGraph`有三个东西.

1. **State.**输入式命令 (TypedDict或Pydantic模型) 通过图表流动.每个节点都收到完整状态并返回部分更新,LangGraph将其通过每个字段的 *reducer* 合并`operator.add`对于应该积累的列表,默认上重写.
2. **Nodes.** Python 函数`state -> partial_state`每一步都是一个分别的步骤: "调用模型", "运行工具", "总结".
3. **Edges.**节点之间的过渡.静态边缘将移动到一个地方. 条件边缘将接收路由器函数`state -> next_node_name`图表可以分为模型输出.

编译将拓链接,附加一个检查点 (可选但对于生产至关重要),并返回一个可运行的.您使用一个初始状态和一个`thread_id`每一步执行都会有一个关键的检查点`(thread_id, checkpoint_id)`现在,我们要去.

### 它们是四大超级大国.

**Checkpointing.**每个节点过渡都会将新状态写入一个存储器 (在内存中进行测试,Postgres/Redis/SQLite为 prod).再重复,再用相同的方式调用图表.`thread_id`图表从停留的地方恢复.

**Interrupts.**标记一个节点`interrupt_before=["human_review"]`您的API会以"等待批准"回复用户. 随后请求相同的`thread_id`随着`Command(resume=...)`恢复执行.

**Streaming.** `graph.stream(state, mode="updates")`随着这些事件,`mode="messages"`通过模特节点中流动LLM代币. `mode="values"`您可以选择在用户界面中出现什么.

**Time-travel.** `graph.get_state_history(thread_id)`返回检查站的全部日志.`checkpoint_id`为了`graph.invoke`对于调试 ("如果模型选择了工具B?") 和重播生产痕迹的回归测试来说,

### 减少者是个问题

每个状态字段都有减小器.大多数默认值都很好一个新的值覆盖了旧值.但消息列表需要`operator.add`通过减速器将其更新合并.如果两个节点都更新`messages`你忘了了这个`Annotated[list, add_messages]`减速器是图书馆唯一的微妙东西;把它做得好,剩下的都是编曲.

### 通过4个节点的 ReAct图

生产 ReAct 代理是四个节点和两个边缘:

1. `agent`将当前消息历史记录传递给LLM. 返回助理消息 (可能包含工具_调用).
2. `tools`执行最后一个助理消息中的任何工具_调用,将工具结果添加为工具消息.
3. 一个条件边缘`agent`航线到`tools`如果最后一个消息有工具_调用,否则`END`现在,我们要去.
4. 一个静态边缘`tools`回到`agent`现在,我们要去.

您可以在约40行代码中获得全 ReAct循环 (思维 → 行动 → 观察 → 思考 → ...) 通过检查点,中断和流.

### 状态图与发送 (预测)

`Send(node_name, state)`让节点发送平行子图. 举个例子:代理决定同时查询三个检索器. 每个`Send`通过状态减小器,它们的输出融合. 这就是LangGraph在没有线程原始的情况下表达了乐队员-工作者模式.

### 字幕

编译图可以是另一个图中的节点. 外面图看到单个节点;内部图有自己的状态和检查点.这就是团队构建监督员工代理的方式:监督员工图将用户意图导向每个域名的员工子图.

```figure
l5-state-graph-ledger
```

## 建立它

### 步骤1:状态和节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages`错误的原因是,它是最常见的LangGraph错误.

### 步骤2:用线程运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每次更新都是一个命令`{node_name: state_delta}`你的前端可以向用户界面传输这些,以便用户看到"代理正在考虑...打电话搜索_网...得到结果...回复".

### 步骤3:添加一个人-在循环中中断

标记一个节点,以便执行停止运行之前.

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态,检查点和线程都在中断期间持续存在. 除了执行时,没有任何东西被记住.

### 步骤4:调试时间旅行

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

通过`None`通过一个值,在恢复之前将其添加到该点状态的更新中. 这就是你在没有重新运行整个对话的情况下重复一个坏代理运行的方式.

### 步骤5:换取检查点进行生产

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

石,雷迪斯和后生已经出货了.`MemorySaver`任何持续在重启过程中都需要真正的商店.

## 技能

> 你把代理作为图形,而不是作为图形.`while True`子,子.

在你拿到兰格拉夫之前,做一个60秒的设计:

1. **Name the nodes.**任何单独的决定或副作用都是节点. "代理认为", "工具运行", "评论员批准", "响应流".如果你不能列出它们,任务还没有代理形状.
2. **Declare the state.**单词单词,每一个单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词单词`messages`提升任务特定领域 (一个工作`plan`其他`budget`计数器`retrieved_docs`列表) 到最高水平.
3. **Draw the edges.**只有下一步取决于模型输出.每个条件边缘需要一个具有命名分支的路由器函数.
4. **Choose a checkpointer up front.** `MemorySaver`对于测试, Postgres/Redis/SQLite. 没有一个 没有检查点意味着没有简历,没有中断,没有时间旅行.
5. **Decide interrupts before tools run, not after.**通过将边缘进入一个副作用节点,以便您可以在损害之前取消;验证将在模型边缘取消,
6. **Stream by default.** `mode="updates"`对于UI,`mode="messages"`对于模型节点内部的代币级流,`mode="values"`在评估期间,

拒绝运送没有检查点的LangGraph代理.拒绝运送中断后的代理.拒绝运送一个`messages`没有字段`add_messages`作为其减小剂.

## 运动

1. **Easy.**通过计算器工具和网页搜索工具实现上述四节点 ReAct 图.`list(app.get_state_history(config))`返回至少四个检查站,进行两轮对话.
2. **Medium.**添加一个`planner`之前运行的节点`agent`编写一个结构化`plan: list[str]`现在,我在美国.`agent`测试中失败,如果`plan`检查点简历 (错误减小器) 上丢失.
3. **Hard.**建立一个监督图,该图将三个子图之间的路线 (`researcher`现在`writer`现在`reviewer`) 使用`Send`每个子图都有自己的状态和检查点.`interrupt_before=["writer"]`确认从前一个检查点的时间旅行只运行了叉子分支.

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| StateGraph | "The LangGraph graph" | The builder object you add nodes and edges to before compile. |
| Reducer | "How the field merges" | A function `(old, new) -> merged` applied when a node returns an update for that field; default is overwrite, `add_messages` appends. |
| Thread | "A conversation ID" | A `thread_id` string that scopes all checkpoints for one session. |
| Checkpoint | "A paused state" | A persisted snapshot of the full graph state after a node transition, keyed on `(thread_id, checkpoint_id)`. |
| Interrupt | "Pause for a human" | `interrupt_before` / `interrupt_after` stop execution at a node boundary; resume with `Command(resume=...)`. |
| Time-travel | "Fork from a prior step" | `graph.invoke(None, config_with_old_checkpoint_id)` replays from that checkpoint forward. |
| Send | "Parallel subgraph dispatch" | A constructor a node can return to spawn N parallel executions of a target node. |
| Subgraph | "A compiled graph as a node" | A compiled StateGraph used as a node in another graph; preserves its own state scope. |

## 进一步阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)对国家图表,减小器,检查点和中断的常规参考.
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/)本课中使用的心理模型,直接从源头.
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) Postgres/SQLite/Redis 商店,检查点名字空间和线程ID的细节.
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) `interrupt_before`现在`interrupt_after`现在`Command(resume=...)`它们是"化"的.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629)每一个LangGraph代理所实施的模式;阅读它,以推理后果理性.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) 什么图形 (链,路由器,管弦工作者,评价者优化器) 首选,何时.
- 阶段11 · 09 (函数调用) 每一个LangGraph代理节点重复使用工具调用原始.
- 11 · 14阶段 (模式文本协议) 外部工具发现,将其插入到一个LangGraph`ToolNode`通过MCP适配器.
- 阶段11 · 17 (代理框架交易) 什么时候选择LangGraph而不是CrewAI,AutoGen或Agno.
