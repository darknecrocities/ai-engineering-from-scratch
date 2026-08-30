# 状态图管弦乐  持久执行和检查点

> 代理是一个状态机;节点是函数;边缘是过渡;状态是每个节点后的检查点.在最后一个成功检查点中恢复任何失败. 兰格格拉夫是2026年低级状态调整模型的参考.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## 学习目标

- 描述兰格拉夫的核心模型:状态机,具有打字状态,函数节点,条件边缘和节点后检查点.
- 文件强调的四个功能:持久执行,流媒体,人在循环,全面的内存.
- 解释LangGraph支持的三个管弦乐拓:监督,同行 (群) 和层次 (嵌套子图).
- 实现一个具有输入状态,条件边缘和检查点/恢复周期的 stdlib状态图.

## 问题

代理人和工作流程都有一个问题:当40步运行在38步失败时,你想从38步开始,而不是重新开始.二级状态模型让运营商在一个库中重新尝试,该库假设新的运行.

兰格拉夫的设计答案:状态是一个第一类类的类型对象,突变是明确的,并且检查点在每个节点之后仍然存在.`load_state(session_id)`给我打电话.

## 概念

### 图表

图表由:

- **State type.**一个打字的定律 (或皮达因模型),每个节点都会读取和变异.
- **Nodes.**纯功能的`(state) -> state_update`更新后将合并到状态.
- **Edges.**节点之间的条件或直接过渡.
- **Entry and exit.** `START`其他`END`卫兵节点标记了边界.

举个例子:一个代理人`classify`现在`refund`现在`bug`现在`sales`现在`done`节点 作为图的路由工作流程.

### 持续执行

每个节点返回后,运行时间将状态串行并将其写入一个检查点 (SQLite,Postgres,Redis,定制).在步骤N中失败时,运行时间可以`resume(session_id)`接下来从步骤N+1进行精确状态.

拉格格拉夫文件明确强调生产用户在哪里重要:克拉纳,Uber,J.P.摩根.

### 流媒体

每个节点都能产生部分输出. 图表向调用者传输每个节点-delta事件,以便随着图表运行 UI 更新.

### 轮中的人

检查和修改节点之间的状态. 实现:在关键节点之前暂停,向人类表现状态,接受修改,恢复. 检查点使这很容易,因为状态已经串行.

### 记忆

短期 (运行中对话历史状态) 和长期 (通过检查点加上单独的长期存储器持续的跨运行). 兰格拉夫通过工具与外部内存系统 (Mem0,定制) 集成.

### 三种拓

1. **Supervisor.**专业的子管. `create_supervisor()`在`langgraph-supervisor`(尽管2026年兰格链团队建议通过直接调用工具来进行更多的语境控制).
2. **Swarm / peer-to-peer.**代理人直接通过共享工具表面交付.
3. **Hierarchical.**监管部门管理子监管部门,作为嵌套子图.

### 在这个模式出现错误的地方

- **Checkpoints too small.**只有检查对话转换, 工具状态和记忆写不可回收. 完整状态必须串行.
- **Non-deterministic nodes.**简历假设节点输入产生相同状态更新. 随机种子,墙钟,外部API必须捕获.
- **Over-use of conditional edges.**图表的每个边缘都是条件的,这是一个无法推理的状态机器.

```figure
langgraph-state
```

## 建立它

`code/main.py`执行一个 stdlib 状态图:

- `State`一个字符号的字符号`messages`现在`step`现在`route`现在`output`现在`human_approval`现在,我们要去.
- `Node`可调用状态检查和返回更新命令.
- `StateGraph`节点+边缘+条件边缘+运行+续航.
- `SQLiteCheckpointer`将每个节点后的状态串行;`load(session_id)`恢复.
- 展示图:分类 -> 分类(退款 / 错误 / 销售) -> 人类门 -> 发送.

运行它:

```
python3 code/main.py
```

后续的痕迹显示,第一次跑步失败了,

## 用它

- **LangGraph**参考,生产准备. 使用`create_react_agent`现在`create_supervisor`没有任何可能的图表.
- **AutoGen v0.4**演员模式替代性高竞争性场景.
- **Claude Agent SDK**管理带有内置的会议商店.
- **Custom**当你需要对状态形状或检查点后端的确切控制时.

## 运送它

`outputs/skill-state-graph.md`在任何目标运行时间内生成一个长图形状态图,

## 运动

1. 添加一个条件边缘从`classify`为了`end`继续运行,然后再追踪人类的设置.`route`通过手动.
2. 换一个真实的SQLite检查点,按步骤测量序列化.
3. 实现平行边缘:两个节点同时运行,通过定制减速器合并.
4. 阅读`langgraph-supervisor`转移玩具到`create_supervisor`比较了痕迹的形状.
5. 添加流量:每一个节点在运行时都会产生部分状态.

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| State graph | "Agent as state machine" | Typed state + nodes + edges + reducers |
| Checkpointer | "Persistence backend" | Serializes state after every node; enables resume |
| Reducer | "State merger" | Function that combines current state with a node's update |
| Conditional edge | "Branch" | Edge chosen by a function of state |
| Subgraph | "Nested graph" | A graph used as a node inside another graph |
| Durable execution | "Resume from failure" | Restart at the last successful node with exact state |
| Supervisor | "Router LLM" | Central dispatcher for specialist subagents |
| Swarm | "P2P agents" | Agents hand off via shared tools; no central router |

## 进一步阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)参考文件
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/)监督模式API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)演员模式替代
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)会议店和副行
