# 基于角色的代理团队 角色,任务,过程

> 两个顶级形状: 团队 (自主,基于角色的协作) 和流动 (事件驱动,定决). CrewAI是2026年参考实现,其文件是直接的: "对于任何准备生产的应用程序,从流动开始".

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 minutes

## 学习目标

- 给CrewAI的四个原始人 (代理,任务,机组人员,过程) 名称,以及每个人的所有物.
- 区分序列,层次和计划的共识过程; 每个工作负载中选择一个.
- 区分Crew (自主角色) 和Flows (事件驱动的确定性),并解释 doc's生产建议.
- 具有电线工具的`@tool`装饰师和`BaseTool`部分类;关于结构化输出与自由文本的理由.
- 给出四种CrewAI内存类型,以及每种类型的效果.
- 执行一个由三名特工组成的工作组 (研究人员,作家,编辑) 制作简报.
- 发现三种 CrewAI失败模式:快速膨胀,管理者-LLM税,脆弱的交付.

## 问题

采用多代理框架的团队都在同一墙上. "自主合作"在演示中听起来很好.然后一个客户提交了一个错误,你需要确定性重播.或者金融问一个LLM路由人员每次运行成本多少.或者在调用时需要知道哪个代理停滞在3AM.

纯粹的DAG回答所有,但失去探索形式一个脑风暴代理需要.

团队的分离是诚实的. 团队为合作,基于角色,探索工作. 活动驱动,代码所有,可审计的生产流动. 同样的框架,两个形状,每表面选择.

## 概念

### 四个原始

机组人员的表面很小,记住这个,剩下的都在配置.

- **Agent.** `role + goal + backstory + tools + (optional) llm`后台故事是承载性的.它塑造音调,判断,当代理停止.工具是代理可以调用的功能 (下面更多).
- **Task.** `description + expected_output + agent + (optional) context + (optional) output_pydantic`能重复使用的工作单位.`expected_output`现在,我们要做什么?`context`列出出输出输入的上游任务. `output_pydantic`它们的结构是结构性的.
- **Crew.**集装箱,拥有列表`agents`列表`tasks`其他`process`其他选择性`memory`其他`verbose`其他`manager_llm`设置
- **Process.**执行策略:序列,层次,共识 (计划). 选择运行的形状.

经纪人不会直接见到彼此,任务是指标人员,机组人员是排序任务,过程决定谁选择下一个任务.这是整个心理模型.

> **Validated against**更新版本可能会更改名称或合并过程类型; 查看[CrewAI Processes docs](https://docs.crewai.com/concepts/processes)在依赖特定的形状之前.

### 序列与等级与共识

- **Sequential.**任务按声明顺序运行.任务N的输出可用为 `context`需要一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个小组,一个,一个小组,一个,一个小组,一个,一个小组,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,一个,
- **Hierarchical.**经理代理 (分别的LLM调用) 间专业人员的路线.`manager_llm`管理员每次选择下一个任务,可以拒绝或重新引导. 使用当你有四个或更多的专家,订单真正取决于之前的输出.
- **Consensus.**现在,我们需要在公开API中进行计划,但目前没有实现. 文件保留了未来基于投票的程序的名字.

层次性增加每轮LLM调用 (经理) 在每个专业调用之上.代币成本可以在五步运行中增加三倍.只需要路由时支付.

### 机组与流动

这就是医生在2026年将会带来的框架.

- **Crew.**基于LLM的自主化.框架在运行时选择形状.好用于:研究,大脑风暴,第一轮草稿,无论路径是答案的一部分.难以重复.难以测试.
- **Flow.**根据事件的图表,你拥有.`@start`标记入口.`@listen(topic)`标志着一个步骤,当另一个步骤发射这个话题时,一个步骤.每个步骤是简单的Python (可以内部调用一个船员).

医生2026年产品建议:从流动开始.`Crew.kickoff()`流动给你审计轨道,机组给你探索.编写,不要挑选.

### 工具集成

给一个代理一个工具的三个方法.

1. **`@tool` decorator.**字符是方案,文件串是 LLM看到的描述.最适合一次性帮助者.

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` subclass.**基于类的工具,有明确的 args 方案,支持异步,重试. 使用工具有状态 (客户端,缓存) 或需要结构化的 args.

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Built-in toolkits.**机组人员运输的第一方适配器: `SerperDevTool`现在`FileReadTool`现在`DirectoryReadTool`现在`CodeInterpreterTool`现在`RagTool`现在`WebsiteSearchTool`只有一个进口.

结构化输出使用Pydantic.`output_pydantic=MyModel`工作人员验证了LLC反应与模型,或者强迫或重新尝试.`expected_output`文字字符串. 免费文本输出对于草稿来说很好;结构化输出是下游流可以消耗的.

### 记忆

机组人员可以同时启动四种内存类型.

> **Validated against**最新版本通过统一的路线`Memory`下面的概念模型仍然存在,但公共类面可能会崩到一个`Memory`在更新的版本中,进入点;[CrewAI memory docs](https://docs.crewai.com/concepts/memory)对于当前的API.

- **Short-term.**只有一个次,就会谈缓冲器,最后就会被抹去.
- **Long-term.**连续运行.存储在向量DB (默认Chroma,可交换). 通过与当前任务相似性获取.
- **Entity.**实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体实体
- **Contextual.**收集时间,在代理需要时,提取相关的记忆,而不是预装.

启动机组使用`memory=True`存储器是CrewAI在较薄的框架中获得的位置之一;纯的LangGraph需要你自己将这些设置都线索.

### 角色为基础的团队适合时

- 只有三到六名代理人,有名字,有合作工作流程,编写,审查,规划,大脑风暴.
- 路由,在 LLM关于下一步的判断是价值的一部分 (层次).
- 任何地方,团队更喜欢阅读.`role + goal + backstory`而不是阅读图表的定义.

### 当他们没有

- 确定性 DAG 具有严格的排序.使用LangGraph (课程 13).图形形是正确的抽象; CrewAI的角色框架是摩擦.
- 连续性连续化提示包括背景故事和之前的输出.
- 单代理循环. 跳过框架;一个代理循环 (课 1) 加上工具注册表更短.

简短版本:CrewAI坐落在"基于角色的合作"角落.

### 依赖性形状

 Python 3.10 到 3.13 使用 `uv`星数:看看[crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)据悉,AWS Bedrock 集成已有记录;供应商基准报告了QA 工作负载上的快速相比,但方法 (数据集,硬件,评估指标) 并没有公布,因此只将框架供应商数字视为指向.

### 在这个模式出现错误的地方

- **Prompt-bloat from backstories.**每个代理和五名代理团队的2000字背景故事在第一次工具调用之前会燃烧背景预算. 保持背景故事在200字以下. 重复代理中短语;不要重复房子风格五次.
- **Manager-LLM token tax.**在一个五任务组上,这是六个LLM电话而不是五个,而管理员电话带有完整的任务列表加上之前的输出. 切换到序列,除非路由取决于输出.
- **Brittle handoffs.**任务N的`expected_output`任务N+1读为`context`法律法师生产了四个. 下游代理的广告.`output_pydantic`在任务N上,所以任务N+1读取输入的对象,而不是自由文本.
- **Crew-as-prod.**无轮装机出货. 输出可变性很高;重播是不可能的; 在调用时不能分辨一个坏跑与一个好的. 装用流.

```figure
ae-crew-vs-flow
```

## 建立它

`code/main.py`执行了两种形状的SDLB版本,加上一个三名特工机组.

形状:

- `Agent`现在`Task`数据类与CrewAI的表面相匹配.
- `SequentialCrew.kickoff(inputs)`按声明顺序执行任务,按输出线程进行编程.`context`现在,我们要去.
- `HierarchicalCrew.kickoff(topic)`总经理将每次选出下一个专家,
- `Flow`随着`@start`其他`@listen(topic)`装饰器,一个小事件循环,一个痕迹.
- `tool(name)`装饰师反映了CrewAI的`@tool`它们的形状.
- `Memory`随着`short_term`现在`long_term`现在`entity`商店; 嘲笑的相似性使用了 numpy.
- 假的LLM响应是硬码字符串,关键字角色加输入前. 没有网络. 确定性.

具体演示:研究人员,作家,编辑团队制作"代理工程2026"的简报.研究人员拉出 (嘲笑) 来源.作家草案.编辑紧缩.同一个团队通过流来显示确定性形状.

运行它:

```bash
python3 code/main.py
```

后续人员线路输出`context`管理者选择 (研究人员,作家,编辑,然后"完成"),流动运行相同的三个步骤,有明确的主题 (`researched`现在`drafted`现在`edited`),工具调用通过`@tool`长期记忆,在两次击中幸存下来.

机组人员的踪迹是流动的,管理者原则上可以重新订购. 流量痕迹是固定的. 这种选择是教训.

## 用它

- **CrewAI Flow**尽管流量只是一个要求`Crew.kickoff()`流量给出了审计界限.
- **CrewAI Crew (Sequential)**对于清晰的合作工作,特别是第一份草案和审查循环.
- **CrewAI Crew (Hierarchical)**路由取决于输出,并且您有四名或更多的专家.
- **LangGraph**对于明确状态机器,持久简历,严格的排序.
- **AutoGen v0.4**(课 14) 演员模型同步和故障隔离.
- **OpenAI Agents SDK**(课 16) 对于使用手柄和护的OpenAI第一产品.
- **Claude Agent SDK**(课 17) 对于用品,包括用品和会议店.

## 运送它

`outputs/skill-crew-or-flow.md`对于一个任务,选出 Crew vs Flow,并设置最小的实现. 硬拒绝了 Crew-without-backstory,Flow-without-explicit-topics,有不到三位专家的层次性.

## 陷

- **Backstory as flavor.**测试每位代理的3种变体,变体是真实的.
- **Skipping `expected_output`.**没有每项任务的合同,下游任务都会接收到 LLM所产生的任何东西.
- **Memory always-on.**长期写每一个运行. 矢量DB增长. 检索变得杂. 范围写到事实持续的任务.
- **Manager prompt drift.**如果路由变得奇怪,放下语音模式,然后读.
- **Tool side effects in Crews.**邮件,删除,支付属于流动步骤,从来没有一个船员工具.

## 运动

1. 转换序列机组成流量,计算变化量下降的触摸点,注意可读性下降的地方.
2. 通过检查检索,检查对实体的检索.
3. 执行一个层次性进程,管理者拒绝向编辑提供路线,直到编辑输出至少有三个段落.
4. 电线`BaseTool`为了在网上搜索的子类.`@tool`装饰品版本.
5. 加入`output_pydantic=Brief`编辑任务,在哪里`Brief`没有`title`现在`summary`现在`sections`让编写任务输出错误的JSON一次;验证CrewAI在追踪中重新尝试行为.
6. 读一读CrewAI的文件介绍,把玩具带到真实中.`crewai`什么保证是SDLB版本错过的?
7. 导向代理运营或长 (课 24) 进行真正的运行.

## 关键词

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "Persona" | Role + goal + backstory + tools |
| Task | "Unit of work" | Description + expected output + assignee + optional structured output |
| Crew | "Agent team" | Container for Agents + Tasks + Process |
| Process | "Execution strategy" | Sequential / Hierarchical / Consensus (planned) |
| Flow | "Deterministic workflow" | Event-driven, code-owned, testable |
| Backstory | "Persona prompt" | Tone and judgment shaper for the Agent |
| `@tool` | "Function tool" | Decorator that turns a function into a tool the Agent can call |
| `BaseTool` | "Class tool" | Class-based tool with args schema, retries, async support |
| Entity memory | "Per-entity facts" | Memory scoped to a customer / account / issue |
| Long-term memory | "Cross-run memory" | Vector-backed memory that survives between kickoffs |
| Contextual memory | "Just-in-time retrieval" | Memory pulled at the moment the Agent needs it |
| Manager LLM | "Router agent" | Extra LLM in Hierarchical process that picks the next task |
| `expected_output` | "Task contract" | String that tells the Agent (and audit) what shape to return |

## 进一步阅读

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction):概念和建议的生产路径
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows)活动驱动的形状`@start`现在`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)其他`@tool`现在`BaseTool`集成工具包
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory)短期,长期,实体,文本性
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)什么时候多剂帮助,什么时候不帮助
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)另一个国家机器
