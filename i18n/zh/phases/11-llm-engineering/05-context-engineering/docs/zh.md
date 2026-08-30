# 文本工程:Windows,预算,内存和检索

> 提示工程是一个子集.语境工程是整个游戏.提示是你输入的字符串.背景是所有进入模型窗口的东西:系统说明,检索的文件,工具定义,对话历史,几次示例,以及提示本身.2026年最好的人工智能工程师是语境工程师.他们决定什么进去,什么留下,以及什么顺序.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10 (LLMs from Scratch), Phase 11 Lesson 01-02
**Time:** ~90 minutes
**Related:**阶段11 · 15 (即时缓存) 缓存友好的布局是文本工程的延伸. 5 · 28 (长文本评估) 阶段是如何使用NIAH/RULER测量中途丢失.

## 学习目标

- 计算所有语境窗口组件 (系统提示,工具,历史记录,检索文件,生成头空间) 的代币预算
- 实现文本窗口管理策略:截图,总结和对话历史滑动窗口
- 优先考虑和排序文本组件,以最大限度地使模型关注最相关的信息
- 建立一个基于查询类型和可用窗口空间的语境组装器,

## 问题

克劳德·奥普斯 4.7 有200K的代币窗口 (1M在beta). GPT-5 有400K. 双子座 3 Pro 有2M. Llama 4 声称10M. 这些数字听起来很大,直到你填充它们.

系统提示:500个代码. 50个工具的工具定义:8,000个代码. 获取的文档:4,000个代码. 对话历史 (10轮):6,000个代码. 当前用户查询:200个代码. 代码预算 (最大输出):4,000个代码. 总数: 22,700个代码. 这仅占128K窗口的18%.

但注意力并不是按照背景长度进行线性扩展. 一种拥有128K语境代币的模型在尼拉变压器中支付了四方形注意力成本 (O(n^2),尽管大多数生产模型使用高效的注意力变体. 更重要的是,检索的精度会降低. 模型在长文本中难以找到信息. 等人研究 (2023) 显示,在长文本开始和结束时,LLM几乎完全准确地获取信息,但在中间的信息 (文本中的位置为40-70%) 的准确度下降了10-20%. 这种"中途失落"效果因模型而异,但影响了所有当前的建筑.

实际的教训:拥有200K代币并不意味着使用200K代币是有效的.一个精心策划的10K代币背景通常比一个倾倒的100K代币背景更高. 语境工程是在语境窗口内最大化信号-噪音比率的学科.

每个你放入窗口的代币都会取代一个可能带有更多相关信息的代币. 每个无关的工具定义,每一个过时的对话转,每一个不回答问题的检索文本,

## 概念

### 文本窗口是稀缺的资源

设想文本窗口是RAM,而不是磁盘. 它快速,直接访问,但有限. 你不能容纳一切. 你必须选择.

```mermaid
graph TD
    subgraph Window["Context Window (128K tokens)"]
        direction TB
        S["System Prompt\n~500 tokens"] --> T["Tool Definitions\n~2K-8K tokens"]
        T --> R["Retrieved Context\n~2K-10K tokens"]
        R --> H["Conversation History\n~2K-20K tokens"]
        H --> F["Few-shot Examples\n~1K-3K tokens"]
        F --> Q["User Query\n~100-500 tokens"]
        Q --> G["Generation Budget\n~2K-8K tokens"]
    end

    style S fill:#1a1a2e,stroke:#e94560,color:#fff
    style T fill:#1a1a2e,stroke:#0f3460,color:#fff
    style R fill:#1a1a2e,stroke:#ffa500,color:#fff
    style H fill:#1a1a2e,stroke:#51cf66,color:#fff
    style F fill:#1a1a2e,stroke:#9b59b6,color:#fff
    style Q fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#0f3460,color:#fff
```

每个组件都在争夺空间.添加更多工具定义意味着对话历史的空间减少.添加更多获取的文本意味着少量示例的空间.文本工程是分配这个预算的艺术,以最大限度地提高任务性能.

### 迷失在中间

环境工程中最重要的实验发现.模型更好地关注文本的开始和结束时的信息.中间的信息获得了较低的注意力分数,更有可能被忽视.

等人 (2023) 系统地测试了这一点.他们将相关文件放在20个无关的文件中,并测量了答案的准确性.当相关文件是第一个或最后的时,准确性为85-90%.当它在中间时 (20个位置的10个),准确性下降到60-70%.

这对工程业产生了直接影响:

- 首先要把最重要的信息放在第一位 (系统提示,关键指令)
- 关键词: 关键词: 关键词: 关键词:
- 处理环境中部为最低优先级区域
- 如果要在中间包含信息,最后重复关键点

```mermaid
graph LR
    subgraph Attention["Attention Distribution Across Context"]
        direction LR
        P1["Position 0-20%\nHIGH attention\n(system prompt)"]
        P2["Position 20-40%\nMODERATE"]
        P3["Position 40-70%\nLOW attention\n(lost in middle)"]
        P4["Position 70-90%\nMODERATE"]
        P5["Position 90-100%\nHIGH attention\n(current query)"]
    end

    style P1 fill:#51cf66,color:#000
    style P2 fill:#ffa500,color:#000
    style P3 fill:#ff6b6b,color:#fff
    style P4 fill:#ffa500,color:#000
    style P5 fill:#51cf66,color:#000
```

### 文本组件

**System prompt**克劳德代码使用大约6,000个代币用于系统提示,包括工具定义和行为说明.保持紧密.系统提示中的每个字都在每个API调用中重复.

**Tool definitions**每个工具都会添加50-200个代币 (名称,描述,参数方案).每一个50个代币的50个工具在任何对话发生之前,每个代币都会达到7,500个代币.

**Retrieved context**检索的质量直接决定了响应的质量. 检索不良比没有检索更糟糕 - - 它会填满窗口的噪音,并积极误导模型.

**Conversation history**交换时间:每次用户消息和助理响应. 交谈长度随着交谈时间的推移而增长. 每次交换时50次交谈的200个代币相当于10,000个历史代币. 大多数是与当前查询无关的.

**Few-shot examples**输入/输出对,证明所需的行为. 两到三个精心选择的例子通常会提高输出质量,超过数千个指令代币.

**Generation budget**模型的代币为模型的响应保留. 如果填充到容量的窗口,模型就没有答案的空间. 储备至少2000至4,000代币用于生成.

### 环境压缩策略

**History summarization**总结对话的时间: "我们讨论了X,决定了Y,用户想要Z"在100代币中取代了2000代币的10次转换. 运行总结时历史超过门 (例如5000代币).

**Relevance filtering**您的文件在一个门以下. 如果您检索了10个块,但只有3个是相关的,则丢弃其他7.比10个中等的部分更好有3个非常相关的块.

**Tool pruning**编码问题不需要日历工具.编程问题不需要文件系统工具. 这可以将工具定义从8,000个代币降至1,000个.

**Recursive summarization**首先要总结每个部分,然后要总结.一个50页的文档变成一个500个代币的摘要,捕捉到关键点.

### 记忆系统

文本工程跨越了三个时间视野.

**Short-term memory**直接存储在文本窗口中.随着每次转折而增长.通过总结和缩短来管理.

**Long-term memory**文件:在对话中持续存在的事实和偏好. "用户更喜欢TypeScript". "项目使用PostgreSQL."存储在数据库中,在会议开始时获取.Claude Code将这些存储在CLAUDE.md文件中.ChatGPT将其存储在其内存功能中.

**Episodic memory**通过"Auth"模块调试了类似的问题. 存储为嵌入式,当当前对话与过去事件匹配时获取.

```mermaid
graph TD
    subgraph Memory["Memory Architecture"]
        direction TB
        STM["Short-term Memory\n(current conversation)\nDirect in context window"]
        LTM["Long-term Memory\n(facts, preferences)\nDB -> retrieved on session start"]
        EM["Episodic Memory\n(past interactions)\nEmbeddings -> retrieved on similarity"]
    end

    Q["Current Query"] --> STM
    Q --> LTM
    Q --> EM

    STM --> CW["Context Window"]
    LTM --> CW
    EM --> CW

    style STM fill:#1a1a2e,stroke:#51cf66,color:#fff
    style LTM fill:#1a1a2e,stroke:#0f3460,color:#fff
    style EM fill:#1a1a2e,stroke:#e94560,color:#fff
    style CW fill:#1a1a2e,stroke:#ffa500,color:#fff
```

### 动态语境组件

关键见解:不同的查询需要不同的语境.静态系统提示 +静态工具 +静态历史是浪费的.最好的系统每一个查询动态组装语境.

1. 分类查询意图
2. 选择相关工具 (不是所有工具)
3. 检索相关文件 (不是固定集)
4. 包含相关历史转折 (不是全部历史)
5. 添加与任务类型相匹配的几次示例
6. 按重要点排序:最先关键,最后重要,中间是可选的

这就是区分一个好的人工智能应用程序和一个伟大的应用程序的原因.

```figure
lost-in-the-middle
```

## 建立它

### 步骤1: 代币计数器

构建一个简单的代币计数 (使用白空分数的近似,因为确切的计数取决于代币表).

```python
import json
import numpy as np
from collections import OrderedDict

def count_tokens(text):
    if not text:
        return 0
    return int(len(text.split()) * 1.3)

def count_tokens_json(obj):
    return count_tokens(json.dumps(obj))
```

### 步骤2: 环境预算管理员

预算管理员会追踪每个组件使用多少代币,并执行限制.

```python
class ContextBudget:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.max_tokens = max_tokens
        self.generation_reserve = generation_reserve
        self.available = max_tokens - generation_reserve
        self.allocations = OrderedDict()

    def allocate(self, component, content, max_tokens=None):
        tokens = count_tokens(content)
        if max_tokens and tokens > max_tokens:
            words = content.split()
            target_words = int(max_tokens / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        used = sum(self.allocations.values())
        if used + tokens > self.available:
            allowed = self.available - used
            if allowed <= 0:
                return None, 0
            words = content.split()
            target_words = int(allowed / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        self.allocations[component] = tokens
        return content, tokens

    def remaining(self):
        used = sum(self.allocations.values())
        return self.available - used

    def utilization(self):
        used = sum(self.allocations.values())
        return used / self.max_tokens

    def report(self):
        total_used = sum(self.allocations.values())
        lines = []
        lines.append(f"Context Budget Report ({self.max_tokens:,} token window)")
        lines.append("-" * 50)
        for component, tokens in self.allocations.items():
            pct = tokens / self.max_tokens * 100
            bar = "#" * int(pct / 2)
            lines.append(f"  {component:<25} {tokens:>6} tokens ({pct:>5.1f}%) {bar}")
        lines.append("-" * 50)
        lines.append(f"  {'Used':<25} {total_used:>6} tokens ({total_used/self.max_tokens*100:.1f}%)")
        lines.append(f"  {'Generation reserve':<25} {self.generation_reserve:>6} tokens")
        lines.append(f"  {'Remaining':<25} {self.remaining():>6} tokens")
        return "\n".join(lines)
```

### 步骤3: 失落的中部重组

实施重组策略:最重要的项目首先和最后,最不重要的项目在中间.

```python
def reorder_lost_in_middle(items, scores):
    paired = sorted(zip(scores, items), reverse=True)
    sorted_items = [item for _, item in paired]

    if len(sorted_items) <= 2:
        return sorted_items

    first_half = sorted_items[::2]
    second_half = sorted_items[1::2]
    second_half.reverse()

    return first_half + second_half

def score_relevance(query, documents):
    query_words = set(query.lower().split())
    scores = []
    for doc in documents:
        doc_words = set(doc.lower().split())
        if not query_words:
            scores.append(0.0)
            continue
        overlap = len(query_words & doc_words) / len(query_words)
        scores.append(round(overlap, 3))
    return scores
```

### 步骤4:对话历史压缩机

总结一下,老话语回归,回归代币预算.

```python
class ConversationManager:
    def __init__(self, max_history_tokens=5000):
        self.turns = []
        self.summaries = []
        self.max_history_tokens = max_history_tokens

    def add_turn(self, role, content):
        self.turns.append({"role": role, "content": content})
        self._compress_if_needed()

    def _compress_if_needed(self):
        total = sum(count_tokens(t["content"]) for t in self.turns)
        if total <= self.max_history_tokens:
            return

        while total > self.max_history_tokens and len(self.turns) > 4:
            old_turns = self.turns[:2]
            summary = self._summarize_turns(old_turns)
            self.summaries.append(summary)
            self.turns = self.turns[2:]
            total = sum(count_tokens(t["content"]) for t in self.turns)

    def _summarize_turns(self, turns):
        parts = []
        for t in turns:
            content = t["content"]
            if len(content) > 100:
                content = content[:100] + "..."
            parts.append(f"{t['role']}: {content}")
        return "Previous: " + " | ".join(parts)

    def get_context(self):
        parts = []
        if self.summaries:
            parts.append("[Conversation Summary]")
            for s in self.summaries:
                parts.append(s)
        parts.append("[Recent Conversation]")
        for t in self.turns:
            parts.append(f"{t['role']}: {t['content']}")
        return "\n".join(parts)

    def token_count(self):
        return count_tokens(self.get_context())
```

### 步骤5:动态工具选择器

仅包括与当前查询相关的工具. 分类意图,然后过.

```python
TOOL_REGISTRY = {
    "read_file": {
        "description": "Read contents of a file",
        "tokens": 120,
        "categories": ["code", "files"],
    },
    "write_file": {
        "description": "Write content to a file",
        "tokens": 150,
        "categories": ["code", "files"],
    },
    "search_code": {
        "description": "Search for patterns in codebase",
        "tokens": 130,
        "categories": ["code"],
    },
    "run_command": {
        "description": "Execute a shell command",
        "tokens": 140,
        "categories": ["code", "system"],
    },
    "create_calendar_event": {
        "description": "Create a new calendar event",
        "tokens": 180,
        "categories": ["calendar"],
    },
    "list_emails": {
        "description": "List recent emails",
        "tokens": 160,
        "categories": ["email"],
    },
    "send_email": {
        "description": "Send an email message",
        "tokens": 200,
        "categories": ["email"],
    },
    "web_search": {
        "description": "Search the web for information",
        "tokens": 140,
        "categories": ["research"],
    },
    "query_database": {
        "description": "Run a SQL query on the database",
        "tokens": 170,
        "categories": ["code", "data"],
    },
    "generate_chart": {
        "description": "Generate a chart from data",
        "tokens": 190,
        "categories": ["data", "visualization"],
    },
}

def classify_intent(query):
    query_lower = query.lower()

    intent_keywords = {
        "code": ["code", "function", "bug", "error", "file", "implement", "refactor", "debug", "test"],
        "calendar": ["meeting", "schedule", "calendar", "appointment", "event"],
        "email": ["email", "mail", "send", "inbox", "message"],
        "research": ["search", "find", "what is", "how does", "explain", "look up"],
        "data": ["data", "query", "database", "chart", "graph", "analytics", "sql"],
    }

    scores = {}
    for intent, keywords in intent_keywords.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return ["code"]

    max_score = max(scores.values())
    return [intent for intent, score in scores.items() if score >= max_score * 0.5]

def select_tools(query, token_budget=2000):
    intents = classify_intent(query)
    relevant = {}
    total_tokens = 0

    for name, tool in TOOL_REGISTRY.items():
        if any(cat in intents for cat in tool["categories"]):
            if total_tokens + tool["tokens"] <= token_budget:
                relevant[name] = tool
                total_tokens += tool["tokens"]

    return relevant, total_tokens
```

### 步骤 6: 完整的环境组合管道

根据查询,动态组装最佳的文本.

```python
class ContextEngine:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.budget = ContextBudget(max_tokens, generation_reserve)
        self.conversation = ConversationManager(max_history_tokens=5000)
        self.system_prompt = (
            "You are a helpful AI assistant. You have access to tools for "
            "code editing, file management, web search, and data analysis. "
            "Use the appropriate tools for each task. Be concise and accurate."
        )
        self.knowledge_base = [
            "Python 3.12 introduced type parameter syntax for generic classes using bracket notation.",
            "The project uses PostgreSQL 16 with pgvector for embedding storage.",
            "Authentication is handled by Supabase Auth with JWT tokens.",
            "The frontend is built with Next.js 15 using the App Router.",
            "API rate limits are set to 100 requests per minute per user.",
            "The deployment pipeline uses GitHub Actions with Docker multi-stage builds.",
            "Test coverage must be above 80% for all new modules.",
            "The codebase follows the repository pattern for data access.",
        ]

    def assemble(self, query):
        self.budget = ContextBudget(self.budget.max_tokens, self.budget.generation_reserve)

        system_content, _ = self.budget.allocate("system_prompt", self.system_prompt, max_tokens=1000)

        tools, tool_tokens = select_tools(query, token_budget=2000)
        tool_text = json.dumps(list(tools.keys()))
        tool_content, _ = self.budget.allocate("tools", tool_text, max_tokens=2000)

        relevance = score_relevance(query, self.knowledge_base)
        threshold = 0.1
        relevant_docs = [
            doc for doc, score in zip(self.knowledge_base, relevance)
            if score >= threshold
        ]

        if relevant_docs:
            doc_scores = [s for s in relevance if s >= threshold]
            reordered = reorder_lost_in_middle(relevant_docs, doc_scores)
            doc_text = "\n".join(reordered)
            doc_content, _ = self.budget.allocate("retrieved_context", doc_text, max_tokens=3000)

        history_text = self.conversation.get_context()
        if history_text.strip():
            history_content, _ = self.budget.allocate("conversation_history", history_text, max_tokens=5000)

        query_content, _ = self.budget.allocate("user_query", query, max_tokens=500)

        return self.budget

    def chat(self, query):
        self.conversation.add_turn("user", query)
        budget = self.assemble(query)
        response = f"[Response to: {query[:50]}...]"
        self.conversation.add_turn("assistant", response)
        return budget


def run_demo():
    print("=" * 60)
    print("  Context Engineering Pipeline Demo")
    print("=" * 60)

    engine = ContextEngine(max_tokens=128000, generation_reserve=4000)

    print("\n--- Query 1: Code task ---")
    budget = engine.chat("Fix the bug in the authentication module where JWT tokens expire too early")
    print(budget.report())

    print("\n--- Query 2: Research task ---")
    budget = engine.chat("What is the best approach for implementing vector search in PostgreSQL?")
    print(budget.report())

    print("\n--- Query 3: After conversation history builds up ---")
    for i in range(8):
        engine.conversation.add_turn("user", f"Follow-up question number {i+1} about the implementation details of the system")
        engine.conversation.add_turn("assistant", f"Here is the response to follow-up {i+1} with technical details about the architecture")

    budget = engine.chat("Now implement the changes we discussed")
    print(budget.report())

    print("\n--- Tool Selection Examples ---")
    test_queries = [
        "Fix the bug in auth.py",
        "Schedule a meeting with the team for Tuesday",
        "Show me the database query performance stats",
        "Search for best practices on error handling",
    ]

    for q in test_queries:
        tools, tokens = select_tools(q)
        intents = classify_intent(q)
        print(f"\n  Query: {q}")
        print(f"  Intents: {intents}")
        print(f"  Tools: {list(tools.keys())} ({tokens} tokens)")

    print("\n--- Lost-in-the-Middle Reordering ---")
    docs = ["Doc A (most relevant)", "Doc B (somewhat relevant)", "Doc C (least relevant)",
            "Doc D (relevant)", "Doc E (moderately relevant)"]
    scores = [0.95, 0.60, 0.20, 0.80, 0.50]
    reordered = reorder_lost_in_middle(docs, scores)
    print(f"  Original order: {docs}")
    print(f"  Scores:         {scores}")
    print(f"  Reordered:      {reordered}")
    print(f"  (Most relevant at start and end, least relevant in middle)")
```

## 用它

### 带管理的环境

克劳德代码使用层次方法管理文本.系统提示包含行为规则和工具定义 (~6K代币).当您打开文件时,其内容被注入为文本.当您搜索时,结果被添加.旧对话转换总结.CLAUDE.md提供长期内存,持续在整个会议中.

关键的工程决定:Claude Code不把整个代码库放入文本中. 它在需求上检索相关文件.

### 动态文本加载

库尔索将整个代码库索引成嵌入式.当你输入查询时,它会使用向量相似性检索最相关的文件和代码块.只有这些块进入文本窗口. 500K 行的代码库被压缩成5到10个最相关的代码块.

现在,我们需要的东西,

### 长期记忆助理

聊天GPT将用户的偏好和事实存储为长期内存.每次对话开始,相关的记忆都会被检索到系统提示中. "用户更喜欢Python"的代价为5个代币,但在对话中保存了数百个重复指令的代币.

### 作为文本工程的RAG

复苏增强的生成是文本工程正式化. 您在查询时检索相关文件,并将它们注入文本窗口. 整个RAG管道-- 碎片化,嵌入,检索,重新排名-- 存在于解决一个问题:将正确的信息放入文本窗口.

## 运送它

这一课产生了`outputs/prompt-context-optimizer.md`-- 复用提示,审计一个语境组装策略并建议优化. 给它提供系统提示,工具数量,平均历史长度和检索策略,

它还产生了`outputs/skill-context-engineering.md`--基于任务类型,背景窗口大小和延迟预算的决策框架.

## 运动

1. 加入一个"代币废物检测器"到 ContextBudget类. 它应该标记使用超过30%的预算的组件,并建议针对每个组件类型的压缩策略 (概括历史,剪切工具,重新排名文件).

2. 实现检索文本的语义排序.如果两个检索文档80%以上相似 (通过词汇重叠或嵌入式相似性),只保留更高分数的文本.测量这项文本的恢复额.

3. 建立一个"语境重播"工具. 给出对话转录,通过语境引擎重播它,并可视化预算分配如何随时变化. 随着时间的推移,绘制每个组件的代币使用情况. 确定文本开始压缩的转折.

4. 实现基于优先级的工具选择器. 取而代之的是,将每个工具分配到当前查询中的相关性分数. 包含工具在下降相关性顺序中,直到工具预算被耗尽. 与 5, 10, 20 和 50 个工具的任务性能进行比较.

5. 建立一个多策略背景压缩机. 实施三种压缩策略 (缩小,总结,取取出关键句子) 并根据20份文件进行比较. 测量压缩比和信息保留之间的权衡 (压缩版本是否仍然包含查询答案?).

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Context window | "How much the model can read" | The maximum number of tokens (input + output) the model processes in a single forward pass -- 400K for GPT-5, 200K (1M beta) for Claude Opus 4.7, 2M for Gemini 3 Pro |
| Context engineering | "Advanced prompt engineering" | The discipline of deciding what goes into the context window, in what order, and at what priority -- encompasses retrieval, compression, tool selection, and memory management |
| Lost-in-the-middle | "Models forget stuff in the middle" | Empirical finding that LLMs attend better to the beginning and end of context, with 10-20% accuracy drop for information placed in the middle |
| Token budget | "How many tokens you have left" | An explicit allocation of context window capacity across components (system prompt, tools, history, retrieval, generation) with per-component limits |
| Dynamic context | "Loading stuff on the fly" | Assembling the context window differently for each query based on intent classification, relevant tool selection, and retrieval results |
| History summarization | "Compressing the conversation" | Replacing verbatim old conversation turns with a concise summary, reducing token cost while preserving key information |
| Tool pruning | "Only including relevant tools" | Classifying query intent and only including tool definitions that match, reducing tool token cost by 60-80% |
| Long-term memory | "Remembering across sessions" | Facts and preferences stored in a database and retrieved at session start -- CLAUDE.md, ChatGPT Memory, and similar systems |
| Episodic memory | "Remembering specific past events" | Past interactions stored as embeddings and retrieved when the current query is similar to a past conversation |
| Generation budget | "Room for the answer" | Tokens reserved for the model's output -- if the context fills the window completely, the model has no room to respond |

## 进一步阅读

- [Liu et al., 2023 -- "Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172)模型在长时间的背景下与信息扎
- [Anthropic's Contextual Retrieval blog post](https://www.anthropic.com/news/contextual-retrieval)-- 如何让人类对文本意识的部分检索进行处理,
- [Simon Willison's "Context Engineering"](https://simonwillison.net/2025/Jun/27/context-engineering/)-- 博客文章命名了该学科,
- [LangChain documentation on RAG](https://python.langchain.com/docs/tutorials/rag/)-- 实际实施以检索增强发电作为文本工程模式
- [Greg Kamradt's Needle in a Haystack test](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)-- 标志显示所有主要模型中存在位置依赖的检索失败
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102)背景长度为什么驱动内存和延迟,以及KV缓存,MQA和GQA如何改变预算计算.
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369)两个期的推断,使得长时间的提示在TTFT中昂贵,但在TPOT中便宜;
- [Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (EMNLP 2023)](https://arxiv.org/abs/2305.13245)通过集成的查询注意力纸, 无损质量, 切断了生产解码器中的KV内存8倍.
