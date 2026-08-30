# الوكيل الآلات الدولة  الرسومات، العقد، نقاط التفتيش

> حلقة ReAct مكتوبة يدويا هي `while True`نفس الحلقة المكتوبة على الرسم البياني الصريح هي شيء يمكنك التفتيش، والقطع، والفرع، والسفر عبر الزمن.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 minutes

## المشكلة

أنت ترسل وكيل يدعو وظيفة. يعمل لمدة ثلاث دورات، ثم يحدث شيء خاطئ: النموذج يحاول أداة تعيد 500، المستخدم يغير رأيه في منتصف المهمة، أو الوكيل يقرر إعادة طلب دون توقيع بشري.`while True:`لا يمكنك أن تعيقه، لا يمكنك أن تعيد التفريغ، ولا يمكنك أن تتحول إلى "ماذا لو كان النموذج قد اختار الأداة الأخرى". في اللحظة التي ترسلها هذا بعد عرض تجريبي،

الخطوة التالية واضحة بمجرد رؤيتها الوكيل هو بالفعل آلة حالة  نظام استئناف بالإضافة إلى تاريخ الرسائل بالإضافة إلى المنتظرات الأداة مكالمات بالإضافة إلى العمل التالي. اجعل آلة الحالة واضحة: العقدة لـ "النموذج يفكر" "أداة تعمل" "إنسان يوافق" والحواف للانتقالات المشروطة بينها. بمجرد أن يكون الرسم البياني واضحًا ، يحصل الحزام على أربعة أشياء مجانًا: التفتيش (إنقاذ حالة بين الخطوات) ، والقاطع (وقف للإنسان) ، والتشغيل (الرموز التدريبية والأحداث المتوسطة) ، والسفر عبر الزمن (العودة إلى حالة سابقة ومحاولة فرع مختلف).

إن تنفيذ مرجعية لهذا التجريد هو LangGraph. إنه ليس إطار عميل بمعنى LangChain ("هنا AgentExecutor ، حظاً طيباً"). إنه جدول تشغيل مع حالة من الدرجة الأولى ، ومثابرة من الدرجة الأولى ، ومقاطعات من الدرجة الأولى. حلقة العميل هي شيء ترسم ، وليس شيء تكتب به يدك.

## المفهوم

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

أ`StateGraph`لديه ثلاثة أشياء

1. **State.**إشارة مدمجة (TypedDict أو نموذج Pydantic) التي تتدفق عبر الرسم البياني. كل عقدة تتلقى الحالة الكاملة وتعيد تحديثًا جزئيًا ، والذي يدمجها LangGraph باستخدام *reducer* لكل حقل `operator.add`بالنسبة لقوائم يجب أن تتراكم، إعادة كتابة حسب الاختيار.
2. **Nodes.**وظائف Python `state -> partial_state`كل خطوة منفصلة: "تصل النموذج" "تشغيل الأدوات" "تجميع".
3. **Edges.**الانتقال بين العقد. الحواف الدولية تذهب إلى مكان واحد. الحواف الشروطية تأخذ وظيفة الجهاز التوجيه`state -> next_node_name`حتى يمكن أن تتفرق الرسم البياني على النموذج الخارجي.

تقوم بتجميع الرسم البياني. تقوم بتجميع يربط التطبيقات، ويربط نقطة التفتيش (اختيارية ولكن ضرورية للإنتاج) ، ويرجع إمكانية التشغيل. تستدعيها مع حالة أولية و `thread_id`كل خطوة من الإعدام تستمر بمراقبة معلقة`(thread_id, checkpoint_id)`. . .

### القوى العظمى الأربعة

**Checkpointing.**كل انتقال عقد يكتب الحالة الجديدة إلى مخزن (في الذاكرة للتجارب، Postgres/Redis/SQLite للإنتاج). استأنف عن طريق استدعاء الرسم البياني مرة أخرى بنفس `thread_id`الرسم البياني يستمر من حيث توقف

**Interrupts.**قم بتشخيص العقدة`interrupt_before=["human_review"]`وتوقف التنفيذ قبل تشغيل تلك العقدة. الحالة لا تزال. API الخاص بك يستجيب للمستخدم مع "انتظار الموافقة". طلب لاحقا لنفس `thread_id`مع`Command(resume=...)`يستأنف الإعدام

**Streaming.** `graph.stream(state, mode="updates")`يُعطى الدولة ديلتا كما يحدث.`mode="messages"`يُدفق رموز الـ LLM داخل عقدة النموذج. `mode="values"`يُعطى صورًا مفصلة. تختار ما ستظهر في واجهتك التفاعلية.

**Time-travel.** `graph.get_state_history(thread_id)`يعيد سجل المراقبة الكاملة. اجتياز أي سابقة `checkpoint_id`إلى`graph.invoke`و أنت تشرق من تلك النقطة. عظيم للتحليل ("ماذا لو كان النموذج قد اخترت الأداة B بدلا؟") و للاختبارات التراجعة التي تعيد تشغيل آثار الإنتاج.

### القلص هو النقطة

كل حقل حالة لديه خفض. معظم الافتراضات تصلح  قيمة جديدة تغطي القديمة. ولكن القوائم الرسائل تحتاج `operator.add`لذا يتم إضافة رسائل جديدة بدلاً من استبدالها. الحواف المتوازية تجمع تحديثاتها من خلال القلل. إذا تم تحديث كلا العقدين`messages`و نسيتِ`Annotated[list, add_messages]`و الفائز الثاني في الصمت و تخسر نصف الجولة و القلل هو الشيء الوحيد الخفيف في المكتبة

### الرسم البياني ReAct في أربعة عقدات

وكيل ReAct الإنتاج هو أربع عقدين وعضوين:

1. `agent` يدعو الجامعة مع تاريخ الرسالة الحالية. يعيد رسالة المساعد (التي قد تحتوي على tool_calls).
2. `tools` تنفيذ أي tool_calls في آخر رسالة المساعد، ويربط نتائج الأداة كرسائل الأداة.
3. حافة مشروطة من`agent`تلك الطرق إلى`tools`إذا كانت الرسالة الأخيرة تحتوي على tool_calls ، وإلا `END`. . .
4. حافة ثابتة من`tools`عودوا إلى`agent`. . .

هذا هو الأمر. تحصل على حلقة ReAct كاملة (الفكر → العمل → الملاحظة → الفكر → ...) مع التقاطع، والقطع، والتشغيل، في حوالي 40 سطر من الشفرة.

### StateGraph vs Send (موقع)

`Send(node_name, state)`يسمح للعقد بإرسال المخططات الفرعية المتوازية. مثال: الوكيل يقرر استفسار ثلاثة متسابقين في وقت واحد. كل `Send`يخلق تنفيذ متوازي للعقدة المستهدفة؛ وتتدمج نتائجها من خلال خفض الحالة. هكذا يعبر لانغغغراف عن نمط الموسيقي-العمال دون خيط البدائيات.

### المخطوطات الفرعية

يمكن أن يكون الرسم البياني المجمّع عقدة في الرسم البياني الآخر. الرسم البياني الخارجي يرى عقدة واحدة؛ الرسم البياني الداخلي له حالته الخاصة ومراقبها الخاص. هكذا تقوم فرق بناء وكلاء عامل المشرف: يرسل الرسم البياني المشرف نية المستخدم إلى سبغراف عامل لكل نطاق.

```figure
l5-state-graph-ledger
```

## بناءها

### الخطوة الأولى: الحالة والعقد

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

`add_messages`هو القيادة التي تجمع قائمة الرسائل بدلا من إعادة كتابتها. إنسيانها هو أخطاء LangGraph الأكثر شيوعا.

### الخطوة الثانية: تشغيل بشبكة

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

كل تحديث هو أمر`{node_name: state_delta}`. يمكن أن تقوم الجبهة بتدفق هذه إلى واجهة المستخدم حتى يرى المستخدمون "الوكيل يفكر...

### الخطوة الثالثة: إضافة الإنسان في الحلقة المقاطعة

قم بتشخيص العقدة حتى يتوقف التنفيذ قبل تشغيله

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

الحالة، نقطة التفتيش، والخيط يستمرون طوال المقاطعة لا يوجد شيء في الذاكرة إلا أثناء الإعدام

### الخطوة الرابعة: السفر عبر الزمن لإعداد التحليلات

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

يمر`None`عندما تقوم المدخل بإعادة تشغيل من نقطة التفتيش المحددة؛ إعادة إعطاء قيمة يضيفها كتحديث لحالة نقطة التفتيش تلك قبل استئنافها. هكذا تقوم بإعادة تشغيل وكيل سيء دون إعادة تشغيل المحادثة بأكملها.

### الخطوة 5: تبادل نقطة التفتيش للإنتاج

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

(سكلايت) و (ريديس) و (بوسغريس) أرسلت`MemorySaver`أي شيء يستمر عبر إعادة التشغيل يريد متجر حقيقي

## المهارة

> أنتِ تبنيين العملاء كرسومات، وليس كـ`while True`حلقات

قبل أن تصل إلى لنجراف، قم بتصميم 60 ثانية:

1. **Name the nodes.**كل قرار منفصل أو عمل يؤثر جانبي هو عقدة. "الوكيل يفكر،" "الأداة تعمل،" "المراجع يوافق،" "تدفقات الاستجابة".
2. **Declare the state.**الحد الأدنى من النمطDict مع خفض لكل حقل القائمة. لا تضع كل شيء في `messages`؛ رفع الحقول المحددة للمهمة (عمل`plan`، أ`budget`العداد، a `retrieved_docs`(قائمة) إلى المستوى الأعلى.
3. **Draw the edges.**ثابتة ما لم تعتمد الخطوة التالية على إصدار النموذج. كل حافة مشروطة تحتاج إلى وظيفة توجيه مع فرع مسمى.
4. **Choose a checkpointer up front.** `MemorySaver`لا يتم شحن دون واحد  لا يوجد نقطة تفتيش لا تعني لا وجود ليرة الذكر، لا وجود لقطعة، لا وجود للسفر عبر الزمن.
5. **Decide interrupts before tools run, not after.**الموافقة تذهب على الحافة إلى عقدة تأثير جانبي حتى تتمكن من إلغاء قبل الضرر؛ التحقق من التحقق من التحقق من التحقق من النموذج حتى تتمكن من رفض المكالمات السيئة رخيصة.
6. **Stream by default.** `mode="updates"`للصفحة الواحدة، `mode="messages"`للتدفق على مستوى الرمز داخل عقدة النموذج ، `mode="values"`لقطات مفصلة خلال التقييم.

رفض شحن وكيل لنجراف الذي لا يمتلك نقطة التفتيش رفض شحن واحد الذي يقاطع بعد التأثير الجانبي رفض شحن`messages`الحقل بدون`add_messages`كمخفضة لها

## التمارين

1. **Easy.**تنفيذ الرسم البياني ReAct الأربعة العقدة أعلاه مع أداة الحاسبة و أداة بحث الويب. تحقق من أن `list(app.get_state_history(config))`يعود أربع نقاط تفتيش على الأقل للحوار المزدوج.
2. **Medium.**إضافة`planner`العقدة التي تمر قبلها`agent`وكتب كتابة منظمة`plan: list[str]`إلى الولاية`agent`علامة خطة الخطوات كما فعلت. فشل في الاختبار إذا`plan`ضاعت في سيرته الذاتية في نقطة التفتيش (المخفض الخاطئ).
3. **Hard.**بناء الرسم البياني للإشراف الذي يتوجب بين ثلاث صور فرعية (`researcher`،`writer`،`reviewer`) باستخدام `Send`كل فرعي لديه حالته ومراقبه الخاص`interrupt_before=["writer"]`على الرسم البياني الخارجي حتى يستطيع الإنسان الموافقة على البحث المختصر. تأكيد أن السفر عبر الزمن من نقطة تفتيش سابقة يعد فقط الفرشة المفترقة.

## الشروط الرئيسية

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

## المزيد من القراءة

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) الإشارة القنوية لـ StateGraph، والخفضات، والمركبات التفتيشية، والقاطعات.
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/)النموذج العقلي الذي تستخدمه هذه الدروس مباشرة من المصدر
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) التفاصيل على متاجر Postgres/SQLite/Redis، ومناطق أسماء نقاط التفتيش، وتعريفات الأسلاك.
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) `interrupt_before`،`interrupt_after`،`Command(resume=...)`، و النمط من إصدار الحالة.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) النمط الذي ينفذ كل وكيل لنجراف؛ اقرأه للحصول على دليل التفكير.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) أي أشكال الرسم البياني (سلسلة، راوتر، موظف أوركستراتور، مقدم تقييم-تحسين) تفضل ومتى.
- المرحلة 11 · 09 (تدعو الوظيفة)  أداة-دعوة البدائية كل عقدة عامل LangGraph إعادة استخدامها.
- المرحلة 11 · 14 (مثال بروتوكول السياق)  اكتشاف أداة خارجية تتصل مع LangGraph `ToolNode`عبر جهاز التكيف MCP
- المرحلة 11 · 17 (تبادلات إطار العملاء)  متى لاختيار LangGraph على CrewAI، AutoGen، أو Agno.
