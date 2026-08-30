# एजेंट स्टेट मशीन  ग्राफ, नोड्स, चेकपॉइंट

> एक ReAct लूप जो हाथ से लिखा गया है `while True`एक स्पष्ट ग्राफ के रूप में लिखा हुआ एक ही लूप कुछ है कि आप चेकपॉइंट, बाधित, शाखा, और समय यात्रा कर सकते हैं. एजेंट नहीं बदला है. उसके चारों ओर के हार्नेस है.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 minutes

## समस्या

आप एक कार्य कॉल एजेंट भेजते हैं. यह तीन बारी के लिए काम करता है, फिर कुछ गलत हो जाता हैः मॉडल एक उपकरण का प्रयास करता है जो 500 वापस करता है, उपयोगकर्ता अपने मन को बदल देता है मध्य कार्य, या एजेंट एक आदेश वापस करने का फैसला करता है मानव हस्ताक्षर किए बिना.`while True:`लूप में कोई हुक नहीं है. आप इसे रोक नहीं सकते, आप इसे वापस नहीं कर सकते, और आप "क्या होगा अगर मॉडल अन्य उपकरण चुना था. " क्षण आप एक डेमो के बाद इस शिप, एजेंट एक काला बॉक्स बन जाता है या तो काम किया या नहीं किया.

एक बार जब आप इसे देखते हैं तो अगला कदम स्पष्ट है। एजेंट पहले से ही एक राज्य मशीन है  सिस्टम प्रॉम्प्ट प्लस संदेश इतिहास प्लस लंबित उपकरण कॉल प्लस अगली कार्रवाई। राज्य मशीन को स्पष्ट बनाएंः "मॉडल सोचता है", "एक उपकरण चलता है", "एक मानव अनुमोदित करता है", और उनके बीच के सशर्त संक्रमण के लिए किनारे। एक बार जब ग्राफ स्पष्ट हो जाता है, तो हर्नस को चार चीजें मुफ्त में मिलती हैंः चेकपॉइंटिंग (चरणों के बीच स्थिति सहेजें), इंटरट्यूट्स (एक इंसान के लिए विराम), स्ट्रीमिंग (स्ट्रीम टोकन और मध्यवर्ती घटनाएं), और समय यात्रा (एक पूर्व स्थिति में वापस लौटें और एक अलग शाखा का प्रयास करें) ।

इस अमूर्तता का संदर्भ कार्यान्वयन लैंगग्राफ है। यह लैंगचेन अर्थ में एक एजेंट फ्रेमवर्क नहीं है ("यहां एक एजेंट निष्पादक है, शुभकामनाएं") । यह प्रथम श्रेणी की स्थिति, प्रथम श्रेणी की दृढ़ता और प्रथम श्रेणी के विराम के साथ एक ग्राफ रनटाइम है। एजेंट लूप कुछ ऐसा है जिसे आप आकर्षित करते हैं, न कि कुछ ऐसा जिसे आप हाथ से लिखते हैं।

## अवधारणा

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

ए `StateGraph`तीन चीजें हैं।

1. **State.**एक टाइप किया गया डिक्ट (TypedDict या Pydantic मॉडल) जो ग्राफ के माध्यम से बहता है। प्रत्येक नोड को पूर्ण स्थिति प्राप्त होती है और आंशिक अद्यतन लौटाता है, जिसे लैंगग्राफ प्रति फ़ील्ड *reducer* का उपयोग करके मिलाता है `operator.add`उन सूचियों के लिए जो जमा होने चाहिए, डिफ़ॉल्ट रूप से ओवरराइट करें।
2. **Nodes.**पायथन फ़ंक्शंस `state -> partial_state`प्रत्येक एक अलग कदम हैः "मॉडल को कॉल करें", "उपकरण चलाएं", "संक्षेप लें।
3. **Edges.**नोड्स के बीच संक्रमण. स्थैतिक किनारे एक स्थान पर जाते हैं. सशर्त किनारे रूटर फ़ंक्शन लेते हैं`state -> next_node_name`तो ग्राफ मॉडल आउटपुट पर शाखा कर सकते हैं।

आप ग्राफ संकलित करते हैं. संकलित टॉपलॉजी को बांधता है, एक चेक पॉइंटर (वैकल्पिक लेकिन उत्पादन के लिए आवश्यक) संलग्न करता है, और एक रनएबल वापस करता है। आप इसे एक प्रारंभिक स्थिति और एक के साथ कॉल करते हैं `thread_id`निष्पादन के प्रत्येक चरण में एक चेकपोस्ट है जो किशिंग पर है`(thread_id, checkpoint_id)`. .

### चार सुपर पावर

**Checkpointing.**प्रत्येक नोड संक्रमण एक स्टोर में नई स्थिति लिखता है (टेस्ट के लिए मेमोरी में, पोस्टग्रेस/रेडिस/एसक्यूएलइट के लिए प्रोड) । उसी के साथ ग्राफ को फिर से कॉल करके पुनः आरंभ करें ।`thread_id`. ग्राफ को फिर से शुरू होता है जहां यह रुक गया था.

**Interrupts.** के साथ एक नोड को चिह्नित करें`interrupt_before=["human_review"]`और निष्पादन उस नोड चलाने से पहले बंद हो जाता है. राज्य बरकरार है. अपने एपीआई उपयोगकर्ता के लिए प्रतिक्रिया "अनुमोदन के लिए प्रतीक्षा". एक बाद में अनुरोध के लिए एक ही `thread_id`के साथ`Command(resume=...)`निष्पादन को फिर से शुरू कर देता है।

**Streaming.** `graph.stream(state, mode="updates")`राज्य डेल्टा के रूप में वे होते हैं।`mode="messages"`मॉडल नोड्स के अंदर LLM टोकन स्ट्रीम करता है। `mode="values"`आप चुनते हैं कि आप अपने UI में क्या सतह पर आने के लिए.

**Time-travel.** `graph.get_state_history(thread_id)`चेकपॉइंट लॉग को वापस करता है। किसी भी पूर्व पास करें।`checkpoint_id``graph.invoke`डिबगिंग के लिए अच्छा ("क्या होगा अगर मॉडल ने इसके बजाय उपकरण बी चुना था? ") और उत्पादन के निशान को फिर से खेलने वाले प्रतिगमन परीक्षण के लिए।

### घटाने वाले हैं मुद्दा

प्रत्येक राज्य क्षेत्र में एक घटाने वाला है. अधिकांश डिफ़ॉल्ट ठीक हैं  एक नया मान पुराने को ओवरराइट करता है. लेकिन संदेश सूचियों की जरूरत है `operator.add`तो नए संदेशों को बदलने के बजाय जोड़ते हैं. समानांतर किनारे अपने अद्यतनों को घटाने के माध्यम से मिलाते हैं। यदि दो नोड्स दोनों अद्यतन `messages`और तुम भूल गए `Annotated[list, add_messages]`रिड्यूसर पुस्तकालय में एकमात्र सूक्ष्म चीज है; इसे सही करें और बाकी रचनाएं।

### चार नोड्स में ReAct ग्राफ

एक उत्पादन ReAct एजेंट में चार नोड्स और दो किनारे होते हैंः

1. `agent` वर्तमान संदेश इतिहास के साथ LLM को बुलाता है। सहायक संदेश (जो tool_calls हो सकता है) लौटाता है।
2. `tools` अंतिम सहायक संदेश में किसी भी tool_call निष्पादित करता है, उपकरण संदेश के रूप में उपकरण परिणामों को जोड़ता है।
3.  से एक सशर्त किनारा`agent`जो मार्गों के लिए `tools`यदि अंतिम संदेश में tool_calls है, तो अन्यथा `END`. .
4.  से एक स्थैतिक किनारा`tools`वापस `agent`. .

यह है. आपको लगभग 40 लाइनों में कोड के साथ चेकपॉइंटिंग, इंटरक्यूज और स्ट्रीमिंग के साथ पूरा ReAct लूप (Think → Action → Observation → Thought → ...) मिलता है।

### StateGraph बनाम भेजें (फैनआउट)

`Send(node_name, state)`उदाहरण: एजेंट एक ही समय में तीन रिट्रीवरों से क्वेरी करने का निर्णय लेता है। प्रत्येक `Send`यह एक समानांतर निष्पादन को जन्म देता है लक्ष्य नोड; उनके आउटपुट राज्य घटाने के माध्यम से विलय करते हैं। यह है कि कैसे LangGraph मूलभूत थ्रेडिंग के बिना ऑर्केस्ट्रेटर-कामगार पैटर्न व्यक्त करता है।

### उपग्राफ

एक संकलित ग्राफ एक अन्य ग्राफ में एक नोड हो सकता है। बाहरी ग्राफ एक एकल नोड देखता है; आंतरिक ग्राफ में अपनी स्थिति और अपने स्वयं के चेकपोइंट हैं। इस तरह टीमें पर्यवेक्षक-कार्यकर्ता एजेंट बनाते हैंः पर्यवेक्षक ग्राफ उपयोगकर्ता के इरादे को प्रति डोमेन कार्यकर्ता उपग्राफ में रूट करता है।

```figure
l5-state-graph-ledger
```

## इसे बनाओ

### चरण 1: स्थिति और नोड्स

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

`add_messages`यह कम करने वाला है जो संदेश सूची को ओवरराइट करने के बजाय जमा करता है। इसे भूलना सबसे आम लैंगग्राफ बग है।

### चरण 2: एक धागे के साथ चलाएं

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

हर अद्यतन एक आदेश है`{node_name: state_delta}`. आपका फ्रंटेंड इन यूजर यूजर को स्ट्रीम कर सकता है ताकि उपयोगकर्ता देख सकें "एजेंट सोच रहा है ... खोज_वेब को कॉल कर रहा है ... परिणाम मिला ... जवाब दे रहा है। "

### चरण 3: एक मानव-इन-द-लूप इंटरट्यूड जोड़ें

एक नोड को चिह्नित करें ताकि निष्पादन चलाने से पहले रुक जाए।

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

राज्य, चेकपॉइंट, और धागा सभी विराम के दौरान बरकरार रहते हैं।

### चरण 4: डिबगिंग के लिए समय यात्रा

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

गुजर रहा है`None`जब इनपुट दिए गए चेकपॉइंट से फिर से चलाता है; एक मान पारित करने से इसे फिर से शुरू करने से पहले उस चेकपॉइंट की स्थिति में अपडेट के रूप में जोड़ा जाता है। इस तरह आप पूरी बातचीत को फिर से चलाए बिना एक खराब एजेंट को पुनः उत्पन्न करते हैं।

### चरण 5: उत्पादन के लिए चेकपोइंट को स्विच करें

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis, और Postgres शिप कर रहे हैं। `MemorySaver`फिर से शुरू करने के बाद भी जो कुछ भी बरकरार रहता है, उसे एक असली दुकान चाहिए।

## कौशल

> आप एजेंटों को ग्राफ के रूप में बनाते हैं, न कि `while True`लूप्स।

लैंगग्राफ तक पहुंचने से पहले, 60 सेकंड का डिज़ाइन करेंः

1. **Name the nodes.**प्रत्येक निर्णय या साइड इफेक्टिंग क्रिया एक नोड है। "एजेंट सोचता है", "उपकरण चलाता है", "रिव्यूर अनुमोदित करता है", "रिस्पॉन्स स्ट्रीम।" यदि आप उन्हें सूचीबद्ध नहीं कर सकते हैं, तो कार्य अभी तक एजेंट के आकार में नहीं है।
2. **Declare the state.**प्रत्येक सूची क्षेत्र के लिए एक घटाने वाला न्यूनतम टाइपडिक्ट। सब कुछ में नहीं भरें `messages`; कार्य-विशिष्ट क्षेत्र (एक कार्यक्षेत्र)`plan`, ए `budget`काउंटर, ए `retrieved_docs`सूची) को शीर्ष स्तर तक पहुंचाया गया है।
3. **Draw the edges.**जब तक अगले चरण मॉडल आउटपुट पर निर्भर नहीं करता है तब तक स्थिर। प्रत्येक सशर्त किनारे को नामित शाखाओं के साथ एक राउटर फ़ंक्शन की आवश्यकता होती है।
4. **Choose a checkpointer up front.** `MemorySaver`परीक्षण के लिए, Postgres/Redis/SQLite किसी भी अन्य चीज के लिए। एक के बिना जहाज नहीं करते हैं  कोई चेकपोइंटर का मतलब कोई रिज्यूमे नहीं है, कोई रुकावट नहीं है, कोई समय यात्रा नहीं है।
5. **Decide interrupts before tools run, not after.**अनुमोदन एक साइड-इफेक्टिंग नोड में बढ़ता है ताकि आप नुकसान से पहले रद्द कर सकें; सत्यापन मॉडल के बाहर बढ़ता है ताकि आप सस्ते में बुरे कॉल को अस्वीकार कर सकें।
6. **Stream by default.** `mode="updates"`यूआई के लिए, `mode="messages"`मॉडल नोड्स के अंदर टोकन स्तर पर स्ट्रीमिंग के लिए, `mode="values"`मूल्यांकन के दौरान पूर्ण स्नैपशॉट के लिए।

एक LangGraph एजेंट भेजने से इनकार करें जो कोई चेक पॉइंटर नहीं है।`messages`बिना क्षेत्र `add_messages`इसके घटाने वाले के रूप में।

## व्यायाम

1. **Easy.**एक कैलकुलेटर उपकरण और एक वेब खोज उपकरण के साथ ऊपर चार-नोड ReAct ग्राफ को लागू करें। यह सत्यापित करें कि `list(app.get_state_history(config))`दो-चक्र वार्ता के लिए कम से कम चार चेकपोस्ट लौटाता है।
2. **Medium.**एक जोड़ें `planner`पहले से चलती नोड `agent`और एक संरचित लिखता है `plan: list[str]`राज्य में.`agent`यदि आप परीक्षा में असफल हो जाते हैं तो`plan`चेकपॉइंट रिज्यूमे (गलत रेड्यूसर) पर खो गया है।
3. **Hard.**एक पर्यवेक्षण ग्राफ बनाएं जो तीन उपग्राफों के बीच मार्ग (`researcher`,`writer`,`reviewer`) का उपयोग करके `Send`. प्रत्येक उपग्राफ के अपने राज्य और चेक पॉइंटर्स है. एक जोड़ें`interrupt_before=["writer"]`एक मानव अनुसंधान संक्षिप्त अनुमोदन करने के लिए बाहरी ग्राफ पर. पुष्टि करें कि एक पूर्व चेकपॉइंट से समय यात्रा केवल कांटा शाखा फिर से चलाता है.

## प्रमुख शर्तें

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

## आगे पढ़ना

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) स्टेटग्राफ, रिड्यूसर, चेक पॉइंटर्स और इंटरट्यूट्स के लिए कैनोनिक संदर्भ।
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) इस पाठ में उपयोग किए जाने वाले मानसिक मॉडल, सीधे स्रोत से।
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) Postgres/SQLite/Redis स्टोर, चेकपॉइंट नाम स्थान और थ्रेड आईडी पर विवरण।
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) `interrupt_before`,`interrupt_after`,`Command(resume=...)`, और संपादित-राज्य पैटर्न.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) प्रत्येक लैंगग्राफ एजेंट द्वारा लागू किए जाने वाले पैटर्न; तर्क के लिए इसे पढ़ें।
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) कौन सा ग्राफ आकार (चैन, राउटर, ऑर्केस्ट्रेटर-वर्कर, मूल्यांकनकर्ता-अनुकूलनकर्ता) पसंद करना है और कब।
- चरण 11 · 09 (फंक्शन कॉल)  उपकरण-कॉल आदिम प्रत्येक LangGraph एजेंट नोड पुनः उपयोग करता है।
- चरण 11 · 14 (मॉडल कॉन्टेक्स्ट प्रोटोकॉल)  बाहरी उपकरण की खोज जो लैंगग्राफ में प्लग करता है `ToolNode`एमसीपी एडाप्टर के माध्यम से।
- चरण 11 · 17 (एजेंट फ्रेमवर्क ट्रेडऑफ)  जब CrewAI, AutoGen, या Agno पर LangGraph चुनें।
