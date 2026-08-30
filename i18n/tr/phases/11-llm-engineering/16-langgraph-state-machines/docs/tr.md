# Ajan Devlet Makineleri  Grafikler, düğmeler, Kontrol Noktalar

> Elle yazılmış bir ReAct döngüsü bir `while True`Bu, bir açık grafik olarak yazılan aynı döngüdür. kontrol noktası, kesinti, dal ve zaman yolculuğu yapabileceğiniz bir şey.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 minutes

## Sorun

Bir fonksiyon çağıran ajan göndersiniz. Üç tur için çalışır, sonra bir şey ters gider: model 500'i geri veren bir aracı dener, kullanıcı görev ortasında fikrini değiştirir veya ajan bir siparişi insan imzasız geri ödeme yapmaya karar verir.`while True:`Bu, bir demo'dan sonra gönderdiğiniz an, ajan ya işe yaradı ya da çalışmadı.

Bir sonraki adım, gördüğünüzde açık olur. Ajan zaten bir devlet makinesi  sistem tesisi artı mesaj geçmişi artı bekleyen araç çağrıları artı bir sonraki eylem. Devlet makinesini açık bir şekilde yapın: "model düşünür", "bir araç çalışır", "bir insan onaylar" ve aralarındaki koşullu geçişler için kenarlar. Grafiğin açık olduğu zaman, harness dört şeyi ücretsiz olarak alır: kontrol noktası (adılar arasında durum kaydet), kesintiler (insan için durak), akış (akış tokeni ve ara olaylar) ve zaman yolculuğu (önceki bir duruma geri dönüp farklı bir dal denemek).

Bu soyutlamanın referans uygulanması LangGraph. LangChain anlamında bir ajan çerçevesidir ("burada bir AgentExecutor, iyi şanslar"). Birinci sınıf durum, birinci sınıf ısrar ve birinci sınıf kesintilerle bir grafik çalıştırma süresi.

## Anlaşım

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

A.`StateGraph`Üç şey var.

1. **State.**Grafiği akıtır. Her düğüm tam durumu alır ve kısmi bir güncelleme gönderir, LangGraph'in her alan için *reducer* kullanarak birleştirdiği `operator.add`Toplanması gereken listeler için, varsayılan olarak yazılmasını değiştirin.
2. **Nodes.**Python fonksiyonları `state -> partial_state`Her biri ayrı bir adım: "modelle çağırabilir", "alçaları çalıştırır", "cümle yaparlar".
3. **Edges.**Kodular arasındaki geçişler. Statik kenarlar bir yere gider. Şartlı kenarlar bir yönlendirme işlevi alır.`state -> next_node_name`Yani grafik model çıkışına dalışabilir.

Grafi oluşturur. Topolojiyi bağlar, bir kontrol noktasını bağlar (özel ama üretim için gerekli) ve bir çalıştırılabilirini gönderir.`thread_id`Her atışın bir kontrol noktası vardır .`(thread_id, checkpoint_id)`- Evet .

### Dört süper güç

**Checkpointing.**Her düğüm geçimi yeni durumu bir depoya yazar (testler için hafıza, prod için Postgres/Redis/SQLite).`thread_id`Grafik durduğu yerden devam ediyor.

**Interrupts.**Bir düğüm ile işaretleyin `interrupt_before=["human_review"]`Bu durum devam eder. API'niz kullanıcıya "ilkelme bekliyor" cevabını verir.`thread_id`- Evet .`Command(resume=...)`İdamı yeniden başlatıyor.

**Streaming.** `graph.stream(state, mode="updates")`Bu durumlar, Delta eyaletlerini de etkiledi.`mode="messages"`LLM tokenlerini model düğümler içinde akıtır. `mode="values"`Uygulama alanında neyi açacaklarını seçersiniz.

**Time-travel.** `graph.get_state_history(thread_id)`Kontrol noktasının tamamını gönderir.`checkpoint_id`- ...`graph.invoke`Bu durum, " modelin yerine araç B seçtiği olsaydı ne olurdu?" ve üretim izlerini tekrarlayan gerileme testleri için harika.

### Kısıtlayıcılar önemli .

Her durum alanında bir azaltıcı vardır. Çoğu varsayılanlar iyi  yeni bir değer eski değerleri üstü yazıyor. Ama mesaj listeleri gerekir `operator.add`Bu nedenle, yeni mesajlar değiştirmek yerine eklenir. Düz kenarları güncellemelerini azaltıcı aracılığıyla birleştirir.`messages`Ve sen unutmuşsun.`Annotated[list, add_messages]`Kısaltıcı kütüphanede tek ince şey, doğru yaparsanız geri kalanı yazarsınız.

### ReAct grafik 4 düğümde

Bir üretim ReAct ajanı dört düğüm ve iki kenardan oluşur:

1. `agent` mevcut mesaj geçmişi ile LLM'yi çağırır. Yardımcı mesajını gönderir ( tool_calls içerebilir).
2. `tools` son asistan mesajında herhangi bir tool_call'u gerçekleştirir, araç sonuçlarını araç mesajları olarak ekler.
3. Şartlı bir kenar `agent`Bu yollar `tools`Eğer son mesajda tool_calls varsa, başka bir şekilde `END`- Evet .
4. `tools`Geri dön .`agent`- Evet .

Tam ReAct döngüsünü (Though → Action → Observation → Thought → ...) yaklaşık 40 satır kodla kontrol, kesinti ve akışla elde ediyorsunuz.

### StateGraph vs Gönder (fanout)

`Send(node_name, state)`Bir düğüm paralel altgrafları gönderir. Örnek: ajan üç geri alıcıyı bir anda sormaya karar verir. Her biri `Send`LangGraph, hedef düğümün paralel bir yürütmesini sağlar; çıkışları durum azaltıcısı aracılığıyla birleşir.

### Altyazılar

Bir grafik başka bir grafikte bir düğüm olabilir. Dış grafik tek bir düğüm görür; iç grafik kendi durumuna ve kendi kontrol noktalarına sahiptir.

```figure
l5-state-graph-ledger
```

## Yapın

### Adım 1: Durum ve düğümler

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

`add_messages`Bu, mesaj listesini üst yazmak yerine toplayarak azaltır.

### Adım 2: İpuçla çalıştır

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

Her güncelleme bir diktedir .`{node_name: state_delta}`Ön uçlarınız bunları kullanıcı arayüzüne aktarır böylece kullanıcılar "Agent düşünüyor"... arama web... sonuç aldı... cevap veriyor".

### Adım 3: Bir insan-da-da-da-da döngü kesintisi ekleyin

Bir düğüm işaretleyin, böylece çalıştırmadan önce çalıştırma durur.

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

Durum, kontrol noktası ve ip kesinti boyunca devam ediyor.

### Adım 4: Debugging için zaman yolculuğu

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

Geçmek .`None`Bu, bir değer geçerek, tekrar başlamadan önce bu kontrol noktasının durumuna bir güncelleme olarak ekler.

### Adım 5: Kontrol noktasını üretim için değiştir

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis ve Postgres gönderiliyor.`MemorySaver`Yeniden başlatma sırasında devam eden her şey gerçek bir mağazaya ihtiyaç duyar.

## Yetenek

> Ajanları grafik olarak inşa ediyorsun, değil.`while True`- Çubuklar.

LangGraph'e ulaşmadan önce 60 saniyelik bir tasarım yapın:

1. **Name the nodes.**Her ayrı karar veya yan etkisi olan eylem bir düğümdür. "Agent düşünür," "üçüm çalışır," "temizleyici onaylar," " yanıt akışları".
2. **Declare the state.**Her liste alanı için bir azaltıcı ile en az Tiplenmiş Dikt.`messages`; görev-sözlü alanları (bir çalışma `plan`, a `budget`karşılama, bir `retrieved_docs`listesi) en üst seviyeye kadar.
3. **Draw the edges.**Bir sonraki adım model çıkışına bağlı değilse, her koşullu kenarın isimli dallarla bir yönlendirme fonksiyonu olması gerekir.
4. **Choose a checkpointer up front.** `MemorySaver`Testler için Postgres/Redis/SQLite başka bir şey için.
5. **Decide interrupts before tools run, not after.**Onaylamalar kenarında yan etkileme düğümüne gider böylece zarar vermeden iptal edebilirsiniz; onaylama modelin kenarında gider böylece kötü çağrıları ucuz bir şekilde reddedebilirsiniz.
6. **Stream by default.** `mode="updates"`UI için, `mode="messages"`Modelle düğümler içinde token seviyesinde akış için, `mode="values"`değerlendirme sırasında tam anlık fotoğraflar için.

Kontrol noktası olmayan LangGraph ajanını göndermeyi reddedin yan etkiden sonra kesilen bir ajanı göndermeyi reddedin bir kontrol noktası olmayan LangGraph ajanını göndermeyi reddedin yan etkiden sonra kesilen bir ajanı göndermeyi reddedin bir kontrol noktası olmayan bir ajanı göndermeyi reddedin bir kontrol noktası olmayan bir ajanı göndermeyi reddedin bir kontrol noktası olmayan bir ajanı göndermeyi reddedin bir kontrol noktası olmayan bir ajanı göndermeyi reddedin bir kontrol noktası olmadan bir kontrol noktası göndermeyi reddedin bir kontrol noktası için bir kontrol noktası göndermeyi reddedin`messages` olmadan alan`add_messages`- Kısıtlayıcı olarak.

## Egzersizler

1. **Easy.**Yukarıdaki dört düğümlü ReAct grafiğini bir hesap makinesi aracı ve bir web arama aracı ile uygulayın.`list(app.get_state_history(config))`İki dönüşlü bir konuşma için en az dört kontrol noktasını geri gönderir.
2. **Medium.**Bir ekle`planner`Önceden giden düğüm`agent`ve yapılandırılmış bir yazı yazar.`plan: list[str]`- Eyalete.`agent`Plan adımlarını işaretleyin.`plan`kontrol noktası özetlemesinde kaybolur (sahte azaltıcı).
3. **Hard.**Üç alt grafik arasında yol alan bir denetim grafiği oluştur (`researcher`- Evet .`writer`- Evet .`reviewer`) kullanılarak `Send`Her altgrafın kendi durumu ve kontrol noktası vardır.`interrupt_before=["writer"]`Bir insan araştırma raporu onaylayabilsin diye dış grafikte.

## Anahtar Terimler

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

## Daha Fazla Okumak

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) StateGraph, azaltıcılar, kontrol noktaları ve kesintiler için kanonik referans.
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) bu dersin kullandığı zihinsel model, doğrudan kaynağından.
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) Postgres/SQLite/Redis depoları, kontrol noktaları isim alanları ve ip kimlikleri üzerindeki detaylar.
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) `interrupt_before`- Evet .`interrupt_after`- Evet .`Command(resume=...)`, ve düzenleme durumu kalıbı.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) her LangGraph ajanının uyguladığı örneği; mantık izlenimi mantıklılığı için okuyun.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) hangi grafik şekilleri ( zincir, yönlendirici, orkestrasyon-işçiler, değerlendirici-optimalisyoncu) tercih etmek ve ne zaman.
- Eğlence çağrısı primitifleri her LangGraph ajan düğümünü tekrar kullanır.
- 11 · 14 aşama (Model Kontext Protokolü)  LangGraph'e bağlanan dış araç keşfi `ToolNode`MCP adaptörü üzerinden.
- Eğlence 11 · 17 (Agent çerçeve pazarlamaları)  LangGraph'i CrewAI, AutoGen veya Agno'dan ne zaman seçmek.
