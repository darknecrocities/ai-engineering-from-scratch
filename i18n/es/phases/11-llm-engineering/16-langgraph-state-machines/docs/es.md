# Máquinas del Estado agente  Gráficos, nodos, puestos de control

> Un bucle ReAct escrito a mano es un `while True`El mismo bucle escrito como un gráfico explícito es algo que se puede hacer en punto de control, interrumpir, ramificar y viajar en el tiempo.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 minutes

## El problema

Se envía un agente de llamada de función. Funciona durante tres vueltas, luego algo sale mal: el modelo prueba una herramienta que devuelve 500, el usuario cambia de opinión en medio de la tarea, o el agente decide reembolsar un pedido sin una firma humana.`while True:`no se puede pausar, no se puede volver a doblar, y no se puede ramificar en "qué pasa si el modelo hubiera escogido la otra herramienta". En el momento en que envías esto después de una demostración, el agente se convierte en una caja negra que o funcionó o no.

El siguiente paso es obvio una vez que lo veas. El agente ya es una máquina de estado  sistema de respuesta más el historial de mensajes más llamadas de herramienta pendientes más la siguiente acción. Haga que la máquina del estado sea explícita: nodos para "el modelo piensa", "una herramienta funciona", "un humano aprueba", y bordes para las transiciones condicionales entre ellas. Una vez que el gráfico es explícito, el arnés obtiene cuatro cosas de forma gratuita: control (salvar estado entre pasos), interrupciones (pausa para un humano), transmisión (tokens de flujo y eventos intermedios) y viaje en el tiempo (reincorporarse a un estado anterior y probar una rama diferente).

La aplicación de referencia de esta abstracción es LangGraph. No es un marco de agente en el sentido de LangChain ("aquí hay un AgentExecutor, buena suerte"). Es un tiempo de ejecución de gráfico con estado de primera clase, persistencia de primera clase y interrupciones de primera clase. El bucle de agente es algo que dibujas, no algo que escribes a mano.

## El concepto

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

¿ Qué es esto ?`StateGraph`tiene tres cosas.

1. **State.**Un dictado tipado (modelo TypedDict o Pydantic) que fluye a través del gráfico. Cada nodo recibe el estado completo y devuelve una actualización parcial, que LangGraph fusiona utilizando un *reducer* por campo `operator.add`para las listas que deben acumularse, sobrescribir por defecto.
2. **Nodes.**Funciones de Python `state -> partial_state`Cada uno es un paso discreto: "llamar al modelo", "ejecutar herramientas", "resumir".
3. **Edges.**Transiciones entre nodos. los bordes estáticos van a un lugar. los bordes condicionales toman una función de enrutador`state -> next_node_name`Así que el gráfico puede ramificar en la salida del modelo.

Compila la gráfica. Compila se une a la topología, se une un checkpointer (opcional pero esencial para la producción), y devuelve un ejecutable.`thread_id`Cada paso de la ejecución es un punto de control con teclado .`(thread_id, checkpoint_id)`¿ Qué ?

### Las cuatro superpotencias

**Checkpointing.**Cada transición de nodo escribe el nuevo estado a una tienda (en memoria para pruebas, Postgres/Redis/SQLite para prod).`thread_id`El gráfico se repite donde se detuvo.

**Interrupts.**Marque un nodo con `interrupt_before=["human_review"]`La API responde al usuario con "esperando aprobación". Una solicitud posterior a la misma `thread_id`con`Command(resume=...)`reanudará la ejecución.

**Streaming.** `graph.stream(state, mode="updates")`Los estados deltas como ocurren.`mode="messages"`transmite los tokens de LLM dentro de los nodos del modelo. `mode="values"`Es el momento de elegir qué aparecer en la interfaz de usuario.

**Time-travel.** `graph.get_state_history(thread_id)`devuelve el registro completo del puesto de control.`checkpoint_id`¿ Qué ?`graph.invoke`Y se forja desde ese punto. Es ideal para el depuración ("¿qué pasa si el modelo hubiera escogido la herramienta B en su lugar?") y para las pruebas de regresión que reproducen rastros de producción.

### Los reducidores son el punto

Cada campo de estado tiene un reducidor. La mayoría de los valores predeterminados están bien  un nuevo valor sobreescribe el viejo. Pero las listas de mensajes necesitan `operator.add`Los bordes paralelos fusionan sus actualizaciones a través del reducidor. Si dos nodos actualizan ambos`messages`y te olvidaste de la`Annotated[list, add_messages]`El reducidor es la única cosa sutil en la biblioteca; hazlo bien y el resto compone.

### El gráfico ReAct en cuatro nodos

Un agente ReAct de producción es de cuatro nodos y dos bordes:

1. `agent` llama al LLM con el historial de mensajes actual.
2. `tools` ejecuta cualquier llamada de herramienta en el último mensaje de asistente, añade los resultados de la herramienta como mensajes de herramienta.
3. Un borde condicional de `agent`que rutas a `tools`si el último mensaje tiene herramientas_llamadas, de otro modo `END`¿ Qué ?
4. Un borde estático de `tools`De vuelta a`agent`¿ Qué ?

Así es todo. Obtienes el bucle completo de ReAct (Pensamiento → Acción → Observación → Pensamiento → ...) con puntos de control, interrupciones y transmisión, en aproximadamente 40 líneas de código.

### StateGraph vs Enviar (fanout)

`Send(node_name, state)`El agente decide consultar tres extractores a la vez.`Send`La función de la función de reducción de los niveles de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de los grupos de trabajo de trabajo de los grupos de trabajo de trabajo de los grupos de trabajo de trabajo de los grupos de trabajo de trabajo de los grupos de trabajo de trabajo de los grupos de trabajo de trabajo de los grupos de trabajo de trabajo de los grupos de trabajo de trabajo de los grupos de trabajo de trabajo de trabajo de los grupos de trabajo de trabajo de trabajo de trabajo de los grupos de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo de trabajo

### Subgrafos

Un gráfico compilado puede ser un nodo en otro gráfico. El gráfico externo ve un solo nodo; el gráfico interno tiene su propio estado y sus propios puntos de control. Así es como los equipos construyen agentes supervisores de trabajadores: el gráfico supervisor rúa la intención del usuario a un subgrafo de trabajadores por dominio.

```figure
l5-state-graph-ledger
```

## Construye el mismo

### Paso 1: estado y nodos

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

`add_messages`Es el reducidor que hace que la lista de mensajes se acumula en lugar de sobrescribir.

### Paso 2: ejecutar con un hilo

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

Cada actualización es un dictado .`{node_name: state_delta}`Su frontend puede transmitir esto a la interfaz de usuario para que los usuarios vean "el agente está pensando... llamando a search_web... obtuvo el resultado... respondiendo".

### Paso 3: añadir una interrupción humana en el circuito

Marque un nodo para que la ejecución se detenga antes de ejecutarse.

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

El estado, el punto de control y el hilo persisten durante la interrupción.

### Paso 4: Viaje en el tiempo para el depuración

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

Pasando .`None`Cuando la entrada se repite desde el punto de control dado, pasar un valor lo añade como una actualización al estado de ese punto de control antes de reanudar. Así es como se reproduce un agente mal ejecutado sin volver a ejecutar toda la conversación.

### Paso 5: cambiar el punto de control para la producción

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis y Postgres están enviados.`MemorySaver`Todo lo que persiste en los reinicios necesita una tienda real.

## La habilidad

> Construyes agentes como gráficos, no como`while True`- ¿Qué es eso?

Antes de llegar a LangGraph, haz un diseño de 60 segundos:

1. **Name the nodes.**Cada decisión discreta o acción secundaria es un nodo. "El agente piensa, " "la herramienta se ejecuta, " "el revisor aprueba, " " "los flujos de respuesta". Si no puedes enumerarlos, la tarea aún no está en forma de agente.
2. **Declare the state.**Tipo mínimo Dict con un reducidor para cada campo de lista. No enchufes todo en `messages`• elevar los campos específicos de las tareas (un trabajo `plan`, una `budget`Contador, un `retrieved_docs`El Consejo de Ministros de la Unión Europea ha aprobado el proyecto de ley de la Unión Europea.
3. **Draw the edges.**Esta estática a menos que el siguiente paso dependa de la salida del modelo.
4. **Choose a checkpointer up front.** `MemorySaver`No se envíe sin uno  ningún punto de control significa ningún currículum, ninguna interrupción, ningún viaje en el tiempo.
5. **Decide interrupts before tools run, not after.**Las aprobaciones van en el borde a un nodo de efecto secundario para que puedas cancelar antes de dañar; la validación va en el borde fuera del modelo para que puedas rechazar malas llamadas a bajo costo.
6. **Stream by default.** `mode="updates"`para la interfaz de usuario, `mode="messages"`para la transmisión a nivel de token dentro de los nodos del modelo, `mode="values"`para las instantáneas completas durante la evaluación.

Rechazar el envío de un agente LangGraph que no tiene un punto de control. Rechazar el envío de uno que interrumpa *después* del efecto secundario. Rechazar el envío de un`messages`campo sin `add_messages`como su reducidor.

## Los ejercicios

1. **Easy.**Implemente el gráfico de cuatro nodos ReAct de arriba con una herramienta de calculadora y una herramienta de búsqueda web.`list(app.get_state_history(config))`devuelve al menos cuatro puestos de control para una conversación de dos vueltas.
2. **Medium.**Añadir un`planner`nodo que se ejecuta antes `agent`y escribe un estructurado `plan: list[str]`En el estado.`agent`Marque los pasos del plan como se ha hecho.`plan`se pierde en un currículum de control (reducidor incorrecto).
3. **Hard.**Construir un gráfico de supervisión que se ejecuta entre tres subgrafos (`researcher`¿ Qué ?`writer`¿ Qué ?`reviewer`) utilizando `Send`Cada subgrafo tiene su propio estado y punto de control.`interrupt_before=["writer"]`Confirmar que el viaje en el tiempo desde un punto de control anterior re-correce sólo la rama bifurcada.

## Términos clave

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

## Leer más

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) referencia canónica para StateGraph, reducidores, puntos de control y interrupciones.
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) el modelo mental que utiliza esta lección, directamente de la fuente.
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) el detalle de las tiendas Postgres/SQLite/Redis, los espacios de nombres de los puntos de control e identificadores de hilos.
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)¿ Qué es esto ?`interrupt_before`¿ Qué ?`interrupt_after`¿ Qué ?`Command(resume=...)`, y el patrón de edición-estado.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) el patrón que cada agente de LangGraph implementa; lea para el razonamiento de la racionalidad de rastreo.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) qué formas de gráfico (cadena, enrutador, orquestrador-trabajadores, evaluador-optimizador) preferir y cuándo.
- Fase 11 · 09 (Llamada de funciones)  la herramienta-llamada primitiva cada nodo agente LangGraph reutiliza.
- Fase 11 · 14 (Modelo de Protocolo de Contexto)  Descubrimiento de herramientas externas que se enchufen en un LangGraph `ToolNode`a través del adaptador MCP.
- Fase 11 · 17 (Compromiso de marco de agentes)  cuándo elegir LangGraph sobre CrewAI, AutoGen o Agno.
