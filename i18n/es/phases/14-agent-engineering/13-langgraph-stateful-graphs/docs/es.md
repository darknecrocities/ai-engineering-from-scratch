# Orquestación de gráficos estatales  Ejecución duradera y puntos de control

> El agente es una máquina de estado; los nodos son funciones; los bordes son transiciones; el estado se pone en punto de control después de cada nodo.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## Objetivos de aprendizaje

- Describa el modelo principal de LangGraph: máquina de estado con estado tipado, nodos de función, bordes condicionales y puntos de control post-nodo.
- Nombren las cuatro capacidades que destacan los documentos: ejecución duradera, transmisión, humano en el bucle, memoria integral.
- Explica las tres topologías de orquestación que LangGraph admite: supervisor, peer-to-peer (crujo), jerárquico (subgrafos anidados).
- Implemente un gráfico de estado stdlib con estado tipado, bordes condicionales y un ciclo de punto de control/resumen.

## El problema

Los agentes y los flujos de trabajo comparten un problema: cuando una ejecución de 40 pasos falla en el paso 38, desea reanudar desde el paso 38, no empezar de nuevo. Los modelos de estado de segunda clase dejan a los operadores pirateando retos alrededor de una biblioteca que asume nuevas ejecuciones.

La respuesta de diseño de LangGraph: el estado es un objeto tipado de primera clase, las mutaciones son explícitas y los puntos de control persisten después de cada nodo.`load_state(session_id)`¿Qué pasa?

## El concepto

### El gráfico

Un gráfico se define por:

- **State type.**Un dictado tipado (o modelo Pydantic) que cada nodo lee y muta.
- **Nodes.**Funciones puras`(state) -> state_update`Las actualizaciones se fusionan en estado después de regresar.
- **Edges.**Transiciones condicionales o directas entre nodos.
- **Entry and exit.** `START`y `END`Los nodos de la centinela marcan el límite.

Ejemplo: un agente con `classify`¿ Qué ?`refund`¿ Qué ?`bug`¿ Qué ?`sales`¿ Qué ?`done`los nodos  un flujo de trabajo de enrutamiento como un gráfico.

### Ejecución duradera

Después de que cada nodo regrese, el tiempo de ejecución serializa el estado y lo escribe a un punto de control (SQLite, Postgres, Redis, personalizado).`resume(session_id)`y recoger desde el paso N + 1 con estado exacto.

Los documentos de LangGraph destacan explícitamente a los usuarios de producción donde esto importa: Klarna, Uber, JP Morgan. La afirmación no es la forma del gráfico; es que la forma del gráfico más el punto de control hace que la recuperación sea barata.

### En streaming

Cada nodo puede producir una salida parcial. El gráfico transmite eventos por nodo-delta al llamador para que las UI se actualicen a medida que se ejecuta el gráfico.

### Hombre en el ciclo

Inspeccionar y modificar el estado entre los nodos. Implementaciones: pausa antes de un nodo crítico, estado de superficie a un humano, acepta modificaciones, reanuda. El checkpointer hace esto fácil porque el estado ya está serializado.

### La memoria

A corto plazo (dentro de una ejecución  historial de conversaciones en estado) y a largo plazo (a través de las carreras  persistente a través del checkpointer más una tienda a largo plazo separada). LangGraph se integra con sistemas de memoria externa (Mem0, personalizado) a través de herramientas.

### Tres topologías

1. **Supervisor.**El router central LLM envía a los sub-agentes especializados. `create_supervisor()`En el`langgraph-supervisor`(aunque el equipo de LangChain en 2026 recomienda hacer esto a través de herramientas que piden directamente más control de contexto).
2. **Swarm / peer-to-peer.**Los agentes se entregan directamente a través de una superficie compartida de herramientas.
3. **Hierarchical.**Supervisores que gestionan subsupervisores, implementados como subgrafos en niedo.

### Cuando este patrón va mal

- **Checkpoints too small.**Sólo el punto de control de conversación gira deja el estado de la herramienta y la memoria escribe irrecuperable.
- **Non-deterministic nodes.**Resume asume que las entradas de nodos producen la misma actualización de estado. Se deben capturar semillas aleatorias, relojes de pared, API externas.
- **Over-use of conditional edges.**Un gráfico con cada borde condicional es una máquina de estado que no se puede razonar. Prefiere cadenas lineales con ramas ocasionales.

```figure
langgraph-state
```

## Construye el mismo

`code/main.py`Implementa un gráfico de estados de stdlib:

- `State` un dictado escrito con `messages`¿ Qué ?`step`¿ Qué ?`route`¿ Qué ?`output`¿ Qué ?`human_approval`¿ Qué ?
- `Node` Callable tomando estado y devolviendo un dictado de actualización.
- `StateGraph` nodos + bordes + bordes condicionales + ejecutar + reanudar.
- `SQLiteCheckpointer`(falsas en memoria)  serializa estado después de cada nodo; `load(session_id)`- ¿Qué es eso?
- Un gráfico de demostración: clasificar -> rama(reembolso / error / ventas) -> puerta humana -> enviar.

- ¿Qué quieres decir ?

```
python3 code/main.py
```

El rastro muestra la primera carrera fallando en la puerta humana, persistencia, luego reanudar la producción final.

## Usalo

- **LangGraph** el referente, listo para producción.`create_react_agent`¿ Qué ?`create_supervisor`, o construir su propio gráfico.
- **AutoGen v0.4**(Lección 14)  Modelo alternativo de actores para escenarios de alta competencia.
- **Claude Agent SDK**(Lección 17)  Arnes administrado con tienda de sesiones integrada.
- **Custom** cuando necesite un control exacto sobre la forma del estado o el backend del checkpointer.

## Envío

`outputs/skill-state-graph.md`genera un gráfico de estado en forma de LangGraph en cualquier tiempo de ejecución objetivo con control y resumen conectado.

## Los ejercicios

1. Añadir un borde condicional de `classify`¿ Qué ?`end`Cuando la confianza de clasificación está por debajo de un umbral, reanuda la carrera después de un conjunto humano.`route`manualmente.
2. Cambiar el falso de SQLite por un verdadero punto de control SQLite.
3. Implementar bordes paralelos: dos nodos se ejecutan simultáneamente, se fusionan por un reducidor personalizado. ¿Qué compra el estado inmutable aquí?
4. Leer .`langgraph-supervisor`- ¿Qué es lo que se hace?`create_supervisor`Comparar las formas de las huellas.
5. Añadir streaming: cada nodo produce estado parcial mientras se ejecuta. Imprimir los deltas a su llegada.

## Términos clave

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

## Leer más

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) los documentos de referencia
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) API de patrón de supervisión
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) Alternativa de actor-modelo
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) Tienda de sesiones y subagentes
