# Equipos de agentes basados en el papel  Roles, tareas, procesos

> Cuatro primitivas: agente, tarea, tripulación, proceso. Dos formas de nivel superior: equipos (autónomo, colaboración basada en roles) y flujos (evento-driven, determinista). CrewAI es la implementación de referencia de 2026, y sus documentos son contundentes: "para cualquier aplicación lista para la producción, comience con un flujo".

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 minutes

## Objetivos de aprendizaje

- Nombre a los cuatro primitivos de CrewAI (Agencia, tarea, tripulación, proceso) y lo que cada uno posee.
- Distinguir el proceso de Consenso secuencial, jerárquico y planeado; elegir uno por carga de trabajo.
- Distinguir a los equipos (basados en roles autónomos) de los flujos (determinísticos basados en eventos) y explicar la recomendación de producción de los docentes.
- Herramientas de alambre con el `@tool`decoratora y`BaseTool`Subclase; razonamiento sobre las salidas estructuradas vs texto libre.
- Nombre de los cuatro tipos de memoria CrewAI y cuando cada uno paga.
- Implementar un equipo de tres agentes (investigador, escritor, editor) que produzca un resumen.
- Detecta los tres modos de falla de CrewAI: Inflación rápida, impuestos de gerente-LLM, entregas frágiles.

## El problema

Los equipos que adoptan marcos multi-agentes se encuentran en la misma pared. "La colaboración autónoma" suena bien en una demostración. Luego un cliente presenta un error y necesitas una reproducción determinista. O las finanzas preguntan cuánto cuesta un equipo enrutado por LLM por carrera. O en la llamada necesita saber qué agente se detuvo a las 3 am.

Los equipos de formación libre y de formación en LLM no responden a ninguna de estas preguntas, pero los DAG puros responden a todas pero pierden la forma exploratoria que necesita un agente de lluvia de ideas.

La división de CrewAI es honesta sobre el comercio. equipos para el trabajo colaborativo, basado en el papel, exploratorio. flujos para la producción impulsada por eventos, propiedad de código, auditable. El mismo marco, dos formas, elegir por superficie.

## El concepto

### Cuatro primitivos

La superficie de la tripulación es pequeña, memorizad esto y el resto está configurado.

- **Agent.** `role + goal + backstory + tools + (optional) llm`La historia de fondo es cargadora. modela el tono, el juicio, cuando el agente se detiene. Las herramientas son funciones que el agente puede llamar (más abajo).
- **Task.** `description + expected_output + agent + (optional) context + (optional) output_pydantic`Una unidad de trabajo reutilizable.`expected_output`Es el contrato.`context`En el caso de las actividades de gestión de los datos, el número de datos de gestión de los datos de gestión de los datos de gestión de los datos de gestión de los datos de gestión de los datos de gestión de los datos de gestión de los datos de gestión de los datos de gestión de los datos de gestión de los datos de gestión de los datos de gestión de datos de la gestión de datos de la gestión de datos de datos de la gestión de datos de datos de la gestión de datos de datos de datos de la gestión de datos de datos de datos de datos de la gestión de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de cuyo sitio web.`output_pydantic`fuerza una forma estructurada.
- **Crew.**Container, posee la lista de`agents`, la lista de `tasks`, el `process`, y opcionales `memory`¿ Qué es eso ?`verbose`¿ Qué es eso ?`manager_llm`configuración.
- **Process.**Estrategia de ejecución: secuencial, jerárquico, consenso (planificado).

Los agentes no se ven directamente, las tareas son de referencia, la tripulación secuencia las tareas, el proceso decide quién elige la siguiente tarea, ese es todo el modelo mental.

> **Validated against**CrewAI 0.86 (2026-05). Las versiones más recientes pueden renombrar o fusionar los tipos de proceso; compruebe el [CrewAI Processes docs](https://docs.crewai.com/concepts/processes)antes de depender de una forma específica.

### Secuenciales vs Jerárquicos vs Consenso

- **Sequential.**Las tareas se ejecutan en orden de declaración.`context`El costo más bajo, más predecible, se utiliza cuando se fija el pedido.
- **Hierarchical.**Un agente gerente (llamada separada de LLM) rutas entre los especialistas.`manager_llm`Configurar o un error predeterminado. El administrador selecciona la siguiente tarea cada ronda y puede rechazar o redirigir.
- **Consensus.**Los documentos reservan el nombre para un futuro proceso basado en el voto.

Hierarquical añade una llamada de LLM por ronda (el gerente) en la parte superior de cada llamada especializada. El costo de las fichas puede triplicar en una carrera de cinco pasos. Pague solo cuando necesite el enrutamiento.

### Los equipos contra los flujos

Este es el marco con el que los doctores lideran en 2026.

- **Crew.**La autonomía impulsada por el LLM. El marco elige la forma en el tiempo de ejecución. Es bueno para: investigación, lluvia de ideas, primeros proyectos, dondequiera que el camino sea parte de la respuesta. Es difícil de reproducir. Es difícil de probar. Es barato para el prototipo.
- **Flow.**Grafico basado en eventos que posees.`@start`marca la entrada. `@listen(topic)`Es un paso que dispara cuando otro paso emite ese tema. Cada paso es Python simple (puede llamar a un equipo internamente).

Las recomendaciones de producción de los doctores para 2026: comience con un flujo.`Crew.kickoff()`El flujo te da el rastro de auditoría, la tripulación te da la exploración.

### Integración de herramientas

Hay tres formas de darle a un agente una herramienta.

1. **`@tool` decorator.**Las funciones puras se convierten en herramientas. La firma es el esquema; la cadena de documentos es la descripción que el LLM ve. Lo mejor para los ayudantes únicos.

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` subclass.**Herramienta basada en clases con esquema de args explícito, soporte de asíncrono, retries. Utilice cuando la herramienta tiene estado (un cliente, una caché) o necesita args estructurados.

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

3. **Built-in toolkits.**CrewAI envía adaptadores de primera parte: `SerperDevTool`¿ Qué ?`FileReadTool`¿ Qué ?`DirectoryReadTool`¿ Qué ?`CodeInterpreterTool`¿ Qué ?`RagTool`¿ Qué ?`WebsiteSearchTool`- Un cable con una importación.

Las salidas estructuradas utilizan Pydantic.`output_pydantic=MyModel`La respuesta de la MLL contra el modelo y o bien obliga o retenta.`expected_output`las salidas de texto libre son buenas para los borradores; las salidas estructuradas son lo que los flujos aguas abajo pueden consumir.

### Cuchillos de memoria

CrewAI saca cuatro tipos de memoria de la caja.

> **Validated against**CrewAI 0.86 (2026-05). Las últimas versiones tratan todo a través de un sistema unificado `Memory`El modelo conceptual de abajo sigue vigente, pero la superficie de la clase pública puede colapsar a una sola`Memory`punto de entrada en versiones más recientes; comprobar [CrewAI memory docs](https://docs.crewai.com/concepts/memory)para la API actual.

- **Short-term.**Puente de conversación en una sola carrera.
- **Long-term.**Persistido a través de ejecuciones. Almacenado en un vector DB (Chroma por defecto, intercambiable). Recuperado por similitud con la tarea actual.
- **Entity.**"El cliente X está en el plan empresarial" Es clave por entidad, no por similitud.
- **Contextual.**Recuperación en tiempo de montaje, extrae la memoria relevante en el momento en que el agente la necesita, no precargada.

Habilitar a la tripulación con `memory=True`El sistema de memoria de CrewAI es uno de los lugares donde CrewAI gana su mantenimiento frente a los marcos más delgados; LangGraph puro requiere que usted cable de cada uno de ellos usted mismo.

### Cuando los equipos basados en el papel se ajustan

- De tres a seis agentes con roles nombrados y un flujo de trabajo colaborativo.
- En el caso de los servicios de gestión de la empresa, el valor de la empresa es el valor de la empresa.
- En cualquier lugar el equipo es más feliz leyendo .`role + goal + backstory`que leer una definición de gráfico.

### Cuando no lo hacen

- Los DAG deterministas con orden estricto. Utilice LangGraph (lección 13). La forma del gráfico es la abstracción correcta; el marco de rol de CrewAI es la fricción.
- Los presupuestos de latencia subsegundos. Jerárquico añade viajes de ida y vuelta. Incluso Sequential serializa las instrucciones que incluyen historias de fondo y salidas anteriores.
- Los bucles de agente único. Salta el marco; un bucle de agente (lección 1) más un registro de herramientas es más corto.

La lección 17 (Tradeoffs de Marco de Agentes) expone esto en una matriz.

### Forma de dependencia

Independiente de LangChain. Python 3.10 a 3.13. utiliza `uv`Cuenta de estrellas: mira[crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)La integración de AWS Bedrock está documentada; los puntos de referencia de los proveedores informan de una velocidad sustancial frente a LangGraph en las cargas de trabajo de QA, pero la metodología (dataset, hardware, métrica de evaluación) no se publica, por lo que tratar los números de los proveedores de marco como direccionales.

### Cuando este patrón va mal

- **Prompt-bloat from backstories.**Una historia de fondo de 2000 palabras por agente y un equipo de cinco agentes quema el presupuesto de contexto antes de la primera llamada de herramienta. Mantenga las historias de fondo de menos de 200 palabras.
- **Manager-LLM token tax.**El proceso jerárquico agrega una llamada de LLM del gerente antes de cada llamada especialista. En un equipo de cinco tareas que es seis llamadas de LLM en lugar de cinco, y la llamada del gerente lleva la lista completa de tareas más las salidas anteriores.
- **Brittle handoffs.**La tarea N's `expected_output`La tarea N+1 lo lee como `context`La LLM produjo cuatro, los agentes de la corriente baja, los ad-libs.`output_pydantic`En la tarea N, la tarea N+1 lee un objeto mecanografiado, no texto libre.
- **Crew-as-prod.**El equipo de forma libre se envía a la producción sin envoltura de flujo. La variabilidad de salida es alta; la repetición es imposible; en la llamada no puede diferenciar una carrera mala contra una buena.

```figure
ae-crew-vs-flow
```

## Construye el mismo

`code/main.py`Implementa versiones de STDlib de ambas formas más un equipo de tres agentes.

Forma:

- `Agent`¿ Qué ?`Task`las clases de datos que coinciden con la superficie de la CrewAI.
- `SequentialCrew.kickoff(inputs)`ejecuta tareas en orden de declaraciones, trenzando las salidas como `context`¿ Qué ?
- `HierarchicalCrew.kickoff(topic)`Agrega un agente gerente que escoge al próximo especialista cada ronda, se detiene en "hecho".
- `Flow`con`@start`y `@listen(topic)`decoratores, un pequeño circuito de eventos, y un rastro.
- `tool(name)`Decorador que refleja el de CrewAI `@tool`¿Qué forma tiene?
- `Memory`con`short_term`¿ Qué ?`long_term`¿ Qué ?`entity`Las tiendas; la similitud burlada utiliza numpy.
- Las respuestas de LLM son cadenas codificadas con teclas de papel más prefijo de entrada.

Demo de concreto: investigador, escritor, equipo de redacción que produce un resumen sobre "ingeniería de agentes 2026".

- ¿Qué quieres decir ?

```bash
python3 code/main.py
```

Las cubiertas de rastro: secuencia de salida de la tripulación a través de los hilos `context`, equipo jerárquico con seleccionar el gerente (investigador, escritor, editor, luego "hecho"), flujo que se ejecuta los mismos tres pasos con temas explícitos (`researched`¿ Qué ?`drafted`¿ Qué ?`edited`), las llamadas a través de la herramienta`@tool`, y la memoria a largo plazo sobreviviendo a través de dos patadas.

El rastro de la tripulación es fluido, el gerente podría en principio reordenar el rastro de flujo está fijo esa elección es la lección

## Usalo

- **CrewAI Flow**Incluso cuando el flujo es un paso que llama`Crew.kickoff()`El flujo da el límite de auditoría.
- **CrewAI Crew (Sequential)**para el trabajo colaborativo de ordenamiento claro, especialmente los primeros proyectos y los ciclos de revisión.
- **CrewAI Crew (Hierarchical)**cuando el enrutamiento depende de la salida y usted tiene cuatro o más especialistas.
- **LangGraph**(Lección 13) para máquinas de estado explícito, currículum duradero, ordenamiento estricto.
- **AutoGen v0.4**(Lección 14) para la concurrencia del modelo actor y el aislamiento de fallos.
- **OpenAI Agents SDK**(Ley 16) para los productos OpenAI-first con cargas y barandillas.
- **Claude Agent SDK**(Ley 17) para productos de primera clase con subagentes y tienda de sesiones.

## Envío

`outputs/skill-crew-or-flow.md`Selecciona Crew vs Flow para una tarea y plantea la implementación mínima. Hard rechaza sobre temas de Crew-sin historia trasera, Flow-sin temas explícitos, Jerárquico con menos de tres especialistas.

## Las trampas

- **Backstory as flavor.**Se dan formas a las salidas, se prueban tres variantes por agente, la varianza es real, se escoge una y se congela.
- **Skipping `expected_output`.**Sin un contrato por tarea, las tareas posteriores se hacen cargo de lo que el LLM produjo.
- **Memory always-on.**El largo plazo escribe cada ejecución. El vector DB crece. La recuperación se hace ruidosa. El alcance escribe a tareas donde el hecho es persistente.
- **Manager prompt drift.**Si el enrutamiento se vuelve raro, deja en modo verbal y lee.
- **Tool side effects in Crews.**Un equipo puede llamar a una herramienta más veces de lo esperado.

## Los ejercicios

1. Convierta a la tripulación de la secuencia a un flujo, cuenta los puntos de contacto donde baja la variabilidad, nota donde baja la legibilidad.
2. Añadir memoria de entidad a la tripulación: los hechos sobre un cliente persisten a través de los arranques.
3. Implemente un proceso jerárquico en el que el gerente se niega a dirigir al editor hasta que la salida del escritor tenga al menos tres párrafos.
4. El cable a `BaseTool`Subclase para una búsqueda web (follando). Comparar la forma de rastreo con la `@tool`versión decorativa.
5. Añadir`output_pydantic=Brief`a la tarea de editor, donde `Brief`¿ Qué ?`title`¿ Qué ?`summary`¿ Qué ?`sections`. Hacer que la salida de la tarea de escritora JSON malformado una vez; verificar el comportamiento de CrewAI de nuevo en el rastreo.
6. Lea la introducción de los documentos de CrewAI.`crewai`¿Qué garantías se saltaron de la versión de STDlib?
7. ¿Qué rastros se perdieron en la versión de Stdlib?

## Términos clave

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

## Leer más

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction): conceptos y la vía de producción recomendada
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows): forma basada en eventos, `@start`¿ Qué ?`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)¿ Qué es esto ?`@tool`¿ Qué ?`BaseTool`, herramientas incorporadas
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory): a corto plazo, a largo plazo, entidad, contexto
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): cuando ayuda el multi-agente y cuando no
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): la alternativa de la máquina estatal
