# La interfaz de herramientas  Por qué los agentes necesitan una entrada/participación estructurada

> Un modelo de lenguaje produce tokens. Un programa toma acciones. La brecha entre estos dos es la interfaz de herramientas: un contrato que permite al modelo solicitar una acción y el host ejecutarla. Cada 2026 pila  función que llama a OpenAI, Anthropic y Gemini; MCP's `tools/call`La función de A2A es una codificación diferente del mismo bucle de cuatro pasos.

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11 (LLM completion APIs)
**Time:** ~45 minutes

## Objetivos de aprendizaje

- Explica por qué un LLM que sólo puede generar texto no puede, por sí solo, tomar medidas contra el mundo real.
- Dibujar el bucle de llamada de herramienta de cuatro pasos (describir → decidir → ejecutar → observar) y nombrar quién es el dueño de cada paso.
- Escriba una descripción de herramienta en tres partes: nombre, entrada de esquema JSON y función de ejecutor determinista.
- Distinguir entre herramientas puras y herramientas con efectos secundarios y explicar por qué la división es importante para la seguridad.

## El problema

Un LLM emite una distribución de probabilidades sobre el siguiente token. Esa es toda la superficie de salida. Si le preguntas a un modelo de chat "cuál es el tiempo en Bengaluru en este momento", puede escribir una frase plausible, pero no puede marcar en una API meteorológica. La frase puede ser correcta por casualidad o tres días de obsolescencia.

Cerrar esa brecha es el propósito de la interfaz de la herramienta. El programa host  su agente runtime, Claude Desktop, ChatGPT, Cursor, o un script personalizado  anuncia una lista de herramientas llamables al modelo. El modelo, cuando decide que es necesaria una acción, emite una carga útil estructurada que nombra una herramienta y sus argumentos. El anfitrión analiza la carga útil, ejecuta la herramienta en realidad y devuelve el resultado. El bucle continúa hasta que el modelo decide que no se necesitan más llamadas.

La primera versión de este contrato fue lanzada en junio de 2023 como parámetro de "funciones" de OpenAI.`tool_use`Bloques en Claude 2.1.`functionDeclarations`Un año después, cada proveedor expone la misma forma: una lista de herramientas tipo JSON-Schema, una llamada de herramienta de carga útil JSON. El Protocolo Contextual Modelo (novembre 2024) generalizó el contrato por lo que un registro de herramientas sirve a cada modelo. A2A (abril 2026, v1.0) layered el mismo primitivo para la delegación de agente a agente.

El bucle de cuatro pasos es la invariante debajo de todo esto.

## El concepto

### Paso uno: describir

El anfitrión declara cada herramienta con tres campos.

- **Name.**Un identificador estable y legible por máquina.`get_weather`, no "la cosa del tiempo".
- **Description.**Un breve de un párrafo en lenguaje natural. "Use cuando el usuario pregunta sobre las condiciones actuales de una ciudad específica. No se use para datos históricos".
- **Input schema.**Un objeto de esquema JSON (proyecto 2020-12) que describe los argumentos de la herramienta.

Los proveedores modernos enseran estas declaraciones en el sistema de solicitud utilizando una plantilla específica para el proveedor, por lo que usted como solicitante solo se ocupa del formulario estructurado.

### Paso dos: decidir

Dado el mensaje del usuario y las herramientas disponibles, el modelo elige uno de tres comportamientos.

1. **Answer directly**No hay llamadas de herramientas.
2. **Call one or more tools.**Emite objetos de llamadas estructurados.`parallel_tool_calls: true`(por defecto en OpenAI y Gemini, opta por Anthropic) el modelo puede emitir múltiples llamadas en un solo giro.
3. **Refuse.**Las salidas estructuradas en modo estricto pueden producir una tipografía `refusal`bloqueo en lugar de una llamada.

Una carga útil de llamada de herramienta tiene tres campos estables: una llamada `id`, una herramienta `name`, y un JSON `arguments`El id existe para que el host pueda correlacionar el resultado posterior con la llamada específica, lo que importa cuando las llamadas paralelas vuelven fuera de orden.

### Paso tres: ejecutar

El host recibe la llamada, valida los argumentos contra el esquema declarado y ejecuta el ejecutor. Argumentos inválidos significan que el modelo alucinó un campo o usó el tipo incorrecto  un modo de falla muy común en modelos débiles. Los hosts de producción hacen una de tres cosas en argumentos inválidos: fallan rápidamente y superfieren el error al modelo, reparan el JSON con un parser restringido o intentan de nuevo el modelo con el error de validación incluido en el pedido.

El ejecutor en sí mismo es código ordinario. Python, TypeScript, un comando shell, una consulta de base de datos. Produce un resultado, que generalmente es una cadena, pero puede ser cualquier valor JSON o un bloque de contenido estructurado (texto, imagen o referencia de recursos en MCP). El resultado debe ser serializable.

### Paso cuatro: observar

El anfitrión añade el resultado de la herramienta a la conversación (como `tool`mensaje de papel con la coincidencia `id`El modelo ahora tiene la salida de la herramienta en contexto y puede producir una respuesta final o solicitar más llamadas.

### La confianza se dividió

Las herramientas vienen en dos sabores que son importantes para la seguridad.

- **Pure.**Solo para lectura, determinista, sin efectos secundarios. `get_weather`¿ Qué ?`search_docs`¿ Qué ?`get_current_time`- Es seguro llamar especulativamente.
- **Consequential.**Mutó el estado, gastó dinero, tocó los datos del usuario. `send_email`¿ Qué ?`delete_file`¿ Qué ?`execute_trade`- Tiene que estar cerrado.

La "Regla de dos" de Meta para la seguridad de los agentes 2026 dice que un solo giro puede combinar al máximo dos de: entradas no confiables, datos sensibles, acción consecuente. La interfaz de herramienta es donde se aplica esa regla  rechazando llamadas, solicitando confirmación del usuario o aumentando los escalones. Ver Fase 13 · 15 para el capítulo completo de seguridad y Fase 14 · 09 para las políticas de permisos a nivel de agente.

### Donde vive el bucle

| Context | Who describes | Who decides | Who executes |
|---------|---------------|-------------|--------------|
| Single-turn function calling (OpenAI/Anthropic/Gemini) | App developer | LLM | App developer |
| MCP | MCP server | LLM via MCP client | MCP server |
| A2A | Agent Card publisher | Calling agent | Called agent |
| Web browser (function-calling agent) | Browser extension / WebMCP | LLM | Browser runtime |

Los nombres de las columnas cambian, la estructura no.

### ¿Por qué no simplemente pedir al modelo que emita JSON?

"Pide al modelo que responda en JSON" fue el patrón de llamada de pre-función. No funciona entre el 5 y el 15 por ciento del tiempo en los modelos fronterizos y mucho más en los modelos más pequeños. Los modos de falla incluyen aparatos que faltan, vírgenes que siguen atrás, campos alucinados y tipos incorrectos. Luego necesita un pase de reparación JSON, una nueva prueba o un decodificador restringido.

Llamando a funciones nativas es mejor por tres razones. Primero, el proveedor entrena el modelo de extremo a extremo en la forma exacta de llamada, por lo que la tasa válida de JSON sube al 98 a 99 por ciento en modo estricto. En segundo lugar, la carga útil de la llamada se encuentra en su propio espacio de protocolo, no dentro del texto libre  por lo que una llamada de herramienta nunca se filtra en la respuesta visible para el usuario. En tercer lugar, los proveedores hacen cumplir el cumplimiento de esquemas con decodificación restringida (el modo estricto de OpenAI, el de Anthropic `tool_use`, de los Géminis.`responseSchema`La salida está garantizada para validar.

La fase 13 · 02 recorre las tres APIs de proveedores lado a lado.

### Disruptores de circuitos

El bucle termina cuando el modelo deja de emitir llamadas o el host alcanza un recuento máximo de vueltas. Los hosts de producción establecen esto entre 5 y 20 vueltas. Más allá de eso, casi seguramente estás en un bucle que el modelo no puede salir. Claude Code por defecto es de 20; OpenAI Assistants a 10; el modo de agente de Cursor a 25.

La alternativa  bucles ilimitados  aparece cada seis meses como "agente gastó $400 en llamadas API durante la noche" post mortem. No envíe sin un límite.

La fase 14 · 12 abarca la recuperación de errores y la auto-curación en profundidad; la fase 17 abarca los límites de la tasa de producción.

### Donde la Fase 13 va desde aquí

- Las lecciones 02 a 05 pulirán la superficie de llamada de herramienta a nivel de proveedor.
- Las lecciones 06 a 14 generalizan el ciclo en MCP.
- Las lecciones 15 a 18 defienden el bucle contra servidores hostiles, usuarios adversarios y superficies de autor remotas no autenticadas.
- Las lecciones 19 a 22 amplían el patrón a la colaboración entre agentes, observabilidad, enrutamiento y envasado.
- La lección 23 crea un ecosistema completo usando cada primitivo.

Cada lección restante es una elaboración de este ciclo de cuatro pasos.

```figure
tp-tool-loop
```

## Usalo

`code/main.py`ejecuta el bucle de cuatro pasos sin un LLM. Una función de "decider" falso simula el modelo mediante la coincidencia de patrones en el mensaje del usuario; el ejecutor, el validador de esquema y el arnés de observación de pasos son reales. ejecuta para ver la coreografía completa de solicitud / respuesta con estado intermedio impresible, luego reemplaza el decider falso con cualquier proveedor real en una lección posterior.

Qué ver:

- El registro de herramientas contiene tres campos por herramienta: nombre, descripción, esquema y una referencia de ejecutor.
- El validador es un subconjunto mínimo de JSON Schema (tipos, requeridos, enum, min/max) escrito solo en stdlib.
- El número de iteraciones del bucle se limita a cinco.

## Envío

Esta lección produce`outputs/skill-tool-interface-reviewer.md`. Dado un borrador de definición de herramienta (nombre + descripción + esquema + esquema de ejecutor), la habilidad lo audita para la aptitud del bucle: es el nombre máquina estable, es la descripción un breve uso completo, el esquema utiliza JSON Schema 2020-12 correctamente, y es la clasificación pura-vs-consecuenciales explícita.

## Los ejercicios

1. Añadir una cuarta herramienta a `code/main.py`llamado`get_stock_price(ticker)`. Escriba su descripción como "Usa cuando el usuario solicita un precio actual de las acciones por ticker. No utilice para precios históricos o resúmenes de mercado".

2. Rompe el validador de esquema.`arguments`El objeto carece de un campo requerido, y confirme que el host lo rechaza antes de ejecutarlo. Luego pasa una llamada con un campo desconocido adicional. Decide: debe rechazar o ignorar el host. Justifique su elección con un argumento de seguridad.

3. Clasifique cada herramienta en el arnés como pura o consecuente.`consequential: true`señalar a las entradas del registro que lo necesitan, y cambiar el bucle para imprimir una línea "confirmaría con el usuario" cada vez que se elija una herramienta consecuente. Esta es la forma de la puerta de confirmación que necesita cada host de producción.

4. Dibujar el bucle de cuatro pasos en papel con la tabla de columna del proveedor arriba llenado para su cliente favorito (Claude Desktop, Cursor, ChatGPT, o una pila personalizada).

5. Lea la guía de llamadas de función de OpenAI de arriba a abajo. Identifique el campo que se encuentra en la solicitud pero no en el bucle de cuatro pasos como se presenta aquí. Explica lo que agrega y por qué es conveniente en lugar de esencial.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Tool | "A thing the model can call" | A triple of name + JSON-Schema-typed input + executor function |
| Function calling | "Native tool use" | Provider-level API support for emitting structured tool calls instead of prose |
| Tool call | "The model's request to act" | A JSON payload with `id`, `name`, `arguments` emitted by the model |
| Tool result | "What the tool returned" | The executor's output, wrapped in a `tool` role message with matching id |
| Parallel tool calls | "Many calls at once" | Multiple call objects in one model turn, independent and orderable by id |
| Strict mode | "Guaranteed JSON" | Constrained decoding that forces the model's output to validate against the declared schema |
| Pure tool | "Read-only tool" | No side effects; safe to re-run |
| Consequential tool | "Action tool" | Mutates external state; requires gate, audit, or user confirmation |
| Four-step loop | "The tool-call cycle" | describe → decide → execute → observe |
| Host | "Agent runtime" | The program that holds the tool registry, calls the model, and runs the executor |

## Leer más

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) referencia canónica para las declaraciones de herramientas y formas de llamadas de estilo OpenAI
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) El de Claude `tool_use`- ¿ Qué ?`tool_result`formato de bloque
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling)¿ Qué es esto ?`functionDeclarations`y semántica paralela en Gemini
- [Model Context Protocol — Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) la generalización actual sin estado y agnóstica del proveedor de la interfaz de herramientas
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) el dialecto de esquema de cada herramienta moderna API habla
