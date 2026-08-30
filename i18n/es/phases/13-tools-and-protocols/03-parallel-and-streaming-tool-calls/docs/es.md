# Llamadas paralelas y transmisión con herramientas

> Tres búsquedas de tiempo independientes en serie son tres viajes de ida y vuelta. ejecutarlos en paralelo y el tiempo total se desploma a la llamada única más lenta. Cada proveedor fronterizo ahora emite múltiples llamadas de herramienta en un solo giro. La recompensa es real; la tubería es sutil. Esta lección camina ambas mitades: el fan-out paralelo y el reensamblaje de argumentos transmitidos, con énfasis en la trampa de correlación de identidad.

**Type:** Build
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## Objetivos de aprendizaje

- ¿ Por qué ?`parallel_tool_calls: true`existe y cuándo desactivarlo.
- Correlación de los trozos de argumento en streaming con la identificación de llamada de herramienta correcta durante el ventilador paralelo.
- Reassemblar parcialmente `arguments`las cadenas en JSON completo sin analizar temprano.
- Ejecutar un indicador meteorológico de tres ciudades que demuestre latencia secuencial vs paralelo.

## El problema

Sin llamadas paralelas, un agente respondiendo a "cuál es el tiempo en Bengaluru, Tokio y Zurich" hace esto:

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

Tres viajes de ida y vuelta de LLM, cada uno de los cuales también paga la latencia del ejecutor.

Con llamadas paralelas:

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

Un viaje de ida y vuelta de LLM. El tiempo de ejecución es el máximo de los tres, no la suma. Los índices de producción en OpenAI, Anthropic y Gemini muestran una reducción del 60 a 70 por ciento en el reloj de pared en las cargas de trabajo de ventilador.

Cuando las tres llamadas terminan fuera de orden, sus resultados deben llevar la coincidencia `tool_call_id`Cuando los resultados se transmiten, se debe reunir fragmentos de argumento parciales en JSON completo antes de ejecutar. Gemini 3 agregó ids únicos en parte para resolver un problema en el mundo real donde dos llamadas paralelas a la misma herramienta eran indistinguibles.

## El concepto

### Permitir el paralelo

- **OpenAI.** `parallel_tool_calls: true`en el por defecto.`false`para forzar a la serie.
- **Anthropic.**Paralelamente a través de `disable_parallel_tool_use: false`(por defecto en Claude 3.5 y más).`true`para la serie.
- **Gemini.**Siempre paralela .`tool_config.function_calling_config.mode = "AUTO"`Deje que el modelo decida.

Deshabilitar el paralelo cuando las herramientas tengan dependencias de orden (`create_file`Entonces ...`write_file`), cuando la salida de una llamada informa la entrada de otra, o cuando el limitador de velocidad no puede manejar el ventilador.

### Correlación de identidad

Cada llamada que emite el modelo tiene un`id`Cada resultado que devuelva el anfitrión debe incluir la misma identificación.

- **OpenAI.** `tool_call_id`en cada mensaje de papel de herramienta.
- **Anthropic.** `tool_use_id`en cada uno `tool_result`El bloque.
- **Gemini.** `id`en cada uno `functionResponse`(Gemini 3 y arriba; Gemini 2 coincidió por nombre que rompió para llamadas paralelas del mismo nombre).

### Ejecutar llamadas simultáneamente

El anfitrión ejecuta el ejecutor de cada llamada en su propio hilo, coroutine o trabajador remoto.`asyncio.gather`El orden de finalización es impredecible  el id es el identificador.

Un error común: responder con resultados en orden de lista de llamadas en lugar de orden de finalización. Esto suele funcionar porque el modelo solo se preocupa por `tool_call_id`, pero si un resultado se deja caer o se duplica, la presentación fuera de orden hace que el descomposición sea más difícil.

### Las llamadas de herramientas de transmisión

Cuando el modelo fluye,`arguments`Tres flujos separados de trozos para tres llamadas paralelas se interponen en el cable.

Forma por proveedor:

- **OpenAI.**Cada pieza es`choices[0].delta.tool_calls[i].function.arguments`El pedazo lleva`index`Se acumula por índice, se lee `id`cuando aparece por primera vez, y analizar JSON cuando `finish_reason = "tool_calls"`¿ Qué ?
- **Anthropic.**Los eventos de transmisión son`message_start`, luego uno .`content_block_start`por bloque con tipo `tool_use`(contiene identificación, nombre, entrada vacía). `content_block_delta`los eventos llevan `input_json_delta`Los trozos.`content_block_stop`cierra cada cuadra.
- **Gemini.** `streamFunctionCallArguments`(Gemini 3 y arriba) emite trozos con un `functionCallId`Antes de Gemini 3, la transmisión devolvió una llamada completa a la vez.

### JSON parcial y la trampa de análisis precoz

No puedes analizar .`arguments`JSON parcial como `{"city": "Beng`La puerta correcta es la señal de final de llamada del proveedor: OpenAI `finish_reason = "tool_calls"`, Anthropic's `content_block_stop`, o el evento de la línea de salida de Géminis.`json.loads`. Un enfoque más robusto utiliza un parser JSON incremental que produce eventos a medida que se completa la estructura; la guía de transmisión de OpenAI lo recomienda para UX que muestra un indicador de "pensamiento" en vivo. La cuenta de correcciones es poco confiable como prueba de integridad (las correcciones dentro de las cadenas citadas o el contenido escapado causan falsos positivos) y solo debe usarse como una heurística de defecto informal.

### Completado fuera de orden

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

La respuesta de la anfitriona deberá seguir citando las identidades siguientes:

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

El orden en la respuesta no importa para la corrección en OpenAI o Anthropic. Gemini acepta cualquier orden siempre que coincidan las identidades.

### Indicador de referencia: secuencial vs paralelo

El arnés en `code/main.py`Se ejecuta en 1800 ms total. Se ejecuta en paralelo en max ((400, 600, 800) = 800 ms. La diferencia es constante, no proporcional, por lo que los ahorros crecen con el número de herramientas.

Advertencia en el mundo real: llamadas paralelas ponen en riesgo las APIs en el aguas abajo. Una ventaja de 10 vías a un servicio limitado de velocidad fallará. La fase 13 · 17 cubre la presión de retroceso a nivel de puerta de entrada; se planea volver a probar la semántica para una fase futura.

### Reloj de pared de ventilador en streaming

Si el modelo en sí mismo transmite, puede comenzar a ejecutar tan pronto como los argumentos de una llamada estén completos, en lugar de esperar a que todas las llamadas se finalicen. Esta es una optimización de los documentos OpenAI pero no todos los SDK exponen. El arnés de esta lección lo hace: tan pronto como el flujo simulado produzca un objeto de argumento completo, el host inicia esa llamada.

```figure
tp-parallel-fanout
```

## Usalo

`code/main.py`La primera ejecuta tres llamadas meteorológicas simuladas secuencialmente y en paralelo utilizando`concurrent.futures.ThreadPoolExecutor`La segunda mitad reproduce una respuesta de transmisión falsa  trozos de `arguments`para tres llamadas paralelas entrelazadas en una corriente  y las reúne por identificación con `StreamAccumulator`No LLM, no red, sólo la lógica de reensamblaje.

Qué ver:

- El temporizador secuencial alcanza 1,8 segundos y el temporizador paralelo alcanza 0,8 segundos en las mismas latencias falsas.
- El acumulador maneja los trozos que llegan fuera de orden mediante el amortiguamiento por identificación y el análisis solo cuando el JSON de cada llamada esté completo.
- El ejecutor comienza tan pronto como finalizan los argumentos de un ID, no después de que terminen todas las corrientes.

## Envío

Esta lección produce`outputs/skill-parallel-call-safety-check.md`. Dado un registro de herramientas, las auditorías de habilidades que son seguras para paralelar las herramientas, que tienen dependencias de orden y que abrumarían los límites de tasas a continuación  devolver un registro revisado con herramienta por herramienta `parallel_safe`las banderas.

## Los ejercicios

1. - ¿ Qué ?`code/main.py`Confirmar que la relación paralelo-secuencial es aproximadamente `max/sum`(las carreras reales se desvían ligeramente de la ideal debido a la programación de hilos, la serialización y el gasto superior del arnés).

2. Extensión del acumulador para manejar un caso de "llamada fue cancelada en medio de la corriente" dejando caer su amortiguador y emitiendo un `cancelled`¿Qué proveedor documenta explícitamente este caso?`content_block_stop`La semántica y OpenAI `finish_reason: "length"`el comportamiento.

3. Remplaza el hilo de hilo con `asyncio.gather`Se deberían ver pequeñas ganancias en async debido al menor costo de cambio de contexto, pero sólo si los ejecutores hacen I/O real.

4. Elegir dos herramientas que NO deben estar en paralelo (por ejemplo `create_file`Entonces ...`write_file`Añadir un`ordering_dependency`El sistema de programación de dependencias es el mínimo que una futura fase de ingeniería de agentes formaliza.

5. Lea la sección de llamadas paralelas de funciones de OpenAI y la de Anthropic `disable_parallel_tool_use`Documents. Identifique el tipo de herramienta en el mundo real en el que Anthropic recomienda desactivar el paralelismo.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Parallel tool calls | "Fan-out in one turn" | Model emits multiple tool calls in a single assistant message |
| `parallel_tool_calls` | "OpenAI's flag" | Enable or disable multi-call emission |
| `disable_parallel_tool_use` | "Anthropic's inverse" | Opt-out flag; default is parallel enabled |
| Tool call id | "Correlation handle" | Per-call identifier the result message must echo |
| Accumulator | "Stream buffer" | Per-id string buffer for partial `arguments` chunks |
| Out-of-order completion | "Fastest first" | Parallel calls finish in unpredictable order; ids are the glue |
| Dependency graph | "Ordering constraints" | Tools whose outputs feed into inputs of other tools; cannot parallelize |
| Parse-early trap | "JSON.parse exploded" | Attempting to parse an incomplete `arguments` string |
| `streamFunctionCallArguments` | "Gemini 3 feature" | Streamed argument chunks with unique id per call |
| Completion-order reply | "Don't wait for all" | Reply with results as they arrive, keyed by id |

## Leer más

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) El comportamiento predeterminado y la bandera de exclusión
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use)¿ Qué es esto ?`disable_parallel_tool_use`y lotes de resultados
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) Llamadas paralelas correlacionadas con id de Gemini 3
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) reensamblaje de argumentos en fragmentos para flujos OpenAI
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming)¿ Qué es esto ?`content_block_delta`con`input_json_delta`
