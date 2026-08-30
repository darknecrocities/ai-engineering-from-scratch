# Función llamada Deep Dive  OpenAI, Antropic, Gemini

> Los tres proveedores fronterizos convergieron en el mismo ciclo de llamadas de herramientas en 2024 y luego divergieron en todo lo demás.`tools`y `tool_calls`. Uso antropológico `tool_use`y `tool_result`Los Géminis usan`functionDeclarations`Esta lección diferencia los tres lados a la vez para que el código que se envía en un proveedor no se rompa cuando se porta.

**Type:** Build
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** Phase 13 · 01 (the tool interface)
**Time:** ~75 minutes

## Objetivos de aprendizaje

- En el caso de las cargas de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de
- Traducir una declaración de herramienta en los tres formatos de proveedor y predecir dónde las restricciones de modo estricto serán diferentes.
- Usar`tool_choice`en cada proveedor para forzar, prohibir o seleccionar automáticamente las llamadas de herramientas.
- Conozca los límites de durabilidad de cada proveedor (conto de herramientas, profundidad de esquema, longitud de argumento) y las firmas de error emitidas cuando se violan los límites.

## El problema

La forma de una solicitud de llamada a funciones difiere de un proveedor a otro.

**OpenAI Chat Completions / Responses API.**Pasas .`tools: [{type: "function", function: {name, description, parameters, strict}}]`La respuesta del modelo contiene `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`donde`arguments`Es una cadena JSON que debe analizar.`strict: true`) impone el cumplimiento de los esquemas mediante la decodificación restringida.

**Anthropic Messages API.**Pasas .`tools: [{name, description, input_schema}]`La respuesta es:`content: [{type: "text"}, {type: "tool_use", id, name, input}]`- ¿ Qué ?`input`Se ha analizado ya (un objeto, no una cadena).`user`mensaje que contiene un `{type: "tool_result", tool_use_id, content}`El bloque.

**Google Gemini API.**Pasas .`tools: [{functionDeclarations: [{name, description, parameters}]}]`(nido bajo `functionDeclarations`La respuesta llega como `candidates[0].content.parts: [{functionCall: {name, args, id}}]`donde`id`Es único en Gemini 3 y arriba para correlación de llamadas paralelas.`{functionResponse: {name, id, response}}`¿ Qué ?

El mismo bucle, diferentes nombres de campos, diferentes anidamientos, diferentes convenciones de cuerda contra objeto, diferentes mecanismos de correlación. Un equipo que escribe un agente meteorológico en OpenAI paga un puerto de dos días a Anthropic y otro día a Gemini solo por la tubería.

Esta lección construye un traductor que unifica los tres formatos en una declaración de herramienta canónica y rutas en el borde.

## El concepto

### La estructura común

Cada proveedor necesita cinco cosas:

1. **Tool list.**Nombre, descripción y esquema de entrada por herramienta.
2. **Tool choice.**Forzar una herramienta específica, prohibir las herramientas, o dejar que el modelo decida.
3. **Call emission.**Elaboración estructurada de nombres de la herramienta y de los argumentos.
4. **Call id.**Correlación de la respuesta a la llamada correcta (materiales para el paralelo).
5. **Result injection.**Un mensaje o bloqueo que vincula el resultado a la llamada.

### Diferencias de forma, campo por campo

| Aspect | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| Declaration envelope | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema field | `parameters` | `input_schema` | `parameters` |
| Response container | `tool_calls[]` on assistant message | `content[]` of type `tool_use` | `parts[]` of type `functionCall` |
| Arguments type | stringified JSON | parsed object | parsed object |
| Id format | `call_...` (OpenAI generates) | `toolu_...` (Anthropic) | UUID (Gemini 3+) |
| Result block | role `tool`, `tool_call_id` | `user` with `tool_result`, `tool_use_id` | `functionResponse` with matching `id` |
| Force-a-tool | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| Forbid tools | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| Strict schema | `strict: true` | schema-is-schema (always enforced) | `responseSchema` at request level |

### Los límites que realmente alcanzarás

- **OpenAI.**128 herramientas por solicitud. profundidad de esquema 5. cadena de argumentos <= 8192 bytes.`$ref`No , no .`oneOf`- ¿ Qué ?`anyOf`- ¿ Qué ?`allOf`con superposición, todas las propiedades enumeradas en `required`¿ Qué ?
- **Anthropic.**64 herramientas por solicitud. La profundidad del esquema es efectivamente ilimitada pero un límite práctico 10.
- **Gemini.**64 funciones por solicitud. Los tipos de esquemas son un subconjunto de OpenAPI 3.0 (una ligera divergencia con JSON Schema 2020-12).

### `tool_choice`comportamiento

Tres modos que todos soportan, nombrados de manera diferente.

- **Auto.**El modelo elige herramienta o texto.
- **Required / Any.**El modelo debe llamar al menos a una herramienta.
- **None.**El modelo no debe llamar a herramientas.

Además de un modo único para cada proveedor:

- **OpenAI.**Forza una herramienta específica por nombre.
- **Anthropic.**Forzar una herramienta específica por nombre; `disable_parallel_tool_use`bandera separa un solo versus varios.
- **Gemini.** `mode: "VALIDATED"`envía cada respuesta a través de un validador de esquema independientemente de la intención del modelo.

### Llamadas paralelas

La apertura de la AIE `parallel_tool_calls: true`(por defecto) emite varias llamadas en un mensaje de asistente.`tool_call_id`. Antropic históricamente hizo una sola llamada;`disable_parallel_tool_use: false`Gemini 2 permitió llamadas paralelas pero no dio id estables; Gemini 3 añade UUIDs para que las respuestas fuera de orden se correlacionen de forma limpia.

### En streaming

Los tres soportan las llamadas de herramientas de transmisión.

- **OpenAI.**Los trozos del delta de `tool_calls[i].function.arguments`Se acumulan hasta que...`finish_reason: "tool_calls"`¿ Qué ?
- **Anthropic.**Eventos de inicio de bloqueo / delta de bloqueo / parada de bloqueo. `input_json_delta`Los trozos llevan argumentos parciales.
- **Gemini.** `streamFunctionCallArguments`(nuevo en Gemini 3) emite trozos con un `functionCallId`para que múltiples llamadas paralelas puedan intercalarse.

La fase 13 · 03 se centra en la forma de declaración y de llamada única.

### Errores y reparaciones

Los errores de argumento inválido también se ven diferentes.

- **OpenAI (non-strict).**Modelo de devoluciones `arguments: "{bad json}"`Si el análisis JSON falla, se inyecta un mensaje de error y se vuelve a llamar.
- **OpenAI (strict).**La validación ocurre durante la descodificación; JSON inválido es imposible pero `refusal`puede aparecer.
- **Anthropic.** `input`Puede contener campos inesperados; esquema es aconsejable. Valida el lado del servidor.
- **Gemini.**La peculiaridad de OpenAPI 3.0: `enum`en campos de objetos silenciosamente ignorados; validar a sí mismo.

### El patrón de traducción

Una declaración de herramienta canónica en su código se ve así (se elige la forma):

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

Tres funciones diminutas lo traducen a las tres formas de proveedor.`code/main.py`No se requiere red  esta lección enseña las formas, no el HTTP.

Los equipos de producción envuelven este traductor en `AbstractToolset`(IA de la Pidantica),`UniversalToolNode`(LangGraph), o `BaseTool`La fase 13 · 17 envía una puerta de entrada que expone una API en forma de OpenAI frente a cualquiera de las tres.

```figure
function-call-args
```

## Usalo

`code/main.py`define un canónico `Tool`Dataclass y tres traductores que emiten la declaración OpenAI, Anthropic y Gemini JSON. Luego analiza una respuesta de proveedor hecha a mano de cada forma en el mismo objeto de llamada canónica, demostrando que la semántica es idéntica debajo de la piel.

Qué ver:

- Los tres bloques de declaración difieren sólo en el nombre de los envelope y de los campos.
- Los tres bloques de respuesta difieren en el lugar donde se realiza la llamada (nivel superior `tool_calls`¿ Qué ?`content[]`bloque,`parts[]`de entrada).
- Uno .`canonical_call()`extractos de funciones `{id, name, args}`de las tres formas de respuesta.

## Envío

Esta lección produce`outputs/skill-provider-portability-audit.md`. Dado que se integra una llamada de función contra un proveedor, la habilidad produce una auditoría de portabilidad: qué proveedor limita a qué se basa, qué campos necesitan ser renombrados y qué rompe cuando se porta a otro proveedor.

## Los ejercicios

1. - ¿ Qué ?`code/main.py`y comprobar que los tres proveedores de declaraciones JSONs todos serializan la misma subyacente `Tool`Modificar la herramienta canónica para agregar un parámetro enum y confirmar que sólo el traductor Gemini necesita manejar la peculiaridad de OpenAPI.

2. Añadir un`ListToolsResponse`parser para cada proveedor que extrae la lista de herramientas un modelo devuelve después de un `list_tools`OpenAI no tiene uno nativo; note esta asimetría.

3. Implementación `tool_choice`conversión: mapa de un canónico `ToolChoice(mode="force", tool_name="x")`en las tres formas del proveedor.`mode="any"`y `mode="none"`- Revisa la tabla de diferencias de la lección.

4. Seleccione uno de los tres proveedores y lea su guía de llamadas de función de extremo a extremo. Encuentre un campo en su especificación de esquema que los otros dos no admitan.`strict`, Antropical `disable_parallel_tool_use`, Géminis .`function_calling_config.allowed_function_names`¿ Qué ?

5. Escriba un vector de prueba: una llamada de herramienta cuyos argumentos violan el esquema declarado. ejecutarlo a través del validador de cada proveedor (el stdlib en la lección 01 servirá como proxy) y registrar qué errores se producen. Documento que usted usaría en la producción para la estrictidad.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Function calling | "Tool use" | Provider-level API for structured tool-call emission |
| Tool declaration | "Tool spec" | Name + description + JSON Schema input payload |
| `tool_choice` | "Force / forbid" | Auto / required / none / specific-name modes |
| Strict mode | "Schema enforcement" | OpenAI flag that constrains decoding to match schema |
| `tool_use` block | "Anthropic's call shape" | Inline content block with id, name, input |
| `functionCall` part | "Gemini's call shape" | A `parts[]` entry containing name, args, and id |
| Arguments-as-string | "Stringified JSON" | OpenAI returns args as a JSON string, not an object |
| Parallel tool calls | "Fan-out in one turn" | Multiple tool calls in one assistant message |
| Refusal | "Model declines" | Strict-mode-only refusal block instead of a call |
| OpenAPI 3.0 subset | "Gemini schema quirk" | Gemini uses a JSON-Schema-like dialect with minor differences |

## Leer más

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) Referencia canónica, incluidos los modos estrictos y las llamadas paralelas
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)¿ Qué es esto ?`tool_use`y `tool_result`semántica de bloque
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) llamadas paralelas, ID únicos y subconjunto OpenAPI
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) Superficie de la empresa de Géminis
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) Detalles de aplicación de esquemas de modo estricto
