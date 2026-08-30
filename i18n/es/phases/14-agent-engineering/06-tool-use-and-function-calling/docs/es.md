# Uso de herramientas y llamadas de funciones

> Toolformer (Schick et al., 2023) comenzó la anotación de herramientas auto supervisada. Berkeley Function Calling Leaderboard V4 (Patil et al., 2025) establece la barra de 2026: 40% agente, 30% multi-turn, 10% en vivo, 10% no en vivo, 10% alucinación. Se resuelve el giro único.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 minutes

## Objetivos de aprendizaje

- Explica la señal de entrenamiento auto supervisada de Toolformer: mantenga las anotaciones de la herramienta solo cuando la ejecución reduzca la pérdida de los siguientes tokens.
- Nombre de las cinco categorías de evaluación de BFCL V4 y qué medidas cada una.
- Implementar un registro de herramientas stdlib con validación de esquemas, coerción de argumentos y sandboxing de ejecución.
- Diagnóstico de los tres problemas abiertos 2026: cadena de herramientas de horizonte largo, toma de decisiones dinámicas y memoria.

## El problema

El uso temprano de herramientas se preguntó: ¿puede el modelo predecir una llamada de función correcta?

El modelo de evaluación de las herramientas de la empresa de evaluación de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de la calidad de

## El concepto

### El equipo de herramientas (Schick et al., NeurIPS 2023)

Idea: dejar que el modelo anote su propio corpus de preentrenamiento con llamadas de API de candidatos. Para cada candidato, ejecutarlo. Mantenga la anotación solo si la inclusión del resultado de la herramienta reduce la pérdida en el siguiente token.

Las herramientas cubiertas: calculadora, sistema de calificación, motores de búsqueda, traductor, calendario. La señal de autocontrol se refiere puramente a si la herramienta ayuda a predecir el texto  no etiquetas humanas.

Resultado de escala: el uso de herramientas surge a escala. Los modelos más pequeños se ven perjudicados por las anotaciones de herramientas; los modelos más grandes ganan. Esta es la razón por la cual los modelos fronterizos 2026 tienen un fuerte uso de herramientas incorporados, mientras que la mayoría de los modelos 7B necesitan una ajuste de uso de herramientas explícito para ser confiables.

### El nivel de clasificación de las funciones de Berkeley V4 (Patil et al., ICML 2025)

BFCL es la evaluación de facto de 2026.

- **Agentic (40%)** trayectorias de agentes completos: memoria, decisiones de varios turnos y dinámicas.
- **Multi-Turn (30%)** conversaciones interactivas con cadenas de herramientas.
- **Live (10%)** Invitaciones reales presentadas por el usuario (distribución más dura).
- **Non-Live (10%)** casos de ensayo sintéticos.
- **Hallucination (10%)** detectar cuando no se debe llamar a ninguna herramienta.

V3 introdujo la evaluación basada en el estado: después de una secuencia de herramientas, compruebe el estado real de la API (por ejemplo, "¿se ha creado el archivo?") en lugar de coincidir con el AST de las llamadas de la herramienta. V4 agregó categorías de búsqueda web, memoria y sensibilidad al formato.

En el 2026 se encontró que la llamada de la función de giro único está casi resuelta. Las fallas se concentran en la memoria (cargar con el contexto a través de los turnos), la toma de decisiones dinámicas (escoler herramientas basadas en resultados previos), cadenas de horizonte largo (drift después de más de 20 pasos) y la detección de alucinaciones (rechazar llamar cuando ninguna herramienta encaja).

### Esquema de herramienta

Cada proveedor tiene un esquema. difieren en detalles pero comparten la misma forma:

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Utilizaciones antropológicas `input_schema`Directamente. OpenAI utiliza`function.parameters`. Ambos aceptan JSON Schema. Las descripciones son cargadoras  el modelo las lee para elegir la herramienta correcta. Las descripciones de herramientas malas son la causa principal de fallos de herramientas seleccionadas incorrectamente.

### Validación de los argumentos

No confía en ninguna llamada de herramienta.

1. **Type coercion.**El modelo puede devolver una cadena "5" donde el esquema dice int. Forzar si no es ambigua; rechazar si no.
2. **Enum validation.**Si el esquema dice `status in {"open", "closed"}`y las emisiones de modelo `"in_progress"`, rechazar con un error descriptivo.
3. **Required fields.**Falta el campo requerido -> observación de error inmediato de vuelta al modelo, no un accidente.
4. **Format validation.**Datos, correos electrónicos, URLs  validar con parseres de concreto, no regex.

Cada falla de validación debe devolver una observación estructurada para que el modelo pueda volver a intentar con la forma correcta.

### Llamadas paralelas de herramientas

Los proveedores modernos admiten llamadas paralelas de herramientas en un solo turno de asistente.

1. El modelo emite 3 llamadas de herramientas con distinción `tool_use_id`S.
2. El tiempo de ejecución los ejecuta (en paralelo si es independiente).
3. Cada resultado se remonta a como un`tool_result`bloque correlacionado por `tool_use_id`¿ Qué ?

Regla de ingeniería: tratar las identificaciones de correlación como carga, cambiarlas y obtener la ruta de herramienta a resultado equivocado.

### El sandboxing

La ejecución de herramientas es el límite de la caja de arena. Véase la lección 09 para detalles. versión corta: cada herramienta debe especificar la superficie de lectura/escritura, acceso a la red, tiempo de espera, límite de memoria.`run_shell(cmd)`es una bandera roja; específico `git_status()`Es más seguro.

```figure
tool-routing
```

## Construye el mismo

`code/main.py`Implemente un registro de herramientas de forma de producción:

- Validador de subconjunto de JSON Schema (sólo stdlib).
- Registro de herramienta con descripción, esquema de entrada, tiempo de espera y ejecutor.
- La coerción de argumentos y la validación enum.
- Envío paralelo de herramientas con identificadores de correlación.
- Observaciones de errores como cadenas estructuradas.

- ¿Qué quieres decir ?

```
python3 code/main.py
```

El rastro muestra a un mini agente llamando a tres herramientas en un turno, con una llamada deliberadamente malformada que es rechazada con un error descriptivo en el que el modelo puede actuar.

## Usalo

Cada proveedor tiene su propio esquema de herramientas  Antropic, OpenAI, Gemini, Bedrock. Utilice una capa de traducción (OpenAI Agents SDK, Vercel AI SDK, LangChain Tool Adapter) si necesita multi-proveedor. BFCL es el punto de referencia  ejecutarlo contra su agente antes de enviar si el uso de herramientas es central para el producto.

## Envío

`outputs/skill-tool-registry.md`genera un catálogo de herramientas, esquema y registro para un dominio de tarea determinado. Incluye controles de calidad de descripción (¿dice la descripción de cada herramienta al modelo cuándo usarla?).

## Los ejercicios

1. Añadir una herramienta "no-op" que permite al modelo rechazar explícitamente el uso de cualquier otra herramienta.
2. Implemente la coerción de argumentos para int-as-string y float-as-string. ¿Dónde comienza la coerción a ocultar insectos reales?
3. Añadir un tiempo de espera por herramienta y un interruptor de circuito (rechazar la herramienta durante 60 años después de 3 fallos consecutivos). ¿Qué cambia esto en la forma en que el modelo se recupera?
4. Lea la descripción de BFCL V4. Seleccione una categoría (por ejemplo, "multi-turn") y ejecute 10 instrucciones de ejemplo a través de su agente.
5. ¿Qué fue lo que Pydantic/Zod captó que el juguete perdió?

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Function calling | "Tool use" | Structured-output tool invocation with validated schema |
| Toolformer | "Self-supervised tool annotation" | Schick 2023 — keep tool calls whose results reduce next-token loss |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 benchmark: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination |
| Tool schema | "Function signature for the model" | name, description, JSON Schema of arguments |
| tool_use_id | "Correlation ID" | Ties a tool call to its result; essential for parallel dispatch |
| Hallucination detection | "Know when not to call" | V4 category: refuse to call when no tool fits |
| Argument coercion | "String-to-int repair" | Narrow fixes for predictable schema-mismatch; reject if ambiguous |
| Sandboxing | "Tool execution boundary" | Per-tool read/write surface, network, timeout, memory cap |

## Leer más

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) Anotado de herramientas auto supervisadas
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) Valoración de referencia de 2026
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) esquema de herramientas de producción en el SDK de Claude Agent
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) Tipo de herramienta de función y Guardrails
