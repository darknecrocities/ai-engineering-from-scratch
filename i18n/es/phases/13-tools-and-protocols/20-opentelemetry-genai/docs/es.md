# OpenTelemetry GenAI  Herramienta de seguimiento de llamadas de extremo a extremo

> Un agente llama a cinco herramientas, tres servidores MCP y dos subagentes. Necesitas un rastro en todo. Las convenciones semánticas de OpenTelemetry GenAI (atributos estables en v1.37 y superior) son el estándar 2026, nativo apoyado por Datadog, Langfuse, Arize Phoenix, OpenLLMetry y AgentOps. Esta lección nombra los atributos requeridos, recorre la jerarquía de espacio (herramienta agente → LLM →), y envía un emisor de espacio de espacio stdlib que puede conectar a cualquier exportador OTel.

**Type:** Build
**Languages:** Python (stdlib, OTel span emitter)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## Objetivos de aprendizaje

- Nombre de los atributos OTel GenAI requeridos para un período de LLM y un período de ejecución de herramientas.
- Construir una jerarquía de rastreo que cubra el bucle de agentes, llamada LLM, llamada de herramientas y envío de clientes MCP.
- Decidir qué contenido capturar (opt-in) vs redactar (por defecto).
- Emite extensiones a un coleccionista local (Jaeger, Langfuse) sin reescribir el código de la herramienta.

## El problema

Un defecto de febrero de 2026: el usuario informa "mi agente a veces tarda 30 segundos en responder; otras veces 3 segundos". No hay rastro. Los registros muestran la llamada de LLM, pero no el envío de la herramienta, no el servidor MCP ida y vuelta, no el sub-agente.

Sin rastreo de extremo a extremo, no se puede encontrar esto.

Las convenciones se establecieron en 2025-2026 bajo el grupo de convenciones semánticas OpenTelemetry. Definen nombres de atributos estables para que Datadog, Langfuse, Phoenix, OpenLLMetry y AgentOps analicen todos los mismos intervalos.

## El concepto

### Jerarquía de la España

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

Todo se anida bajo una identificación de rastro.

### Los atributos requeridos

En el semestre 2025-2026, se aplicarán:

- `gen_ai.operation.name`¿ Qué es esto ?`"chat"`¿ Qué ?`"text_completion"`¿ Qué ?`"embeddings"`¿ Qué ?`"execute_tool"`¿ Qué ?`"invoke_agent"`¿ Qué ?
- `gen_ai.provider.name`¿ Qué es esto ?`"openai"`¿ Qué ?`"anthropic"`¿ Qué ?`"google"`¿ Qué ?`"azure_openai"`¿ Qué ?
- `gen_ai.request.model` cadena de modelo solicitada (por ejemplo `"gpt-4o-2024-08-06"`¿Qué es lo que se hace?
- `gen_ai.response.model` el modelo realmente sirvió.
- `gen_ai.usage.input_tokens`- ¿ Qué ?`gen_ai.usage.output_tokens`¿ Qué ?
- `gen_ai.response.id` Identificación de respuesta del proveedor para correlación.

Para las extensiones de las herramientas:

- `gen_ai.tool.name` Identificador de herramienta.
- `gen_ai.tool.call.id` el identificador de llamada específico.
- `gen_ai.tool.description` Descripción de la herramienta (opcional).

Para las franjas de agentes:

- `gen_ai.agent.name`- ¿ Qué ?`gen_ai.agent.id`- ¿ Qué ?`gen_ai.agent.description`¿ Qué ?

### Tipos de espinacas

- `SpanKind.CLIENT`para llamadas que cruzan un límite de proceso (proveedor de LLM, servidor MCP).
- `SpanKind.INTERNAL`para los pasos del propio bucle del agente y la ejecución de la herramienta.

### Captura de contenido de opción

Por defecto, los intervalos llevan métricas y tiempos  no instrucciones o completos.`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`El contenido de la información en el archivo de datos de la empresa es el contenido de la información de la empresa.

### Eventos en las franjas

Los eventos de nivel de tokens se pueden agregar como eventos de período:

- `gen_ai.content.prompt` mensajes de entrada.
- `gen_ai.content.completion` mensajes de salida.
- `gen_ai.content.tool_call` llamada de herramienta tal como se grabó.

Se realizan eventos en orden temporal dentro de un lapso para una reproducción detallada.

### Exportadores

OTel abarca las exportaciones a:

- **Jaeger / Tempo.**OSS, en el lugar.
- **Langfuse.**Específico de observabilidad de LLM; visualiza el uso de tokens.
- **Arize Phoenix.**Evals + trazaje combinado.
- **Datadog.**Comercial; nativo parses `gen_ai.*`Los atributos.
- **Honeycomb.**Orientación a columnas; amigable con las consultas.

Todos hablan OTLP, el formato de cable.

### Propagación a través de los MCP

Cuando un cliente MCP llama a un servidor, inyecta el encabezado traceparent W3C en la solicitud. Streamable HTTP admite encabezados estándar. Stdio no lleva encabezados HTTP de forma nativa; la hoja de ruta 2026 de la especificación discute la adición de un `_meta.traceparent`campo en llamadas JSON-RPC.

Hasta que los buques: incluyan el rastreador en el `_meta`El servidor registra la identificación de rastreo.

### Las métricas

Junto con las extensiones, el semconv de la GenAI define métricas:

- `gen_ai.client.token.usage` histograma.
- `gen_ai.client.operation.duration` histograma.
- `gen_ai.tool.execution.duration` histograma.

Utilice estos para los paneles que no necesitan detalles por llamada.

### Capas de agenteOps

AgentOps (fundado en 2024) se especializa en observabilidad GenAI. Envuelve marcos populares (LangGraph, Pydantic AI, CrewAI) para emitir espacios OTel automáticamente.

```figure
t3-span-waterfall
```

## Usalo

`code/main.py`Emite extensiones en forma de OTel a un stdout (en formato similar a OTLP-JSON) para un agente que llama a un LLM, envía dos herramientas y hace un viaje de ida y vuelta de MCP. Ningún exportador real  la lección se centra en el conjunto de formas y atributos de extensión. Pegar la salida en un espectador compatible con OTLP o simplemente leerlo.

Qué ver:

- El rastro de identidad se comparte en todos los espacios.
- Los vínculos entre padres e hijos se codifican a través de `parentSpanId`¿ Qué ?
- Requerido `gen_ai.*`Los atributos están poblados.
- La captura de contenido está apagada por defecto; un escenario lo activa a través de env var.

## Envío

Esta lección produce`outputs/skill-otel-genai-instrumentation.md`- Dado que la base de código de agentes, la habilidad produce un plan de instrumentación: dónde añadir las extensiones, qué atribuye a la población y qué exportadores se dirigen.

## Los ejercicios

1. - ¿ Qué ?`code/main.py`Cuenta los intervalos e identifica cuál es el cliente vs. interno.

2. Activar la captura de contenido (env var) y confirmar `gen_ai.content.prompt`y `gen_ai.content.completion`No obstante, el informe de la Comisión no se aplica a los Estados miembros.

3. Añadir la métrica de ejecución de la herramienta `gen_ai.tool.execution.duration`y emitirlo como muestra de histograma por llamada.

4. Propagar un rastreador de un agente padre en el espacio de una solicitud de MCP `_meta.traceparent`Verifique si el servidor MCP vería la misma identificación de rastreo.

5. Lea la especificación de semconv de OTel GenAI. Identifique un atributo que no emite el código de esta lección en el semconv. Añade.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | Open standard for traces, metrics, logs |
| GenAI semconv | "GenAI semantic conventions" | Stable attribute names for LLM / tool / agent spans |
| `gen_ai.*` | "The attribute namespace" | All GenAI attributes share this prefix |
| Span | "Timed operation" | A unit of work with a start, end, and attributes |
| Trace | "Cross-span ancestry" | Tree of spans sharing a trace id |
| SpanKind | "CLIENT / SERVER / INTERNAL" | Hints about span direction |
| OTLP | "OpenTelemetry Line Protocol" | Wire format for exporters |
| Opt-in content | "Prompt / completion capture" | Off by default; env var to enable |
| traceparent | "W3C header" | Propagates trace context across services |
| Exporter | "Backend-specific shipper" | Component that sends spans to Jaeger / Datadog / etc. |

## Leer más

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) convenciones canónicas para los espacios, métricas y eventos de GenAI
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) Lista de atributos de duración de la MLL y de la ejecución de las herramientas
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) A nivel de agente `invoke_agent`el tiempo
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) Fuente de verdad alojada en GitHub
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) Integrar la producción a través de la marcha
