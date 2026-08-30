# Selección de la pila de observabilidad del LLM

> El mercado de observabilidad de 2026 se divide en dos categorías. Las plataformas de desarrollo (LangSmith, Langfuse, Comet Opik) incluyen monitoreo con evaluaciones, gestión de la sesión, repeticiones de sesiones. Las herramientas de acceso/instrumentación (Helicone, SigNoz, OpenLLMetry, Phoenix) se centran en la telemetría. Langfuse es un núcleo con licencia MIT con un fuerte balance OSS (50K eventos / mes en la nube gratuita). Phoenix es OpenTelemetry nativo bajo la Licencia Elastic 2.0  excelente para la visualización de deriva / RAG, no un backend de producción persistente. Arize AX utiliza la integración de Iceberg/Parquet de copia cero que afirma que es 100 veces más barato que la observabilidad monolítica. LangSmith lidera para LangChain/LangGraph, $ 39 / usuario / mes, auto-host en Enterprise sólo. Helicone es basado en proxy con configuración de 15-30 minutos, 100K req / mo libre, pero menos profundidad en las huellas del agente. Modelo de producción común: Gateway (Helicone/Portkey) + plataforma eval (Phoenix/TruLens) pegada por OpenTelemetry.

**Type:** Learn
**Languages:** Python (stdlib, toy trace-sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Objetivos de aprendizaje

- Distinguir las plataformas de desarrollo (agrupadas: evals + prompts + sesiones) de las herramientas de gateway/telemetría (solo traces + métricas).
- Mapa de seis herramientas principales (Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik) a sus casos de licencia, precios y uso de puntos dulces.
- Explica el patrón de cola de OpenTelemetry que le permite combinar una herramienta de pasarela con una plataforma de evaluación separada.
- Nombre el diferenciador de costos para 2026 (enfoque de copia cero de Arize AX vs ingesta monolitica) y indique el multiplicador aproximado de 100x.

## El problema

Usted envió una función de LLM. Funciona. No tiene visibilidad en fallos rápidos, bucles de herramientas, regresiones de latencia, picos de costos o tasa de éxito de caché rápido. Usted Google "observabilidad de LLM" y obtener ocho herramientas todas afirmando que resuelven el mismo problema a tres puntos de precio diferentes.

No resuelven el mismo problema. LangSmith responde "¿Por qué esta ejecución de LangGraph falló?" Phoenix responde "¿Mi tubería RAG está driftando?" Helicone responde "¿Qué aplicación está quemando tokens?" Langfuse responde "¿Puedo auto-host todo el asunto?" diferentes herramientas, diferentes audiencias.

La selección implica cuatro ejes: pila (LangChain? SDK crudo? multivendor?), tolerancia a las licencias (sólo MIT? Elastico OK? multa comercial?), presupuesto (nive libre? $100/mo? $1000/mo?), y auto-host (debe ser agradable de tener? nunca?).

## El concepto

### Dos categorías

**Development platforms**Se ejecuta experimentos, ver qué prompt funcionó, regresión de un conjunto de datos, nuevo prompt contra los viejos ganadores.

**Gateway/telemetry tools**Infraestructura de la información de la información de la información de la información de la información de la información.

### Saldo de la OSS de la Langfuse

- Licenciado por Apache / MIT; auto-host a través de Docker.
- Número gratuito en la nube: 50 mil eventos al mes.
- Evals, gestión rápida, rastreos, conjuntos de datos, cobertura razonable de las cuatro características de la plataforma de desarrollo.
- Un punto positivo: desea características de la clase LangSmith pero debe ser auto-host o permanecer con licencia OSS.

### Phoenix (Arize)  Telemetría-primero, OpenTelemetría-nativo

- Licencia Elastica 2.0; auto-host trivial.
- Excelente en RAG y visualización de deriva.
- No diseñado como backend de producción persistente  observabilidad en el tiempo de desarrollo.
- Punto positivo: desarrollo de tuberías RAG, depuración de deriva, parejas con una puerta de entrada separada para la producción.

### Arize AX  el juego de la escala

- Integración de datos de la laguna de cero copias a través de Iceberg/Parquet.
- Las matemáticas: almacena rastros en su propio Parquet en S3; Arize lee directamente.
- En el caso de los Estados miembros, el número de datos de datos de los centros de investigación y de investigación es de 10 millones.

### LangSmith  LangChain/LangGraph primero

- Comercial, 39 dólares al mes, auto-host sólo en Enterprise.
- Lo mejor de su clase para las pilas de LangChain y LangGraph.
- Un equipo comprometido con LangChain, dispuesto a pagar.

### Helicone  base de proxy mínimo viable

- 15-30 minutos de configuración al cambiar su `OPENAI_API_BASE`a la proxy de Helicone.
- Licenciado por MIT; 100 mil recetas por mes gratis, pagados $20 por mes.
- Incluye fallover, almacenamiento en caché, límites de tasas  actúa como una puerta de entrada también.
- Menos profundidad en las huellas de agente / múltiples pasos.
- Sweet spot: inicio rápido, aplicación de pila única, necesita gateway + observabilidad en uno.

### Opik (Comet)  Plataforma de desarrollo OSS

- Apache 2.0, totalmente OSS.
- Un rasgo similar a Langfuse con herencia cometa.
- Los equipos de ML ya están en Comet, quieren la observabilidad de LLM en el mismo panel.

### SigNoz  OpenTelemetry-first APM completo

- Apache 2.0 maneja APM general más LLM a través de OpenTelemetry.
- Punto dulce: observabilidad unificada entre los servicios y las llamadas de LLM.

### El pegamento: OpenTelemetry + Convenciones semánticas de GenAI

OpenTelemetry publicó las convenciones semánticas de GenAI a finales de 2025 (`gen_ai.system`¿ Qué ?`gen_ai.request.model`¿ Qué ?`gen_ai.usage.input_tokens`Las herramientas que consumen OTel pueden interactuar.

1. Emite OTel con convenciones de GenAI de cada llamada de LLM.
2. Ruta a puerta de entrada (Helicone / Portkey) para el día a día.
3. Plataforma de evaluación de doble nave (Phoenix / Langfuse) para regresiones.
4. Archivo en el lago de datos (Iceberg) para análisis a largo plazo a través de Arize AX o DuckDB.

### La trampa: el instrumento en la capa equivocada

La instrumentación dentro de su marco de agentes (por ejemplo, añadiendo rastros LangSmith) lo une a ese marco. La instrumentación en la capa HTTP/OpenAI-SDK (a través de OpenLLMetry o su puerta de enlace) es portátil.

### Muestras  no se puede guardar todo

En el caso de las solicitudes de más de 1 millón de personas/día, la retención de datos completos cuesta más que las solicitudes de LLM. Muestra por regla: 100% de errores, 100% de alto costo, 5% de éxito.

### Números que debes recordar

- Nube libre de Langfuse: 50K eventos/mes.
- LangSmith: 39 dólares al usuario por mes.
- Helicona libre: 100K reacutivos al mes.
- Arize AX reclama: ~ 100 veces más barato que el monolito a escala.
- Convenciones de OpenTelemetry GenAI: 2025 transporte marítimo, 2026 ampliamente adoptado.

```figure
i4-otel-glue
```

## Usalo

`code/main.py`simula un día de 1M de seguimiento en todas las estrategias de retención (100% ingesta, muestreo, muestreo + errores).

## Envío

Esta lección produce`outputs/skill-observability-stack.md`. Dado el tamaño, la escala, el presupuesto, la postura de la licencia, elige la herramienta (s).

## Los ejercicios

1. Su equipo en LangChain quiere la observabilidad de OSS, elija Langfuse o Opik y justifica.
2. Con 5M traces/día con Datadog cotiza 150K$/mes, computa equilibración para Arize AX.
3. Diseñar un atributo de OpenTelemetry GenAI establecido por la guía de su organización debe exigir en cada llamada de LLM.
4. ¿Debemos discutir si Phoenix es suficiente para la producción?
5. Helicone es 20 ms de carga por proxy. ¿A P99 TTFT 300 ms, es aceptable? ¿Qué pasa si SLA es 100 ms?

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OpenLLMetry | "OTel for LLMs" | Open-source OpenTelemetry instrumentation for LLMs |
| GenAI conventions | "OTel attributes" | Standard OTel attribute names for LLM calls |
| LangSmith | "LangChain observability" | Commercial platform bundled with LangChain ecosystem |
| Langfuse | "OSS LangSmith" | MIT OSS with similar feature set |
| Phoenix | "Arize dev tool" | OpenTelemetry-native dev/eval platform |
| Arize AX | "scale observability" | Commercial zero-copy Iceberg/Parquet observability |
| Helicone | "proxy observability" | HTTP proxy collecting LLM telemetry + gateway features |
| Opik | "Comet LLM" | Apache 2.0 OSS dev platform from Comet |
| Session replay | "trace rerun" | Replay a full agent session with tool calls |
| Eval | "offline test" | Running candidate model/prompt over labeled dataset |

## Leer más

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
