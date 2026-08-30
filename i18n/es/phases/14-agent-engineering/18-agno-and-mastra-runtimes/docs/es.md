# Tiempos de ejecución de los agentes de producción  Instalación rápida y flujos de trabajo tipografizados

> Un agente de producción optimiza el tiempo de ejecución de lo que los marcos de prototipos ignoran: el costo de instanciación, las superficies de flujo de trabajo tipografadas y un backend listo para la entrega. El emparejamiento de 2026: Agno (Python) tiene como objetivo la instanciación de agente de microsecondas y los backends FastAPI sin estado. Mastra envía agentes, herramientas, flujos de trabajo, enrutamiento de modelo unificado y almacenamiento compuesto en el sustrato de Vercel AI SDK.

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## Objetivos de aprendizaje

- Identifique los objetivos de rendimiento de Agno y cuándo importan.
- Nombre de los tres primitivos de Mastra  Agentes, Herramientas, Flujos de Trabajo  y los adaptadores de servidor compatibles.
- Explica por qué un backend FastAPI sin estado y con escala de sesión es la vía de producción Agno recomendada.
- Seleccione Agno vs Mastra para una pila dada (Python-primero vs TypeScript-primero).

## El problema

LangGraph, AutoGen, CrewAI son un sistema de trabajo muy pesado. Los equipos que quieren "solo el bucle de agente, rápido, en mi tiempo de ejecución" pueden llegar a Agno (Python) o Mastra (TypeScript). Ambos intercambian algunos de los primitivos de propiedad del sistema por velocidad bruta y un ajuste más ajustado a la pila circundante.

## El concepto

### Agno

- Python runtime, anteriormente Phi-data.
- "Sin gráficos, cadenas o patrones complicados, solo pitón puro".
- Objetivos de rendimiento de sus documentos: ~ 2μs instanciación de agente, ~ 3,75 KiB de memoria por agente, ~ 23 proveedores de modelos.
- Camino de producción: backend FastAPI sin estado de sesión. Cada solicitud inicia un nuevo agente; el estado de sesión vive en un DB.
- Multimodal nativo (texto, imagen, audio, video, archivo) y RAG agente.

Los objetivos de velocidad importan cuando tienes miles de agentes de corta duración por segundo (fán de chat, canalizaciones de evaluación), pero son menos importantes cuando un agente corre durante 10 minutos.

### El Mastra

- TypeScript, construido en el SDK de Vercel AI.
- Tres primitivos:**Agents**¿ Qué ?**Tools**(Tipo de zona), **Workflows**¿ Qué ?
- Modelo unificado de router  3.300+ modelos en 94 proveedores (marzo 2026).
- Almacenamiento compuesto: memoria, flujos de trabajo, observabilidad a diferentes fondos; ClickHouse recomendado para observabilidad a escala.
- Apache 2.0 con `ee/`directorios con licencia empresarial disponible en la fuente.
- Adaptadores de servidores para Express, Hono, Fastify, Koa; integración de primera clase Next.js y Astro.
- Naves Mastra Studio (host local:4111) para el depuración.
- 22k+ GitHub estrellas, 300k+ descargas semanales en 1.0 (Jan 2026).

### Posicionamiento

Tampoco se trata de ser LangGraph.

- **Language fit.**Agno para equipos de Python; Mastra para TypeScript.
- **Runtime ergonomics.**Agno = casi cero gastos generales; Mastra = integrado con el ecosistema Vercel.
- **Observability.**Ambos se integran con Langfuse/Phoenix/Opik (Lección 24) pero Mastra Studio es de primera parte.

### ¿Cuándo elegir cada uno?

- **Agno**Python backend, muchos agentes de corta duración, fuertes requisitos de perf, tienda FastAPI.
- **Mastra** Backend de TypeScript, Next.js / Vercel desplegado, enrutamiento unificado de modelos multi-proveedor, herramientas tipo Zod.
- **LangGraph**(Lección 13)  cuando el estado duradero y el razonamiento gráfico explícito importan más que la velocidad bruta.
- **OpenAI / Claude Agent SDK** cuando se desea la forma productiva del proveedor (lecciones 1617).

### Cuando este patrón va mal

- **Perf-for-perf's-sake.**Elegir Agno porque "2μs" suena bien cuando la carga de trabajo es una llamada lenta de agente por solicitud.
- **Ecosystem lock-in.**La integración con sabor a Vercel de Mastra es un plus en Vercel, un menos en otros lugares.
- **Enterprise license confusion.**El de Mastra.`ee/`Los directorios están disponibles en fuente, no Apache 2.0.

```figure
wb-runtime-spawn
```

## Construye el mismo

Esta lección es principalmente comparativa  ningún artefacto de código único haría justicia a ambos marcos. Ver `code/main.py`para un juguete lado a lado: un mínimo de " ejecutar un agente, transmitir la salida, persistir sesión " flujo implementado dos veces (una vez en forma de Agno, una vez en forma de Mastra).

- ¿Qué quieres decir ?

```
python3 code/main.py
```

Dos rastros estructuralmente diferentes pero funcionalmente equivalentes.

## Usalo

- **Agno** Backend Python que necesita velocidad y forma FastAPI.
- **Mastra** Backend de TypeScript con muchos proveedores y primitivos de flujo de trabajo.
- Ambos barcos tienen ganchos de observación de primera parte.

## Envío

`outputs/skill-runtime-picker.md`elige Agno, Mastra, LangGraph o un SDK de proveedor basado en la pila, el presupuesto de latencia y la forma operativa.

## Los ejercicios

1. Lea los documentos de Agno, lleva el ciclo de ReAct a Agno. ¿Qué ha desaparecido?
2. Lea los documentos de Mastra. Portar el mismo bucle a Mastra. ¿Qué cambió en la mecanografía de herramientas (Zod vs nada)?
3. Métese la latencia de instanciación de agente en su pila. ¿Las 2μs de Agno importan para su carga de trabajo?
4. Diseñar una migración: si has estado ejecutando CrewAI en Python, ¿qué se rompe si te mueves a Agno?
5. Lea el libro de Mastra `ee/`¿Qué restricciones afectarían a un fork de código abierto?

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agno | "Fast Python agents" | Stateless session-scoped agent runtime |
| Mastra | "TypeScript agents on Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "Multi-provider access" | Single client for 3,300+ models across 94 providers |
| Composite storage | "Multiple backends" | Memory/workflows/observability each to a different store |
| Mastra Studio | "Local debugger" | localhost:4111 UI for introspecting agents |
| Source-available | "Not OSS" | License permits source reading but restricts commercial use |

## Leer más

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) objetivos de rendimiento, integración de FastAPI
- [Mastra docs](https://mastra.ai/docs) primitivos, adaptadores de servidores, modelo de enrutador
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) la alternativa de gráfico estatal
- [Comet Opik](https://www.comet.com/site/products/opik/) comparaciones de observabilidad citadas por las integraciones de Mastra
