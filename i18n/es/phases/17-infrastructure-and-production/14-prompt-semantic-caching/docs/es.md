# Economía del caché rápido y del caché semántico

> **Pricing snapshot dated 2026-04.**Las afirmaciones numéricas a continuación reflejan las tarjetas de tarifas de los proveedores capturadas en la publicación de esta lección; verifique con los documentos vinculados antes de cotizarlos en aguas subyacentes.

> El caché de instrucciones de L2 (nivel de proveedor) reutiliza la atención KV para los prefijos repetidos  Los documentos de caché de instrucciones de Anthropic anuncian hasta una reducción de costes del 90% y una reducción de latencia del 85% en instrucciones largas; para Claude 3.5 Sonnet se leen en caché $0.30/M vs $3,00/M fresco con un TTL de 5 minutos y una prima de escritura de 2 veces para la opción de TTL de 1 hora (docs.anthropic.com, 2026-04). La caché de instancias OpenAI se aplica automáticamente para las instancias ≥1024 tokens y precios entradas cachées a aproximadamente un descuento del 90% frente a frescos (platform.openai.com, 2026-04); la tasa exacta caché por modelo depende de la tarjeta de tasa en vivo. El L1 (nivel de aplicación) de almacenamiento en caché semántico omite el LLM en su totalidad en la incorporación de hits de similitud. Vendedor "95% de exactitud" se refiere a la corrección de coincidencia, no la tasa de impacto  las tasas de impacto de producción reportadas van desde el 10% (chat abierto) hasta el 70% (FAQ estructurado); ninguno de los proveedores publica una línea de base oficial, por lo que trata esto como telemetría comunitaria en lugar de garantías. Los obstáculos de producción: la paralelalización mata el caché (las solicitudes paralelas emitidas antes de que la primera caché escriba pueden inflar el gasto varias veces), y el contenido dinámico dentro del prefijo evita que el caché golpee por completo. ProjectDiscovery informó que se movió de 7% a 74% de la tasa de hits (2025-11) moviendo el texto dinámico fuera del prefijo cachéable.

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## Objetivos de aprendizaje

- Distinguir entre el caché de prompto/prefijo L2 (reutilización de KV en el proveedor) y el caché semántico L1 (eliminación de LLM en prompts similares).
- Explica el trabajo de Anthropic `cache_control`marcado explícito y las dos opciones TTL (5 minutos vs. 1 hora) con sus multiplicadores de precios.
- Calcule los ahorros mensuales esperados dados el índice de éxito, la mezcla de respuesta/prompto y los precios de los tokens.
- Nombre el patrón anti-paralelalización que infla las facturas en 5-10 veces y el patrón anti-contenido dinámico que colapsar tasa de impacto.

## El problema

Si usted añade la caché de instrucciones a su servicio RAG. La factura se mantiene plana. mide la tasa de hits; es 7%. Sus instrucciones se ven estáticas pero no lo son.

Separadamente, su agente ejecuta diez llamadas paralelas de herramientas por pregunta de usuario. Todas las diez llegan al proveedor antes de que finalice la primera escritura en caché. Diez escribe, cero lee. Su factura es 5-10 veces lo que "con caché" se suponía que costaría.

El caché es un protocolo, no una bandera.

## El concepto

### L2  almacenamiento en caché de los servicios de proveedores

El proveedor almacena el KV de atención para un prefijo cachéable y lo reutiliza en la siguiente solicitud que coincide con el prefijo.

**Anthropic (Claude 3.5 / 3.7 / 4 series)**: explícito `cache_control`TTL: 5 minutos (costos de escritura 1.25x base) o 1 hora (costos de escritura 2x base).$0.30/M on Claude 3.5 Sonnet vs $3,00/M fresco  10 veces más barato (docs.anthropic.com, a partir de 2026-04). Las tarifas difieren por modelo (Opus/Haiku publicado por separado); siempre compruebe la página de precios en vivo.

**OpenAI**El sistema de almacenamiento en caché automático para las instrucciones ≥1024 tokens (platform.openai.com, 2026-04). No hay bandera explícita. La entrada en caché es aproximadamente 10 veces más barata que la nueva en las tarjetas de tasa gpt-4o/gpt-5. Ni los documentos ni las notas de liberación publican una línea de base oficial de la tasa de éxito; los informes comunitarios se agrupan alrededor de 3060% con un diseño de solicitud cuidadoso.`usage.cached_tokens`para medir la suya.

**Google (Gemini)**: caching de contexto a través de API explícita; 1M-token context significa que el caching paga aún más.

**Self-hosted (vLLM, SGLang)**: Fase 17 · 06 cubre el mismo patrón en su propio cálculo.

### L1  Caching semántico a nivel de aplicación

Antes de llamar al LLM, hash el prompt, embebegue y busque una solicitud similar almacenada en caché (similaridad de cosinos por encima del umbral, típicamente 0.95+).

Fuente abierta: Redis Vector Similarity, GPTCache, Qdrant. Comercial: Portkey Cache, Helicone Cache.

Las afirmaciones de precisión del proveedor se refieren a la frecuencia con la que la respuesta devuelta en caché fue semánticamente apropiada  no a la frecuencia con la que se golpeó.

- Convite de tiempo libre: 10-15%.
- Preguntas frecuentes / apoyo estructurados: 40-70%.
- Las preguntas de código: 20-30% (variantes pequeñas matan los hits).
- Agentes de voz que repiten las instrucciones: 50-80% (conjunto fijo de normalización de voz).

### El patrón antiparallelización

Su agente hace 10 llamadas de herramientas en paralelo. Todas las 10 tienen el mismo mensaje de sistema de 4K-token. Las escrituras de caché antropópica son por solicitud; la primera escritura de caché se completa alrededor de 300 ms después de que el proveedor vea el mensaje. Las solicitudes de 2-10 llegan en la misma ventana de milisegundos y cada uno ve caché perdido. Pagas 10 primas de escritura, 0 descuentos de lectura.

Corrección: lote con secuencia-primero  hacer la solicitud 1 solo, luego disparar 2-10 una vez que la caché de 1 se ha llenado. Agrega 300 ms a la primera llamada de herramienta; ahorra 5-10 veces la factura.

### El antipatrón de contenido dinámico

Su mensaje del sistema se parece a:

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

Cada solicitud es única, cada solicitud escribe, cero hits.

Corrección: mover todo lo realmente estático al prefijo cachéable; agregar contenido dinámico después del límite de caché:

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery pasó de 7% a 74% de la tasa de caché de esta manera y publicó la anatomía.

### Batch de pila + caché para cargas de trabajo nocturnas

Las API de lote (fase 17 · 15) ofrecen un descuento del 50% a las 24 horas de cambio. La entrada almacenada en caché en la parte superior te da ~ 10 veces más. Las cargas de trabajo de clasificación, etiquetado y generación de informes de la noche a la mañana pueden caer a ~ 10% del costo sincrónico sin caje mediante la pila.

### Números que debes recordar

Los puntos de precios se capturan 2026-04 de los documentos de los vendedores vinculados y se desvían cada pocos meses  volver a comprobar antes de depender de ellos.

- Anthropic se lee en caché: $0.30/M en Claude 3.5 Sonnet, aproximadamente 10 veces más barato que la entrada reciente (docs.anthropic.com).
- Premia de escritura de caché antropico: 1.25x (5 min TTL) o 2x (1 hora TTL).
- OpenAI auto-cache: se aplica a las instrucciones ≥1024 tokens; entradas almacenadas en caché a un precio de aproximadamente el 10% de las entradas nuevas en las tarjetas de tasa corriente (platform.openai.com).
- Taxa de hits de caché semántico (reportado por la comunidad): ~10% de chat abierto; hasta ~70% de preguntas frecuentes estructuradas. No es una línea de base documentada por el proveedor.
- ProjectDiscovery: 7% → 74% de tasa de éxito al mover dinámica fuera del prefijo (blog de proyecto, 2025-11).
- Anti-patrón de paralelalización: informes típicos de inflación de facturas 510x cuando N solicitudes paralelas omiten la primera caché de escritura.

```figure
semantic-cache-hit
```

## Usalo

`code/main.py`Simula la caché L1 + L2 en cargas de trabajo mixtas.

## Envío

Esta lección produce`outputs/skill-cache-auditor.md`. Dado el modelo y el tráfico inmediato, las auditorías de caché y recomienda la reestructuración.

## Los ejercicios

1. - ¿ Qué ?`code/main.py`- ¿Cuánto cambia la cuenta?
2. Su solicitud del sistema tiene una fecha. Mueva. Muéstren antes / después de golpear la tasa matemática.
3. Calcule el equilibrio de 1 hora TTL (2x escribir) vs 5 minutos TTL (1.25x escribir) dada la tasa de llegada de su solicitud.
4. El caché semántico en el umbral de 0.95 alcanza el 20%. en 0.85 alcanza el 50% pero ve respuestas almacenadas en caché incorrectas.
5. Se hacen 10 subcuestiones paralelas por pregunta de usuario.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| L2 prompt cache | "prefix cache" | Provider stores KV for repeated prefix |
| `cache_control` | "Anthropic cache marker" | Explicit attribute marking cacheable blocks |
| Cache write premium | "write tax" | Extra cost for first miss-to-cache (1.25x or 2x) |
| L1 semantic cache | "embedding cache" | App-level hash-and-embed before calling LLM |
| GPTCache | "LLM caching lib" | Popular OSS L1 cache library |
| Cache hit rate | "hits / total" | Fraction of requests served from cache |
| Parallelization anti-pattern | "the N-write trap" | N parallel requests miss cache N times |
| Dynamic content trap | "the time-in-prompt trap" | Dynamic bytes in prefix kill hit rate |
| RadixAttention | "intra-replica cache" | SGLang's prefix-cache implementation |

## Leer más

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) oficial `cache_control`la semántica y los TTL.
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) comportamiento de almacenamiento automático en caché y elegibilidad.
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
