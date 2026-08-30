# Las APIs de lote  el 50% de descuento como estándar de la industria

> Cada proveedor principal envía una API de lote sincronizada con un descuento del 50% y una respuesta de 24 horas. OpenAI, Anthropic, Google y la mayoría de las plataformas de inferencia (Tier de lote de Fireworks, lote de Together) implementan el mismo patrón. El lote de pila con almacenamiento en caché rápido y las tuberías de la noche a la mañana caen a ~10% del coste sincrónico-desarrollado. La regla es brutalmente simple: si no es interactiva, pertenece al lote. Las líneas de generación de contenido, la clasificación de documentos, la extracción de datos, la generación de informes, el etiquetado en masa, el etiquetado de catálogos  cualquier cosa que tolere la latencia de 24 horas es dinero que queda en la mesa hasta que se mueva al lote. El patrón de producción de 2026 es la triaje de cada nueva carga de trabajo de LLM en tres carriles: interactiva (sincrónica con el almacenamiento en caché), seminteractiva (cuota sincrónica con fallback), lote (por la noche, entradas almacenadas en caché apiladas). Las cargas de trabajo que pretenden ser interactivas pero toleran minutos de latencia pierden más.

**Type:** Learn
**Languages:** Python (stdlib, toy batch-vs-sync cost simulator)
**Prerequisites:** Phase 17 · 14 (Prompt & Semantic Caching)
**Time:** ~45 minutes

## Objetivos de aprendizaje

- Nombre de las tres API de lote de proveedores (OpenAI, Anthropic, Google) y las garantías comunes de descuento del 50% + 24 horas de respuesta.
- Calcule el coste de la empilación de lote + entrada almacenada en caché en una carga de trabajo de clasificación durante la noche y compare con la línea de base sincrónica sin almacenamiento.
- Tria una carga de trabajo en lote interactivo / semi-interactivo / y justifique el carril.
- Nombrar las dos trampas: interactividad parcial (el usuario espera más rápido que 24h) y derivación del esquema de salida (el formato de archivo de lote difiere por proveedor).

## El problema

Su equipo envía una línea de generación de informes nocturna. 50.000 documentos, resumen cada uno, agrupar los resúmenes, redactar un informe ejecutivo.

El lote te ofrece un 50% de descuento. También puedes habilitar el caché de la solicitud del sistema (compartido en todas las llamadas 50k).

El lote es la palanca más barata en el conjunto de herramientas de costos de LLM que nadie tira. La razón es principalmente organizativa: los equipos piensan "en tiempo real" cuando el SLA en realidad es "por la mañana". Esta lección es sobre no dejar el 90% de la factura en la mesa.

## El concepto

### Las tres partidas de API

**OpenAI Batch API**: Cargar el archivo JSONL con una lista de solicitudes. Promesa respuesta de 24 horas (generalmente ~ 2-8 horas en la práctica). Descuento del 50% en tokens de entrada y salida. `/v1/batches`Las entradas elegibles para almacenamiento en caché también obtienen precios de entrada almacenados en caché en la parte superior.

**Anthropic Message Batches**JSONL carga, 24 horas de vuelta, descuento del 50%.`cache_control` las escrituras en caché son explícitas, las lecturas ocurren automáticamente dentro del lote.

**Google Vertex AI Batch Prediction**BigQuery o GCS entrada. Descuento similar del 50% para Gemini.

### Semántica: asíncrona, no lenta

El lote es "prometo que regresaré dentro de las 24 horas"  no "esto tomará 24 horas".

### Se apila con caché

Una resumen de 50k de documentos con el mismo sistema de 4K-token de la solicitud:

- Sincronización sin caché: 50000 × ($input × 4000 + $de salida × 200) a las velocidades completas.
- Caché sincrónico: el pedido del sistema se cacha después de la primera escritura; los 49999 restantes obtienen 10 veces más barato entrada.
- Batch caché: todo lo anterior más un 50% de descuento tanto en lectura como en escritura.

La pila: lote + caché = ~10% de la factura sincronizada sin caché. Cualquier carga de trabajo que se ejecuta durante la noche y tiene un mensaje compartido del sistema debe usar esto.

### Clasificación de la carga de trabajo

**Interactive** el usuario espera la respuesta. TTFT importa. llamada sincrónica con caché rápido. No puede batch.

**Semi-interactive** el usuario envía una tarea, verifica en minutos. colas de sincronización con fallback para sincronizar si el lote no está disponible.

**Batch** el usuario espera resultados "a la mañana" o "a la próxima hora".

Error común: clasificar todo como interactivo porque la tubería es producción.

### La trampa de interactividad parcial

Algunas características parecen interactivas pero toleran 5-10 minutos. Ejemplo: un informe de salud de los clientes nocturno con botón "refrenchar". El usuario hace clic en refrescar; esperar 10 minutos está bien. El equipo lo envía como sincrono. 50 actualizaciones simultáneas cuestan 10 veces lo que costaría el envío en lote y entregado por correo electrónico.

La pregunta que se debe hacer: "¿Qué significa 24 horas para este usuario?" Si la respuesta es "no se darían cuenta", en lote.

### La trampa de esquema de salida

Los formatos de archivos de lote difieren por proveedor:

- OpenAI: JSONL, una solicitud por línea.
- Antropic: JSONL, un mensaje por línea; formato de respuesta incorporado.
- Vertex: tabla BigQuery o prefijo GCS con TFRecord.

Escribir "un cliente de lote" entre los proveedores significa código de adaptador por proveedor. Gateways que anuncian lote multi-provedor (Portkey, LiteLLM algunos niveles) todavía envuelven el formato crudo.

### Números que debes recordar

- Descuento por lotes entre los proveedores: 50% fijo en entradas + salidas.
- SLA de giro: 24 horas garantizadas, 2-6 horas típicas P50.
- Partido apilado + entrada almacenada en caché: ~10% del coste sincronizado sin caché.
- Regla de clasificación de carga de trabajo: si la latencia de 24 horas es aceptable, siempre se realice el lote.

```figure
batch-lane-triage
```

## Usalo

`code/main.py`Computa costos en sincronización, sincronización + caché, lote y lote + caché para una carga de trabajo de 50k documentos.

## Envío

Esta lección produce`outputs/skill-batch-triager.md`- Dadas las características de la carga de trabajo, se clasifican en lotes interactivos/semi/partidos y se estiman los ahorros.

## Los ejercicios

1. - ¿ Qué ?`code/main.py`. Para un pipeline de 100k-doc con un sistema de 3K-token y 500-token de salida, calcular el ahorro de la pila completa (batch + caché) frente a la línea de base de sincronización.
2. Seleccione tres características en un producto real que conozca. Triega cada una en interactivo/semi/parcela.
3. Un usuario se queja de que su informe tomó 3 horas. ¿Fue un error de selección de lote o un interactivo legítimo?
4. Su SLA de retorno de API de lote es de 24 horas pero P99 es de 20 horas. ¿Cómo se comunica esto al usuario  ¿cuál es el comportamiento del sistema en aguas abajo en el caso de borde?
5. Computación de equilibrio: ¿a qué longitud de prefijo compartido se hace batch + cache más barato que correr de la noche a la mañana en su propia GPU reservada?

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Batch API | "async discount" | 50% off with 24h turnaround |
| JSONL | "batch format" | One JSON request per line; OpenAI/Anthropic standard |
| Message Batches | "Anthropic batch" | Anthropic's batch API product name |
| Batch prediction | "Vertex batch" | Vertex AI's batch API product |
| Turnaround SLA | "24h promise" | Guarantee, not typical; typical is 2-6h |
| Workload triage | "interactivity decision" | Interactive / semi / batch routing decision |
| Output schema | "response format" | Per-provider JSONL layout; not portable |
| Stacked discount | "batch + cache" | ~10% of uncached sync bill when both apply |

## Leer más

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) formato JSONL y `/v1/batches`¿Qué es lo que se dice?
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) formato de lote y `cache_control`interacción.
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini) Semántica de lotes de Géminis.
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
