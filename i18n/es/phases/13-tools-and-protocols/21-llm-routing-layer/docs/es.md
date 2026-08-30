# LLM Routing Layer  LiteLLM, OpenRouter, Portkey

> El bloqueo del proveedor es caro. Las diferentes cargas de trabajo para llamar a herramientas se adaptan a diferentes modelos. Las pasarelas de enrutamiento dan una superficie de API, retas, fallas, seguimiento de costos y barandillas. Tres arquetipos dominan 2026: LiteLLM (auto-hostado de código abierto), OpenRouter (SaaS administrado), Portkey (producción de grado, de código abierto en marzo de 2026). Esta lección nombra los criterios de decisión y se encuentra en una puerta de enrutamiento de STDlib.

**Type:** Learn
**Languages:** Python (stdlib, routing + failover + cost tracker)
**Prerequisites:** Phase 13 · 02 (function calling), Phase 13 · 17 (gateways)
**Time:** ~45 minutes

## Objetivos de aprendizaje

- Distinguir entre las opciones de enrutamiento de auto-hostaje, administrado y de producción.
- Implementar una cadena de retroceso que reteste las fallas de los proveedores en un orden de prioridad definido.
- Seguir el costo por solicitud y el uso de tokens entre los proveedores.
- Decidir entre LiteLLM, OpenRouter y Portkey para una determinada restricción de producción.

## El problema

En situaciones en las que el enrutamiento del proveedor sea importante:

1. **Cost.**Claude Sonnet cuesta 3 veces lo que Haiku, para una tarea de triaje, Haiku es suficiente, para una tarea de síntesis, Sonnet vale la pena.

2. **Failover.**OpenAI tiene una mala hora, todas las solicitudes fallan, quieres regresar automáticamente a Anthropic sin volver a desplegar.

3. **Latency.**Una interfaz de chat en vivo necesita un tiempo rápido para el primer token.

4. **Compliance.**Los usuarios de la UE deben permanecer en las regiones de la UE.

5. **Experimentation.**A/B dos modelos en la misma carga de trabajo.

La codificación manual de todo esto por integración es repetitiva. Una puerta de enrutamiento da una API compatible con OpenAI y maneja el resto.

## El concepto

### Forma de proxy compatible con OpenAI

Todos hablan OpenAI, la puerta de enrutamiento expone.`/v1/chat/completions`, acepta el esquema OpenAI, y internos proxies a Antropic / Gemini / Cohere / Ollama / cualquier cosa.

### Los alias de modelo

En lugar de un ID de instantáneo fijado, tu código dice `our_smart_model`Cuando un proveedor envía una nueva generación, cambia el alias de lado del servidor; su código no toca nada.

### Las cadenas de retroceso

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

Los gateways definen esto en una configuración. Los retrospectivos cuentan contra un presupuesto para que las cascadas de retroceso no exploten el costo.

### Caching semántico

Las instrucciones idénticas o casi idénticas se encuentran en un caché en lugar del proveedor. Los ahorros en los bucles de agentes repetidos pueden ser del 30 al 60 por ciento. Las claves se basan en la incorporación; las instrucciones casi idénticas comparten un espacio de caché.

### Barras de seguridad

Nivel de entrada:

- **PII redaction.**Regex o ML antes de enviar las instrucciones.
- **Policy violations.**Rechazar las instrucciones con contenido prohibido.
- **Output filters.**Escarba los acabados para las fugas.

Portkey y Kong tienen barandillas de seguridad, y LiteLLM las deja opcionales.

### Límites de tasas por clave

Una clave de API = un equipo. Los presupuestos por clave impiden que un equipo consuma la cuota compartida. La mayoría de las puertas de acceso soportan esto.

### Compromiso entre empresas auto-hospedadas y empresas administradas

| Factor | LiteLLM (self-hosted) | OpenRouter (managed) | Portkey (production) |
|--------|----------------------|----------------------|----------------------|
| Code | Open source, Python | Managed SaaS | Open source (Mar 2026) + managed |
| Setup | Deploy a proxy | Sign up | Either |
| Providers | 100+ | 300+ | 100+ |
| Billing | Your own keys | OpenRouter credits | Your own keys |
| Observability | OpenTelemetry | Dashboard | Full OTel + PII redaction |
| Best for | Teams that want full control | Rapid prototyping | Production with compliance |

LiteLLM gana cuando tienes un equipo SRE y quieres soberanía de datos. OpenRouter gana cuando quieres una sola suscripción y no hay infra. Portkey gana cuando necesitas barandillas y cumplimiento fuera de la caja.

### Seguimiento de costes

Cada solicitud lleva`provider`¿ Qué ?`model`¿ Qué ?`input_tokens`¿ Qué ?`output_tokens`. Multiplicar por precios por modelo por token (extraído de una hoja de precios que mantiene la puerta de entrada).

### MCP más enrutamiento

Una puerta de enlace puede enrutar tanto las llamadas de LLM como las solicitudes de muestreo de MCP. Cuando el modelo de una solicitud de muestreoPreferencias prefiere un modelo específico, la puerta de enlace se traduce a la parte posterior derecha. Aquí es donde la fase 13 · 17 (puerta de enlace de MCP) y la puerta de enlace de esta lección a veces se fusionan en un solo servicio.

### Estrategias de enrutamiento

- **Static priority.**Primero en la lista, retrocede en el error.
- **Load balancing.**En forma de round-robin o ponderada.
- **Cost-aware.**Elija el modelo más barato que reúna latencia / calidad.
- **Latency-aware.**Elige el modelo más rápido en los últimos N minutos.
- **Task-aware.**Las rutas de clasificación rápida codifican a un modelo, resumen a otro.

```figure
tp-router-failover
```

## Usalo

`code/main.py`Implementa una puerta de enrutamiento en ~ 150 líneas: acepta solicitudes en forma de OpenAI, se traduce a estatuas por proveedor, ejecuta una cadena de fallback de prioridad, rastrea el costo por solicitud y aplica un pase de redacción de PII en las entradas.

Qué ver:

- `ROUTES`dict: alias -> lista ordenada por prioridad de proveedores concretos.
- Recupera el ciclo de retroceso en 5xx.
- El rastreador de costos multiplica el uso de tokens por tasas por modelo.
- El redactor de PII examina los patrones en forma de SSN antes de reenviarlos.

## Envío

Esta lección produce`outputs/skill-routing-config-designer.md`. Dado un perfil de carga de trabajo (latencia, coste, cumplimiento), la habilidad selecciona LiteLLM / OpenRouter / Portkey y produce una configuración de enrutamiento.

## Los ejercicios

1. - ¿ Qué ?`code/main.py`- Activa el escenario de interrupción; confirma que el segundo proveedor se retrocede y que el coste se atribuye correctamente.

2. Agregue caché semántico: SHA256 del prompt es una clave de búsqueda; los toques de caché regresan instantáneamente.

3. Añadir un clasificador de orden que envía "código ..." a un alias que favorece la inteligencia y "resumir ..." a un alias que favorece la velocidad.

4. Diseño de presupuestos por equipo: cada equipo tiene un límite de gasto mensual; Gateway rechaza las solicitudes una vez que se alcanza el límite.

5. Lea los documentos LiteLLM, OpenRouter y Portkey uno al lado del otro.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Routing gateway | "LLM proxy" | One-API-surface layer in front of many providers |
| OpenAI-compatible | "Speaks the OpenAI schema" | Accepts `/v1/chat/completions` shape, translates to any backend |
| Model alias | "our_smart_model" | Name in your code that the gateway maps to a concrete model |
| Fallback chain | "Retry list" | Ordered list of providers attempted on failure |
| Semantic caching | "Prompt-embedding cache" | Key is embedding of the prompt; near-duplicates share a cache hit |
| Guardrails | "Input/output filters" | Redact PII, reject policy violations |
| Per-key rate limit | "Team budget" | Quota scoped to an API key |
| Cost tracking | "Per-request spend" | Aggregate token usage x price per model |
| LiteLLM | "The open proxy" | Self-hostable OSS routing gateway |
| OpenRouter | "The managed SaaS" | Hosted gateway with credit-based billing |
| Portkey | "The production option" | Open-source + managed with guardrails built in |

## Leer más

- [LiteLLM — docs](https://docs.litellm.ai/) Puerta de enrutamiento auto-alojada
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) Routing administrado SaaS
- [Portkey — docs](https://portkey.ai/docs) Enrutamiento de producción con barandillas
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) Guía de decisión
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) Encuesta de proveedores
