# Puertas de acceso de IA  LiteLLM, Portkey, Kong AI Gateway, Bifrost

> Una puerta de entrada se encuentra entre sus aplicaciones y los proveedores de modelos. Las características principales son el enrutamiento del proveedor, retroceso, retemplajes, limitación de tarifas, referencias secretas, observabilidad, barandillas.**LiteLLM**es MIT OSS con más de 100 proveedores, compatible con OpenAI, pero se descompone alrededor de ~ 2000 RPS (8 GB de memoria, fallas en cascada en benchmarks publicados); mejor para Python, <500 RPS, desarrollo / prototipo. **Portkey**está en el plano de control (garderas, redacción de PII, detección de jailbreak, pistas de auditoría), fue Apache 2.0 de código abierto marzo de 2026, 20-40 ms de latencia por encima, $49/mo production tier. **Kong AI Gateway** built on Kong Gateway — Kong's own benchmark on same 12 CPUs: 228% faster than Portkey, 859% faster than LiteLLM; $Precio de 100/modelo/mes (máximo 5 en nivel Plus); adecuado para empresas si ya está en Kong. **Bifrost**(Maxim AI)  retemplajes automáticos con configurable backkoff, regreso a Anthropic en OpenAI 429. **Cloudflare / Vercel AI Gateways** administrado, operaciones cero, retraso básico. Residencia de datos impulsa la decisión de auto-host; Portkey y Kong se sientan en el medio con OSS + administrado opcional.

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 minutes

## Objetivos de aprendizaje

- Enumera las seis características principales de la puerta de enlace (routing, fallback, retemptadas, límites de velocidad, secretos, observabilidad, barandillas).
- Mapa de cuatro puertas de acceso 2026 (LiteLLM, Portkey, Kong AI, Bifrost) para escalar los techos y casos de uso.
- Cita el índice de referencia Kong (228% vs Portkey, 859% vs LiteLLM) y explica por qué importa para > 500 RPS.
- Elija auto-hosted vs administrado dado el presupuesto de residencia de datos y operaciones.

## El problema

Su producto llama OpenAI, Anthropic y un Llama auto-hosted. Cada proveedor tiene un SDK diferente, modelo de error, límite de tarifa y esquema de auth.

Reinventando esto en la capa de aplicación, se combina cada servicio con cada proveedor. Una capa de puerta de enlace lo consolida en un proceso con una API (generalmente compatible con OpenAI) que se distribuye a los proveedores.

## El concepto

### Seis características centrales

1. **Provider routing** OpenAI, Anthropic, Gemini, auto-hosted, etc. detrás de una API.
2. **Fallback** en 429, 5xx, o fallas de calidad, vuelva a intentarlo en otro lugar.
3. **Retries** retroceso exponencial, intentos limitados.
4. **Rate limits** por inquilino, por llave, por modelo.
5. **Secret references** sacar las credenciales de la bóveda en el tiempo de ejecución (nunca en la aplicación).
6. **Observability** ATRITUDOS OTEL + GenAI (fase 17 · 13) + atribución de costes.
7. **Guardrails** Reducción de PII, detección de jailbreak, filtros de temas permitidos.

### LiteLLM  MIT OSS, Python

- 100+ proveedores, compatibles con OpenAI, configuración del router, retroceso, observabilidad básica.
- Se rompe alrededor de 2000 RPS en el punto de referencia de Kong; 8 GB de memoria, fallas en cascada bajo carga sostenida.
- Mejor ajuste: aplicación Python, < 500 RPS, puertas de acceso de desarrollo/estagiado, enrutamiento experimental.
- Costo: $0 para OSS; existe un nivel libre de nube.

### Portkey  posicionamiento del plano de control

- Apache 2.0 OSS a partir de marzo de 2026. Rastreos de seguridad, redacción de PII, detección de jailbreak, rastro de auditoría.
- 20-40 ms por solicitud de latencia general.
- $49 / mes para el nivel de producción con retención + SLA.
- Mejor adaptación: industrias reguladas que necesitan barandillas + observabilidad en conjunto.

### Kong AI Gateway  el juego de la escala

- Construido en Kong Gateway (producto de API maduro, lua+OpenResty).
- El propio índice de referencia de Kong en el equivalente de 12 CPU: 228% más rápido que Portkey, 859% más rápido que LiteLLM.
- Precio: $ 100 / modelo / mes, máximo 5 en el nivel Plus.
- Mejor ajuste: ya en Kong; > 1000 RPS; dispuesto a licenciar.

### Bifrost (Maxim AI)

- Pruebas automáticas con retroceso configurable.
- Fallback a Anthropic en OpenAI 429 es una receta canónica.
- Nuevo participante, comercial.

### Puerta de entrada de IA de Cloudflare / Puerta de entrada de IA de Vercel

- Gestionado, operaciones cero, retoma y observabilidad básica.
- Mejor ajuste: aplicaciones JavaScript de Edge en Cloudflare/Vercel.
- Limitado en comparación con Kong/Portkey en barandillas y límites de velocidad.

### Auto-hosted vs administrado

La residencia de datos es la función de forzamiento. Cuidado de salud y finanzas auto-hosting por defecto (LiteLLM o Portkey OSS o Kong). Productos de consumo administrados por defecto (Cloudflare AI Gateway) o de nivel medio (Portkey administrado).

### Presupuesto de la latencia

- LiteLLM: 5-15 ms de carga aérea típico.
- Portero: 20-40 ms por encima.
- 3 a 8 ms por encima.
- Cloudflare/Vercel: 1-3 ms de gastos generales (avantage de borde).

La latencia de la puerta de enlace se agrega directamente a TTFT. Para TTFT P99 < 100 ms SLA, Kong o Cloudflare. Para P99 < 500 ms, cualquier.

### Materia de semántica de límite de tasas

El token-bucket simple funciona hasta una escala moderada. Multi-tenant requiere una ventana deslizante + franquicia de explosión + tiering por tenente. LiteLLM navega token-bucket; Kong navega ventana deslizante; Portkey navega en niveles.

### Puerta de entrada + observabilidad + enrutamiento componer

La fase 17 · 13 (observabilidad) + 16 (routing de modelo) + 19 (gateways) son la misma capa en producción. Elija una herramienta que cubra las tres o cableálas cuidadosamente: la mayoría de las implementaciones de 2026 combinan Helicone (observabilidad) o Portkey (garderrails) con Kong (escala) para funciones divididas.

### Números que debes recordar

- LiteLLM: rompe a ~ 2000 RPS, memoria de 8 GB.
- Portkey: 20-40 ms por encima; Apache 2.0 desde marzo de 2026.
- Kong: 228% más rápido que Portkey, 859% más rápido que LiteLLM.
- Precio de Kong: $ 100 / modelo / mes, 5 max en el nivel Plus.
- Cloudflare/Vercel: 1-3 ms de carga en el borde.

```figure
mx-gateway-fallback
```

## Usalo

`code/main.py`Simula el enrutamiento de la puerta de enlace con fallback en 3 proveedores bajo la inyección 429/5xx. Informes latencia, tasa de retiro y tasa de impacto de fallback.

## Envío

Esta lección produce`outputs/skill-gateway-picker.md`Dada la escala, la postura de las operaciones, el cumplimiento, el presupuesto de latencia, elige una puerta de entrada.

## Los ejercicios

1. - ¿ Qué ?`code/main.py`. Configurar fallback desde OpenAI→Anthropic→auto-hosted. ¿Cuál es la tasa de impacto esperada con una tasa de error del proveedor del 5%?
2. Su SLA es TTFT P99 < 200 ms en una línea de base de 300 ms. ¿Qué pasarelas se mantienen dentro del presupuesto?
3. Un cliente de atención médica requiere auto-hosting + redacción de PII + auditoría.
4. Comparar LiteLLM vs Kong: ¿a qué límite RPS debe migrar un equipo?
5. Diseñar una política de límite de tarifas para un SaaS multi-arrendatario: nivel gratuito, nivel de prueba, nivel pagado.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Gateway | "API broker" | Process sitting between apps and providers |
| LiteLLM | "the MIT one" | Python OSS, 100+ providers, breaks at 2K RPS |
| Portkey | "guardrails gateway" | Control plane + observability, Apache 2.0 |
| Kong AI Gateway | "the scale one" | Built on Kong Gateway, benchmark leader |
| Bifrost | "Maxim's gateway" | Retries + Anthropic fallback recipe |
| Cloudflare AI Gateway | "edge managed" | Edge-deployed managed gateway, zero-ops |
| PII redaction | "data scrub" | Regex + NER mask before sending to model |
| Jailbreak detection | "prompt injection guard" | Classifier on user input |
| Audit trail | "regulated log" | Immutable record of every LLM call |
| Token-bucket | "simple rate limit" | Refill-based rate limiter |
| Sliding-window | "precise rate limit" | Time-windowed rate limiter; better fairness |

## Leer más

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
