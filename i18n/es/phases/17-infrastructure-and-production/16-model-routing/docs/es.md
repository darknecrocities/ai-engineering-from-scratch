# El modelo de enrutamiento como una primitiva de reducción de costos

> Un corredor dinámico evalúa cada solicitud (tipo de tarea, longitud de token, similitud de incorporación, confianza) y envía consultas simples a un modelo barato, aumentando las complejas a un modelo fronterizo. También se llama cascada de modelos. Los estudios de caso de producción muestran una reducción de costes de 20-60% en la calidad de iso en las implementaciones de EE.UU./Reino Unido/UE; una mejora de la eficiencia de enrutamiento del 30% en el SaaS de gran volumen se convierte en un ahorro anual de seis cifras. El contexto de 2026 es que los precios de la inferencia LLM cayeron ~ 10 veces por año  un token de clase GPT-4 se fue de $20/M to ~$0,40/M desde finales de 2022 hasta 2026. La mayor parte de la caída es mejor para las pilas (fase 17 · 04-09), no para el hardware. El enrutamiento es cómo se convierte esa caída de precios en margen sin regresión del producto. El modo de falla es la deriva del modelo barato: la ruta empuja el 40% a un modelo más débil, la calidad cae del 3-5% en las tareas de razonamiento, nadie se da cuenta por un cuarto. Rutas de puertas por métricas de calidad en línea, no sólo conjuntos de evaluaciones fuera de línea.

**Type:** Learn
**Languages:** Python (stdlib, toy cascading router simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 19 (AI Gateways)
**Time:** ~60 minutes

## Objetivos de aprendizaje

- Explica el modelo en cascada: barato primero con control de confianza, escalada en baja confianza.
- Enumera las cuatro señales de enrutamiento (clasificación de tareas, longitud de la tarea, incorporación de similitud con el conjunto duro conocido, confianza en sí mismo desde el primer paso).
- Calcule el coste combinado esperado en la división de rotación de destino y la tolerancia a pérdidas de calidad.
- Nombre de la métrica de monitoreo de deriva (puerta de calidad en línea) que atrapa el modelo barato.

## El problema

Su servicio cuesta $80k/mes en GPT-5. sus análisis muestran que el 70% de las consultas son simples: "¿qué hora es en París?" "refrasear esta frase". Un modelo de clase Haiku maneja perfectamente a 3% del costo. 30% necesita el razonamiento de GPT-5  codificación, matemáticas, planificación en múltiples pasos.

Si se envía el 70% a barato y el 30% a caro, su factura cae alrededor del 65% en la misma calidad del producto. Esto es enrutamiento. El truco es construir el corredor sin regredir la calidad.

## El concepto

### Cuatro señales de enrutamiento

1. **Task classification**Se puede ser un clasificador basado en reglas, un LLM pequeño (Haiku-clase a $0.25/M), o incorporar similitud con baldes etiquetados.

2. **Prompt length**Las señales de +4K a menudo necesitan fronteras para la coherencia.

3. **Embedding similarity to known-hard set**Si la consulta está cerca (cosin > 0,88) de un cubo conocido de durabilidad, escala directamente a la frontera.

4. **Self-confidence from first-pass**Si las pruebas de registro del modelo muestran baja confianza O se niegan O se expone el lenguaje de cobertura, vuelva a intentarlo en frontera.

### Tres patrones

**Pre-route**(clasificador por delante): ~ 5-10 ms de latencia añadida; más rápido en general.

**Cascade**(Primero barato, escala en baja confianza): ~1.2x latencia media (corrida barata más verificación), ~2x en escalada.

**Ensemble route**(se ejecuta a bajo costo y fronterizo en paralelo para una muestra, seleccionar un modelo de recompensa): la más alta calidad, el mayor coste; utilizar sólo para A/B crítico.

### Aplicación

Las pasarelas de IA (fase 17 · 19) exponen el enrutamiento.`router`Por ejemplo, el sistema de conexión de acceso a Internet (POS) es un sistema de conexión de acceso a Internet (POS) que permite a los usuarios acceder a Internet en forma automática.

Fuente abierta: RouteLLM (LMSYS), No Diamond (comercial), Prompt Mule.

### La curva de precios de 2026

| Model class | Late 2022 | 2026 | Change |
|-------------|-----------|------|--------|
| GPT-4-level quality | ~$20/M | ~$0.40/M | 50x cheaper |
| Frontier (GPT-5, Claude 4) | — | ~$3-10/M | new tier |

La mayor parte de la mejora es el servicio de eficiencia  las lecciones básicas en la Fase 17 · 04-09 se convirtieron en caídas de costos del lado del proveedor.

### La deriva es el verdadero riesgo

Su ruta envía el 40% al modelo barato. Durante seis meses, la distribución de tareas cambia (los usuarios se vuelven más sofisticados, hacen preguntas más largas). El router no se da cuenta porque su clasificador fue entrenado en datos de Q1. La calidad cae silenciosamente. Nadie se queja lo suficientemente fuerte. En un benchmark de competidores descubres que perdiste.

Rutas de puertas por métricas de calidad en línea:

- El usuario sube/baja por ruta.
- Juez de LLM automático en una muestra retenida (5%) por ruta.
- Taxa de escalación: si la cascada está aumentando en la ruta superior a > 30%, el modelo barato está siendo sobre-enrutado.
- Taxa de rechazo por ruta.

### Números que debes recordar

- 2026 ahorros de enrutamiento en iso-calidad: estudios de caso del 20-60%.
- Descenso de los precios de los LLM 2022-2026: ~ 10 veces por año agregado.
- GPT-4 nivel 2022 vs 2026: ~$20/M → ~$0,40/M.
- Impacto de latencia en cascada: ~ 1,2x mediana, ~ 2x escalada (~ 10% del tráfico).

```figure
model-cascade-router
```

## Usalo

`code/main.py`La información de la empresa se centra en la información de los usuarios y en la información de los usuarios.

## Envío

Esta lección produce`outputs/skill-router-plan.md`- Dado el volumen de trabajo y el presupuesto de calidad, elige un patrón de enrutamiento y señales.

## Los ejercicios

1. - ¿ Qué ?`code/main.py`¿En qué piso de precisión la cascada supera la ruta previa?
2. Su base de usuarios es de 30% empresarial (cuestiones complejas), 70% de nivel gratuito (simple). Diseñar la división de enrutamiento. ¿Qué métricas en línea se abren?
3. Una ruta reduce la calidad en un 2% pero ahorra un 40%. ¿Es un barco?
4. Implementar una verificación de confianza utilizando logprobs de OpenAI / APIs antropológicas. ¿Cuál es el umbral con el que comienza?
5. En seis meses, la tasa de escalada sube del 8% al 22%. Diagnóstico de tres causas y la solución para cada uno.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Model routing | "cost broker" | Dynamic choice of model per request |
| Model cascade | "cheap-first escalate" | Run cheap, fall through to frontier on low confidence |
| Pre-route | "classify first" | Classifier up front; no re-run |
| Ensemble route | "parallel pick" | Run multiple, reward-model picks best |
| Escalation rate | "uprouted %" | Fraction of cascade requests that escalated |
| RouteLLM | "LMSYS router" | OSS router library |
| Not Diamond | "commercial router" | SaaS model-routing product |
| Drift | "cheap creep" | Distribution shift without router noticing |
| Online quality gate | "live check" | Automated LLM-judge sampling live traffic |

## Leer más

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) Puerta de entrada multimodelo con primitivas de enrutamiento.
