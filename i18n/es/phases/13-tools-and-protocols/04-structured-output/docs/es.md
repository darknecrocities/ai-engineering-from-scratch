# Producción estructurada  JSON Schema, Pydantic, Zod, Descodage restringido

> "Pregúntale bien al modelo para devolver JSON" falla del 5 al 15 por ciento de las veces, incluso en los modelos fronterizos. Las salidas estructuradas cierran esa brecha con decodificación restringida: el modelo se impide literalmente emitir un token que violaría el esquema.`responseSchema`, Pydantic AI `output_type`, y Zod's `.parse`Esta lección construye el validador de esquemas y los estudiantes de contrato de modo estricto utilizarán para cada tubería de extracción de producción.

**Type:** Build
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## Objetivos de aprendizaje

- Escriba un esquema JSON 2020-12 para un objetivo de extracción utilizando las restricciones correctas (enum, min/max, requerido, patrón).
- Explica por qué el modo estricto y la decodificación limitada dan garantías diferentes de la "validez después de la generación".
- Distingue los tres modos de falla: error de análisis, violación de esquema, rechazo de modelo.
- Enviar una tubería de extracción con reparación tipográfica y manipulación tipográfica de rechazo.

## El problema

Un agente que lee un correo electrónico de pedido de compra necesita convertir el texto libre en `{customer, line_items, total_usd}`- Tres enfoques.

**Approach one: prompt for JSON.**"Responde en JSON con campos cliente, line_items, total_usd. " Funciona del 85 al 95 por ciento del tiempo en los modelos fronterizos. Fallece en seis formas: brace faltante, coma posterior, tipos equivocados, campos alucinados, truncados en el límite de tokens, prosa filtrada como "Aquí está tu JSON:".

**Approach two: validate after generation.**Generar libremente, analizar, validar contra el esquema, volver a intentar en caso de fallo. Confiable pero caro  pagas por cada nuevo intento, y los errores de truncado cuestan un giro extra por ocurrencia.

**Approach three: constrained decoding.**El proveedor impone el esquema en el momento de decodificar. Los tokens inválidos se enmasculan fuera de la distribución de muestreo. La salida está garantizada para analizar y garantizada para validar. El fracaso se desmorona en un modo: rechazo (el modelo decide que la entrada no encaja con el esquema).

Cada proveedor fronterizo de 2026 envía alguna forma de enfoque 3.

- **OpenAI.** `response_format: {type: "json_schema", strict: true}`Además`refusal`en la respuesta si el modelo disminuye.
- **Anthropic.**Ejecución del régimen en `tool_use`los insumos; `stop_reason: "refusal"`No es una cosa, pero `end_turn`Sin herramientas, la señal es la llamada.
- **Gemini.** `responseSchema`a nivel de solicitud; en 2026 Gemini lanzará restricciones gramaticales a nivel de tokens para tipos seleccionados.
- **Pydantic AI.** `output_type=InvoiceModel`emite una estructura `RunResult`se escribe en `InvoiceModel`¿ Qué ?
- **Zod (TypeScript).**Parser de tiempo de ejecución que valida la salida del proveedor con un esquema Zod; empareja con OpenAI `beta.chat.completions.parse`¿ Qué ?

El hilo común: declarar el esquema una vez, hacer cumplirlo de extremo a extremo.

## El concepto

### JSON Schema 2020-12  la lengua franca

Cada proveedor acepta JSON Schema 2020-12. Los constructos que más usas:

- `type`: uno de los `object`¿ Qué ?`array`¿ Qué ?`string`¿ Qué ?`number`¿ Qué ?`integer`¿ Qué ?`boolean`¿ Qué ?`null`¿ Qué ?
- `properties`: mapa del nombre de campo a subesquema.
- `required`: lista de nombres de campos que deben aparecer.
- `enum`: conjunto cerrado de valores permitidos.
- `minimum`- ¿ Qué ?`maximum`(números), `minLength`- ¿ Qué ?`maxLength`- ¿ Qué ?`pattern`¿Qué es eso?
- `items`: subesquema aplicado a todos los elementos del matriz.
- `additionalProperties`¿ Qué es esto ?`false`prohíbe campos adicionales (el valor predeterminado varía según el modo).

El modo estricto de OpenAI añade tres requisitos: cada propiedad debe estar en la lista `required`¿ Qué ?`additionalProperties: false`En todas partes, y no hay nada sin resolver.`$ref`Si rompe esto, la API devuelve 400 en el momento de la solicitud.

### Pydantic, la unión de Python

Pydantic v2 genera esquema JSON a partir de modelos en forma de clase de datos a través de `model_json_schema()`La IA de Pydantic envuelve esto para que escribas:

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

y el marco de agentes traduce el esquema en OpenAI modo estricto, Antropic `input_schema`, o Géminis .`responseSchema`La salida del modelo regresa como una tipografía.`Invoice`En el caso de los datos de la aplicación, el valor de la información se eleva.`ValidationError`con vías de error de tipografía.

### Zod, la unión de TypeScript

Zod (`z.object({customer: z.string(), ...})`El SDK Node de OpenAI expone `zodResponseFormat(Invoice)`que se traduce en la carga útil del esquema JSON de la API.

### Rechazos

El modo estricto no puede obligar al modelo a responder. Si la entrada no encaja en el esquema ("el correo electrónico era un poema, no una factura"), el modelo emite una`refusal`El código debe tratar esto como un resultado de primera clase, no como un fracaso. La negativa también es útil como señal de seguridad: un modelo solicitado para extraer un número de tarjeta de crédito de un correo electrónico de contenido protegido devuelve una negativa con la razón de seguridad adjunta.

### Descodación limitada en el espacio abierto

Las implementaciones de pesos abiertos utilizan tres técnicas.

1. **Grammar-based decoding**(El artículo`outlines`¿ Qué ?`guidance`¿ Qué ?`lm-format-enforcer`): construir un automático finito determinista a partir del esquema; en cada paso, enmascarar los logits de tokens que violarían el FSM.
2. **Logit masking with a JSON parser**: ejecutar un parser de JSON en streaming en un bloqueo con el modelo; en cada paso, calcular el conjunto de tokens válidos-próximamente.
3. **Speculative decoding with a verifier**El modelo de proyecto barato propone tokens, el verificador aplica el esquema.

Los proveedores comerciales escogen uno de estos detrás de escena. El estado de la técnica de 2026 es más rápido que la generación ordinaria para los resultados estructurados cortos y aproximadamente la misma velocidad para los largos.

### Los tres modos de falla

1. **Parse error.**La salida no es válida JSON. No puede ocurrir en modo estricto. Aún puede ocurrir en proveedores no estrictos.
2. **Schema violation.**La salida analiza pero viola el esquema. No puede suceder bajo modo estricto.
3. **Refusal.**El modelo se desacelera, debe ser tratado como un resultado tipado.

### Estrategia de retraso

Cuando estás fuera del modo estricto (uso de herramientas antropicas, OpenAI no estricto, Gemini más viejo), el patrón de recuperación es:

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

Un retraso es generalmente suficiente. Tres retrosas atrapan flocos de modelo débil. Más allá de tres es una señal de un mal esquema: el modelo no puede satisfacerlo para algunas entradas, y el prompt o el esquema necesita fijarse.

### Apoyo a los modelos pequeños

El decodificación restringida funciona en modelos pequeños. Un modelo abierto de parámetro 3B con aplicación gramatical supera a un modelo de parámetro 70B con incitación bruta en tareas estructuradas. Esta es la razón principal por la que las salidas estructuradas son importantes para la producción: separa la fiabilidad del tamaño del modelo.

```figure
constrained-decoding
```

## Usalo

`code/main.py`Envía un validador mínimo de JSON Schema 2020-12 en stdlib (tipos, requeridos, enum, min/max, patrón, elementos, propiedades adicionales).`Invoice`El sistema de análisis de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos

Qué ver:

- El validador devuelve una letra`[ValidationError]`Esta es la forma que desea que aparezca en la solicitud de retoma.
- La rama de rechazo NO vuelve a intentarlo. Registra y devuelve una negativa tipografada. Fase 14 · 09 utiliza rechazos como señal de seguridad.
- El `additionalProperties: false`comprobar los incendios en la entrada de prueba adversaria, mostrando por qué el modo estricto cierra la puerta a los campos alucinados.

## Envío

Esta lección produce`outputs/skill-structured-output-designer.md`. Dado un objetivo de extracción de texto libre (facturas, boletos de soporte, currículos, etc.), la habilidad produce un esquema JSON 2020-12 que es estrictamente compatible con el modo y un modelo Pydantic que lo refleja, con rechazo de tipo y manejo de nuevo.

## Los ejercicios

1. - ¿ Qué ?`code/main.py`Añadir un cuarto caso de prueba cuyo`total_usd`Confirme que el validador lo rechaza con el`minimum`el camino de restricción.

2. Extensión del validador para soportar `oneOf`El caso común: `line_item`es un producto o un servicio, etiquetado por `kind`. El modo estricto tiene reglas sutiles aquí; revisa la guía de salidas estructuradas de OpenAI.

3. Escriba el mismo esquema de factura que un modelo base Pydantic y compara `model_json_schema()`Identifique los conjuntos de Pydantic de campo por defecto que la versión de rodaje manual omite.

4. Mide las tasas de rechazo. Construye diez entradas que no deben ser extraíbles (una letra de canción, una prueba de matemáticas, un correo electrónico en blanco) y ejecutalas a través de un proveedor real con modo estricto. Cuente los rechazos frente a las salidas alucinadas. Esta es su verdadera base para los retemplazos conscientes de rechazo.

5. Lea la guía de salida estructurada de OpenAI de arriba a abajo. Identifique el constructo que prohíbe explícitamente en modo estricto que permite el simple esquema JSON. Luego diseñe un esquema que use el constructo prohibido no esencialmente y refactor para que sea estrictamente compatible.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| JSON Schema 2020-12 | "The schema spec" | IETF-draft schema dialect every modern provider speaks |
| Strict mode | "Guaranteed schema" | OpenAI flag that enforces schema via constrained decoding |
| Constrained decoding | "Logit masking" | Decode-time enforcement that masks invalid next-tokens |
| Refusal | "Model declines" | Typed outcome when input cannot fit the schema |
| Parse error | "Invalid JSON" | Output did not parse as JSON; impossible under strict |
| Schema violation | "Wrong shape" | Parsed but violated types / required / enum / range |
| `additionalProperties: false` | "No extras allowed" | Forbids unknown fields; required in OpenAI strict |
| Pydantic BaseModel | "Typed output" | Python class that emits and validates JSON Schema |
| Zod schema | "TypeScript output type" | TS runtime schema for provider output validation |
| Grammar enforcement | "Open-weights constrained decode" | FSM-based logit masking, as in outlines / guidance |

## Leer más

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) Requisitos estrictos de modo, rechazos y esquemas
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) Agosto 2024 puesta en marcha de la garantía de descifrado
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) enlaces de tipo output_type que se serializan a cada proveedor
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) la especificación canónica
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) Notas de despliegue de empresas y advertencias de modo estricto
