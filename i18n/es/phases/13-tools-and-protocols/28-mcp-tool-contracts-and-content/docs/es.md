# Contratos y contenido de las herramientas de MCP

> Una herramienta es segura para automatizar sólo cuando el descubrimiento, los argumentos, los resultados, la paginado y el transporte de metadatos se acuerdan en un contrato.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 07, 09, and 10
**Time:** ~120 minutes

## Objetivos de aprendizaje

- Definir las entradas y salidas de herramientas con JSON Schema 2020-12.
- Valida los resultados estructurados sin suponer que sean objetos JSON.
- Elija entre texto, imagen, audio, enlaces de recursos y recursos integrados.
- Rechazar el inseguro`x-mcp-header`las definiciones antes de que una herramienta llegue al modelo.
- Encode los valores de parámetro-título y verifique la paridad exacta de encabezado a cuerpo.
- Paginación transversal del cursor sin interpretar los valores del cursor.
- Enlazar y autorizar`completion/complete`¿Qué sugerencias?

## El problema

Llamar a una función Python es fácil. Llamar a una capacidad remota a través de un host de IA es un problema de contrato.

El servidor publica un descriptor. El cliente convierte ese descriptor en contexto de modelo e interfaz de usuario. El modelo crea argumentos. Una puerta de entrada puede enrutar la solicitud desde encabezados espejo. El servidor ejecuta la herramienta. El cliente decide entonces si el resultado es lo suficientemente seguro y válido para regresar al modelo.

Un límite débil corrompe toda la cadena.

Considere cinco fracasos:

- El descriptor dice que el resultado es un objeto, pero el servidor devuelve una matriz.
- El cliente deja de paginarse cuando `nextCursor`es una cuerda vacía.
- Un parámetro de token se refleja en un encabezado HTTP y se vuelve visible para los intermediarios.
- Un valor de enrutamiento de Unicode se envía como una cabecera en bruto, luego la puerta de entrada y el origen interpretan diferentes bytes.
- Un punto final de finalización sugiere un entorno de producción para un llamador que no puede acceder a él.

Ninguno de estos fallos se soluciona con una mejor solicitud.

## El oleoducto de contrato

Trata cada llamada de herramienta como cinco puertas:

1. **Discover.**Lea una lista de herramientas determinista y pagineada.
2. **Admit.**Valida cada descriptor y aplica la política de seguridad local.
3. **Invoke.**Valida los argumentos y construye metadatos de transporte.
4. **Execute.**Ejecutar el controlador y clasificar los fallos correctamente.
5. **Consume.**Valida los bloques de contenido y la salida estructurada antes de utilizar el modelo.

```figure
mcp-contract-pipeline
```

El host es el propietario de las puertas de admisión y consumo. Un servidor no puede obligar a un cliente a confiar en sus anotaciones, esquemas o salidas.

## JSON Es un límite de tiempo de ejecución

En MCP `2026-07-28`¿ Qué ?`inputSchema`y `outputSchema`usar JSON Schema. Cuando `$schema`Si no está presente, el dialecto predeterminado es 2020-12.

El esquema de entrada debe ser un objeto de esquema. Una herramienta sin argumentos debe decir exactamente lo que acepta:

```json
{
  "type": "object",
  "additionalProperties": false
}
```

Esto es más estricto que`{ "type": "object" }`, que acepta propiedades arbitrarias.

Un esquema de salida es opcional. Una vez que un servidor publica uno, cada herramienta completa
el resultado se compromete a devolver conforme `structuredContent`, incluidos los resultados
con`isError: true`. La bandera de error clasifica el resultado de ejecución; no
Los clientes deben validar el resultado en su lugar.
de confiar en el descriptor.

### Contenido estructurado es cualquier valor JSON

No codifique en forma dura `structuredContent`como un diccionario.

- un objeto;
- una matriz;
- una cuerda;
- un número;
- un booleano;
- `null`¿ Qué ?

Esta herramienta devuelve una matriz:

```json
{
  "name": "tag_catalog",
  "inputSchema": {
    "type": "object",
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "array",
    "items": {"type": "string"}
  }
}
```

Su resultado exitoso es válido:

```json
{
  "resultType": "complete",
  "content": [
    {
      "type": "text",
      "text": "[\"contracts\", \"mcp\", \"stateless\"]"
    }
  ],
  "structuredContent": ["contracts", "mcp", "stateless"],
  "isError": false
}
```

Para la compatibilidad, los resultados estructurados también deben incluir JSON serializado en un bloque de texto.`structuredContent`- Sí, es cierto.

### Un pequeño validador todavía enseña el límite

La lección utiliza un subconjunto deliberado de JSON Schema porque se mantiene dentro de la biblioteca estándar de Python.

- tipos de objeto, matriz, cadena, número entero, número, booleano y nulo;
- las propiedades requeridas;
- `additionalProperties: false`El artículo 1
- elementos de matriz;
- valores enum;
- longitud mínima de la cuerda.

La lección reutilizable es cuando se realiza la validación: después del descubrimiento de los descriptores, antes de la ejecución de los argumentos y antes del consumo para los resultados estructurados.

## Los bloques de contenido tienen diferentes costos

El `content`array puede combinar varios tipos de contenido.

| Type | Use it for | Main boundary |
|------|------------|---------------|
| `text` | Human and model-readable summaries | Treat text as untrusted output |
| `image` | Visual evidence encoded as base64 | Validate media type and size |
| `audio` | Spoken or recorded output encoded as base64 | Validate media type and duration limits |
| `resource_link` | A URI the client may fetch later | Reauthorize the later resource read |
| `resource` | Data embedded directly in the result | Enforce payload and content limits now |

Un enlace a un recurso no es prueba de que el recurso aparezca en `resources/list`El cliente sigue aplicando su política de recursos cuando sigue el URI.

Un recurso incorporado evita otro viaje de ida y vuelta, pero aumenta el tamaño de la respuesta actual. Utilice enlaces para artefactos grandes o que cambian de forma independiente. Utilice recursos incorporados para pequeñas pruebas que deben viajar de forma atómica con el resultado.

La lección es `evidence_bundle`El cliente valida cada bloque antes de aceptar el resultado.

## `x-mcp-header`¿Está enrutando metadatos

Una propiedad dentro .`inputSchema`puede declarar `x-mcp-header`. En Streamable HTTP, el cliente refleja ese argumento en `Mcp-Param-{name}`¿ Qué ?

```json
{
  "region": {
    "type": "string",
    "x-mcp-header": "Region"
  }
}
```

Con `region: "eu-west"`, el transporte puede emitir:

```http
Mcp-Param-Region: eu-west
```

La anotación existe para que un balanceador de carga, una puerta de entrada o un motor de política puedan encaminar sin analizar el cuerpo JSON.

El protocolo limita la anotación:

- el nombre del encabezado no está vacío y sigue la sintaxis de los tokens de nombre de campo HTTP;
- los nombres de encabezados son únicos sin importar el caso;
- el tipo de propiedad es cadena, número entero o booleano;
- `number`no se permite;
- La anotación aparece sólo en un miembro directo de `inputSchema.properties`El artículo 1
- Los valores enteros permanecen dentro de `-9007199254740991`por el`9007199254740991`¿ Qué ?

La regla de ubicación es sintáctica y está cerrada.
No sólo las propiedades que su validador comprende.
anotación bajo el objeto anidado `properties`, una `oneOf`rama,`items`, una
definición alcanzada por `$ref`, o cualquier esquema de salida. Resolviendo una referencia hace
no convertir el nodo de referencia en una propiedad directa de nivel superior.

Esta lección añade una política de implementación: rechazar descriptores que reflejan nombres como `password`¿ Qué ?`secret`¿ Qué ?`token`¿ Qué ?`api_key`, o`authorization`. La especificación oficial aconseja a los autores del servidor no reflejar parámetros sensibles. Un cliente puede convertir ese consejo en una regla de admisión dura.

Auditar el nombre del encabezado, no su valor.`Mcp-Param-Region`mientras que mantiene`eu-west`fuera del evento de auditoría.

### Valores de codificación antes de construir encabezados HTTP

Un valor de parámetro puede viajar como texto plano sólo cuando es una cadena no vacía
de caracteres ASCII visibles de `!`por el`~`y no se parece a la
Todo lo demás utiliza esta forma exacta:

```text
=?base64?{Base64UTF8}?=
```

`Base64UTF8`es base 64 estándar sobre los bytes UTF-8 exactos. No cortar,
Encode Unicode, cadenas vacías, espacios,
las pestañas, los caracteres de control, CR o LF, el espacio blanco de dirección o de dirección posterior, y cualquier
valor que comienza con `=?base64?`. codificar un valor de sentinela de nuevo es
¿Qué permite al receptor recuperar el texto original literal en lugar de decodificar
como una sintaxis de transporte.

Los booleanos se traducen en letras pequeñas .`true`o `false`. Integres rendidos en base 10 y
Los valores fuera de ese rango deben permanecer dentro del rango de enteros seguros de JavaScript.
se rechazan en lugar de ser redondeadas por un intermediario.

### El servidor verifica la copia espejo

La generación de encabezados es sólo la mitad del cliente.
el servidor deberá:

1. encontrar reconocido `Mcp-Param-*`nombres sin tener en cuenta el caso de título-nombre;
2. Descifrar la forma exacta de base64 sentinela cuando esté presente;
3. comparar exactamente el texto decodificado con el correspondiente argumento de cuerpo JSON;
4. rechazar una falta, duplicación, inesperada, malformada o no coincidente
   cabezal reconocido antes de su envío.

El rechazo es HTTP `400`con código de error JSON-RPC `-32020`Ni el
El valor del cuerpo ni su forma de encabezado codificada pertenecen al registro de auditoría.
el nombre de encabezado reconocido y la categoría de rechazo solamente.

`code/main.py`Modela este límite directamente. [Lesson 09](../../09-mcp-transports/)
cubre el orden de validación HTTP Streamable más amplio, incluido el método y
Paridad de protocolo-versión.

## Los maldictores de la página son opacos

Las operaciones de lista MCP utilizan la paginado del cursor. El servidor selecciona el tamaño de la página y el formato del cursor. El cliente obtiene una decisión:

```python
if result.get("nextCursor") is None:
    break
cursor = result["nextCursor"]
```

No escribas esto:

```python
if not result.get("nextCursor"):
    break
```

Una cadena vacía es un cursor válido. La verdad se detendría demasiado pronto.

Los clientes no deben decodificar un cursor, incrementarlo, compararlo con un cursor anterior para ordenar o inferir un número de página. Un servidor puede firmar un cursor, vincularlo a una versión de catálogo o mapearlo en estado privado. Ese es el detalle de implementación del servidor.

El servidor muestra regresa deliberadamente `""`El cliente debe enviar ese valor exacto en la segunda solicitud.

```text
<first request with no cursor>
<second request with cursor "">
```

Los cursores inválidos producen parámetros inválidos JSON-RPC, código `-32602`¿ Qué ?

## La finalización es una superficie autorizada

`completion/complete`El formulario de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de

Una solicitud de finalización nombra una referencia y el argumento que se completa:

```json
{
  "method": "completion/complete",
  "params": {
    "ref": {
      "type": "ref/prompt",
      "name": "deployment_review"
    },
    "argument": {
      "name": "environment",
      "value": "st"
    }
  }
}
```

El resultado devuelve un máximo de 100 valores y puede informar `total`Además`hasMore`¿ Qué ?

Aplicar el mismo límite de autorización utilizado por el prompt o recurso de referencia.`development`y `staging`Sólo un operador puede recibir`production`¿ Qué ?

La finalización de la producción también requiere:

- validación de entrada;
- filtración de la información de los llamados;
- solicitar desacusación en el cliente;
- limitación de velocidad en el servidor;
- el conteo de resultados limitados;
- registros que no exponan valores sensibles de sugerencia.

La finalización es asistencia, no el bypass de descubrimiento.

## Dos capas de error

Mantenga los errores de protocolo separados de los errores de ejecución de la herramienta.

Utilice un error JSON-RPC cuando la solicitud de MCP no pueda ser enviada correctamente:

- nombre desconocido de la herramienta;
- forma de solicitud malformada;
- los metadatos de la solicitud que faltan;
- Cursor inválido.

Utilice un resultado completo de la herramienta con `isError: true`cuando la invocación llegue a la herramienta y la herramienta informe de un fallo que pueda ser actualizado:

- no se dispone de una fuente de informe;
- una fecha está fuera del rango de apoyo;
- una regla de negocio rechaza la operación solicitada.

Los modelos a menudo pueden reparar un error de ejecución de herramienta. No pueden reparar un servidor que violó su propio esquema de salida.

Si la herramienta declara un esquema de salida, modelar un fallo actuable dentro de ese
El esquema.`route_report`el fallo devuelve su región solicitada con
`accepted: false`, junto con el texto de error legible por el hombre y `isError: true`¿ Qué ?

## Construye el mismo

`code/main.py`construye ambos lados del límite con la biblioteca estándar Python.

El servidor implementa:

- validación de metadatos del MCP por solicitud;
- `server/discover`con herramientas y capacidades de finalización;
- determinista `tools/list`la paginado;
- cuatro descriptores de herramientas, incluido uno que debe rechazarse;
- salida estructurada de matriz;
- cada tipo de bloque de contenido de herramienta actual;
- una puerta de paridad HTTP que descifra los encabezados de parámetros reconocidos y
  devuelve HTTP `400`más JSON-RPC `-32020`en caso de falta de coincidencia;
- finalización autorizada y limitada.

El cliente realiza:

- admisión de descriptor;
- árbol lleno `x-mcp-header`la validación de la colocación y la política de campo sensible;
- codificación exacta del valor ASCII o base64 UTF-8 visible claramente;
- un bucle de cursor opaco que siga una cadena vacía;
- argumentos y validación de resultados;
- validación de bloques de contenido;
- header de auditoría que contenga nombres pero no valores.

El descriptor deliberadamente inseguro es el de enseñanza de datos.

## Usalo

Desde la raíz del repositorio:

```bash
cd phases/13-tools-and-protocols/28-mcp-tool-contracts-and-content/code
python3 main.py
python3 -m unittest discover tests -v
```

Las impresiones demo admitidas herramientas, el descriptor rechazado, ambos pagination
solicitudes, contenido estructurado de la matriz, tipos de bloques de contenido, encabezado espejo
nombres, ya sea el valor requerido de codificación, el estado de paridad HTTP, y
valores de finalización filtrados por el receptor.

## Laboratorio interactivo

Abriéndome .`code/main.py`y localizar`TOOLS`¿ Qué ?

1. Cambiar`tag_catalog.outputSchema.type`de la`array`¿ Qué ?`object`¿ Qué ?
2. Ejecutar la demostración. El cliente debe rechazar el conjunto devuelto.
3. Restablezca el esquema.
4. Mantenga la primera página.`nextCursor`¿ Cómo ?`""`, luego hacer la página final regresar
   `nextCursor: None`en lugar de omitir el campo.
5. Haga los análisis y comparar el rastro del cursor.
6. Añadir`x-mcp-header: "Authorization"`a una propiedad de cuerdas.
7. La admisión de descriptor de confirmación lo rechaza antes de invocarlo.
8. Prueba .`region`valores que contienen Unicode, una nueva línea, espacios circundantes, y
   el texto literal `=?base64?SGVsbG8=?=`. Descifrar cada encabezado emitido y probar
   El valor original sobrevive exactamente.
9. Mueve la anotación hacia abajo `oneOf`¿ Qué ?`items`, o un `$ref`Confirma la definición.
   cada descriptor se rechaza incluso si esa rama nunca es utilizada por la demostración.
10. Elimine el encabezado reconocido o cambie su valor decodificado.
    estado de las declaraciones de límite `400`y código JSON-RPC `-32020`¿ Qué ?

El punto no es memorizar una forma JSON, es ver cómo falla cada puerta en el límite que la posee.

## Laboratorio de práctica

Extendiendo el laboratorio de contrato con un `search_evidence`- ¿Qué?

Requisitos:

1. Su esquema de entrada acepta `query`¿ Qué ?`limit`, y una caja fuerte .`region`campo de enrutamiento.
2. Su esquema de salida es una matriz de objetos con `uri`¿ Qué ?`title`, y `score`¿ Qué ?
3. El resultado incluye texto de compatibilidad y un enlace de recursos por elemento.
4. Los argumentos rechazan propiedades desconocidas.
5. `limit`se limita por la validación de la solicitud.
6. Un llamador sin acceso a una URI nunca ve esa URI a través de la finalización o salida de la herramienta.
7. Las pruebas incluyen una puntuación no conforme, una anotación de encabezado inválida y una lista de dos páginas.
8. Las pruebas de valores de encabezado abarcan ASCII visible, Unicode, caracteres de control,
   espacio blanco, texto de aspecto sentinela, y ambos límites de números enteros seguros de JavaScript.
9. El fijo HTTP acepta nombres de encabezados insensibles a los casos pero rechaza los faltantes
   o valores reconocidos no coincidentes con el estado `400`y código `-32020`¿ Qué ?

## Artículo enviado

`outputs/skill-mcp-contract-reviewer.md`Es una habilidad plana de revisión reutilizable. Dale un descriptor de herramientas, resultados de muestra, comportamiento de paginado y política de finalización. devuelve una decisión de admisión, un plan de validación de resultados, una política de encabezado y pruebas de falla concretas.

## Verifique el hecho

La lección está completa cuando estas declaraciones son ciertas:

- `tools/list`devuelve el mismo orden lógico en llamadas repetidas.
- El cliente realiza una segunda solicitud cuando `nextCursor`¿ Es verdad ?`""`¿ Qué ?
- El descriptor de encabezado sensible no seguro está excluido mientras que otras herramientas siguen disponibles.
- Una matriz pasa su esquema de salida de matriz.
- Un objeto falla en ese mismo esquema de matriz.
- Los resultados de error no pueden omitir o violar un esquema de salida publicado.
- El texto, la imagen, el audio, el enlace de recursos y los bloques de recursos incorporados validan.
- Los eventos de auditoría de encabezado contienen nombres y no valores.
- ASCII visible en plano sigue siendo plano; Unicode, control, relleno, vacío, y
  valores de aspecto sentinel viaje de ida y vuelta a través de base64 exacto codificación UTF-8.
- Los números enteros especulares fuera del rango seguro de JavaScript se rechazan.
- Anotadas en `oneOf`¿ Qué ?`items`, objetos anidados, `$ref`las definiciones, o
  los regímenes de salida se rechazan durante la admisión.
- Los nombres de encabezados reconocidos que no son sensibles a los casos sólo pasan cuando el valor descifrado
  corresponde exactamente al cuerpo; copias faltantes o no coincidentes producen HTTP `400`
  y JSON-RPC `-32020`¿ Qué ?
- El análisis nunca regresa .`production`¿ Qué ?
- Una falla de herramienta utiliza `isError: true`Una llamada de protocolo malformada utiliza JSON-RPC `error`¿ Qué ?

## Modos de falla de producción

| Failure | What the learner sees | Correct response |
|---------|-----------------------|------------------|
| Client assumes object output | Valid arrays fail or are silently wrapped | Validate against the published schema without object-only types |
| Empty cursor treated as false | Final pages disappear | Continue whenever `nextCursor` is present and non-null |
| Sensitive value mirrored | Secret appears in proxy, WAF, or trace data | Reject the descriptor and keep secrets in protected request data |
| Raw Unicode or whitespace mirrored | Gateway and origin disagree or the value is normalized | Use exact base64 UTF-8 sentinel encoding and compare after decoding |
| Annotation hidden in a schema branch | A client misses routing metadata during admission | Traverse the entire schema tree and allow only direct top-level properties |
| Large integer mirrored | JavaScript intermediary rounds the routing value | Reject values outside the JavaScript safe integer range |
| Header and body disagree | Gateway routes one target while the origin executes another | Reject before dispatch with HTTP `400` and JSON-RPC `-32020` |
| Output schema ignored | Downstream code consumes corrupt structure | Validate before model or application use |
| Resource link trusted automatically | Caller follows an unauthorized URI | Reauthorize every resource read |
| Completion shares global suggestions | Hidden tenant names leak | Filter by caller, reference, and authorization |
| Tool annotations treated as policy | Destructive operation bypasses confirmation | Enforce authorization and approval outside annotations |
| One malformed tool breaks discovery | Entire server becomes unavailable | Reject the bad descriptor and admit valid tools independently |

## Conexión de Capstone

La piedra angular de la Fase 13 necesita una puerta de entrada que pueda fusionar herramientas de varios servidores.

Utilice el artefacto para calificar cuatro piezas de evidencia de piedra angular:

- descubrimiento determinista y completo en páginas;
- la validación del descriptor antes de la exposición del modelo;
- la salida estructurada validada más los bloques de contenido limitados;
- la finalización y el enrutamiento de metadatos que preserven los límites de autorización.

No reclame la compatibilidad de la puerta de enlace de un exitoso `tools/call`Captura el descriptor, el rastro de la página, el conjunto de herramientas admitidas, el conjunto de herramientas rechazado y un resultado validado.

## Términos clave

| Term | Meaning |
|------|---------|
| `inputSchema` | JSON Schema object defining accepted tool arguments |
| `outputSchema` | Optional JSON Schema defining `structuredContent` |
| `structuredContent` | Any JSON value produced by a tool result |
| Content block | Typed text, image, audio, resource link, or embedded resource |
| `x-mcp-header` | Schema annotation that mirrors a primitive argument into Streamable HTTP metadata |
| Opaque cursor | Server-issued pagination token whose value the client does not interpret |
| Completion reference | Prompt name or resource URI/template whose argument is being completed |
| Admission | Client decision to expose or reject a discovered descriptor |

## Leer más

- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Completion](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/completion)
- [MCP Pagination](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/pagination)
- [MCP Streamable HTTP Parameter Headers](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http#custom-headers-from-tool-parameters)
