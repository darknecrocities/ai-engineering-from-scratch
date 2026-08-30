# Ingeniería de conformidad de MCP: versionamiento, pruebas y operaciones

> Un servidor no es conforme porque su camino feliz trabajó a través de un SDK. La conformidad vive en el cable, en los límites de la versión, a través de intermediarios y durante el rollback.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 17 (gateways), Phase 13 · 30 (registry admission)
**Time:** ~100 minutes

## Objetivos de aprendizaje

- Convierta las reglas normativas de MCP en transcripciones de oro y negativas.
- Manténgase estricto .`2026-07-28`comportamiento separado de la herencia limitada de retroceso.
- Distinguir campos adictivos desconocidos de un campo desconocido inválido `resultType`¿ Qué ?
- Compare la evidencia cruda de JSON-RPC con una vista normalizada de SDK.
- Prueba la integridad de la cabeza y el cuerpo a través de un límite de proxy real.
- Las liberaciones de la puerta con transcripción editada, salud y evidencia de retroceso.

## El problema

Su cliente llama .`tools/list`y obtiene herramientas.

Ese resultado deja sin respuesta preguntas importantes:

- ¿La solicitud contenía metadatos modernos del protocolo por solicitud?
- ¿ Qué ?`MCP-Protocol-Version`¿ Qué ?`Mcp-Method`, y `Mcp-Name`¿Combina con el cuerpo JSON-RPC?
- ¿La respuesta contiene un valor válido?`resultType`¿O el SDK lo sintetizó?
- ¿El cliente conservaría un futuro campo aditivo?
- ¿Un error reconocido en la actualidad desencadenaría accidentalmente un apretón de manos heredado?
- ¿Se ha conservado el estado de origen y el error JSON-RPC?
- ¿El serializar de notificación emitió una respuesta prohibida?
- ¿Pueden las operaciones probar por qué se promovió una liberación o se retrocedió sin guardar secretos?

Conformance es un conjunto de invariantes observables. Construye un arnés que capture esos invariantes antes de que el tráfico de producción tenga que descubrirlos.

```figure
mcp-conformance-operations
```

## Comience con las versiones de Eras

MCP `2026-07-28`El sistema de datos de la información de la información de la persona que solicita la información de la persona que solicita la información de la persona que solicita la información.`params._meta.io.modelcontextprotocol/protocolVersion`y `params._meta.io.modelcontextprotocol/clientCapabilities`Las claves con espacios de nombres son importantes , al descubierto .`protocolVersion`o `clientCapabilities`Los alias están malformados. Cuando los encabezados de enrutamiento espejo están presentes en el límite HTTP, sus valores deben estar de acuerdo con el cuerpo JSON-RPC.`resultType`¿ Qué ?

Versiones a través de `2025-11-25`Usar la era de inicialización anterior.`resultType`se interpretará como completa sólo después de que el cliente haya seleccionado la época anterior.

No crea un validador permisorio que acepte ambas formas a la vez.

| Branch | Entry evidence | Missing `resultType` | Initialization |
|---|---|---|---|
| Modern | Successful `server/discover` or recognized modern response | Invalid | Not the default path |
| Legacy | Configured allowlist plus a valid legacy `initialize` result after an inconclusive modern probe | Interpreted as complete | Required by that era |

La separación impide que un compañero moderno malformado sea recompensado con una validación más débil.

### Modo estricto

El modo estricto requiere pruebas de comportamiento moderno.`server/discover`prueba la rama moderna. Un error JSON-RPC moderno reconocido también lo prueba. Correcta la solicitud o detente. Nunca rebajar porque el servidor regresó`-32020`¿ Qué ?`-32021`, o`-32022`¿ Qué ?

### Modo de retroceso

El modo fallback realiza una sonda moderna limitada. Un tiempo de espera, respuesta vacía, conexión cerrada o respuesta no reconocida es inconclusivo. No prueba que el par sea heredado. Sólo un punto final configurado explícitamente o listado para la compatibilidad puede recibir una sonda heredada limitada, y el cliente selecciona la rama heredada solo después de validar la prueba de la sonda.`initialize`el resultado y la revisión negociada del legado.

Fallback no es tried legacy después de cualquier error. Un error moderno reconocido contiene información de corrección útil.

Esto evita que un atacante, interrupción o filtro de proxy de obligar a rebajar la respuesta moderna.

Sin ese hecho, un campo faltante puede parecer aceptable en una prueba y inválido en otra.

## Construye un cuerpo de transcripciones

Un dispositivo de transcripción registra lo que cruzó el límite, no sólo la llamada SDK:

```json
{
  "name": "golden-modern-list",
  "era": "modern",
  "headers": {
    "MCP-Protocol-Version": "2026-07-28",
    "Mcp-Method": "tools/list"
  },
  "request": {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {
      "_meta": {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {}
      }
    }
  },
  "responseStatus": 200,
  "responseBody": {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
      "resultType": "complete",
      "tools": []
    }
  }
}
```

Mantenga dos clases de accesorios.

### Transcripciones doradas

Las transcripciones doradas demuestran un comportamiento aceptado:

- solicitud de descubrimiento o método moderno con metadatos y encabezados correspondientes
- resultado completo con los campos requeridos
- `input_required`resultado cuando el método puede solicitar más entradas
- Resultado de extensión sólo después de que se anunciara la capacidad correspondiente
- resultado legado sin `resultType`, pero sólo en la era del legado seleccionado
- procesamiento de notificaciones sin respuesta JSON-RPC

Una transcripción dorada es precisa, no grande. Mantenga las identidades volátiles y las estampillas de tiempo deterministas o normalizálas antes de la comparación.

### Transcripciones negativas

Las transcripciones negativas demuestran el comportamiento de rechazo:

- Desajuste de cabeza y cuerpo
- Falta de capacidades de solicitud
- versión de protocolo no compatible
- falta de modernos `resultType`
- desconocido o no anunciado `resultType`
- respuesta `jsonrpc`que no sea `2.0`o un ID que difiere en valor o tipo JSON
- una respuesta que contiene ambas `result`y `error`, o ninguno de los dos
- un error sin un número entero `code`y cuerda`message`
- un error de protocolo conocido asignado al estado HTTP incorrecto
- respuesta emitida para una notificación
- Envase de JSON-RPC malformada
- colapso de un error de protocolo

Para cada caso negativo, asegure el límite de rechazo y el código de error estable. La llamada falló es demasiado débil.`-32020`pueden parecer fracasos mientras cuentan historias completamente diferentes a los operadores.

La fijación de incompatibilidad de encabezado debe incluir la respuesta HTTP 400 JSON-RPC real del servidor con el ID de solicitud de coincidencia y código de error `-32020`. Aplicar automáticamente cuando el validador local observa .`HeaderMismatch`Un caso con HTTP 500 y ningún cuerpo falla incluso cuando el código de rechazo local era correcto. Un arnés que se detiene después de que su propio validador de solicitud lanza ha probado sólo a sí mismo, no el comportamiento de cable del servidor.

El proyecto oficial de conformidad MCP es útil como una suite externa y una referencia de versiones. Guarde sus transcripciones locales también. Capturan su proxy, SDK, autenticación, extensiones y ruta de liberación, que una suite general no puede conocer.

## Los valores de encabezado deben coincidir con el cuerpo de RPC

En el moderno HTTP Streamable, los intermediarios pueden encaminar o hacer cumplir la política utilizando encabezados espejo. El cuerpo JSON-RPC sigue siendo la fuente de verdad del protocolo.

Valida en el siguiente orden:

1. Analizar y validar los tipos de envase y metadatos JSON-RPC.
2. Comparar`MCP-Protocol-Version`con`params._meta.io.modelcontextprotocol/protocolVersion`¿ Qué ?
3. Comparar`Mcp-Method`con`method`¿ Qué ?
4. Cuando el método tiene un nombre de enrutamiento, compara `Mcp-Name`con el valor corporal correspondiente.
5. Una vez que se establezca la igualdad, se decidirá si se admite la versión y el conjunto de capacidades correspondientes.

Este orden distingue la discrepancia .`-32020`de la versión no compatible `-32022`También impide que una puerta de entrada autorice el nombre del encabezado mientras el origen ejecuta un nombre de cuerpo diferente.

Los nombres de campos HTTP son insensibles a los casos, mientras que sus valores siguen siendo sensibles a los casos. Normaliza los nombres de encabezados antes de buscar y rechaza los duplicados contradictorios. Para un espacio blanco inseguro, no ASCII o de seguimiento o de seguimiento.`Mcp-Name`, descifrar el exacto`=?base64?{Base64EncodedValue}?=`Rechazar un sentinela incompleto, inválido Base64, inválido UTF-8, o valor cruo inseguro con `-32020`. El espacio blanco en bruto circundante es inválido incluso cuando el cuerpo contiene los mismos caracteres porque ese valor requiere codificación sentinel antes del transporte.

Un intermediario puede rechazar HTTP malformado antes de que una solicitud llegue al servidor MCP, por lo que su falla puede ser un error HTTP sin JSON-RPC. Captura si un rechazo vino del intermediario o del origen. El servidor MCP de origen debe usar el contrato de error de protocolo cuando maneja una solicitud JSON-RPC válida.

## Los campos desconocidos no son resultados desconocidos

La compatibilidad con el futuro requiere dos reglas diferentes.

### Campos adictivos desconocidos

Objetos de resultado y `_meta`Los mapas pueden obtener campos. Un validador debe conservar o ignorar un campo aditivo según su función, a menos que el campo viole un contrato reservado.`futureHint`junto a un resultado conocido.

Si usted es un proxy transparente, conservar un campo desconocido es generalmente más seguro que deshacerse de él. Si usted es un cliente de aplicación, ignorarlo puede ser válido. Su prueba diferencial aún debe revelar que el SDK lo omitió por lo que el comportamiento es deliberado.

### No se sabe .`resultType`

`resultType`El uso de los resultados modernos`complete`o `input_required`. Una extensión puede añadir otro valor sólo cuando se publicitara su capacidad.`task`en el contexto de la capacidad negociada.

El cliente no sabe el ciclo de vida que desecharía.

La misma respuesta en bruto puede, por lo tanto, contener un campo desconocido aceptable y un tipo de resultado desconocido inaceptable.

El discriminador es sólo la primera capa. Valida la carga útil específica del método después de él.`tools/list`El resultado necesita un `tools`array cuyos descriptores tienen nombres únicos no vacíos, descripciones útiles y raíz de objeto `inputSchema`Los valores de la información`task`El resultado es válido sólo para un beneficiario `tools/call`con la capacidad de tareas y requiere`taskId`, estado conocido, creación y actualización de timestamps, y `ttlMs`, más un intervalo de votación opcional válido.`completion/complete`El resultado requiere un `completion`objeto con un valor de cadena no superior a 100, un número entero no negativo opcional `total`que no sea menor que los valores devueltos, y un booleano opcional `hasMore`Un buen escrito .`resultType`no puede hacer un conformante de carga útil malformado.

## La notificación invariante

Una notificación JSON-RPC no tiene `id`El receptor no debe enviar una respuesta de éxito o error JSON-RPC.

Para una forma de notificación HTTP aceptada, el arnés espera una HTTP `202`con un cuerpo vacío.`2026-07-28`no define notificaciones centrales de cliente a servidor sobre Streamable HTTP. La muestra utiliza una notificación de extensión de curso con espacio de nombres solo para probar la invariante de serie de una sola dirección. No lo presente como un nuevo método central.

Prueba el serializer, no sólo el manipulador.`None`Mientras que el middleware lo envuelve en un objeto de éxito JSON. Captura los bytes de salida final.

## Añadir un SDK Diferencial

Los SDK a menudo convierten objetos de cable en tipos de lenguaje convenientes. Eso es útil, pero un objeto normalizado no puede probar lo que se recibió.

Para cada dispositivo de alto riesgo, captura:

1. Estatus bruto, encabezados y cuerpo de respuesta antes de la descifrado SDK.
2. Valor de retorno normalizado o excepción de SDK.
3. La proyección semántica esperada para la era seleccionada.
4. Los campos levantados, sintetizados, despojados o modificados por el SDK.

La muestra permite eliminar solo SDK de contabilidad de cajeros conocidos como `resultType`¿ Qué ?`_meta`¿ Qué ?`ttlMs`, y `cacheScope`La aplicación de la aplicación de carga útil.`futureHint`porque ese campo semántico desconocido desapareció.

No asuma que cada diferencia es un error de SDK. El punto es hacer visible la transformación. Decida si su componente es un punto final de aplicación, que puede ignorar un campo aditivo, o un intermediario transparente, que debe preservarlo.

Ejecutar el diferencial contra cada SDK y versión que envíe. Si dos SDK normalizan la misma transcripción de manera diferente, la política de liberación debe decir qué comportamiento es aceptable en lugar de elegir la salida más conveniente después del hecho.

## Captura pruebas de proxy

La mayoría de las fallas de MCP de producción ocurren en más de un proceso.

| View | Minimum evidence |
|---|---|
| Ingress | request headers, JSON-RPC body, content type, authenticated route, receive time |
| Origin | forwarded headers and body digest, origin status, response headers and body |
| Egress | client-visible status, headers, body, and send time |

La muestra detecta dos transformaciones comunes:

- un error HTTP 400 o 404 JSON-RPC de origen se convierte en un proxy genérico 500
- el cuerpo de salida JSON-RPC difiere del cuerpo de origen

Añadir afirmaciones específicas de implementación para el tipo de contenido, `Accept`, compresión, SSE escaneado por solicitud, encabezados de caché y correlación de rastreo. Captura ambos lados de la terminación de TLS cuando la política lo permita. Nunca registres credenciales sólo para probar la ruta.

## Reedición antes de que las pruebas dejan la memoria

La redacción es parte de las operaciones de conformidad, no un trabajo de limpieza posterior. Aplica antes de la serialización, hashing, registros, artefactos de prueba o cargas de fallas.

La casilla de muestra pliega los nombres de las claves y elimina los separadores antes de combinarlos, luego sustituye recursivamente los valores de las claves como `Authorization`¿ Qué ?`Cookie`¿ Qué ?`Set-Cookie`¿ Qué ?`X-Api-Key`¿ Qué ?`accessToken`¿ Qué ?`clientSecret`¿ Qué ?`registrationAccessToken`¿ Qué ?`token`¿ Qué ?`password`¿ Qué ?`secret`, y `api_key`. la canonización y el denilista deben utilizar la misma forma para que las variantes camel, con hifas, subyacentes y puntos no puedan pasar por alto la política de los demás.`query`pueden contener datos personales o regulados.

El archivo de datos de archivo de datos de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archivo de archi

## Haga de la salud y el retroceso parte de la puerta

El protocolo de conformidad es necesario pero no suficiente para la liberación. Un candidato conformante puede todavía tiempo fuera, fuga de memoria o sobrecarga de una dependencia.

Definir una ventana de salud antes de la puesta en marcha:

- el número mínimo de muestras
- tasa máxima de error
- el percentil máximo de latencia
- limitaciones de saturación o de recursos
- duración de la observación
- comparación con el límite de referencia admitido

También debe definir la evidencia de retroceso antes del lanzamiento:

- versión exacta anterior
- Digestión de pruebas de admisión
- SHA-256 artefactos y pines de descripción
- estado actual del Registro
- Resultado actual de salud
- procedimiento de restauración de la ruta
- una certificación sobre esos campos exactos de una identidad de controlador de liberación de confianza

Requerir que el objetivo de retroceso sea verificado y saludable antes de la promoción, no solo después de que el candidato falla.

Si un candidato falla y el objetivo de retroceso carece de esa evidencia, detenga el tráfico en lugar de adivinar.

No reduzca la preparación para las verificaciones de veracidad, como una versión no vacía, `healthy: "yes"`La muestra requiere tipos exactos, un estado activo, tres digestos SHA-256, un firme de confianza y una certificación HMAC-SHA-256 válida sobre la carga útil completa de retroceso. Su clave de demostración determinista es un dispositivo no secreto. Inyecta una clave protegida, un resultado de verificación de KMS o un verificador de certificación de llave pública en el límite de liberación en producción.

La puerta de liberación también rechaza la transcripción vacía, el diferencial SDK o la evidencia de proxy.

## Construye el mismo

Ejecutar el arnés estándar de la biblioteca:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
```

La demostración ejecuta exactamente quince transcripciones doradas y negativas, incluyendo resultados de finalización válidos y malformados, compara un resultado crudo con una vista SDK, inspecciona un proxy que se derrumbó un error de origen, evalúa la salud, autentica la evidencia de retroceso y selecciona ese objetivo.

Forma esperada:

```json
{
  "transcriptsPassed": 15,
  "transcriptsTotal": 15,
  "sdkDroppedFields": ["futureHint"],
  "proxyIssues": [
    "proxy collapsed a protocol error into HTTP 500",
    "proxy changed the origin JSON-RPC body"
  ],
  "releaseAction": "rollback",
  "evidenceDigest": "..."
}
```

Leer .`code/main.py`en este orden:

1. `validate_request()`aplica las normas de solicitud y encabezado específicas de la época.
2. `validate_result()`Se trata de un sistema de control de la información y de la información que se utiliza para determinar si se trata de un sistema de control de datos.
3. `select_era()`La Comisión ha adoptado una política de retroceso estricta y limitada.
4. `run_transcript()`evalúa las luces doradas y negativas.
5. `compare_sdk_view()`expone las diferencias de normalización.
6. `inspect_proxy()`compara la entrada, origen y salida de pruebas.
7. `redact()`elimina secretos obvios antes de que se hacha la evidencia.
8. `rollback_evidence_ready()`valida los campos de pines exactos y el certificado de liberación de confianza.
9. `ReleaseGate.evaluate()`se une a la conformidad no vacía, SDK, proxy, salud y pruebas de retroceso.

## Usalo

- Conduce el arnés en cuatro puntos:

1. En cada cambio de implementación con un adaptador de prueba en proceso.
2. Contra los binarios de cliente y servidor construidos sobre el transporte real.
3. A través del proxy o puerta de enlace desplegados en un entorno de puesta en escena.
4. Durante el despliegue de canarios con pruebas de salud y retroceso.

Mantenga los mismos nombres de casos estables a través de capas. `negative-header-body-mismatch`El registro de pruebas será diferente porque el límite cambió; el requisito no debería ser el mismo.

Almacenar esquemas de fijación en control de versión. Almacenar pruebas de ejecución editadas en su sistema de liberación. Almacenar capturas crudas de corta duración sólo bajo controles de acceso de incidentes.

## Laboratorio interactivo

### Laboratorio A: demostrar el límite de la era

Desde el`code`directorio, Python abierto:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/code
python3 -q
```

- ¿Qué quieres decir ?

```python
from main import *
validate_result({"tools": []}, "legacy")
validate_result({"tools": []}, "modern")
```

La llamada heredada se inhala .`complete`La llamada moderna se eleva`ProtocolViolation`Ahora prueba de retroceso:

```python
select_era({"kind": "timeout"}, "fallback")
select_era(
    {"kind": "timeout"},
    "fallback",
    legacy_allowed=True,
    legacy_evidence={"kind": "initialize_success", "protocolVersion": LEGACY_VERSION},
)
select_era({"kind": "jsonrpc_error", "code": -32021}, "fallback")
```

La primera hora de espera no se cierra porque el silencio no es evidencia heredada. La segunda llamada selecciona herencia sólo porque la configuración lo permite y se observó un resultado de inicialización heredada válido. El error de capacidad faltante reconocido prueba la rama moderna.

### Laboratorio B: campo aditivo versus discriminador

```python
validate_result({"resultType": "complete", "tools": [], "futureHint": True}, "modern")
validate_result({"resultType": "future_mode", "tools": []}, "modern")
```

El primer resultado conserva `futureHint`La segunda se rechaza porque el discriminador del ciclo de vida es desconocido.

### Laboratorio C: inspeccionar una transformación de SDK

```python
compare_sdk_view(
    {"resultType": "complete", "tools": [], "futureHint": {"mode": "new"}},
    {"tools": []},
)
```

Decida si su componente puede ignorar `futureHint`Si no se puede hacer una elección, no se debe borrar el diferencial.

### Laboratorio D: repara el proxy

Modifique el intercambio de demostraciones para que la salida preserve el estado de origen y el cuerpo.`python3 main.py`Los problemas de proxy deberían desaparecer, pero el diferencial SDK aún bloquea la promoción.`futureHint`en la vista SDK y observar el cambio de acción a `promote`Cuando todas las fuentes de evidencia pasen.

## Laboratorio de práctica

Añadir transcripciones de SSE a la solicitud al arnés.

Requisitos:

- Captura el estado de la respuesta, el tipo de contenido, los eventos SSE ordenados y la terminación del stream.
- Demostrar que cada evento JSON-RPC tiene un resultado o error válido específico de la época.
- Añadir un caso negativo para un proxy que amortiza el flujo completo antes de reenviar.
- Añadir un caso negativo para un evento SSE cuyo ID JSON-RPC difiere de la solicitud.
- Rediseña los datos del evento antes de escribir pruebas.
- Incluya la duración del flujo, la latencia del primer evento y el recuento de eventos en la ventana de salud.
- Haz que la puerta de liberación escoge sólo un objetivo de retroceso evidenciado cuando la corriente falla.

El éxito significa que el mismo caso se ejecuta directamente y a través del proxy, con un informe que identifica el límite exacto que cambió el comportamiento.

## Artículo enviado

Esta lección nos lleva .`outputs/skill-mcp-conformance-release-gate.md`. Utilice para convertir un servidor, cliente, puerta de enlace o cambio de SDK en una matriz de conformidad con versión y decisión de liberación. El artefacto requiere evidencia de alambre bruto, casos negativos, selección explícita de la era, diferenciales de SDK, prueba de proxy, redacción, umbrales de salud y evidencia de retroceso.

## Verifique el hecho

Ejecutar la suite demo y determinista:

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

La verificación deberá demostrar:

- Cada transcripción dorada y negativa incluida alcanza su resultado esperado
- Las solicitudes modernas requieren las claves de metadatos con nombres espaciados
- Los nombres de encabezados HTTP se combinan de manera insensible a los casos y se codifican `Mcp-Name`los valores se decodifican exactamente
- encabezado y cuerpo desajuste devuelve el código moderno desajuste
- se validan la versión de respuesta, la identificación, la exclusividad del resultado o el error, la forma del error y el mapeo HTTP
- se aplican los requisitos de la lista de herramientas, tareas y carga útil de finalización específicos del método
- Cada observación .`HeaderMismatch`requiere un HTTP 400 JSON-RPC real `-32020`respuesta
- en bruto`Mcp-Name`el espacio blanco es rechazado mientras que los viajes de ida y vuelta del espacio blanco con código sentinela exacto
- un desaparecido`resultType`sólo es válido en la era de legado seleccionada
- los campos aditivos sobreviven a la validación en bruto mientras que los tipos de resultados desconocidos fallan
- los tipos de resultados de extensión requieren su capacidad anunciada
- los errores modernos reconocidos nunca causan retrocesos legales
- las notificaciones no producen respuesta JSON-RPC
- Se distingue la eliminación de la contabilidad de SDK y la pérdida de campo semántico
- se detecta el colapso del error de proxy y las credenciales se eliminan recursivamente en camelCase y variantes de separador
- La promoción requiere una transcripción no vacía, SDK, proxy y pruebas operacionales saludables.
- La promoción y el retorno requieren un objetivo de retorno autenticado, fijado, activo y saludable.

## Modos de falla de producción

| Failure | What the weak test reports | What the harness must prove |
|---|---|---|
| SDK synthesizes a missing discriminator | “tools/list passed” | Raw modern result lacked `resultType` and is invalid |
| Client downgrades after `-32021` | “legacy retry worked” | Recognized modern error forbids fallback |
| Unknown result type treated as complete | “response parsed” | Unadvertised lifecycle discriminator is rejected |
| Proxy authorizes one tool and origin executes another | “request reached server” | `Mcp-Name` equals the body routing name at every hop |
| Harness throws before reading the server response | “header mismatch test passed” | HTTP 400 and JSON-RPC `-32020` response are captured and validated |
| Proxy turns origin 400 into generic 500 | “upstream error” | Origin and egress statuses and JSON-RPC bodies are preserved |
| Notification middleware emits `{result: null}` | “handler returned none” | Final egress body is empty and no JSON-RPC response exists |
| SDK strips an additive field | “typed objects match” | Raw and normalized views show the exact dropped field |
| Failure artifact leaks a bearer token | “debug bundle uploaded” | Redaction occurred before hashing, logging, or upload |
| Credential key style bypasses redaction | “denylist contains api_key” | CamelCase and separator variants share one canonical denylist form |
| Canary has no samples but appears healthy | “zero errors” | Minimum sample count is enforced |
| Rollback selects an unknown build | “previous deployment restored” | Target version, admission digest, pins, status, and health are present |

## Regla de funcionamiento

Prueba los bytes que envías, los bytes de cada intermediario, la semántica que exponen cada SDK y las operaciones de prueba que utilizarán bajo presión. La compatibilidad es una rama explícita.

## Leer más

- [MCP 2026-07-28 base protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP version negotiation](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Official MCP conformance project](https://github.com/modelcontextprotocol/conformance)
