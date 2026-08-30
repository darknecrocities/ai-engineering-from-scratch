# Transporte de MCP: estudio y streaming HTTP sin estado

> El transporte transporta mensajes MCP. No proporciona el estado del protocolo faltante.`2026-07-28`, local stdio y remoto Streamable HTTP ambos llevan solicitudes auto-describir.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 07 and 08
**Time:** ~65 minutes

## Objetivos de aprendizaje

- Elija el estudio para procesos infantiles locales y Streamable HTTP para servicios de red.
- Implementar el moderno contrato HTTP Streamable de un solo punto final, solo POST.
- Reflejar y validar las versiones, métodos y encabezados de nombres de MCP en relación con el cuerpo JSON-RPC.
- Entrega de SSE a escala de solicitud y de larga duración `subscriptions/listen`flujo correctamente.
- Migra las implementaciones HTTP+SSE basadas en sesiones y heredadas sin presentar el comportamiento heredado como moderno.

## El problema

Anteriormente, las revisiones de HTTP Streamable combinaban la negociación de protocolos con el comportamiento de conexión y sesión.`Mcp-Session-Id`, exponer una corriente de GET independiente, aceptar DELETE para la terminación de la sesión y reanudar la SSE con `Last-Event-ID`¿ Qué ?

MCP `2026-07-28`El servidor de HTTP puede validar esos encabezados contra el cuerpo antes de la ejecución.

El resultado es más fácil de escalar y más fácil de razonar. También significa que un servidor que enseña el transporte 2025 como actual está enseñando el modelo de falla y seguridad equivocado.

## El concepto

### estudio

La unión de estudio es para un subproceso lanzado por el cliente:

- El cliente escribe un mensaje UTF-8 JSON-RPC por línea a stdin.
- El servidor escribe un mensaje UTF-8 JSON-RPC por línea a stdout.
- El servidor escribe diagnósticos a Stderr.
- El servidor sale rápidamente en el EOF.
- Cada solicitud moderna lleva versiones y capacidades de cliente en `params._meta`¿ Qué ?

El proceso puede funcionar para muchas llamadas, pero no es una sesión de protocolo moderno. Si sale inesperadamente, las solicitudes en vuelo se pierden. Reiniciar el proceso, redescubrir, volver a listar, volver a abrir suscripciones y volver a intentar operaciones seguras con nuevos ID de solicitud.

### HTTP en 2026-07-28

Un servidor moderno expone un punto final de MCP, como `/mcp`, que acepta POST.

Cada solicitud o notificación JSON-RPC es un nuevo POST HTTP. El cuerpo contiene un mensaje JSON-RPC. Los clientes no envían respuestas JSON-RPC al servidor.

Para una solicitud, el servidor devuelve:

- `Content-Type: application/json`con una respuesta JSON-RPC; o
- `Content-Type: text/event-stream`con notificaciones relacionadas con dicha solicitud, seguidas de la respuesta final JSON-RPC.

Para una notificación aceptada, el servidor devuelve `202 Accepted`Sin cuerpo.

Los clientes anuncian ambos tipos de respuesta:

```http
Accept: application/json, text/event-stream
```

### Solo POST significa sólo POST

El HTTP de transmisión moderna no tiene flujo de GET independiente y no tiene punto final de sesión DELETE.

- `GET /mcp`retorno `405 Method Not Allowed`¿ Qué ?
- `DELETE /mcp`retorno `405 Method Not Allowed`¿ Qué ?
- `Mcp-Session-Id`se ignora y nunca se acuña o se hace eco.
- `Last-Event-ID`se ignora porque las corrientes modernas no son reanulables.

Si una secuencia de solicitudes se rompe antes de su respuesta final, el cliente ha perdido esa solicitud en vuelo. Puede emitir una nueva solicitud con un nuevo ID JSON-RPC cuando la retoma sea segura. No debe intentar la reanudación de la secuencia.

### Validación del origen

Los servidores validan `Origin`En caso de que el encabezado esté presente y no esté explícitamente permitido, devuelva `403 Forbidden`Un cliente no navegador puede omitir`Origin`, lo que las normas oficiales de transporte permiten.

Los servidores locales deben vincularse a `127.0.0.1`Los servicios de red todavía necesitan autenticación y autorización en cada solicitud.

Utilice la coincidencia exacta de origen después de la configuración canónica.`origin.startswith("https://trusted.example")`son inseguras porque pueden aceptar sufijos controlados por el atacante.

### Título de metadatos HTTP requerido

Cada solicitud POST moderna incluye:

```http
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes_search
```

Reglas de encabezado:

- `MCP-Protocol-Version`es requerida y debe ser igual `params._meta.io.modelcontextprotocol/protocolVersion`¿ Qué ?
- `Mcp-Method`se requiere y debe ser igual al JSON-RPC `method`¿ Qué ?
- `Mcp-Name`es necesario para `tools/call`¿ Qué ?`resources/read`, y `prompts/get`¿ Qué ?
- `Mcp-Name`igual `params.name`, o`params.uri`por`resources/read`¿ Qué ?
- Los valores de encabezado son sensibles a los casos aunque los nombres de encabezado son insensibles a los casos.

No seguro o no ASCII `Mcp-Name`Los valores utilizan el sentinela exacto UTF-8 Base64:

```text
=?base64?{Base64EncodedValue}?=
```

El servidor decodifica ese valor antes de compararlo con el cuerpo.

Los encabezados espejos que faltan, están malformados o no coinciden devuelven HTTP `400`con código JSON-RPC `-32020`. Si el encabezado y el cuerpo coinciden en una versión que el servidor no admite, devuelva HTTP `400`con`-32022`y datos de error exactos como `{"supported":["2026-07-28"],"requested":"2027-01-01"}`¿ Qué ?

Un método moderno desconocido devuelve HTTP `404`con JSON-RPC `-32601`. El cuerpo JSON-RPC es importante porque un cliente de doble era lo utiliza para distinguir un error moderno de un error de punto final heredado.

### ETS con escala de solicitud

Un servidor puede elegir SSE para una solicitud de larga duración:

```text
POST tools/call id=41
  <- notifications/progress related to id=41
  <- notifications/progress related to id=41
  <- JSON-RPC response id=41
stream closes
```

El servidor no debe enviar solicitudes independientes de JSON-RPC en esta corriente. Muestras, elicitación e interacciones de raíces utilizan resultados de Requestas de viaje múltiple. Cerrar la corriente de respuesta cancela esa solicitud.

No agregue ID de evento SSE para reproducir. `Last-Event-ID`La reanudación no forma parte de la revisión moderna.

### Cambios de larga duración con suscripciones/audición

Las notificaciones de cambio utilizan una solicitud abierta por el cliente, no una GET independiente:

```json
{
  "jsonrpc": "2.0",
  "id": "listen-1",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true,
      "resourceSubscriptions": ["notes://note-1"]
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    }
  }
}
```

La respuesta POST es una corriente SSE de larga duración.`notifications/subscriptions/acknowledged`. El reconocimiento, cada notificación de cambio y el resultado final llevan `io.modelcontextprotocol/subscriptionId`En el`_meta`El servidor puede emitir comentarios de SSE como mantenientes. Cuando la corriente baja, el cliente reemite `subscriptions/listen`con una nueva identificación de solicitud y reajusta los datos afectados.

`resources/subscribe`y `resources/unsubscribe`No los uses en una conexión moderna.

### Estado de aplicación explícita

La eliminación de sesiones de protocolo no prohíbe los flujos de trabajo con estado. El servidor puede acuñar un mango de estado opaco y devolverlo como resultado de herramienta normal. El cliente pasa ese mango como un argumento explícito en llamadas posteriores.

Enlace las manijas al principal autenticado, haga que no sean probables, expire y autorice todo uso. Esto hace que el estado sea visible en la capa de aplicación en lugar de ocultarlo en la afinidad del transporte.

El fallo causado por el estado de réplica oculta es mecánico:

1. La solicitud A llega a la réplica 1 y crea un borrador en la memoria de ese proceso.
2. La respuesta no devuelve un proyecto de manipulación porque la implementación asume que la conexión identifica el proyecto.
3. La solicitud B es un nuevo POST y llega a la réplica 2.
4. Replica 2 tiene metadatos de protocolo válidos pero no hay forma de nombrar o cargar el borrador, por lo que el flujo de trabajo falla o lee el objeto local incorrecto.
5. El enrutamiento pegajoso parece arreglar el síntoma hasta que una reiniciación, despliegue, reprogramar o fallar sobre el siguiente pedido.

El límite correcto tiene dos partes. El contexto del protocolo se mantiene en cada solicitud. El estado de aplicación duradera vive en una tienda compartida bajo un mango memorizado por el servidor devuelto al cliente. La siguiente llamada suministra el manejo, cualquier réplica carga el mismo registro, y la autorización une el registro al principal y al inquilino autenticados. La memoria de réplica puede almacenar un registro en caché, pero no puede ser la única copia requerida para la corrección.

Seleccione el mecanismo de estado por vida. Las variables locales de solicitud pueden servir una llamada. Una continuación corta de MRTR puede usar integridad protegida `requestState`Una tarea de proyecto o duradera necesita un manejo explícito más persistencia compartida, vencimiento, control de concurrencia e idempotencia. ninguno de esos objetos es una sesión de protocolo MCP.

### Compatibilidad con la doble era HTTP

Un cliente que admite servidores modernos y antiguos intenta primero un POST moderno. Si recibe HTTP `400`¿ Qué ?`404`, o`405`, inspeccionará el cuerpo:

- Un error moderno reconocido JSON-RPC prueba que el servidor es moderno. Correcta la solicitud o vuelva a probar una versión anunciada. No rebajes el nivel.
- Un cuerpo vacío o una respuesta no reconocida puede indicar un servidor HTTP+SSE heredado. Sólo entonces prueba el antiguo punto final GET y espera su legado `endpoint`El evento.

Un servidor puede soportar ambas épocas durante la migración enrutando metadatos modernos a la implementación moderna de solo POST y manteniendo puntos finales heredados separados para clientes antiguos. Nunca describa el comportamiento heredado GET, DELETE, session id o replay como parte de `2026-07-28`¿ Qué ?

```figure
tp-transport-handshake
```

## Usalo

`code/main.py`Implementa un servidor HTTP Streamable finito y moderno con la biblioteca estándar Python. Valida los encabezados Origin y Mirrored, ignora los encabezados de sesión eliminados, devuelve JSON para llamadas normales y demuestra un límite `subscriptions/listen`El flujo de SSE.

```bash
cd code
python3 main.py --probe
python3 -m unittest discover tests -v
```

La sonda comprueba:

- se rechazará el origen inválido;
- el descubrimiento tiene éxito sin un ID de sesión;
- `Mcp-Session-Id`y `Last-Event-ID`se ignoran;
- los resultados de las discrepancias de encabezado `-32020`El artículo 1
- versiones no compatibles `-32022`con exactitud`supported`y `requested`datos;
- una notificación aceptada sin id devuelve HTTP `202`sin cuerpo;
- Obtener y borrar el retorno `405`El artículo 1
- `subscriptions/listen`es un flujo de respuesta POST cuyo reconocimiento, notificación y resultado final llevan su identificación de suscripción.

## Envío

Esta lección nos lleva .`outputs/skill-mcp-transport-migrator.md`. Elimina las sesiones de protocolo modernas, agrega validación de cabezas y cuerpo, sustituye GET independiente por `subscriptions/listen`, y mantiene cualquier puente heredado visiblemente separado.

## Los ejercicios

1. Retirada`Mcp-Method`de una POST. Confirmar HTTP `400`y error `-32020`¿ Qué ?
2. Envía la versión de encabezado y cuerpo correspondiente `2027-01-01`Confirmar el HTTP`400`, error `-32022`, y datos exactos `{"supported":["2026-07-28"],"requested":"2027-01-01"}`¿ Qué ?
3. Envía un centinela Base64 .`Mcp-Name`Para un URI de recurso no ASCII. Confirme que el valor decodificado se compara con `params.uri`¿ Qué ?
4. Rompe la transmisión de escucha finita antes de su respuesta final, reediciéndola con un nuevo ID JSON-RPC y herramientas de reafirmación.
5. Añadir un manejo explícito del flujo de trabajo a la herramienta ping. Atarlo a un sujeto de autorización sin usar afinidad de conexión.

## Términos clave

| Term | Meaning |
|------|---------|
| stdio | Newline-delimited JSON-RPC over a client-launched subprocess |
| Streamable HTTP | Single endpoint where each modern message is a new POST |
| Request-scoped SSE | POST response stream containing related notifications and final response |
| `subscriptions/listen` | Long-lived POST request for opted-in change notifications |
| Header mismatch | HTTP `400` and JSON-RPC `-32020` when mirrored headers disagree with body |
| Origin validation | DNS-rebinding defense for incoming connections, not authentication |
| Explicit state handle | Application token passed as an ordinary argument instead of hidden session state |
| Legacy bridge | Separate earlier-era behavior kept only for compatibility |

## Leer más

- [MCP Transport Overview](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
