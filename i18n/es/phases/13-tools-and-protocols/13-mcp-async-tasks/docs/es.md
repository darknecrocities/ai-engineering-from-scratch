# Extensión de las tareas del MCP: trabajo duradero en un núcleo sin estatus

> El MCP sin estatus no significa que cada operación debe terminar en una sola solicitud. La extensión oficial de tareas da a los trabajos de larga duración un mango duradero explícito. Un servidor puede devolver ese mango desde `tools/call`, cualquier instancia puede responder .`tasks/get`, y la entrada del cliente llega a través de `tasks/update`Sin reanudar las sesiones de protocolo.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 11 (stateless MRTR), Phase 13 · 12 (elicitation)
**Time:** ~90 minutes

## Objetivos de aprendizaje

- Distinguir el transporte de protocolos sin estado de origen del estado de tareas de aplicación duraderas.
- Negociar el acuerdo`io.modelcontextprotocol/tasks`extensión de las capacidades por solicitud y `server/discover`¿ Qué ?
- Regresa un mensaje dirigido al servidor`CreateTaskResult`con`resultType: "task"`sólo después de una creación duradera.
- Encuesta con `tasks/get`, cumple las tareas introducidas con `tasks/update`, y solicitar la cancelación de la cooperación con `tasks/cancel`¿ Qué ?
- Retira el viejo .`tasks/status`¿ Qué ?`tasks/result`, y `tasks/list`las suposiciones.
- Suscribirse a las notificaciones opcionales de tareas a través de `subscriptions/listen`en una corriente de respuesta POST SSE.
- Expirado de tarea modelo, reinicio de recuperación, deduplicación de la clave de entrada y errores de ejecución correctamente.

## Por qué las tareas son una extensión

Las tareas aparecieron por primera vez como una característica central experimental en 2025-11-25.`io.modelcontextprotocol/tasks`extensión para que los clientes y servidores puedan optar por el ciclo de vida extra sin expandir el protocolo principal para todos.

La especificación de extensión sigue siendo un borrador de superficie a pesar de que es el hogar oficial actual de tareas. Enlazar la versión de extensión compatible con su SDK, ejecutar escenarios de conformidad y aislar los adaptadores de cable de su dominio de trabajo y almacenamiento.

Utilice una tarea cuando la operación tenga una o más de estas propiedades:

- Puede sobrevivir a un tiempo de espera normal.
- Una cola de trabajadores o un sistema de trabajo externo ya posee la ejecución.
- El cliente necesita recuperarse después de su propio reinicio.
- La operación se detiene para la entrada del usuario o modelo durante la ejecución.
- La cancelación y la recuperación de resultados duraderos son requisitos del producto.

No crea una tarea para una búsqueda determinista barata. Un manejo, persistencia, votación, vencimiento y cancelación son una complejidad real.

## El núcleo de los apátridas, la aplicación estatal

MCP 2026-07-28 se elimina `initialize`¿ Qué ?`notifications/initialized`, sesiones de protocolo, y `Mcp-Session-Id`Eso no prohíbe los productos de estado.

Una identificación de tarea es el estado de aplicación explícito:

- El servidor lo insiste antes de devolverlo.
- El cliente puede almacenarlo y volver a hacer una encuesta después de reiniciar.
- La identificación puede ser enviada a cualquier réplica respaldada por la misma tienda duradera.
- La autorización se verifica en cada método de tarea.
- La expiración y eliminación se definen por campos de tareas, no por un período de vida del transporte.

Esto es operativamente diferente del estado oculto conectado a una conexión.

Mantenga cuatro vidas separadas:

| State | Lifetime | Where it belongs |
|---|---|---|
| Protocol metadata | One request | `params._meta`, validated again on every call |
| Transport work | One stdio request or HTTP response | In-flight coordinator with a bounded deadline |
| MRTR continuation | One retry sequence | Integrity-protected `requestState`, plus replay controls when needed |
| Durable task | Across requests, replicas, restarts, and reconnects | Shared application store keyed by an authorized `taskId` |

El traslado de un registro de tareas a la memoria de proceso no hace que MCP sea estado.`tasks/get`Persiste antes de devolver el mango, luego haga que cada método de tarea resuelva el mismo registro compartido bajo los controles del inquilino y del principal.

## Negociación de la capacidad

El cliente anuncia apoyo en cada solicitud elegible:

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "extensions": {
        "io.modelcontextprotocol/tasks": {}
      }
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "lesson-client",
      "version": "1.0.0"
    }
  }
}
```

El servidor devuelve exactamente `supportedVersions`, capacidades,`ttlMs`, y `cacheScope`de la`server/discover`La aplicación de las tecnologías de la información y de la información es una de las principales modalidades de aplicación de las tecnologías de la información y de la información.`tools/list`Ese resultado devuelve un determinista`generate_report`descriptor, objeto válido `inputSchema`¿ Qué ?`resultType: "complete"`, metadatos de identidad del servidor, y pistas de caché público.

Un método de tarea de un cliente que no declaró las devoluciones de extensión `-32021`, Falta de capacidad requerida para el cliente, con `data.requiredCapabilities`se fija en `{"extensions":{"io.modelcontextprotocol/tasks":{}}}`Una cadena de protocolo no soportada devuelve .`-32022`con exactitud`supported`y `requested`datos; una versión que no está disponible o no está en cadena se devuelve `-32602`¿ Qué ?

Un sobre sin un JSON-RPC `id`El receptor puede procesarlo, pero no emite ningún resultado o error JSON-RPC.`202 Accepted`sin organismo para una notificación aceptada.

En la actualidad, sólo`tools/call`soporta la ejecución de tareas aumentadas. Diseñe su abstracción interna para que los tipos de solicitudes futuros no requieran reescribir almacenamiento.

## Creación de tareas dirigidas por servidores

La vieja bandera del cliente .`params._meta.task.required`El cliente declara soporte de extensión, luego el servidor decide si un determinado `tools/call`se convierte en una tarea.

Solicitud:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "generate_report",
    "arguments": {"size": "large"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Respuesta:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "task",
    "taskId": "tsk_786512e29e0d",
    "status": "working",
    "statusMessage": "Preparing report outline.",
    "createdAt": "2026-08-21T10:30:00Z",
    "lastUpdatedAt": "2026-08-21T10:30:00Z",
    "ttlMs": 900000,
    "pollIntervalMs": 1000
  }
}
```

El servidor no debe devolver este mango hasta que un `tasks/get`En una tienda de almacenamiento consistente, espere a la visibilidad de lectura antes de responder. de lo contrario un cliente puede recibir un ID de aspecto válido y recibir inmediatamente "no se encuentra".

Una respuesta a la tarea no se solicita en el sentido de que el cliente no solicita el modo de tarea.

## La forma de la tarea

Cada tarea lleva consigo:

- `taskId`: identificador estable generado por el servidor;
- `status`¿ Qué es esto ?`working`¿ Qué ?`input_required`¿ Qué ?`completed`¿ Qué ?`cancelled`, o`failed`El artículo 1
- `createdAt`y `lastUpdatedAt`: sello de tiempo ISO 8601;
- `ttlMs`: duración de caducidad desde la creación, o `null`sin límite anunciado;
- opcional `pollIntervalMs`: la cadencia mínima de encuestas sugerida del servidor actual;
- opcional `statusMessage`: contexto orientado al usuario o al modelo.

Los campos específicos de estado aparecen sólo cuando sean relevantes:

- `input_required`incluye `inputRequests`¿ Qué ?
- `completed`incluye la solicitud original `result`¿Qué forma tiene?
- `failed`incluye un JSON-RPC `error`Objeto.

El cliente debe honrar .`pollIntervalMs`Un servidor puede limitar las tasas de encuestas más agresivas y puede cambiar el intervalo durante la vida de la tarea.

## Encuesta con `tasks/get`

El cliente pide una instantánea actual:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/get
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tasks/get",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

`tasks/get`El resultado siempre ha sido el mismo.`resultType: "complete"`La tarea anida todavía puede tener`status: "working"`o `status: "input_required"`¿ Qué ?

Esta distinción evita un error común del parser:

```text
result.resultType = complete    means the tasks/get RPC finished
result.status = working        means the represented job is still running
```

No hay ninguna .`tasks/result`Cuando la tarea se complete, el siguiente `tasks/get`La respuesta se enmarca en el original `CallToolResult`en el`result`¿Qué es esto ?

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "completed",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:34:12Z",
  "ttlMs": 900000,
  "result": {
    "resultType": "complete",
    "content": [
      {"type": "text", "text": "Generated large report with approved outline."}
    ],
    "structuredContent": {"size": "large", "approved": true},
    "isError": false,
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "tasks-demo",
        "version": "1.0.0"
      }
    }
  },
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "tasks-demo",
      "version": "1.0.0"
    }
  }
}
```

El exterior .`resultType`dice el `tasks/get`El RPC completado.`result.resultType`dice que la llamada de herramienta original se completó. ese discriminador anidado es necesario.`CallToolResult`También debe llevar su propio `io.modelcontextprotocol/serverInfo`Esta lección incluye en lugar de almacenar una carga útil no tipificada.

No hay ninguna .`tasks/list`Los servidores sin sesión no pueden inferir con seguridad qué tareas pertenecen a una lista escaneada por conexión. Las aplicaciones que necesitan historial deben exponer una herramienta de dominio autorizado con filtros explícitos y reglas de propiedad.

## Ingreso durante la ejecución de tareas

La entrada de tarea y el MRTR central se parecen, pero utilizan continuidades diferentes.

### Entrada necesaria antes de crear tareas

El núcleo de retorno `resultType: "input_required"`del original `tools/call`El cliente lo cumple y vuelve a intentar la llamada original.

### Entrada necesaria después de la creación de tareas

Establezca la tarea para `input_required`- ¿ Qué ?`tasks/get`expone lo sobresaliente `inputRequests`, y el cliente envía respuestas a través de `tasks/update`El cliente no vuelve a intentar el original .`tools/call`¿ Qué ?

Instantánea:

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "input_required",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:31:00Z",
  "ttlMs": 900000,
  "inputRequests": {
    "approve_outline": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Approve the generated report outline?",
        "requestedSchema": {
          "type": "object",
          "properties": {"approved": {"type": "boolean"}},
          "required": ["approved"]
        }
      }
    }
  }
}
```

Actualización:

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/update
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tasks/update",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "inputResponses": {
      "approve_outline": {
        "action": "accept",
        "content": {"approved": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

La respuesta de éxito es un reconocimiento vacío más `resultType: "complete"`El cambio de estado puede ser consistente, así que el cliente continúa haciendo encuestas o escuchando.

Cada uno .`inputRequests`La clave debe ser única durante toda la vida de la tarea.`tasks/get`Las imágenes instantáneas pueden mostrar la misma clave pendiente; los clientes deduplican la interfaz de usuario y los servidores ignoran las respuestas de las claves desconocidas, reemplazadas o ya cumplidas.`input_required`hasta que se contesten todas las llaves requeridas.

## La cancelación es cooperativa

`tasks/cancel`El trabajo puede terminar primero, ignorar la cancelación o la transición más tarde.

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/cancel
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tasks/cancel",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Para los tres métodos de tarea,`Mcp-Name`espejos`params.taskId`. No repite el nombre del método JSON-RPC. `code/main.py`centraliza esta regla en `make_http_request`¿ Qué ?

El trabajador de la lección honra la cancelación inmediatamente, haciendo llamadas repetidas idempotentes.

No se utilice `notifications/cancelled`La notificación pertenece a la solicitud de cancelación, no a tareas duraderas.

La distinción es importante en el límite de enrutamiento. La cancelación de solicitudes se dirige a una operación JSON-RPC en vuelo o a su respuesta HTTP escaneada por la solicitud.`tools/call`Ya ha regresado .`resultType: "task"`, que la solicitud está completa y que el cierre de su transporte no puede nombrar o detener el trabajo duradero. `tasks/cancel`El nuevo RPC está autorizado.`params.taskId`, refleja esa identidad en`Mcp-Name`, resuelve el backend de la tarea, registra la intención de cancelación de la cooperativa y devuelve un reconocimiento sin afirmar que el trabajador ha parado.

Por lo tanto, una puerta de entrada debe mantener los coordinadores de las solicitudes y las rutas de tareas en diferentes tablas. La tabla de las solicitudes puede desaparecer cuando finalice la respuesta. La ruta de tareas debe sobrevivir hasta que expire el estado terminal y la retención. [Lesson 29: MCP Reliability, Cancellation, and Flow Control](../../29-mcp-reliability-cancellation-and-flow-control/docs/en.md)construye la carrera, el tiempo de espera, la impotencia, la presión y retomar las reglas para ambos caminos.

## Notificaciones opcionales

Las encuestas son la línea de base. Un cliente que quiere actualizaciones push envía`subscriptions/listen`Con ID de tarea. Para Streamable HTTP, este es un POST cuya respuesta es un flujo de SSE escaneado por solicitud. No hay un flujo de eventos GET independiente y ninguna sesión de protocolo para mantenerlo vivo.

El servidor reconoce las identidades aceptadas con `notifications/subscriptions/acknowledged`y luego puede enviar instantáneas completas a través de `notifications/tasks`. El reconocimiento y cada notificación de tarea`io.modelcontextprotocol/subscriptionId`En el`_meta`, igual al `subscriptions/listen`Cada notificación de tarea es equivalente a lo que `tasks/get`regresaría en ese momento.

Los clientes deben declarar la extensión de tareas. Deben volver a conectarse y reanudar desde ID de tareas duraderas en lugar de depender de la repetición de eventos o `Last-Event-ID`¿ Qué ?

## Semántica del fracaso

Utilice las dos capas de error correctamente.

### Error de protocolo

Parámetros de método inválidos o un id de tarea desconocido devuelven un error JSON-RPC, comúnmente `-32602`. Falta de declaraciones de apoyo a la extensión `-32021`con el objeto de capacidad requerido.

### Resultados de la ejecución de tareas

- Un resultado normal de la herramienta con `isError: true`es todavía un `completed`la tarea porque la llamada de herramienta produjo su resultado definido.
- Un error JSON-RPC durante la ejecución diferida hace la tarea `failed`y almacena ese error JSON-RPC en `error`¿ Qué ?
- El rechazo del usuario puede producir `cancelled`, un resultado de rechazo completado, u otro resultado seguro específico del dominio.

## Durabilidad, vencimiento y propiedad

Persiste al menos el ID de tarea, estado, timestamps, ttl, intervalo de encuestas, propiedad original de la operación, resultado o error, solicitudes de entrada pendientes y todas las entradas emitidas.

La clave de almacenamiento debe incluir o resolver un inquilino y un principal autorizado.`tasks/get`¿ Qué ?`tasks/update`¿ Qué ?`tasks/cancel`, y suscripción.

`ttlMs`El cliente puede tratarlo como un respaldo cuando una tarea ha dejado de producir actualizaciones observables. Un servidor puede fallar y luego eliminar una tarea expirada. No lo describa como una promesa de conservar un resultado completado durante tantos milisegundos después de la finalización.

El curso escribe un archivo temporal y renombre automáticamente. Un servicio de réplica múltiple debe utilizar una tienda duradera compartida y un contrato de arrendamiento de trabajadores o control de concurrencia equivalente.

```figure
tp-task-lifecycle
```

## Construye el mismo

`code/main.py`Implementa un servicio de tareas deterministas:

- `server/discover`retorno `supportedVersions`, las pistas de caché, y la extensión de tareas.
- `tools/list`devuelve un determinista, cachéable `generate_report`Descriptor con un esquema de entrada válido.
- `tools/call`crea y persiste la tarea antes de regresar `resultType: "task"`¿ Qué ?
- Una nueva instancia de servicio recarga la misma tarea, demostrando la recuperación de reinicio.
- `tasks/get`devuelve instantáneas completas de tareas.
- El trabajador se mueve de `working`¿ Qué ?`input_required`¿ Qué ?
- `tasks/update`acepta una respuesta en el formulario y devuelve un reconocimiento completo vacío.
- El trabajador almacena un nido`CallToolResult`con su propio `resultType`y la identidad del servidor, luego las transiciones a `completed`¿ Qué ?
- `tasks/cancel`El Consejo de Ministros de la Unión Europea ha adoptado una decisión en el marco de la cual se ha adoptado una decisión.
- Los conjuntos de constructor HTTP `Mcp-Name`¿ Qué ?`params.taskId`por`tasks/get`¿ Qué ?`tasks/update`, y `tasks/cancel`¿ Qué ?
- Los asistentes de notificación utilizan `notifications/subscriptions/acknowledged`y `notifications/tasks`, ambos etiquetados con la solicitud de escucha ID.
- Las notificaciones sin ID no producen respuesta JSON-RPC.

El trabajador avanza explícitamente en lugar de dormir en un hilo de fondo. Eso hace que cada transición de estado sea determinista y mantiene el ejemplo de protocolo separado de la mecánica de cola.

## Usalo

Desde la raíz del repositorio:

```bash
cd phases/13-tools-and-protocols/13-mcp-async-tasks/code
python3 main.py
python3 -m unittest discover tests -v
```

Secuencia de resultados esperada:

```text
id=0 resultType=complete status=ack
id=1 resultType=task status=working
id=2 resultType=complete status=working
id=3 resultType=complete status=input_required
id=4 resultType=complete status=ack
id=5 resultType=complete status=completed
```

También verifique que`tasks/status`¿ Qué ?`tasks/result`, y `tasks/list`método de devolución no encontrado en el servicio moderno.
Verifique eso .`tools/list`es determinista y cada método de tarea HTTP actual refleja su ID de tarea a través de `Mcp-Name`¿ Qué ?

## Envío

`outputs/skill-task-store-designer.md`Ahora produce un diseño consciente de la extensión: negociación de capacidad, creación duradera antes de regreso, métodos actuales, flujo de actualización de entrada, propiedad, vencimiento, cancelación, suscripción y migración de los métodos experimentales eliminados.

## Los ejercicios

1. Añadir una segunda clave de entrada pendiente. Envía una parcial `tasks/update`y demostrar que la tarea sigue .`input_required`Hasta que se respondan las dos llaves.
2. Añadir la propiedad del inquilino a la tienda y rechazar un ID de tarea válido presentado por el principal autenticado incorrecto.
3. Añadir un contrato de arrendamiento de trabajadores con vencimiento. Demostrar que dos instancias de servicio no pueden completar la misma tarea simultáneamente.
4. Implementar un adaptador SSE de respuesta POST para `subscriptions/listen`No agregue GET,`Last-Event-ID`, o un encabezado de sesión.
5. Añadir limpieza de vencimiento. Distinguir una tarea vencida de una identificación de tarea malformada sin filtración de existencia entre los inquilinos.

## Términos clave

| Term | Meaning in the current extension |
|------|----------------------------------|
| Tasks extension | Optional `io.modelcontextprotocol/tasks` capability for durable async work |
| `CreateTaskResult` | Server-directed `resultType: "task"` response to an eligible request |
| `tasks/get` | Poll a full current task snapshot, including terminal result or pending input |
| `tasks/update` | Submit responses to a task's outstanding `inputRequests` |
| `tasks/cancel` | Acknowledge cooperative cancellation intent |
| `input_required` | Task status indicating client input is outstanding |
| `pollIntervalMs` | Server-suggested minimum delay before another poll |
| `ttlMs` | Expiry duration measured from task creation |
| Durable-before-return | Rule that the task id must resolve before its handle is sent |
| `notifications/tasks` | Optional full task snapshot delivered on a subscribed SSE response |

## Compatibilidad con el legado

La superficie experimental 2025-11-25 utilizó el aumento de tareas solicitadas por el cliente, `tasks/status`¿ Qué ?`tasks/result`, y opcionales `tasks/list`Un cliente actual utiliza la capacidad de extensión, acepta manipulaciones dirigidas por servidores, encuestas `tasks/get`, suministra entrada con `tasks/update`, y lee el resultado final de la foto de tarea.

## Leer más

- [Official MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
