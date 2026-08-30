# Confiabilidad, cancelación y control de flujo de los MCP

> Un ID de solicitud correlaciona un mensaje. No hace que un efecto secundario sea seguro, detenga a un trabajador o proteja un flujo de un consumidor lento.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 09 and 13
**Time:** ~120 minutes

## Objetivos de aprendizaje

- Implemente la señal de cancelación correcta para stdio y Streamable HTTP.
- Resolver carreras de finalización y cancelación sin enviar mensajes después de la cancelación.
- Cancelación de solicitud por separado de duradero `tasks/cancel`¿Qué es lo que se dice?
- Construir decisiones de retoma de efectos secundarios y claves explícitas de impotencia.
- Limitar las colas de progreso mientras se conservan las respuestas finales.
- Recuperar los flujos a través de reconectar, re-accionar y el retroceso nervioso.

## El problema

El camino feliz esconde los errores más caros de los sistemas distribuidos.

Un cliente llama a una herramienta. El servidor comienza a trabajar. El progreso llega. Un proxy amortiza la corriente. El cliente alcanza su tiempo límite y se desconecta. El servidor termina un milisegundo más tarde. El cliente vuelve a intentar con un nuevo ID JSON-RPC. La mutación se ejecuta dos veces.

Cada componente se comportó localmente, el sistema falló a nivel mundial.

MCP define el comportamiento de mensajes y transporte, pero su aplicación aún posee:

- presupuestos temporales;
- la capacidad de las empresas;
- filas de puertas;
- la clasificación de los ensayos de nuevo;
- estado de tarea duradero;
- La Comisión ha adoptado una política de reconnexión y de reajuste.

Esta lección construye esas decisiones en un simulador determinista.
No hay interrupciones, conexiones o fallos aleatorios.
Una prueba de hilo sincronizado obliga a dos clientes de contabilidad a competir
para la misma clave de impotencia.

## El pedido de cancelación es específico para el transporte

La intención es la misma en todos los transportes: el cliente ya no necesita un resultado en vuelo.

### estudio

stdio utiliza un canal bidireccional compartido. Un cliente envía una notificación:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/cancelled",
  "params": {
    "requestId": 41,
    "reason": "User closed the operation"
  }
}
```

La notificación es de fuego y olvido. El servidor no emite ninguna respuesta JSON-RPC a ella.

El servidor debe dejar de trabajar, liberar recursos y evitar enviar una respuesta para la solicitud cancelada. Puede ignorar la cancelación cuando la solicitud es desconocida, ya terminada o no se puede detener de forma segura.

Las notificaciones de cancelación mal formadas, desconocidas y ya completadas son ignoradas.

### HTTP en transmisión

El cliente cancela cerrando el flujo de respuesta de esa solicitud.

No publiques`notifications/cancelled`El cierre de la transmisión es la señal de cancelación.

Una vez que el servidor observe la desconexión, debe dejar de funcionar y no debe enviar más mensajes para esa solicitud.

### La cancelación enviada por el servidor es estrecha

Un servidor no utiliza `notifications/cancelled`En el estudio, la cancelación enviada por el servidor se reserva para terminar una llamada de un cliente.`subscriptions/listen`Mantenga ese camino separado de la cancelación ordinaria de las solicitudes del cliente.

## La cancelación es una carrera

Dos órdenes de eventos son válidas.

### La cancelación gana

```text
request starts
client sends cancellation signal
server marks request cancelled
worker reaches completion
server suppresses the response
```

### La finalización gana

```text
request starts
worker commits the result
server sends the response
cancellation arrives late
server ignores the late notification
```

El cliente también debe ignorar una respuesta tardía para una solicitud que ya abandonó. La latencia de la red significa que ninguna de las partes puede probar qué evento la otra parte observó primero.

```figure
mcp-reliability-race
```

La lección es `RequestCoordinator`almacena un estado terminal. `complete()`No puede devolver respuesta después de la cancelación.

## Las horas de espera necesitan dos relojes

Un solo temporizador de inactividad no es suficiente.

Utilice dos límites:

1. **Idle timeout.**El tiempo que la solicitud no puede producir actividad útil.
2. **Maximum timeout.**El presupuesto absoluto del reloj de pared desde el inicio de la solicitud.

El progreso puede restablecer el reloj inactivo.

```text
start: 0 ms
progress: 400 ms
progress: 800 ms
progress: 1200 ms
idle timeout: 500 ms
maximum timeout: 2000 ms
```

A 1500 ms, la solicitud sigue activa porque el último progreso es de sólo 300 ms. A 2000 ms, el plazo máximo lo anula incluso si otro suceso de progreso llega a 1999 ms.

El progreso es opcional. Un servidor puede aceptar un token de progreso y no emitir actualizaciones. Nunca convierta la presencia de un token en un tiempo infinito.

Los valores de progreso de los MCP deben aumentar. Las notificaciones se detienen después de la finalización o la cancelación.

## No es posible solicitar la cancelación`tasks/cancel`

Estos mecanismos resuelven diferentes vidas.

| Mechanism | Target | Signal | What success means |
|-----------|--------|--------|--------------------|
| Request cancellation on stdio | One in-flight RPC | `notifications/cancelled` | Client abandoned the request; server should stop if practical |
| Request cancellation on HTTP | One in-flight response stream | Close the stream | Client abandoned the request; server should stop if practical |
| `tasks/cancel` | One durable Task | Ordinary MCP request | Server acknowledged cancellation intent |

Un éxito .`tasks/cancel`El resultado no demuestra que el trabajador se haya detenido.`working`El trabajo puede concluirse antes de que el puesto de control del trabajador observe la bandera.

No borre el estado de tarea duradera cuando se cierra la conexión HTTP. La razón para crear una tarea es que su ciclo de vida sobrevive a una solicitud y una conexión.

## Un nuevo ID JSON-RPC no es idempotencia

Los ID de JSON-RPC correlacionan las solicitudes y respuestas. No identifican una operación de negocio.

Supongamos que un cliente presenta una carga con un ID `41`, pierde la respuesta, y vuelve a intentar con id `42`El servidor ve dos mensajes diferentes. sin una clave de aplicación, no puede saber que representan una sola caja.

Una clave de idempotency identifica la intención de negocio:

```json
{
  "name": "charge_account",
  "arguments": {
    "account": "acct-7",
    "cents": 1200,
    "idempotencyKey": "checkout-7"
  }
}
```

Los servidores almacenan:

- la llave;
- una huella digital de los argumentos de operación;
- el resultado prometido.

La misma clave y los mismos argumentos devuelven el resultado almacenado. La misma clave con argumentos diferentes se rechaza. Esto evita que la reutilización accidental de la clave mute una operación de negocio diferente.

### El límite del libro mayor debe ser atómico y duradero

Esta secuencia no es segura:

```text
check key
run mutation
store result
```

Dos trabajadores pueden observar una llave faltante y ejecutar la mutación.
después del efecto pero antes de que la tienda crea la misma ambigüedad en la nueva prueba.

La lección utiliza un libro mayor SQLite respaldado por archivos. `BEGIN IMMEDIATE`Se trata de un
control de clave, efecto de negocio simulado, contador de ejecución y resultado almacenado en
Dos conexiones independientes de contabilidad que compiten con la misma clave
Por lo tanto, observen un resultado comprometido y una ejecución.
El libro mayor mantiene ese registro.

Cada valor de retorno se reconstruye a partir de JSON almacenado.
el objeto mutable que contiene el libro mayor, por lo que cambiar un diccionario devuelto no puede
corrupto resultados de reproducción posterior.

El efecto de negocio del simulador es el contador de recepción y ejecución dentro del
Una llamada de pago real, implementación o API externa es
La producción de los productos de la industria de la producción de la carne de vacuno no se hace atómica simplemente escribiendo una tabla local.
transacción compartida de base de datos, caja de salida transaccional o proveedor de servicios de alta corriente
La seguridad de los sistemas de seguridad de seguridad de los sistemas de seguridad de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de los sistemas de seguridad de seguridad de los sistemas de seguridad de los sistemas de seguridad de seguridad de los sistemas de seguridad de los sistemas de seguridad de seguridad de los sistemas de seguridad de los sistemas de seguridad de seguridad de los sistemas de seguridad de los sistemas de seguridad de seguridad de los sistemas de seguridad de seguridad de los sistemas de seguridad de seguridad de los sistemas de seguridad de seguridad de los sistemas de seguridad de seguridad de los sistemas de seguridad de seguridad de los sistemas de seguridad de seguridad de los sistemas de seguridad de seguridad de seguridad de los sistemas de seguridad de seguridad de seguridad de los sistemas de seguridad de seguridad de seguridad de los sistemas de seguridad de seguridad de seguridad de seguridad de seguridad de los sistemas de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de los sistemas de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de los sistemas de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de los sistemas de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de seguridad de
Replicaciones múltiples o sobrevivir a un reinicio.

### Matriz de retraso

Clasificar los retos antes de implementarlos.

| Class | Example | Retry rule |
|------|---------|------------|
| Safe | Deterministic read with no side effect | Retry with a new JSON-RPC id after the failure boundary is understood |
| Conditional | Mutation with a durable idempotency key | Retry with the same key and identical arguments |
| Unsafe | Mutation without business deduplication | Do not retry automatically; reconcile first |

Anotadas de herramientas como `readOnlyHint`y `idempotentHint`El contrato de aplicación y la implementación del servidor deciden la seguridad de volver a probar.

## La presión es parte de lo correcto

Un productor de SSE puede generar progreso más rápido de lo que un cliente, proxy o red puede consumirlo.

Use una cola limitada y defina lo que puede perderse.

El progreso es reemplazable. Un valor de progreso posterior reemplaza a uno anterior por el mismo token. Una respuesta final JSON-RPC no es reemplazable.

El amortiguador de lecciones aplica esta política:

1. Coalesce el progreso adyacente para la misma señal.
2. Deja el progreso más antiguo cuando se alcance la capacidad.
3. Marque la corriente como necesitando una revisión autorizada.
4. Preserva la respuesta final.
5. Rechazar un estado en el que preservar la respuesta final requeriría dejar caer otra respuesta final.

Esta es una pérdida limitada con recuperación explícita.

### Puente de apoyo

Un servidor puede transmitir correctamente mientras un proxy inverso mantiene eventos en un buffer.

Para una respuesta de SSE, envíe:

```http
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

La especificación HTTP 2026 Streamable recomienda `X-Accel-Buffering: no`para que los proxies compatibles lleven eventos inmediatamente.

Para las corrientes tranquilas de larga duración, emitir periódicamente un comentario de la SSE:

```text
:
```

El cliente ignora las líneas de comentarios, los intermediarios ven el tráfico y tienen menos probabilidades de cerrar una conexión ociosa.

No restablezca el tiempo de inactividad semántica de una operación simplemente porque haya llegado un comentario de transporte.

## Reconectar significa recoger

El HTTP de transmisión moderna no admite SSE reiniciado a través de `Last-Event-ID`¿ Qué ?

Después de un`subscriptions/listen`caídas de corriente:

1. Abre una nueva solicitud de escucha con un nuevo ID JSON-RPC.
2. Restaurar el filtro de suscripción deseado.
3. Reemporte las herramientas, recursos, instrucciones o tareas afectadas de métodos autorizados.
4. Estado de aplicación deduplicado por identificadores estables.
5. No repitas una mutación insegura sólo porque su respuesta se perdió.

El plan de recuperación de la muestra establece explícitamente `sendLastEventId`para falsear y enumera recursos para reafirmar.

### Prevenir la reconexión de un rebaño

Si 10,000 clientes se reconnectan exactamente en un segundo, el servidor de recuperación falla de nuevo.

La lección calcula el jitter determinista a partir del número de identificación del cliente y el número de intento para que las pruebas sigan siendo reproducibles:

```text
attempt 0: up to 250 ms
attempt 1: up to 500 ms
attempt 2: up to 1000 ms
...
cap: 8000 ms
```

La producción puede utilizar la aleatoriedad de tiempo de ejecución o de seguridad criptográfica.

## Construye el mismo

`code/main.py`construye cinco componentes pequeños de fiabilidad.

### `RequestCoordinator`

- iniciar una solicitud en vuelo con plazos inactivos y máximos;
- emitirá notificaciones monótonas de progreso;
- Produce la señal correcta de cancelación de stdio o HTTP;
- ignora las notificaciones de cancelación inválidas;
- expone explícitamente las carreras terminales de cancelación y finalización;
- reservará la cancelación enviada por el servidor para suscripciones de estudio.

### `MutationLedger`

- demostrar que dos ID de JSON-RPC se ejecutan dos veces sin una clave de negocio;
- utiliza una transacción SQLite respaldada por archivos para la verificación de claves, efecto simulado,
  el contador de ejecución y el compromiso de resultados;
- deduplica argumentos correspondientes bajo una clave de idempotencia en diferentes
  conexiones de libros principales;
- Rechaza una clave reutilizada con diferentes argumentos;
- devuelve copias defensivas y conserva registros comprometidos a través de reabertura.

### `DurableTaskService`

- reconoce una solicitud de cancelación;
- mantiene la tarea `working`hasta un puesto de control de trabajadores;
- demuestra por qué el reconocimiento no es el estado final.

### `BoundedSseBuffer`

- se fusionan o disminuyen los progresos bajo presión;
- registros que requieren una revisión autorizada;
- nunca deja caer la respuesta final.

### Ajudantes de recuperación

- Retorno de los encabezados de las SSE seguros por proxy y de los comentarios de mantenimiento;
- crear un plan de reconexión y reajuste;
- Los retemplazos de dispersión con retroceso exponencial determinista y nerviosismo.

## Usalo

Desde la raíz del repositorio:

```bash
cd phases/13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/code
python3 main.py
python3 -m unittest discover tests -v
```

La demostración se ejecuta a ambos lados de la carrera central, ejecuta una transacción
mutación deduplicada en un libro mayor temporal respaldado por archivos, sobrecarga una
Buffer de progreso, y muestra una tarea duradera que se mueve de la cancelación reconocida
a la cancelación observada por los trabajadores.

## Laboratorio interactivo

Ejecutar cuatro órdenes de eventos sin añadir dormir.

1. Inicio de la solicitud `A`, cancelarlo, luego llamar`complete()`¿ Qué ?
2. Inicio de la solicitud `B`, completarlo, y luego entregar la cancelación.
3. Inicio de la solicitud `C`, emitir progreso antes de cada plazo ocioso, luego cruzar el plazo máximo.
4. Inicio de la solicitud `D`sobre Streamable HTTP y cierre su flujo de respuesta.

Registro para cada escenario:

- el estado de la solicitud terminal;
- si existe una respuesta final;
- la señal de cancelación colocada en el cable;
- ¿Qué evento debe ignorar el cliente?

Entonces cambie .`D`La operación es idéntica, pero la señal de cancelación debe cambiar.

## Laboratorio de práctica

Añadir un`reserve_inventory`mutación a `MutationLedger`¿ Qué ?

Requisitos:

1. La clave vincula el SKU, la cantidad, el inquilino y el nombre de la operación.
2. Un retraso con la misma clave y los mismos argumentos devuelve la primera reserva.
3. Un nuevo intento con cantidad cambiada falla sin otra reserva.
4. Una ejecución que se cometió pero perdió su respuesta puede reconciliarse por clave.
5. El resultado no registra datos secretos ni de pago.
6. El nuevo intento automático se desactiva cuando el cliente no proporciona una llave.
7. Añade una caída de suscripción simulada y revise el registro de inventario antes de decidir qué hacer a continuación.
8. Iniciar dos conexiones de libro mayor en una barrera y enviar la misma clave
   - Asegurar que se ha cometido una reserva.
9. Mutar el primer objeto de reserva devuelto.
   el resultado almacenado no cambió.
10. Cierra y vuelve a abrir el archivo de contabilidad, y luego reconcilia la reserva con la clave.

Mantenga el laboratorio honesto: si el inventario vive en otro servicio, explique si
que el servicio acepta la misma clave de idempotencia o si una caja de salida transaccional
Los puentes locales se comprometen con el efecto remoto.

## Artículo enviado

`outputs/skill-mcp-reliability-reviewer.md`Es una habilidad plana de revisión de fiabilidad. Dale una operación MCP, transporte, política de tiempo de espera, comportamiento de retraso, política de cola y plan de recuperación. devuelve una tabla de carrera, clasificación de retraso, límite de impotencia, controles de control de flujo y fichas de falla.

## Verifique el hecho

La lección está completa cuando estas declaraciones son ciertas:

- La cancelación de estudio envía `notifications/cancelled`y no recibe respuesta.
- La cancelación de HTTP en streaming cierra el flujo de solicitud y no envía POST de cancelación.
- Cancelación antes de completar suprime la respuesta final.
- El completo antes de cancelar conserva la respuesta e ignora la cancelación tardía.
- El progreso puede restablecer el tiempo de inactividad pero nunca el tiempo máximo.
- Una nueva identificación JSON-RPC solo ejecuta la mutación de nuevo.
- Una clave de idempotencia y argumentos idénticos se ejecutan una vez en una concurrencia
  carrera de dos conexiones.
- Un registro comprometido sobrevive a la reapertura y la reproducción devuelve una copia defensiva.
- Mutar un resultado devuelto no puede alterar el resultado almacenado.
- El amortiguador limitado se mantiene dentro de la capacidad y conserva la respuesta final.
- Reconnect utiliza una nueva solicitud, no envía `Last-Event-ID`, y reafirma el estado afectado.
- `tasks/cancel`El reconocimiento deja la tarea no terminal hasta que el trabajador la observe.

## Modos de falla de producción

| Failure | Observable symptom | Correct response |
|---------|--------------------|------------------|
| HTTP client POSTs cancellation notification | Server and client disagree about request lifetime | Close the request's SSE response stream |
| Server responds after accepted cancellation | Client receives an unusable late result | Stop work and suppress further messages when cancellation wins |
| Progress resets every deadline | Hung work survives forever | Keep a separate absolute maximum timeout |
| New RPC id treated as deduplication | Charge, deployment, or deletion runs twice | Add a durable application idempotency key |
| Key check and effect are separate | Concurrent workers both observe a missing key | Commit key claim, effect record, and result atomically |
| In-memory ledger used across replicas | Restart or another worker forgets prior commits | Use shared durable storage or upstream idempotency |
| Stored mutable result returned directly | Caller mutation corrupts later replays | Serialize committed results and return defensive copies |
| Key reused with changed arguments | One key aliases two business intents | Store and compare an argument fingerprint |
| Unbounded progress queue | Memory rises with a slow consumer | Coalesce and drop replaceable progress within a bound |
| Final response dropped under pressure | Client cannot know the request outcome | Reserve capacity or evict progress, never the final response |
| Proxy buffers SSE | Progress arrives in bursts or after timeout | Disable buffering and configure compatible proxy timeouts |
| `Last-Event-ID` assumed | Client resumes from state the server does not support | Reconnect with a new request and refetch |
| Every client reconnects immediately | Recovery creates another outage | Use capped exponential backoff with jitter |
| Task ack treated as final cancellation | Worker keeps running after UI says stopped | Poll the Task until a terminal status |

## Conexión de Capstone

La piedra angular del ecosistema de herramientas debe tratar la fiabilidad como evidencia ejecutable, no como un párrafo en un diagrama de arquitectura.

Requieren estos artefactos:

- una transcripción de la carrera de cancelación para cada transporte;
- una mesa de ensayo de nuevo para cada mutación expuesta;
- un registro de la clave de desempotencia y un dispositivo de desajuste;
- una transcripción simultánea de la misma clave, una verificación de reapertura y una verificación de alias de mutación;
- un resultado de sobrecarga de buffer limitado;
- cabeceras de SSE de proxy inverso y política de inactividad;
- un plan de reconexión que nombre los métodos de reajuste autorizados;
- un rastro duradero de cancelación de tareas cuando la piedra angular utiliza tareas.

Una solicitud verde en un proceso local prueba sólo el camino feliz. La piedra angular está lista para la producción cuando las respuestas perdidas, la cancelación tardía, los consumidores lentos y las rebañas reconectadas tienen resultados deterministas.

## Términos clave

| Term | Meaning |
|------|---------|
| Request cancellation | Abandonment of one in-flight MCP request |
| Cancellation race | Competition between terminal completion and cancellation events |
| Idle timeout | Limit since the last useful request activity |
| Maximum timeout | Absolute limit from request start, unaffected by progress |
| Idempotency key | Application identifier that deduplicates one business intent |
| Atomic ledger | Durable boundary that commits the key claim, effect record, and result as one unit |
| Backpressure | Control applied when producers outpace consumers |
| Progress coalescing | Replacing older progress with a newer authoritative value |
| Refetch | Reading current state again after a stream gap |
| Jitter | Deliberate variation that spreads retries across time |

## Leer más

- [MCP Cancellation](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/cancellation)
- [MCP Progress](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/progress)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Tasks Extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
