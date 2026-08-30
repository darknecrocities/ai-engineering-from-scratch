# Entrada de las puertas de acceso de los MCP sin estatus y registro

> Un gateway debe hacer explícita cada ruta. El protocolo 2026-07-28 le da método, nombre, versión, capacidad, identidad, caché y límites de rastreo sin una sesión de transporte.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13 · 15 (security), Phase 13 · 16 (authorization)
**Time:** ~75 minutes

## Objetivos de aprendizaje

- Agrega varios servidores MCP detrás de un punto final 2026-07-28 sin afinidad de sesión.
- Valida los metadatos y los encabezados de enrutamiento por solicitud antes de la política o del reenvío.
- Fusión de herramientas con espacios de nombres estables, orden determinístico, pines descriptor, RBAC y caché privado.
- Trate los registros como evidencia de descubrimiento que aún requiere política de admisión.
- SSE con escala de ruta según la solicitud, `subscriptions/listen`, MRTR retestes, y las llamadas de extensión de tareas correctamente.
- Isolar el apoyo de la mano y la sesión de la herencia del camino moderno.

## El problema

Conectar un cliente directamente a un servidor es simple. Una implementación más grande necesita una respuesta consistente a preguntas más difíciles:

- ¿Qué servidores están permitidos?
- ¿Qué director puede ver y llamar a cada herramienta?
- ¿Qué sucede cuando dos personas de fondo exponen el mismo nombre?
- ¿Cómo se revisan los cambios en el descriptor?
- ¿Dónde se aplican los límites de tipos y los eventos de auditoría?
- ¿Puede alguna instancia manejar la siguiente solicitud?

Una puerta de enlace se encuentra entre los clientes y los servidores de MCP de fondo. Presenta un punto final de MCP, aplica políticas transversales y reenvía solicitudes aprobadas.

Los diseños de pasarelas más antiguos a menudo multiplican una sesión de cliente en varias sesiones de backend y se reescriben `Mcp-Session-Id`El núcleo 2026-07-28 no tiene sesiones de protocolo.

## El concepto

### El camino de la puerta moderna

Para cada solicitud:

1. Identifique el principal de la autorización de transporte.
2. Validación`MCP-Protocol-Version`¿ Qué ?`Mcp-Method`¿ Qué ?`Mcp-Name`, y `params._meta`¿ Qué ?
3. Autorice el principal, el recurso, el método, la herramienta y los argumentos.
4. Aplicar la política de descripción, registro, tarifa y datos.
5. Crear una nueva solicitud independiente para el backend seleccionado.
6. Valida el resultado de backend y devuelve un resultado de la puerta de entrada.
7. Graba un evento de auditoría sin registrar secretos.

Ningún paso necesita una sesión de protocolo oculto. El estado de aplicación puede existir en bases de datos, manuales explícitos, tareas o estado MRTR protegido por integridad.

### La política de tiempo de ejecución es la decisión principal de la puerta de entrada

La admisión decide qué versión de backend puede entrar en la puerta de enlace. No autoriza una llamada en vivo. Para cada solicitud, la puerta de enlace recalcula la política desde el principal autenticado, emisor y recurso, inquilino, método y nombre coincidentes, argumentos normalizados, pin de descripción admitido, salud de backend actual, intersección de capacidades, clasificación de datos, estado de tasa y cualquier aprobación vinculada a la acción.

Esto importa. Un registro de registro puede permanecer activo mientras se revoca el papel de un usuario. Un descriptor puede permanecer fijado mientras un argumento de destino cruza un límite de inquilino. Un backend puede permanecer aprobado mientras la política de cuarentena incidente cambia de estado llamadas. La política de tiempo de ejecución es por lo tanto la decisión principal de permitir o negar, con registro y evidencia de descriptor como entradas.

No almacenar en caché una decisión de permitir en una conexión o en un identificador de sesión eliminado. Si la política no está disponible, siga una política de falla declarada por clase de operación. Un defecto seguro es que no se cierre para cambios de estado y lecturas sensibles, mientras que las vías de lectura pública explícitamente aprobadas pueden utilizar una política de última hora de corta duración solo cuando su modelo de riesgo lo permita. Registre qué versión de política y ruta de falla tomó la decisión, luego valida el resultado de retorno antes de devolverlo.

### Un punto final de POST

El HTTP de transmisión moderna envía cada mensaje JSON-RPC a través de POST:

```text
POST /mcp
Authorization: Bearer <gateway-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.search
Accept: application/json, text/event-stream
```

La puerta de enlace puede devolver JSON o SSE escaneado por solicitud para ese POST. GET y DELETE devuelven 405 para solicitudes modernas. `Mcp-Session-Id`y `Last-Event-ID`no crear autoridad, afinidad o comportamiento repetición.

Los valores de encabezado y cuerpo deben estar de acuerdo.`-32020`Esto permite que los balanceadores de carga, pasarelas y limitadores de velocidad se dirigen sin analizar todo el cuerpo mientras se conserva la integridad de extremo a extremo.

Valida en un orden exacto: tipos de JSON-RPC y metadatos, igualdad de encabezado y cuerpo, luego soporte para la versión coincidente. Una incompatibilidad devuelve HTTP 400 con `-32020`Si el encabezado y el cuerpo coinciden en una versión no compatible, devuelva HTTP 400 con `-32022`y `data`Es exactamente .`{"supported":["2026-07-28"],"requested":"<actual>"}`Un método desconocido devuelve HTTP 404 con `-32601`¿ Qué ?

`ProtocolError`lleva opcional `data`, y la puerta de enlace la serializa en el objeto de error JSON-RPC.`id`Una notificación HTTP aceptada devuelve 202 con un cuerpo vacío.

### Implementar el descubrimiento en cada capa

La puerta de entrada se ejecuta `server/discover`También descubre cada backend para que conozca las versiones de protocolo, capacidades y extensiones.

Ejemplo de resultado de la puerta de entrada:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": true}
  },
  "ttlMs": 30000,
  "cacheScope": "private",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "enterprise-gateway",
      "version": "2.0.0"
    }
  }
}
```

Publicidad sólo la intersección de capacidades que la puerta de entrada puede honrar de extremo a extremo. Una función de backend no es automáticamente segura de exponer. Una función de puerta de entrada sin camino de backend no es útil para la publicidad.

`serverInfo`No se utilice como prueba de registro o de editor.

### Capacidades de cliente por solicitud

Cada solicitud enviada necesita una actualización `_meta`Envase:

```json
{
  "io.modelcontextprotocol/protocolVersion": "2026-07-28",
  "io.modelcontextprotocol/clientCapabilities": {},
  "io.modelcontextprotocol/clientInfo": {
    "name": "enterprise-gateway",
    "version": "1.0.0"
  }
}
```

No copiar ciegamente las capacidades del cliente externo a un backend. La puerta de entrada es el cliente del backend.

### Espacio de nombres determinista

Fusión de herramientas de backend bajo nombres públicos estables:

```text
notes.search
notes.create
issues.list
issues.open
```

Mantenga un mapa desde el nombre público hasta el nombre de la herramienta original y de la parte posterior. Nunca elija la primera o última colisión. Un nombre público es parte del contrato de aprobación y auditoría, por lo que cambiarlo es una migración.

`tools/list`Cuando la visibilidad difiere por principal, el rendimiento`cacheScope: private`- Un límite .`ttlMs`reduce la carga de descubrimiento de backend sin permitir que una lista específica del usuario se filtrara en contextos de autorización.

Cada descriptor de herramientas expuesto incluye un nombre estable, descripción y raíz de objeto `inputSchema`. El espacio de nombres no puede eliminar los campos de descriptor requeridos. El resultado de la lista completa también incluye `resultType`, metadatos de identidad del servidor, y pistas de caché.

### Descriptores aprobados con pin

En el momento de la admisión, canonice el descriptor completo y almacene su digesto bajo el nombre público calificado.

Si cambia:

- Retirarlo de `tools/list`¿ Qué ?
- Rechaza las llamadas directas.
- Emite un evento de auditoría.
- Requerir una nueva aprobación política o humana antes de actualizar el pin.

Una puerta de entrada es un punto de aplicación central útil, pero no convierte un descriptor de primera vista en uno seguro.

### Los registros ayudan a descubrir, no a decidir

Un registro `server.json`El registro respaldado por paquetes puede verse así:

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "com.example/notes",
  "description": "Example notes MCP server.",
  "version": "1.0.0",
  "packages": [
    {
      "registryType": "npm",
      "identifier": "@example/notes-mcp",
      "version": "1.0.0",
      "transport": {"type": "stdio"}
    }
  ]
}
```

Los metadatos de publicación no contienen la decisión de seguridad de la puerta de entrada.

```json
{
  "registryName": "com.example/notes",
  "registryVersion": "1.0.0",
  "publisher": {"namespace": "com.example", "status": "verified"},
  "provenance": {
    "source": "registry.modelcontextprotocol.io",
    "recordId": "com.example/notes@1.0.0"
  },
  "admission": {"status": "approved", "reviewedBy": "gateway-policy"}
}
```

La puerta de entrada verifica el `server.json`La puerta de entrada todavía necesita una política de admisión.

Para cada final admitido, registren:

- Registro y registro exacto.
- Espacio de nombres de editores verificado o evidencia de dominio.
- Se permite el transporte y el punto final.
- Versión fija o política de actualización aprobada.
- Digestión de artefactos o descriptores.
- Emitente de autorización y recurso.
- Revisor, tiempo de aprobación y caducidad.

No acepte un servidor porque su nombre de visualización se asemeja a un producto familiar. No trate la presencia del registro como una revisión de seguridad operativa. Los servidores privados pueden ser admitidos a través del mismo esquema de pruebas incluso cuando nunca aparecen en un registro público.

Esta lección implementa la costura de la puerta de entrada: unir la evidencia de publicación a la admisión local antes de que un backend se convierta en en rutable. [Lesson 30: MCP Registry Supply Chain, Admission, Drift, and Rollback](../../30-mcp-registry-supply-chain-and-drift/docs/en.md)construye el plano de control completo para la prueba exacta del espacio de nombres, la procedencia de los artefactos, los pines inmutables, la deriva del descriptor en vivo, la reconciliación del estado del Registro, un libro mayor de admisiones que sea evidente y el retroceso respaldado por pruebas. Mantenga ese estado de la cadena de suministro separado de la decisión de ejecución por solicitud anterior.

### Medición de credenciales

La puerta de entrada autentica a sus llamadas y se autentica por separado a los backends.

Mantenga explícitos estos vínculos:

```text
outer principal -> gateway role and policy
backend issuer + resource -> backend registration and token
```

Nunca pases el token de puerta de entrada externa a un backend. Nunca vuelvas a usar un token de backend en un emisor o recurso diferente. Si una herramienta actúa en nombre de un usuario final, preserva esa delegación con un modelo de intercambio o reclamaciones diseñado en lugar de imitar al usuario con una credencial de servicio compartida.

### Límites de tarifas sin sesiones

Los límites clave por principal autenticado, emisor, recurso, herramienta pública, clase de costo y ventana de tiempo.

Aplique una validación barata antes de consumir un trabajo costoso.

### Auditoría de la cadena de decisiones

Graba lo suficiente para reconstruir una llamada:

- Identificadores de solicitud y rastreo.
- Principio y emisor autenticados.
- Herramienta pública y ruta de backend.
- Versión de pin de descriptor.
- La decisión política y la razón.
- La clase de latencia y resultados.
- Identificador de ronda o tarea MRTR, cuando proceda.

Tokens de portador de redacción, códigos de autorización, tokens de actualización, secretos crudos y argumentos sensibles innecesarios.

### ETS con escala de solicitud

Un POST normal puede devolver el SSE escaneado por la solicitud cuando se ejecuta una sola solicitud.

No cree una corriente de GET separada y no prometa reproducción de último evento-ID.

### Notificaciones de cambios de larga duración

Para notificaciones de cambio de lista y recursos, un cliente actual envía `subscriptions/listen`Los filtros de notificación utilizan los campos planos exactos `toolsListChanged`¿ Qué ?`promptsListChanged`¿ Qué ?`resourcesListChanged`, y `resourceSubscriptions`¿Qué es esto ?

```json
{
  "jsonrpc": "2.0",
  "id": "listen-tools",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

El primer evento reconoce el subconjunto soportado. Su identificador de suscripción es el ID JSON-RPC de la solicitud que abrió la transmisión:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": "listen-tools"
    },
    "notifications": {
      "toolsListChanged": true
    }
  }
}
```

La puerta de entrada envía entonces sólo los tipos de cambios reconocidos.`io.modelcontextprotocol/subscriptionId`En el`params._meta`. No hay reproducción automática o re-audición automática. En reconectar, el cliente vuelve a abrir la suscripción y actualiza las listas en las que se basa. Un cerramiento gracioso iniciado por el servidor devuelve un resultado completo final etiquetado con la misma identificación de suscripción.

El camino moderno reemplaza`resources/subscribe`¿ Qué ?`resources/unsubscribe`, y sin solicitar, GET streaming independiente. Mantén sólo en una versión de puerta de ruta más antigua.

### MRTR a través de una puerta de entrada

Cuando un backend regresa `resultType: input_required`, la puerta de entrada sólo puede reenviar ese resultado si el cliente externo admite la solicitud de entrada necesaria.`requestState`byte por byte a menos que la puerta de entrada termine deliberadamente y reemite la interacción.

El cliente vuelve a probar la herramienta pública original con un ID JSON-RPC nuevo y `inputResponses`La puerta de entrada reautorizará la reutilización, comprobará la misma ruta pública y enviará una nueva solicitud de backend.

### tareas de enrutamiento de extensión

Las tareas son una extensión oficial identificada por `io.modelcontextprotocol/tasks`No son un reemplazo de sesión principal.

El cliente declara la extensión dentro de las capacidades del cliente por solicitud, y la puerta de entrada lo anuncia en descubrimiento solo cuando puede preservar el ciclo de vida de extremo a extremo.`tools/call`, el backend solo decide si devuelve el resultado ordinario o `resultType: task`. Un resultado de la tarea lleva `taskId`¿ Qué ?`status`, sello de tiempo,`ttlMs`, y una opción `pollIntervalMs`La tarea debe ser duraderamente legible antes de que se envíe el resultado.

La puerta de entrada registra la ruta principal y backend autenticadas para el identificador de tarea opaco.`tasks/get`¿ Qué ?`tasks/update`, y `tasks/cancel`uso de llamadas `params.taskId`¿ Cómo ?`Mcp-Name`, que da a los intermediarios una clave de enrutamiento. `tasks/get`retorno `resultType: complete`con el estado de tarea actual y enmarca el resultado final o el error de protocolo en un estado terminal. `tasks/update`Envía con llave`inputResponses`para entradas de tareas pendientes y devuelve un reconocimiento completo vacío. `tasks/cancel`Es una intención cooperativa con un reconocimiento completo vacío, no una garantía de que el trabajo se detenga.

No implementar nuevos `tasks/list`o `tasks/result`Los métodos de trabajo son los más antiguos, pero son de un modelo experimental.`tasks/get`El cliente responde a través de ellos `tasks/update`El cliente sigue votando en el intervalo sugerido; la creación de tareas sigue siendo dirigida al servidor.

El estado de ruta de tarea duradera es el dato de aplicación teclado por el manejo de tarea, no una sesión de protocolo.

### Fronteras de compatibilidad

Si la puerta de entrada debe servir a un cliente o backend más viejo:

- Detecta la época explícitamente.
- Mantenga la inicialización, las sesiones de transporte, los flujos GET, las suscripciones de recursos y el vocabulario de tareas antiguas dentro de un adaptador heredado.
- Nunca filtrar un ID de sesión heredado en el enrutamiento moderno o autorización.
- Prefiero una sonda de descubrimiento limitada y una política de retroceso explícita a la rebaja silenciosa.

```figure
t3-gateway-funnel
```

## Construye el mismo

`code/main.py`El gateway proporciona una puerta de acceso de protocolo en proceso y dos servidores backend.`tools/list`, enrutamiento con espacios de nombres, Registro `server.json`Además del estado de admisión externa, pines descriptores, RBAC, límites de tasas de la clave principal, decisiones de auditoría y un modelo `subscriptions/listen`Confirmación de la SSE.

El modelo recibe cuerpos de solicitud analizados, encabezados de enrutamiento y una identidad de portador autenticada. No es un adaptador HTTP completo y no analiza `Content-Type`o el completo `Accept`conecta con el adaptador HTTP de la Lección 09, que requiere `Content-Type: application/json`y un `Accept`valor que contiene ambos `application/json`y `text/event-stream`¿ Qué ?

- ¿Qué quieres decir ?

```bash
cd phases/13-tools-and-protocols/17-mcp-gateways-and-registries
python3 code/main.py
python3 -m unittest discover code/tests -v
```

La demostración imprime la identificación de la solicitud externa y la identificación de la solicitud de backend fresca para que el hop sin estado sea visible.

## Usalo

Replace los objetos de fondo en proceso con clientes reales de protocolo de corriente. Mantenga las mismas costuras:

- Registro de admisión antes de la conexión.
- Descubrimiento de retroceso antes de la exposición de la capacidad.
- Nombre público calificado antes de la autorización.
- Pinta de descriptor antes de lista o llamada.
- Metadatos frescos por solicitud antes de ser enviados.
- Validación del resultado antes de regresar.

## Envío

Esta lección nos lleva .`outputs/skill-gateway-bootstrap.md`Produce un diseño moderno de puerta de enlace que cubre la entrada, el descubrimiento, la admisión, los espacios de nombres, la autorización, el almacenamiento en caché, la transmisión, las suscripciones, MRTR, tareas, observabilidad y aislamiento heredado.

## Los ejercicios

1. Añadir contexto de rastreo a los metadatos externos y enviados de la solicitud y registrar la correlación en el evento de auditoría.
2. Añadir un backend y ruta con funciones`tasks/get`por tarea ID en `Mcp-Name`¿ Qué ?
3. Cambiar un descriptor de backend y demostrar que tanto el descubrimiento como la llamada directa están bloqueados.
4. Añadir una capacidad de servidor específica del principal y explicar por qué el descubrimiento debe permanecer en caché privado.
5. Escriba una interfaz de adaptador heredado sin agregar ningún estado heredado al moderno `Gateway`clase.

## Términos clave

| Term | Meaning |
|------|---------|
| MCP gateway | Policy and routing server between clients and backend MCP servers |
| Admission record | Evidence and policy decision allowing one backend into the gateway |
| Qualified tool name | Stable public route such as `notes.search` |
| Descriptor pin | Approved digest checked during discovery and dispatch |
| Private cache scope | Cached result restricted to one authorization context |
| Request-scoped SSE | Streaming response attached to one POST request |
| `subscriptions/listen` | Client-opened SSE stream for selected long-lived change notifications |
| Task route | Application mapping from an opaque task id to its backend |
| Legacy adapter | Explicit version-gated boundary for old handshake and session behavior |

## Leer más

- [Streamable HTTP transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [Official Registry server.json requirements](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
