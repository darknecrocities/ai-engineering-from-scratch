# Aplicaciones de MCP sobre el Protocolo de apátridas

> Un resultado interactivo es todavía una herramienta MCP y el intercambio de recursos. El núcleo 2026-07-28 hace que ese intercambio sea autónomo, mientras que la extensión Apps añade la superficie del navegador sandboxed.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources)
**Time:** ~75 minutes

## Objetivos de aprendizaje

- Publicidad de las aplicaciones de MCP a través de `server/discover`y capacidades de extensión por solicitud.
- Declarar una`ui://`recurso en una herramienta antes de que la herramienta sea llamada.
- Regresa los resultados completos de la herramienta y los recursos en el cable sin estado de 2026-07-28.
- Separar las aplicaciones `ui/initialize`mensaje de puente desde el apretón de manos del núcleo de MCP eliminado.
- Aplicar la validación de origen, sandboxing, CSP y permisos de menor privilegio.

## El problema

Un resultado de texto puede describir una línea de tiempo. No puede dar al usuario una línea de tiempo que pueda filtrar, inspeccionar o actuar sobre.

MCP Apps resuelve el problema de presentación con una extensión opcional.`ui://`El host puede recoger y revisar ese recurso antes de que la herramienta se ejecute, renderizarlo en un iframe sandboxed y mediar todas las acciones de la aplicación a través de un puente JSON-RPC.

El protocolo principal cambió en 2026-07-28. No envuelva una aplicación en el viejo ciclo de vida de la conexión:

- No hay núcleo .`initialize`solicitud o `notifications/initialized`notificación.
- No hay ninguna .`Mcp-Session-Id`¿Qué es eso?
- Cada solicitud lleva la versión del protocolo y las capacidades del cliente en `params._meta`¿ Qué ?
- Un servidor implementa `server/discover`para que los clientes puedan inspeccionar versiones, capacidades centrales y extensiones.
- Cada resultado exitoso tiene un resultado .`resultType`- ¿Qué es eso?
- El HTTP transmitible utiliza un POST por solicitud. Los puntos de entrada modernos GET y DELETE devuelven 405.

El puente de Apps todavía tiene un método llamado `ui/initialize`Pertenece al dialecto de iframe postMessage. No recrea una sesión central de MCP.

## El concepto

### Dos protocolos, una característica

Mantenga las capas explícitas:

1. El núcleo del MCP lleva `server/discover`¿ Qué ?`tools/list`¿ Qué ?`tools/call`¿ Qué ?`resources/list`, y `resources/read`¿ Qué ?
2. La extensión de MCP Apps declara la interfaz de usuario y define el puente iframe-host.
3. Las reglas de la caja de arena del navegador limitan lo que la interfaz de usuario puede alcanzar.

El identificador de extensión es `io.modelcontextprotocol/ui`. Ambos pares optan. Un cliente envía soporte de extensión dentro del objeto de capacidades en cada solicitud:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "server/discover",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/ui": {}
        }
      },
      "io.modelcontextprotocol/clientInfo": {
        "name": "timeline-host",
        "version": "1.0.0"
      }
    }
  }
}
```

`clientInfo`Se trata de datos auto-reportados, no de una identidad de autorización.

### Descubrir antes de hacer la traducción

El resultado de descubrimiento del servidor anuncia la extensión:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {},
    "resources": {},
    "extensions": {
      "io.modelcontextprotocol/ui": {}
    }
  },
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "timeline-app-server",
      "version": "2.0.0"
    }
  }
}
```

El servidor debe soportar el descubrimiento. Un cliente no se ve obligado a llamar al descubrimiento antes de cada acción porque cada acción lleva sus propias capacidades.

### Declarar la interfaz de usuario en la definición de herramienta

El contrato de aplicaciones modernas une una interfaz de usuario a la herramienta en `tools/list`¿Qué es esto ?

```json
{
  "name": "notes_timeline",
  "description": "Render a timeline of notes.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  },
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline.html"
    }
  }
}
```

Este es un proceso de pre-llamada de metadatos. El host puede precargar, almacenar en caché y revisar la seguridad del HTML antes de que un resultado le pida que lo muestre.`_meta.ui.resourceUri`forma.

`tools/list`Es caché en el núcleo actual. Incluye orden determinístico,`ttlMs`, y `cacheScope`- Usar .`private`cuando las herramientas visibles varían según el usuario o el token.

### Regresa los datos, luego deja que el host vincule la vista

La llamada de herramienta devuelve contenido ordinario más datos estructurados:

```json
{
  "resultType": "complete",
  "content": [
    {"type": "text", "text": "Timeline ready."}
  ],
  "structuredContent": {
    "notes": [
      {"id": "note-1", "title": "Discover", "created": "2026-07-28"}
    ]
  },
  "isError": false
}
```

El host ya sabe qué vista pertenece a la herramienta. Evite inventar un nuevo bloque de contenido solo para repetir el URI.

### Sirve la aplicación como recurso

El servidor anuncia `resources`En el descubrimiento, por lo que también implementa el obligatorio`resources/list`La entrada de la lista determinista incluye el URI canónico, un nombre estable, descripción y tipo MIME.`resultType`, metadatos de identidad del servidor, `ttlMs`, y `cacheScope`, como la lista de herramientas deterministas.

El anfitrión envía`resources/read`En Streamable HTTP, la solicitud tiene:

```text
POST /mcp
MCP-Protocol-Version: 2026-07-28
Mcp-Method: resources/read
Mcp-Name: ui://notes/timeline.html
```

Los valores de encabezado y el cuerpo JSON-RPC deben coincidir.`-32020`¿ Qué ?

El resultado contiene el recurso HTML y las sugerencias de caché:

```json
{
  "resultType": "complete",
  "contents": [
    {
      "uri": "ui://notes/timeline.html",
      "mimeType": "text/html;profile=mcp-app",
      "text": "<!doctype html>...",
      "_meta": {
        "ui": {
          "csp": {
            "connectDomains": [],
            "resourceDomains": [],
            "frameDomains": [],
            "baseUriDomains": []
          },
          "permissions": {}
        }
      }
    }
  ],
  "ttlMs": 60000,
  "cacheScope": "public"
}
```

### Cachar los recursos de la interfaz de usuario como contenido ejecutable

Un recurso de aplicación no es intercambiable con la prosa ordinaria. Su entrada en caché puede ejecutar código puente, renderizar datos de herramienta y solicitar acciones mediadas por el host.`ui://`URI, identidad y versión admitidas del servidor, digestión de contenido de recursos y contexto de autorización cuando `cacheScope`Nunca vuelva a utilizar un recurso privado de la aplicación en todos los principales porque el HTML o sus metadatos de política pueden diferir incluso cuando el URI es idéntico.

Invalida la entrada cuando su `ttlMs`expira, la herramienta es `_meta.ui.resourceUri`cambios de enlace, cambios en la versión del servidor o en el pin de descriptor admitido, o una suscripción reconocida para cambiar recursos nombra el URI. Recargar y volver a aplicar la revisión de CSP y permisos antes de volver a instalar. Un iframe obsoleto no debe mantener permisos más amplios simplemente porque una nueva versión de recurso aún no se ha cargado.

### Rechazar la ambigüedad del cable antes de la política de características

La validación tiene un orden deliberado. primero valida la forma JSON-RPC y requiere metadatos de protocolo de cadena más un mapa de capacidad del cliente objeto. Luego compara los encabezados de enrutamiento con el cuerpo. Sólo entonces decide si se admite la versión del protocolo correspondiente. Este orden impide que un proxy y el servidor interpreten diferentes solicitudes.

| Condition | HTTP | JSON-RPC error |
|-----------|------|----------------|
| Header and body version, method, or name disagree | 400 | `-32020` |
| Header and body agree on an unsupported version | 400 | `-32022`, with `data` exactly `{"supported":["2026-07-28"],"requested":"<actual>"}` |
| `resources/read` lacks the Apps extension capability | 400 | `-32021`, with `data.requiredCapabilities.extensions.io.modelcontextprotocol/ui` |
| Method is unknown | 404 | `-32601` |

Una notificación JSON-RPC no tiene `id`, por lo que el servidor nunca emite una respuesta JSON-RPC para él. Una notificación HTTP aceptada devuelve 202 con un cuerpo vacío. Un error puede cambiar el estado HTTP, pero aún no puede crear un cuerpo de error JSON-RPC para una notificación.

### La caja de arena es un límite, no un veredicto de confianza

Un host controla el iframe. La aplicación no puede leer directamente las cookies del host, el almacenamiento local o la página DOM. Todos los trabajos privilegiados deben cruzar el puente.

Utilice estos valores predeterminados:

- Deja todas las listas de dominios de CSP vacías, luego añade sólo los orígenes que la aplicación necesita.`connectDomains`para extraer, XHR y WebSocket; uso `resourceDomains`para scripts, estilos, imágenes y fuentes.
- En el caso práctico, agrupar código y datos.
- No solicite permiso de cámara, micrófono o ubicación a menos que una característica visible lo necesite.
- Pinar `postMessage`a la exactitud del origen de los pares y rechazar los eventos de cualquier otro origen.
- Trate los argumentos de las herramientas, los resultados de las herramientas, el texto de los recursos y los mensajes de puente como entradas no confiables.
- Mantenga el consentimiento del usuario en el host.

No copiar un fijo `sandbox`El host debe elegir banderas basadas en el modelo de origen de la aplicación y su propio diseño de aislamiento.

Un dominio permitido sigue siendo un camino de exfiltración.`connectDomains: ["https://api.example.com"]`significa que cualquier guión que se ejecuta dentro de la aplicación puede enviar datos permitidos allí. La coincidencia exacta del origen evita la confusión del destino, pero no decide si la carga útil es adecuada. Mantenga el acceso de conexión vacío por defecto, evite colocar fichas portadoras en el iframe, operaciones de estrecha representación a través del host cuando sea práctico, limite los tamaños de respuesta y solicitud y controle qué acción del usuario causó cada solicitud saliente. Tratar`resourceDomains`separadamente de `connectDomains`; el permiso para cargar una fuente o un guión no debe permitir cargar datos arbitrarios.

### El puente de Apps tiene su propio ciclo de vida

El puente de aplicaciones es un dialecto JSON-RPC sobre `postMessage`Puede intercambiarse .`ui/initialize`y `ui/*`Notificaciones y puede proxy métodos de aspecto de núcleo tales como `tools/call`¿ Qué ?

La vista envía `ui/initialize`con`appInfo`y un `appCapabilities`Objeto. El host devuelve sus capacidades y contexto del host. Sólo después de esa respuesta el View envía `ui/notifications/initialized`El anfitrión debe esperar esta notificación de aplicaciones antes de enviar mensajes a la vista.

Ese apretón de manos local crea un puente entre un iframe y un host frame. No negocia la versión del protocolo MCP, no crea estado del servidor ni crea una sesión de transporte.`notifications/initialized`fue eliminado, mientras que las aplicaciones `ui/notifications/initialized`Una solicitud central generada por una llamada de herramienta puenteada es una nueva solicitud autónoma con un nuevo ID JSON-RPC y metadatos completos de la solicitud.

### Contexto del anfitrión, acciones y revocación

El host sigue siendo la autoridad después de la inicialización del puente. Una vista puede solicitar una acción de herramienta, navegación, uso de clipboard u otro efecto privilegiado solo a través de una capacidad que el host anunció. El host valida la solicitud tipada, el usuario actual, el objetivo y los argumentos, aplica la política de aprobación y puede rechazarla. Un clic de botón y un mensaje de puente válido expresan la intención; ninguno otorga autoridad.

Trate el tema, el tamaño y la accesibilidad como contextos de cambio en lugar de entradas de renderización de una sola vez:

- Aplique los tokens de color y tipografía proporcionados por el host, y luego reaccione cuando cambie la preferencia de tema o contraste.
- Deja que la vista informe las dimensiones deseadas, pero deja que el anfitrión cubra y aplique el tamaño de iframe para que el contenido no pueda escapar de su diseño o crear superposiciones engañosas.
- Preserva el orden del teclado, el enfoque visible, los nombres accesibles, el estado del lector de pantalla, el contraste suficiente, el zoom y el comportamiento de movimiento reducido dentro del iframe.
- Re-test transferencia de enfoque entre los controles de host y Control de vista después de cambiar de tamaño y volver a renderizar.

Las capacidades pueden ser revocadas mientras la aplicación esté abierta porque el usuario cambia de cuenta, cambia de política, un servidor está en cuarentena o el host restringe el consentimiento.`ui/initialize`. En caso de revocación, rechazar las llamadas privilegiadas pendientes, detener la actividad de red que ya no se ajusta a las políticas, eliminar el estado de renderización sensible y volver a montar o volver a escribir cuando el recurso de la interfaz de usuario mismo ya no sea admitido.

### El retroceso es parte del contrato.

Un servidor consciente de las aplicaciones todavía puede servir a los hosts que no anuncian la extensión de la interfaz de usuario:

- Regresa la misma herramienta sin `_meta.ui`En el`tools/list`¿ Qué ?
- Mantenga un resultado de texto útil para `tools/call`¿ Qué ?
- Rechazar`resources/read`para la interfaz de usuario con un error de capacidad faltante.
- Nunca asuma que exista un iframe al decidir si la herramienta está completa.

```figure
t3-ui-sandbox
```

## Construye el mismo

`code/main.py`Construye un pequeño modelo de protocolo en proceso sin un SDK. Valida el envase de solicitud actual y los valores de enrutamiento HTTP Streamable, anuncia Apps a través de `server/discover`, enumera herramientas y recursos, ejecuta la herramienta y sirve un recurso HTML independiente.

El modelo recibe cuerpos ya analizados y encabezados de enrutamiento.`Content-Type`o `Accept`. Utilice la Lección 09 para el adaptador HTTP de transmisión completo que requiere `Content-Type: application/json`y un `Accept`valor que contiene ambos `application/json`y `text/event-stream`¿ Qué ?

- ¿Qué quieres decir ?

```bash
cd phases/13-tools-and-protocols/14-mcp-apps
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Inspeccionar cuatro cosas en la salida:

1. Cada llamada es independiente.
2. Cada solicitud tiene`_meta`las capacidades.
3. `resources/list`devuelve un descriptor estable antes de leer cualquier recurso.
4. Cada resultado tiene`resultType`y metadatos de identidad del servidor.
5. No aparece ningún identificador de sesión principal.

## Usalo

Comience con`server/discover`- Confirmar .`io.modelcontextprotocol/ui`aparece en el mapa de extensiones del servidor. Luego llame `tools/list`La primera respuesta declara el recurso. La segunda sigue siendo una herramienta de uso sólo de texto.

Leer .`ui://notes/timeline.html`Busca en el HTML para`hostOrigin`y el `event.origin`Esas dos líneas son la prueba mínima visible de que el puente no utiliza un blanco de tarjeta salvaje.

## Envío

Esta lección nos lleva .`outputs/skill-mcp-apps-spec.md`. Utilice para revisar un contrato de aplicación antes de escribir código marco. Se obliga al autor a indicar el envase principal actual, la negociación de extensiones, fallback, recurso de UI, política de caché, CSP, permisos, métodos de puente y límite de consentimiento.

## Los ejercicios

1. Cambia la capacidad del cliente a un mapa de extensión vacío.`tools/list`mantiene la herramienta pero elimina la unión de la interfaz de usuario.
2. Envía`Mcp-Name: ui://notes/other.html`Con un cuerpo que lee la línea de tiempo.`-32020`¿ Qué ?
3. Cambiar el recurso a `cacheScope: private`Describir la condición específica del usuario que la justifica.
4. Mueve el guión a `https://static.example.com/app.js`Añadir ese origen a `resourceDomains`y explicar el nuevo riesgo de la cadena de suministro.
5. Añadir un`notes_open`la herramienta y el botón de ruta haga clic en el host. Mantenga la aprobación del usuario en el host.

## Términos clave

| Term | Meaning |
|------|---------|
| MCP Apps | Optional extension for interactive HTML rendered by an MCP host |
| `io.modelcontextprotocol/ui` | Extension identifier advertised by both peers |
| `ui://` | Resource scheme for an App's UI template |
| `text/html;profile=mcp-app` | MIME type for MCP App HTML |
| `server/discover` | Current RPC for protocol and capability discovery |
| `resources/list` | Mandatory resource listing method when the server advertises resources |
| `resultType` | Required discriminator for modern successful results |
| `ui/initialize` | First Apps bridge request, separate from removed core initialization |
| `ui/notifications/initialized` | Apps View readiness notification sent after the host responds |
| CSP | Browser policy that restricts scripts, styles, images, and network origins |
| Text fallback | Tool behavior retained for a host without Apps support |

## Leer más

- [MCP 2026-07-28 base protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Apps overview](https://modelcontextprotocol.io/extensions/apps/overview)
- [MCP Apps build guide](https://modelcontextprotocol.io/extensions/apps/build)
- [Official extension support matrix](https://modelcontextprotocol.io/extensions/client-matrix)
