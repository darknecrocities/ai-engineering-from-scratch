# Recursos y instrucciones de MCP: Contexto direccionable para servidores sin estatus

> Las herramientas realizan operaciones. Los recursos exponen el contenido direccionable. Invita a los paquetes de plantillas de mensajes seleccionadas por el usuario. Un buen servidor MCP mantiene esos contratos separados y predecibles.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lesson 07 (Building an MCP Server), Phase 13, Lesson 09 (MCP Transports)
**Time:** ~60 minutes

## Objetivos de aprendizaje

- Elige entre herramientas, recursos y instrucciones de la intención del consumidor.
- Publicitar el recurso y la superficie de inmediato a través de obligatorios `server/discover`¿ Qué ?
- Construir un determinista `resources/list`y `prompts/list`Los resultados.
- Aplicar`ttlMs`y `cacheScope`sin filtración de datos específicos del usuario.
- Regresa el error JSON-RPC `-32602`para una URI de recurso inválida o desconocida.
- Abre una`subscriptions/listen`POST-respuesta de transmisión y correlacionar cada evento por ID de suscripción.
- Trate el contenido de los recursos y las plantillas de solicitud como salida de servidor no confiable.

## Comience con el consumidor

La forma más fácil de usar mal MCP es comenzar con el código de implementación. Una consulta de base de datos se convierte en una herramienta porque las funciones son familiares. Un flujo de trabajo reutilizable se convierte en un recurso porque se almacena en un archivo. Un prompt se convierte en política oculta porque el host puede inyectarlo.

Comience con quién elige y qué espera.

| Primitive | Primary intent | Selection owner | Typical result |
|---|---|---|---|
| Tool | Perform an operation | Model or application | Structured action result |
| Resource | Read content at a URI | Host, application, or user | Text or binary content |
| Prompt | Start a reusable message workflow | User through host UI | One or more prompt messages |

Una nota en `notes://note-1`es un recurso porque es un contenido direccionable. `delete_note`Es una herramienta porque cambia el estado.`review_note`es una solicitud porque un usuario elige un flujo de trabajo de revisión preparado.

No exponga una operación como las tres sólo para parecer completa. Cada superficie adicional necesita descubrimiento, autorización, almacenamiento en caché, manejo de errores, pruebas y documentación.

## El envase de los apátridas 2026-07-28

Esta lección tiene como objetivo la revisión del protocolo de MCP `2026-07-28`. No hay una sesión de apretón de mano de inicialización o protocolo en este perfil.`_meta`¿Qué es eso?

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Un servidor debe implementar `server/discover`. Sus resultados publicitarios apoyados
las versiones, las capacidades de recursos y de ejecución rápida, la identidad de la implementación, y
Un cliente puede llamar a otro método directamente, pero el descubrimiento le da
una instantánea estable antes de que construya una interfaz de usuario.

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "resources": {"listChanged": true, "subscribe": true},
    "prompts": {"listChanged": true}
  },
  "ttlMs": 3600000,
  "cacheScope": "public"
}
```

Un resultado normal se declara .`"resultType": "complete"`La respuesta .`_meta`identifica la ejecución de servicio con `io.modelcontextprotocol/serverInfo`. Esta información es útil para el diagnóstico. No es una identidad de autenticación. Una solicitud que contiene una revisión no respaldada devuelve `-32022`con la revisión solicitada y las revisiones soportadas del servidor.

El contrato sin estado cambia sus instintos de diseño. Una lista no puede depender de una llamada previa en una conexión. La autorización puede cambiar el conjunto visible porque las credenciales son la entrada de solicitud, pero el historial de conexión no debe.

## Los recursos son contratos estables de URI

Un recurso es el contenido identificado por un URI. Diseña el URI antes del manipulador.

Las propiedades de las URI son buenas:

- Lo suficientemente estable como para marcar o pasar entre solicitudes.
- Espacio de nombres en el dominio del servidor.
- Independiente de una identificación de proceso o conexión.
- Validado antes del acceso al almacenamiento.
- Autorizado en cada lectura.

`notes://note-1`es mejor que `note-1`porque su espacio de nombres es explícito. Un servidor de archivos puede usar `file://`URI, pero aún debe comprobar los límites de los directorios configurados después de resolver los vínculos simbólicos y los segmentos relativos.

`resources/list`Retorna los recursos actualmente visibles para el llamador. Se clasifica por una clave estable como URI. El orden determinístico evita que se pierdan caché ruidosos, se cambien instantáneas y se salten entre actualizaciones.

```json
{
  "resultType": "complete",
  "resources": [
    {
      "uri": "notes://note-1",
      "name": "Architecture decision",
      "description": "Why the service uses a stateless boundary",
      "mimeType": "text/markdown"
    }
  ],
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

`resources/read`devuelve uno o más elementos de contenido. Un URI desconocido no es una lectura vacía exitosa. La especificación de recursos actual asigna URI de recursos inválidos o desconocidos a parámetros inválidos JSON-RPC, código `-32602`¿ Qué ?

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "error": {
    "code": -32602,
    "message": "Unknown or invalid resource URI",
    "data": {
      "uri": "notes://missing"
    }
  }
}
```

Esta distinción permite separar la ausencia de un documento vacío válido y evita que se vuelva a buscar más ampliamente.

### Modelos de recursos

Una plantilla de recursos describe una familia de URI parámetrizados. Utilice una cuando enumere cada elemento concreto sería caro o ilimitado. Por ejemplo, `notes://projects/{project}/decisions/{decision}`le dice a un cliente cómo formar una dirección válida sin devolver cada decisión.

Una plantilla no debilita la validación. Analizar variables, aplicar autorizaciones, hacer cumplir límites de longitud y caracteres y construir consultas de almacenamiento con parámetros tipados. Nunca concatenar una cola URI arbitraria en un camino del sistema de archivos o en una declaración de base de datos.

### El contenido no es una instrucción confiable

El servidor debe limitar el tamaño del contenido, devolver un tipo MIME preciso, editar los campos a los que el llamador no puede acceder y evitar devolver registros no relacionados.

## Las instrucciones son plantillas controladas por el usuario

Las instrucciones MCP están diseñadas para la selección explícita del usuario. Un host puede renderizarlas como comandos de corte, elementos de menú o botones de flujo de trabajo. El protocolo no requiere una interfaz de usuario.

`prompts/list`Cada respuesta necesita un nombre estable, una descripción útil y declaraciones de argumento que permitan al host recopilar información antes de que se haga una solicitud.`prompts/get`¿ Qué ?

```json
{
  "resultType": "complete",
  "prompts": [
    {
      "name": "review_note",
      "title": "Review a note",
      "description": "Review one note for a named concern",
      "arguments": [
        {
          "name": "uri",
          "description": "The note resource URI",
          "required": true
        }
      ]
    }
  ],
  "ttlMs": 600000,
  "cacheScope": "public"
}
```

`prompts/get`El host decide cómo los mensajes devueltos entran en el contexto del modelo y mantiene su propia política de confianza en una mayor prioridad.

Valida los argumentos de la solicitud en el límite del servidor. Un URI de la solicitud debe pasar la misma verificación de autorización que una lectura directa de recursos. No haga de una solicitud un canal lateral alrededor del acceso a recursos.

## Las señales de caché son parte de la corrección

`ttlMs`Indica al cliente cuánto tiempo puede reutilizarse un resultado. `cacheScope`describe quién puede compartir ese valor almacenado en caché.

| Scope | Meaning | Typical use |
|---|---|---|
| `public` | May be reused across users when authorization permits | Public prompt catalog |
| `private` | Bound to the requesting user or credential context | User-owned note content |

Elige un TTL de la velocidad de cambio de los datos y el daño de la latencia. Cinco minutos pueden adaptarse a un catálogo público de solicitudes.

El MCP sólo define `public`y `private`¿ Cómo ?`cacheScope`Para obtener un resultado secreto o cambiante rápidamente, devuelva`cacheScope: "private"`con`ttlMs: 0`, luego aplicar cualquier regla más estricta de no almacenar en la política de caché del host. `no-store`no es un MCP en sí mismo `cacheScope`El valor.

Las sugerencias de caché nunca reemplazan la autorización. Una clave de caché debe incluir todas las dimensiones de la solicitud que cambien la visibilidad, incluido el cursor de inquilino, usuario, alcance, localización y paginado. Si una caché compartida no puede expresar esas dimensiones de manera segura, use `private`con un TTL cero y una política de no tienda a nivel de host.

## Suscripciones Utilice un flujo de respuesta abierto por el cliente

El patrón de suscripción moderno reemplaza al anterior `resources/subscribe`RPC y el antiguo punto final del evento HTTP GET.

El cliente envía`subscriptions/listen`En HTTP Streamable este es un POST cuya respuesta permanece abierta como un flujo SSE.`notifications`El objeto es una lista de permisos. Un servidor no debe entregar tipos de notificación que no fueron solicitados.

```json
{
  "jsonrpc": "2.0",
  "id": 17,
  "method": "subscriptions/listen",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    },
    "notifications": {
      "resourcesListChanged": true,
      "promptsListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

El ID de solicitud es el ID de suscripción. Antes de cualquier evento solicitado, el servidor envía `notifications/subscriptions/acknowledged`Su filtro contiene sólo el subconjunto que el servidor acepta.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "notifications": {
      "resourcesListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

Cada evento posterior en esa corriente lleva los mismos metadatos.

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "uri": "notes://note-1"
  }
}
```

La notificación dice que el recurso ha cambiado.`resources/read`No asume que el evento contenga el nuevo documento.

Varias suscripciones pueden compartir un canal de estudio. El ID de suscripción permite al cliente demultiplejarlas.`resultType: "complete"`la respuesta correlacionada con la solicitud original.

No utilice un flujo de suscripción como sesión de protocolo. Una lectura posterior sigue siendo una solicitud completa que puede llegar a cualquier instancia de servidor saludable.

```figure
t3-primitive-sort
```

## Laboratorio interactivo

Utilice la figura para clasificar cinco capacidades de un rastreador de proyectos: detalles de emisión, crear un problema, plantilla de revisión de sprint, política del proyecto y problema de cierre. Luego decida qué listas se pueden guardar en caché públicamente, qué lecturas deben permanecer privadas y qué recursos merecen notificaciones de actualización.

Para cada clasificación, nombre el elegidor. Si el modelo realiza una acción, use una herramienta. Si un host lee contenido con direcciones URI, use un recurso. Si el usuario inicia un flujo de trabajo de mensajes preparados, use un prompt.

## Laboratorio de práctica

Ejecutar el simulador desde la raíz de repositorio:

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

Inspectar la transcripción en este orden:

1. Confirmarlo .`server/discover`publicitará la revisión actual y ambas capacidades.
2. Confirmar que los resultados de la lista están clasificados y utilizar `resultType: "complete"`¿ Qué ?
3. Confirmar la lista y leer los resultados contienen pistas de caché intencionales.
4. Cambia la URI de lectura a `notes://missing`y observar .`-32602`¿ Qué ?
5. Confirmar el reconocimiento de suscripción precede el evento de recurso.
6. Confirmar el evento y cerrar graciosamente ambos llevar la identificación de suscripción `5`¿ Qué ?

El modelo Python no abre una conexión HTTP real. Representa los mensajes que un SDK debe colocar en el flujo de respuesta escaneado por solicitud. Utilice un SDK oficial para el enmarcado y el transporte en producción.

## Artículo enviado

`outputs/skill-primitive-splitter.md`Es una revisión de diseño reutilizable para la selección primitiva de MCP. Ahora verifica el descubrimiento determinista, el alcance de la caché, el comportamiento URI inválido y los filtros de suscripción modernos.

La lección también nos lleva .`assets/primitive-split.svg`, una versión estática del límite primitivo y de suscripción para el estudio fuera de línea.

## Verifique el hecho

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

Resultado esperado: el programa principal imprime una transcripción JSON y el comando de prueba informa de al menos doce pruebas de aprobación.

## Conexión de Capstone

Utilice este contrato cuando su servidor de capstone exponga conocimientos direccionables además de acciones. Incluya una instantánea de catálogo determinista, una lectura de recursos autorizados, una resolución rápida, un caso URI inválido y una transcripción de suscripción.

Su evidencia debe demostrar que ninguna lista depende del historial de conexión y que un evento de suscripción nunca otorga acceso al recurso subyacente.

## Los ejercicios

1. Añadir un`notes://projects/{project}/notes/{id}`la plantilla de recursos y validar ambas variables.
2. Añadir páginas a `resources/list`Mientras que se conserva el orden determinista.
3. Cambiar un recurso a `cacheScope: "private"`con`ttlMs: 0`, añadir una política de no tienda a nivel de host, y explicar la amenaza que justifica ambos controles.
4. Añadir una suscripción de cambio de lista de preguntas y demostrar que no se envía ningún evento cuando el filtro omite `promptsListChanged`¿ Qué ?
5. Crear dos suscripciones simultáneas y demostrar que cada evento lleva la ID de solicitud correcta.
6. Añadir una autorización sujeta al manejero de lectura y demostrar que una entrada de caché no puede cruzar sujetos.

## Términos clave

- **Resource:**Contenido con direcciones URI expuesto por un servidor MCP.
- **Prompt:**Una plantilla de mensaje controlada por el usuario expuesta por un servidor MCP.
- **Deterministic list:**Un resultado de descubrimiento con membresía estable y orden para las mismas entradas de solicitud.
- **`ttlMs`:**Cachar la duración de la frescura en milisegundos.
- **`cacheScope`:**El límite de compartimiento para un resultado almacenado en caché.
- **`subscriptions/listen`:**Una solicitud de larga duración cuyo flujo de respuesta proporciona notificaciones filtradas explícitamente.
- **Subscription ID:**El ID original de solicitud de escucha, repetido en metadatos de notificación.
- **Invalid parameters:**Erro JSON-RPC `-32602`, utilizado para una URI de recurso inválida o desconocida.
- **Unsupported protocol version:**Erro JSON-RPC `-32022`, incluyendo `supported`y `requested`Las revisiones.
- **`server/discover`:**Método servidor obligatorio que devuelve las revisiones, capacidades, identidad y sugerencias opcionales de caché compatibles.

## Leer más

- [MCP 2026-07-28 Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP 2026-07-28 Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP 2026-07-28 Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Caching](https://modelcontextprotocol.io/specification/2026-07-28/basic/utilities/caching)
