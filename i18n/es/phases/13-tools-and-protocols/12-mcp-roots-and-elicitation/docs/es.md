# El alcance explícito y la obtención de la licencia de apatrida

> Las raíces se deprecian en MCP 2026-07-28 y nunca fueron una caja de seguridad. Coloque alcance en argumentos de herramientas visibles o en URIs de recursos, autorízala en el servidor y use MRTR cuando una herramienta realmente necesita la entrada del usuario. El usuario ve la decisión, el modelo ve el mango y cualquier instancia del servidor puede procesar la retrayectoria.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 11 (stateless MRTR)
**Time:** ~60 minutes

## Objetivos de aprendizaje

- Reemplazar las raíces obsoletas con parámetros explícitos del espacio de trabajo, URI de recursos o configuración de servidor.
- Indicaciones de alcance separadas de la autorización, contención de ruta y sandboxing del sistema operativo.
- Modo de entrega de formulario `elicitation/create`a través de un MRTR `input_required`el resultado.
- Publicitar soporte de elicitación en las capacidades del cliente por solicitud y rechazar los modos no soportados.
- Validación`accept`¿ Qué ?`decline`, y `cancel`como resultados distintos.
- Atardecer la confirmación destructiva a un principal autenticado, argumentos originales, conjunto de candidatos y caducidad.

## Dos problemas que parecen ser similares

Una herramienta de notas recibe la siguiente solicitud: " Eliminar el antiguo informe TPS".

El servidor debe responder a dos preguntas diferentes.

1. ¿En qué espacio de trabajo puede tocar esta operación?
2. ¿A cuál de las tres notas coincidencias se refería el usuario?

La primera es el alcance y la autorización. La segunda es la desambiguación interactiva. La mezcla de ellos conduce a diseños peligrosos, como tratar una carpeta proporcionada por el cliente como prueba de que el llamador puede eliminar todo lo que hay dentro de ella.

## Las raíces son una superficie de migración

Las revisiones anteriores de MCP permitían a un cliente anunciar Roots y notificar a un servidor cuando la lista cambiaba. Roots eran una guía informativa. No restringían lo que el proceso del servidor podía leer, no autorizaban al llamador y no creaban una caja de arena del sistema operativo.

El MCP 2026-07-28 se desestima `roots/list`y `notifications/roots/list_changed`Prefiero uno de estos reemplazos explícitos:

- ¿ Qué es esto ?`workspaceUri`o `directory`argumento de herramienta cuando el alcance varía por llamada.
- Un URI de recurso cuando la operación ya apunta a un recurso.
- Configuración de servidor cuando una implementación posee un espacio de trabajo fijo.
- Un proceso sandbox o sistema de archivos encarcelado cuando el código debe ser técnicamente incapaz de escapar.

Si todavía se necesita una integración existente para el 2026-07-28 `roots/list`durante la ventana de deprecación, el servidor lo incrusta en MRTR `inputRequests`. No debe enviar una solicitud inversa en vivo. Es un adaptador de migración; los nuevos operadores deben aceptar un alcance explícito en su lugar.

El modelo puede ver y repetir un mango explícito.

### La regla de tres capas

Un URI explícito aún no se autoriza.

1. **Authorization:**¿Está autorizado este director autenticado a usar este espacio de trabajo?
2. **Containment:**¿El URI objetivo normalizado permanece dentro del límite del espacio de trabajo autorizado?
3. **Sandbox:**¿Puede el sistema operativo evitar que un servidor comprometido se escape de todos modos?

El servidor ejecutable mantiene una lista de permisos de los URI de espacio de trabajo autorizados, normaliza los caminos codificados por ciento, verifica un límite real del componente de camino y vuelve a comprobar la contención inmediatamente antes de la eliminación.

Las comprobaciones ingenuas de prefijos de cadena están equivocadas:

```text
allowed:   file:///work/notes
attacker:  file:///work/notes-evil/secret.md
traversal: file:///work/notes/%2e%2e/private.md
```

Los dos caminos hostiles comienzan con una cadena engañosa. Normaliza primero, luego compara los componentes del camino. Un servidor de sistema de archivos de producción también debe defenderse contra las carreras de enlaces simbólicos y la semántica de la ruta específica de la plataforma.

## La solicitud sigue existiendo, pero la entrega ha cambiado

La elicitación es la característica actual del cliente para recopilar la entrada del usuario durante `tools/call`¿ Qué ?`prompts/get`, o`resources/read`. El nombre del método permanece `elicitation/create`Lo que cambió fue la dirección del flujo del cable.

Un servidor 2026-07-28 no envía una solicitud inversa JSON-RPC.`InputRequiredResult`¿Qué es esto ?

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "delete_choice": {
        "method": "elicitation/create",
        "params": {
          "mode": "form",
          "message": "Choose one matching note and confirm deletion.",
          "requestedSchema": {
            "type": "object",
            "properties": {
              "note_id": {
                "type": "string",
                "enum": ["note-3", "note-7", "note-14"]
              },
              "confirm": {"type": "boolean"}
            },
            "required": ["note_id", "confirm"]
          }
        }
      }
    },
    "requestState": "integrity-protected-delete-state"
  }
}
```

El host hace la versión del formulario. El usuario puede aceptar, rechazar o rechazarlo explícitamente.`tools/call`con una identificación nueva:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes_delete",
    "arguments": {
      "workspaceUri": "file:///Users/alice/Documents/Notes",
      "title": "TPS report"
    },
    "inputResponses": {
      "delete_choice": {
        "action": "accept",
        "content": {"note_id": "note-14", "confirm": true}
      }
    },
    "requestState": "integrity-protected-delete-state",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

El servidor verifica el estado de eco, valida la respuesta contra el esquema esperado, verifica que la nota seleccionada estaba en el conjunto de candidatos firmados, reautorizará el espacio de trabajo, volverá a comprobar la contención y luego se eliminará.

## Las negociaciones de capacidad son por solicitud

Un cliente que admite la elicitación en modo de formulario declara:

```json
{
  "io.modelcontextprotocol/clientCapabilities": {
    "elicitation": {"form": {}}
  }
}
```

Una capacidad de elicitación vacía,`"elicitation": {}`, sigue siendo equivalente a un apoyo de compatibilidad sólo en forma.`"elicitation": {"form": {}}`También admite el modo de formulario. Una declaración de URL solamente, `"elicitation": {"url": {}}`El servidor no debe incorporar un modo ausente de las capacidades de la solicitud actual, incluso si una solicitud anterior lo publicitó.

Cada solicitud también lleva`io.modelcontextprotocol/protocolVersion`. Una versión que no está presente o no está en cadena se devuelve `-32602`Una cadena no soportada devuelve .`-32022`con exactitud`supported`y `requested`Datos. Datos de apoyo de elicitación faltantes o sólo en URL `-32021`con`data.requiredCapabilities`se fija en `{"elicitation":{"form":{}}}`¿ Qué ?

Un sobre sin un JSON-RPC `id`es una notificación. Procesarlo sin emitir un éxito o respuesta de error JSON-RPC. En Streamable HTTP, una notificación aceptada recibe `202 Accepted`Sin cuerpo.

`clientInfo`El uso de la información de la información de usuario debe incluirse para el diagnóstico, pero es auto-relatado y no puede identificar al usuario para su autorización.

El servidor implementa `server/discover`y los retornos `supportedVersions`, capacidades,`ttlMs`, y `cacheScope`con`resultType: "complete"`. No anuncia Roots para este diseño moderno.`tools/list`Ese resultado devuelve la determinística`notes_delete`descriptor, un objeto válido `inputSchema`, metadatos de identidad del servidor, y pistas de caché público.

## Modo de formulario

El modo de formulario utiliza un esquema JSON restringido diseñado para diálogos utilizables. La raíz es un objeto y sus propiedades son campos primitivos planos o matrices enum compatibles.

Utilice el modo de formulario para:

- elegir uno de varios candidatos;
- la confirmación de una operación destructiva;
- la recogida de preferencias no sensibles;
- recoger un pequeño número de valores que el usuario, no el modelo, debe decidir.

No utilice el modo de formulario para contraseñas, claves de API, tokens de acceso o credenciales de pago. Esos secretos pasarían por el cliente MCP y podrían llegar a registros o contexto del modelo.

El servidor valida el contenido devuelto de nuevo. La validación del formulario del lado del cliente mejora la experiencia de usuario, pero no crea confianza.

## Modo de URL

El modo URL envía una URL web segura para una interacción fuera de banda:

```json
{
  "method": "elicitation/create",
  "params": {
    "mode": "url",
    "message": "Connect the report service to continue.",
    "url": "https://mcp.example.com/connect/report-service"
  }
}
```

Utilice cuando la información sensible debe ir directamente a un flujo web controlado por el servidor, como la autorización de terceros. El cliente muestra el destino completo y obtiene su consentimiento antes de abrirlo. No debe pre-recoger la URL.

Un `accept`respuesta significa que el usuario ha aceptado abrir la URL. No demuestra que el flujo externo se haya completado. En el retraso, el servidor verifica su propio estado y completa o devuelve otro `input_required`el resultado.

La elicitación de URL no es un reemplazo de la autorización entre el cliente MCP y el servidor MCP. Es para una interacción externa que el servidor MCP necesita realizar en nombre del usuario. El servidor debe vincular al usuario del navegador al mismo principal autenticado que comenzó la operación MCP.

## Ramas de respuesta

Tratar las acciones como decisiones de producto, no como alias:

| Action | Meaning | Safe server behavior |
|--------|---------|----------------------|
| `accept` | User submitted the interaction | Validate content and continue |
| `decline` | User explicitly refused | Return a complete, non-error refusal outcome |
| `cancel` | User dismissed or could not finish | Stop safely and allow a later retry |

Nunca interprete el contenido faltante como consentimiento. Nunca convierta el declive en un bucle repetido.

## Protección del Estado MRTR destructivo

La lista de candidatos no puede vivir solo en un valor Base64 de solicitud o sin firmar. Un cliente controla todo lo que envía de vuelta.

La lección firma una carga útil estatal que contiene:

- el principal autenticado;
- método de origen;
- digestión de `workspaceUri`y `title`El artículo 1
- las identidades permitidas de las notas indicadas en el formulario;
- fase de operación;
- corto plazo.

Antes de la mutación, el servidor también verifica el registro de notas en vivo. Esto captura carreras de eliminación y un objetivo se desplazó fuera del espacio de trabajo después de que se mostró el formulario.

En caso de una acción financiera o irreversible única, el HMAC por sí solo no impide que un estado válido se repita dentro de su vencimiento. Almacenar y consumir un nonce exactamente una vez en una tienda de reproducción compartida por cada instancia de manipulador. La lección inyecta una tienda limitada y cortada con TTL y mantiene su reclamo atómico mientras realiza la eliminación en memoria. Una base de datos de producción debe combinar la reclamación de noción y la mutación en una transacción o en un límite de escritura condicional equivalente.

Valida la interacción antes de reclamar la noción.`cancel`No realiza ninguna mutación y deja el estado retrativable hasta la expiración.`decline`es terminal, así que la lección consume el nonce sin borrar nada.

```figure
t3-roots-boundary
```

## Construye el mismo

`code/main.py`demuestra una moderna `notes_delete`herramienta:

- `tools/list`devuelve un descriptor determinístico y cachéable con el espacio de trabajo y el esquema de título requeridos.
- El alcance es explícito.`workspaceUri`¿Qué es eso?
- La configuración del servidor autoriza ese espacio de trabajo para el director de la lección.
- La normalización de URI rechaza la confusión de prefijos y el cruce codificado.
- Cada eliminación destructiva requiere una elicitación en modo de forma.
- La elicitación viaja hacia dentro.`resultType: "input_required"`¿ Qué ?
- Firmado .`requestState`se une a la lista exacta de candidatos y a los argumentos originales.
- Una tienda de reproducción inyectada rechaza el mismo estado aceptado o rechazado en todas las instancias del servidor.
- El retraso utiliza un nuevo ID de solicitud y devuelve `resultType: "complete"`¿ Qué ?

El almacenamiento de datos está en la memoria por lo que el comportamiento del protocolo es fácil de inspeccionar.

## Usalo

Desde la raíz del repositorio:

```bash
cd phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/code
python3 main.py
python3 -m unittest discover tests -v
```

Los puntos de control previstos:

- Discovery anuncia herramientas sin raíces.
- Retorno de descubrimiento de herramientas `notes_delete`con`resultType`, identidad del servidor, y pistas de caché.
- Solicitud de identificación`1`devuelve el formulario en `inputRequests.delete_choice`¿ Qué ?
- Solicitud de identificación`2`se hace eco del estado firmado y completa la eliminación.
- Un camino de prefijo y un camino de cruce codificado no pueden contenerse.
- Un título cambiado no puede reutilizar el estado de confirmación original.
- Una disminución deja la nota sin cambios.
- Dos objetos del servidor que comparten el estado de nota y repetición no pueden ejecutar una confirmación.
- Las declaraciones de formulario vacío y explícito funcionan, mientras que el soporte solo para URL devuelve exactas `-32021`requisitos de los formularios.
- Las versiones no compatibles utilizan el mismo `-32022`forma de los datos.
- Una notificación sin id no produce respuesta JSON-RPC.

## Envío

`outputs/skill-elicitation-form-designer.md`Diseña el alcance explícito, las verificaciones de autorización, el formulario MRTR, las ramas de respuesta y la vinculación de estado. Se niega a tratar las raíces desactualizadas como una caja de arena o a recopilar secretos a través del modo de formulario.

## Los ejercicios

1. Sustituye la memoria de repetición de almacenamiento con SQLite. Utilice una transacción para reclamar el noce y borrar la nota, luego demuestre que dos procesos no pueden comprometerse ambos.
2. Añadir`url`• la capacidad de negociación y un flujo de configuración fuera de banda.`inputResponses`¿ Qué ?
3. Reemplaza el mapa de notas en memoria con una base de datos temporal de SQLite. Reverifique la autorización y contención dentro de la transacción de mutación.
4. Añadir una política de enlace simbólico para una implementación real del sistema de archivos. Explicar por qué la contención léxica URI sola no puede detener una fuga de enlace simbólico.
5. Diseñar un adaptador 2025-11-25 que mapee la salida del procesador MRTR moderno a la elicitación iniciada por el servidor heredado. Manténgalo aislado del procesador actual.

## Términos clave

| Term | Meaning in 2026-07-28 |
|------|------------------------|
| Roots | Deprecated informational workspace hints, not authorization or sandboxing |
| Explicit scope | Workspace, directory, or resource handle visible in request arguments |
| Containment | Normalized path-component check that keeps a target inside a boundary |
| Elicitation | Client feature for obtaining user input during an MCP operation |
| Form mode | In-band structured user input using a restricted flat schema |
| URL mode | Out-of-band interaction for sensitive or external workflows |
| MRTR | Stateless input-required result followed by a fresh retry |
| `requestState` | Opaque state echoed exactly and integrity-checked by the server |
| Decline | Explicit user refusal |
| Cancel | Dismissal or incomplete interaction without approval |

## Compatibilidad con el legado

Para un compañero fijado a 2025-11-25, `roots/list`¿ Qué ?`notifications/roots/list_changed`, y en vivo por el servidor iniciado `elicitation/create`Etiquetar el legado del adaptador. No permita que una lista de raíz heredada para eludir la autorización del servidor, y no llevar supuestos de sesión de protocolo en el procesador moderno.

## Leer más

- [MCP 2026-07-28 Elicitation](https://modelcontextprotocol.io/specification/2026-07-28/client/elicitation)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Roots deprecation](https://modelcontextprotocol.io/specification/2026-07-28/client/roots)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
