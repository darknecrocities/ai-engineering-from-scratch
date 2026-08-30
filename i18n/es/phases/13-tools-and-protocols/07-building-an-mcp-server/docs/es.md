# Construir un servidor MCP: Python sin estado y TypeScript

> Un servidor MCP moderno no recuerda un apretón de manos. Valida los metadatos en cada solicitud, ejecuta un procesador y devuelve un resultado tipado.

**Type:** Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 13, Lesson 06
**Time:** ~85 minutes

## Objetivos de aprendizaje

- Implementación obligatoria `server/discover`para MCP `2026-07-28`¿ Qué ?
- Valida la versión del protocolo y las capacidades del cliente en cada solicitud.
- Exponer herramientas, recursos y instrucciones con ordenamiento de lista determinista.
- Regreso .`resultType`, identidad del servidor, y indicios de caché sobre los resultados correctos.
- Servir el mismo contrato sin estado sobre el estudio de línea nueva y limitada en Python y TypeScript.

## El problema

Un servidor que almacena las capacidades del cliente después del primer mensaje es fácil de construir y difícil de operar. El mismo proceso puede servir a clientes secuenciales. Una solicitud remota puede aterrizar en un trabajador diferente. Una declaración de capacidad obsoleta puede filtrar el comportamiento a través de los límites de autorización.

MCP `2026-07-28`La aplicación puede mantener notas duraderas, trabajos o manipulaciones de estado explícito. Lo que no puede mantener es el estado de protocolo oculto que cambia la forma en que se descifre una solicitud posterior.

Esta lección construye un servidor de notas dos veces. Las versiones de Python y TypeScript usan solo sus bibliotecas estándar para el núcleo del protocolo. Ambos exponen los mismos métodos y aplican el mismo contrato de cable.

## El concepto

### El moderno bucle de envío

```text
read one JSON-RPC line
parse the envelope
if it is a notification, do not respond
validate params._meta for this request
route by method
wrap success with resultType and serverInfo
write one JSON-RPC response line
forget request-scoped metadata
```

Tres reglas del estudio siguen siendo importantes:

- Escriba sólo mensajes JSON-RPC a stdout. Envía diagnósticos a stderr.
- Delimitar los mensajes con una línea nueva y vertiente cada respuesta.
- Salir inmediatamente cuando el stdin llegue a la Oficina de Ejecuciones.

La vida útil del proceso es una vida útil del transporte.

### Validación de la solicitud

Cada solicitud debe contener:

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Se requieren los dos primeros campos. `clientInfo`Se recomienda validar una forma de identidad actual, pero no tratarla como autenticación.

Si la versión no está soportada, devuelva el código `-32022`con`requested`y `supported`. Los metadatos de la solicitud faltantes son parámetros inválidos, código `-32602`Nunca llenes los campos que faltan de una llamada anterior.

### Descubrimiento obligatorio

Los servidores modernos deben implementar `server/discover`. Un resultado completo de descubrimiento incluye versiones modernas compatibles, capacidades, instrucciones opcionales, sugerencias de caché y la identidad del servidor en el resultado `_meta`¿Qué es esto ?

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

Discovery no desbloquea el servidor. Un cliente puede llamar`tools/list`sin llamar descubrimiento porque`tools/list`ya contiene los mismos metadatos de la solicitud.

### Herramientas

`tools/list`El orden estable mejora la caché de la respuesta y mantiene el contexto del modelo estable.`ttlMs`y `cacheScope`¿ Qué ?

`tools/call`devuelve bloques de contenido y `isError`. Utilice un error JSON-RPC cuando el envase del protocolo o los parámetros del método son inválidos.`isError: true`cuando se ejecuta una invocación de herramienta válida pero la herramienta en sí misma falla.

Las anotaciones de las herramientas siguen siendo indicios, no ejecuciones:

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

El host debe usarlos para la confirmación y presentación. El servidor debe seguir aplicando la autorización real.

### Recursos

`resources/list`devuelve descriptores de URI estables. `resources/read`devuelve el contenido escrito. ambos son caché en `2026-07-28`, por lo que ambos incluyen`ttlMs`y `cacheScope`¿ Qué ?

Usar`cacheScope: "private"`Una caché compartida no debe reutilizar una respuesta privada en contextos de autorización.

La entrega de cambios moderna no utiliza `resources/subscribe`Un cliente abre .`subscriptions/listen`y las peticiones `resourceSubscriptions`La lección 10 construye ese flujo.

### Las instrucciones

`prompts/list`es cachéable y determinista. `prompts/get`El resultado de la solicitud de solicitud de presentación está completo, pero no es uno de los resultados cachéables o de lectura que requieren sugerencias de caché.

### Cada resultado exitoso es escrito

Los ejemplos utilizan un envase para cada éxito:

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

Lista, lectura y manipulador de descubrimientos añadir `ttlMs`Además`cacheScope`La centralización de este envoltorio evita que un manipulador omita silenciosamente los campos de resultados modernos.

### No se iniciaron solicitudes del servidor

Un servidor moderno puede enviar notificaciones relacionadas con una solicitud de cliente o notificaciones en una apertura de cliente `subscriptions/listen`No debe enviar su propia solicitud JSON-RPC.

Cuando un manipulador necesita muestreo, elicitación o entrada de raíces, devuelve un `input_required`El cliente cumple las solicitudes de entrada integradas y vuelve a probar el método original con un nuevo ID de solicitud.

### Compatibilidad explícita con el legado

Un servidor de doble era también puede implementar la `2025-11-25`El sistema de control de la carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de carga de`_meta`los campos están presentes y el comportamiento legado cuando recibe `initialize`¿ Qué ?

No ponga un `2026-07-28`No se puede hacer un sello de la mano.`resultType`El código de esta lección es deliberadamente moderno sólo para que sus invariantes permanezcan visibles.

```figure
t3-dispatch-loop
```

## Usalo

Ejecutar la demostración y pruebas finitas del servidor Python:

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

Ejecutar el puerto TypeScript con un ejecutor TypeScript:

```bash
npx tsx main.ts --demo
```

La demostración envía .`server/discover`, enumera cada primitivo, invoca herramientas y muestra un error de versión no soportado.

## Envío

Esta lección nos lleva .`outputs/skill-mcp-server-scaffolder.md`Produce un plan de servidor moderno con un contrato de descubrimiento, validación por solicitud, listas deterministas cachéables y un adaptador heredado aislado opcional.

## Los ejercicios

1. Eliminar las capacidades de una solicitud y demostrar que el servidor no reutiliza la declaración de la solicitud anterior.
2. Revierten el `TOOLS`¿ Qué ?`PROMPTS`Confirmar que todos los resultados de la lista permanecen estables.
3. Añadir un destructivo `notes_delete`La aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la ley es.`destructiveHint`sólo como una pista de experiencia.
4. Añadir`resources/templates/list`con`ttlMs`¿ Qué ?`cacheScope`, y el orden determinista.
5. Construir un adaptador separado para `2025-11-25`Añadir pruebas que demuestren que una solicitud moderna nunca entra.

## Términos clave

| Term | Meaning |
|------|---------|
| Stateless server | Handles each request from its own metadata without protocol-session memory |
| `server/discover` | Mandatory modern method that advertises versions and capabilities |
| Complete result | Successful modern result with `resultType: "complete"` |
| Cacheable result | Discovery, list, or resource-read result with `ttlMs` and `cacheScope` |
| Deterministic list | Same logical registry produces the same item order |
| Server identity | Recommended `io.modelcontextprotocol/serverInfo` in result `_meta` |
| Tool error | Valid tool call that returns content with `isError: true` |
| Protocol error | Invalid JSON-RPC or MCP request returned through `error` |

## Leer más

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
