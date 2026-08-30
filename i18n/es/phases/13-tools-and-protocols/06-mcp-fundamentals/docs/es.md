# Principios básicos de los MCP: Solicitudes de apatrida y JSON-RPC

> El MCP moderno no tiene apretón de manos y no tiene sesión de protocolo. Cada solicitud debe llevar suficientes metadatos para ser entendidos, autorizados, enrutados y retestados por sí solo.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 01 through 05
**Time:** ~55 minutes

## Objetivos de aprendizaje

- Distinguir los primitivos del servidor de MCP de sus características del lado del cliente.
- Construir solicitudes y respuestas válidas de JSON-RPC 2.0 para MCP `2026-07-28`¿ Qué ?
- Agregue la versión del protocolo, las capacidades del cliente e identidad del cliente a cada solicitud.
- Usar`server/discover`y manejar .`UnsupportedProtocolVersionError`sin dar una mano.
- Recoger una solicitud independiente de la validación hasta un resultado completo.

## El problema

Un servidor MCP puede recibir dos solicitudes consecutivas de diferentes clientes, con diferentes capacidades, en el mismo proceso o en el mismo servidor HTTP. Si el servidor recuerda lo que la solicitud anterior declaró, puede aplicar los permisos equivocados o devolver la forma incorrecta del cable.

MCP `2026-07-28`El núcleo del protocolo es estatal. Un servidor debe decidir cómo manejar la solicitud actual de la solicitud actual, no del historial de conexión.

Esto cambia el modelo mental. La antigua secuencia era conexión primero, apretón de manos segundo, operaciones tercero.

1. El cliente envía una solicitud de auto-descripción.
2. El servidor valida la versión y las capacidades de esa solicitud.
3. El servidor maneja el método.
4. El servidor devuelve un resultado de tipografía o un error JSON-RPC.

La siguiente solicitud repite el mismo proceso desde cero.

## El concepto

### Primitivas del servidor

Los servidores MCP exponen tres primitivas primarias:

1. **Tools**Las acciones controladas por modelos, descubiertas con `tools/list`y invocado con `tools/call`¿ Qué ?
2. **Resources**Los datos de la información de la URI se encuentran en el`resources/list`y recuperado con `resources/read`¿ Qué ?
3. **Prompts**son plantillas reutilizables, descubiertas con `prompts/list`y se traduce con `prompts/get`¿ Qué ?

Las raíces, la toma de muestras y la tala permanecen en el `2026-07-28`Es un esquema de compatibilidad, pero está desactualizado. Las nuevas implementaciones deben utilizar herramientas o recursos explícitos para las raíces, API directas de proveedores de modelos para la muestreo y stderr o OpenTelemetry para el registro. La obtención de datos sigue disponible a través de las solicitudes de viaje de ida y vuelta múltiples, donde un servidor devuelve una solicitud de entrada y el cliente retoma la operación original. Un servidor moderno nunca inicia una solicitud independiente de JSON-RPC.

### Envase de JSON-RPC

MCP utiliza JSON-RPC 2.0:

- Solicitud: `{jsonrpc, id, method, params}`
- Respuesta: `{jsonrpc, id, result}`o `{jsonrpc, id, error}`
- Notificación: `{jsonrpc, method, params}`sin ninguna`id`

La solicitud `id`correlaciona una respuesta. No crea una sesión de protocolo.

### Metadatos requeridos de la solicitud

Cada solicitud moderna lleva un`_meta`objeto dentro `params`¿Qué es esto ?

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/list",
  "params": {
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

Se requiere la versión del protocolo y las capacidades del cliente. Se recomienda la identidad del cliente. Se trata de datos de visualización y depuración auto-reportados, no de una credencial de seguridad.

El servidor no debe inferir ninguno de estos valores de una solicitud anterior, un proceso de estudio, una conexión HTTP o un encabezado de transporte solo.

### Resultados completos y identidad del servidor

Cada resultado moderno exitoso incluye`resultType`. Un resultado final normal utiliza `"complete"`Los servidores también deben identificarse en los metadatos de los resultados:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "resultType": "complete",
    "tools": [],
    "ttlMs": 30000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "notes-server",
        "version": "1.0.0"
      }
    }
  }
}
```

`tools/list`¿ Qué ?`resources/list`¿ Qué ?`prompts/list`¿ Qué ?`resources/templates/list`¿ Qué ?`resources/read`, y `server/discover`Los resultados se pueden almacenar en caché.`ttlMs`y `cacheScope`. Un defecto seguro es `ttlMs: 0`y `cacheScope: "private"`Los elementos de la lista deben tener orden determinista para que las respuestas equivalentes produzcan claves de caché estables y contexto de modelo estable.

### Descubrimiento sin apretar la mano

Cada servidor moderno debe implementar`server/discover`El cliente puede llamar antes de otro método para obtener:

- `supportedVersions`
- servidor `capabilities`
- uso opcional `instructions`
- identidad del servidor en resultado `_meta`
- indicios de caché

El descubrimiento es útil, pero no es una puerta.`tools/list`En primer lugar, porque esa solicitud ya cuenta con su versión y sus capacidades de protocolo.

Si la versión solicitada no es compatible, el servidor devuelve el código JSON-RPC `-32022`con:

```json
{
  "requested": "2027-01-01",
  "supported": ["2026-07-28"]
}
```

El cliente selecciona una versión moderna compatible entre sí y vuelve a intentar con un nuevo ID de solicitud JSON-RPC.

### Un ciclo de vida de la solicitud

Traza una solicitud moderna en este orden:

1. Parsear un sobre JSON-RPC.
2. Confirmarlo .`jsonrpc`¿ Es verdad ?`"2.0"`, un `id`existe, `method`es una cuerda, y `params`es un objeto.
3. Requerir la cadena de versiones y el objeto de capacidad en `params._meta`; los metadatos malformados o faltantes son `-32602`¿ Qué ?
4. En un límite HTTP, comparar la versión, el método y los encabezados de nombres aplicables con el cuerpo.`-32020`incluso cuando uno de los dos valores de versión no esté soportado.
5. Una vez que se establezca la igualdad, rechace una versión coincidente pero no respaldada con `-32022`¿ Qué ?
6. Compruebe las capacidades requeridas, luego viaja por `method`y validar los argumentos específicos del método.
7. Autentique y autorice la operación de concreto antes de que su manipulador se ejecute.
8. Regresa un resultado completo con identidad del servidor.
9. Olvídate de los metadatos del protocolo.

Esta orden impide que dos componentes interpreten llamadas diferentes.`Mcp-Name: notes.read`mientras el origen ejecuta `params.name: notes.delete`También mantiene entradas malformadas, confusión de encabezado, negociación de versiones, falla de capacidad, autorización y falla de manipulador como evidencia distinta.

Cerrar stdin o una respuesta HTTP termina la actividad de transporte. No termina una sesión de protocolo porque el MCP moderno no tiene sesión de protocolo.

### Compatibilidad explícita con el legado

Versiones a través de `2025-11-25`uso `initialize`¿ Qué ?`notifications/initialized`, las capacidades de conexión-escalado, y, en anteriores Streamable HTTP, sesiones de protocolo opcionales. Ese comportamiento sigue siendo relevante cuando un cliente de doble era habla con un servidor viejo.

Mantenga las eras separadas. Una solicitud moderna se identifica por los metadatos requeridos por solicitud. Una conexión heredada se selecciona solo a través del sendero de retroceso documentado. No envíe `initialize`como el defecto para un `2026-07-28`¿Qué es eso?

Instatado tiene por tanto un significado específico de la época.`2026-07-28`En el caso de las aplicaciones de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la ley, se puede considerar que la aplicación de la misma es una de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la aplicación de la ley.`2025-11-25`La implementación de una doble era no es una máquina de estado permisivo. Es un núcleo moderno sin estado junto a un adaptador heredado aislado, con una decisión de selección explícita antes de que se ejecute cualquiera de los paresores.

Ninguno de los dos significados prohíbe el estado de aplicación duradero. Un flujo de trabajo, tarea o borrador puede vivir detrás de un mango opaco en una tienda compartida. El cliente envía ese mango como entrada ordinaria, y cada réplica autentica y autoriza su uso. El contexto del protocolo no debe filtrarse a esa tienda como sustituto de la sesión eliminada.

```figure
mcp-tool-call
```

## Usalo

`code/main.py`crea, valida, rastrea y envía mensajes MCP modernos sin un marco. ejecuta:

```bash
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Cuidado con tres invariantes en la salida:

- Cada solicitud repite su `_meta`los campos.
- Cada resultado exitoso es`resultType: "complete"`y incluye la identidad del servidor.
- El resultado de la lista está ordenado deterministicamente y tiene sugerencias explícitas de caché.

## Envío

Esta lección nos lleva .`outputs/skill-mcp-handshake-tracer.md`El nombre histórico del archivo sigue siendo estable, pero el artefacto es ahora un rastreador de solicitudes sin estado. Audita cada mensaje de forma independiente y etiqueta el tráfico de apretones de manos heredados solo cuando está realmente presente.

## Los ejercicios

1. Cambiar la versión de protocolo de una solicitud a `2027-01-01`Confirme que el código de error es`-32022`y los datos anuncian la versión compatible.
2. Retirada`io.modelcontextprotocol/clientCapabilities`Confirmar que el servidor no reutiliza las capacidades de la primera solicitud.
3. Revuelve el registro de herramientas en memoria.`tools/list`todavía devuelve el mismo orden determinista.
4. Cambiar`cacheScope`de la`public`¿ Qué ?`private`Explicar qué contextos de autorización pueden reutilizar la respuesta en cada caso.
5. Añadir una opción `clientInfo`El requisito debe permanecer válido porque se recomienda la identidad del cliente, no se requiere.

## Términos clave

| Term | Meaning |
|------|---------|
| Stateless protocol | Every request supplies the metadata needed to interpret it |
| Request metadata | Version, client capabilities, and recommended client identity in `params._meta` |
| `server/discover` | Mandatory server method for versions, capabilities, instructions, and identity |
| `resultType` | Discriminator on every successful modern result |
| Cacheable result | Result that includes required `ttlMs` and `cacheScope` hints |
| Protocol era | Modern per-request metadata or legacy connection-scoped initialization |
| Transport lifetime | Process, connection, or response-stream lifetime, not protocol session state |
| `-32022` | Unsupported protocol version error with requested and supported versions |

## Leer más

- [MCP Architecture](https://modelcontextprotocol.io/specification/2026-07-28/architecture)
- [MCP Base Protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP 2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
