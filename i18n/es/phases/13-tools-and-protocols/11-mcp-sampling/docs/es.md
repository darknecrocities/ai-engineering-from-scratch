# Introducción del modelo de MCP: muestreo de migración y MRTR de apátridas

> MCP 2026-07-28 deprecia la muestreo para nuevos diseños y elimina el canal de solicitud de servidor a cliente. Si un flujo de trabajo existente todavía necesita el modelo del cliente, el servidor devuelve un `input_required`El cliente retoma la solicitud original con la salida del modelo. El bucle de razonamiento se vuelve explícito, limitado y sin estado en la capa de protocolo.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**Time:** ~75 minutes

## Objetivos de aprendizaje

- Explica por qué la muestreo se ha desactualizado en MCP 2026-07-28 y elige el modelo de integración directa predeterminado para los nuevos servidores.
- Implementar un flujo de trabajo de compatibilidad que lleve `sampling/createMessage`a través de las solicitudes de viajes múltiples y redondos (MRTR).
- Coloque la revisión del protocolo y las capacidades del cliente en cada solicitud `_meta`Objeto.
- Regreso .`resultType: "input_required"`y volver a probar el método original con un nuevo ID JSON-RPC.
- Protección de la integridad `requestState`y se vinculen al principio, método, argumentos y vencimiento.
- Los bucles con modelo de ayuda ligada con controles de capacidad, aprobación, validación de respuesta y un límite redondo.

## La decisión antes del Protocolo

Una herramienta como `summarize_repo`Necesita dos tipos de trabajo:

1. Trabajo determinista: lista de archivos, lectura de archivos permitidos, validación de caminos y ensamblaje de contenido.
2. Trabajo de modelo: elegir archivos representativos y sintetizar el resumen.

Ahora tienes dos arquitecturas válidas.

### Nuevo servidor: se integra directamente con un proveedor de modelos

Este es el estándar actual. El servidor posee la selección de modelos, credenciales, presupuestos, retemplazos y observabilidad.`tools/call`el resultado para el cliente de MCP.

Elige esto cuando el servidor ya sea un servicio alojado o cuando el comportamiento predecible del modelo sea más importante que el uso del modelo del host.

### Flujo de trabajo de muestreo existente: migrarlo a MRTR

El muestreo todavía existe durante su ventana de deprecación. Un servidor dirigido a 2026-07-28 no puede enviar una transmisión en vivo `sampling/createMessage`En cambio, el cliente debe incorporar esa solicitud en un`InputRequiredResult`¿ Qué ?

Elegir este camino de compatibilidad sólo cuando se utiliza el modelo del cliente y las credenciales es un requisito real del producto.

## El contrato de apatrida

El protocolo de julio de 2026 no tiene`initialize`el intercambio, no `notifications/initialized`, y no .`Mcp-Session-Id`Cada solicitud contiene la información que antes vivía en el apretón de manos:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

El servidor valida la revisión en cada solicitud. Una versión que no está disponible o sin cadena es un parámetro inválido.`-32602`Una cadena no soportada devuelve .`-32022`con datos exactos `{"supported":["2026-07-28"],"requested":"<client version>"}`. Una capacidad de muestreo faltante regresa .`-32021`con`data.requiredCapabilities`se fija en `{"sampling":{}}`¿ Qué ?

Un sobre sin un JSON-RPC `id`El receptor puede procesarlo, pero no emite una respuesta de éxito ni una respuesta de error. Un adaptador HTTP transmitible devuelve `202 Accepted`sin organismo para una notificación aceptada.

El servidor también implementa `server/discover`con el exacto `supportedVersions`clave, capacidades, `ttlMs`, y `cacheScope`para que un cliente pueda aprender y almacenar en caché el contrato del servidor antes de llamar a una herramienta.`tools`, el servidor también implementa obligatorio `tools/list`Es determinista .`summarize_repo`el descriptor incluye un objeto válido `inputSchema`¿ Qué ?`resultType: "complete"`, metadatos de identidad del servidor, y pistas de caché público.

Cada resultado moderno exitoso tiene un discriminador:

- `resultType: "complete"`significa que la operación ha terminado.
- `resultType: "input_required"`significa que el cliente debe cumplir con las solicitudes incorporadas y volver a intentarlo.
- Las extensiones pueden definir tipos de resultados adicionales.`"task"`En la Lección 13.

## Una ronda de MRTR

El servidor no puede llamar al cliente mientras se maneja la solicitud. En su lugar devuelve este resultado:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "pick_files": {
        "method": "sampling/createMessage",
        "params": {
          "messages": [
            {
              "role": "user",
              "content": {
                "type": "text",
                "text": "Choose three representative files and return a JSON array."
              }
            }
          ],
          "systemPrompt": "Return only the requested value.",
          "modelPreferences": {
            "costPriority": 0.8,
            "intelligencePriority": 0.2
          },
          "maxTokens": 400
        }
      }
    },
    "requestState": "opaque-integrity-protected-value"
  }
}
```

El cliente verifica que admite la muestra, aplica sus políticas de aprobación y modelo y obtiene una respuesta de modelo. Luego envía una nueva solicitud con un id JSON-RPC diferente:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "inputResponses": {
      "pick_files": {
        "role": "assistant",
        "content": {
          "type": "text",
          "text": "[\"README.md\", \"server.py\", \"docs/intro.md\"]"
        },
        "model": "host-model",
        "stopReason": "endTurn"
      }
    },
    "requestState": "opaque-integrity-protected-value",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}}
    }
  }
}
```

El retraso no es una continuación de una sesión de protocolo. Es una nueva solicitud que repite el método y los argumentos originales, añadiendo sólo los de la ronda actual `inputResponses`, y los ecos .`requestState`byte por byte.

El MRTR sólo está permitido en `tools/call`¿ Qué ?`prompts/get`, y `resources/read`Un servidor no debe regresar .`input_required`de métodos no relacionados.

## Estado de la mayoría de las partes

Esta lección necesita dos modelos:

1. `pick_files`devuelve una matriz JSON.
2. `summary`devuelve la prosa final.

Cada retraso solo contiene las respuestas de esa ronda. El servidor, por lo tanto, pone la fase y los datos intermedios validados en la siguiente `requestState`¿ Qué ?

Tratar ese valor como controlado por el atacante. Firmar un nombre de fase crudo no es suficiente.

- el principal autenticado, no auto-informado `clientInfo`El artículo 1
- el método de origen;
- un resumen de los argumentos originales;
- una caducidad corta;
- la fase actual y los valores intermedios validados.

Utilice HMAC cuando no se requiere confidencialidad. Utilice cifrado autenticado cuando el cliente no debe leer el estado. Rechazar una firma mala, valor expirado, cambio de principal o argumentos cambiados con `-32602`¿ Qué ?

El cliente no debe analizar ni modificar `requestState`Su único trabajo es hacer eco de la cuerda exacta en el retraso.

## Las preferencias de modelo son indicios

`costPriority`¿ Qué ?`speedPriority`, y `intelligencePriority`Las preferencias de la probabilidad de un cliente pueden ser ignoradas porque el cliente posee una política de modelo.

Mantenga .`includeContext`En el`"none"`Si mantiene un flujo de muestreo heredado. Otros modos de contexto aumentan el riesgo de fugas y son en sí mismos desactualizados.

## Las medidas de seguridad

El cliente es el límite de confianza para las solicitudes de muestreo integradas.

- Muestre al usuario lo que el servidor le pide al modelo cuando la política requiere aprobación.
- Un servidor malicioso puede crear un bucle de gasto de modelo.
- Valida cada respuesta de muestreo antes de usarla como un nombre de archivo, URL o entrada de herramienta.
- Limite los bytes y tokens por ronda.
- Rechazar una solicitud de entrada que no se declaró en las capacidades actuales del cliente.
- Mantenga la salida del modelo fuera de las decisiones de autorización.
- Registre el método de origen y la clave de entrada-solicitud sin registrar el contenido de la solicitud sensible.

`clientInfo`y `serverInfo`Los datos de identidad de los usuarios son metadatos de visualización y diagnóstico.

```figure
t3-sampling-flip
```

## Construye el mismo

`code/main.py`Implementa el flujo completo de dos rondas sin paquete de terceros:

- `server/discover`retorno `supportedVersions`, anuncia el soporte de herramientas, y devuelve sugerencias de caché.
- `tools/list`devuelve un determinista, cachéable `summarize_repo`Descriptor con un esquema de entrada de objeto.
- `tools/call`valida los metadatos por solicitud.
- El primer resultado se incorpora `sampling/createMessage`para la selección de archivos.
- El primer retraso valida el resultado del modelo y incorpora una segunda solicitud.
- Protegida por HMAC `requestState`se realiza la fase entre las solicitudes independientes.
- El resultado final utiliza `resultType: "complete"`¿ Qué ?

El modelo de anfitrión falso hace que el ejemplo sea determinista.`fake_host_model`La máquina del lado del servidor debe permanecer determinista y testable.

## Usalo

Desde la raíz del repositorio:

```bash
cd phases/13-tools-and-protocols/11-mcp-sampling/code
python3 main.py
python3 -m unittest discover tests -v
```

Los puntos de control previstos:

- Discovery devuelve un resultado completo con `ttlMs`y `cacheScope`¿ Qué ?
- El descubrimiento de herramientas devuelve el mismo descriptor clasificado con `resultType`, identidad del servidor, y pistas de caché.
- Capacidades faltantes y versiones no compatibles utilizan exacto `-32021`y `-32022`datos de error.
- Una notificación sin id no produce respuesta JSON-RPC.
- Los documentos de identificación de la solicitud son `[1, 2, 3]`, que demuestra que cada ronda de MRTR es independiente.
- Los dos primeros resultados son:`input_required`¿ Qué ?
- El resultado final es `complete`y contiene los archivos seleccionados más un resumen.
- Cambiar los argumentos originales en una nueva prueba falla en la verificación del estado de solicitud.

## Envío

`outputs/skill-sampling-loop-designer.md`Es un programa de migración que decide si se debe eliminar la muestreo a favor de la integración directa del modelo. Si se requiere compatibilidad, produce las rondas MRTR, la vinculación del estado, la puerta de capacidad, el presupuesto, la validación y el plan de eliminación.

## Los ejercicios

1. Cambiar la respuesta de selección de archivos a JSON inválido. Confirmar el servidor devuelve `-32602`en lugar de confiar en la salida del modelo.
2. Cambiar`audience`Explique por qué el estado sellado bloquea la reutilización de las solicitudes cruzadas.
3. Añadir una tercera ronda que le pida al anfitrión que critique el resumen. Llevar el resumen anterior dentro del estado firmado y limitar todo el flujo en tres rondas.
4. Elimine la muestreo reemplazando la llamada de regreso de host falso con un adaptador de modelo propiedad del servidor.
5. Añadir una prueba de vencimiento utilizando un valor de estado que es un segundo después de su fecha límite.

## Términos clave

| Term | Meaning in 2026-07-28 |
|------|------------------------|
| Sampling | Deprecated feature that asks the client's model for a completion |
| MRTR | Stateless retry pattern for client input required during a request |
| `InputRequiredResult` | Result with `resultType: "input_required"` |
| `inputRequests` | Server-assigned map of embedded elicitation, sampling, or roots requests |
| `inputResponses` | Current round's client results keyed like `inputRequests` |
| `requestState` | Opaque server state echoed exactly by the client and verified by the server |
| `resultType` | Required discriminator for modern MCP results |
| Direct model integration | Recommended replacement for new servers that need model inference |
| Capability gate | Rule that prevents sending an embedded request the client did not advertise |
| Loop budget | Maximum rounds, tokens, bytes, time, and spend allowed for the operation |

## Compatibilidad con el legado

Un cliente fijado a 2025-11-25 puede seguir utilizando el antiguo servidor iniciado `sampling/createMessage`No haga que el camino de sesión sea la arquitectura de un servidor 2026-07-28.

Los SDK oficiales pueden traducir modernos `input_required`Ese shim es un límite de compatibilidad, no el permiso para añadir nueva lógica dependiente de la sesión.

## Leer más

- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP Sampling deprecation](https://modelcontextprotocol.io/seps/2577-deprecate-roots-sampling-and-logging)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
