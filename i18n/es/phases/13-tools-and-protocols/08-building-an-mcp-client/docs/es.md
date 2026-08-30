# Construir un cliente MCP: Descubrimiento, enrutamiento y retroceso de doble era

> Un cliente MCP moderno repite su contrato en cada solicitud. Su decisión de compatibilidad más difícil es saber cuándo un servidor viejo es realmente viejo y cuándo un servidor moderno está reportando un error corregible.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lesson 07
**Time:** ~85 minutes

## Objetivos de aprendizaje

- Construir todos los MCP `2026-07-28`solicitud con metadatos actuales.
- Probe los servidores de estudio con `server/discover`y seleccionar una versión compatible entre sí.
- Autorizar una sonda de legado limitado sólo para los pares explicitamente autorizados.
- Sólo acepta una era de legado después de validar un positivo .`initialize`resultado para una revisión respaldada.
- Combine listas de herramientas deterministas sin sobrescribir silenciosamente las colisiones.
- Ruta llamadas al compañero que posee cada herramienta sin inventar sesiones de protocolo.

## El problema

Un agente de alojamiento generalmente habla con más de un servidor MCP. Debe descubrir cada servidor, fusionar catálogos de herramientas, resolver nombres duplicados, llamadas de ruta y recuperarse de fallas de transporte.

El `2026-07-28`La revisión hace que el estado estacionario sea más sencillo porque cada solicitud es autónoma.

- un servidor moderno que admita la versión preferida;
- un servidor moderno que devuelva una versión reconocida o un error de encabezado;
- Un servidor heredado que nunca ha oído hablar de `server/discover`El artículo 1
- Un servidor heredado que permanece en silencio hasta que recibe `initialize`¿ Qué ?

Tratar cada error de sonda como legado es peligroso. Una solicitud moderna malformada, un servidor sobrecargado, un proceso muerto y un servidor viejo pueden producir el mismo tiempo o cerrar la conexión. Esas señales son ambigüas. El cliente debe combinar la intención explícita del operador con pruebas positivas de protocolo antes de elegir la era legada.

## El concepto

### Un compañero, no una sesión de protocolo

Mantenga un registro de transporte para cada proceso o punto final del servidor:

- Función de manija de transporte o de envío;
- la época y la versión del protocolo seleccionados;
- las capacidades del servidor que hayan sido descubiertas por última vez;
- la última lista de herramientas deterministas;
- las identificaciones de las solicitudes pendientes de correlación;
- salud del transporte.

Esto es la contabilidad del cliente. No es el estado de sesión del protocolo. En el MCP moderno, el servidor todavía recibe la versión actual y las capacidades en cada solicitud.

### Construye cada solicitud moderna desde cero

```python
def modern_request(request_id, method, params, version, capabilities):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {
            **params,
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": version,
                "io.modelcontextprotocol/clientCapabilities": capabilities,
                "io.modelcontextprotocol/clientInfo": CLIENT_INFO,
            },
        },
    }
```

No adjunta metadatos una vez a un objeto de conexión y asuma que alcanzó el cable.

### Descubrimiento moderno

`server/discover`Retorna las versiones compatibles, las capacidades del servidor, las instrucciones, las sugerencias de caché y la identidad del servidor recomendada. Un cliente elige la versión moderna más alta compatible mutuamente.

Discovery es opcional para un cliente moderno, pero se recomienda en el estudio. Algunos servidores legacy aceptan una operación antes de la inicialización, por lo que enviar `tools/list`primero puede producir un éxito ambiguo. `server/discover`crea un límite de la era limpia.

### La sonda de compatibilidad de estudio

Un cliente de estudio de doble época envía `server/discover`con sus metadatos modernos preferidos antes de cualquier otra solicitud. Hay tres clases de resultados:

1. **DiscoverResult.**El servidor es moderno. Seleccione una versión compatible y continúe con metadatos por solicitud.
2. **Recognized modern error.**El servidor es moderno.`-32022`, elegir entre `data.supported`Para errores de encabezado o capacidad, corrija la solicitud. No envíe `initialize`¿ Qué ?
3. **Ambiguous signal.**Un error JSON-RPC no reconocido, tiempo de espera, cierre de conexión o respuesta vacía no identifica una era.

Los errores de protocolo modernos reconocidos incluyen:

- `-32020`HeaderNo coincide
- `-32021`Falta de capacidad requerida
- `-32022`No soportado ProtocoloVersión

Los errores modernos reconocidos siguen siendo modernos incluso cuando el peer está en la lista de permisos heredados. Una vez que un servidor demuestra que entiende el vocabulario de errores moderno, enviando `initialize`Sería una rebaja.

No se trate`-32601`El sistema de detección de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de

### La autorización es la intención del operador, no la evidencia

La compatibilidad con el legado debe ser una propiedad explícita de una configuración de pares fijada:

```python
client.add_server("archive", archive_transport, allow_legacy=True)
```

Enlace esa opción al comando o punto final configurado. No use una tarjeta salvaje que permita a un servidor arbitrario optar por semántica más débil.`allow_legacy=True`fracasa después de un resultado ambiguo y nunca recibe`initialize`¿ Qué ?

El autor de permisos concede permiso para sondear.`initialize`en el plazo exigido por el transporte, requiere entonces todo lo siguiente:

- un JSON-RPC `2.0`la respuesta con la identificación de la solicitud correspondiente;
- Es exactamente uno .`result`Y no .`error`El artículo 1
- ¿ Qué es esto ?`protocolVersion`en el conjunto de revisiones legales configurados del cliente;
- un objeto valorado `capabilities`campo;
- ¿ Qué es esto ?`serverInfo`objeto con una cadena no vacía `name`y `version`los campos.

Una suspensión de tiempo, cierre de conexión, respuesta de error, resultado malformado, identificación no coincidente o revisión no compatible no se cierra. Sólo un resultado positivo estructuralmente válido selecciona la era heredada. El código pasa `legacy_probe_timeout_ms`a la adaptación de transporte; un verdadero estudio o adaptador HTTP debe hacer cumplir ese plazo en lugar de simplemente registrarlo.

Cache la época seleccionada para el transportista y no vuelva a buscar antes de cada llamada.

### Legacy es una rama de compatibilidad

Una vez que la sonda limitada devuelva pruebas de legado positivo válidas, el cliente utiliza la versión de legado seleccionada exactamente como se define en esa revisión:

1. Verifique el envase de respuesta y la identificación de correlación.
2. Verifique si la revisión negociada está en el conjunto de legado configurado.
3. Registra las capacidades validadas y la identidad del servidor.
4. Envía`notifications/initialized`Sólo después de que todos los cheques pasen.
5. Utilice las formas de solicitud heredadas para esa vida útil de transporte.

Esta rama existe para la interoperabilidad con pares conocidos. No es el diseño predeterminado para nuevos servidores o nuevas solicitudes. Si el transporte se reinicia o su punto final cambia, desecha la caché de la era de pares y negocie de nuevo.

### Herramientas de descubrimiento y almacenamiento en caché

Para cada compañero activo, llame `tools/list`Un resultado moderno incluye:`resultType`¿ Qué ?`ttlMs`, y `cacheScope`. Respecto de la pista de frescura dentro del contexto de la autorización correcta. Recuperación después de la expiración o un evento de cambio de lista suscrito.

Los clientes deben tratar a un desaparecido .`resultType`desde un servidor heredado como `"complete"`No requieren campos de caché modernos para una respuesta de una era anterior negociada.

El servidor debe devolver orden determinista. El cliente también debe ordenar antes de la fusión para que el orden del registro local no dependa del momento de inicio del proceso.

### Fusión de espacios de nombres seguros de colisión

Dos servidores pueden exponer ambos .`search`. Elige una política declarada:

1. **Prefix on collision.**Mantenga el primer nombre canónico y exponga las colisiones posteriores como `<server>/<tool>`¿ Qué ?
2. **Reject on collision.**No cargue el duplicado y no aparezca un error de configuración claro.
3. **Silent overwrite.**Nunca uses esto. Esconde qué servidor recibe una acción seleccionada por el modelo.

Almacenar los nombres canónicos y locales.`tools/call`utiliza el nombre local declarado por el servidor propietario.

### Enrutamiento de una llamada

El enrutamiento es una búsqueda pura:

```text
canonical tool name
  -> peer name + local tool name
  -> new JSON-RPC request id
  -> modern request metadata or explicit legacy shape
  -> matching response id
```

No envíe una llamada cuando el transporte de su propietario no esté disponible.`tools/list`. Las solicitudes modernas perdidas en vuelo en un transporte roto pueden ser retomadas con un nuevo ID JSON-RPC cuando la política de seguridad de la operación lo permita.

### Notificaciones y suscripciones

Los cambios de lista y recursos modernos sólo llegan en un cliente abierto `subscriptions/listen`El cliente envía el filtro de notificación, espera.`notifications/subscriptions/acknowledged`, y correlaciona los eventos con el ID de solicitud de escucha en los metadatos de notificación.

En el momento de desconectar, abra una nueva solicitud de escucha y revise las listas o recursos pertinentes.`Last-Event-ID`¿ Qué ?

### No se iniciaron solicitudes del servidor

Los servidores modernos no llaman al cliente con solicitudes independientes de JSON-RPC para muestreo, elicitación o raíces.`input_required`, y el cliente retoma la solicitud original después de cumplir con las solicitudes de entrada integradas.

No bloquee el lector de respuesta del mismo mientras realiza la entrada. Preserva la correlación y crea un nuevo ID JSON-RPC para la nueva prueba.

```figure
tp-client-merge
```

## Usalo

`code/main.py`El sistema de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de la red de transporte de transporte de la red de transporte de transporte de la red de transporte de transporte de transporte de la red de transporte de transporte de transporte de transporte de la red de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de transporte de

```bash
cd code
python3 main.py
python3 -m unittest discover tests -v
```

Las pruebas demuestran límites que las demostraciones normales pierden:

- las solicitudes modernas repiten metadatos;
- `-32022`retrocederá en el descubrimiento moderno sin inicialización;
- errores modernos reconocidos nunca se rebajan, incluso para un compañero autorizado;
- Los tiempos de interrupción, el cierre de la conexión, las respuestas vacías y errores no reconocidos no se activan `initialize`sin licencia;
- un compañero autorizado se convierte en legado sólo después de un valido, respaldado `initialize`el resultado;
- los resultados de legado malformados y no soportados no permiten que el peer esté disponible;
- se almacenará en caché una era seleccionada con éxito para la vida útil del transporte.

## Envío

Esta lección nos lleva .`outputs/skill-mcp-client-harness.md`. Se plantea el estampado de solicitudes moderno, la negociación de la era de estudio, la fusión determinista del espacio de nombres, el enrutamiento y una rama de compatibilidad heredada cerrada por fallas.

## Los ejercicios

1. Hacer una devolución de servidor falso `-32022`Confirmar que el cliente falla en lugar de enviar`initialize`¿ Qué ?
2. Permite un servidor legado falso, haz que sea limitado `initialize`sondear tiempo fuera, y probar que el par se queda `unknown`y no está disponible.
3. Añadir`cacheScope: "private"`Las listas de herramientas para dos contextos de autorización. Confirmar que el cliente nunca comparte el resultado almacenado de un contexto con el otro.
4. Cambia la política de colisión a rechazo y haz que la inicialización fracase con ambos nombres de pares en el error.
5. Añadir un finito `subscriptions/listen`En la pérdida de flujo, escucha de nuevo con un nuevo ID de solicitud y herramientas de reajuste.

## Términos clave

| Term | Meaning |
|------|---------|
| Peer | Client-side record for one server transport and its discovered data |
| Protocol era | Modern per-request metadata or legacy initialization semantics |
| Discovery probe | Initial `server/discover` used to identify the stdio era |
| Recognized modern error | Error that proves modern behavior and forbids legacy fallback |
| Legacy allowlist | Operator configuration permitting one bounded compatibility probe for a pinned peer |
| Positive legacy evidence | Valid, correlated `initialize` result for an explicitly supported legacy revision |
| Merged namespace | Canonical tool names across all active peers |
| Collision policy | Prefix or reject rule for duplicate tool names |
| Era cache | Selected modern or legacy behavior stored for one transport peer |
| Transport recovery | Restart or reconnect, rediscover, relist, and retry safely with a new id |

## Leer más

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Versioning](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
