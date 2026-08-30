# Seguridad de MCP: Metadatos envenenados, enrutamiento y estado de MRTR

> Sin estatus no significa que no tenga confianza, significa que cada solicitud expone la evidencia que un servidor y una puerta de entrada necesitan para validar la llamada de forma independiente.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~60 minutes

## Objetivos de aprendizaje

- Trate las descripciones de las herramientas, las anotaciones, la información del cliente y la información del servidor como datos no confiables.
- Detecta envenenamiento de metadatos, cambios en el descriptor y colisiones de nombres entre servidores.
- Valida los metadatos de la solicitud 2026-07-28 y los encabezados de enrutamiento HTTP en streaming.
- Proteja el MRTR `requestState`contra la manipulación y vincular la confirmación a los argumentos exactos.
- Aplicar límites de autorización y tarifa a un principal, no una sesión de protocolo eliminado.

## El problema

Un modelo lee las descripciones de herramientas para decidir qué llamar. Un router lee los nombres de herramientas para decidir a dónde enviar una solicitud. Un usuario lee las etiquetas para decidir qué aprobar. Un descriptor malicioso puede atacar a los tres.

La guía oficial de seguridad de MCP es directa: las descripciones y anotaciones deben ser tratadas como no confiables a menos que provengan de un servidor de confianza. Incluso entonces, la confianza de implementación puede cambiar. Una actualización del servidor, un paquete comprometido, un error de registro o una fusión de pasarelas pueden alterar lo que el modelo ve.

El protocolo actual también cambia los límites de seguridad. En 2026-07-28 no hay apretón de manos y ninguna sesión de transporte.`Mcp-Session-Id`no es un diseño actual.

## El concepto

### Siete superficies de ataque que vale la pena comprobar

Utilice una lista concreta en lugar de las vagas instrucciones para tener cuidado.

1. **Metadata poisoning.**Una descripción contiene instrucciones no relacionadas con el comportamiento declarado de la herramienta.
2. **Descriptor rug pull.**Cambios de nombre, descripción, esquema o anotación previamente aprobados.
3. **Cross-server shadowing.**Dos fondos exponen el mismo nombre de herramienta no calificado y el enrutamiento elige uno en silencio.
4. **Header and body confusion.** `Mcp-Method`o `Mcp-Name`no está de acuerdo con la solicitud JSON-RPC.
5. **Capability escalation.**Un peer reclama una extensión o característica cliente y el servidor comete errores en esa declaración de autorización.
6. **MRTR state tampering.**Un cliente cambia`requestState`, responde a una pregunta diferente, o reutiliza la confirmación con diferentes argumentos.
7. **Supply-chain identity confusion.**Un nombre de pantalla familiar se trata como prueba de identidad del editor o del servidor.

Las superficies se superponen. El encapsulado de hash ayuda con los cambios de descriptor pero no prueba que el primer descriptor fuera seguro. El escaneo estático capta frases obvias pero no instrucciones sutiles. El espaciamiento de nombres evita una clase de colisión pero no un servidor malicioso con espacios de nombres.

### El envase de solicitud actual es evidencia, no identidad

Cada solicitud de 2026-07-28 contiene:

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "elicitation": {"form": {}}
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "security-lab",
      "version": "1.0.0"
    }
  }
}
```

Valida la versión y la forma de capacidad en cada solicitud. Utilice las capacidades para elegir una forma de respuesta compatible. No use `clientInfo`Se declara como un principal autenticado.

La misma advertencia se aplica a `io.modelcontextprotocol/serverInfo`En el caso de los registros, el registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de registro de datos de datos de registro de datos de registro de datos de registro de datos de registro de datos de registro de datos de datos de registro de datos de registro de datos de registro de datos de datos de registro de datos de datos de registro de datos de datos de registro de datos de datos de registro de datos de datos de datos de registro de datos de datos de registro de datos de datos de datos de registro de datos de datos de registro de datos de datos de registro de datos de datos de registro de datos de datos de registro de datos de datos de datos de datos de datos de datos de registro de datos de datos de datos de registro de datos de datos de datos de datos de registro de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos

### Valida el enrutamiento antes de la política

Para`tools/call`, HTTP transmitible incluye:

```text
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.export
```

El método de encabezado debe ser igual al método del cuerpo.`params.name`Rechazar el desacuerdo con`-32020`antes de seleccionar un backend, aplicar RBAC o consumir un token de límite de tasa.

Este orden cierra una ambigüedad común: un componente autoriza el cuerpo mientras que otro se dirige por el encabezado.

La validación por cable sigue una secuencia exacta. Valida los tipos de JSON-RPC y metadatos, compara los valores de encabezado con el cuerpo, luego compruebe si la versión coincidente es compatible. Un encabezado no coincidente devuelve HTTP 400 con `-32020`Si el encabezado y el cuerpo coinciden en una versión no compatible, devuelva HTTP 400 con `-32022`y `data`Es exactamente .`{"supported":["2026-07-28"],"requested":"<actual>"}`Un método desconocido devuelve HTTP 404 con `-32601`¿ Qué ?

Cada objeto de error incluye opcional `data`Cuando el contrato requiere información estructurada de recuperación, una notificación no tiene`id`, por lo que nunca recibe un éxito JSON-RPC o respuesta de error. Una notificación HTTP aceptada devuelve 202 con un cuerpo vacío.

### Enfilar todo el descriptor

Un hash de descripción solo pierde cambios de esquema y anotación. Canonizar y hash los campos de descriptor aprobados por el usuario:

```python
normalized = json.dumps(tool, sort_keys=True, separators=(",", ":"))
digest = hashlib.sha256(normalized.encode()).hexdigest()
```

Guarde el archivo bajo una llave calificada como `notes.export`, junto con la evidencia de la editorial y el tiempo de aprobación fuera de este ejemplo de juguete.

En cada refresco:

- La clave desconocida: cuarentena hasta la revisión.
- La misma llave, diferente digestión: cuarentena como un tirón de alfombra hasta que sea reaprobado.
- Nombre duplicado no calificado: requiere espaciado de nombres determinista.
- El botón del escáner: bloquear y revisar el descriptor completo.

La igualdad de hash prueba estabilidad, no seguridad.

### El escaneo estático es un cable triplicado

Los patrones simples pueden marcar etiquetas de rol, pasos de instrucción, ocultación, acceso secreto y destinos de red ocultos. Son lo suficientemente baratos para el tiempo de instalación e información informática.

Una descripción segura puede contener una frase marcada en una advertencia legítima. Una descripción maliciosa puede evitar cada frase. Trata la salida del escáner como evidencia de revisión, no una puntuación automática de inocencia.

### Espacio de nombres antes de la fusión

Supongamos que dos servidores exponen ambos .`search`Nunca dejes que la orden del descubrimiento decida cuál gana.

```text
notes.search
issues.search
```

El nombre calificado es el nombre de la puerta de entrada pública. Registra el mapeo de backend por separado. Los nombres estables hacen aprobación, auditoría, pines hash, y `Mcp-Name`el enrutamiento se refiere al mismo objeto.

### Las capacidades son declaraciones de compatibilidad

Por petición `clientCapabilities`El protocolo de acceso no permite que el cliente tenga acceso a herramientas, datos o acciones.

La autorización sigue procedente de la política de capital y recursos autenticada.

1. Autentica las credenciales de transporte.
2. Valida la versión, los encabezados y la forma de solicitud.
3. Verifique la compatibilidad de las capacidades.
4. Autorice el principal, la herramienta, el recurso y los argumentos.
5. Ejecutar o solicitar la entrada del usuario.

### Proteja la confirmación de MRTR sin estado

Una herramienta consecuente puede necesitar una confirmación del usuario. El MCP actual utiliza Solicitudes de viaje de ida y vuelta múltiples en lugar de una llamada de regreso de servidor a cliente.

Primera respuesta:

```json
{
  "resultType": "input_required",
  "inputRequests": {
    "confirm": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Export notes to archive?",
        "requestedSchema": {
          "type": "object",
          "properties": {
            "confirm": {"type": "boolean"}
          },
          "required": ["confirm"]
        }
      }
    }
  },
  "requestState": "opaque-integrity-protected-value"
}
```

El cliente obtiene la entrada y vuelve a probar el método original con un nuevo ID JSON-RPC:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes.export",
    "arguments": {"query": "private", "destination": "archive"},
    "requestState": "opaque-integrity-protected-value",
    "inputResponses": {
      "confirm": {
        "action": "accept",
        "content": {"confirm": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

Cada uno .`inputRequests`valor es una solicitud integrada completa con `method`y `params`Su clave debe coincidir con la entrada correspondiente en `inputResponses`Una elicitación de formulario utiliza una raíz de objeto`requestedSchema`, y el cliente debe haber declarado la capacidad de elicitación de formulario antes de que el servidor lo solicite.

La capacidad actual tiene dos declaraciones de formulario válidas. `{"elicitation":{}}`La Comisión ha adoptado una decisión sobre la aplicación de la ley de derechos humanos.`{"elicitation":{"form":{}}}`Una declaración de URL sólo como `{"elicitation":{"url":{}}}`El servidor devuelve HTTP 400 con `-32021`y `data.requiredCapabilities`igual a `{"elicitation":{"form":{}}}`¿ Qué ?

Tratar`requestState`El código de lección utiliza HMAC y el argumento exacto para hacer visible el límite.

El libro mayor no debe vivir dentro de un objeto de puerta de entrada. El modelo ejecutable inyecta una tienda de reproducción limitada, recortada TTL que puede ser compartida por múltiples instancias de puerta de entrada. Su reclamo atómico es el límite de ejecución: solo una aceptación validada o una disminución terminal explícita consume estado. Una respuesta malformada o `cancel`La flota de producción necesita la misma reclamación condicional en el almacenamiento duradero compartido.

No almacenar el contexto oculto de confirmación en una sesión de protocolo. Cualquier instancia de servidor debe ser capaz de validar la reutilización.

### Regla de dos para las llamadas de alto riesgo

Clasificar una llamada a lo largo de tres ejes:

- Consume entradas no confiables.
- Puede acceder a datos sensibles.
- Esto provoca una acción externa consecuente.

Un solo paso automático no debe combinar los tres. Dividirlo, reducir los privilegios o solicitar la entrada explícita del usuario a través de MRTR. Esta es una heurística de diseño, no una capacidad de protocolo.

### Reducir la autoridad antes de la ejecución

La falta de estatus por sí sola no es seguridad. Elimina el historial oculto del protocolo, pero una solicitud independiente todavía puede pedir a un controlador con más poder que filtra datos o haga un cambio irreversible. La seguridad proviene de reducir la autoridad en cada frontera:

1. **Typed verb.**Exponer una operación limitada como `archive_note`, no un genérico .`run`o `request`instrumento que pueda expresar poderes no relacionados.
2. **Validated arguments.**Utilice un esquema cerrado donde sea práctico, rechace campos desconocidos, normalice identificadores una vez, tamaños de límites y valide el destino, el inquilino y la propiedad de los recursos antes de la evaluación de la política.
3. **Current authorization.**Enlace el principal autenticado al verbo exacto, el recurso, el entorno y los argumentos normalizados.
4. **Action-bound approval.**Para una llamada consecuente, vincular la aprobación a un digesto del verbo escrito y los argumentos normalizados, además de la política principal, de vencimiento y de una sola vez. Cualquier campo cambiado requiere una nueva decisión.
5. **First-class refusal.**Rechazar el modelo, la aprobación expirada, el rechazo de los usuarios y el destino inseguro como resultados ordinarios que no ejecutan ningún efecto secundario.
6. **Redacted audit evidence.**Registra quién preguntó, qué descriptor y versión de política admitida se utilizaron, qué objetivo normalizado fue autorizado, por qué la decisión permitió o rechazó, y si la ejecución comenzó.

Cada paso restringe lo que puede hacer el siguiente componente. El procesador final debe recibir un comando de dominio ya validado, no texto de modelo crudo más amplias credenciales. Repita toda la cadena en un retraso MRTR, actualización de tareas o llamada de pasarela. Una aprobación anterior no convierte las solicitudes posteriores en tráfico de sesión de confianza.

### Caminos de interacción actuales y antiguos

Roots, Sampling y Logging se deprecian para las nuevas implementaciones 2026-07-28. Un gateway puede conservar el código de canal de solicitud más antiguo solo como una ruta de compatibilidad con versiones.

No construya una nueva defensa alrededor de un limitador de muestreo por sesión. Aplique cuotas a la ventana de tiempo, de capital, de emisor, de recurso, de herramienta y de tiempo autenticados. Para el trabajo interactivo actual, inspeccione las solicitudes de entrada y respuestas de MRTR.

### Control de transporte de personas sin nacionalidad

- Acceptar los mensajes modernos de MCP en el único punto final de POST.
- Retorna 405 para GET y DELETE modernos.
- No se acuña ni dependa de `Mcp-Session-Id`¿ Qué ?
- Ignora la sesión heredada y repite los encabezados como entradas de autoridad.
- Regresa JSON o SSE escaneado por solicitud para ese POST.
- Usar`subscriptions/listen`únicamente para las notificaciones de cambios de larga duración que hayan sido aceptadas.

```figure
tp-tool-poisoning
```

## Construye el mismo

`code/main.py`Implementa un pequeño modelo de puerta de enlace de seguridad en proceso. Canonicizan y pinen descriptores de herramientas completas, informan de intoxicación y sombreamiento de metadatos, validan el envelope de solicitud moderno y los valores de enrutamiento, y realizan una exportación confirmada en dos rondas con firmas `requestState`y una tienda de reproducción compartida inyectada.

El modelo comienza después de que un adaptador HTTP haya analizado el cuerpo JSON y los encabezados de enrutamiento.`Content-Type`o `Accept`. Conectar el mismo despachador al adaptador HTTP Streamable completo de la Lección 09 , lo que requiere `Content-Type: application/json`y un `Accept`valor que contiene ambos `application/json`y `text/event-stream`¿ Qué ?

- ¿Qué quieres decir ?

```bash
cd phases/13-tools-and-protocols/15-mcp-security-tool-poisoning
python3 code/main.py
python3 -m unittest discover code/tests -v
```

La muestra muta intencionalmente un descriptor.El escáner y la comparación de digestión producen hallazgos independientes.`input_required`respuesta y retrocesión sin estatus.

## Usalo

Reemplazar`SAFE_TOOLS`Con una instantánea normalizada de sus propios servidores aprobados. Mantenga las credenciales y secretos fuera de la instantánea. Revise cada descriptor nuevo o cambiado antes de actualizar su digesto.

En una puerta de entrada, ejecuta las mismas comprobaciones durante el descubrimiento y de nuevo antes de enviar. Una caché puede reducir el trabajo de descubrimiento, pero una aprobación en caché debe expirar o ser invalidada cuando el descriptor cambie.

## Envío

Esta lección nos lleva .`outputs/skill-mcp-threat-model.md`Produce un modelo de amenaza de protocolo de corriente en metadatos, enrutamiento, capacidad, autorización, MRTR, almacenamiento en caché, registro y límites de compatibilidad.

## Los ejercicios

1. Enlazar la decisión de autorización principal y actual autenticada con el estado MRTR sellado, y luego rechazar una nueva prueba bajo un principal diferente.
2. Reemplazar la memoria de repetición con un inserto condicional persistente y demostrar que dos procesos no pueden reclamar un solo nonce.
3. Inyectar una falla después de la reclamación de repetición pero antes de una exportación simulada. Definición y prueba la regla de transacción o idempotencia que hace que la recuperación sea segura.
4. Cambiar las herramientas `inputSchema`Confirme que el pin de descriptor entero lo capta.
5. Añadir una política que rechaza la caché pública cuando `tools/list`Se diferencia por el principal.
6. Modela un servidor más antiguo detrás de la puerta de entrada.`2025-11-25`el ramo de compatibilidad.

## Términos clave

| Term | Meaning |
|------|---------|
| Metadata poisoning | Instructions or deceptive claims embedded in a tool descriptor |
| Rug pull | Change to a previously approved descriptor |
| Tool shadowing | Ambiguous routing caused by duplicate unqualified names |
| Header mismatch | Routing header and JSON-RPC body disagreement, error `-32020` |
| Hash pin | Digest of the complete approved descriptor |
| MRTR | Stateless response and retry pattern for server-requested input |
| `requestState` | Opaque round-trip value that must be treated as untrusted input |
| Capability declaration | Statement of protocol compatibility, not authorization |
| Implicit form support | An empty `elicitation` capability object, equivalent to form support |
| Qualified tool name | Stable gateway name such as `notes.search` |

## Leer más

- [MCP security and trust guidance](https://modelcontextprotocol.io/specification/2026-07-28#security-and-trust--safety)
- [Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Streamable HTTP transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Deprecated features](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)
