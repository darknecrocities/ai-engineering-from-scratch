# Autorizamiento de MCP: CIMD, vinculación del emisor, PKCE y Step-Up

> Una solicitud remota de MCP es estatalizada, pero su autorización no es anónima. Atarde cada credencial al emisor que la creó y cada token al recurso que la recibe.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 15 (security)
**Time:** ~90 minutes

## Objetivos de aprendizaje

- Descubra los servidores de autorización a través de metadatos de recursos protegidos.
- Prefiere Documents de Metadatos de Identificación de Cliente sobre Registro Dinámico de Cliente obsoleto.
- Declarar lo correcto `application_type`cuando sea inevitable una vía de compatibilidad con el DCR.
- Valida la respuesta de autorización `iss`y aislar las credenciales por emisor.
- Utilice PKCE, indicadores de recursos, validación de audiencia y escopo incremental.
- Enviar las solicitudes autorizadas del MCP 2026-07-28 sin sesiones de protocolo.

## El problema

Un servidor MCP remoto puede leer registros privados, escribir sistemas externos o desencadenar trabajos costosos.

- ¿Qué servidor de autorización emitió la credencial?
- ¿Para qué recurso MCP es el token?
- ¿Qué cliente y redirección URI completaron el flujo?
- ¿Qué operaciones aprobó el usuario?
- ¿Este pedido exacto todavía encaja en esa aprobación?

El perfil de autorización 2026-07-28 endurece la inscripción de clientes y el manejo de emisores. Prefiere documentos de metadatos de ID de clientes, deprecia la inscripción dinámica de clientes, requiere el derecho `application_type`en DCR, valida las respuestas de los emisores de la RFC 9207 y prohíbe la reutilización de las credenciales entre los emisores.

Estas reglas complementan el núcleo de los apátridas, no restauran un apretón de manos o un`Mcp-Session-Id`¿ Qué ?

## El concepto

### Conozca los tres papeles

- **MCP client:**Envía solicitudes en nombre de un propietario de recursos.
- **MCP resource server:**acepta el token de acceso y sirve al punto final del MCP.
- **Authorization server:**autentica al propietario del recurso, recauda su consentimiento y emite tokens.

El servidor de recursos y el servidor de autorización pueden operarse juntos, pero mantener sus identificadores y responsabilidades de validación separados.

### La autorización se aplica a HTTP

La especificación de autorización MCP se aplica a los transportes basados en HTTP. Un servidor de estudio local se ejecuta bajo el límite de confianza del proceso y el sistema operativo. No agregue un flujo OAuth falso del navegador al estudio simplemente por simetría.

Para el HTTP remoto Streamable, envíe el token portador en el `Authorization`En cualquier solicitud, nunca lo coloque en la URL.

### Comience con metadatos de recursos protegidos

El servidor de recursos publica metadatos de RFC 9728:

```json
{
  "resource": "https://notes.example.com/mcp",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:delete", "notes:read", "notes:write"]
}
```

El cliente comienza desde la URL de recurso MCP, saca este documento, selecciona un servidor de autorización anunciado y luego saca los metadatos OAuth o OpenID Connect de ese servidor.

Preserva la ruta de recursos al construir la RFC 9728 conocida URL. Para el recurso `https://notes.example.com/mcp`, esta lección utiliza`https://notes.example.com/.well-known/oauth-protected-resource/mcp`- Dejar caer el .`/mcp`el sufijo puede seleccionar metadatos para un recurso protegido diferente del mismo origen.

No adivines el servidor de autorización a partir de un nombre de host. No sigas a un emisor descubierto en un cuerpo de error no validado. Mantén una política en la que el cliente está dispuesto a confiar en los emisores.

### Verificar los metadatos del servidor de autorización

Los metadatos deben exponer los puntos finales y los controles soportados:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "code_challenge_methods_supported": ["S256"],
  "authorization_response_iss_parameter_supported": true,
  "client_id_metadata_document_supported": true
}
```

Requerir S256 para PKCE. Registrar la cadena exacta de emisor. Ese valor exacto se convierte en la clave para el registro y almacenamiento de tokens.

### Seguir la prioridad de registro

Utilice la información del cliente pre-registrada cuando el cliente ya tenga una relación explícita con el emisor seleccionado. De lo contrario, prefiere Documentación de Metadatos de ID del cliente cuando el servidor de autorización anuncia soporte. Utilice DCR solo como la retroceso de compatibilidad obsoleta, luego solicite información del cliente si ninguno de esos mecanismos está disponible.

### Preferir documentos de metadatos de ID del cliente

Un documento de metadatos de ID de cliente le da al servidor de autorización una URL HTTPS que es tanto el identificador del cliente como la ubicación de sus metadatos:

```json
{
  "client_id": "https://client.example.com/oauth/metadata.json",
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

El servidor de autorización recoge y valida el documento.`client_id`Debe ser una URL HTTPS con una ruta, y el valor dentro del documento debe ser igual a esa URL exactamente.`client_id`¿ Qué ?`client_name`, y `redirect_uris`- ¿ Qué ?`application_type`El nuevo uso obligatorio de la misma es específicamente el camino de DCR.

Trate la obtención del documento como una operación sensible a SSRF. Resolver y validar el destino, rechazar direcciones de recorrido, privadas, locales de enlace y de otro modo no permitidas, volver a comprobar después de redirecciones y cambios en DNS, limitar redirecciones, bytes y tiempo, requiere JSON y solo de acuerdo con controles de caché HTTP validados. Trate `client_name`y otros campos de visualización como texto no confiable.

CIMD elimina la necesidad de imprimir un nuevo identificador dinámico para cada primer contacto. No elimina la validación de URI redirigida, la política del emisor o el consentimiento del usuario.

### DCR es un camino de compatibilidad

El registro de clientes dinámico sigue disponible para servidores de autorización antiguos, pero se ha desactualizado para las nuevas implementaciones de MCP.

Cuando utilice DCR, debe declarar `application_type`¿Qué es esto ?

```json
{
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

- Utilización de escritorio, móvil, línea de comandos y clientes de loopback `native`¿ Qué ?
- Uso de aplicaciones de navegador alojadas remotamente `web`y redirecciones remotas HTTPS.

Salvar el campo puede ser por defecto `web`en una implementación de registro OpenID Connect y hacer una redirección de loopback legítima falla.

Mantenga el código DCR detrás de una decisión de retroceso explícita. No retroceda silenciosamente después de un fallo de validación CIMD arbitrario. Eso podría convertir un fallo de seguridad en una ruta de inscripción más débil.

### Las credenciales vinculativas al emisor

Almacenar material de inscripción con moneda del emisor bajo el emisor exacto:

```text
issuer_credentials[issuer] = pre_registered_or_dcr_client
tokens[(issuer, resource)] = access_token
```

Si el descubrimiento de recursos protegidos cambia desde `https://auth-one.example`¿ Qué ?`https://auth-two.example`, reevaluar la confianza. Nunca envíe el secreto del cliente del primer emisor, la identificación del cliente DCR, el token de acceso de registro, el token de actualización o el token de acceso al segundo.

Un ID de cliente de CIMD es diferente porque es una URL HTTPS auto-alojada, no una credencial acuñada por un servidor de autorización. La misma URL de CIMD es portátil: un nuevo emisor de confianza saca y valida el documento sin volver a registrarse en DCR. Las respuestas y tokens de autorización aún se validan y almacenan bajo el nuevo emisor.

### Código de autorización con PKCE

El flujo interactivo es:

1. Generar una alta entropía `code_verifier`¿ Qué ?
2. Derivar el S256 `code_challenge`¿ Qué ?
3. Envía la solicitud de autorización con exactitud `client_id`¿ Qué ?`redirect_uri`¿ Qué ?`scope`¿ Qué ?`code_challenge`, y `resource`¿ Qué ?
4. Recibir una respuesta de autorización que contenga `code`y, cuando se haya proporcionado, `iss`¿ Qué ?
5. Validación`iss`contra el emisor registrado exacto antes de utilizar cualquier campo de respuesta.
6. Cambiar el código con `code_verifier`, el mismo redireccionamiento de URI, y el mismo `resource`¿ Qué ?
7. Guarde el token resultante en la sección `(issuer, resource)`¿ Qué ?

El `resource`El parámetro de RFC 8707 aparece tanto en las solicitudes de autorización como en las de tokens.

### Validación`iss`exactamente

La RFC 9207 evita que una respuesta de autorización de un emisor se confunda con una respuesta de otro.

¿ Cuándo ?`iss`Si el código está presente, compare con el emisor registrado sin doblar el caso, cambios de la columna de seguimiento, eliminación de puerto predeterminado o normalización de codificación por ciento. En caso de desajuste, no actúe sobre el código ni incluso muestre detalles de error controlados por el atacante de esa respuesta.

Un servidor de autorización que incluye `iss`publicidad `authorization_response_iss_parameter_supported: true`Los clientes actuales aún validan un regalo .`iss`Incluso cuando ese anuncio no está.

### Valida la audiencia en el servidor MCP

El servidor de recursos sólo acepta tokens emitidos por sí mismo:

```text
token.issuer == configured_authorization_server
token.audience == canonical_mcp_resource
```

Los tokens inválidos, vencidos, de emisor incorrecto o de audiencia incorrecta reciben 401. El servidor MCP no debe aceptar ni transitar un token destinado a otro servicio.

### Solicitar el menor alcance de corriente

Si una herramienta posterior requiere más, el servidor devuelve 403 con un desafío de alcance autorizado:

```text
WWW-Authenticate: Bearer error="insufficient_scope",
  scope="notes:delete",
  resource_metadata="https://notes.example.com/.well-known/oauth-protected-resource/mcp"
```

El cliente explica el nuevo permiso, obtiene el consentimiento, realiza un nuevo flujo de autorización con el conjunto de alcance combinado y vuelve a probar la solicitud de MCP con un nuevo ID JSON-RPC.

No asuma que el ámbito de aplicación cuestionado es un subconjunto de `scopes_supported`El desafío es autoritario para la operación actual.

### Autorización y cable de MCP sin estatus

Una llamada de herramienta autorizada todavía lleva el envase completo de la solicitud actual:

```text
POST /mcp
Authorization: Bearer <access-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.delete
```

```json
{
  "jsonrpc": "2.0",
  "id": 12,
  "method": "tools/call",
  "params": {
    "name": "notes.delete",
    "arguments": {"id": "note-7"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "oauth-lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

El token autoriza al principal, los metadatos de la solicitud negocian el comportamiento del protocolo, ninguno sustituye al otro.

Valida el cable en un orden fijo: tipos de JSON-RPC y metadatos, igualdad de encabezado y cuerpo, luego soporte de protocolo. Una desajuste de enrutamiento o encabezado de versión devuelve HTTP 400 con `-32020`Si el encabezado y el cuerpo coinciden en una versión no compatible, devuelva HTTP 400 con `-32022`y `data`Es exactamente .`{"supported":["2026-07-28"],"requested":"<actual>"}`Un método desconocido devuelve HTTP 404 con `-32601`¿ Qué ?

Cada error de solicitud, incluido 401 tokens inválidos y 403 alcance insuficiente, es un envase de error JSON-RPC con la solicitud original `id`. La información de recuperación estructurada pertenece a un error opcional `data`¿ Qué es ?`WWW-Authenticate`El título de respuesta HTTP sigue siendo el de una notificación.`id`Una notificación HTTP aceptada devuelve 202 con un cuerpo vacío.

El servidor implementa `server/discover`y publicitarias, por lo que también implementa el obligatorio`tools/list`Los descriptores de herramientas tienen nombres, descripciones y raíz de objeto estables.`inputSchema`La lista es determinista y devuelve `resultType`, metadatos de identidad del servidor, un límite `ttlMs`, y `cacheScope`• El descubrimiento y una lista de herramientas independientes del usuario pueden estar disponibles antes de la autorización.

### No hay pasaportes simbólicos

Un servidor MCP no debe reenviar el token de acceso MCP del cliente a una API en aguas subidas. Obtener un token en aguas subidas separado con el público adecuado o usar un diseño explícito de intercambio de tokens. La validación de audiencia solo funciona cuando los servicios rechazan tokens acuñados para otra persona.

### Tokens de actualización

Los tokens de actualización son opcionales. Cuando se emiten, guardanlos confidencialmente y los contratan por emisor y recurso. No asuma que existen.

```figure
t3-scope-stepup
```

## Construye el mismo

`code/main.py`Es un protocolo en proceso y simulador de autorización. Implementa el descubrimiento de recursos protegidos, los metadatos del servidor de autorización, la inscripción en CIMD, la retroceso de DCR con versiones, las comprobaciones de tipos de aplicaciones, PKCE, validación de emisores, tokens vinculados a recursos, aumento de alcance, `server/discover`¿ Qué ?`tools/list`, y una solicitud de herramienta sin estado.

El modelo recibe cuerpos de solicitud analizados y encabezados de enrutamiento.`Content-Type`o `Accept`. Conectarlo al adaptador HTTP de la Lección 09, que requiere `Content-Type: application/json`y un `Accept`valor que contiene ambos `application/json`y `text/event-stream`¿ Qué ?

- ¿Qué quieres decir ?

```bash
cd phases/13-tools-and-protocols/16-mcp-security-oauth-2-1
python3 code/main.py
python3 -m unittest discover code/tests -v
```

La salida muestra el descubrimiento primero, la inscripción en CIMD, una lectura ordinaria, dos pasos separados de alcance y el almacenamiento de credenciales con llave de emisor.

## Usalo

Mapa de los objetos del simulador a los componentes de producción:

- `ResourceServer.protected_resource_metadata`se convierte en el punto final de la RFC 9728.
- `AuthorizationServer.metadata`se convierte en RFC 8414 o OpenID Connect descubrimiento.
- `Client.enroll`se convierte en resolución CIMD más una rama explícita de compatibilidad DCR.
- Credenciales de clientes emitidas por el emisor y `tokens_by_issuer_resource`Una URL CIMD puede permanecer portátil mientras sus resultados de autorización permanezcan vinculados al emisor.
- `ResourceServer.handle`Se convierte en un middleware que valida los encabezados actuales de MCP, el token y el alcance de las herramientas antes de enviar mientras se mantiene cada error de solicitud en un envase JSON-RPC correspondiente.

## Envío

Esta lección nos lleva .`outputs/skill-oauth-scope-planner.md`. Ahora se diseña la prioridad de inscripción, el almacenamiento de credenciales vinculadas al emisor, el tipo de solicitud, PKCE, los indicadores de recursos, los desafíos de alcance y el límite actual de las solicitudes de apatridia.

## Los ejercicios

1. Añadir la rotación de la ficha de actualización y rechazar la reutilización de la ficha de actualización anterior.
2. Añadir una lista de permisos de emisor. Al cambiar de emisor, reutilice solo una URL CIMD portátil; rechace todas las credenciales y tokens emitidas anteriormente por el emisor.
3. Añadir una caducidad a los códigos de autorización y confirmar que un intercambio tardío falla.
4. Construir una variante del cliente web con una redirección remota HTTPS y comparar sus metadatos DCR con el cliente nativo.
5. Añadir un segundo recurso bajo el mismo emisor. Confirmar su token de acceso no se puede utilizar en el primer recurso.

## Términos clave

| Term | Meaning |
|------|---------|
| Protected-resource metadata | RFC 9728 document that identifies the resource and authorization servers |
| CIMD | HTTPS metadata document whose URL is the OAuth client identifier |
| DCR | Deprecated dynamic client enrollment retained for compatibility |
| `application_type` | `native` or `web`, used to validate redirect URI rules |
| PKCE | Verifier and S256 challenge that protect an intercepted authorization code |
| `iss` | RFC 9207 authorization response issuer identifier |
| Resource indicator | RFC 8707 parameter that binds a token request to an MCP resource |
| Audience | Resource for which a token is valid |
| Step-up | New consent and token issuance for an additional current-operation scope |
| Issuer-bound credentials | Registration and token records isolated by exact authorization server issuer |

## Leer más

- [MCP 2026-07-28 authorization specification](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
- [RFC 9728: OAuth 2.0 Protected Resource Metadata](https://www.rfc-editor.org/rfc/rfc9728)
- [RFC 8707: Resource Indicators for OAuth 2.0](https://www.rfc-editor.org/rfc/rfc8707)
- [RFC 9207: OAuth 2.0 Authorization Server Issuer Identification](https://www.rfc-editor.org/rfc/rfc9207)
- [OAuth Client ID Metadata Document draft](https://datatracker.ietf.org/doc/draft-ietf-oauth-client-id-metadata-document/)
