# Cadena de suministro de registros de MCP: admisión, derivación y retroceso

> Una entrada en el registro le dice lo que un editor declaró.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 17 (gateways and registries), Phase 13 · 18 (production authentication)
**Time:** ~90 minutes

## Objetivos de aprendizaje

- Publicación del Registro separada, procedencia del paquete, descubrimiento en tiempo de ejecución y aprobación local.
- Verifique un espacio de nombres del servidor MCP sin confiar en el nombre dentro de su propio registro.
- Pin publicación inmutable, fuente de ejecución, procedencia y evidencia en vivo del descriptor.
- Detectar cambios en el estado del registro y la deriva del tiempo de ejecución después de la admisión.
- Volver el enrutamiento a una versión previamente admitida sin reescribir el historial.
- Mantenga un libro de admisión que explique cada decisión.

## El problema

¿ Qué encuentras ?`com.example/inventory`Su descripción se ve correcta, su paquete existe, el servidor responde.`server/discover`¿ Qué ?

No es un hecho, sino una cadena de hechos de diferentes autoridades:

1. Un editor autenticado para un espacio de nombres presentó un registro.
2. Un registro de paquetes servía a un artefacto con una identidad y digestión específicas.
3. Un punto final en ejecución informó una versión del protocolo, capacidades, herramientas e información del servidor de diagnóstico.
4. Su organización decidió que esta combinación exacta era permitida.

Una publicación válida puede ser desactualizada. Una etiqueta de paquete puede apuntar a un artefacto inesperado si no se pincha su digesto. Un servidor puede agregar una herramienta destructiva después de la revisión. Una copia de seguridad puede elegir silenciosamente una versión que nunca fue admitida.

El fijo es un controlador de admisión con pruebas en cada frontera.

## El registro es un índice, no su sistema de aprobación

El Registro oficial de MCP almacena los metadatos del servidor.`server.json`Registro de nombres de una versión del servidor y declara uno o más paquetes o puntos finales remotos. reglas de publicación añaden autenticación del espacio de nombres, comprobantes de propiedad del paquete, reglas de registro restringidas y una ubicación de metadatos de editor estrecha.

En el caso de los controles, las preguntas de publicación son respondidas.

| Boundary | Question | Evidence owner |
|---|---|---|
| Namespace | Was the publisher allowed to use this name? | Registry authentication plus your verified namespace input |
| Record | What did the publisher declare for this version? | Immutable `server.json` digest |
| Execution source | Which package or remote endpoint will execute? | Declared source fields, verified ownership result, transport, and trusted digest |
| Runtime | What does the endpoint expose now? | `server/discover` and tool descriptors |
| Admission | Did your policy approve this exact set? | Local pin and ledger entry |
| Operations | Is it still safe, and what can replace it? | Drift checks, status sync, health, and rollback route |

La versión del esquema de Registro y la versión del protocolo MCP son independientes.`2025-12-11`esquema del servidor mientras el servidor en vivo admite MCP `2026-07-28`Nunca deducir uno del otro.

```figure
mcp-registry-admission
```

## Siete controles en una sola decisión de admisión

### 1. Verificación del espacio de nombres

Los nombres oficiales del Registro utilizan espacios de nombres autenticados. Un dominio verificado puede mapear a un prefijo de dominio invertido. Por ejemplo, el control de `example.com`puede establecer `com.example/*`¿ Qué ?

No acepte una verificación de prefijos de cadena:

```python
server_name.startswith("com.example")
```

Eso también acepta .`com.exampleevil/tool`. Divide el nombre en `/`, requieren una bala no vacía, y comparar el segmento del espacio de nombres exactamente. Lo más importante, pasar el espacio de nombres verificado en la admisión del resultado de autenticación. No derivar confianza del registro no confiable.

Los espacios de nombres respaldados por GitHub y los espacios de nombres respaldados por dominios utilizan diferentes caminos de autenticación. Normaliza cualquiera de los caminos en una entrada de admisión: la cadena de espacio de nombres verificada exacta.

### 2. Provenencia de la unión

Para un registro de paquete, la declaración y el artefacto recogido deberán unirse en campos explícitos:

- tipo de registro de paquetes
- Identificador del paquete
- versión del paquete
- Resultado verificado de propiedad
- Digest de artefactos descargados

También valida el transporte declarado del paquete. Un registro con solo un punto final remoto es válido y no puede ser rechazado por falta de un paquete. Para una fuente remota, unirse a la URL declarada y el tipo de transporte a la propiedad del punto final verificada de forma independiente y a un digesto de la conexión de confianza o pruebas de implementación.

El código de lección soporta ambos tipos de fuente y hashes la fuente seleccionada junto con la fuente del Registro, el nombre del servidor, la versión del Registro, el registro de registro y el registro de evidencia.

Nunca acepte un documento que sólo sea proporcionado por el artefacto que usted está tratando de verificar, pero calculelo en un límite de compra de confianza o recibalo de un servicio de paquetes cuyo resultado de verificación usted valida.

### 3. Enlace la decisión, no sólo la versión

Las versiones del Registro son identificadores únicos de publicación. Los metadatos publicados son inmutables. Un registro cambiado requiere una nueva versión. Se recomienda versionar semánticamente, pero el Registro no lo requiere y no acepta rangos de versiones.

Esto significa que`^1.4`No es un pin de admisión. Ni es lastest. Un pin útil contiene:

```json
{
  "server": "com.example/inventory",
  "version": "1.0.0",
  "recordDigest": "...",
  "source": {"kind": "package", "registryType": "pypi"},
  "sourceDigest": "...",
  "toolsetDigest": "...",
  "provenanceDigest": "...",
  "registryStatus": "active"
}
```

El enchufe de varias capas permite identificar qué límite cambió. Un cambio de registro de digestión bajo la misma versión de Registro es un fallo de integridad del Registro. Un cambio de digestión de origen bajo la misma coordenada del paquete o implementación remota es un fallo de integridad de la fuente de ejecución. Un cambio de digestión de conjunto de herramientas es la deriva de tiempo de ejecución.

### 4. Detección de deriva en vivo

La entrada debe observar el servidor que realmente recibirá el tráfico.`server/discover`, lista o obtenga de otra manera los descriptores de herramientas expuestos a través de su camino de confianza, y verifique:

- `2026-07-28`¿ Está en ?`supportedVersions`
- todas las capacidades requeridas localmente están presentes
- Cada descriptor de herramientas tiene la superficie de identidad y esquema requerida
- el digesto de descriptor normalizado coincide con el pin admitido en las comprobaciones posteriores

El resultado opcional `_meta["io.modelcontextprotocol/serverInfo"]`El valor es un contexto de visualización, registro y depuración auto-relatado. Regístalo como evidencia de diagnóstico, pero nunca lo use para establecer el espacio de nombres, propiedad del paquete, propiedad del punto final, admisión o cualquier otra decisión de seguridad.`serverInfo`alias afuera `_meta`no es el campo de contratación y no debe ser promovido a prueba de diagnóstico.

Normaliza sólo los campos cuyo orden no tiene significado. La muestra clasifica la lista de herramientas por nombre estable antes de hash, por lo que un cambio inofensivo en el orden de la lista no causa derivación. No descarta los campos de descripción. Una nueva herramienta, un esquema cambiado, una descripción cambiada o nuevas anotaciones cambian el pin.

La muestra trata descriptores malformados y cualquier cambio de digestión de descriptor como deriva, cuarentena el pin, elimina su ruta activa y bloquea esa versión como un objetivo de retroceso. Una política de producción puede permitir un cambio editorial solo a través de una nueva revisión, porque las descripciones influyen en la selección de herramientas de modelo.

### 5. El estado del registro es estado en vivo

La API del Registro adjunta un nivel de respuesta `_meta`Objeto junto a cada registro del servidor.`_meta["io.modelcontextprotocol.registry/official"]`- Pasen la respuesta .`_meta`objetar la admisión y leer `_meta["io.modelcontextprotocol.registry/official"].status`- Un directo .`_meta.status`El valor no es la forma oficial del cable.`_meta`El estado puede ser:

- `active`: devuelto por defecto y elegible para la admisión local
- `deprecated`: todavía se puede descubrir con una advertencia, pero ya no es una opción automática segura
- `deleted`: oculto por defecto mientras su registro histórico permanece disponible a través de vistas eliminadas o incrementales

Sincronización de estado después de la admisión. Si una versión activa se vuelve obsoleta o eliminada, cuarentena su pin y deje de enrutar nuevo trabajo a ella. Guarde la evidencia.

Los metadatos personalizados proporcionados por el editor pertenecen sólo a la sección `_meta.io.modelcontextprotocol.registry/publisher-provided`Los datos de respuesta administrados por el Registro son separados.

### 6. El retroceso significa la restauración de la ruta

Una publicación inmutable no se edita durante el rollback. Rollback selecciona un pin previamente admitido y actualmente elegible y cambia la ruta activa.

Un objetivo seguro debe:

1. Tener un registro de admisión completado.
2. Todavía tienes un estado activo de registro bajo tu póliza.
3. No estar en cuarentena por pruebas de seguridad.
4. Aún resuelva al paquete fijado y el conjunto de descriptor en vivo.
5. Pasar los controles de salud actuales.

La muestra se centra en las tres primeras condiciones. Un reconciliador real debe recoger el paquete y volver a comprobar el punto final en vivo antes de la activación.

### 7. Añadir un libro mayor de admisiones

Una base de datos de admisiones dice lo que está activo.

Cada entrada muestra contiene una secuencia, tiempo, evento, servidor, versión, resultado, razones, evidencia, el hash de la entrada anterior y su propio hash. Cambiar un resultado anterior rompe la verificación de esa entrada y cada enlace posterior.

Esto es evidente, no mágicamente a prueba de manipulación. Encuentra libros de contabilidad periódicos en un dominio de confianza separado, como metadatos de liberación firmados o almacenamiento de escritura una vez. Restringe quién puede agregar. Mantenga tokens de autorización, credenciales de paquetes, argumentos de herramientas y datos de puntos finales privados fuera de evidencia.

## Construye el mismo

El controlador ejecutable está en `code/main.py`Sólo utiliza la biblioteca estándar de Python.

Comience con la demostración finita:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
```

La demostración realiza cinco operaciones:

1. Admite que`1.0.0`con espacios de nombres, procedencia del paquete, protocolo, capacidades y herramientas correspondientes.
2. Admite que`1.1.0`y hacerla activa.
3. Observe una herramienta de eliminación inesperada en el tiempo de ejecución.
4. Observar el estado del Registro para `1.1.0`¿ Cómo se hace ?`deprecated`¿ Qué ?
5. Restablecer el enrutamiento a los que aún están admitidos `1.0.0`¿Qué es eso?

Forma esperada:

```json
{
  "admitted": [true, true],
  "driftAllowed": false,
  "rollbackAllowed": true,
  "activeVersion": "1.0.0",
  "ledgerValid": true
}
```

Lea la ejecución en este orden:

1. `namespace_for_domain()`y `namespace_matches()`establecer la autoridad exacta para nombrar.
2. `digest()`y `normalized_tools()`producen pruebas deterministas.
3. `RegistryAdmissionController.admit()`se une a la publicación, procedencia, tiempo de ejecución y política.
4. `check_live()`compara una nueva observación con el pin.
5. `observe_registry_status()`las versiones de cuarentena cuyo estado de registro cambie.
6. `rollback()`sólo activa un objetivo admisible previamente admitido.
7. `AdmissionLedger.verify()`detecta cambios en el historial registrado.

## Usalo

Ponga el controlador entre el descubrimiento y el enrutamiento:

```text
Registry sync -> artifact verifier -> live discovery -> admission controller -> route table
                                               |                 |
                                               v                 v
                                          evidence store    admission ledger
```

Usar identidades separadas para estos trabajos. Un trabajador de sincronización de Registro necesita acceso a metadatos. Un verificador de artefactos necesita acceso a paquetes. Un reconcillador de rutas necesita permiso para activar un pin aprobado. Ninguno de ellos necesita todas las credenciales.

Hacer que la declaración de lanzamiento sea explícita. Approved significa la política aprobada por la evidencia. Active significa que la ruta la selecciona actualmente. Quarantined significa que no puede recibir nuevos trabajos. Superseded significa que otra versión admitida está activa. No codifique los cuatro significados en un booleano.

Ejecutar la admisión antes de exponer un servidor en `tools/list`De lo contrario, el cliente puede descubrir una herramienta durante la brecha entre la publicación y la evaluación de políticas.

## Laboratorio interactivo

Verá que un límite falla a la vez.

### Laboratorio A: colisión en el espacio de nombres

Abre una cáscara de Python desde el directorio de código:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/code
python3 -q
```

Entonces corre:

```python
from main import namespace_matches
namespace_matches("com.example/inventory", "com.example")
namespace_matches("com.exampleevil/inventory", "com.example")
```

El primer resultado es `True`La segunda es:`False`. Reemplazar la comparación exacta con `startswith`y observar por qué el segundo nombre cruza la frontera. Restaurar la comparación exacta antes de continuar.

### Laboratorio B: deriva del descriptor

```python
from main import *
times = iter(f"2026-08-21T12:00:{n:02d}+00:00" for n in range(10))
c = RegistryAdmissionController(clock=lambda: next(times))
meta = {OFFICIAL_META_KEY: {"status": "active"}}
c.admit(sample_record("1.0.0"), meta, "com.example", evidence_for("1.0.0"), sample_live("1.0.0"))
c.check_live("com.example/inventory", "1.0.0", sample_live("1.0.0", True))
```

Inspeccionar las razones y el estado de la ruta. El registro del paquete y el Registro no cambió. La superficie de la herramienta de tiempo de ejecución lo hizo, por lo que el controlador puso en cuarentena y desactivó el pin.

### Laboratorio C: estado y retroceso

Admite que`1.1.0`, marque que se ha desactualizado, y prueba ambos objetivos de retroceso:

```python
c.admit(sample_record("1.1.0"), meta, "com.example", evidence_for("1.1.0"), sample_live("1.1.0"))
c.observe_registry_status("com.example/inventory", "1.1.0", "deprecated")
c.rollback("com.example/inventory", "1.1.0", "unsafe retry")
c.rollback("com.example/inventory", "1.0.0", "restore known release")
c.ledger.verify()
```

El objetivo de cuarentena es rechazado, el pin activo anterior es aceptado, el libro mayor sigue siendo válido.

## Laboratorio de práctica

Extenda el controlador con una puerta de aprobación para dos personas.

Requisitos:

- Las aprobaciones se almacenan como referencias de evidencia firmada, no como nombres mutables en el pin.
- Requerir dos identidades diferentes de revisores para un conjunto de herramientas que contiene una herramienta con `destructiveHint: true`¿ Qué ?
- Rechazar las identidades duplicadas de los revisores.
- Conservar el intento de admisión original en el libro mayor cuando la aprobación no esté completa.
- Añadir pruebas para cero, uno, duplicado y dos aprobaciones distintas.
- No registres firmas, credenciales o argumentos completos de herramientas privadas.

El éxito significa que una herramienta destructiva no puede activarse hasta que ambas identidades aprueben el registro exacto, el paquete y el conjunto de herramientas.

## Artículo enviado

Esta lección nos lleva .`outputs/skill-mcp-registry-admission.md`. Utilice como un libro de ejecución plano y reutilizable cuando revise una nueva versión de Registro o investigue la deriva. Define las entradas, reglas de rechazo, paquete de pruebas, reconciliación de estado y prueba de retroceso sin depender de los nombres de clases de muestra.

## Verifique el hecho

Ejecutar la demostración y la suite determinista:

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

La verificación deberá demostrar:

- límites exactos del espacio de nombres rechazar prefijos parecidos
- Sólo el estado oficial del Registro con espacios de nombres puede hacer que una versión sea elegible
- se rechazan los paquetes no verificados o no coincidentes y las pruebas remotas
- Los metadatos de los editores no pueden imitarse a los metadatos administrados por el Registro
- el ordenamiento de herramientas se normaliza sin ocultar cambios en el descriptor
- las estructuras de paquetes y herramientas malformadas se rechazan de forma segura
- `serverInfo`permanece diagnóstico y nunca proporciona autoridad de admisión
- Descriptor de cuarentena de deriva, desactiva y bloquea el retroceso al pin
- cambios de estado de los pines activos de cuarentena
- el rollback no puede seleccionar una versión en cuarentena o desconocida
- se detecta una manipulación del libro mayor

## Modos de falla de producción

| Failure | Why it happens | Required response |
|---|---|---|
| Name looks valid but namespace was never authenticated | Policy trusted record text | Reject until a trusted namespace verifier supplies the exact prefix |
| Same package coordinate returns new bytes | Mutable upstream or compromised distribution | Stop activation, retain both digests, investigate the fetch boundary |
| “Latest” changes without review | Floating selection escaped the pin | Resolve only exact admitted versions and digests |
| New tool appears after approval | Runtime drift or a different deployment | Quarantine the route and capture a fresh descriptor observation |
| Deprecated version remains active | Status sync is missing or delayed | Reconcile status on a schedule and before activation |
| Deleted record disappears from default sync | Client requested only active records | Use incremental or deleted-aware reconciliation and preserve local history |
| Rollback target was never admitted | Route control and approval state are disconnected | Refuse rollback and run a new admission for the target |
| Ledger verifies locally after an attacker rewrites all entries | Hash chain has no external anchor | Publish signed ledger heads to a separate trust domain |
| Evidence contains bearer tokens or tool arguments | Logging copied whole requests | Redact at collection time and store only the minimum proof |

## Regla de funcionamiento

Respuestas de publicación ¿Puede esta identidad publicar este nombre? Respuestas de admisión ¿Executaremos este artefacto exacto y exponeremos este comportamiento exacto?

## Leer más

- [Official Registry server.json requirements](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [Official Registry OpenAPI contract](https://registry.modelcontextprotocol.io/openapi.yaml)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
