# Descubrimiento de habilidades y revelación progresiva

> Una habilidad se vuelve útil antes de que su cuerpo se cargue. Su nombre y descripción ganan un lugar en el catálogo; sus archivos más profundos ganan contexto solo cuando la tarea les llega.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 22 (Agent Skills: Portable Contract and Runtime Boundary)
**Time:** ~105 minutes

## Objetivos de aprendizaje

- Construir una tubería de descubrimiento del sistema de archivos que separe el alcance, la validación, la política de colisión y la publicación del catálogo.
- Explica los tres niveles de divulgación: metadatos de catálogo, instrucciones activas y recursos específicos de tareas.
- Referencias de diseño para que un agente pueda llegar directamente al detalle requerido sin cargar todo el paquete.
- Espacio de catálogo presupuestario independiente del contexto de las competencias activas.
- Rechazar el recorrido de la ruta y escapar de la sincronía cuando una habilidad lee sus propios recursos.

## El problema

Su agente tiene 200 habilidades instaladas.`SKILL.md`, archivo de referencia, script y plantilla al inicio de la sesión enterrarían la tarea actual en un procedimiento no relacionado. Cargar nada obligaría al usuario a recordar las rutas exactas del sistema de archivos.

El compromiso habitual es el catálogo: mostrar al modelo una identidad compacta y una descripción de enrutamiento para cada habilidad elegible, y luego cargar el cuerpo completo sólo después de la selección.

En primer lugar, el descubrimiento no es solo una búsqueda de archivos recursiva. Las habilidades pueden existir en el proyecto, usuario, administrador, plugin o escopo integrado. Dos paquetes pueden compartir un nombre. Un enlace simbólico puede apuntar fuera de la raíz de confianza. Un paquete malformado puede consumir espacio en el catálogo o convertirse en imposible de invocar.

En segundo lugar, la revelación progresiva puede convertirse en una confusión progresiva.`SKILL.md`Si cada guía apunta a tres archivos más, la carga se convierte en un gráfico sin límites.

Un buen tiempo de ejecución hace que el descubrimiento sea determinista y la divulgación intencional.

## El concepto

### Discovery es un pipeline de compiladores

Trate el sistema de archivos como entrada de origen. No publique caminos crudos directamente al modelo.

```figure
skill-discovery-pipeline
```

Cada etapa debe producir datos estructurados y fallos estructurados.

- ¿Qué raíces fueron buscadas?
- ¿Qué candidatos fueron encontrados?
- ¿Qué candidatos fueron rechazados, y por qué?
- ¿Qué paquete ganó una colisión?
- ¿Qué catálogos fueron reducidos o omitidos por razones presupuestarias?

Sin esa evidencia, "el modelo no usó mi habilidad" es casi imposible de diagnosticar.

### El alcance es la política de tiempo de ejecución

La especificación portátil define un paquete de habilidades, no un sendero de instalación universal o orden de prioridad.

Un tiempo de ejecución genérico podría utilizar estos escopo:

| Scope | Example root | Intended ownership |
|---|---|---|
| Workspace | `<repo>/.agents/skills/` | Project maintainers |
| User | `<user-data>/skills/` | One developer |
| Administrator | `<system>/skills/` | Machine or organization policy |
| Plugin | A signed plugin bundle | Plugin publisher and installer |
| Built-in | Runtime package | Runtime vendor |

A partir de agosto de 2026, el Codex documenta el proyecto de descubrimiento de `$CWD/.agents/skills`El código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código de código`SKILL.md`; verificar la corriente [Codex skill documentation](https://learn.chatgpt.com/docs/build-skills)cuando escribe un adaptador.

Nunca inventes prioridad de nombres de directorios, declare como política y prueba.`Scope`Así que el mismo conjunto de candidatos siempre resuelve de la misma manera.

### Las colisiones necesitan una identidad más allá `name`

Dos paquetes nombrados `release-readiness`Una puede ser una anulación del espacio de trabajo y otra un usuario predeterminado.

```json
{
  "name": "release-readiness",
  "description": "Inspect a release candidate for this repository.",
  "scope": "workspace",
  "source": "/repo/.agents/skills/release-readiness",
  "selected": true
}
```

Las políticas comunes de colisión incluyen:

| Policy | Benefit | Risk |
|---|---|---|
| Keep every candidate | Nothing is hidden | The model sees ambiguous names |
| Highest-precedence scope wins | Simple invocation | A local package can shadow a trusted one |
| Reject duplicates | No silent shadowing | Legitimate overrides stop working |
| Qualify names by source | Explicit identity | User-facing names become longer |

Elige una política para el anfitrión. Preserva los candidatos rechazados o sombrados en la diagnóstico incluso cuando no estén en el catálogo de modelos.

### Tres niveles de divulgación

La especificación de habilidades de agente describe la carga en etapas.

```figure
skill-disclosure-levels
```

#### Nivel 1: Metadatos de catálogo

El modelo necesita suficiente información para distinguir la habilidad de los vecinos.

Una descripción útil tiene dos cláusulas:

```yaml
description: Validate a release candidate and produce a readiness report. Use when the user asks whether a version, tag, or package is ready to publish.
```

La primera cláusula establece la capacidad, la segunda el límite de disparos, la lección 25 evalúa este límite con las instrucciones positivas y cercanas a la falta.

#### Nivel 2: instrucciones activas

Después de la activación, el cuerpo debe funcionar como un mapa y un procedimiento.`SKILL.md`Es una señal de diseño, no un objetivo para llenar.

El cuerpo debe contener:

- el límite de tarea;
- el flujo de trabajo predeterminado;
- condiciones de sucursal;
- las referencias directas a archivos más profundos;
- contratos de herramientas y guiones;
- el comportamiento de falla y detención;
- la producción esperada y su verificación.

No muevan el flujo de trabajo central a una referencia simplemente para acortar el archivo de entrada.

#### Nivel 3: recursos de apoyo

Las referencias proporcionan prosa o datos. Los guiones proporcionan un cálculo determinista. Los activos se copian, llenan o transforman en resultados en lugar de tratarlos como instrucciones.

| Directory | Model reads it? | Model executes it? | Typical content |
|---|:---:|:---:|---|
| `references/` | Yes, when needed | No | schemas, policies, domain guides |
| `scripts/` | May inspect it | Through a permitted tool | validators, converters, collectors |
| `assets/` | Only if useful | No | templates, fixtures, images, starter files |

Estos nombres son convenciones, no capacidades mágicas. El host todavía necesita acceso a archivos y una herramienta de ejecución.

### Las referencias específicas de cada rama superan las descargas de temas

Escriba el archivo de entrada como un mapa de decisión:

```markdown
## Choose the path

- For a Python package, read `references/python-release.md`.
- For a container image, read `references/container-release.md`.
- For a documentation-only release, read `references/docs-release.md`.
- If the release combines artifact types, read only the guides for those artifacts.
```

Esto da a cada referencia una condición de carga observable.`references/`No lo hace.

Mantenga el gráfico de referencia superficial.`SKILL.md`Un salto hace que la accesibilidad sea probable y reduce la posibilidad de que una restricción necesaria nunca entre en contexto.

```figure
skill-reference-map
```

### El presupuesto del catálogo y el contexto activo son presupuestos diferentes

- ¿ Qué ?`c_i`ser el costo del catálogo serializado de la habilidad `i`¿ Qué ?`B_c`el presupuesto del catálogo, `b_j`el coste del cuerpo activo, y `r_k`los recursos realmente cargados.

```text
catalog_cost = sum(c_i for every published skill)
active_cost = sum(b_j for every activated skill) + sum(r_k for every disclosed resource)
```

Reducir un presupuesto no reduce automáticamente el otro. Las descripciones cortas pueden ahorrar espacio en el catálogo mientras que un cuerpo de 900 líneas activado todavía abrumará la tarea. Dividir el cuerpo en referencias puede reducir el costo activo solo cuando el tiempo de ejecución e instrucciones realmente eviten cargar ramas irrelevantes.

El Codex actualmente presupuesta la lista inicial de habilidades en el 2 por ciento del contexto
el valor de 8.000 caracteres es un valor de
el retroceso sólo cuando ese tamaño no se conoce; no es un segundo límite combinado con
Cuando el catálogo exceda el presupuesto aplicable,
Las descripciones pueden ser abreviadas o omitidas.
Política del Código, no una propiedad de la norma de Habilidades de Agentes.

### Los caminos de recursos son un límite de confianza

Una habilidad debe leer sólo archivos dentro de su paquete.

```text
references/../../../../.ssh/config
references/external-link -> /private/company-secrets
```

Resolver la raíz del paquete y el candidato con la semántica del sistema de archivos, rechazar las entradas absolutas y verificar que el candidato resuelto permanece bajo la raíz resuelta. Decidir si los enlaces simbólicos están permitidos antes de la descubrimiento. Si es posible, comprobar el objetivo resuelto cada vez.

```figure
skill-resource-containment
```

La contención de vías no establece la confianza del contenido. Una referencia válida dentro del paquete puede aún contener instrucciones maliciosas. La lección 26 maneja esa amenaza.

### La carga debe ser observable

Registrar eventos de divulgación sin registrar secretos:

```json
{
  "event": "skill.resource.loaded",
  "skill": "release-readiness",
  "resource": "references/python-release.md",
  "reason": "candidate contains pyproject.toml",
  "bytes": 2840
}
```

La razón convierte una elección de contexto en evidencia revisable. También ayuda a identificar instrucciones que hacen que el agente cargue cada archivo "sólo por si acaso".

## Construye el mismo

`code/main.py`construye un motor determinista de descubrimiento y divulgación.

La superficie de descubrimiento incluye:

- `Scope`para los metadatos de origen y de precedencia;
- `SkillCandidate`para un candidato no validado del sistema de archivos;
- `discover_scope(scope)`enumerar directorios de competencias inmediatas;
- `resolve_collisions(candidates, precedence)`aplicar una política declarada;
- `CatalogEntry`y `build_catalog(...)`publicar metadatos limitados;
- `CatalogBudget`para dar cuenta de las entradas serializadas sin pretender caracteres son tokens universales.

La superficie de divulgación incluye:

- `load_skill_body(entry, ...)`para la activación de nivel 2;
- `validate_reference(skill_dir, reference)`para contención de rutas;
- `load_reference(...)`para las lecturas de nivel 3 limitadas.

Dirige el laboratorio:

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Este bloque requiere un clon local y resuelve la raíz del repositorio de cualquier
directorio de trabajo dentro de ese clon.

La demostración crea espacios temporales de proyecto y usuario, inserta una colisión, construye un catálogo bajo un presupuesto deliberadamente pequeño, activa una habilidad y intenta una lectura de referencia válida y una fuga de paso.

### ¿Por qué el descubrimiento es superficial?

`discover_scope`verifica directorios de niños inmediatos para `SKILL.md`No trata recursivamente a todos los anidados .`SKILL.md`El sistema de seguridad de los dispositivos de seguridad de los usuarios es un sistema de seguridad de seguridad de los usuarios.

### ¿Por qué el laboratorio no analiza YAML arbitrario

El laboratorio soporta la materia frontal escalar necesaria para su catálogo. Un tiempo de ejecución de producción debe usar un parser YAML seguro con un esquema explícito, límites de tamaño y construcción de objetos personalizados desactivados. "Stdlib-solo" es una restricción de enseñanza, no permiso para inventar un dialecto parcial de YAML en silencio.

## Usalo

Aplicar esta lista de verificación a cualquier adaptador de descubrimiento:

1. Enumera cada raíz configurada y quién puede escribir a ella.
2. Indique si se permiten paquetes sin enlaces.
3. Valida el nombre del paquete, el nombre del directorio, los metadatos requeridos y el tamaño del cuerpo de entrada.
4. Preservar la fuente y el alcance en la identidad interna.
5. Declarar y probar el comportamiento de nombre duplicado.
6. Medir el catálogo serializado exacto enviado al modelo.
7. Registrar por qué se cargó un cuerpo o recurso.
8. Mantenga las lecturas de recursos dentro de la raíz del paquete resuelto.
9. Fallar claramente cuando falta un archivo de referencia.
10. Reconstruir el catálogo cuando las instalaciones o políticas cambien.

## Envío

Esta lección produce la`skill-catalog-builder`Bandeje. Escanea raíces ordenadas explícitamente, rechaza archivos de entrada sin enlaces y incompatibilidades de directorios de nombres, resuelve colisiones entre ámbitos, rechaza duplicados de igual precedencia y ajusta los metadatos seleccionados a los presupuestos de entrada, descripción y caracteres serializados declarados.

Su informe JSON contiene entradas seleccionadas, candidatos sombreados, entradas omitidas, errores de validación, precedencia y uso presupuestario.

## Los ejercicios

1. Añadir un alcance de plugin y colocarlo entre el usuario y la prioridad incorporada.
2. Cambiar la política de colisión de la mayor prioridad a nombres calificados.
3. Añadir un límite de tamaño de byte a `load_reference`Prueba un archivo exactamente en el límite y un byte por encima de él.
4. Crear dos descripciones que suenen casi idénticas y reescribirlas para que los límites del gatillo no se superpongan.
5. Añadir un manifiesto que contiene hashes para cada referencia y guión. Detectar un recurso modificado antes de cargarlo.
6. Instrumenta la demostración para informar los números de byte de nivel 1, nivel 2 y nivel 3 por separado.

## Términos clave

| Term | What people say | What it actually means |
|---|---|---|
| Skill discovery | "Find every SKILL.md" | Search configured scopes, validate packages, attach provenance, and apply policy |
| Skill catalog | "The list of installed skills" | Compact model-visible routing metadata for eligible packages |
| Collision policy | "Which duplicate wins" | A declared rule for same-name candidates from different sources |
| Progressive disclosure | "Lazy loading" | Staged context admission from catalog to body to branch-specific resources |
| Reference graph | "Files linked by the skill" | The reachable resource structure and its load conditions |
| Path containment | "Stay in the folder" | Verify resolved resource targets remain inside the resolved package root |

## Leer más

- [Agent Skills specification](https://agentskills.io/specification)para la forma del paquete y los niveles de divulgación progresiva.
- [Optimizing skill descriptions](https://agentskills.io/skill-creation/optimizing-descriptions)para metadatos de enrutamiento de catálogo.
- [Agent Skills best practices](https://agentskills.io/skill-creation/best-practices)para referencias directas y tamaño de archivo de entrada.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills)para los actuales ámbitos de descubrimiento del Codex y los límites del catálogo.
