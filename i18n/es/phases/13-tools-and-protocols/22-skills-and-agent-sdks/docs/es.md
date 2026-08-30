# Habilidades de agente: contrato portátil y límite de tiempo de ejecución

> Una habilidad no es un pedido largo con un mejor nombre de archivo. Es un paquete descubrible de instrucciones, recursos y ayudantes ejecutables que entra en el contexto de un agente a través de un contrato de tiempo de ejecución.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 01 (The Tool Interface), Phase 13 · 05 (Tool Schema Design)
**Time:** ~90 minutes

## Objetivos de aprendizaje

- Definir una habilidad de agente sin confundirla con un prompt, instrucciones de repositorio, una herramienta, un gancho, un subagente o un plugin.
- Lea el portátil .`SKILL.md`el contrato y separarlo de las extensiones específicas del tiempo de ejecución.
- Explicar el descubrimiento, la selección, la activación, la carga de recursos, el uso de herramientas y la verificación como etapas distintas del ciclo de vida.
- Valida un paquete de habilidades antes de que un tiempo de ejecución lo coloque en el catálogo de un agente.
- Elija entre una habilidad, herramienta MCP, gancho, subagente o código ordinario para una tarea concreta.

## El primer éxito en diez minutos

Haz esto antes de la larga explicación.
el completo revisor paquete en un agente real host, invocarlo, verificar el
Esto demuestra el ciclo de vida con un resultado observable.

### Prevuelo para el laboratorio de anfitrión real

El punto de control del host real requiere Node.js, `npx`, Python 3, uno seleccionado
un host capaz de competir, y escribir acceso al proyecto o el alcance de usuario que elija en
Verifique primero los comandos locales:

```bash
node --version
npx --version
python3 --version
```

Decida qué host y alcance utilizará antes de la instalación.
no está disponible, lea esta lección en el sitio web o continúe con
El ejercicio manual de paquete de abajo.
no comprueba el descubrimiento del host, la invocación, la ejecución de un guión en paquete, o
Desinstala el comportamiento. Mantenga las observaciones marcadas pendientes.

### 1. Comience en un directorio de trabajo vacío

Ejecutar estos comandos desde cualquier directorio de padres donde usted sigue aprendiendo a trabajar:

```bash
mkdir -p agent-skills-first-run
cd agent-skills-first-run
TARGET_ROOT="$(pwd -P)"
printf 'TARGET_ROOT=%s\n' "$TARGET_ROOT"
ls -A
```

El comando final no debe imprimir nada.
directorio vacío para que la revisión tenga un límite claro.

Crear un directorio para su primera habilidad:

```bash
mkdir -p my-first-skill
```

Crear`my-first-skill/SKILL.md`con este contenido:

```markdown
---
name: my-first-skill
description: Turn rough meeting notes into a compact decision record when the user asks to capture a technical decision.
---

# Decision record

Extract the decision, context, alternatives, owner, and next review date.
If the notes do not contain a decision, ask one clarifying question instead
of inventing one.
```

Verifique que ha creado el archivo en el directorio previsto:

```bash
test -f my-first-skill/SKILL.md
```

Ningún código de salida y salida 0 significa que el archivo existe.

### 2. Instalar el paquete completo de revisores

Quédate en el interior .`agent-skills-first-run`y ejecutar:

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-contract-reviewer --full-depth
```

Seleccione el host de agente y el alcance que está utilizando.
`skill-contract-reviewer`y el destino que escribió.`--full-depth`es
Se requiere porque la habilidad de esta lección es un paquete anidado con referencias, un
El guión, y un activo.

Se ha establecido`SKILL_ROOT`El fabricante de instalación deberá presentar un documento de información sobre el
ser el directorio que contiene los instalados `SKILL.md`, no la fuente de la lección
directorio y no el espacio de trabajo actual:

```bash
# Replace the placeholder with the destination printed by the installer.
SKILL_ROOT="$(cd "/absolute/path/to/skill-contract-reviewer" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\n' "$SKILL_ROOT"
```

Si la sesión del agente ya estaba abierta, comience una nueva sesión o use la sesión del host
No asumas que cada host vuelva a cargar su catálogo.

### 3. Lo invocará explícitamente

En el agente instalado, con `agent-skills-first-run`como el trabajo
directorio, utilizar la sintaxis que apoya ese host:

| Host | Explicit invocation |
|---|---|
| Codex | `skill-contract-reviewer`, or choose it from `/skills`, then provide the review request |
| Claude Code | `/skill-contract-reviewer` followed by the review request |
| Portable fallback | `Use skill-contract-reviewer to review the target package.` |

Utilice los valores absolutos impresos para `SKILL_ROOT`y `TARGET_ROOT`en el
Requerir al host para expandirlos antes de la ejecución y mostrar el exacto
comando resuelto, no un comando que dependa del directorio de trabajo de procesos:

```text
Use skill-contract-reviewer to review <TARGET_ROOT>/my-first-skill. The installed bundle root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/check_skill.py <TARGET_ROOT>/my-first-skill. Before running it, show the fully resolved argv. Return the validation report, selected primitives, and one sentence for each selection. Include the resolved script path, resolved target path, cwd, argv, and exit code as execution evidence.
```

El comando resuelto debe tener esta forma, sin que quede ningún titular de lugar:

```bash
python3 "/absolute/install/path/skill-contract-reviewer/scripts/check_skill.py" \
  "/absolute/workspace/path/agent-skills-first-run/my-first-skill"
```

Un resultado exitoso tiene las tres propiedades:

1. El anfitrión encuentra .`skill-contract-reviewer`por nombre.
2. El revisor lee el contrato de paquete y ejecuta su validador en paquete.
3. La respuesta contiene un informe de validación sin error estructural para el
   muestra, más una selección primitiva justificada.

La evidencia de ejecución también debe nombrar el camino del guión, el camino del objetivo, el cwd, el exacto
Un informe fluido sin esos campos no puede ser
prueba que el guión de acompañamiento instalado se ejecutó.

Si el host informa que la habilidad no está disponible, verifique la instalación
el destino, volver a escanear o reiniciar una vez, y volver a intentar la solicitud explícita.
reescribir la descripción de las habilidades para ocultar una falla de instalación.

### 4. Selección implícita de la sonda

Comience un nuevo turno de agente y ingrese la misma tarea sin nombrar la habilidad:

```text
Review <TARGET_ROOT>/my-first-skill as a reusable agent package and tell me whether its package contract is valid.
```

Si el anfitrión expone habilidades seleccionadas, registre si eligió
`skill-contract-reviewer`Si el anfitrión no expone el enrutamiento, marque implícito
La invocación explícita es el fallback portátil.

### 5. Limpiar

Eliminar sólo el paquete de revisores instalado:

```bash
npx skills remove skill-contract-reviewer
```

Seleccione el mismo host y alcance utilizados durante la instalación.
La Comisión ha presentado una propuesta de resolución en el marco de la cual se ha aprobado el Reglamento (CE) n° 1408/71 de la Comisión.`skill-contract-reviewer`debe informar que
No está disponible.`my-first-skill`para las lecciones posteriores, o eliminar la
directorio del laboratorio después de terminar la pista.

## El problema

Supongamos que su equipo tiene un flujo de trabajo de liberación confiable. encuentra los cambios fusionados, revisa las notas de migración, actualiza el registro de cambios, ejecuta un comando de embalaje y produce una lista de revisión.

El proceso de trabajo se puede pegar fácilmente y difícilmente de manejar. El proceso de trabajo no tiene una identidad estable, ninguna regla de descubrimiento, ningún límite de recursos, ninguna forma de paquete verificable y ninguna respuesta a preguntas básicas: ¿Quién puede invocarlo? ¿Cuándo debe seleccionarlo el modelo? ¿Qué scripts puede ejecutar? ¿Qué archivos son confiables? ¿Qué sobrevive cuando se compacta el contexto?

El error opuesto es tratar cada instrucción reutilizable como una habilidad.`SKILL.md`produce un directorio que se ve portátil mientras depende del comportamiento indocumentado de un host.

La primera tarea de ingeniería es la clasificación, decidir qué es el artefacto antes de decidir cómo empacarlo.

## El concepto

### Habilidades para codificar conocimientos procesales

Una habilidad de agente es un directorio cuyo punto de entrada es `SKILL.md`. El archivo de entrada contiene la materia frontal de YAML seguida de instrucciones de Markdown.

```figure
skill-package-anatomy
```

El directorio, no el archivo Markdown solo, es la unidad desplegable.`SKILL.md`con referencias faltantes es un paquete roto incluso si su material frontal se analiza.

### Las abstracciones vecinas

| Artifact | Primary job | Loaded or run when | What it should not impersonate |
|---|---|---|---|
| Prompt | Shape one model interaction | Included by an application or user | A versioned package with resources |
| Repository instructions | Explain one codebase's standing rules | A coding runtime enters that scope | A reusable task workflow |
| Agent skill | Supply reusable procedural knowledge | Explicit or implicit activation | A hard authorization boundary |
| MCP tool | Expose a typed remote capability | The model or application calls it | A detailed operating procedure |
| Hook | Run deterministic logic on an event | The declared event occurs | Probabilistic model routing |
| Subagent | Delegate work with separate context and state | An orchestrator creates or calls it | A static instruction bundle |
| Plugin | Distribute a larger runtime extension | The host installs or enables it | The portable skill contract itself |
| Learned skill library | Store behavior discovered through experience | A policy retrieves a prior program or trajectory | A standards-based `SKILL.md` package |

Un servidor MCP puede exponer el registro de liberación. Un gancho puede prohibir presiones directas. Un subagente puede auditar de forma independiente al candidato. Estas piezas se componen porque mantienen diferentes responsabilidades.

### La palabra "habilidad" nombra dos ideas diferentes

Los sistemas de investigación a veces llaman un programa aprendido, una trayectoria exitosa o un fragmento de política específico del medio ambiente una habilidad. Un agente puede crear estos artefactos durante la exploración, recuperarlos por similitud de tarea, ejecutarlos y revisar la biblioteca a partir de comentarios.

Un agente habilidad en esta mini-track es diferente. Es un paquete de autor con un contrato declarado del sistema de archivos, catálogo de metadatos, divulgación progresiva, invocación mediada por tiempo de ejecución, y herramientas controladas por el host. Puede ser generado o mejorado por un agente, pero el aprendizaje no es necesario para el formato.

| Dimension | Agent Skill package | Learned skill library |
|---|---|---|
| Primary unit | `SKILL.md` directory | Program, policy, trajectory, or memory record |
| Creation | Authored, generated, or curated | Usually discovered from environment experience |
| Selection | Catalog description plus runtime policy | Retrieval or policy over task state |
| Execution | Model follows instructions and calls host tools | Environment runs a stored behavior or code artifact |
| Portability | Package contract can cross compatible hosts | Often tied to one environment and action space |
| Evaluation | Routing, artifact, safety, and host compatibility | Reward, success rate, transfer, and library growth |

Ambas ideas incluyen competencias reutilizables, y no deben compartir las reclamaciones de ejecución simplemente porque comparten un nombre.

### El núcleo portátil

La especificación de habilidades de agente requiere dos campos de materia frontal:

```yaml
---
name: release-readiness
description: Inspect a release candidate when the user asks whether a version is ready to publish.
---
```

`name`El identificador estable debe cumplir las reglas de denominación de la especificación y coincidir con el directorio principal. `description`Es la documentación y el enrutamiento de metadatos.

Los campos opcionales portátiles son:

| Field | Purpose | Portability note |
|---|---|---|
| `license` | State the terms for the package | Core specification |
| `compatibility` | State environmental requirements | Core specification |
| `metadata` | Carry string-valued extension data | Core specification |
| `allowed-tools` | Suggest pre-approved tools | Experimental; host support varies |

El cuerpo de Markdown tiene las instrucciones operativas. Debe definir el flujo de trabajo, los puntos de decisión, el comportamiento de falla y los caminos directos hacia los recursos de soporte.

```markdown
# Release readiness

Use this workflow for a release candidate, not for ordinary development builds.

1. Read `references/release-policy.md`.
2. Run `python3 scripts/inspect_release.py --format json`.
3. Stop if the report contains a blocking failure.
4. Produce the checklist from `assets/release-checklist.md`.
5. Ask for approval before any publish or tag action.
```

### Las extensiones de tiempo de ejecución son una segunda capa

Algunos hosts aceptan una configuración adicional de frontmatter o compañero.

| Behavior | Example host extension | Portable core? |
|---|---|:---:|
| Hide a skill from model routing while keeping direct user invocation | `disable-model-invocation` | No |
| Hide a skill from the user's command menu while allowing model routing | `user-invocable` | No |
| Show argument help in a command menu | `argument-hint` | No |
| Run the skill in delegated context | `context`, `agent` | No |
| Pin model or reasoning settings | `model`, `effort` | No |
| Register lifecycle automation | `hooks` | No |
| Disable implicit invocation in Codex | `agents/openai.yaml` policy | No |

Tratar cada extensión como un adaptador. Mantenga el flujo de trabajo principal válido sin él, documenta el fallback y prueba al host que lo consume. Un tiempo de ejecución puede ignorar un campo desconocido, rechazarlo o preservarlo sin implementar el comportamiento.

### El material de frente es metadatos ejecutables

Los metadatos cambian el comportamiento del sistema antes de que se lea el cuerpo de habilidades.

- Un malformado .`name`puede hacer que el descubrimiento fracasen.
- Un poco vagos .`description`puede encaminar las peticiones equivocadas.
- Una bandera sólo humana puede eliminar la habilidad del catálogo del modelo.
- Una asignación de herramientas puede cambiar si un anfitrión pide permiso.
- Un ajuste de contexto puede mover la ejecución a una sesión de agente separada.

Revise la materia frontal como código de configuración, valida, versionarla e incluye su comportamiento en evals.

### El ciclo de vida de las habilidades

```figure
skill-runtime-lifecycle
```

Cada flecha es un límite con sus propios modos de fracaso.

1. **Discovery**encuentra posibles paquetes en ubicaciones configuradas.
2. **Validation**Rechaza los paquetes malformados o inseguros antes de la publicación del catálogo.
3. **Cataloging**expone un compacto `name`y `description`, no el paquete completo.
4. **Selection**decide si la habilidad es relevante.
5. **Activation**carga el cuerpo en un contexto visible del modelo.
6. **Disclosure**sólo lee referencias o activos cuando una sucursal los requiere.
7. **Execution**utiliza herramientas de host bajo las reglas de permiso y aislamiento del host.
8. **Verification**verifica el artefacto producido independientemente de la reclamación del modelo.

La desintegración de estas etapas causa malos modelos mentales. Una habilidad descubierta no es activa. Una habilidad activa no está autorizada a hacer todo lo que describe. Una llamada permitida a herramientas no es prueba de que el resultado es correcto.

### Las habilidades y las herramientas son ortogonales

MCP responde: "¿Qué capacidades puede requerir esta aplicación, y cuáles son sus esquemas?" Una habilidad responde: "¿Cómo debe un agente abordar esta clase de tarea?"

```figure
skill-tool-orthogonality
```

La habilidad puede nombrar una herramienta, pero el host posee el registro de capacidades reales. Si la herramienta está ausente, la habilidad debe indicar una caída o falla claramente. Nunca debe implicar que nombrar una capacidad la crea.

### Las competencias y las instrucciones de repositorio son ámbitos diferentes

Las instrucciones de repositorio describen el entorno en el que ya estás: comandos, convenciones, archivos generados y límites.

Cuando ambos se aplican, la solicitud activa del usuario y las reglas del repositorio restringen la habilidad.

### Las habilidades no se importan entre sí

Una habilidad puede dirigir al agente a invocar otra, pero esto no es una importación a nivel de lenguaje. La segunda habilidad sigue pasando por el descubrimiento en el tiempo de ejecución, la elegibilidad, la activación, los permisos y el manejo del contexto.

Escriba las dependencias entre habilidades como bordes de flujo de trabajo observables:

```markdown
After producing the candidate changelog, invoke the `release-risk-review` skill.
Pass the candidate path and require a blocking or non-blocking verdict.
If that skill is unavailable, stop and report the missing dependency.
```

Esto hace que la dependencia sea verificable y le da al anfitrión la oportunidad de hacer cumplir la política.

## Construye el mismo

`code/main.py`El sistema de control de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de datos de la red de red de datos de la red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de red de

El validador expone:

- `parse_frontmatter(text)`para separar metadatos del cuerpo.
- `validate_skill_text(text, directory_name, allowed_runtime_extensions=())`para comprobar los campos requeridos, el nombre, las extensiones desconocidas, la presencia del cuerpo y los límites portátiles.
- `ValidationIssue`y `SkillReport`para devolver evidencia estructurada en lugar de un booleano opaco.
- `FrontmatterSyntaxError`para entradas que no puedan ser interpretadas de manera segura.

El elegidor expone `TaskShape`y `select_primitives(task)`.Mapaza las necesidades de una tarea con código ordinario, instrucciones de repositorio, una habilidad, un gancho, un subagente o una herramienta MCP.

Dirige el laboratorio:

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/22-skills-and-agent-sdks
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Este bloque de comando requiere un clon local y debe comenzar desde cualquier lugar dentro
Ese clon así .`git rev-parse --show-toplevel`puede resolver la raíz del repositorio.

La demostración imprime JSON para una habilidad portátil válida, una habilidad ampliada para host, un paquete inválido y varias decisiones de forma de tarea.

### Las cuestiones relativas a la orden de validación

Validar los hechos estructurales baratos antes de reglas de contenido más profundas:

```figure
skill-validation-order
```

Este orden evita que los errores secundarios ocultan la primera invariante rota.

## Usalo

Antes de escribir una habilidad, llene esta tarjeta de decisión:

| Question | If yes | Likely primitive |
|---|---|---|
| Does this need reusable model judgment across several steps? | The procedure is stable but decisions vary | Skill |
| Must this happen every time an event fires? | Missing one execution is unacceptable | Hook or application code |
| Does the model need an external capability with typed inputs? | The operation lives outside model context | Tool or MCP server |
| Does the work need isolated context, state, or ownership? | A separate worker returns a bounded result | Subagent |
| Is this guidance specific to one repository? | It describes local commands and constraints | Repository instructions |
| Is one interaction enough? | No package lifecycle is needed | Prompt |

Muchos flujos de trabajo de producción utilizan más de una fila. La tarjeta evita que un artefacto pretenda proporcionar cada propiedad.

## Envío

Esta lección produce la`skill-contract-reviewer`el paquete de abajo `outputs/`Contiene:

- un portátil`SKILL.md`que revise un paquete de competencias propuesto;
- las listas de verificación de referencia para el contrato portátil y la selección primitiva;
- un guión de validación determinista;
- los accesorios de forma de tarea que cubren las instrucciones, las habilidades, las herramientas, los ganchos, el código ordinario y los subámbulos.

Instalar el paquete completo, no sólo su archivo de entrada:

```bash
cd "$(git rev-parse --show-toplevel)"
python3 scripts/install_skills.py /tmp/aiefs-skills --phase 13 --type skill
```

El instalador del curso informa cada habilidad copiada de la Fase 13 y escribe
`/tmp/aiefs-skills/manifest.json`Este destino limpio comprueba la forma del paquete .
El primer bucle de éxito de arriba verifica el descubrimiento y la invocación en un host real.

Las siguientes lecciones profundizan cada etapa del ciclo de vida. La lección 24 construye el descubrimiento y la divulgación progresiva. La lección 25 construye la política de invocación y el enrutamiento. La lección 26 separa los permisos del sandboxing. La lección 27 convierte todo el paquete en un artefacto de liberación evaluado.

## Los ejercicios

1. Clasifique cinco flujos de trabajo de su propio equipo utilizando `TaskShape`Defiende cada caso en el que elijas más de un primitivo.
2. Añadir pruebas de límite que demuestren que un 500 caracteres `compatibility`el valor pasa y un valor de 501 caracteres falla como error de especificación.
3. Añadir una extensión de tiempo de ejecución a la lista de permisos. Escribir una prueba que demuestre que el mismo archivo todavía es distinguible de una habilidad portátil sólo.
4. Divide una llamada de 400 líneas en `SKILL.md`, una referencia, un contrato de guión y una plantilla de salida.
5. Diseñar una respuesta de falla para una habilidad que se refiere a una herramienta MCP no disponible. No sustituya silenciosamente una herramienta con permisos más amplios.
6. Revise una habilidad existente y etiquete cada oración como enrutamiento, procedimiento, política, indicador de referencia o contrato de salida.

## Términos clave

| Term | What people say | What it actually means |
|---|---|---|
| Agent skill | "A saved prompt" | A discoverable directory of procedural instructions and optional resources |
| Portable core | "Fields every runtime shares" | The contract defined by the Agent Skills specification |
| Runtime extension | "Extra frontmatter" | Host-specific configuration whose behavior requires a compatible adapter |
| Activation | "The skill ran" | The skill body entered model-visible context; execution may come later |
| Skill dependency | "Import another skill" | A runtime-mediated invocation edge with availability and policy checks |
| Tool contract | "A function schema" | Inputs, outputs, permissions, side effects, errors, and evidence for a capability |

## Leer más

- [Agent Skills specification](https://agentskills.io/specification)para el contrato de directorio portátil y de material frontal.
- [Agent Skills best practices](https://agentskills.io/skill-creation/best-practices)para el alcance, las instrucciones y la organización de los recursos.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills)para el comportamiento actual de descubrimiento y invocación del Codex.
- [Claude Code skills](https://code.claude.com/docs/en/skills)para la invocación, el argumento, la herramienta y las extensiones de contexto delegado de un tiempo de ejecución.
