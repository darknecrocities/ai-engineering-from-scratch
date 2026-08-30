# Invocación de habilidades y enrutamiento

> La invocación es una decisión de la autoridad seguida de una decisión de relevancia.Una buena descripción ayuda al modelo a elegir; una buena política decide si esa elección es permitida.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 24 (Skill Discovery and Progressive Disclosure)
**Time:** ~105 minutes

## Objetivos de aprendizaje

- Distinguir entre la invocación explícita del usuario, la invocación implícita del modelo, la invocación de la aplicación y la invocación de habilidades.
- Modelo de visibilidad humana y modelo de elegibilidad como dimensiones políticas independientes.
- Escriba descripciones de enrutamiento con disparadores positivos y límites cercanos a la falta.
- Separar la elegibilidad, la selección, la activación, la vinculación de los argumentos y la ejecución en rastros y pruebas.
- Adaptar campos de invocación específicos del tiempo de ejecución sin presentarlos como material portátil.

## El problema

Usted instala un `database-migration`La habilidad. El usuario puede ejecutarlo por nombre, pero el modelo también ve su descripción y la selecciona cuando alguien hace una pregunta general de base de datos. La habilidad luego propone un cambio de esquema para una tarea que solo necesitaba una explicación.

Usted añade`user-invocable: false`En otro tiempo de ejecución, ese campo es ignorado.`disable-model-invocation: true`En el tiempo de ejecución que lo entienda, el usuario todavía puede invocarlo explícitamente.

No hay nada malo con los nombres de campos. El modelo es incorrecto. "El usuario puede verlo", "el modelo puede seleccionarlo", "la aplicación puede cargarlo previamente", y "las herramientas dentro de él pueden ejecutar" son hechos separados.`invocable`No puedo expresarlas.

El routing tiene un segundo modo de falla. Si las descripciones son vagas, varias habilidades se vuelven plausibles. Si las descripciones están llenas de palabras clave, las tareas no relacionadas las desencadenan. El catálogo es una interfaz probabilística: lo suficientemente compacta como para encajar, lo suficientemente específica como para rutas.

## El concepto

### Cinco canales pueden iniciar el ciclo de vida

| Actor | Invocation shape | Typical use | Main risk |
|---|---|---|---|
| Human user | Names a skill in the UI or prompt | Deliberate workflow selection | User expects availability or authority the host does not grant |
| Model or autonomous agent | Selects a catalog entry from task context | Automatic expert procedure | False-positive routing |
| Application | Activates or preloads a skill through runtime code | Fixed product workflow | Hidden coupling to one host |
| Another skill or subagent | Requests an exact skill as a workflow dependency | Composition | Cycles, missing dependency, or context bleed |
| Evaluation harness | Activates an exact skill under a fixed scenario | Repeatable measurement | Tests the skill while accidentally bypassing the production policy under study |

La especificación portátil de habilidades de agente define el paquete. no estandariza una interfaz de usuario de comandos de corte universal, bandera de enrutamiento implícito, API de aplicaciones o ciclo de vida de subagente.

### Las cinco etapas de invocación

```figure
skill-invocation-stages
```

Usa estas palabras con precisión:

- **Eligible**significa que la política permite a este actor solicitar la habilidad.
- **Selected**significa que el usuario lo nombró o que un router lo consideró relevante.
- **Activated**significa que sus instrucciones han entrado en el contexto de trabajo.
- **Executing**significa que el agente ha comenzado a trabajar en modelos o en herramientas de conformidad con dichas instrucciones.
- **Completed**significa que la producción ha cumplido una verificación de éxito independiente.

Un rastro que sólo registra .`skill_used=true`Esconde el límite donde ocurrió un fracaso.

### Invocación humana y modelo de forma de matriz 2x2

| Human can invoke | Model can invoke | Mode | Suitable examples |
|:---:|:---:|---|---|
| Yes | Yes | Shared | Code explanation, test planning, documentation review |
| Yes | No | Human-only | Publish preparation, billing export, destructive cleanup plan |
| No | Yes | Model-only | Internal style guide, domain reference, automatic support procedure |
| No | No | Disabled or application-only | Staged rollout, deprecated package, programmatic preload |

La matriz es un modelo de política, no YAML estándar.

Un host actual utiliza `disable-model-invocation: true`para la fila de sólo humanos y `user-invocable: false`Para la fila de modelo sólo. El predeterminado es ambos. Otro host utiliza `agents/openai.yaml`con`allow_implicit_invocation: false`Los servidores de la red de acceso pueden ignorar los servidores de la red de acceso.

El detalle confuso es importante:`user-invocable: false`No significa "el modelo no puede usar esto". Elimina la invocación directa del usuario en el host que lo define. `disable-model-invocation: true`No significa "la habilidad está desactivada". Elimina la selección iniciada por el modelo mientras se mantiene el acceso explícito del usuario.

### La invocación explícita es la identidad primero

Una invocación explícita proporciona identidad directamente:

```text
/release-readiness v2.4.0
```

o bien:

```text
release-readiness check v2.4.0 without publishing
```

Documento de interfaces del Código actual `/skills`para la selección y los nombres de destrezas en las solicitudes de invocación explícita.`/skill-name`La sintaxis exacta, la visibilidad del menú, las reglas de citas y la expansión de variables pertenecen al host.

Una solicitud explícita sigue aprobando la política. Nombrar una habilidad no debe pasar por alto los permisos faltantes, las restricciones del espacio de trabajo, las puertas de aprobación o el aislamiento en el tiempo de ejecución.

### La invocación implícita es la descripción primero

Para el enrutamiento implícito, el modelo ve inicialmente los metadatos del catálogo en lugar del cuerpo completo.

Debilidad:

```yaml
description: Helps with releases.
```

En exceso de ancho:

```yaml
description: Use for release, version, package, build, deploy, publish, tag, changelog, GitHub, CI, or software tasks.
```

Con un límite:

```yaml
description: Inspect an already prepared release candidate and produce a readiness report. Use when the user asks whether a version, tag, package, or image is ready to publish; do not use for ordinary build failures or feature development.
```

La versión limitada contiene:

1. **Capability:**inspeccionar a un candidato preparado.
2. **Output:**informe de preparación.
3. **Positive boundary:**pregunta si un artefacto de liberación está listo.
4. **Negative boundary:**las construcciones y el desarrollo ordinarios están fuera de alcance.

Los límites negativos son útiles cuando dos habilidades cercanas comparten vocabulario.

### El enrutamiento es la clasificación con una opción de abstenerse

Para una habilidad .`s`y la solicitud `x`, imaginen un puntaje del router:

```text
score(s, x) = capability_match + trigger_match + context_match - exclusion_match - ambiguity_penalty
```

El resultado exacto puede ser una decisión de LLM en lugar de aritmética. El principio de ingeniería sigue siendo válido: la selección debe superar un umbral y una habilidad en competencia. Cuando la evidencia es débil, absténgase.

```figure
skill-routing-abstention
```

Para las habilidades de alto impacto, el enrutamiento implícito puede ser inapropiado incluso con una descripción fuerte.

### La elegibilidad debe preceder al ranking

No marque cada habilidad descubierta, elija el partido más fuerte y revise la política de una habilidad después. Un partido bloqueado de la cima impide incorrectamente que un candidato elegible con menos puntuaciones sea considerado.

Utilice este orden para el enrutamiento implícito:

1. El filtro descubrió habilidades del actor solicitante y el adaptador de anfitrión activo.
2. Solo califique a los candidatos elegibles.
3. Seleccione el partido más fuerte elegible si elimina el umbral y las reglas de ambigüedad.
4. Abstenerse cuando ningún candidato es elegible o si no hay una puntuación suficientemente fuerte.

Supongamos que`incident-triage`puntuaciones `0.80`pero su extensión host deshabilita la invocación del modelo. `incident-review`puntuaciones `0.55`El router debe evaluar`incident-review`No debería elegir.`incident-triage`, lo niegue y deje de hacerlo.

Este ordenamiento también evita que los cambios de política alteren el significado de una puntuación de relevancia.

### Las evaluaciones de enrutamiento necesitan casi fallas

Los casos positivos demuestran que se recuerda:

```json
{"prompt":"Is version 2.4.0 ready to publish?","expected":"release-readiness"}
```

Los negativos claros demuestran la precisión básica:

```json
{"prompt":"Explain rotary position embeddings.","expected":null}
```

Las pérdidas cercanas exponen la calidad límite:

```json
{"prompt":"Why did today's package build fail?","expected":"build-diagnostics"}
```

Las acciones de la casi perdida`package`y `build`Un conjunto de enrutamiento hecho sólo de positivos obvios y negativos no relacionados exagerará la calidad.

### Los argumentos tienen tres representaciones

Un argumento de invocación cruza varios límites:

```figure
skill-argument-boundaries
```

En cada límite, preserva la intención sin tratar el texto como código.

- El parser de host decide la sintaxis de comandos y la citación.
- La habilidad recibe texto o variables vinculadas de acuerdo con las reglas del anfitrión.
- Las instrucciones validan los valores requeridos y los valores predeterminados.
- Una llamada de herramienta convierte valores en un esquema tipado y los revalida.

No interpolar argumentos crudos en comandos de shell. Prefiere un script invocado con un vector de argumento o una herramienta de MCP mecanografiada.

### La invocación de la solicitud es una orquestación explícita

Un producto puede activar una habilidad porque su flujo de trabajo ya conoce el tipo de tarea. Por ejemplo, un servicio de revisión de la solicitud de puesta puede precargar `pull-request-risk-review`después de que el usuario presione Review.

Esto elimina la incertidumbre de enrutamiento, pero crea una dependencia de la API de tiempo de ejecución.

```figure
skill-host-adapter
```

La habilidad debe permanecer inteligible cuando se abre por un cliente diferente que cumpla.

### La invocación de habilidades es una ventaja como una herramienta

Supongamos que`release-readiness`¿ Qué es eso ?`security-change-review`cuando los archivos de dependencia cambiaron.

El solicitante deberá proporcionar:

- la identidad de las habilidades objetivo;
- una tarea y un artefacto limitados;
- el contrato de respuesta esperado;
- el motivo de la invocación;
- una caída si no está disponible;
- una regla de profundidad máxima o ciclo.

```json
{
  "target_skill": "security-change-review",
  "task": "Review dependency changes in the candidate diff",
  "inputs": ["artifacts/release.diff"],
  "expected": "risk-report.json",
  "max_depth": 2
}
```

La segunda habilidad no se pega ciegamente en la primera. El host decide cómo activarla y si comparte contexto, se ejecuta en un tenedor o regresa a través de un resultado de herramienta.

### El ciclo de vida del contexto es específico para el host

Después de la activación, el cuerpo de habilidades puede permanecer en la conversación, ser resumido durante la compactación o ejecutado en un contexto delegado.

No escriba una habilidad que dependa de una suposición invisible de vida. Coloque resultados duraderos en archivos o estado de tipado, haga segura la reentrada, y diga qué debe recargarse después de la interrupción.

```markdown
On resume, read `artifacts/release-readiness.json` if it exists.
Revalidate the candidate commit before continuing.
Do not repeat an external write whose idempotency key is already recorded.
```

## Construye el mismo

`code/main.py`Implementa la política y el enrutamiento como adaptadores separados.

El modelo incluye:

- `Actor`para los llamadores humanos, modelos, agentes autónomos, aplicaciones, habilidades y sistemas de uso;
- `SkillMetadata`para la identidad de enrutamiento;
- `InvocationPolicy`para la matriz humana/modelo;
- `InvocationRequest`y `InvocationDecision`para entradas y resultados rastreables;
- `CorePolicyAdapter`para comportamiento portátil sin extensiones de host;
- `ExtensionPolicyAdapter`para campos de tiempo de ejecución reconocidos;
- `build_invocation_matrix(policy)`para la vista 2x2;
- `route_request(skills, request, adapter)`para el filtrado de elegibilidad antes de la clasificación de relevancia, la selección y la negación.

- ¿Qué quieres decir ?

```bash
cd phases/13-tools-and-protocols/25-skill-invocation-and-routing
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

La demostración imprime una matriz y decisiones para el modelo humano explícito, implícito, agente autónomo, aplicación, composición de habilidades y canales de aprovechamiento. Sus resultados de adaptación de extensión muestran que se elimina un bloqueo de la combinación léxica superior antes de clasificar una alternativa elegible. También incluye listados de nombres exactos. No se requiere API de modelo. El router determinista existe para hacer inspectables los límites de la política, no para afirmar que la coincidencia léxica reproduce el enrutamiento del modelo de producción.

### ¿Por qué los adaptadores de núcleo y extensión son separados?

Si un parser asigna significado a cada campo de frontmatter observado, promueve silenciosamente las convenciones de tiempo de ejecución en un estándar falso.

El `CorePolicyAdapter`La política de aplicación de la aplicación es la única que utiliza.`ExtensionPolicyAdapter`reconoce un conjunto explícito de campos de acogida y registros en los que el campo cambió la decisión.

## Usalo

Escriba un contrato de invocación antes de publicar una habilidad:

```yaml
actors:
  human: allow
  model: deny
  application: allow
  skill: deny
explicit_name: release-readiness
arguments:
  candidate: required
  publish: fixed_false
ambiguity: ask_user
missing_dependency: stop
context:
  durable_state: artifacts/release-readiness.json
  max_composition_depth: 2
```

Este contrato es una documentación de diseño para adaptadores y ensayos.`SKILL.md`el asunto principal, a menos que una norma lo adopte explícitamente.

## Envío

Esta lección produce la`skill-invocation-router`incluye una referencia de modelo de invocación, una política de host de ejemplo y un CLI no ejecutante que evalúa a un humano, modelo, agente autónomo, aplicación, composición de habilidades o solicitud de aprovechamiento y devuelve una decisión JSON con canal, adaptador, puntaje y razón.

El CLI de una sola solicitud es una investigación de política, no una evaluación de desencadenante completa. Utilice el diseño etiquetado positivo y casi perdido en la Lección 27 para calcular los recuentos de confusión, precisión, recuerdo y estabilidad de la ejecución repetida.

## Los ejercicios

1. Crear las cuatro filas de la matriz de modelo/humano y escribir un caso de uso legítimo para cada una.
2. Añadir activación sólo para aplicaciones a `CorePolicyAdapter`Demostrar que los llamados humanos y modelos siguen siendo negados.
3. Escriba diez puntos cercanos a la falta para una habilidad de despliegue.
4. Añadir un margen de ambigüedad entre los dos puntajes de enrutamiento superiores.`ask`cuando el margen es demasiado pequeño.
5. Añadir una profundidad máxima de composición a las solicitudes de habilidades y detectar un ciclo de dos habilidades.
6. Ejecutar el mismo conjunto etiquetado a través de los adaptadores de núcleo y extensión. Explicar cada decisión cambiada.

## Términos clave

| Term | What people say | What it actually means |
|---|---|---|
| Explicit invocation | "Slash command" | An actor supplies skill identity directly, subject to policy |
| Implicit invocation | "The model chooses" | A router selects from eligible catalog metadata based on task context |
| User-invocable | "Humans can use it" | A host-specific menu or direct-invocation property, not a core field |
| Model-invocable | "The agent can use it" | Eligibility for implicit model selection under host policy |
| Invocation adapter | "Frontmatter parser" | Code that maps a host's fields and APIs into a declared policy model |
| Near miss | "Hard negative" | A non-triggering request that resembles a skill's intended inputs |
| Abstention | "No skill selected" | A deliberate routing result when evidence is absent or ambiguous |

## Leer más

- [Optimizing skill descriptions](https://agentskills.io/skill-creation/optimizing-descriptions)para los desencadenantes positivos, la especificidad y la evaluación.
- [Evaluating skills](https://agentskills.io/skill-creation/evaluating-skills)para el diseño de evaluaciones de activación y salida.
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills)para los controles de invocación explícitos e implícitos del Código vigentes.
- [Claude Code skills](https://code.claude.com/docs/en/skills)para un anfitrión `user-invocable`¿ Qué ?`disable-model-invocation`, argumentos y contexto delegado.
