# Capstone: Envía un paquete de trabajo de agente reutilizable

> La mini-track termina con un paquete que se deja en cualquier repo. once lecciones de superficies comprimidas en un directorio que se puede`cp -r`La piedra angular es el artefacto del currículo.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phases 14 · 31 to 14 · 41
**Time:** ~75 minutes

## Objetivos de aprendizaje

- Envuelve las siete superficies de la mesa de trabajo en un directorio de entrada.
- Enlaza los esquemas, scripts y plantillas para que un nuevo repo obtenga una base conocida.
- Añadir un único script de instalación que coloque el paquete de forma idempotente.
- Deciden qué queda en el paquete y qué queda fuera, defendiendo el corte para cada uno.

## El problema

Un banco de trabajo que vive en un Google Doc, un historial de chat y tres scripts medio recordados es un banco de trabajo que se reconstruye cada trimestre. La cura es un paquete de versiones: un repo o directorio con las superficies, los esquemas, los scripts y un instalador de un solo comando.

Terminará esta lección con`outputs/agent-workbench-pack/`enviado en disco y un `bin/install.sh`que lo deja en cualquier repo objetivo.

## El concepto

```mermaid
flowchart TD
  Pack[agent-workbench-pack/] --> Docs[AGENTS.md + docs/]
  Pack --> Schemas[schemas/]
  Pack --> Scripts[scripts/]
  Pack --> Bin[bin/install.sh]
  Bin --> Repo[target repo]
  Repo --> Surfaces[all seven workbench surfaces wired]
```

### El diseño del paquete

```
outputs/agent-workbench-pack/
├── AGENTS.md
├── docs/
│   ├── agent-rules.md
│   ├── reliability-policy.md
│   ├── handoff-protocol.md
│   └── reviewer-rubric.md
├── schemas/
│   ├── agent_state.schema.json
│   ├── task_board.schema.json
│   └── scope_contract.schema.json
├── scripts/
│   ├── init_agent.py
│   ├── run_with_feedback.py
│   ├── verify_agent.py
│   └── generate_handoff.py
├── bin/
│   └── install.sh
└── README.md
```

### Lo que se queda dentro, lo que se queda fuera

En:

- Es el contrato.
- Los cuatro guiones de arriba son el tiempo de ejecución.
- Los cuatro documentos son las reglas y la rúbrica.

Fuera:

- Las tareas pertenecen a la tabla del repo objetivo, no al paquete.
- El paquete es agnóstico.
- La manada vive junto a la manada del equipo, no dentro de ella.

### El instalador

Un corto .`bin/install.sh`(o `bin/install.py`):

1. Se niega a instalar en un paquete existente sin `--force`¿ Qué ?
2. Copia el paquete en el repo objetivo.
3. Los cables de CI si un `.github/workflows/`¿Qué es eso?
4. Imprima los siguientes pasos: rellene el tablero, establece comandos de aceptación, ejecuta el script init.

### La versión

El paquete lleva un`VERSION`Los cambios de esquema y los cambios de guión que requieren migraciones golpean el mayor.`agent_state.json`registros de la versión del paquete contra la que se inició.

```figure
wb-pack-install
```

## Construye el mismo

`code/main.py`se ensambla el paquete en `outputs/agent-workbench-pack/`junto a la lección, sembrado con los esquemas y guiones de las lecciones anteriores en esta mini-track y los documentos que ya escribió.

- ¿Qué quieres decir ?

```
python3 code/main.py
```

El guión copia y pin las superficies, escribe el README, imprime el árbol de paquete y sale de cero.

## Modelos de producción en la naturaleza

Un paquete sólo es valioso si sobrevive a horquillas, actualizaciones y un poco hostiles aguas arriba.

**`VERSION` is the contract, not the marketing.**Los problemas principales requieren una migración de estado. los problemas menores requieren una revisión de control. los problemas de parche son sólo de doc. el instalador escribe`.workbench-version`en el repo objetivo en cada instalación; `lint_pack.py`se niega a enviar si la cerradura del objetivo no está de acuerdo con la de la manada `VERSION`Así es como ...`npm`¿ Qué ?`Cargo`, y `pyproject.toml`sobrevivir 10 años de trabajo, nada sobre los agentes cambia las reglas.

**Single source for cross-tool distribution.**Nx barcos uno `nx ai-setup`que se establece `AGENTS.md`¿ Qué ?`CLAUDE.md`¿ Qué ?`.cursor/rules/`¿ Qué ?`.github/copilot-instructions.md`El paquete debe hacer lo mismo; el instalador emite los enlaces de signos (`ln -s AGENTS.md CLAUDE.md`La forma en que se puede utilizar un sistema de codificación es de una manera única, por lo que una sola fuente de verdad se expone a cada agente de codificación.

**`uninstall.sh` that refuses on non-trivial state.**Desinstalar el paquete no debe eliminar los datos del usuario `agent_state.json`¿ Qué ?`task_board.json`, o`outputs/`El desinstalador elimina los esquemas, scripts, documentos y...`AGENTS.md`(con `--keep-agents-md`El Estado pertenece al usuario; el paquete no es su propietario.

**Skill-as-publishable. SkillKit-style distribution.**Los paquetes se envían como una habilidad de SkillKit: `skillkit install agent-workbench-pack`La base de datos es la fuente de la verdad, SkillKit es el canal de distribución, el bloqueo de vendedor se derrumba, las siete superficies permanecen iguales.

## Usalo

Tres lugares los barcos de paquete:

- **As a directory you drop into a repo.** `cp -r outputs/agent-workbench-pack /path/to/repo`¿ Qué ?
- **As a public template repo.**Forja y personalización, con `VERSION`control de la deriva.
- **As a SkillKit skill.**Encableado en el producto de tu agente para que un solo comando lo establezca.

El paquete es la receta, cada instalación es una porción.

## Envío

`outputs/skill-workbench-pack.md`genera un paquete adaptado al proyecto: reglas ajustadas a la historia del equipo, globos de alcance ajustados al repo, dimensiones de rubrica ampliadas con una entrada específica de dominio.

## Los ejercicios

1. Decida cuál de los quintos documentos opcionales merece ser ascendido a la lista canónica.
2. Reescribir el instalador como Python con un `--dry-run`Comparar la ergonomía con la de Bash.
3. Añadir un`bin/uninstall.sh`¿Qué es lo que cuenta como no trivial?
4. Añadir un`lint_pack.py`que falla cuando el paquete se deriva de `VERSION`- Envíala a la CI para el propio reporte del paquete.
5. ¿Cuál es el orden de operaciones que minimiza el tiempo de inactividad?

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Workbench pack | "The starter kit" | A versioned directory carrying all seven surfaces |
| Installer | "Setup script" | `bin/install.sh` that lays the pack down idempotently |
| Pack version | "VERSION" | Major bumps for schema/script changes, patch for doc-only |
| Drop-in pack | "cp -r and go" | Pack works without per-repo customization on day one |
| Forkable template | "GitHub template" | Public repo that GitHub's "Use this template" can clone from |

## Leer más

- Fases 14 · 31 a 14 · 41  cada superficie que este paquete envuelve
- [SkillKit](https://github.com/rohitg00/skillkit) instalar esta habilidad en 32 agentes de IA
- [Nx Blog, Teach Your AI Agent How to Work in a Monorepo](https://nx.dev/blog/nx-ai-agent-skills) Generador de un solo origen en seis herramientas
- [agents.md — the open spec](https://agents.md/) lo que debe implementar el router de su paquete
- [HKUDS/OpenHarness](https://github.com/HKUDS/OpenHarness) Implementación de referencia de un equivalente de envasado
- [andrewgarst/agentic_harness](https://github.com/andrewgarst/agentic_harness) Referencia respaldada por Redis con suite eval
- [Augment Code, A good AGENTS.md is a model upgrade](https://www.augmentcode.com/blog/how-to-write-good-agents-dot-md-files) la barra de calidad de los documentos de empaque
- [Anthropic, Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Anthropic, Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- Fase 14 · 30  Desarrollo de agente basado en la evaluación que consume la puerta de verificación del paquete
- Fase 14 · 41  el índice de referencia antes/después de este paquete mejora en
