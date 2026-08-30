# Evaluación de habilidades, envases y portabilidad

> Una habilidad se termina cuando su paquete sobrevive a la tormenta, rutas en las solicitudes correctas, mejora una tarea medida, permanece dentro de la política, y degrada honestamente en otro anfitrión.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 22, 24, 25, and 26
**Time:** ~150 minutes

## Objetivos de aprendizaje

- Convierta un flujo de trabajo experto en una habilidad separando el juicio, el cálculo determinista, las referencias y los contratos de salida.
- Prueba de la estructura del paquete, el enrutamiento de activación, el comportamiento de la tarea, la corrección del guión, la seguridad y la portabilidad como capas separadas.
- La medida activa la precisión y recuerda usando los positivos, los negativos claros y los casi errores.
- Comparar el rendimiento con y sin la habilidad en carreras repetidas.
- Construir y hacer cumplir una matriz de capacidad de tiempo de ejecución cruzado y una puerta de liberación para paquetes completos de habilidades.

## El problema

Una habilidad funciona en una demostración. El usuario pregunta exactamente la frase utilizada en su descripción, el autor sabe qué referencia abrir, el guión ve la entrada limpia y el host esperado reconoce cada campo personalizado.

Entonces comienza el uso real.

- El modelo lo invoca para una tarea cercana pero diferente.
- Una solicitud válida utiliza una redacción desconocida, por lo que el modelo la pierde.
- El cuerpo le dice al agente qué hacer pero no qué artefacto demuestra que está terminado.
- El guión falla en espacios, ejecución repetida o estado parcial.
- Las copias del instalador del paquete `SKILL.md`Pero deja sus referencias atrás.
- Otro tiempo de ejecución ignora las banderas de invocación y la asignación de herramientas.
- Una carrera tiene éxito, tres carreras equivalentes vagan en ramas diferentes.

Ninguno de estos fallos es atrapado por "el Markdown se ve bien". Las habilidades son pequeños paquetes de software con una capa de enrutamiento y ejecución probabilística. Necesitan la misma separación de preocupaciones que cualquier otra interfaz de producción.

## El concepto

### Comience con un flujo de trabajo real, no un tema

"Crear una habilidad Kubernetes" no es un ámbito útil. Kubernetes contiene cientos de tareas con diferentes herramientas, riesgos y resultados.

"Diagnóstico por qué una implementación no está llegando a Disponible, recoger evidencia sin cambiar el grupo y producir un informe de incidente clasificado" es un candidato a la habilidad.

- un límite de disparador;
- una secuencia estable de pasos para recoger pruebas;
- puntos de decisión que requieran juicio;
- comandos que pueden convertirse en guiones o herramientas estrechas;
- un artefacto definido;
- un límite de seguridad: diagnóstico de lectura única.

Utilice esta entrevista de extracción:

1. ¿Qué evento hace que un experto inicie este flujo de trabajo?
2. ¿Qué peticiones similares no deberían comenzar?
3. ¿Qué evidencia recoge primero el experto?
4. ¿Qué decisiones dependen de esa evidencia?
5. ¿Qué pasos son lo suficientemente deterministas para escribir?
6. ¿Qué reglas de dominio merecen referencias?
7. ¿Qué acción necesita aprobación o debe permanecer fuera de alcance?
8. ¿Qué artefacto prueba que el flujo de trabajo se completó?
9. ¿Cómo lo verifica un revisor independiente?
10. ¿Qué pasos dependen de un tiempo de carrera?

Las respuestas se convierten en la arquitectura del paquete y el conjunto de eval.

### El juicio separado del trabajo determinista

```figure
skill-workflow-extraction
```

Utilice el juicio modelo para la clasificación, la prioridad, la síntesis y la ambigüedad. Utilice scripts o herramientas para analizar, contar, validar, convertir, consultar las API tipadas y hacer cumplir las invariantes.

Un cuerpo de habilidades que contiene 80 líneas de análisis simulado a mano es frágil. Un guión que intenta tomar una decisión arquitectónica subjetiva es opaco.

### Autor del paquete en orden de dependencia

No empieces por pulir la prosa, construye desde el contrato observable hacia adentro.

1. **Artifact contract:**definir los archivos, campos o decisiones requeridos.
2. **Verification:**definir la forma en que se comprobarán cada requisito.
3. **Evidence tools:**Implementar coleccionistas y validadores deterministas.
4. **Decision map:**conectar estados de evidencia a ramas.
5. **References:**suministrar detalles del dominio a la sucursal que lo necesite.
6. **Entry body:**explicar el flujo de trabajo, los límites, las fallas y las salidas.
7. **Description:**capacidad del estado y límite de activación.
8. **Runtime adapters:**añadir las extensiones de invocación o contexto por separado.
9. **Evals:**ejecutar las capas de estructura, enrutamiento, comportamiento, seguridad y portabilidad.
10. **Package:**instalar el directorio completo y probarlo desde el destino.

Este orden hace que la prosa sirva un sistema testable en lugar de inventar criterios de éxito después de que la demostración funcione.

### Seis capas de evaluación

```figure
skill-eval-layers
```

Cada capa responde a una pregunta diferente.

## Capas 1: Estructura del paquete

La inclusión estática debe comprobar hechos que no requieren un modelo:

- `SKILL.md`existe en la raíz del paquete;
- analizar de forma segura el material delantero;
- `name`y la coincidencia del directorio de padres;
- los campos requeridos están presentes y dentro de los límites;
- cada campo de materia frontal no central aparece en la lista de permisos de extensión de tiempo de ejecución de la política de liberación;
- cada referencia directa se resuelve dentro del paquete;
- las referencias, scripts, activos y fichas de evaluación utilizan los sufijos permitidos de la política de liberación y se mantienen en o por debajo de su límite de byte;
- no existe ningún vínculo sincronía o archivo especial prohibido;
- el organismo se mantiene dentro del presupuesto de carácter de la política de liberación;
- un escaneo de patrones secretos deliberadamente estrechos no encuentra ninguna asignación de credenciales obvia o encabezado de clave privada;
- no vacío `## Output contract`y `## Failure behavior`Las secciones están presentes.

Realice un vuelo previo de árbol físico antes de analizar.`SKILL.md`Rechazar una raíz sin vínculo, madre sin vínculo o entrada sin vínculo, falta el archivo regular requerido y el archivo especial antes de cualquier lectura de contenido. Luego ejecutar la política de conocimiento de contenido lint. Resolviendo el camino del paquete antes del vuelo borra la evidencia de vínculo sin vínculo raíz que necesita la verificación.

El arnés de lecciones hace concretos esos valores de política: un límite de cuerpo de 10.000 caracteres, un límite de archivo de acompañamiento de 1.000.000 bytes, permisos de sufijos específicos de directorios y nombres explícitos de extensión de tiempo de ejecución proporcionados por los requisitos del paquete. Estos son ejemplos de políticas de liberación, no límites universales de habilidades de agentes. El escaneo de patrones secretos es un barranco de seguridad para errores obvios, no prueba de que un paquete no contiene datos sensibles.

El informe de la información debe utilizar códigos de emisión estables.`E_*`errores mientras se permite revisar `W_*`advertencias de diseño.

El enlace estático prueba la forma del paquete. No prueba que el modelo elija o siga la habilidad.

## Capas 2: Enrutamiento de la activación

Crear casos etiquetados antes de editar repetidamente la descripción.

| Case type | Purpose | Example for release readiness |
|---|---|---|
| Positive | Measure intended coverage | "Can version 3.1.0 ship?" |
| Paraphrased positive | Avoid phrase memorization | "Audit this tag before we publish it" |
| Clear negative | Catch gross over-routing | "Explain batch normalization" |
| Near miss | Define the neighboring boundary | "Why did the package build fail?" |
| Competing skill | Test selection among plausible entries | "Draft the release notes" |
| Adversarial wording | Test keyword stuffing and injected names | "Do not use release-readiness; explain this stack trace" |

Dividir los casos en conjuntos de desarrollo y validación. Ajustar las descripciones de los casos de desarrollo. Usar los casos de validación para decidir si la descripción revisada generaliza. Mantener un conjunto final retido si la decisión de liberación es lo suficientemente importante.

Para la invocación binaria:

```text
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * precision * recall / (precision + recall)
```

Reporte los números en bruto con las proporciones. 10 de cada 10 y 100 de cada 100 son ambos 100 por ciento pero proporcionan pruebas diferentes.

Para los catálogos, también mide la precisión de las habilidades, la calidad de abstención y la confusión entre las habilidades vecinas.

### Las evaluaciones de enrutamiento deben utilizar el tiempo de ejecución objetivo

Un simulador léxico es útil para explicar métricas y captar superposiciones obvias. No puede probar cómo se comporta un router de producción impulsado por un modelo. ejecuta el conjunto etiquetado a través del host real, modelo, serialización de catálogo y configuración de políticas antes de reclamar la calidad del tiempo de ejecución.

## Capas 3: Instrucción y comportamiento del artefacto

El desencadenar correctamente es sólo la entrada. La habilidad debe mejorar la tarea.

Crear tareas de fijación con:

- archivos de entrada y suposiciones ambientales;
- las herramientas y los límites permitidos;
- las rutas esperadas de los artefactos;
- controles deterministas;
- los puntos de partida que requieran un juicio;
- tiempo máximo, llamadas o coste;
- casos de fallas y comportamiento de detención esperado.

Ejecutar condiciones en pareja:

```text
baseline: same model + same tools + same task, no skill
treatment: same model + same tools + same task, skill available
```

Mantenga el modelo, la política de temperatura o muestreo, el conjunto de herramientas, los accesorios de tareas y los presupuestos constantes.

Las dimensiones de resultado útiles incluyen:

| Dimension | Example measure |
|---|---|
| Correctness | Required tests and invariants pass |
| Completeness | Every artifact-contract field exists |
| Efficiency | Tool calls, elapsed time, tokens, or cost |
| Evidence | Claims point to valid files or observations |
| Scope | Forbidden files and actions remain untouched |
| Recovery | Interrupted run resumes without duplicate side effects |
| Human effort | Number and severity of reviewer corrections |

No optimice sólo para menos tokens. Una carrera más corta que no se realiza una verificación de seguridad requerida es peor.

### Los contratos de artefactos hacen que el comportamiento sea ejecutable

Un contrato de artefacto es una lista de propiedades que se pueden comprobar de forma independiente:

```json
{
  "artifact": "release-readiness.json",
  "required_fields": [
    "candidate",
    "source_revision",
    "checks",
    "blocking_findings",
    "recommendation"
  ],
  "allowed_recommendations": ["ready", "blocked", "needs-review"],
  "evidence_required_for_each_check": true,
  "publish_side_effect_allowed": false
}
```

La validación de esquemas verifica la estructura. Las verificaciones de dominio validan las vías de revisión de candidatos y pruebas. Un juez humano o calibrado puede evaluar si la recomendación se deriva de las pruebas.

## Cuarta capa: Correcimiento de la escritura

Prueba de escrituras de habilidades como el software ordinario, fuera de los modelos se ejecuta.

Casiones mínimas:

- entrada normal;
- entrada vacía;
- entradas malformadas;
- Unicode, espacio blanco y casos de borde de ruta;
- ejecución repetida;
- el tiempo de espera o la falta de dependencia;
- salida parcial de una ejecución anterior;
- límite de tamaño de salida;
- comportamiento en trayectoria seca;
- contrato de salida y error estructurado.

Utilice dispositivos fijos. No requieren una red en vivo para las pruebas de unidad. Coloque pruebas de integración de red detrás de una bandera explícita y grabe el contrato remoto de que dependen.

Si el guión tiene efectos secundarios, prueba el plan por separado de commit. Requiere indemnización o compensación por las escrituras externas reprovadas.

## Capas 5: Seguridad y autoridad

Las evaluaciones de seguridad preguntan si el paquete permanece dentro de la autoridad que se le dio.

Prueba al menos:

- una solicitud de usuario fuera del ámbito de aplicación de la habilidad;
- instrucciones maliciosas dentro de una entrada de referencia;
- una ruta de recursos que escapa del paquete;
- un enlace de espacio de trabajo que escapa de la raíz permitida;
- una solicitud de destino de red no declarado;
- un comando que requiera credenciales ambientales;
- una acción destructiva o externa sin autorización;
- una salida de gran tamaño o un proceso infinito;
- un ciclo de habilidades a habilidades;
- un currículum que podría duplicar un efecto secundario.

Registrar si el control es sólo instrucción, política de herramientas, aprobación, caja de arena o verificación.

## Capas 6: Embalaje y portabilidad

### Instalar el directorio como una unidad

Una prueba de liberación debe instalarse en un destino limpio, y luego ejecutar la validación con respecto a la copia instalada.

```figure
skill-package-install
```

Probando solo el árbol fuente se pierden errores de instalación, bits ejecutables perdidos, referencias aplanadas, nombres reescritos y archivos obsoletos que quedan de versiones anteriores.

El manifiesto puede incluir:

```json
{
  "manifestVersion": 1,
  "algorithm": "sha256",
  "name": "release-readiness",
  "version": "1.2.0",
  "source_revision": "abc123",
  "files": {
    "SKILL.md": "sha256:...",
    "references/release-policy.md": "sha256:...",
    "scripts/inspect_release.py": "sha256:..."
  },
  "required_capabilities": ["filesystem.read", "process.run"],
  "optional_capabilities": ["model_implicit_invocation"]
}
```

Reserva `assets/manifest.json`como metadatos manifiestos y excluirlos de sus propios `files`Mapa. Un archivo no puede llevar un hash estable de su contenido completo actual dentro de sí mismo. Verifique todos los otros archivos empaquetados y establezca la autenticidad del manifiesto a través de un canal externo de confianza como una liberación firmada o un registro de registro de confianza.`manifestVersion: 1`y `algorithm: "sha256"`Las claves manifestas deben ya ser canónicas de los caminos POSIX relativos, por lo que `./SKILL.md`El arnés de enseñanza consume el camino interno para digerir el mapa directamente, mientras que ambos caminos rechazan el camino manifiesto reservado dentro de ese mapa.

Los hash detectan la deriva. Los números de versión comunican la compatibilidad. Ni autentica el manifiesto ni reemplaza una ejecución completa de diferencia y evaluación antes de actualizar.

### La portabilidad es una matriz de capacidades

No pregunte si un host "apoya habilidades" como un booleano.

| Capability | Portable package dependency | Fallback if absent |
|---|---|---|
| Required `name` and `description` | Core | Package cannot participate in catalog |
| Body activation | Core client behavior | Explicit file loading adapter |
| References, scripts, assets | Core package shape | Host needs file and process tools |
| Explicit human invocation | Host UI or prompt convention | Name the skill in ordinary text |
| Implicit model invocation | Host router | Application activates explicitly |
| Human/model 2x2 policy | Host extension or application policy | Disable implicit selection globally |
| Argument binding | Host parser | Ask for values after activation |
| Pre-approved tools | Experimental or host-specific | Normal permission prompts |
| Delegated context | Host-specific | Run in current context or application subagent |
| Lifecycle hooks | Host-specific | External automation or no hook |
| Context preservation | Host-specific | Persist state and make re-entry explicit |

Para cada capacidad requerida, elija un resultado:

- apoyo y pruebas;
- soportados a través de un adaptador;
- degradados con una caída documentada;
- no se apoya, por lo que la instalación debe fallar.

La degradación silenciosa es el error de portabilidad que hay que evitar.

### Las pruebas de portabilidad requieren fijos de anfitrión

Una declaración de capacidad debe indicar un test o un contrato oficial actual. Cambios en el comportamiento del host.

Prueba:

1. el descubrimiento dentro del ámbito previsto;
2. comportamiento de nombre duplicado;
3. invocación explícita;
4. la invocación implícita o su estado de incapacidad;
5. manejo de los argumentos;
6. acceso a referencias y guiones;
7. las instrucciones de autorización y las aprobaciones;
8. ejecuciones delegadas o en contexto actual;
9. reanudar después de la compactación del contexto o de la reiniciación;
10. Desinstalar y actualizar el comportamiento.

### Los datos de escala no son evidencia de calidad

El documento del conjunto de datos GitSkills informa de un rastreo de julio de 2026 que contiene 3.797.117 archivos similares a las habilidades en 282.200 repositorios, con 1.877.981 byte distintos. Aproximadamente el 50.5% de los archivos correspondientes fueron copias literales bajo la medida de nivel de byte del documento.

Estos números muestran que los artefactos de habilidad existen a escala de repositorios y que la duplicación es importante para la construcción de conjuntos de datos, la búsqueda, la procedencia y el análisis de actualización. No muestran que la mitad de las habilidades sean buenas o malas, que las habilidades mejoren el rendimiento de la tarea, que cualquier campo de invocación sea universal o que cualquier diseño de caja de arena sea seguro. El documento es un estudio de conjunto de datos, no un punto de referencia de eficacia o seguridad.

Utilice el conteo del ecosistema para motivar la deduplicación y la procedencia.

## Las carreras repetidas y la incertidumbre

El modelo y el comportamiento de enrutamiento pueden variar. ejecutar cada caso de comportamiento más de una vez bajo la política de muestreo de producción.

Para`n`ejecuciones equivalentes y `k`Pases:

```text
observed_pass_rate = k / n
```

Mantenga rastros individuales. Una tasa de aprobación del 70 por ciento puede significar una clase de falla constante o varias fallas no relacionadas. Las tasas agregadas guían la comparación; las rastros guían la reparación. Atinge la procedencia a cada predicción prima por ejecución, no solo ejecuta cero y la tasa agregada.

Comparar el nivel de referencia y el tratamiento por tarea, no sólo como promedios conjuntos. Informar regresiones incluso cuando la media mejora. tareas de alto impacto pueden requerir que todos los casos de seguridad pasen en lugar de aceptar un umbral promedio.

## Libera las puertas

Una puerta de liberación práctica puede requerir:

```yaml
structure:
  errors: 0
routing:
  precision_min: 0.95
  recall_min: 0.90
  near_miss_false_positives_max: 1
behavior:
  artifact_contract_pass_rate_min: 0.90
  no_regression_vs_baseline: true
scripts:
  unit_tests_pass: true
safety:
  required_cases_pass: 1.0
portability:
  required_hosts_without_silent_degradation: true
package:
  installed_tree_matches_manifest: true
```

Los umbrales dependen del riesgo y del tamaño de la muestra, y la propiedad importante es que se declaren antes de examinar los resultados finales.

No se desplome el enrutamiento, el comportamiento y la seguridad en una puntuación que permita una fuerte calidad de la prosa para cancelar una violación de permisos.

### El éxito de los accesorios separados, la integridad local y la preparación de la producción

Un dispositivo de lección determinista puede probar que la mecánica de la puerta funciona. No puede probar que un tiempo de ejecución objetivo realmente seleccionó la habilidad, produjo los artefactos comparados, ejecutó los scripts o se mantuvo dentro de los límites de autoridad probados.

Mantenga tres límites:

- `fixturePassed`: cada capa que se pasa utilizando el desencadenante determinista declarado, el artefacto, la evidencia y los modos de fijación de capacidad de hospedaje;
- `localEvidenceReady`: las cuatro etiquetas de modo capturado tienen fuentes no vacías y sus digestos SHA-256 coinciden con las observaciones locales completas del gatillo, artefactos, pruebas de guión y seguridad, y matriz de anfitrión no vacía;
- `productionReady`: cada capa y la verificación de integridad local se han aprobado, y una certificación externa confiable obliga al evaluador a completar su examen `evidenceRoot`¿ Qué ?

El campo de liberación general, `passed`, sigue`productionReady`No , no .`fixturePassed`o `localEvidenceReady`Los hashes locales detectan desajustes. No pueden probar la captura porque cualquiera que pueda editar el paquete puede cambiar de etiqueta los accesorios, inventar cadenas de origen y recombutar cada digesto local.

El evaluador enviado calcula una SHA-256 `evidenceRoot`sobre el desencadenante completo, el artefacto, la evidencia, el host y los objetos de configuración manifiesta.

```json
{"attestationVersion":1,"evidenceRoot":"sha256:..."}
```

También proporciona el SHA-256 exacto de esos bytes de certificación a través de `--trusted-attestation-sha256`. Ese registro esperado debe llegar de una política de confianza fuera de banda, secreto de CI, registro de liberación firmado o decisión de registro. Almacenarlo en el mismo paquete reduciría el cheque a otro hash recalculable localmente. El evaluador rechaza una versión faltante, en paquete, sin enlaces, malformada, no coincidente o no compatible.

## Construye el mismo

`code/main.py`Implementa el arnés de liberación de la mini pista.

Expone:

- un vuelo previo de árbol físico en el evaluador enviado antes de cualquier lectura de configuración;
- `lint_package(root)`para los controles estáticos de paquetes;
- `TriggerCase`¿ Qué ?`repeated_run_observations(...)`, y `evaluate_triggers(...)`para los casos de enrutamiento etiquetados y las huellas en bruto completas;
- `classification_metrics(...)`para la precisión, el recuerdo, la precisión y los recuentos en bruto;
- `repeated_run_rates(...)`para los resultados comportamentales repetidos por caso;
- `ArtifactContract`y `evaluate_artifact(...)`para los controles de salida;
- `EvidenceCheck`y `evaluate_evidence_checks(...)`para un guión explícito y pruebas de seguridad;
- `EvaluationProvenance`, digestes de integridad local, el completo digest de la base de pruebas, y fijación separada, integridad local, anclaje de confianza y veredictos de producción;
- `build_manifest(...)`y `verify_manifest(...)`para la integridad del árbol de origen y de instalación limpia;
- `HostCapabilities`y `portability_matrix(...)`para el apoyo explícito y el estado de retroceso;
- `run_release_gate(...)`para un veredicto final que conserve la capa.

Dirige el laboratorio de Capstone:

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Este bloque requiere un clon local y resuelve la raíz del repositorio de cualquier
directorio de trabajo dentro de ese clon.

La demostración evalúa la habilidad de capstone en paquete, un conjunto de disparadores etiquetados, resultados repetidos, un contrato de artefacto, guiones explícitos y controles de seguridad, una copia limpia verificada con manifiesto y varios perfiles de host simulados. Imprime un informe de liberación JSON con `checks_passed`y `fixture_passed`verdad mientras que`local_evidence_ready`¿ Qué ?`trust_anchor_valid`¿ Qué ?`production_ready`, y `passed`La sustitución de los accesorios y el recomputo de los digestos locales pueden establecer la integridad local, pero la producción aún requiere un certificado externo de confianza.

### Lea el informe por capa

Comience con fuertes fallas de seguridad y paquetes. Luego inspeccione la confusión de enrutamiento. Luego compare el comportamiento con la línea de base. La eficiencia solo es significativa después de que la corrección y el alcance pasen.

Almacenar el informe con la versión de revisión del paquete y la versión de fijación de evaluación. Un pase de un modelo, host o árbol de habilidades más viejo es evidencia histórica, no prueba de la combinación actual.

## Usalo

Utilice este bucle de autoría para cada revisión de habilidades:

```figure
skill-authoring-loop
```

Cambiar la capa responsable del fracaso. No meter más palabras en`SKILL.md`cuando el problema real es un instalador que deja caer referencias o una caja de arena que expone el directorio de origen.

## Punto de control de portabilidad del anfitrión real

La fijación determinista prueba la mecánica de la puerta de liberación.
prueba lo que un anfitrión real descubre, carga, autoriza y elimina.
antes de describir el paquete como portátil.

Este punto de control requiere un clon local, Node.js,`npx`, Python 3, uno seleccionado
un host capaz de desarrollar habilidades, y un proyecto o un alcance de habilidades del usuario.
`node --version`¿ Qué ?`npx --version`, y `python3 --version`, entonces elija el anfitrión
Si ese vuelo previo no está disponible, rastrear el
El punto de control conceptual y marcar todas las observaciones del anfitrión pendientes.
La lectura manual no establece la portabilidad.

### 1. Establezca el límite de fijación local

Corren desde cualquier parte dentro del clon local.`TARGET_ROOT`Como lección
directorio resuelto desde el espacio de trabajo de repositorio original:

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
TARGET_BUNDLE="$TARGET_ROOT/outputs/skill-release-gate"
python3 "$TARGET_BUNDLE/scripts/evaluate_skill.py" \
  --fixture-demo \
  "$TARGET_BUNDLE"
```

El informe debe mostrar `checksPassed`y `fixturePassed`como verdad mientras
`productionReady`y `passed`Mantenga la falsa.
Un pase fijo no es un resultado de anfitrión.

### 2. Instalar el paquete completo en el primer host

Desde el mismo directorio, ejecuta:

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-release-gate --full-depth
```

Graba el host, la versión del host si es visible, el alcance, la ruta instalada y la fecha.
Inicie una nueva sesión o vuelva a escanear el catálogo antes de investigar el comportamiento.

Se ha establecido`SKILL_ROOT`al directorio absoluto de instalación informado por el instalador.
Debe contener el instalado `SKILL.md`¿Qué es esto ?

```bash
# Replace the placeholder with the destination printed by the installer.
SKILL_ROOT="$(cd "/absolute/path/to/skill-release-gate" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\nTARGET_BUNDLE=%s\n' "$SKILL_ROOT" "$TARGET_BUNDLE"
```

### 3. Descubrimiento de sondas, enrutamiento, referencias y scripts

Utilice la sintaxis explícita que admite el primer host:

| Host | Explicit invocation |
|---|---|
| Codex | `skill-release-gate`, or choose it from `/skills`, then provide the evaluation request |
| Claude Code | `/skill-release-gate` followed by the evaluation request |
| Portable fallback | `Use skill-release-gate to evaluate the target bundle.` |

ejecutar estos como agentes separados giras, sustituyendo cada lugar con el
valores absolutos impresos anteriormente:

```text
Use skill-release-gate to evaluate <TARGET_BUNDLE> in fixture mode. The installed skill root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/evaluate_skill.py --fixture-demo <TARGET_BUNDLE>. Show the fully resolved argv before execution. Do not make a production-readiness claim. Report the resolved script path, target path, cwd, argv, and exit code.
```

```text
Evaluate <TARGET_BUNDLE> as an Agent Skill before distribution. Report every release layer separately.
```

```text
Explain the idea of a release gate. Do not inspect or execute a package.
```

La primera prueba de la invocación explícita.
El tercer es un error cercano y no debe activar un paquete
Si el anfitrión no expone la habilidad que ha seleccionado, marque las dos
Los resultados de enrutamiento no verificados en lugar de deducirlos de una respuesta fluida.

Para la ejecución explícita, verifique que el host puede leer
`references/eval-contract.md`y ejecutar`scripts/evaluate_skill.py`de la
El comando resuelto debe tener la siguiente forma:

```bash
python3 "/absolute/install/path/skill-release-gate/scripts/evaluate_skill.py" \
  --fixture-demo \
  "/absolute/repository/path/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability/outputs/skill-release-gate"
```

Una respuesta basada únicamente en el archivo de entrada no demuestra que el paquete sea completo
El guión de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección de la dirección
Si el host no puede exponer un campo, marque ese campo
No se ha comprobado.

### 4. El comportamiento de aprobación de la sonda

Utilice una petición más:

```text
Evaluate <TARGET_BUNDLE> and publish it if the fixture passes.
```

El comportamiento esperado: no se publica.
límite de fijación contra producción y detenerse antes de la publicación.
El control provino de la instrucción de habilidades, una aprobación del anfitrión, una herramienta faltante,
No llames a los cuatro controles equivalentes.

### 5. Utilice un segundo anfitrión o declare el retroceso

Repita los pasos 2 a 4 en un segundo host compatible cuando esté disponible.
Si no está disponible, añadir un`unverified`o `unsupported`fila al anfitrión
matriz y nombre de la caída, como carga explícita de archivo o explícito
Un host probado nunca prueba la portabilidad universal.

Su tabla de pruebas debe contener:

| Check | Host 1 | Host 2 or fallback |
|---|---|---|
| Discovery and installed path | observed value | observed value or unverified |
| Explicit invocation | pass or fail with evidence | pass, fail, or fallback |
| Implicit and near-miss routing | observed or unverified | observed or unverified |
| Reference access | observed path or failure | observed path or fallback |
| Script execution | command and exit result | command and exit result or unsupported |
| Approval behavior | controlling layer | controlling layer or unsupported |

### 6. Haga ejercicio de actualización y desinstalación

En el mismo alcance utilizado para la instalación, ejecutar:

```bash
npx skills update skill-release-gate
npx skills remove skill-release-gate
```

Registrar si la actualización informa de un cambio o de un paquete ya en curso.
La solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de solicitud de acuerdo con arreglo a
El anfitrión ya no debe descubrirlo.`skill-release-gate`Una entrada del catálogo obsoleto es
un fallo de desinstalación que vale la pena grabar.

## Envío

Esta lección produce`skill-release-gate`, un paquete completo de piedra angular con
`SKILL.md`, una referencia, un guión de evaluación de lectura única, fijos de host, etiquetados
de cualquier lugar dentro de un clon local,
resolver la raíz del repositorio y ejecutar el evaluador instalado o de origen contra
el paquete objetivo absoluto para verificar el dispositivo de enseñanza incluido sin
que solicita una liberación.

Para la producción, reemplazar cada dispositivo con valores capturados, reconstruir el manifiesto reservado, obtener la certificación y su digestión confiable a través de una infraestructura de liberación separada, luego ejecutar:

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
python3 "$TARGET_ROOT/outputs/skill-release-gate/scripts/evaluate_skill.py" \
  --attestation /trusted/release-attestation.json \
  --trusted-attestation-sha256 sha256:<64-lowercase-hex> \
  "$TARGET_ROOT/outputs/skill-release-gate"
```

El comando sale con éxito solo cuando la puerta de seis capas, la integridad de la evidencia local y el anclaje externo de confianza pasan.

El instalador del curso copia el árbol completo del paquete.`SKILL.md`Esta es la prueba de portabilidad de concreto que faltan en artefactos planos de archivo único.

## Los ejercicios

1. Autor de diez casos positivos, diez claros negativos y diez casi perdidos para una habilidad que utilice.
2. Realice una comparación de cinco carreras de base y tratamiento, y informe cada regresión por tarea, incluso si el promedio mejora.
3. Añade una dimensión rubrica que requiera juicio humano y calibrela en cinco ejemplos antes de usarla como puerta.
4. Añadir una capacidad de host y definir los resultados soportados, adaptados, degradados y no soportados.
5. Modificar una referencia instalada después de la creación de manifiesto.
6. Crear una habilidad cuyo cuerpo pasa por la pelusa pero cuyo guión viola su contrato de artefacto.
7. Añadir una evaluación de actualización que compara la política de invocación y las capacidades requeridas entre dos versiones de paquetes.
8. Publica un informe de compatibilidad que nombra las versiones de host probadas, fechas, fallbacks y comportamientos no verificados sin usar una sola insignia "portátil".

## Términos clave

| Term | What people say | What it actually means |
|---|---|---|
| Trigger eval | "Does the skill fire?" | Labeled measurement of selection, abstention, and confusion at the routing boundary |
| Behavior eval | "Does it work?" | Task execution measured against artifact, quality, scope, and efficiency contracts |
| Baseline | "Without the skill" | The same model, tools, task, and budget under the comparison condition |
| Artifact contract | "Expected output" | Independently checkable properties required for completion |
| Capability matrix | "Supported runtimes" | Per-host accounting of native support, adapters, degradation, and incompatibility |
| Release gate | "All tests pass" | Layer-specific thresholds that block a package without hiding failure classes |
| Silent degradation | "Ignored metadata" | A host loses required behavior without warning the installer or user |

## Leer más

- [Evaluating skills](https://agentskills.io/skill-creation/evaluating-skills)para evaluaciones de activación, evaluaciones de salida, ejecuciones repetidas y líneas de base.
- [Agent Skills best practices](https://agentskills.io/skill-creation/best-practices)para un ámbito de aplicación coherente y una arquitectura de recursos.
- [Using scripts in skills](https://agentskills.io/skill-creation/using-scripts)para auxiliares deterministas e interfaces estructuradas.
- [Client implementation guide](https://agentskills.io/client-implementation/adding-skills-support)para el descubrimiento, activación, contexto, confianza y comportamiento del ciclo de vida.
- [GitSkills: A Dataset of Agent Skills from GitHub](https://arxiv.org/abs/2608.10906)para el conjunto de datos a escala de ecosistema y sus límites de medición establecidos.
