# ¿Por qué multi-agente?

> Un agente golpea una pared, el movimiento inteligente no es un agente más grande, sino más agentes.

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Objetivos de aprendizaje

- Identificar el límite máximo de un agente único (desbordamiento de contexto, experiencia mixta, cuello de botella secuencial) y explicar cuándo dividir en múltiples agentes es el movimiento correcto
- Compara patrones de orquestación (pipelina, ventilador paralelo, supervisor, jerárquico) y seleccione el correcto para una estructura de tarea dada
- Diseñar un sistema multiagente con límites claros de rol, estado compartido y contrato de comunicación
- Analiza las diferencias entre la complejidad de varios agentes (latencia, costo, dificultad para deshacerse) y la simplicidad de un solo agente

## El problema

Construiste un único agente en la Fase 14. Funciona. Puede leer archivos, ejecutar comandos, llamar a las API y razonar sobre los resultados. Luego lo apuntas a una base de código real: 200 archivos, tres idiomas, pruebas que dependen de la infraestructura y un requisito para investigar las API externas antes de escribir código.

El agente se ahoga. No porque el LLM sea tonto, sino porque la tarea excede lo que un bucle de agente puede manejar. La ventana de contexto se llena de contenido de archivo. El agente olvida lo que leyó hace 40 llamadas de herramienta. Trata de ser un investigador, un codificador y un revisor a la vez, y hace mal los tres.

Este es el techo de un solo agente.

- **More context than fits in one window**- leer 50 archivos se hace pasar de 200 mil tokens
- **Different expertise at different stages**- la investigación requiere una motivación diferente a la generación de código
- **Work that can happen in parallel**- ¿Por qué leer tres archivos secuencialmente cuando puedes leerlos simultáneamente?

## El concepto

### El techo de un solo agente

Un agente único es un bucle, una ventana de contexto, un sistema de instrucciones.

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

Tres cosas se rompen:

1. **Context saturation**En la curva 30, el agente ha consumido 150 mil tokens de contenido de archivos, salidas de comandos y razonamiento previo.

2. **Role confusion**- un mensaje de sistema que dice "es un investigador, codificador, revisor y probador" produce un agente que hace medio estudio, medio código y nunca termina la revisión.

3. **Sequential bottleneck**- el agente lee el archivo A, luego el archivo B, luego el archivo C. Tres llamadas en serie de LLM.

### La solución multiagente

Divide el trabajo. Dale a cada agente un trabajo, una ventana de contexto y una llamada de sistema sintonizada para ese trabajo:

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

Cada agente tiene:
- Una solicitud de sistema enfocada ("Usted es un revisor de código. Su único trabajo es encontrar errores. ")
- Su propia ventana de contexto (no contaminada por el trabajo de otros agentes)
- Un contrato de entrada/salida claro (recibe notas de investigación, código de salida)

### Sistemas reales que hacen esto

**Claude Code subagents**- cuando Claude Code genera un subagente con`Task`El padre mantiene su contexto limpio, el niño hace un trabajo enfocado y devuelve un resumen.

**Devin**- ejecuta un agente de planificación, un agente de codificación y un agente del navegador. El planificador divide el trabajo en pasos. El codificador escribe código. El navegador investiga la documentación. Cada uno tiene un contexto separado.

**Multi-agent coding teams (SWE-bench)**- los sistemas de mejor rendimiento en el banco SWE utilizan un investigador que lee la base de código, un planificador que diseña la corrección y un codificador que la implementa.

**ChatGPT Deep Research**- genera múltiples agentes de búsqueda en paralelo, cada uno explorando un ángulo diferente, y luego sintetiza los resultados.

### El espectro

El multi-agente no es binario, es un espectro:

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**Single agent**- Un bucle, un prompt.

**Subagents**- un padre da a luz a los hijos para subtareas enfocadas el padre mantiene el plan los niños reportan esto es lo que hace Claude Code

**Pipeline**- agentes se ejecutan en secuencia. la salida del agente A se convierte en la entrada del agente B. Es bueno para flujos de trabajo en etapas: investigación -> código -> revisión -> prueba.

**Team**Los agentes funcionan en paralelo con un bus de mensajes compartidos cada uno tiene un papel un orquestrador coordina es bueno cuando se necesitan diferentes habilidades simultáneamente

**Swarm**- muchos agentes idénticos o casi idénticos con estado compartido. sin orquestaje fijo. agentes recogen el trabajo de una cola. bueno para tareas paralelas de alto rendimiento.

### Los cuatro patrones multi-agentes

#### Modelo 1: oleoductos

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

Cada agente transforma los datos y los transmite, sencillo de razonar, el fracaso en una etapa bloquea el resto.

#### Modelo 2: Descanso / In-fan

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

Dividir el trabajo en agentes paralelos, luego fusionar los resultados.

#### Modelo 3: Orquesta-trabajador

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

Un orquestrador inteligente decide qué hacer, delega a los trabajadores y sintetiza los resultados.

#### Patrón 4: El grupo de compañeros

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

No hay orquestaje central, los agentes se comunican entre pares, las decisiones surgen de la interacción, más difícil de deshacer, pero se extiende a muchos agentes.

### Cuando NO utilizar multi-agentes

Cada mensaje entre agentes es un punto de falla potencial. Desarreglar pasa de "leer una conversación" a " rastrear mensajes a través de cinco agentes".

**Stay single-agent when:**
- La tarea se ajusta a una ventana de contexto (bajo ~ 100k tokens de datos de trabajo)
- No necesitas diferentes instrucciones del sistema para diferentes etapas
- La ejecución secuencial es lo suficientemente rápida.
- La tarea es lo suficientemente simple como para que dividirla añade más gastos generales que valor

**The complexity cost:**
- Cada límite de agente es un paso de compresión perdida: el contexto completo del agente A se resume en un mensaje para el agente B
- La lógica de coordinación (quién hace qué, cuándo, en qué orden) es su propia fuente de errores
- Aumenta la latencia: N agentes significa N llamadas de LLM en serie mínimo, más si necesitan hablar de un lado a otro
- Multiplice de costos: cada agente quema tokens de forma independiente

Regla de oro: si una tarea requiere menos de 20 llamadas de herramientas y encaja en 100k tokens, manténla un agente único.

```figure
swarm-messages
```

## Construye el mismo

### Paso 1: El agente único sobrecargado

Aquí hay un solo agente que intenta hacer todo. Tiene un enorme sistema de respuesta y una ventana de contexto que contiene investigación, código y reseñas:

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Problemas con este enfoque:
- La ventana de contexto crece con cada etapa.
- El sistema de instrucción es genérico. No se puede ajustar para cada etapa.
- Nada funciona en paralelo.

### Paso 2: Agentes especializados

Ahora lo dividimos.

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

Cada especialista tiene un mensaje enfocado, cada uno tiene una ventana de contexto limpia con sólo la entrada que necesita.

### Paso 3: Coordinar a través de mensajes

Envía a los especialistas con un mensaje explícito:

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Cada agente recibe sólo los mensajes dirigidos a él, sin contaminación de contexto, los 50 mil tokens de lectura de documentación del investigador nunca entran en el contexto del revisor.

### Paso 4: Compare

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

La versión multi-agente utiliza más tokens totales (tres agentes, tres llamadas LLM separadas), pero el contexto de cada agente se mantiene limpio.

## Usalo

Esta lección produce una invitación reutilizable para decidir cuándo ir a multi-agente.`outputs/prompt-multi-agent-decision.md`¿ Qué ?

## Los ejercicios

1. Añadir un cuarto especialista: un agente "tester" que recibe código del codificador y revisa la retroalimentación del revisor, luego escribe pruebas
2. Modificar la línea de tubería para que el revisor pueda enviar retroalimentación al codificador para un bucle de revisión (máximo 2 rondas)
3. Convierta la tubería secuencial en un ventilador: ejecuta el investigador y un agente "analista de requisitos" en paralelo, luego fusione sus salidas antes de pasar al codificador

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Swarm | "A hive mind of AI agents" | A set of peer agents with shared state and no fixed leader. Behavior emerges from local interactions. |
| Orchestrator | "The boss agent" | An agent whose tools include spawning and managing other agents. It plans and delegates but may not do the actual work. |
| Coordinator | "The traffic cop" | A non-agent component (often just code, not an LLM) that routes messages between agents based on rules. |
| Consensus | "The agents agree" | A protocol where multiple agents must reach agreement before proceeding. Used when conflicting outputs need resolution. |
| Emergent behavior | "The agents figured it out themselves" | System-level patterns that arise from agent interactions but were not explicitly programmed. Can be useful or harmful. |
| Fan-out / fan-in | "Map-reduce for agents" | Splitting a task across parallel agents (fan-out), then combining their results (fan-in). |
| Message passing | "Agents talk to each other" | The communication mechanism between agents: structured data sent from one agent to another, replacing shared context windows. |

## Leer más

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977)- estudio de patrones de múltiples agentes
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155)- El marco de conversación multiagente de Microsoft
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code)- cómo Claude Code delega con la tarea
- [CrewAI documentation](https://docs.crewai.com/)- marco multiagente basado en el papel
