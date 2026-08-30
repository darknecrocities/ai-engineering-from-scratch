# A2A  Protocolo entre agentes

> MCP es agente a herramienta. A2A (Agent2Agent) es un protocolo abierto para permitir que los agentes opacos construidos en diferentes marcos colaboren. Lanzado por Google en abril de 2025, donado a la Fundación Linux en junio de 2025, alcanzando la versión 1.0 en abril de 2026 con más de 150 partidarios, incluidos AWS, Cisco, Microsoft, Salesforce, SAP y ServiceNow. Absorbió el ACP de IBM y añadió la extensión de pagos AP2. Esta lección incluye la tarjeta de agente, el ciclo de vida de la tarea y los dos vínculos de transporte.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## Objetivos de aprendizaje

- Distinguir entre el agente a la herramienta (MCP) y los casos de uso entre agentes (A2A).
- Publica una tarjeta de agente en `/.well-known/agent.json`con habilidades y metadatos de los puntos finales.
- Siga el ciclo de vida de la tarea (sendido → trabajando → requerido de entrada → completado / fallado / cancelado / rechazado).
- Utilice los mensajes con partes (texto, archivo, datos) y artefactos como salidas.

## El problema

Un agente de servicio al cliente debe delegar la redacción de informes a un agente de redacción especializado.

- Funciona pero cada emparejamiento es una sola vez.
- Base de código compartida requiere que los dos agentes ejecuten el mismo marco.
- MCP: no encaja: MCP es para llamar a herramientas, no para dos agentes colaborando mientras se conserva el razonamiento interno opaco de cada agente.

A2A llena la brecha. Modela la interacción como un agente envía una tarea a otro, con un ciclo de vida, mensajes y artefactos. El estado interno del agente llamado se mantiene opaco.

A2A es el protocolo "dejen que los agentes de los marcos hablen entre sí".

## El concepto

### Agente Carte

Cada agente de A2A publica una tarjeta en `/.well-known/agent.json`¿Qué es esto ?

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

El descubrimiento se basa en URL: trae la tarjeta, aprende la URL del punto final A2A, enumera habilidades.

### Carteles de agente firmados (AP2)

La extensión AP2 (septiembre 2025) añade firmas criptográficas a las tarjetas de agentes. Un editor firma su propia tarjeta con un JWT; los consumidores verifican. Previene la imitación.

### Ciclo de vida de las tareas

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

Los clientes comienzan con `tasks/send`. Los agentes llamados pasan por los estados; los clientes se suscriben a las actualizaciones de los estados a través de SSE o encuestas.

### Mensajes y partes

Un mensaje contiene una o más partes:

- `text` contenido simple.
- `file` base64 mancha con mimeType.
- `data` la carga útil de JSON (entrada estructurada para el agente llamado).

Ejemplo:

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artículos

Las salidas son artefactos, no cadenas crudas. Un artefacto es una salida nombrada y tipada:

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Los artefactos pueden ser transmitidos como trozos.

### Dos obligaciones de transporte

1. **JSON-RPC over HTTP.** `/a2a`punto final, POST para las solicitudes, SSE opcional para la transmisión.
2. **gRPC.**Para entornos empresariales donde el gRPC es nativo.

Ambas vinculaciones tienen la misma forma lógica del mensaje.

### Preservación de la apertura

Un principio clave de diseño: el estado interno del agente llamado es opaco. El llamador ve el estado de tarea y los artefactos. La cadena de pensamiento del agente llamado, sus llamadas de herramienta, su delegación de subagentes  son todas invisibles. Esto es diferente de MCP, donde las llamadas de herramientas son transparentes.

Racionalización: A2A permite a los competidores colaborar sin revelar los datos internos. A2A puede ser "llamar a este agente de servicio al cliente" sin que el que llama aprenda cómo ese agente implementa el servicio.

### Línea de tiempo

- **2025-04-09.**Google anuncia A2A.
- **2025-06-23.**Donado a la Fundación Linux.
- **2025-08.**Absorbe el ACP de IBM.
- **2025-09.**Naves de extensión AP2 (pagos por agentes).
- **2026-04.**v1.0 lanzado con más de 150 organizaciones de apoyo.

### Relación con la MCP

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | JWT-signed Agent Cards (AP2) |
| Transport | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

Utilice MCP cuando desea invocar una herramienta específica. Utilice A2A cuando desea delegar una tarea completa a otro agente. Muchos sistemas de producción utilizan ambos: un agente utiliza MCP para su capa de herramientas y A2A para su capa de colaboración.

```figure
a2a-task-lifecycle
```

## Usalo

`code/main.py`La aplicación de un mínimo de A2A: un agente de investigación publica su tarjeta, un agente de redacción recibe un `tasks/send`con partes que incluyen un PDF y una instrucción de texto, las transiciones a través de trabajar → input_required → working → completado, y devuelve un artefacto de texto.

Qué ver:

- Forma de tarjeta de agente JSON.
- asignación de tareas y transiciones de estado.
- Mensajes con piezas de tipo mixto.
- En medio de la tarea, se requiere entrada de la rama.
- El artefacto regresa al final.

## Envío

Esta lección produce`outputs/skill-a2a-agent-spec.md`. Dado que un nuevo agente que debe ser llamado por otros agentes, la habilidad produce el JSON de la tarjeta de agente, esquema de habilidades y plan de punto final.

## Los ejercicios

1. - ¿ Qué ?`code/main.py`. Rastrear todo el ciclo de vida de la tarea, incluida la pausa requerida para la entrada en la que el agente llamado solicite una aclaración.

2. Añadir una tarjeta de agente firmada, firmar con HMAC sobre el JSON canónico de la tarjeta, escribir un verificador y confirmar que falla en una tarjeta mutada.

3. Implementar la transmisión de tareas: el agente de redacción emite tres trozos de artefactos incrementales sobre SSE y el solicitante los acumula.

4. Diseñar un agente A2A que envuelva un servidor MCP. Mapa cada herramienta MCP a una habilidad A2A. Observe los compromisos  ¿qué opacidad se pierde?

5. Lea el anuncio de A2A v1.0 e identifique la única característica que aún no está implementada por ningún marco a partir de abril de 2026. (Intenta: se refiere a la delegación de tareas multi-hop).

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | "Agent-to-Agent protocol" | Open protocol for opaque agent collaboration |
| Agent Card | "`.well-known/agent.json`" | Published metadata describing an agent's skills and endpoint |
| Skill | "A callable unit" | A named operation the agent supports (analog to MCP tool) |
| Task | "Unit of delegation" | A work item with a lifecycle and final artifact |
| Message | "Task input" | Carries Parts (text, file, data) |
| Part | "Typed chunk" | `text` / `file` / `data` element of a message |
| Artifact | "Task output" | Named, typed output returned on completion |
| AP2 | "Agent Payments Protocol" | Signed Agent Cards extension for trust and payments |
| Opacity | "Black-box collaboration" | Called agent's internals are hidden from caller |
| Input-required | "Task pause" | Lifecycle state when the agent needs more info |

## Leer más

- [a2a-protocol.org](https://a2a-protocol.org/latest/) especificación canónica A2A
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) Implementaciones de referencia y KDD
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) Transferencia de gobernanza de junio de 2025
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) hoja de ruta y impulso de los socios
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) Nota de liberación de la versión 1.0 y orientación retrocompatible
