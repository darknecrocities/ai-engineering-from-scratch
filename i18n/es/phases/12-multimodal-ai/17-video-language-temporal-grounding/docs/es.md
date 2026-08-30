# Modelos de lenguaje en video: Tokens temporales y Grounding

> El video no es una pila de fotos. Un clip de 5 segundos tiene orden causal, verbos de acción y tiempo de evento que un modelo de imagen no puede representar. Video-LLaMA (Zhang et al., junio 2023) envió el primer video-LLM abierto con tierra audiovisual. VideoChat y Video-LLaVA escalaron el patrón. Para 2025, el TMRoPE de Qwen2.5-VL cerró la brecha con los modelos patentados fronterizos. Cada sistema resolvió tokens temporales de manera diferente  Q-former por clip, concat-pool por frame, TMRoPE por token. Esta lección lee los patrones, construye un muestreo de marco uniforme frente a dinámico y evalúa las tareas de tierra temporal.

**Type:** Build
**Languages:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)
**Time:** ~180 minutes

## Objetivos de aprendizaje

- Explicar por qué la codificación posicional temporal cambia el rendimiento de VLM de vídeo independientemente del codificador de visión.
- Comparar muestreo de fotogramas uniforme, dinámico-FPS, y impulsado por eventos en tokens-por-segundo vs precisión de tierra.
- Describa los diseños Q-ex-per-clip (Video-LLaMA) vs. pooled-per-frame (Video-LLaVA) vs. M-RoPE-per-token (Qwen2.5-VL).
- Nombre de los cuatro puntos de referencia de vídeo: VideoMME, TempCompass, EgoSchema, Video-MMMU.

## El problema

Un video de 1 minuto a 30 FPS es de 1800 cuadros. A 196 tokens visuales por cuadro (ViT-B a 224), es decir, 352k tokens  más grandes que cualquier contexto LLM de la era de 2024.

Existen tres estrategias de reducción:

1. Envases de submuestras (1-8 FPS dependiendo del contenido).
2. Agrupar las fichas de cada marco de forma agresiva (3x3 o 4x4 pool bilinear).
3. Compresar a través de un Q-former que toma un clip de 16 cuadros y saque 64 tokens.

Cada cambio es diferente. la submuestreo pierde detalles temporales. el pooling pierde detalles espaciales. el Q-former pierde tanto un poco pero ahorra tokens.

La codificación de la posición temporal es el otro eje: ¿cómo sabe el modelo que el marco 5 vino antes que el marco 6? Las opciones incluyen simple RoPE temporal 1D (Video-LLaMA), incorporaciones temporales aprendidas (Video-LLaVA) y TMRoPE (Qwen2.5-VL, 3D completo).

## El concepto

### Video-LLaMA: Q-former por clip + rama de audio

Video-LLaMA (2023) fue el primer video-LLM abierto.

- Clip de 16 cuadros a 2 FPS (de ahí 8 segundos).
- Las funciones de ViT por marco -> Video Q-former que atende cruzando a través de los 16 cuadros -> 32 consultas aprendidas -> LLM.
- Ramo de audio paralelo: forma de onda -> Encoder de audio ImageBind -> Audio Q-former -> 32 consultas -> LLM.

Fuerza: razonamiento conjunto audiovisual. debilidad: longitud fija del clip, sin tierra de tiempo arbitrario.

### VideoChat y Video-LLaVA

VideoChat mantuvo la idea de Video-LLaMA, pero dejó caer el audio y simplificó. Video-LLaVA (Lin et al., 2023) entrenó un único codificador visual en ambas imágenes y marcos de video ("alineamiento antes de la proyección"), dando una representación unificada.

Ninguno de ellos maneja videos largos.

### Qwen2.5-VL y TMRoPE

Qwen2.5-VL introdujo TMRoPE  Embedding de posición rotaria de modalidad temporal. Cada token de parche lleva una posición (t, h, w) donde t es el sello de tiempo real (no el índice de marco).

Diferencias clave de la simple incorporación temporal:

- El modelo ve "a 4.2 segundos" no "a marco 15".
- Cada token visual gira de forma independiente por su sello de tiempo.
- Si muestra a 2 FPS aquí y 4 FPS allí, TMRoPE maneja el espaciamiento desigual de forma nativa.

TMRoPE permite las consultas "¿a qué segundo salta el gato?" El modelo puede emitir "a 4.2 segundos". Video-LLaMA sólo podría decir "a principios en el clip".

### Estrategias de muestreo de marco

Uniforme: muestra N marcos uniformemente a lo largo de la duración.

FPS dinámico: muestra adaptativamente basada en la intensidad del movimiento. flujo óptico o diferenciación de marco elige segmentos de alta movimiento para muestreo más denso.

Event-driven: ejecuta un detector ligero, muestra más donde ocurre la acción.

Cuadro de teclado + contexto: muestra en los límites de la toma + algunos cuadros adyacentes.

### Reunión por marco

A 1 FPS y 576 tokens por fotograma, un clip de 5 minutos es de 172.800 tokens.

3x3 bilinear pool se reduce a 64 tokens por marco -> 19.200 tokens durante 5 minutos.

Reúne de manera más agresiva (6x6 -> 16 tokens por fotograma) para flujos de trabajo de agentes donde el detalle espacial importa menos.

### Los cuatro puntos de referencia de vídeo

- VideoMME: comprensión completa de vídeo, corto + medio + largo.
- TempCompass: razonamiento temporal de gran tamaño, preguntas "antes" / "después".
- EgoSchema: video en primera persona de largo horizonte.
- Video-MMMU: preguntas de vídeo multimodales multidisciplinares.

Una evaluación completa de video-VLM alcanza a los cuatro. Enfatizan diferentes ejes  TempCompass se trata de pedir, EgoSchema es de aproximadamente 3+ minutos de razonamiento, VideoMME abarca duradas.

### Formatos de salida de tierra

Formatos de salida para la tierra temporal:

- Texto libre: "El gato salta alrededor de la marca de 4 segundos".
- JSON estructurado: `{"event": "jump", "start": 4.1, "end": 4.3}`El Qwen2.5VL está equipado con esto.
- Basado en tokens: especial `<time>4.1</time>`Los tokens se entrelazaron con la respuesta.

El formato de salida JSON de Qwen2.5VL analiza directamente.

### 2026 mejores prácticas

Para VLMs de vídeo en 2026:

- Encodrador: SigLIP 2 con M-RoPE o TMRoPE (Qwen2.5-VL).
- Muestreo de fotogramas: FPS dinámico (1-4 dependiendo del movimiento) con tapa máxima de fotograma.
- El conjunto por marco: 3x3 bilinear.
- Resultado: JSON estructurado con campos de tiempo + evento.
- Indicadores de referencia: VideoMME + TempCompass para el general; EgoSchema para el largo horizonte.

```figure
video-temporal-patches
```

## Usalo

`code/main.py`incluye:

- Muestras de marcos de FPS uniformes y dinámicos.
- Un evaluador de la tierra temporal de juguete: dado un evento de "verdad fundamental" en el tiempo T y una salida de modelo, califique la precisión con tolerancia.
- Una comparación entre Video-LLaMA (16 cuadros, Q-former), Video-LLaVA (8 cuadros, MLP), Qwen2.5-VL (FPS dinámico + TMRoPE).

## Envío

Esta lección produce`outputs/skill-video-vlm-frame-planner.md`. Dado una tarea de vídeo (monitoreo, reconocimiento de acción, tierra temporal, resumen), elige el muestreo de fotogramas, factor de agrupación, formato de salida y nivel de precisión esperado.

## Los ejercicios

1. Para una demostración de cocción de 3 minutos, escoge uniforme vs FPS dinámico.

2. TMRoPE añade qué específicamente una simple tabla de incorporación temporal no puede hacer?

3. Escriba un esquema JSON para la conexión temporal que un VLM puede aprender a emitir. Incluya casos de error.

4. Leer la sección 3 de Video-LLaVA sobre "Alignment Before Projection". ¿Por qué es mejor que entrenar codificadores de imágenes y videos separados?

5. Dado el ranking de VideoMME, ¿cuál es la brecha entre el modelo abierto superior y el modelo propietario superior a partir de 2026? ¿Cuánto de esa brecha se atribuye a la codificación temporal vs escala LLM básica?

## Términos clave

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Temporal grounding | "Time-localized answers" | VLM outputs a specific timestamp range for when an event happens |
| TMRoPE | "Time-Multimodal RoPE" | 3D rotary position with absolute timestamps, used by Qwen2.5-VL |
| Dynamic FPS | "Motion-aware sampling" | Sample more frames in high-motion segments, fewer in static ones |
| Frame pooling | "Spatial compress per frame" | Reduce patches per frame with bilinear interpolation before the LLM |
| Video Q-former | "Clip compressor" | Cross-attention bottleneck mapping N frames to K learned queries |
| VideoMME | "Video bench" | Comprehensive short/medium/long video benchmark, 2500+ samples |

## Leer más

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
