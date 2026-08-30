# Por qué los transformadores  Los problemas con las RNN

> RNNs procesan tokens uno a la vez. Transformers procesan todos los tokens a la vez. Esa sola apuesta arquitectónica cambió cada curva de escala en el aprendizaje profundo después de 2017.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 minutes

## El problema

Antes de 2017, cada modelo de secuencia de última generación en el planeta  lenguaje, traducción, habla  era una red neuronal recurrente. LSTMs y GRU ganaron puntos de referencia de traducción equivalentes a ImageNet durante media década. Fueron la única herramienta que alguien tenía.

El cálculo secuencial significaba que no se podía paralelalizar a lo largo del eje del tiempo:`t+1`Necesita el estado oculto de la señal .`t`Una secuencia de 1.024 tokens significaba 1.024 pasos en serie en una GPU que puede hacer 1.000.000 operaciones de puntos flotantes por ciclo.

Los gradientes desaparecientes significaban que la información 50 tokens atrás ya estaba comprimida a través de 50 no linealidades. Las unidades recurrentes de gated (LSTM, GRU) suavizaron la aplastamiento pero nunca lo eliminaron. Las dependencias de largo alcance  "el libro que leí el verano pasado en un avión a Kioto fue..."  rotineamente fracasó.

Los estados ocultos de ancho fijo significaban que el codificador comprimió toda la secuencia de fuente en un solo vector antes de que el decodificador viera algo.

El artículo de 2017 "Attención es todo lo que necesitas" propuso algo radical: dejar de recurrir por completo. Dejar que cada posición atenda a cada otra posición en paralelo. Entrenar en una gran matriz multiplicación en lugar de 1.024 secuenciales.

El resultado domina todas las modalidades para 2026. lenguaje (GPT-5, Claude 4, Llama 4), visión (ViT, DINOv2, SAM 3), audio (Whisper), biología (AlphaFold 3), robótica (RT-2).

## El concepto

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Recurrence as a bottleneck.**Un RNN computa `h_t = f(h_{t-1}, x_t)`Cada paso depende del anterior. No puedes calcular.`h_5`antes de`h_4`En las GPU modernas con más de 10.000 núcleos paralelos, esto desperdicia el 99% del silicio en una larga secuencia.

**Attention as a broadcast.**Cuentas de autoatención `output_i = sum_j(a_ij * v_j)`para cada par `(i, j)`La matriz de atención N×N completa un matmul en lote. Ningún paso depende de otro.

**The speedup is not a constant.**Es la diferencia entre `O(N)`profundidad en serie y `O(1)`En la práctica, los transformadores se entrenan 510x más rápido por época en hardware coincidente a N=512, y la brecha se amplía con la longitud de la secuencia hasta que se golpea el `O(N²)`pared de memoria de la atención (que Flash Attention posteriormente arregló  ver Lección 12).

**What transformers cost.**Escales de memoria de atención como `O(N²)`Para el contexto de 2K, bien. Para el contexto de 128K, necesitas ventanas correderas, extrapolación RoPE, mosaico de atención flash, o variantes de atención lineal.`O(N)`en tiempo y memoria; los transformadores intercambian tiempo por memoria y luego ganan el tiempo de vuelta a través del paralelismo.

**The inductive bias shift.**Los transformadores no asumen nada.  cada par es un candidato a la atención. Es por eso que los transformadores necesitan más datos para entrenar bien, pero escalar más una vez que lo tengan. Chinchilla (2022) formalizó esto: dado suficientes tokens, un transformador siempre supera un RNN de igual número de parámetros.

```figure
rnn-vs-parallel
```

## Construye el mismo

No hay red neuronal aquí Simulado el cuello de botella del núcleo numéricamente para que sienta la brecha en su portátil.

### Paso 1: medir la profundidad en serie

¿ Qué ?`code/main.py`Construimos dos funciones: una codifica una secuencia como una cadena de adiciones (serial, como un RNN). otra lo codifica como una reducción paralela (broadcast, como la atención).

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

Los datos de la secuencia de la CPU son de un solo tipo, pero la secuencia de la CPU es de un solo tipo.`sum()`se implementa en C y se repite sin gastos generales de intérprete por paso.

### Paso 2: contar las operaciones teóricas

Los algoritmos añaden N. La diferencia es *profundidad de dependencia*: cuántas operaciones deben ocurrir secuencialmente antes de que pueda comenzar el siguiente. RNN profundidad = N. Profundidad de atención = log(N) con una reducción de árbol, o 1 con un escaneo paralelo.

### Paso 3: Escalación empírica en secuencias largas

Impresamos una tabla de tiempo que hace que la brecha O(N) sea visible. En un portátil Mac 2026, las secuencias de menos de 1.000 elementos son demasiado rápidas para medir. Las secuencias de 100.000 muestran un escaneo lineal limpio. Escala eso a un transformador de 16.384 tokens con un equivalente LSTM de 12 capas y ves por qué el entrenamiento de relojes murales fue un bloqueador en 2016.

## Usalo

¿Cuándo elegir un RNN en 2026?

| Situation | Pick |
|-----------|------|
| Streaming inference, one token at a time, constant memory | RNN or state-space model (Mamba, RWKV) |
| Very long sequences (>1M tokens) where attention memory explodes | Linear attention, Mamba 2, Hyena |
| Edge device with no matmul accelerator | Depthwise-separable RNN still wins on FLOPs/watt |
| Anything else (training, batched inference, context up to 128K) | Transformer |

Los modelos de espacio estatal (SSM) como Mamba son esencialmente RNNs con parametrización estructurada que les da lo mejor de ambos: `O(N)`La mayoría de los laboratorios fronterizos entrenan modelos híbridos de transformadores SSM+ (por ejemplo, Jamba, Samba)  la recurrencia no está muerta, es un componente.

## Envío

¿ Qué ?`outputs/skill-architecture-picker.md`La habilidad elige una arquitectura para un nuevo problema de secuencia dada la longitud, el rendimiento y las limitaciones del presupuesto de entrenamiento.

## Los ejercicios

1. **Easy.**¿ Qué ?`rnn_style`de la`code/main.py`y reemplazar el estado oculto escalar con un vector de longitud-64 de estados ocultos.
2. **Medium.**Implemente una suma de prefijos paralelas (escaneo de Hillis-Steele) en Python puro. Verifique que produce la misma salida numérica que un escaneo en serie en longitud 1024. Cuente la profundidad.
3. **Hard.**Portar la reducción de estilo de atención a PyTorch en la GPU. tiempo tanto como varía la longitud de la secuencia de 64 a 65.536.

## Términos clave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Recurrence | "RNNs are sequential" | Computation where step `t` depends on step `t-1`, forcing serial execution along the time axis. |
| Serial depth | "How deep the graph is" | Longest chain of dependent ops; bounds wall-clock even on infinite hardware. |
| Attention | "Let tokens look at each other" | Weighted sum `sum_j a_ij v_j` where `a_ij` comes from a similarity score between positions i and j. |
| Context window | "How much the model sees" | Number of positions an attention layer can take as input; quadratic memory cost scales here. |
| Inductive bias | "Assumptions baked into the architecture" | Prior about what the data looks like; CNNs assume translation invariance, RNNs assume recency. |
| State-space model | "RNN with algebra behind it" | Recurrence parameterized for parallel training via structured state-space matrices. |
| Quadratic bottleneck | "Why context costs so much" | Attention memory = `O(N²)` in sequence length; Flash Attention hides the constants, not the scaling. |

## Leer más

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) el artículo que mató la recurrencia en la PNL convencional.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) donde nació la atención, enlazado en un RNN.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) el papel original de LSTM, por escrito.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) respuesta recurrente moderna a los transformadores.
