# Evaluación: índices de referencia, Evals, LM Harness

> Ley de Goodhart: cuando una medida se convierte en un objetivo, deja de ser una buena medida. Cada juego de laboratorio fronterizo marca un punto de referencia. Las puntuaciones de MMLU aumentan mientras que los modelos aún no pueden contar confiablemente el número de R en "fresa". La única evaluación que importa es su evaluación - en su tarea, con sus datos.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lessons 01-05 (LLMs from Scratch)
**Time:** ~90 minutes

## Objetivos de aprendizaje

- Construir un arnés de evaluación personalizado que ejecute puntos de referencia de múltiples opciones y de límite abierto en comparación con un modelo de lenguaje
- Explicar por qué los valores de referencia estándar (MMLU, HumanEval) se saturan y no diferencian los modelos fronterizos
- Implementar evaluaciones específicas de tareas con métricas adecuadas: coincidencia exacta, F1, BLEU y puntuación LLM-as-judge
- Diseñar una suite de evaluación personalizada dirigida a su caso de uso específico en lugar de depender únicamente de tablas de clasificación públicas

## El problema

MMLU fue publicado en 2020 con 15.908 preguntas en 57 temas. En tres años, los modelos fronterizos lo saturaron. GPT-4 obtuvo 86,4%. Claude 3 Opus obtuvo 86,8%. Llama 3 405B obtuvo 88,6%.

Mientras tanto, esos mismos modelos fallan en tareas que un niño de 10 años maneja sin pensar. Claude 3.5 Sonnet, con un puntaje del 88.7% en MMLU, inicialmente no podía contar las letras en "fresa" -- una tarea que requiere cero conocimiento del mundo y cero razonamiento, sólo la iteración a nivel de personajes. HumanEval prueba la generación de código con 164 problemas. Los modelos obtienen un puntaje de más del 90% mientras producen código que se estropea en los casos de borde que cualquier desarrollador junior atraparía.

La brecha entre el rendimiento de los valores de referencia y la fiabilidad del mundo real es el problema central de la evaluación de los LLM. Los puntos de referencia le dicen cómo funciona un modelo en el punto de referencia. No te dicen casi nada sobre cómo ese modelo se desempeñará en tu tarea específica, con tus datos específicos, bajo tus modos de falla específicos. Si estás construyendo un bot de atención al cliente, MMLU es irrelevante. Si estás construyendo un asistente de código, HumanEval sólo cubre la generación a nivel de función, no dice nada sobre depurar, refactorar o explicar el código en archivos.

Necesitas evaluaciones personalizadas. No porque los puntos de referencia sean inútiles, son útiles para la selección aproximada de modelos, sino porque la evaluación final debe coincidir exactamente con las condiciones de implementación.

## El concepto

### El paisaje de Eval

Hay tres categorías de evaluación, cada una con un coste y una calidad de señal diferentes.

**Benchmarks**Los modelos de evaluación de la calidad de la información de los usuarios son un conjunto de pruebas estandarizadas. MMLU, HumanEval, SWE-bench, MATH, ARC, HellaSwag. Se ejecuta un modelo contra el índice de referencia y se obtiene una puntuación. La ventaja: todos utilizan la misma prueba, por lo que se pueden comparar los modelos. La desventaja: los modelos y los datos de capacitación cada vez más contaminan estos índices de referencia. Los laboratorios entrenan en datos que incluyen preguntas de referencia. Las puntuaciones aumentan. La capacidad puede no.

**Custom evals**Los datos de base de datos de SQL se evalúan en su esquema de base de datos. Estos son caros de crear pero son la única evaluación que predice el rendimiento de producción.

**Human evals**El chatbot Arena ha recogido más de 2 millones de votos de preferencia humana en más de 100 modelos.$0.10-$Las medidas de seguridad deben ser adoptadas en el marco de la aplicación de la Directiva.

```mermaid
graph TD
    subgraph Eval["Evaluation Landscape"]
        direction LR
        B["Benchmarks\n(MMLU, HumanEval)\nCheap, standardized\nGameable, stale"]
        C["Custom Evals\nYour task, your data\nHighest signal\nExpensive to build"]
        H["Human Evals\n(Chatbot Arena)\nGold standard\nSlow, costly"]
    end

    B -->|"rough model selection"| C
    C -->|"ambiguous cases"| H

    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style H fill:#1a1a2e,stroke:#e94560,color:#fff
```

### Por qué se rompen los índices de referencia

Tres mecanismos hacen que los puntajes de referencia dejen de reflejar la capacidad real.

**Data contamination.**Los cuerpos de entrenamiento raspan Internet. Las preguntas de referencia se transmiten en línea. Los modelos ven las respuestas durante el entrenamiento. Esto no es engaño en el sentido tradicional - los laboratorios no incluyen intencionalmente datos de referencia. Pero el raspado a escala web hace casi imposible excluirlos.

**Teaching to the test.**Los laboratorios optimizan las mezclas de entrenamiento para el rendimiento de referencia. Si el 5% de la mezcla de entrenamiento es una opción múltiple al estilo MMLU, el modelo aprende el formato y la distribución de la respuesta. MMLU es una opción múltiple de cuatro vías. Los modelos aprenden que la distribución de la respuesta es aproximadamente uniforme en A / B / C / D, lo que ayuda incluso cuando el modelo no conoce la respuesta.

**Saturation.**Cuando cada modelo fronterizo obtiene una puntuación del 85-90% en un índice de referencia, el índice de referencia deja de discriminar. El restante 10-15% de las preguntas pueden ser ambigüas, etiquetadas erróneamente o requerir conocimiento de dominio oscuro.

### Perplejidad: un examen médico rápido

La perplejidad mide lo sorprendente que es un modelo por una secuencia de tokens.

```
PPL = exp(-1/N * sum(log P(token_i | context)))
```

Una perplejidad de 10 significa que el modelo es, en promedio, tan incierto como elegir uniformemente entre 10 opciones en cada posición de token. Bajo es mejor. GPT-2 obtiene una perplejidad de ~30 en WikiText-103. GPT-3 obtiene ~20. Llama 3 8B obtiene ~7.

La perplejidad es útil para comparar modelos en el mismo conjunto de pruebas, pero tiene puntos ciegos. Un modelo puede tener baja perplejidad al ser bueno en predecir patrones comunes mientras que es terrible en patrones raros pero importantes. Tampoco dice nada sobre la instrucción, el razonamiento o la exactitud de los hechos.

### Licenciatura en Derecho como juez

Utilice un modelo fuerte para evaluar la producción de un modelo más débil. La idea es simple: pídale a GPT-4o o Claude Sonnet que califique una respuesta en una escala de 1-5 para la corrección, la utilidad y la seguridad. Esto cuesta alrededor de $0.01 por juicio con GPT-4o-mini y se correlaciona sorprendentemente bien con los juicios humanos - alrededor del 80% de acuerdo en la mayoría de las tareas.

El prompt de puntuación es más importante que el modelo. Un prompt vago ("Rate this response") produce puntuaciones ruidosas. Un prompt estructurado con una rúbrica ("Score 5 si la respuesta es factualmente correcta y cita una fuente, 4 si es correcta pero sin fuente, 3 si es parcialmente correcta...") produce puntuaciones consistentes y reproducibles.

Los modos de falla: los modelos de juez muestran sesgo de posición (prefieren la primera respuesta en comparaciones pares), sesgo de verbosidad (prefieren respuestas más largas) y auto-preferencia (GPT-4 tasa de GPT-4 de salida más alta que las salidas equivalentes Claude).

### Las calificaciones de ELO de las comparaciones de parejas

El enfoque de Chatbot Arena. Muestre dos respuestas a la misma solicitud de diferentes modelos. Un humano (o juez de LLM) elige la mejor. De miles de estas comparaciones, computa una calificación ELO para cada modelo - el mismo sistema utilizado en ajedrez.

Las ventajas de ELO: el ranking relativo es más confiable que el puntuación absoluta, maneja las correcciones con gracia y converge con menos comparaciones que marcar cada salida de forma independiente. A principios de 2026, los rankings de Chatbot Arena muestran GPT-4o, Claude 3.5 Sonnet y Gemini 1.5 Pro dentro de 20 puntos ELO entre sí en la cima.

```mermaid
graph LR
    subgraph ELO["ELO Rating Pipeline"]
        direction TB
        P["Prompt"] --> MA["Model A Output"]
        P --> MB["Model B Output"]
        MA --> J["Judge\n(Human or LLM)"]
        MB --> J
        J --> W["A Wins / B Wins / Tie"]
        W --> E["ELO Update\nK=32"]
    end

    style P fill:#1a1a2e,stroke:#0f3460,color:#fff
    style J fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#51cf66,color:#fff
```

### Cuadro de Evaluación

**lm-evaluation-harness**(EleutherAI): el marco de evaluación estándar de código abierto. Soporta más de 200 puntos de referencia. ejecuta cualquier modelo de Hugging Face contra MMLU, HellaSwag, ARC, etc. con un solo comando.

**RAGAS**El marco de evaluación específico para las tuberías RAG mide la fidelidad (¿correcta la respuesta al contexto recuperado?), la relevancia (¿el contexto recuperado es relevante para la pregunta?), y la corrección de la respuesta.

**promptfoo**: evaluación basada en configuración para la ingeniería de prompto. Definir casos de prueba en YAML, ejecutar contra múltiples modelos, obtener un informe de aprobación/fallo. Útil para las instrucciones de prueba de regresión - asegúrese de que un cambio rápido no rompe los casos de prueba existentes.

### Construir Evals personalizados

La única evaluación que importa para la producción.

1. **Define the task.**¿Qué debe hacer exactamente el modelo? Sea preciso. "Responda a las preguntas" es demasiado vaga. "Dado un correo electrónico de queja del cliente, extraer el nombre del producto, la categoría del problema y el sentimiento" es una tarea que puede evaluar.

2. **Create test cases.**Un mínimo de 50 para un eval de prototipo, 200+ para la producción. Cada caso de prueba es un par (entrada, expect_output). Incluye casos de borde: entradas vacías, entradas adversarias, entradas ambigüas, entradas en otros idiomas.

3. **Define scoring.**Aplicación exacta para resultados estructurados. BLEU/ROUGE para similitud de texto. LLM-as-judge para calidad abierta. F1 para tareas de extracción. Combine múltiples métricas con pesos.

4. **Automate.**Cada eval se ejecuta con un comando. No hay pasos manuales. Almacenar los resultados en un formato que permite la comparación a lo largo del tiempo.

5. **Track over time.**Una puntuación de evaluación no tiene sentido en aislamiento. Necesitas la línea de tendencia. ¿La puntuación mejoró después del último cambio de aviso? ¿Regresó después de cambiar de modelo? Versión de su evaluación junto con sus instrucciones.

| Eval Type | Cost per judgment | Agreement with humans | Best for |
|-----------|------------------|----------------------|----------|
| Exact match | ~$0 | 100% (when applicable) | Structured output, classification |
| BLEU/ROUGE | ~$0 | ~60% | Translation, summarization |
| LLM-as-judge | ~$0.01 | ~80% | Open-ended generation |
| Human eval | $0.10-$2.00 | N/A (is the ground truth) | Ambiguous, high-stakes tasks |

```figure
perplexity-loss
```

## Construye el mismo

### Paso 1: Un marco mínimo de igualdad

Definir las abstracciones centrales. Un caso eval tiene una entrada, una salida esperada y un dictado de metadatos opcionales. Un puntero toma una predicción y una referencia y devuelve una puntuación entre 0 y 1.

```python
import json
from collections import Counter

class EvalCase:
    def __init__(self, input_text, expected, metadata=None):
        self.input_text = input_text
        self.expected = expected
        self.metadata = metadata or {}

class EvalSuite:
    def __init__(self, name, cases, scorers):
        self.name = name
        self.cases = cases
        self.scorers = scorers

    def run(self, model_fn):
        results = []
        for case in self.cases:
            prediction = model_fn(case.input_text)
            scores = {}
            for scorer_name, scorer_fn in self.scorers.items():
                scores[scorer_name] = scorer_fn(prediction, case.expected)
            results.append({
                "input": case.input_text,
                "expected": case.expected,
                "prediction": prediction,
                "scores": scores,
            })
        return results
```

### Paso 2: Puntuación de las funciones

Construye una coincidencia exacta, un token F1, y un puntero simulado de LLM como juez.

```python
def exact_match(prediction, expected):
    return 1.0 if prediction.strip().lower() == expected.strip().lower() else 0.0

def token_f1(prediction, expected):
    pred_tokens = set(prediction.lower().split())
    exp_tokens = set(expected.lower().split())
    if not pred_tokens or not exp_tokens:
        return 0.0
    common = pred_tokens & exp_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def llm_judge_simulated(prediction, expected):
    pred_words = set(prediction.lower().split())
    exp_words = set(expected.lower().split())
    if not exp_words:
        return 0.0
    overlap = len(pred_words & exp_words) / len(exp_words)
    length_penalty = min(1.0, len(prediction) / max(len(expected), 1))
    return round(overlap * 0.7 + length_penalty * 0.3, 3)
```

### Paso 3: Sistema de clasificación de ELO

Implemente comparaciones en pares con las actualizaciones de ELO. Este es exactamente el sistema que utiliza Chatbot Arena para clasificar los modelos.

```python
class ELOTracker:
    def __init__(self, k=32, initial_rating=1500):
        self.ratings = {}
        self.k = k
        self.initial_rating = initial_rating
        self.history = []

    def _ensure_player(self, name):
        if name not in self.ratings:
            self.ratings[name] = self.initial_rating

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def record_match(self, player_a, player_b, outcome):
        self._ensure_player(player_a)
        self._ensure_player(player_b)

        ea = self.expected_score(self.ratings[player_a], self.ratings[player_b])
        eb = 1 - ea

        if outcome == "a":
            sa, sb = 1.0, 0.0
        elif outcome == "b":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        self.ratings[player_a] += self.k * (sa - ea)
        self.ratings[player_b] += self.k * (sb - eb)

        self.history.append({
            "a": player_a, "b": player_b,
            "outcome": outcome,
            "rating_a": round(self.ratings[player_a], 1),
            "rating_b": round(self.ratings[player_b], 1),
        })

    def leaderboard(self):
        return sorted(self.ratings.items(), key=lambda x: -x[1])
```

### Paso 4: Calculo de la complejidad

Compute la perplejidad usando probabilidades de token. en la práctica obtendrías esto de las logits del modelo. aquí simulamos con una distribución de probabilidades.

```python
import numpy as np

def perplexity(log_probs):
    if not log_probs:
        return float("inf")
    avg_neg_log_prob = -np.mean(log_probs)
    return float(np.exp(avg_neg_log_prob))

def token_log_probs_simulated(text, model_quality=0.8):
    np.random.seed(hash(text) % 2**31)
    tokens = text.split()
    log_probs = []
    for i, token in enumerate(tokens):
        base_prob = model_quality
        if len(token) > 8:
            base_prob *= 0.6
        if i == 0:
            base_prob *= 0.7
        prob = np.clip(base_prob + np.random.normal(0, 0.1), 0.01, 0.99)
        log_probs.append(float(np.log(prob)))
    return log_probs
```

### Paso 5: Resultados agregados

Computa estadísticas de resumen en una carrera de evaluaciones: media, media, tasa de aprobación en un umbral y desgloses por métrica.

```python
def summarize_results(results, threshold=0.8):
    all_scores = {}
    for r in results:
        for metric, score in r["scores"].items():
            all_scores.setdefault(metric, []).append(score)

    summary = {}
    for metric, scores in all_scores.items():
        arr = np.array(scores)
        summary[metric] = {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "pass_rate": round(float(np.mean(arr >= threshold)), 3),
            "n": len(scores),
        }
    return summary

def print_summary(summary, suite_name="Eval"):
    print(f"\n{'=' * 60}")
    print(f"  {suite_name} Summary")
    print(f"{'=' * 60}")
    for metric, stats in summary.items():
        print(f"\n  {metric}:")
        print(f"    Mean:      {stats['mean']:.3f}")
        print(f"    Median:    {stats['median']:.3f}")
        print(f"    Std:       {stats['std']:.3f}")
        print(f"    Range:     [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"    Pass rate: {stats['pass_rate']:.1%} (threshold >= 0.8)")
        print(f"    N:         {stats['n']}")
```

### Paso 6: Cumple el oleoducto completo

Define una tarea, crea casos de prueba, simula dos modelos, ejecuta evaluaciones, computa ELO a partir de comparaciones pares e imprima el tablero de clasificación.

```python
def demo_model_good(prompt):
    responses = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
        "Who wrote Hamlet?": "William Shakespeare",
        "What language is PyTorch written in?": "Python and C++",
        "What is the boiling point of water?": "100 degrees Celsius",
    }
    return responses.get(prompt, "I don't know")

def demo_model_bad(prompt):
    responses = {
        "What is the capital of France?": "Paris is the capital city of France",
        "What is 2 + 2?": "The answer is four",
        "Who wrote Hamlet?": "Shakespeare",
        "What language is PyTorch written in?": "Python",
        "What is the boiling point of water?": "212 Fahrenheit",
    }
    return responses.get(prompt, "Unknown")

cases = [
    EvalCase("What is the capital of France?", "Paris"),
    EvalCase("What is 2 + 2?", "4"),
    EvalCase("Who wrote Hamlet?", "William Shakespeare"),
    EvalCase("What language is PyTorch written in?", "Python and C++"),
    EvalCase("What is the boiling point of water?", "100 degrees Celsius"),
]

suite = EvalSuite(
    name="General Knowledge",
    cases=cases,
    scorers={
        "exact_match": exact_match,
        "token_f1": token_f1,
        "llm_judge": llm_judge_simulated,
    },
)

results_good = suite.run(demo_model_good)
results_bad = suite.run(demo_model_bad)

print_summary(summarize_results(results_good), "Model A (concise)")
print_summary(summarize_results(results_bad), "Model B (verbose)")
```

El modelo "bueno" da respuestas exactas. El modelo "malo" da paráfrases verbales. La coincidencia exacta castiga severamente al modelo verbales. Token F1 y LLM como juez son más indulgentes. Esto ilustra por qué la elección métrica importa: el mismo modelo se ve grande o terrible dependiendo de cómo lo califiques.

### Paso 7: Torneo ELO

Realice comparaciones en pares entre modelos en múltiples rondas.

```python
elo = ELOTracker(k=32)

for case in cases:
    pred_a = demo_model_good(case.input_text)
    pred_b = demo_model_bad(case.input_text)

    score_a = token_f1(pred_a, case.expected)
    score_b = token_f1(pred_b, case.expected)

    if score_a > score_b:
        outcome = "a"
    elif score_b > score_a:
        outcome = "b"
    else:
        outcome = "tie"

    elo.record_match("model_a_concise", "model_b_verbose", outcome)

print("\nELO Leaderboard:")
for name, rating in elo.leaderboard():
    print(f"  {name}: {rating:.0f}")
```

### Paso 8: Comparancia de perplejidad

Comparar la perplejidad entre "modelos" de diferentes niveles de calidad.

```python
test_text = "The quick brown fox jumps over the lazy dog in the garden"

for quality, label in [(0.9, "Strong model"), (0.7, "Medium model"), (0.4, "Weak model")]:
    log_probs = token_log_probs_simulated(test_text, model_quality=quality)
    ppl = perplexity(log_probs)
    print(f"  {label} (quality={quality}): perplexity = {ppl:.2f}")
```

## Usalo

### El uso de la tecnología de evaluación (EleutherAI)

La herramienta estándar para ejecutar valores de referencia en cualquier modelo.

```python
# pip install lm-eval
# Command line:
# lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B --tasks mmlu --batch_size 8

# Python API:
# import lm_eval
# results = lm_eval.simple_evaluate(
#     model="hf",
#     model_args="pretrained=meta-llama/Llama-3.1-8B",
#     tasks=["mmlu", "hellaswag", "arc_easy"],
#     batch_size=8,
# )
# print(results["results"])
```

### de inmediato

Evaluación basada en configuración para ingeniería rápida. Defina pruebas en YAML y ejecuta contra múltiples proveedores.

```yaml
# promptfoo.yaml
providers:
  - openai:gpt-4o-mini
  - anthropic:claude-3-haiku

prompts:
  - "Answer in one word: {{question}}"

tests:
  - vars:
      question: "What is the capital of France?"
    assert:
      - type: contains
        value: "Paris"
  - vars:
      question: "What is 2 + 2?"
    assert:
      - type: equals
        value: "4"
```

### RAGAS para la evaluación de RAG

```python
# pip install ragas
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
#
# result = evaluate(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision],
# )
# print(result)
```

RAGAS mide lo que faltan las evaluaciones genéricas: si la respuesta del modelo se basa en el contexto recuperado, no solo si la respuesta es "correcta" en el abstracto.

## Envío

Esta lección produce`outputs/prompt-eval-designer.md`-- una solicitud reutilizable que diseña conjuntos de evaluaciones personalizadas para cualquier tarea. Dale una descripción de tarea y genera casos de prueba, funciones de puntuación y una recomendación de umbral de paso / fracaso.

También produce `outputs/skill-llm-evaluation.md`-- un marco de decisión para elegir la estrategia de evaluación adecuada en función de su tipo de tarea, presupuesto y requisitos de latencia.

## Los ejercicios

1. Añadir un puntero de "constancia" que ejecuta la misma entrada a través del modelo 5 veces y mide la frecuencia con la que coinciden las salidas.

2. Extenda el rastreador ELO para soportar múltiples funciones de juez (combinación exacta, F1, LLM-as-judge) y ponga en peso. Compara cómo cambia el tablero de clasificación cuando ponga en peso la coincidencia exacta en comparación con la F1.

3. Construir una suite de eval para una tarea específica: clasificación de correo electrónico en 5 categorías. Crear 100 casos de prueba con ejemplos diversos, incluyendo casos de borde ( correos electrónicos que podrían pertenecer a múltiples categorías, correos electrónicos vacíos, correos electrónicos en otros idiomas). Medir cómo funcionan diferentes "modelos" (basado en reglas, coincidencia de palabras clave, LLM simulado).

4. Implementar la detección de contaminación: dada una serie de preguntas de evaluación y un corpus de formación, comprobar qué porcentaje de preguntas de evaluación (o parafrases cercanas) aparecen en los datos de formación.

5. Construir una herramienta de "modelo diferente". Dado los resultados de evaluación de dos versiones de modelo, resaltar qué casos de prueba específicos mejoraron, que regresaron y que se mantuvieron iguales. Este es el equivalente de evaluación de un código diferente - esencial para entender si un cambio ayudó o perjudicó.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| MMLU | "The benchmark" | Massive Multitask Language Understanding -- 15,908 multiple choice questions across 57 subjects, saturated above 88% by 2025 |
| HumanEval | "Code eval" | 164 Python function-completion problems from OpenAI, tests only isolated function generation |
| SWE-bench | "Real coding eval" | 2,294 GitHub issues from 12 Python repos, measures end-to-end bug fixing including test generation |
| Perplexity | "How confused the model is" | exp(-avg(log P(token_i given context))) -- lower means the model assigns higher probability to the actual tokens |
| ELO rating | "Chess ranking for models" | A relative skill rating computed from pairwise win/loss records, used by Chatbot Arena to rank 100+ models |
| LLM-as-judge | "Using AI to grade AI" | A strong model scores a weaker model's outputs against a rubric, ~80% agreement with human judges at ~$0.01/judgment |
| Data contamination | "The model saw the test" | Training data includes benchmark questions, inflating scores without improving real capability |
| Eval suite | "A bunch of tests" | A versioned collection of (input, expected_output, scorer) triples that measure a specific capability |
| Pass rate | "What percentage it gets right" | Fraction of eval cases scoring above a threshold -- more actionable than mean score because it measures reliability |
| Chatbot Arena | "Model ranking website" | LMSYS platform with 2M+ human preference votes, producing the most trusted LLM leaderboard via ELO ratings |

## Leer más

- [Hendrycks et al., 2021 -- "Measuring Massive Multitask Language Understanding"](https://arxiv.org/abs/2009.03300)-- el artículo de la MMLU, sigue siendo el referente de LLM más citado a pesar de su saturación
- [Chen et al., 2021 -- "Evaluating Large Language Models Trained on Code"](https://arxiv.org/abs/2107.03374)-- el documento de HumanEval de OpenAI, metodología de evaluación de generación de código establecida
- [Zheng et al., 2023 -- "Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685)-- análisis sistemático del uso de los LLM para evaluar los LLM, incluidos los hallazgos de sesgo de posición y sesgo de verbosidad
- [LMSYS Chatbot Arena](https://chat.lmsys.org/)-- plataforma de comparación de modelos de crowdsourcing con 2M+ votos, el ranking de LLM más confiable del mundo real
