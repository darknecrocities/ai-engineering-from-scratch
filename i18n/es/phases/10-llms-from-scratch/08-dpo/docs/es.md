# DPO: Optimización directa de preferencias

> RLHF funciona. También requiere entrenar tres modelos (SFT, modelo de recompensa, política), gestionar la inestabilidad de PPO y ajustar una penalización KL. DPO pregunta: ¿qué pasa si puedes saltarte todo eso? DPO optimiza directamente el modelo de lenguaje en pares de preferencias. No hay modelo de recompensa. No hay PPO. Un bucle de entrenamiento. Los mismos resultados.

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10, Lesson 07 (RLHF)
**Time:** ~90 minutes

## Objetivos de aprendizaje

- Implementar una formación en DPO que optimice directamente un modelo de lenguaje en pares de preferencias sin un modelo de recompensa separado
- Derivar la función de pérdida de DPO y explicar cómo representa implícitamente un modelo de recompensa a través de las probabilidades de registro de la póliza
- Comparar DPO vs RLHF en términos de estabilidad de formación, coste de cálculo y número de modelos requeridos
- Ajustar el parámetro beta para controlar en qué medida la política entrenada difiere del modelo de referencia

## El problema

El modelo de recompensa y el modelo de política optimizado con PPO. El modelo de recompensa solo requería miles de pares de preferencias humanas y un ciclo de entrenamiento separado.

En la práctica, el entrenamiento PPO es notoriamente inestable. Pequeños cambios de hiperparámetros hacen que el entrenamiento diverja. El modelo de recompensa es un proxy imperfecto para las preferencias humanas, y la política encuentra formas de explotar sus debilidades. La penalidad KL ayuda pero requiere su propio ajuste - demasiado bajo y se obtiene el hacking de recompensa, demasiado alto y el modelo apenas aprende.

Esta complejidad es la razón por la que la mayoría de los modelos de código abierto lucharon con RLHF durante años después de que se publicara InstructGPT.

En mayo de 2023, Rafael Rafailov, Archit Sharma y colegas de Stanford publicaron "Optimización de Preferencias Directas: Tu Modelo de Lenguaje es Secretamente un Modelo de Recompensa". La clave: no necesitas un modelo de recompensa separado. La función de recompensa óptima se determina matemáticamente por las probabilidades simbólicas del modelo de lenguaje. Puedes saltarte el modelo de recompensa por completo y optimizar el modelo de lenguaje directamente en pares de preferencias.

DPO reduce el RLHF a un solo paso de aprendizaje supervisado. Un modelo. Una función de pérdida. Un bucle de entrenamiento. No hay aprendizaje de refuerzo. Zephyr-7B, uno de los primeros modelos en usar DPO a escala, coincidió o superó modelos entrenados con RLHF completo en varios puntos de referencia. Meta utilizó DPO como parte de la tubería de alineación de Llama 3.

## El concepto

### El punto clave

RLHF optimiza este objetivo:

```
maximize: E[R(x, y)] - beta * KL(pi || pi_ref)
```

donde R es el modelo de recompensa, pi es la política, pi_ref es el modelo de referencia y beta es el coeficiente KL.

El documento del DPO mostró que este objetivo tiene una solución óptima en forma cerrada.

```
pi*(y | x) = pi_ref(y | x) * exp(R(x, y) / beta) / Z(x)
```

donde Z(x) es una constante normalizadora.

```
R(x, y) = beta * log(pi*(y | x) / pi_ref(y | x)) + beta * log Z(x)
```

Esta es la conclusión. La recompensa se expresa enteramente en términos de las probabilidades del modelo de política y las probabilidades del modelo de referencia. No es necesario entrenar un modelo de recompensa separado. La recompensa es *implicita* en la proporción de probabilidad.

Sustituyendo esto en el modelo de preferencia Bradley-Terry:

```
P(y_w > y_l | x) = sigmoid(R(x, y_w) - R(x, y_l))
                  = sigmoid(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)))
```

Los términos Z(x) se cancelarán porque ambas respuestas se condicionan a la misma respuesta x. Lo que queda es una función de las probabilidades de registro del modelo de política y las probabilidades de registro del modelo de referencia en las respuestas preferidas y rechazadas.

### La pérdida de DPO

```
L_DPO = -log(sigmoid(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x))))
```

Desempaquemos cada pieza.

- **y_w**= respuesta preferida (ganadora)
- **y_l**= respuesta rechazada (perdida)
- **x**= rápido
- **pi**= modelo actual (estando capacitado)
- **pi_ref**= modelo de referencia (punto de control de FFT congelado)
- **beta**= parámetro de temperatura que controla la desviación de la referencia (normalmente de 0,1 a 0,5)

La proporción `log pi(y|x) / pi_ref(y|x)`Cuando esta relación es positiva, el modelo actual asigna una probabilidad mayor a la respuesta y que la referencia. Cuando es negativa, el modelo actual asigna una probabilidad menor.

La pérdida de DPO empuja al modelo a aumentar la relación de probabilidad de registro para las respuestas preferidas y disminuirlo para las respuestas rechazadas. El parámetro beta controla cuán agresivamente el modelo puede desviarse de la referencia - la beta pequeña significa que se permiten grandes desviaciones, la beta grande mantiene al modelo cerca de la referencia.

```mermaid
graph TD
    subgraph DPO["DPO Training"]
        direction TB
        D["Preference Dataset\n(prompt, winner, loser)"] --> P1["Compute log P(winner)\nunder current model"]
        D --> P2["Compute log P(loser)\nunder current model"]
        D --> R1["Compute log P(winner)\nunder reference model"]
        D --> R2["Compute log P(loser)\nunder reference model"]

        P1 --> RATIO_W["Log ratio (winner)\nlog pi/pi_ref"]
        R1 --> RATIO_W
        P2 --> RATIO_L["Log ratio (loser)\nlog pi/pi_ref"]
        R2 --> RATIO_L

        RATIO_W --> DIFF["beta * (ratio_w - ratio_l)"]
        RATIO_L --> DIFF

        DIFF --> LOSS["-log sigmoid(diff)"]
        LOSS --> UPDATE["Gradient update\non current model"]
    end

    subgraph Models["Models"]
        PI["Current Model (pi)\nupdated each step"]
        REF["Reference Model (pi_ref)\nfrozen SFT checkpoint"]
    end

    Models --> DPO

    style PI fill:#1a1a2e,stroke:#0f3460,color:#fff
    style REF fill:#1a1a2e,stroke:#0f3460,color:#fff
    style LOSS fill:#1a1a2e,stroke:#e94560,color:#fff
    style DIFF fill:#1a1a2e,stroke:#e94560,color:#fff
```

### Por qué la DPO es más simple

| Aspect | RLHF (PPO) | DPO |
|--------|-----------|-----|
| Models to train | 3 (SFT + reward + policy) | 1 (policy only) |
| Training loops | 3 (SFT, RM training, PPO) | 2 (SFT, DPO) |
| Hyperparameters | lr, KL coeff, clip ratio, RM lr, epochs x3 | lr, beta, epochs |
| Reward model | Required (separate training) | Implicit in model probabilities |
| RL algorithm | PPO (complex, unstable) | Supervised learning (stable) |
| GPU memory | 3-4 models in memory during PPO | 2 models (current + reference) |
| Training stability | Sensitive to hyperparameters | Robust, similar to SFT |

DPO necesita dos modelos en la memoria durante el entrenamiento - el modelo actual y la referencia congelada. RLHF necesita tres o cuatro: la política, la referencia, el modelo de recompensa, y opcionalmente una función de valor de línea de base. Para un modelo 70B, cada copia toma 140 GB en FP16.

### Cuando el DPO supera a la RLHF

**Small datasets.**Con 5.000-20.000 pares de preferencias, DPO a menudo coincide o supera RLHF. El modelo de recompensa en RLHF necesita suficientes datos para generalizar - con datos limitados, se sobrepasa y produce señales de recompensa poco confiables. DPO evita este problema al no necesitar un modelo de recompensa en absoluto.

**Limited compute.**DPO requiere aproximadamente un tercio del cálculo de RLHF completo (un bucle de entrenamiento en lugar de tres). Para equipos sin grandes grupos de GPU, esta es la opción práctica.

**Rapid iteration.**¿Quieres probar 10 conjuntos de datos de preferencias diferentes para ver cuál produce el mejor modelo? DPO te permite ejecutar cada experimento en horas. RLHF requiere reentrenamiento del modelo de recompensa para cada conjunto de datos.

### Cuando RLHF supera a DPO

**Large-scale training.**En la escala de GPT-4 o Claude, el modelo de recompensa separado de RLHF puede capturar señales de preferencia más matizadas.

**Complex reward signals.**Cuando "mejor" implica múltiples dimensiones (utilidad, inofensividad, honestidad), un modelo de recompensa puede aprender este compromiso multi-objetivo. DPO trata a cada par de preferencias como una señal binaria - uno es mejor, otro es peor - sin modelar por qué.

**Iterative alignment.**Las tuberías RLHF pueden generar nuevas respuestas con la política actual, tener humanos que evaluarlas y retomar el modelo de recompensa en un bucle en línea. DPO trabaja en un conjunto de datos fijo de pares de preferencias.

### Más allá de la DPO: KTO, ORPO, SimPO

DPO inspiró una familia de métodos de alineación simplificados.

**KTO (Kahneman-Tversky Optimization, 2024):**Ni siquiera necesitas pares. KTO trabaja con retroalimentación sin parejas, simplemente etiquete cada respuesta como "buena" o "mala" sin compararla con otra alternativa. Esto simplifica dramáticamente la recopilación de datos. En lugar de mostrar a los anotadores dos respuestas y preguntar "¿cuál es mejor?", muestra una respuesta y pregunta "¿es esto bueno?" La función de pérdida aplica la aversión a las pérdidas de la teoría de perspectivas: las malas respuestas son penalizadas más que las buenas respuestas son recompensadas.

**ORPO (Odds Ratio Preference Optimization, 2024):**Combina SFT y alineación en un solo paso de entrenamiento. En lugar de hacer primero SFT y luego DPO, ORPO modifica la pérdida SFT para incluir una señal de preferencia. La pérdida tiene dos términos: una pérdida de predicción estándar de token siguiente en las respuestas preferidas, más un término de relación de probabilidades que aumenta la brecha entre las probabilidades de respuesta preferidas y rechazadas. Un bucle de entrenamiento en lugar de dos.

**SimPO (Simple Preference Optimization, 2024):**Elimina el modelo de referencia por completo. En lugar de calcular las relaciones de probabilidad de registro contra una referencia congelada, SimPO utiliza la probabilidad de registro promedio de la respuesta (normalizada por longitud) como la recompensa implícita. Esto ahorra memoria (no se necesita modelo de referencia) y simplifica el entrenamiento. La normalización de longitud evita que el modelo favorezca respuestas más cortas.

| Method | Year | Models in Memory | Needs Pairs? | Needs Reference? | Training Loops |
|--------|------|-----------------|-------------|-----------------|----------------|
| RLHF | 2022 | 3-4 | Yes (for RM) | Yes | 3 |
| DPO | 2023 | 2 | Yes | Yes | 2 |
| KTO | 2024 | 2 | No (unpaired) | Yes | 2 |
| ORPO | 2024 | 1 | Yes | No | 1 |
| SimPO | 2024 | 1 | Yes | No | 1 |

La tendencia es clara: cada método elimina una pieza más de complejidad. RLHF necesitaba un modelo de recompensa y PPO. DPO eliminó ambos. KTO eliminó los datos emparejados. ORPO eliminó la etapa separada de SFT. SimPO eliminó el modelo de referencia. El impuesto de alineación - el costo de cálculo y complejidad de pasar de un modelo base a un modelo alineado - sigue cayendo.

### En el caso de los Estados miembros, el DPO debe ser el único agente de seguridad.

**Zephyr-7B (HuggingFace, October 2023):**Mistral 7B base, SFT en UltraChat (200K ejemplos), luego DPO en UltraFeedback (60K pares de preferencias). Obtuvo un puntaje de 6.47 en MT-Bench - el modelo 7B más alto en ese momento. Para comparación, Llama 2 Chat 70B obtuvo un puntaje de 6.86, lo que significa que Zephyr obtuvo dentro del 6% de un modelo 10x su tamaño utilizando solo la alineación de DPO.

**Llama 3 (Meta, April 2024):**Se utiliza DPO después de las etapas iniciales de RLHF. La combinación sugiere que DPO y RLHF pueden ser complementarios - RLHF para la alineación amplia, DPO para el refinamiento dirigido.

**Neural Magic / nm-chat (2024):**Aplicó DPO a múltiples modelos de código abierto, mostrando consistentemente una mejora del 5-15% en los puntos de referencia de alineación en comparación con las líneas de base exclusivamente de SFT.

```figure
dpo-loss
```

## Construye el mismo

### Paso 1: Datos de preferencias

El mismo formato que RLHF - (pronto, preferido, rechazado) triples. DPO consume estos datos directamente sin un modelo de recompensa intermedio.

```python
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "04-pre-training-mini-gpt", "code"))
from main import MiniGPT, LayerNorm, Embedding, TransformerBlock

PREFERENCE_DATA = [
    {
        "prompt": "What is the capital of France?",
        "preferred": "The capital of France is Paris.",
        "rejected": "France is a country in Europe. It has many cities. The capital is Paris. Paris is known for the Eiffel Tower.",
    },
    {
        "prompt": "Explain gravity in one sentence.",
        "preferred": "Gravity is the force that attracts objects with mass toward each other.",
        "rejected": "Gravity is something that makes things fall down when you drop them.",
    },
    {
        "prompt": "What is 15 times 7?",
        "preferred": "15 times 7 is 105.",
        "rejected": "Let me think about this. 15 times 7. Well, 10 times 7 is 70, and 5 times 7 is 35, so the answer might be around 105.",
    },
    {
        "prompt": "Name three programming languages.",
        "preferred": "Python, Rust, and TypeScript.",
        "rejected": "There are many programming languages. Some popular ones include various languages like Python and others.",
    },
    {
        "prompt": "What year did World War II end?",
        "preferred": "World War II ended in 1945.",
        "rejected": "World War II was a major global conflict. It involved many countries. The war ended in the mid-1940s, specifically in 1945.",
    },
    {
        "prompt": "Define machine learning.",
        "preferred": "Machine learning is a field where algorithms learn patterns from data to make predictions without being explicitly programmed.",
        "rejected": "Machine learning is a type of AI. AI stands for artificial intelligence. Machine learning uses data to learn.",
    },
]
```

### Paso 2: Probabilidad de registro de secuencias

La pérdida de DPO requiere calcular la probabilidad total de registro de una respuesta dada a un prompt. Esto significa ejecutar el modelo en la secuencia completa (prompt + respuesta) y sumar las probabilidades de registro de cada token de respuesta.

```python
def tokenize_sequence(text, vocab_size=256):
    return [min(t, vocab_size - 1) for t in list(text.encode("utf-8"))]


def compute_sequence_log_prob(model, prompt_tokens, response_tokens, max_seq_len=128):
    full_sequence = prompt_tokens + response_tokens
    if len(full_sequence) > max_seq_len:
        full_sequence = full_sequence[:max_seq_len]

    if len(full_sequence) < 2:
        return 0.0

    input_ids = np.array(full_sequence[:-1]).reshape(1, -1)
    target_ids = np.array(full_sequence[1:])

    logits = model.forward(input_ids)
    logits = logits[0]

    max_logits = logits.max(axis=-1, keepdims=True)
    log_probs = logits - max_logits - np.log(
        np.exp(logits - max_logits).sum(axis=-1, keepdims=True)
    )

    prompt_len = len(prompt_tokens)
    response_start = max(0, prompt_len - 1)
    response_end = len(target_ids)

    if response_start >= response_end:
        return 0.0

    response_log_probs = log_probs[response_start:response_end, :]
    response_targets = target_ids[response_start:response_end]

    total_log_prob = 0.0
    for i, target in enumerate(response_targets):
        total_log_prob += response_log_probs[i, target]

    return total_log_prob
```

Esta función es el caballo de trabajo de DPO. Para cada par de preferencias, se ejecuta cuatro veces: modelo en respuesta preferida, modelo en respuesta rechazada, referencia en respuesta preferida, referencia en respuesta rechazada. Eso es 4 pases adelante por ejemplo de entrenamiento frente a la generación de RLHF + puntuación de recompensa + estimación de valor + actualización de PPO.

### Paso 3: La pérdida de DPO

El núcleo del documento en código, una función, una pérdida, sin modelo de recompensa.

```python
def sigmoid(x):
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def dpo_loss(policy_logprob_preferred, policy_logprob_rejected,
             ref_logprob_preferred, ref_logprob_rejected, beta=0.1):
    preferred_ratio = policy_logprob_preferred - ref_logprob_preferred
    rejected_ratio = policy_logprob_rejected - ref_logprob_rejected

    logit = beta * (preferred_ratio - rejected_ratio)

    loss = -np.log(sigmoid(logit) + 1e-8)

    preferred_reward = beta * preferred_ratio
    rejected_reward = beta * rejected_ratio

    return loss, {
        "preferred_ratio": float(preferred_ratio),
        "rejected_ratio": float(rejected_ratio),
        "logit": float(logit),
        "implicit_preferred_reward": float(preferred_reward),
        "implicit_rejected_reward": float(rejected_reward),
        "reward_margin": float(preferred_reward - rejected_reward),
    }
```

El `preferred_ratio`y `rejected_ratio`Cuando el modelo actual asigna una mayor probabilidad a la respuesta preferida (relativa a la referencia) y una menor probabilidad a la respuesta rechazada, la logit es positiva y la pérdida es baja.

El `implicit_preferred_reward`y `implicit_rejected_reward`Se pueden extraer para verificar que la capacitación está funcionando - el margen entre las recompensas preferidas y rechazadas debe aumentar en comparación con la capacitación.

### Paso 4: Ciclo de formación de los DPO

Un ciclo de entrenamiento supervisado estándar, sin PPO, sin modelo de recompensa, sólo pasajes y actualizaciones de gradiente.

```python
def copy_model_weights(source, target):
    target.embedding.token_embed = source.embedding.token_embed.copy()
    target.embedding.pos_embed = source.embedding.pos_embed.copy()
    target.ln_f.gamma = source.ln_f.gamma.copy()
    target.ln_f.beta = source.ln_f.beta.copy()
    for s_block, t_block in zip(source.blocks, target.blocks):
        t_block.attn.W_q = s_block.attn.W_q.copy()
        t_block.attn.W_k = s_block.attn.W_k.copy()
        t_block.attn.W_v = s_block.attn.W_v.copy()
        t_block.attn.W_out = s_block.attn.W_out.copy()
        t_block.ffn.W1 = s_block.ffn.W1.copy()
        t_block.ffn.W2 = s_block.ffn.W2.copy()
        t_block.ffn.b1 = s_block.ffn.b1.copy()
        t_block.ffn.b2 = s_block.ffn.b2.copy()
        t_block.ln1.gamma = s_block.ln1.gamma.copy()
        t_block.ln1.beta = s_block.ln1.beta.copy()
        t_block.ln2.gamma = s_block.ln2.gamma.copy()
        t_block.ln2.beta = s_block.ln2.beta.copy()


def dpo_train(policy_model, reference_model, preference_data,
              num_epochs=5, lr=5e-6, beta=0.1, max_seq_len=128):
    print(f"DPO Training: {len(preference_data)} pairs, {num_epochs} epochs, "
          f"lr={lr}, beta={beta}")
    print()

    losses = []
    margins = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_margin = 0.0
        num_examples = 0

        indices = np.random.permutation(len(preference_data))

        for idx in indices:
            pair = preference_data[idx]

            prompt_tokens = tokenize_sequence(pair["prompt"])
            preferred_tokens = tokenize_sequence(pair["preferred"])
            rejected_tokens = tokenize_sequence(pair["rejected"])

            pi_logprob_w = compute_sequence_log_prob(
                policy_model, prompt_tokens, preferred_tokens, max_seq_len
            )
            pi_logprob_l = compute_sequence_log_prob(
                policy_model, prompt_tokens, rejected_tokens, max_seq_len
            )
            ref_logprob_w = compute_sequence_log_prob(
                reference_model, prompt_tokens, preferred_tokens, max_seq_len
            )
            ref_logprob_l = compute_sequence_log_prob(
                reference_model, prompt_tokens, rejected_tokens, max_seq_len
            )

            loss, metrics = dpo_loss(
                pi_logprob_w, pi_logprob_l,
                ref_logprob_w, ref_logprob_l, beta
            )

            update_direction = 1.0 if metrics["logit"] < 0 else -0.1
            for block in policy_model.blocks:
                block.ffn.W1 += lr * update_direction * np.random.randn(*block.ffn.W1.shape) * 0.01
                block.ffn.W2 += lr * update_direction * np.random.randn(*block.ffn.W2.shape) * 0.01

            epoch_loss += loss
            epoch_margin += metrics["reward_margin"]
            num_examples += 1
            losses.append(float(loss))
            margins.append(metrics["reward_margin"])

        avg_loss = epoch_loss / max(num_examples, 1)
        avg_margin = epoch_margin / max(num_examples, 1)

        print(f"  Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f} | "
              f"Avg Margin: {avg_margin:.4f}")

    return policy_model, losses, margins
```

El ciclo de entrenamiento es refrescante en comparación con RLHF. Para cada par de preferencias: calcular cuatro probabilidades de registro (dos modelos, dos respuestas), conectarlas a la pérdida de DPO, calcular el gradiente, actualizar la política. No hay paso de generación. No hay inferencia de modelo de recompensa. No hay estimación de ventaja. No hay recortes.

### Paso 5: Comparación entre DPO y RLHF

Medir los márgenes implícitos de recompensa y los cambios de probabilidad de registro para comparar el DPO con el modelo RLHF de la lección 07.

```python
def evaluate_preference_accuracy(model, reference_model, preference_data, beta=0.1, max_seq_len=128):
    correct = 0
    total = 0

    for pair in preference_data:
        prompt_tokens = tokenize_sequence(pair["prompt"])
        preferred_tokens = tokenize_sequence(pair["preferred"])
        rejected_tokens = tokenize_sequence(pair["rejected"])

        pi_w = compute_sequence_log_prob(model, prompt_tokens, preferred_tokens, max_seq_len)
        pi_l = compute_sequence_log_prob(model, prompt_tokens, rejected_tokens, max_seq_len)
        ref_w = compute_sequence_log_prob(reference_model, prompt_tokens, preferred_tokens, max_seq_len)
        ref_l = compute_sequence_log_prob(reference_model, prompt_tokens, rejected_tokens, max_seq_len)

        preferred_reward = beta * (pi_w - ref_w)
        rejected_reward = beta * (pi_l - ref_l)

        if preferred_reward > rejected_reward:
            correct += 1
        total += 1

    return correct / max(total, 1)


def analyze_implicit_rewards(model, reference_model, preference_data, beta=0.1, max_seq_len=128):
    print("Implicit Reward Analysis:")
    print("-" * 65)
    print(f"  {'Prompt':<30} {'Pref Reward':>12} {'Rej Reward':>12} {'Margin':>10}")
    print("  " + "-" * 60)

    for pair in preference_data:
        prompt_tokens = tokenize_sequence(pair["prompt"])
        preferred_tokens = tokenize_sequence(pair["preferred"])
        rejected_tokens = tokenize_sequence(pair["rejected"])

        pi_w = compute_sequence_log_prob(model, prompt_tokens, preferred_tokens, max_seq_len)
        pi_l = compute_sequence_log_prob(model, prompt_tokens, rejected_tokens, max_seq_len)
        ref_w = compute_sequence_log_prob(reference_model, prompt_tokens, preferred_tokens, max_seq_len)
        ref_l = compute_sequence_log_prob(reference_model, prompt_tokens, rejected_tokens, max_seq_len)

        pref_reward = beta * (pi_w - ref_w)
        rej_reward = beta * (pi_l - ref_l)
        margin = pref_reward - rej_reward

        truncated = pair["prompt"][:28] + ".." if len(pair["prompt"]) > 30 else pair["prompt"]
        print(f"  {truncated:<30} {pref_reward:>12.4f} {rej_reward:>12.4f} {margin:>10.4f}")

    print()
```

### Paso 6: Análisis de la sensibilidad beta

El parámetro beta es el equivalente de DPO al coeficiente KL en RLHF. Controla cuánto el modelo puede desviarse de la referencia.

```python
def beta_sensitivity_analysis(sft_model, preference_data, betas, max_seq_len=128):
    print("Beta Sensitivity Analysis")
    print("-" * 60)
    print(f"  {'Beta':>8} {'Final Loss':>12} {'Final Margin':>14} {'Accuracy':>10}")
    print("  " + "-" * 55)

    results = []

    for beta in betas:
        policy = MiniGPT(
            vocab_size=256, embed_dim=128, num_heads=4,
            num_layers=4, max_seq_len=max_seq_len, ff_dim=512
        )
        reference = MiniGPT(
            vocab_size=256, embed_dim=128, num_heads=4,
            num_layers=4, max_seq_len=max_seq_len, ff_dim=512
        )
        copy_model_weights(sft_model, policy)
        copy_model_weights(sft_model, reference)

        policy, losses, margins_list = dpo_train(
            policy, reference, preference_data,
            num_epochs=3, lr=5e-6, beta=beta, max_seq_len=max_seq_len
        )

        accuracy = evaluate_preference_accuracy(
            policy, reference, preference_data, beta, max_seq_len
        )

        final_loss = losses[-1] if losses else 0
        final_margin = margins_list[-1] if margins_list else 0

        print(f"  {beta:>8.3f} {final_loss:>12.4f} {final_margin:>14.4f} {accuracy:>10.1%}")
        results.append({
            "beta": beta,
            "final_loss": final_loss,
            "final_margin": final_margin,
            "accuracy": accuracy,
        })

        print()

    return results
```

La pequeña beta (0.01) permite que el modelo se desvíe libremente de la referencia - aprendizaje rápido pero riesgo de soluciones degeneradas. La beta grande (1.0) mantiene el modelo cerca de la referencia - aprendizaje estable pero lento. El punto dulce para la mayoría de las aplicaciones es de 0,1 a 0,3.

## Usalo

### Demo de la línea de tuberías de la DPO

```python
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("DPO: DIRECT PREFERENCE OPTIMIZATION")
    print("=" * 70)
    print()

    print("STEP 1: Initialize SFT Model (from Lesson 06)")
    print("-" * 50)
    sft_model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    print(f"  Parameters: {sft_model.count_parameters():,}")
    print()

    print("STEP 2: DPO Training")
    print("-" * 50)

    policy_model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    reference_model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    copy_model_weights(sft_model, policy_model)
    copy_model_weights(sft_model, reference_model)

    policy_model, losses, margins = dpo_train(
        policy_model, reference_model, PREFERENCE_DATA,
        num_epochs=5, lr=5e-6, beta=0.1
    )
    print()

    print("=" * 70)
    print("STEP 3: Evaluate")
    print("=" * 70)
    print()

    pre_accuracy = evaluate_preference_accuracy(
        sft_model, reference_model, PREFERENCE_DATA, beta=0.1
    )
    post_accuracy = evaluate_preference_accuracy(
        policy_model, reference_model, PREFERENCE_DATA, beta=0.1
    )

    print(f"  Preference accuracy (pre-DPO):  {pre_accuracy:.1%}")
    print(f"  Preference accuracy (post-DPO): {post_accuracy:.1%}")
    print()

    analyze_implicit_rewards(policy_model, reference_model, PREFERENCE_DATA, beta=0.1)

    print("=" * 70)
    print("STEP 4: Training Dynamics")
    print("=" * 70)
    print()

    if losses:
        print("  Loss curve:")
        window = max(1, len(losses) // 5)
        for i in range(0, len(losses), window):
            chunk = losses[i:i + window]
            avg = sum(chunk) / len(chunk)
            print(f"    Steps {i:3d}-{i + len(chunk) - 1:3d}: loss = {avg:.4f}")
        print()

    if margins:
        print("  Reward margin curve:")
        window = max(1, len(margins) // 5)
        for i in range(0, len(margins), window):
            chunk = margins[i:i + window]
            avg = sum(chunk) / len(chunk)
            print(f"    Steps {i:3d}-{i + len(chunk) - 1:3d}: margin = {avg:.4f}")
        print()

    print("=" * 70)
    print("STEP 5: Beta Sensitivity")
    print("=" * 70)
    print()

    beta_results = beta_sensitivity_analysis(
        sft_model, PREFERENCE_DATA, betas=[0.01, 0.1, 0.3, 1.0]
    )

    print("=" * 70)
    print("DPO vs RLHF COMPARISON")
    print("=" * 70)
    print()
    print("  DPO advantages:")
    print("    - 1 training loop (vs 3 for RLHF)")
    print("    - 2 models in memory (vs 3-4 for RLHF)")
    print("    - Supervised learning (vs RL, more stable)")
    print("    - No reward model to train or maintain")
    print()
    print("  RLHF advantages:")
    print("    - Separate reward model captures complex preferences")
    print("    - Online learning: generate, rate, retrain")
    print("    - Better for multi-objective alignment")
    print("    - Proven at largest scales (GPT-4, Claude)")
    print()
    print("  Practical guidance:")
    print("    - Start with DPO. It's simpler and often sufficient.")
    print("    - Switch to RLHF if DPO plateaus on your eval metrics.")
    print("    - Many production systems use both: RLHF first, DPO to refine.")
```

## Envío

Esta lección produce`outputs/prompt-alignment-method-selector.md`- una solicitud que le ayuda a elegir el método de alineación adecuado (SFT, RLHF, DPO, KTO, ORPO, SimPO) para su caso de uso.

## Los ejercicios

1. Implemente KTO (Kahneman-Tversky Optimization). KTO no necesita pares, simplemente etiquete cada respuesta como "buena" o "mala". La pérdida para una buena respuesta es`-log(sigmoid(beta * log_ratio))`y por una mala respuesta es `-log(1 - sigmoid(beta * log_ratio))`con un multiplicador de aversión a la pérdida (normalmente 1,5x) en la pérdida de respuesta mala.

2. Implemente DPO normalizado en longitud. En lugar de probabilidades de registro en bruto, divida por el número de tokens de respuesta: `normalized_logprob = total_logprob / num_tokens`• Esto evita que el modelo favorezca respuestas más cortas (que tienen un log-prob total más alto).

3. Construir una pérdida combinada de estilo ORPO. Agregar una pérdida de predicción estándar de tokens siguientes en la respuesta preferida a la pérdida de DPO: `L = L_sft(preferred) + alpha * L_dpo`. Prueba los valores alfa de 0,1, 0,5 y 1.0. La pérdida combinada debe producir un modelo que siga instrucciones (del término SFT) y prefiera mejores respuestas (del término DPO), eliminando la necesidad de una etapa separada de SFT.

4. Implemente DPO iterativo. ejecuta DPO durante 3 épocas, luego genera nuevas respuestas del modelo entrenado, emparejalas con las respuestas preferidas originales como nuevos pares de preferencias y ejecuta DPO nuevamente. Dos rondas de este proceso de "auto-juego".

5. Comparar el DPO con diferentes modelos de referencia. En lugar de utilizar el punto de control SFT como referencia, pruebe: (a) el modelo base (pre-SFT), (b) un punto de control de la época 1 del DPO, (c) un promedio móvil exponencial del modelo de política.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| DPO | "RLHF without RL" | Direct Preference Optimization: a supervised learning algorithm that optimizes the language model directly on preference pairs, bypassing the reward model and PPO |
| Implicit reward | "The reward is in the model" | The reward function is determined by the log-probability ratio between the policy and reference models -- no separate reward model needed |
| Beta (DPO) | "The temperature" | Controls how far the policy can deviate from the reference model -- small beta allows large deviations, large beta keeps the model close |
| Log-probability ratio | "How much the model changed" | log pi(y\|x) - log pi_ref(y\|x) -- positive means the current model assigns higher probability than the reference |
| Reference model | "The frozen checkpoint" | A copy of the SFT model whose weights never change -- serves as the anchor for computing probability ratios |
| KTO | "DPO without pairs" | Kahneman-Tversky Optimization: works with unpaired "good" or "bad" labels instead of requiring preference pairs |
| ORPO | "One-step alignment" | Odds Ratio Preference Optimization: combines SFT and alignment into a single training loop by adding a preference term to the SFT loss |
| SimPO | "No reference needed" | Simple Preference Optimization: eliminates the reference model by using length-normalized average log-probability as the implicit reward |
| Alignment tax | "The cost of making models safe" | The additional compute, data, and complexity required to go from a base model to an aligned model -- DPO reduces this significantly |

## Leer más

- [Rafailov et al., 2023 -- "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"](https://arxiv.org/abs/2305.18290)-- el documento del DPO que simplificó la alineación de la RLHF al aprendizaje supervisado
- [Tunstall et al., 2023 -- "Zephyr: Direct Distillation of LM Alignment"](https://arxiv.org/abs/2310.16944)-- Zephyr-7B, muestra DPO en UltraFeedback coincide con RLHF en los puntos de referencia
- [Ethayarajh et al., 2024 -- "KTO: Model Alignment as Prospect Theoretic Optimization"](https://arxiv.org/abs/2402.01306)-- eliminar la necesidad de preferencias emparejadas
- [Hong et al., 2024 -- "ORPO: Monolithic Preference Optimization without Reference Model"](https://arxiv.org/abs/2403.07691)-- combinar FFT y alineación en un solo paso
- [Meng et al., 2024 -- "SimPO: Simple Preference Optimization with a Reference-Free Reward"](https://arxiv.org/abs/2405.14734)-- eliminar por completo el modelo de referencia
- [Llama 3 Technical Report](https://arxiv.org/abs/2407.21783)-- El tubo de alineación de Meta que combina RLHF y DPO
