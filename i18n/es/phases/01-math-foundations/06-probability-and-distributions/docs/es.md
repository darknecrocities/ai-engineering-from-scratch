# Probabilidad y distribución

> La probabilidad es el lenguaje que utiliza la IA para expresar la incertidumbre.

**Type:** Learn
**Language:**Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## Objetivos de aprendizaje

- Implementar los PMF y los PDF desde cero para las distribuciones Bernoulli, categórica, Poisson, uniforme y normal
- Computa el valor esperado, la varianza y usa el Teorema del límite central para explicar por qué los gaussianos dominan
- Construir las funciones softmax y log-softmax con el truco de estabilidad numérica (sustraer logit máximo)
- Calcular la pérdida de entropía cruzada de logits y conectarla a la probabilidad de log negativo

## El problema

Una salida de clasificador `[0.03, 0.91, 0.06]`Un modelo de lenguaje elige la siguiente palabra de 50.000 candidatos. Un modelo de difusión genera imágenes tomando muestras de distribuciones aprendidas.

Cada predicción que hace un modelo es una distribución de probabilidades. Cada función de pérdida mide cuán lejos está la distribución prevista de la verdadera. Cada paso de entrenamiento ajusta los parámetros para que una distribución se vea más parecida a otra. Sin probabilidad, no puedes leer un solo documento de ML, deshacerte de un solo modelo o entender por qué tu pérdida de entrenamiento es NaN.

## El concepto

### Eventos, espacios de muestra y probabilidad

El espacio de muestra S es el conjunto de todos los resultados posibles. Un evento es un subconjunto del espacio de muestra.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

Tres axiomas definen toda la probabilidad:
1. P(A) >= 0 para cualquier evento A
2. P(S) = 1 (algo siempre sucede)
3. P(A o B) = P(A) + P(B) cuando A y B no pueden ocurrir ambas

Todo lo demás (teorema de Bayes, expectativas, distribuciones) se sigue de estas tres reglas.

### Probabilidad condicional y independencia

P ((A) B) es la probabilidad de A dada que B sucedió.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

Dos eventos son independientes cuando saber uno no te dice nada del otro:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

Los tiros de monedas son independientes, pero los tiros sin reemplazo no lo son.

### Funciones de masa de probabilidad vs. funciones de densidad de probabilidad

Las variables aleatorias discretas tienen una función de masa de probabilidad (PMF). Cada resultado tiene una probabilidad específica que se puede leer directamente.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

Las variables aleatorias continuas tienen una función de densidad de probabilidad (PDF). La densidad en un solo punto no es una probabilidad.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

Esta distinción es importante en ML. Las salidas de clasificación son PMF (elecciones discretas).

### Distribuciones comunes

**Bernoulli:**Un ensayo, dos resultados.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical:**Los modelos de clasificación de clases múltiples (salida de la máxima suavidad).

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform:**Todos los resultados son igualmente probables.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian):**La curva de la campana. Parameterizada por medio (mu) y varianza (sigma^2).

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:**El número de eventos raros en un intervalo fijo.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### Valor esperado y variación

El valor esperado es el resultado medio ponderado.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

Las medidas de variación se extienden alrededor de la media.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

En ML, el valor esperado aparece como la función de pérdida (pérdida promedio sobre la distribución de datos).

### Distribuciones conjuntas y marginales

Una distribución conjunta P ((X, Y) describe dos variables aleatorias juntas.

Ejemplo de PMF conjunta (X = clima, Y = paraguas):

| | Y=0 (no umbrella) | Y=1 (umbrella) | Marginal P(X) |
|---|---|---|---|
| X=0 (sun) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (rain) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Marginal P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

La distribución marginal suma la otra variable:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

Los totales de filas y columnas de la tabla anterior son los márgenes.

### Por qué la distribución normal aparece en todas partes

El Teorema del límite central: la suma (o promedio) de muchas variables aleatorias independientes converge a una distribución normal, independientemente de la distribución original.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

Por eso es que:
- Los errores de medición son aproximadamente normales (muchas pequeñas fuentes independientes)
- Las inicializaciones de peso en las redes neuronales utilizan distribuciones normales
- El ruido gradiente en SGD es aproximadamente normal (suma de muchos gradientes de muestra)
- La distribución normal es la distribución máxima de entropía para una media y varianza dada

### Probabilidades de registro

Las probabilidades crudas causan problemas numéricos. Multiplicar muchas probabilidades pequeñas juntas rápidamente se subtrae a cero.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

Las probabilidades de registro arreglan esto. Las multiplicaciones se convierten en adiciones.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

Reglas:
- log(a * b) = log(a) + log(b)
- Las probabilidades de registro son siempre <= 0 (ya que 0 < P <= 1)
- Más negativo = menos probable
- La pérdida de entropía cruzada es la probabilidad de registro negativo de la clase correcta

### Softmax como distribución de probabilidad

Las redes neuronales producen puntajes en bruto (logits). Softmax los convierte en una distribución de probabilidades válida.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

El truco de softmax: restar la máxima logit antes de exponenciar para evitar el desbordamiento.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax combina softmax y log para la estabilidad numérica. PyTorch utiliza esto internamente para la pérdida de entropía cruzada.

### Muestreo

Muestreo significa extraer valores aleatorios de una distribución.
- Dejar de tomar muestras aleatorias de qué neuronas se desprenden
- Muestras de aumento de datos transformaciones aleatorias
- Los modelos de lenguaje muestran el siguiente token de la distribución prevista
- Modelos de difusión muestran ruido y denotan progresivamente

La muestreo de distribuciones arbitrarias requiere técnicas como muestreo de transformación inversa, muestreo de rechazo o el truco de reparameterización (utilizado en VAEs).

```figure
gaussian-pdf
```

## Construye el mismo

### Paso 1: Bases de probabilidad

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Paso 2: PMF y PDF desde cero

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Paso 3: Valor esperado y variación

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Paso 4: Muestreo de las distribuciones

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Paso 5: Softmax y probabilidades de registro

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Paso 6: Teorema de límite central demostración

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Paso 7: Visualización

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Las implementaciones completas con todas las visualizaciones están en `code/probability.py`¿ Qué ?

## Usalo

Con NumPy y SciPy, todo lo anterior es de una sola línea:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

Construiste esto desde cero, ahora sabes lo que hacen las llamadas de la biblioteca.

## Los ejercicios

1. Implemente muestreo inverso de transformación para la distribución exponencial. Verifique tomando muestras de 10.000 valores y comparando el histograma con el PDF real.

2. Construye una tabla de distribución conjunta para dos dados cargados, computa las distribuciones marginales y comprueba si los dados son independientes.

3. Calcule la pérdida de entropía cruzada para un clasificador de 5 clases que emita logits `[2.0, 0.5, -1.0, 3.0, 0.1]`Cuando la clase correcta es el índice 3. Entonces verifique su respuesta con PyTorch's `nn.CrossEntropyLoss`¿ Qué ?

4. Escriba una función que tome una lista de probabilidades de registro y devuelve la secuencia más probable, la probabilidad total de registro y la probabilidad bruta equivalente. Pruébalo con una oración de 50 palabras donde cada palabra tiene probabilidad 0.01.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Sample space | "All the possibilities" | The set S of every possible outcome of an experiment |
| PMF | "The probability function" | A function that gives the exact probability of each discrete outcome, summing to 1 |
| PDF | "The probability curve" | A density function for continuous variables. Integrate it over an interval to get probability |
| Conditional probability | "Probability given something" | P(A\|B) = P(A and B) / P(B). The foundation of Bayesian thinking and Bayes' theorem |
| Independence | "They don't affect each other" | P(A and B) = P(A) * P(B). Knowing one event tells you nothing about the other |
| Expected value | "The average" | The probability-weighted sum of all outcomes. The loss function is an expected value |
| Variance | "How spread out" | The expected squared deviation from the mean. High variance = noisy, unstable estimates |
| Normal distribution | "The bell curve" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). Appears everywhere due to the CLT |
| Central Limit Theorem | "Averages become normal" | The mean of many independent samples converges to a normal distribution regardless of the source |
| Joint distribution | "Two variables together" | P(X, Y) describes the probability of every combination of X and Y outcomes |
| Marginal distribution | "Sum out the other variable" | P(X) = sum_y P(X, Y). Recovers one variable's distribution from the joint |
| Log probability | "Log of the probability" | log P(x). Turns products into sums, preventing numerical underflow in long sequences |
| Softmax | "Turn scores into probabilities" | softmax(z_i) = exp(z_i) / sum(exp(z_j)). Maps real-valued logits to a valid probability distribution |
| Cross-entropy | "The loss function" | -sum(p_true * log(p_predicted)). Measures how different two distributions are. Lower is better |
| Logits | "Raw model outputs" | Unnormalized scores before softmax. Named after the logistic function |
| Sampling | "Drawing random values" | Generating values according to a probability distribution. How models generate output |

## Leer más

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo)- prueba visual de por qué las medias se vuelven normales
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf)- una referencia concisa que cubre todo aquí y más
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)- por qué es importante la estabilidad numérica y cómo lograrla
