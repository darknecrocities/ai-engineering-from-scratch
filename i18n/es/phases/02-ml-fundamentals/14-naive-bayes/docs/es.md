# Bayes ingenuo

> La suposición "ingenuo" es incorrecta, y funciona de todos modos.

**Type:** Build
**Language:**Python
**Prerequisites:** Phase 2, Lessons 01-07 (classification, Bayes' theorem)
**Time:** ~75 minutes

## Objetivos de aprendizaje

- Implemente Bayes multinomial ingenuos desde cero con suavizamiento Laplace para la clasificación de texto
- Explica por qué la ingenuidad de la suposición de independencia es matemáticamente incorrecta pero produce clasificaciones correctas de clases en la práctica
- Comparar las variantes multinomial, Bernoulli y Gaussian Naive Bayes y seleccionar la correcta para un tipo de característica dado
- Evaluar Bayes Ingenuo contra la regresión logística en datos escasos de alta dimensión y explicar el compromiso de variación de sesgo en el trabajo

## El problema

Necesitas clasificar el texto. Los correos electrónicos en spam o no spam. Las reseñas de los clientes en positivos o negativos. Los boletos de soporte en categorías. Tienes miles de características (una por palabra) y datos de capacitación limitados.

La mayoría de los clasificadores se ahogan aquí. La regresión logística necesita suficientes muestras para estimar miles de pesas de manera confiable. Los árboles de decisión se dividen en una palabra a la vez y se encajan salvajemente. KNN en 10.000 dimensiones no tiene sentido porque cada punto está igualmente lejos de todos los otros puntos.

Bayes ingenuo se encarga de esto. Hace una suposición matemáticamente incorrecta (que cada característica es independiente de cualquier otra característica dada la clase), y todavía supera a los modelos "más inteligentes" en la clasificación de texto, especialmente con pequeños conjuntos de capacitación. Se entrena en un solo paso a través de los datos. Se extiende a millones de características. Produce estimaciones de probabilidad (aunque a menudo mal calibradas debido a la suposición de independencia).

Comprender por qué una suposición errónea conduce a buenas predicciones nos enseña algo fundamental sobre el aprendizaje automático: el mejor modelo no es el más correcto, es el que tiene el mejor compromiso de variaciones de sesgo para sus datos.

## El concepto

### Teorema de Bayes (Revisión Rápida)

El teorema de Bayes inverte las probabilidades condicionales:

```
P(class | features) = P(features | class) * P(class) / P(features)
```

Queremos`P(class | features)`-- la probabilidad de que un documento pertenece a una clase dada las palabras en ella.
- `P(features | class)`-- la probabilidad de ver estas palabras en documentos de esta clase
- `P(class)`-- la probabilidad previa de la clase (¿qué tan común es el spam en general?)
- `P(features)`- la evidencia, igual para todas las clases, así que podemos ignorarlo cuando comparamos

La clase con más alto .`P(class | features)`- ¿Qué?

### La suposición ingenua de la independencia

Computación`P(features | class)`Si se tiene un vocabulario de 10.000 palabras, se necesita estimar una distribución de 2^10.000 combinaciones posibles.

La suposición ingenua: cada característica es condicionalmente independiente dada la clase.

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

En lugar de una distribución conjunta imposible, se estima n distribuciones simples por característica.

Esta suposición es obviamente incorrecta. Las palabras "máquina" y "aprendizaje" no son independientes en ningún documento. Pero el clasificador no necesita estimaciones de probabilidad correctas. Necesita clasificaciones correctas - que clase tiene la mayor probabilidad. La suposición de independencia introduce errores sistemáticos, pero esos errores afectan a todas las clases de manera similar, por lo que el ranking se mantiene correcto.

### Por qué todavía funciona

Tres razones:

1. **Ranking over calibration.**La clasificación sólo necesita que la clase de clasificación superior sea correcta. Incluso si P(spam) = 0.99999 cuando la probabilidad verdadera es 0.7, el clasificador sigue escogiendo spam correctamente. No necesitamos probabilidades correctas. Necesitamos el ganador correcto.

2. **High bias, low variance.**La hipótesis de independencia es una prioridad fuerte. Restringe el modelo fuertemente, lo que evita el sobreajuste. Con datos de entrenamiento limitados, un modelo ligeramente equivocado pero estable supera a un modelo que es teóricamente correcto pero muy inestable.

3. **Feature redundancy cancels out.**Las características correlacionadas proporcionan evidencia redundante. El clasificador cuenta esta evidencia dos veces, pero también la cuenta dos veces para la clase correcta. Si "máquina" y "aprendizaje" siempre aparecen juntos, ambos proporcionan evidencia para la clase "tecnológica". NB las cuenta dos veces, pero las cuenta dos veces para la clase correcta.

Una cuarta razón práctica: Bayes ingenuo es extremadamente rápido. El entrenamiento es un solo paso a través de las frecuencias de conteo de datos. La predicción es una multiplicación de matriz. Puedes entrenar en un millón de documentos en segundos. Esta velocidad significa que puedes iterar más rápido, probar más conjuntos de características y ejecutar más experimentos que con modelos más lentos.

### El matemático paso a paso

Vamos a rastrear a través de un ejemplo concreto. Supongamos que tenemos dos clases: spam y no spam. Nuestro vocabulario tiene tres palabras: "gratuito", "dinero", "reunión".

Datos de formación:
- Los correos electrónicos de spam mencionan "gratuito" 80 veces, "dinero" 60 veces, "encuentro" 10 veces (150 palabras totales)
- Los correos electrónicos no spam mencionan "gratuito" 5 veces, "dinero" 10 veces, "encuentro" 100 veces (115 palabras totales)
- El 40% de los correos electrónicos son spam, el 60% no son spam

Con suavizamiento Laplace (alfa=1):

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

El nuevo correo electrónico contiene: "gratuito" (2 veces), "dinero" (1 vez), "reunión" (0 veces).

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

El spam gana con un gran margen. La palabra "libre" que aparece dos veces es una fuerte evidencia de spam.

### Tres variantes

Bayes Ingenuo viene en tres sabores.`P(feature | class)`de manera diferente.

#### Bayes Ingenuo Multinomio

Modelos de cada característica como un recuento. Mejor para datos de texto donde las características son frecuencias de palabras o valores TF-IDF.

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

El `alpha`La variante de la clasificación de texto es la de Laplace (explicada a continuación).

#### Bayes Ingenuo Gaussiano

Modela cada característica como una distribución normal.

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

Cada clase tiene su propia media y variación por característica. Esto funciona bien cuando las características realmente siguen una curva de campanas dentro de cada clase.

#### Bernoulli Ingenuo Bayes

Modelos de cada característica como binaria (presente o ausente). Mejor para texto corto o vectores de características binarias.

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

A diferencia de Multinomial, Bernoulli penaliza explícitamente la ausencia de una palabra. Si "libre" aparece típicamente en el correo electrónico, pero está ausente de este correo electrónico, Bernoulli cuenta eso como evidencia contra el correo electrónico.

### Cuándo usar cada variante

| Variant | Feature Type | Best For | Example |
|---------|-------------|----------|---------|
| Multinomial | Counts or frequencies | Text classification, bag-of-words | Email spam, topic classification |
| Gaussian | Continuous values | Tabular data with normal-ish features | Iris classification, sensor data |
| Bernoulli | Binary (0/1) | Short text, binary feature vectors | SMS spam, presence/absence features |

### Laplace Smoothing

¿Qué sucede cuando una palabra aparece en los datos de prueba pero nunca aparece en los datos de formación para una clase en particular?

Sin suavizar:`P(word | class) = 0/N = 0`Un cero multiplicado por todo el producto hace`P(class | features) = 0`Una sola palabra invisible destruye toda la predicción, sin importar cuánta evidencia la apoye.

El suavización de la zona de la zona añade un pequeño recuento `alpha`(generalmente 1) a cada número de características:

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

Con alpha=1, cada palabra obtiene al menos una pequeña probabilidad. La palabra "discombobulate" que aparece en un correo electrónico de prueba ya no mata la probabilidad de spam. El suavización tiene una interpretación bayesiana: es equivalente a colocar un Dirichlet uniforme antes de las distribuciones de palabras.

El alfa superior significa una suavizamiento más fuerte (distribuciones más uniformes).

El efecto de alfa:

| Alpha | Effect | When to use |
|-------|--------|-------------|
| 0.001 | Almost no smoothing, trust the data | Very large training set, no unseen features expected |
| 0.1 | Light smoothing | Large training set |
| 1.0 | Standard Laplace smoothing | Default starting point |
| 10.0 | Heavy smoothing, flattens distributions | Very small training set, many unseen features expected |

### Computación del espacio log-espacio

El producto se convierte en cero en el punto flotante aunque el valor verdadero es un número positivo muy pequeño.

La solución: trabajar en el espacio de registro. En lugar de multiplicar las probabilidades, agregar sus logaritmos:

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

Esto convierte la predicción en un producto de puntos:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

Por eso la predicción de Bayes es tan rápida, es la misma operación que un modelo lineal de una sola capa.

### Bayes ingenuo vs Regresión logística

Ambos son clasificadores lineales para el texto. La diferencia es en lo que modelan.

| Aspect | Naive Bayes | Logistic Regression |
|--------|------------|-------------------|
| Type | Generative (models P(X\|Y)) | Discriminative (models P(Y\|X)) |
| Training | Count frequencies | Optimize loss function |
| Small data | Better (strong prior helps) | Worse (not enough to estimate weights) |
| Large data | Worse (wrong assumption hurts) | Better (flexible boundary) |
| Features | Assumes independence | Handles correlations |
| Speed | Single pass, very fast | Iterative optimization |
| Calibration | Poor probabilities | Better probabilities |

Regla de oro: comienza con Bayes Ingenuo. Si tienes suficientes datos y planillas NB, cambia a regresión logística.

### Línea de clasificación

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

En la práctica, trabajamos en el espacio de registro para evitar el flujo inferior de puntos flotantes. en lugar de multiplicar muchas probabilidades pequeñas, sumamos sus logaritmos:

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## Construye el mismo

El código en `code/naive_bayes.py`Implementa tanto MultinomialNB como GaussianNB desde cero.

### MultinomioNB

La implementación desde cero:

1. **fit(X, y)**Para cada clase, cuenta la frecuencia de cada característica. Agregue el suavización Laplace. Compute probabilidades de registro. Guarde los antecedentes de clase (log de frecuencias de clase).

2. **predict_log_proba(X)**Para cada muestra, compute el registro P(clase) + suma del registro P(feature_i ➡ clase) para todas las clases. Esta es una matriz multiplicación: X @ log_probs.T + log_priors.

3. **predict(X)**: Regresa la clase con mayor probabilidad de registro.

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

La clave: después de la adaptación, la predicción es sólo la multiplicación de matriz más un sesgo.

### GaussianNB

Para características continuas, estimamos la media y la variación por clase por característica:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

La predicción utiliza el PDF gaussiano por característica, multiplicado a través de las características (agregado en el espacio de registro).

### Demo: Clasificación del texto

El código genera datos sintéticos de bolsas de palabras que simulan dos clases (artículos tecnológicos vs artículos deportivos). Cada clase tiene una distribución de frecuencia de palabras diferente. MultinomialNB los clasifica utilizando recuentos de palabras.

Los datos sintéticos funcionan así: creamos 200 "palabras" (columnas de características). Las palabras 0-39 tienen alta frecuencia en artículos de tecnología y baja frecuencia en deportes. Las palabras 80-119 tienen alta frecuencia en deportes y baja frecuencia en tecnología. Las palabras 40-79 son de frecuencia media en ambos. Esto crea un escenario realista donde algunas palabras son indicadores de clase fuertes y otras son ruidosos.

### Demo: características continuas

El código genera datos similares a los iris (3 clases, 4 características, grupos gaussianos). GaussianNB clasifica utilizando la media y la varianza por clase. Cada clase tiene un centro diferente (vector medio) y una dispersión diferente (varianza), imitando datos del mundo real donde las mediciones difieren sistemáticamente entre categorías.

El código también demuestra:
- **Smoothing comparison:**Entrenamiento de MultinomialNB con diferentes valores alfa para mostrar el efecto de la fuerza de suavización sobre la precisión.
- **Training size experiment:**Cómo la precisión de NB mejora a medida que los datos de entrenamiento crecen de 20 a 1600 muestras. NB alcanza una precisión decente incluso con muy pocas muestras - esta es su principal ventaja.
- **Confusion matrix:**Precisión por clase, recuerdo y puntaje de F1 para mostrar dónde NB comete errores.

### Velocidad de predicción

La predicción de Bayes ingenuo es una multiplicación de matriz. Para n muestras con d características y k clases:
- MultinomioNB: una matriz multiplicada (n x d) @ (d x k) = O(n * d * k)
- GaussianNB: n * k Evaluaciones Gaussian PDF, cada una con d características = O(n * d * k)

Ambos son lineales en todas las dimensiones. Comparar esto con KNN (que requiere el cálculo de la distancia a todos los puntos de entrenamiento) o SVM con el núcleo RBF (que requiere la evaluación del núcleo contra todos los vectores de soporte). NB es más rápido por órdenes de magnitud en el tiempo de predicción.

## Usalo

Con sklearn, ambas variantes son de una línea:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

Para la clasificación de textos con sklearn:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

El código en `naive_bayes.py`compara las implementaciones desde cero con las de sklearn en los mismos datos para verificar la corrección.

### TF-IDF con Naive Bayes

El conteo de palabras en bruto da a cada palabra el mismo peso por ocurrencia. Pero las palabras comunes como "el" y "es" aparecen con frecuencia en cada clase - no llevan información. TF-IDF (Term Frequency - Inverse Document Frequency) reduce el peso de las palabras comunes y aumenta el peso de las palabras raras y discriminatorias.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

Los valores TF-IDF son no negativos, por lo que funcionan con MultinomialNB. La combinación de TF-IDF + MultinomialNB es una de las líneas de base más fuertes para la clasificación de texto. A menudo supera modelos más complejos en conjuntos de datos con menos de 10.000 muestras de entrenamiento.

### BernoulliNB para texto corto

Para textos cortos (tweet, SMS, mensajes de chat), BernoulliNB puede superar a MultinomialNB. Los textos cortos tienen un bajo conteo de palabras, por lo que la información de frecuencia en la que MultinomialNB se basa es ruidosa.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

El `binary=True`El número de cuenta de la bandera en CountVectorizer convierte todos los números a 0/1.

### Calibración NB Probabilidades

Cuando NB dice P(spam) = 0,95, la probabilidad real podría ser de 0,7. Si necesita estimaciones de probabilidad fiables (por ejemplo, para establecer un umbral o combinar con otros modelos), utilice el CalibradoClassifierCV de sklearn:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

Esto se ajusta a una regresión logística en la parte superior de las puntuaciones en bruto de NB utilizando validación cruzada.

### Gotas comunes

1. **Negative feature values.**MultinomialNB requiere características no negativas. Si tiene valores negativos (como TF-IDF con ciertas configuraciones o características estandarizadas), use GaussianNB en su lugar, o cambie las características para ser positivas.

2. **Zero variance features.**GaussianNB se divide por varianza. Si una característica tiene una varianza cero para una clase (todos los valores son idénticos), el cálculo de probabilidad se rompe. El código agrega un pequeño término de suavizamiento (1e-9) a todas las variancias para evitar esto.

3. **Class imbalance.**Si el 99% de los correos electrónicos no son spam, el P ((no-spam) = 0,99 anterior es tan fuerte que abrumará la evidencia de probabilidad.

4. **Feature scaling.**MultinomialNB no necesita escalado (funciona en recuentos). GaussianNB tampoco necesita escalado (estima estadísticas por característica). Esta es una ventaja sobre la regresión logística y SVM, que son sensibles a las escalas de características.

## Envío

Esta lección produce:
- `outputs/skill-naive-bayes-chooser.md`-- una habilidad para tomar decisiones para elegir la variante de NB correcta
- `code/naive_bayes.py`-- MultinomialNB y GaussianNB desde cero, con comparación sklearn

### Cuando Bayes, el ingenuo, falla

NB falla cuando la suposición de independencia causa clasificaciones incorrectas (no sólo probabilidades incorrectas). Esto ocurre cuando:

1. **Strong feature interactions.**Si la clase depende de la combinación de dos características pero no de ninguna sola (patrones similares a XOR), NB la omitirá por completo.

2. **Highly correlated features with opposing evidence.**Si la característica A dice "spam" y la característica B dice "no spam", pero A y B están perfectamente correlacionados (siempre están de acuerdo en la realidad), NB verá pruebas contradictorias donde no hay ninguna.

3. **Very large training sets.**Con suficientes datos, los modelos discriminatorios como la regresión logística aprenden el verdadero límite de decisión y superan a NB. La suposición de independencia que ayudó con pequeños datos ahora retrasa el modelo.

En la práctica, estos modos de falla son raros para la clasificación de texto. Las características del texto son numerosas, individualmente débiles, y los errores de la suposición de independencia tienden a cancelarse. Para los datos tablales con pocas características fuertemente correlacionadas, considere primero la regresión logística o los modelos basados en árboles.

## Los ejercicios

1. **Smoothing experiment.**Entrenamiento MultinomialNB en datos de texto con valores alfa de 0.01, 0.1, 1.0, 10.0 y 100.0. Precisión de gráfico vs alfa. ¿Dónde alcanza el máximo rendimiento? ¿Por qué hace daño el alfa muy alto?

2. **Feature independence test.**Tomemos un conjunto de datos de texto real. Elige dos palabras que están obviamente correlacionadas ("máquina" y "aprendizaje"). Computa P " palabra 1 " clase) * P " palabra 2 " clase) y compara con P " palabra 1 Y palabra 2 " clase. ¿Qué tan equivocada es la suposición de independencia? ¿Afectará la precisión de clasificación?

3. **Bernoulli implementation.**Extenda el código con una clase BernoulliNB. Convierta bolsas de palabras a binarias (presentes/ausentes) y compara la precisión con MultinomialNB en los datos de texto. ¿Cuándo gana Bernoulli?

4. **NB vs Logistic Regression.**Entrenando a ambos en datos de texto. Comience con 100 muestras de entrenamiento y aumenta a 10.000. Precisión de trama vs tamaño del conjunto de entrenamiento para ambos. ¿En qué punto la Regresión Logística supera a Bayes Ingenuo?

5. **Spam filter.**Construir un clasificador de spam completo: tokenizar el texto de correo electrónico crudo, construir vocabulario, crear funciones de bolso de palabras, entrenar MultinomialNB, evaluar con precisión y recordar (no sólo precisión - ¿por qué?).

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Naive Bayes | "Simple probabilistic classifier" | A classifier that applies Bayes' theorem with the assumption that features are conditionally independent given the class |
| Conditional independence | "Features don't affect each other" | P(A, B \| C) = P(A \| C) * P(B \| C) -- knowing B tells you nothing new about A once you know C |
| Laplace smoothing | "Add-one smoothing" | Adding a small count to every feature to prevent zero probabilities from dominating the prediction |
| Prior | "What you believed before seeing data" | P(class) -- the probability of each class before observing any features |
| Likelihood | "How well the data fits" | P(features \| class) -- the probability of observing these features if the class is known |
| Posterior | "What you believe after seeing data" | P(class \| features) -- the updated probability of the class after observing the features |
| Generative model | "Models how data is generated" | A model that learns P(X \| Y) and P(Y), then uses Bayes' theorem to get P(Y \| X) |
| Discriminative model | "Models the decision boundary" | A model that directly learns P(Y \| X) without modeling how X is generated |
| Log probability | "Avoid underflow" | Working with log P instead of P to prevent the product of many small numbers from becoming zero in floating point |

## Leer más

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html)-- las tres variantes con detalles matemáticos
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf)-- la comparación clásica de Multinomio vs Bernoulli para el texto
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)-- mejoras en la nota de texto
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)-- prueba que NB converge más rápido que LR con menos datos
