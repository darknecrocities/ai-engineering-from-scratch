# Regla de cadena y diferenciación automática

> La regla de la cadena es el motor detrás de cada red neuronal que aprende.

**Type:** Build
**Language:**Python
**Prerequisites:** Phase 1, Lesson 04 (Derivatives & Gradients)
**Time:** ~90 minutes

## Objetivos de aprendizaje

- Construir un motor de autogrado mínimo (clase de valor) que registre las operaciones y computa los gradientes a través del autodiff en modo inverso
- Implementar pasos hacia adelante y hacia atrás a través de un gráfico de cálculo utilizando ordenamiento topológico
- Construir y entrenar un perceptron de varias capas en XOR utilizando sólo el motor de autograd desde cero
- Verificar la corrección de la auto-diff con la verificación de gradientes contra diferencias finitas numéricas

## El problema

Se pueden calcular derivadas de funciones simples. Pero una red neuronal no es una función simple. Son cientos de funciones compuestas juntas: multiplicar matriz, agregar sesgo, aplicar activación, multiplicar matriz de nuevo, softmax, pérdida de entropía cruzada. La salida es una función de una función de una función.

Para entrenar la red, se necesita el gradiente de la pérdida con respecto a cada peso.

La regla de la cadena te da las matemáticas. La diferenciación automática te da el algoritmo. Juntos te permiten calcular gradientes exactos a través de composiciones arbitrarias de funciones en tiempo proporcionales a un solo pase hacia adelante.

Así es como PyTorch, TensorFlow y JAX funcionan.

## El concepto

### La regla de la cadena

Si ...`y = f(g(x))`, el derivado de `y`en lo que respecta a `x`es:

```
dy/dx = dy/dg * dg/dx = f'(g(x)) * g'(x)
```

Multiplicar las derivadas a lo largo de la cadena. Cada enlace contribuye a su derivada local.

Ejemplo: `y = sin(x^2)`

```
g(x) = x^2       g'(x) = 2x
f(g) = sin(g)     f'(g) = cos(g)

dy/dx = cos(x^2) * 2x
```

Para composiciones más profundas, la cadena se extiende:

```
y = f(g(h(x)))

dy/dx = f'(g(h(x))) * g'(h(x)) * h'(x)
```

Cada capa de una red neuronal es un eslabón de esta cadena.

### Gráficos computacionales

Un gráfico computacional hace que la regla de la cadena sea visual. Cada operación se convierte en un nodo. Los datos fluyen hacia adelante a través del gráfico.

**Forward pass (compute values):**

```mermaid
graph TD
    x1["x1 = 2"] --> mul["* (multiply)"]
    x2["x2 = 3"] --> mul
    mul -->|"a = 6"| add["+ (add)"]
    b["b = 1"] --> add
    add -->|"c = 7"| relu["relu"]
    relu -->|"y = 7"| y["output y"]
```

**Backward pass (compute gradients):**

```mermaid
graph TD
    dy["dy/dy = 1"] -->|"relu'(c)=1 since c>0"| dc["dy/dc = 1"]
    dc -->|"dc/da = 1"| da["dy/da = 1"]
    dc -->|"dc/db = 1"| db["dy/db = 1"]
    da -->|"da/dx1 = x2 = 3"| dx1["dy/dx1 = 3"]
    da -->|"da/dx2 = x1 = 2"| dx2["dy/dx2 = 2"]
```

El paso hacia atrás aplica la regla de la cadena en cada nodo, propagando gradientes de salida a entrada.

### Modo de marcha adelante vs Modo inverso

Hay dos maneras de aplicar la regla de la cadena a través de un gráfico.

**Forward mode**El sistema de cálculo de las derivadas de las operaciones de cálculo de las operaciones de cálculo de las operaciones de cálculo de las operaciones de cálculo de las operaciones de cálculo de las operaciones de cálculo de las operaciones de cálculo de las operaciones de cálculo de las operaciones de cálculo de las operaciones de cálculo de las operaciones de cálculo de las operaciones de cálculo de las operaciones de cálculo de las operaciones de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de las operaciones de cálculo de las operaciones de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de cálculo de las operaciones de cálculo de cálculo de cálculo de las operaciones de cálculo de cálculo de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de las operaciones de cálculo de cálculo de las cuentas de cálculo de cálculo de las cuentas de cálculo de las cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cuentas de cu`dx/dx = 1`Es bueno cuando tienes pocas entradas y muchas salidas.

```
Forward mode: seed dx/dx = 1, propagate forward

  x = 2       (dx/dx = 1)
  a = x^2     (da/dx = 2x = 4)
  y = sin(a)  (dy/dx = cos(a) * da/dx = cos(4) * 4 = -2.615)
```

**Reverse mode**comienza en la salida y tira los gradientes hacia atrás.`dy/dy = 1`Es bueno cuando tienes muchas entradas y pocas salidas.

```
Reverse mode: seed dy/dy = 1, propagate backward

  y = sin(a)  (dy/dy = 1)
  a = x^2     (dy/da = cos(a) = cos(4) = -0.654)
  x = 2       (dy/dx = dy/da * da/dx = -0.654 * 4 = -2.615)
```

Las redes neuronales tienen millones de entradas (pesos) y una salida (pérdida). El modo inverso calcula todos los gradientes en un paso hacia atrás.

| Mode | Seed | Direction | Best when |
|------|------|-----------|-----------|
| Forward | `dx_i/dx_i = 1` | Input to output | Few inputs, many outputs |
| Reverse | `dy/dy = 1` | Output to input | Many inputs, few outputs (neural nets) |

### Números dobles para el modo de avanzada

El modo de avanzada puede implementarse con elegancia con números dobles.`a + b*epsilon`donde`epsilon^2 = 0`¿ Qué ?

```
Dual number: (value, derivative)

(2, 1) means: value is 2, derivative w.r.t. x is 1

Arithmetic rules:
  (a, a') + (b, b') = (a+b, a'+b')
  (a, a') * (b, b') = (a*b, a'*b + a*b')
  sin(a, a')         = (sin(a), cos(a)*a')
```

Seem la variable de entrada con derivada 1. La derivada se propaga automáticamente a través de cada operación.

### Construir un motor de Autograd

Un motor de autograd necesita tres cosas:

1. **Value wrapping.**Envuelve cada número en un objeto que almacene su valor y gradiente.
2. **Graph recording.**Cada operación registra sus entradas y la función de gradiente local.
3. **Backward pass.**La clasificación topológica del gráfico, luego caminar en sentido contrario, aplicando la regla de cadena en cada nodo.

Esto es exactamente lo que PyTorch es.`autograd`¿Qué es eso?`torch.Tensor`clase envuelve valores, registra operaciones cuando `requires_grad=True`, y calcula los gradientes cuando llamas`.backward()`¿ Qué ?

### Cómo funciona la PyTorch Autograd bajo el capó

Cuando escribe código PyTorch:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # 7.0 = 2*x + 3 = 2*2 + 3
```

PyTorch internamente:

1. Crea una`Tensor`nodo para `x`con`requires_grad=True`
2. Cada operación (`**`¿ Qué ?`*`¿ Qué ?`+`) crea un nuevo nodo y registra la función retrocediente
3. `y.backward()`desencadena el modo inverso de auto-difusión a través del gráfico registrado
4. Cada nodo es `grad_fn`computa gradientes locales y los pasa a los nodos padres
5. Los gradientes se acumulan en `.grad`Los atributos se añaden (no se sustituyen)

El gráfico es dinámico (definido por ejecución). Un nuevo gráfico se construye en cada paso hacia adelante. Por eso PyTorch admite el flujo de control (si / o, bucles) dentro de los modelos.

```figure
chain-rule
```

## Construye el mismo

### Paso 1: La clase de valor

```python
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

Cada uno .`Value`almacena sus datos numéricos, su gradiente (initialemente cero), una función retrógrada y señala los nodos infantiles que lo produjeron.

### Paso 2: Operaciones aritméticas con seguimiento de gradientes

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
```

Cada operación crea un cierre que sabe cómo calcular los gradientes locales y multiplicarse por el gradiente aguas arriba (`out.grad`¿ Qué es ?`+=`maneja el caso en que se utilice un valor en múltiples operaciones.

### Paso 3: El pase hacia atrás

```python
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
```

El orden topológico asegura que el gradiente de cada nodo se calcula completamente antes de que se propague a sus hijos.

### Paso 4: Más operaciones para un motor completo

La clase de valor básica maneja la adición, multiplicación y relu. Un motor autograd real necesita más. Estas son las operaciones que necesita para construir redes neuronales:

```python
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    def exp(self):
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        import math
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
```

**Why each operation matters:**

| Operation | Backward rule | Used in |
|-----------|--------------|---------|
| `__sub__` | Reuses add + neg | Loss computation (pred - target) |
| `__pow__` | n * x^(n-1) | Polynomial activations, MSE (error^2) |
| `__truediv__` | Reuses mul + pow(-1) | Normalization, learning rate scaling |
| `exp` | exp(x) * upstream | Softmax, log-likelihood |
| `log` | (1/x) * upstream | Cross-entropy loss, log probabilities |
| `tanh` | (1 - tanh^2) * upstream | Classic activation function |

La parte inteligente:`__sub__`y `__truediv__`Se pueden definir en términos de operaciones existentes. Obtienen gradientes correctos de forma gratuita porque la regla de cadena se compone a través de las operaciones subyacentes de adición/mul/pow.

### Paso 5: Mini MLP desde cero

Con una clase completa de valores, puedes construir una red neuronal sin PyTorch, sin NumPy, sólo valores y la regla de cadena.

```python
import random

class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

¿ Qué es esto ?`Neuron`computaciones`tanh(w1*x1 + w2*x2 + ... + b)`- ¿ Qué ?`Layer`Es una lista de neuronas.`MLP`Cada peso es un`Value`, así que llamando`loss.backward()`propaga los gradientes a todos los parámetros.

**Training on XOR:**

```python
random.seed(42)
model = MLP([2, 4, 1])  # 2 inputs, 4 hidden neurons, 1 output

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]  # XOR pattern (using -1/1 for tanh)

for step in range(100):
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 20 == 0:
        print(f"step {step:3d}  loss = {loss.data:.4f}")

print("\nPredictions after training:")
for x, y in zip(xs, ys):
    print(f"  input={x}  target={y:2d}  pred={model(x).data:6.3f}")
```

Esto es microgrado. Un ciclo completo de entrenamiento de red neuronal en Python puro con diferenciación automática.

### Paso 6: Verificación gradual

¿Cómo saber si su auto-difusión es correcta? Compáralo con derivadas numéricas. Esto es comprobar el gradiente.

```python
def gradient_check(build_expr, x_val, h=1e-7):
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff
```

Prueba con una expresión compleja:

```python
def expr(x):
    return (x ** 3 + x * 2 + 1).tanh()

ad, num, diff = gradient_check(expr, 0.5)
print(f"Autodiff:  {ad:.8f}")
print(f"Numerical: {num:.8f}")
print(f"Difference: {diff:.2e}")
# Difference should be < 1e-5
```

La verificación de gradientes es esencial cuando se implementan nuevas operaciones. Si su pase retrocediente tiene un error, la verificación numérica lo detecta.

**When to use gradient checking:**

| Situation | Do gradient check? |
|-----------|-------------------|
| Adding a new operation to your autograd | Yes, always |
| Debugging a training loop that won't converge | Yes, check gradients first |
| Production training | No, too slow (2x forward passes per parameter) |
| Unit tests for autograd code | Yes, automate it |

### Paso 7: Verificar con el cálculo manual

```python
x1 = Value(2.0)
x2 = Value(3.0)
a = x1 * x2          # a = 6.0
b = a + Value(1.0)    # b = 7.0
y = b.relu()          # y = 7.0

y.backward()

print(f"y = {y.data}")          # 7.0
print(f"dy/dx1 = {x1.grad}")   # 3.0 (= x2)
print(f"dy/dx2 = {x2.grad}")   # 2.0 (= x1)
```

Verificación manual: `y = relu(x1*x2 + 1)`Desde entonces .`x1*x2 + 1 = 7 > 0`, Relu es identidad.
`dy/dx1 = x2 = 3`- ¿ Qué ?`dy/dx2 = x1 = 2`- El motor coincide.

## Usalo

### Verificar con PyTorch

```python
import torch

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
a = x1 * x2
b = a + 1.0
y = torch.relu(b)
y.backward()

print(f"PyTorch dy/dx1 = {x1.grad.item()}")  # 3.0
print(f"PyTorch dy/dx2 = {x2.grad.item()}")  # 2.0
```

Su motor calcula el mismo resultado que PyTorch porque las matemáticas son las mismas: auto-difusión en modo inverso a través de la regla de cadena.

### Una expresión más compleja

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = (a * b + c).relu()  # relu(2*(-3) + 10) = relu(4) = 4

f.backward()
print(f"df/da = {a.grad}")  # -3.0 (= b)
print(f"df/db = {b.grad}")  #  2.0 (= a)
print(f"df/dc = {c.grad}")  #  1.0
```

## Envío

Esta lección produce:
- `outputs/skill-autodiff.md`-- una habilidad para construir y deshacer sistemas de autograd
- `code/autodiff.py`-- un motor de autograd mínimo que se puede extender

La clase de valor construida aquí es la base para el ciclo de entrenamiento de la red neuronal en la Fase 3.

## Los ejercicios

1. Añadir`__pow__`a la clase de valor para que pueda calcular`x ** n`Verifique eso .`d/dx(x^3)`En el`x=2`igual `12.0`¿ Qué ?

2. Añadir`tanh`como una función de activación.`tanh'(0) = 1`y `tanh'(2) = 0.0707`(aproximadamente).

3. Construir un gráfico de cálculo para una sola neurona: `y = relu(w1*x1 + w2*x2 + b)`Computa los cinco gradientes y verifica contra PyTorch.

4. Implementar la autoproducción en modo avanzado utilizando números dobles.`Dual`clase y comprobar que da las mismas derivadas que su motor de modo inverso.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Chain rule | "Multiply the derivatives" | The derivative of composed functions equals the product of each function's local derivative, evaluated at the right point |
| Computational graph | "The network diagram" | A directed acyclic graph where nodes are operations and edges carry values (forward) or gradients (backward) |
| Forward mode | "Push derivatives forward" | Autodiff that propagates derivatives from inputs to outputs. One pass per input variable. |
| Reverse mode | "Backpropagation" | Autodiff that propagates gradients from outputs to inputs. One pass per output variable. |
| Autograd | "Automatic gradients" | A system that records operations on values, builds a graph, and computes exact gradients via the chain rule |
| Dual numbers | "Value plus derivative" | Numbers of the form a + b*epsilon (epsilon^2 = 0) that carry derivative information through arithmetic |
| Topological sort | "Dependency order" | Ordering graph nodes so every node comes after all its dependencies. Required for correct gradient propagation. |
| Gradient accumulation | "Add, don't replace" | When a value feeds into multiple operations, its gradient is the sum of all incoming gradient contributions |
| Dynamic graph | "Define by run" | A computation graph rebuilt on every forward pass, allowing Python control flow inside models (PyTorch style) |
| Gradient checking | "Numerical verification" | Comparing autodiff gradients against numerical finite-difference gradients to verify correctness. Essential for debugging. |
| MLP | "Multi-layer perceptron" | A neural network with one or more hidden layers of neurons. Each neuron computes a weighted sum plus bias, then applies an activation function. |
| Neuron | "Weighted sum + activation" | The basic unit: output = activation(w1*x1 + w2*x2 + ... + b). The weights and bias are learnable parameters. |

## Leer más

- [3Blue1Brown: Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8)-- explicación visual de la regla de la cadena en las redes neuronales
- [PyTorch Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)-- cómo funciona el sistema real
- [Baydin et al., Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)-- referencia general
