# Vectores, matrices y operaciones

> Cada red neuronal es sólo una multiplicación de matriz con pasos adicionales.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## Objetivos de aprendizaje

- Construir una clase de Matrix con operaciones de elementos, multiplicación de matriz, transposición, determinante y inversa
- Distinguir la multiplicación por elemento de la multiplicación por matriz y explicar cuándo cada uno se aplica
- Implementar una sola capa de red neuronal densa (`relu(W @ x + b)`) utilizando únicamente la clase Matrix desde cero
- Explicar las reglas de radiodifusión y cómo funciona la adición de sesgos en los marcos de redes neuronales

## El problema

Si quieres construir una red neuronal, lee el código y ve esto:

```
output = activation(weights @ input + bias)
```

Eso es .`@`es la multiplicación de matriz.`weights`Es una matriz.`input`Si no sabes lo que hacen esas operaciones, esta línea es mágica. si lo sabes, es todo el paso hacia adelante de una capa en tres operaciones.

Cada imagen que procesas es una matriz de valores de píxeles. Cada palabra que se incorpora es un vector. Cada capa de cada red neuronal es una transformación de matriz. No puedes construir sistemas de IA sin ser fluido en las operaciones de matriz de la misma manera que no puedes escribir código sin entender variables.

Esta lección construye esa fluidez desde cero.

## El concepto

### Vectores: listas ordenadas de números

Un vector es una lista de números con una dirección y magnitud. En IA, los vectores representan puntos de datos, características o parámetros.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

Un vector 2D `[3, 4]`El punto de referencia es el punto de referencia de la longitud de un triángulo de 3 a 4 y el punto de referencia de la longitud de un triángulo de 3 a 5 (el triángulo de 3 a 4 a 5).

### Matrices: redes de números

Una matriz es una cuadrícula 2D. filas y columnas.

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

En las redes neuronales, las matrices de peso transforman los vectores de entrada en vectores de salida.

### ¿Por qué importa la forma?

La multiplicación de matrices tiene una regla estricta:`(m x n) @ (n x p) = (m x p)`Las dimensiones internas deben coincidir.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

Si obtienes un error de desajuste de forma en PyTorch, es por esto.

### El mapa de operaciones

| Operation | What it does | Neural network use |
|-----------|-------------|-------------------|
| Addition | Element-wise combine | Adding bias to output |
| Scalar multiply | Scale every element | Learning rate * gradients |
| Matrix multiply | Transform vectors | Layer forward pass |
| Transpose | Flip rows and columns | Backpropagation |
| Determinant | Single number summary | Checking invertibility |
| Inverse | Undo a transformation | Solving linear systems |
| Identity | Do-nothing matrix | Initialization, residual connections |

### El elemento-sabio vs. multiplicación de matriz

Esta distinción atraviesa a los principiantes constantemente.

El elemento-sabio: multiplicar las posiciones coinciden. Ambas matrices deben tener la misma forma.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

Multiplicación de matriz: productos de puntos de filas y columnas.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Diferentes operaciones, diferentes resultados, diferentes reglas.

### La radiodifusión

Cuando se añade un vector de sesgo a una matriz de salidas, las formas no coinciden.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Cada marco moderno hace esto automáticamente.

```figure
vector-projection
```

## Construye el mismo

### Paso 1: Clase de vectores

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Paso 2: Clase de matriz con operaciones centrales

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Paso 3: Vean cómo funciona

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Paso 4: Conectar a las redes neuronales

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

Esta es una sola capa densa:`output = relu(W @ x + b)`Cada capa densa en cada red neuronal hace exactamente esto.

## Usalo

NumPy hace todo lo anterior en menos líneas y órdenes de magnitud más rápido.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

El `@`operador en llamadas Python `__matmul__`NumPy lo implementa con rutinas BLAS optimizadas escritas en C y Fortran.

La radiodifusión en NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy transmite automáticamente el sesgo 1D a través de ambas filas. Así es como la adición de sesgos funciona en cada marco de red neuronal.

## Envío

Esta lección produce un prompt para enseñar operaciones de matriz a través de la intuición geométrica.`outputs/prompt-matrix-operations.md`¿ Qué ?

La clase Matrix construida aquí es la base para el marco de la red neuronal mini que construimos en la fase 3, lección 10.

## Los ejercicios

1. **Verify the inverse.**Multiplicado `A @ A.inverse_2x2()`y confirmar que obtiene la matriz de identidad. Prueba con tres diferentes matrices 2x2. ¿Qué sucede cuando el determinante es cero?

2. **Implement 3x3 inverse.**Extenda la clase de Matrix para calcular inversos de matrices 3x3 usando el método de adjugación.`np.linalg.inv`¿ Qué ?

3. **Build a two-layer network.**Utilizando sólo su clase Matrix (sin NumPy), cree una red neuronal de dos capas: entrada (3) -> oculta (4) -> salida (2). Inicializa pesos aleatorios, ejecuta un pase hacia adelante y verifique que todas las formas son correctas.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Vector | "An arrow" | An ordered list of numbers. In AI: a point in high-dimensional space. |
| Matrix | "A table of numbers" | A linear transformation. It maps vectors from one space to another. |
| Matrix multiply | "Just multiply the numbers" | Dot products between every row of the first matrix and every column of the second. Order matters. |
| Transpose | "Flip it" | Swap rows and columns. Turns an m x n matrix into n x m. Critical in backpropagation. |
| Determinant | "Some number from the matrix" | Measures how much the matrix scales area (2D) or volume (3D). Zero means the transformation crushes a dimension. |
| Inverse | "Undo the matrix" | The matrix that reverses the transformation. Only exists when the determinant is not zero. |
| Identity matrix | "The boring matrix" | The matrix equivalent of multiplying by 1. Used in residual connections (ResNets). |
| Broadcasting | "Magic shape fixing" | Stretching a smaller array to match a larger one by repeating along missing dimensions. |
| Element-wise | "Regular multiplication" | Multiply matching positions. Both arrays must have the same shape (or be broadcastable). |

## Leer más

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)- Intuición visual para cada operación que se cubre aquí
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)- las reglas exactas que sigue NumPy
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf)- referencia concisa para el álgebra lineal específico de ML
