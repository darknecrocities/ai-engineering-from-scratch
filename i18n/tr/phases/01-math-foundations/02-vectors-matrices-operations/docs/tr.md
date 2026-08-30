# vektörler, matrisler ve işlemler

> Her sinir ağı sadece ekstra adımlarla matris çarpımı.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## Öğrenme Hedefleri

- Element-bilge işlemler, matris çarpımı, transpose, determinant ve tersine ile bir Matrix sınıfı oluşturun
- Elementli çarpımı matris çarpımından ayırt edin ve her birinin ne zaman uygulanacağını açıklayın
- Tek yoğun nöral ağ katmanı uygula (`relu(W @ x + b)`) sadece sıfırdan Matrix sınıfını kullanıyor
- Yayınlama kurallarını ve sinir ağ çerçevelerinde önyargı eklemenin nasıl çalıştığını açıklayın

## Sorun

Bir sinir ağı inşa etmek istiyorsanız, kodunu okuyun ve şunu görün:

```
output = activation(weights @ input + bias)
```

- Bu ...`@`Bu da matris çarpımı.`weights`- Bir matris.`input`Eğer bu işlemlerin ne olduğunu bilmiyorsanız, bu çizgi sihirli. Eğer biliyorsanız, bu bir katmanın üç işlemde ileriye geçmesi.

Modeliniz işleyen her görüntü, piksel değerlerinin bir matrisidir. Her kata yerleştirme bir vektördür. Her sinir ağının her katmanı bir matris dönüşümüdür. Matris işlemlerinde akıcı olmadan AI sistemlerini inşa edemezsiniz. Değişkenleri anlamadan kod yazabileceğiniz gibi.

Bu ders, bu akıcılığı sıfırdan inşa eder.

## Anlaşım

### vektörler: sayılar sırasıyla listelenir

Bir vektör, yön ve büyüklüğü olan sayılar listesidir. AI'de vektörler veri noktalarını, özelliklerini veya parametreleri temsil eder.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

2 boyutlu bir vektör .`[3, 4]`Düzlemde koordinatlara (3, 4) işaret eder. Uzunluğu (geniüt) 5 ( 3-4-5 üçgen)

### Matrisler: Sayılar ağı

Bir matris 2 boyutlu bir şebeke. Satırlar ve sütunlar.

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

Nöral ağlarda ağırlık matrisleri giriş vektörlerini çıkış vektörlerine dönüştürür. 784 giriş ve 128 çıkış olan bir katman 128x784 ağırlık matrisini kullanır.

### Neden şekiller önemlidir?

Matrix çarpımı sıkı bir kuralı vardır:`(m x n) @ (n x p) = (m x p)`İç boyutlar aynı olmalı.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

PyTorch'de bir şekil eşleşme hatası varsa, bunun nedeni budur.

### Harita operasyonları

| Operation | What it does | Neural network use |
|-----------|-------------|-------------------|
| Addition | Element-wise combine | Adding bias to output |
| Scalar multiply | Scale every element | Learning rate * gradients |
| Matrix multiply | Transform vectors | Layer forward pass |
| Transpose | Flip rows and columns | Backpropagation |
| Determinant | Single number summary | Checking invertibility |
| Inverse | Undo a transformation | Solving linear systems |
| Identity | Do-nothing matrix | Initialization, residual connections |

### Element-wise vs matris çarpımı

Bu fark yeni başlayanları sürekli şaşırtıyor.

Element açısından: eşleşen pozisyonları çarpın.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

Matrix çarpımı: Satır ve sütunların nokta ürünleri. İç boyutlar eşleşmelidir.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Farklı operasyonlar, farklı sonuçlar, farklı kurallar.

### Yayınlama

Bir çıkış matrisine bir önyargı vektörü eklediğinizde şekiller eşleşmez.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Her modern çerçeve bunu otomatik olarak yapar. Şekiller yanlış görünse de kod çalışsa da, anlayış karışıklığı önler.

```figure
vector-projection
```

## Yapın

### Adım 1: Vektör sınıfı

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

### Adım 2: Temel işlemleri olan matris sınıfı

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

### Üçüncü adım: Çalışın

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

### Dördüncü adım: Nöral ağlara bağlan

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

Bu tek yoğun bir katman:`output = relu(W @ x + b)`Her sinir ağının yoğun katmanları tam olarak bunu yapar.

## Kullan

NumPy yukarıdaki her şeyi daha az çizgi ve büyüklük sıralarında daha hızlı yapar.

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

- Evet .`@`Python çağrılarında operatör `__matmul__`NumPy, C ve Fortran'da yazılmış optimize edilmiş BLAS rutinleri ile uyguluyor. Aynı matematik, 100 kat daha hızlı.

NumPy'de yayınlama:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy, her iki satır boyunca 1D tersi otomatik olarak yayınlar.

## Gönder

Bu ders, geometrik sezgilerle matris işlemlerini öğretmek için bir ipucu üretir.`outputs/prompt-matrix-operations.md`- Evet .

Burada inşa edilen Matrix sınıfı, 3. aşamada oluşturduğumuz mini sinir ağı çerçevesinin temelidir. 10. ders.

## Egzersizler

1. **Verify the inverse.**Çoklu `A @ A.inverse_2x2()`2x2 matrisleri ile deneyin. belirleyici sıfır olduğunda ne olur?

2. **Implement 3x3 inverse.**Matrix sınıfını 3x3 matrisler için ters hesaplamak için adjugaat yöntemi kullanarak genişlet. NumPy'nin `np.linalg.inv`- Evet .

3. **Build a two-layer network.**Sadece Matrix sınıfınızı kullanarak (NumPy yok), iki katlı bir nöron ağı oluşturun: giriş (3) -> gizli (4) -> çıkış (2). Kasıtlı ağırlıkları başlatın, ileri geçiş çalıştırın ve tüm şekillerin doğru olduğunu doğrulayın.

## Anahtar Terimler

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

## Daha Fazla Okumak

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)- Burada ele alınan her operasyon için görsel sezgisellik
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)- NumPy'nin takip ettiği kesin kurallar
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf)- ML spesifik çizgi cebir için kısa bir referans
