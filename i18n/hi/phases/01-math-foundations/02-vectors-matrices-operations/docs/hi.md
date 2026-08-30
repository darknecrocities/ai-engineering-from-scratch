# वेक्टर, मैट्रिक्स और ऑपरेशन

> प्रत्येक तंत्रिका नेटवर्क अतिरिक्त चरणों के साथ मैट्रिक्स गुणन है।

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## सीखने के लक्ष्य

- तत्व-बुद्धिमान संचालन, मैट्रिक्स गुणन, ट्रांसपोस, निर्धारक और विपरीत के साथ मैट्रिक्स वर्ग का निर्माण करें
- तत्व-बुद्धिमान गुणा को मैट्रिक्स गुणा से अलग करें और समझाएं कि प्रत्येक कब लागू होता है
- एक एकल घने तंत्रिका नेटवर्क परत को लागू करें (`relu(W @ x + b)`) केवल स्क्रैच से मैट्रिक्स वर्ग का उपयोग कर
- प्रसारण नियमों और तंत्रिका नेटवर्क ढांचे में पूर्वाग्रह जोड़ने का काम कैसे करता है, इसकी व्याख्या करें

## समस्या

आप एक तंत्रिका नेटवर्क बनाना चाहते हैं. आप कोड पढ़ते हैं और यह देखते हैंः

```
output = activation(weights @ input + bias)
```

यह `@`मैट्रिक्स गुणा है।`weights`एक मैट्रिक्स है।`input`यदि आप नहीं जानते कि वे ऑपरेशन क्या करते हैं, तो यह रेखा जादू है. अगर आप जानते हैं, तो यह तीन ऑपरेशन में एक परत के पूरे आगे पारित है.

आपके मॉडल द्वारा संसाधित की जाने वाली प्रत्येक छवि पिक्सेल मानों की एक मैट्रिक्स है. प्रत्येक शब्द एम्बेडिंग एक वेक्टर है. प्रत्येक तंत्रिका नेटवर्क की प्रत्येक परत मैट्रिक्स परिवर्तन है. आप मैट्रिक्स संचालन में धाराप्रवाह होने के बिना एआई सिस्टम नहीं बना सकते हैं, जिस तरह आप चर को समझने के बिना कोड नहीं लिख सकते हैं।

यह सबक उस धाराप्रवाहता को शून्य से बनाती है।

## अवधारणा

### वेक्टरः संख्याओं की क्रमबद्ध सूची

वेक्टर एक दिशा और परिमाण के साथ संख्याओं की एक सूची है। एआई में, वेक्टर डेटा बिंदुओं, विशेषताओं या मापदंडों का प्रतिनिधित्व करते हैं।

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

एक 2D वेक्टर `[3, 4]`एक विमान पर निर्देशांक (3, 4) पर इंगित करता है। इसकी लंबाई (महानता) 5 (तीन-चार-पांच त्रिकोण) है।

### मैट्रिक्सः संख्याओं के ग्रिड

एक मैट्रिक्स एक 2D ग्रिड है। पंक्तियों और स्तंभों। एक m x n मैट्रिक्स में m पंक्तियों और n स्तंभ हैं।

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

न्यूरल नेटवर्क में, वजन मैट्रिक्स इनपुट वेक्टरों को आउटपुट वेक्टरों में बदल देते हैं। 784 इनपुट और 128 आउटपुट वाली परत में 128x784 वजन मैट्रिक्स का उपयोग किया जाता है।

### आकार क्यों मायने रखते हैं

मैट्रिक्स गुणन का एक सख्त नियम हैः`(m x n) @ (n x p) = (m x p)`आंतरिक आयामों को मेल खाना चाहिए.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

यदि आप PyTorch में एक आकार असंगतता त्रुटि प्राप्त करते हैं, तो यह क्यों है.

### परिचालन मानचित्र

| Operation | What it does | Neural network use |
|-----------|-------------|-------------------|
| Addition | Element-wise combine | Adding bias to output |
| Scalar multiply | Scale every element | Learning rate * gradients |
| Matrix multiply | Transform vectors | Layer forward pass |
| Transpose | Flip rows and columns | Backpropagation |
| Determinant | Single number summary | Checking invertibility |
| Inverse | Undo a transformation | Solving linear systems |
| Identity | Do-nothing matrix | Initialization, residual connections |

### तत्व-बुद्धि बनाम मैट्रिक्स गुणा

यह अंतर शुरुआती लोगों को लगातार ठोकर खाता है।

तत्व-बुद्धिमानः मिलान वाली स्थिति को गुणा करें. दोनों मैट्रिक्स एक ही आकार के होने चाहिए।

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

मैट्रिक्स गुणाः पंक्तियों और स्तंभों के अंक उत्पाद। आंतरिक आयामों को मेल खाना चाहिए।

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

अलग-अलग ऑपरेशन, अलग-अलग परिणाम, अलग-अलग नियम।

### प्रसारण

जब आप आउटपुट मैट्रिक्स में एक पूर्वाग्रह वेक्टर जोड़ते हैं, तो आकार मेल नहीं खाते हैं। प्रसारण छोटे सरणी को फिट करने के लिए बढ़ाता है।

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

प्रत्येक आधुनिक ढांचे यह स्वचालित रूप से करता है। इसे समझना भ्रम को रोकता है जब आकृति गलत लगती है लेकिन कोड चल रहा है।

```figure
vector-projection
```

## इसे बनाओ

### चरण 1: वेक्टर वर्ग

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

### चरण 2: कोर ऑपरेशन के साथ मैट्रिक्स वर्ग

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

### चरण 3: इसे काम करते हुए देखें

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

### चरण 4: तंत्रिका नेटवर्क से कनेक्ट करें

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

यह एक ही घने परत हैः`output = relu(W @ x + b)`प्रत्येक तंत्रिका नेटवर्क में प्रत्येक घने परत ठीक यही करती है।

## इसका प्रयोग करें

NumPy ऊपर की सभी चीजों को कम लाइनों और बड़े पैमाने के आदेशों में तेजी से करता है।

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

`@`पायथन कॉल में ऑपरेटर `__matmul__`NumPy इसे अनुकूलित BLAS दिनचर्या के साथ लागू करता है C और Fortran में लिखा है. एक ही गणित, 100 गुना तेजी से.

NumPy में प्रसारणः

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy स्वचालित रूप से दोनों पंक्तियों पर 1D पूर्वाग्रह प्रसारित करता है। यह हर तंत्रिका नेटवर्क फ्रेमवर्क में पूर्वाग्रह जोड़ने का काम करता है।

## इसे भेजें

इस पाठ में ज्यामितीय अंतर्ज्ञान के माध्यम से मैट्रिक्स संचालन को सिखाने के लिए एक संकेत उत्पन्न होता है।`outputs/prompt-matrix-operations.md`. .

यहाँ निर्मित मैट्रिक्स वर्ग मिनी न्यूरल नेटवर्क फ्रेमवर्क की नींव है जिसे हम चरण 3, पाठ 10 में बनाते हैं।

## व्यायाम

1. **Verify the inverse.**गुणा करें `A @ A.inverse_2x2()`और पुष्टि करें कि आप पहचान मैट्रिक्स प्राप्त करते हैं. यह तीन अलग 2x2 मैट्रिक्स के साथ कोशिश करें. क्या होता है जब निर्धारक शून्य है?

2. **Implement 3x3 inverse.**3x3 मैट्रिक्स के लिए विपरीत गणना करने के लिए मैट्रिक्स वर्ग का विस्तार करें।`np.linalg.inv`. .

3. **Build a two-layer network.**केवल अपने मैट्रिक्स वर्ग (नहीं NumPy) का उपयोग करके, दो-परत तंत्रिका नेटवर्क बनाएंः इनपुट (3) -> छिपा हुआ (4) -> आउटपुट (2). यादृच्छिक वजन शुरू करें, आगे की पास चलाएं, और सभी आकारों की पुष्टि करें कि वे सही हैं।

## प्रमुख शर्तें

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

## आगे पढ़ना

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)- यहाँ कवर किए गए प्रत्येक ऑपरेशन के लिए दृश्य अंतर्ज्ञान
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)- NumPy के अनुसार सही नियम
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf)- एमएल विशिष्ट रैखिक बीजगणित के लिए संक्षिप्त संदर्भ
