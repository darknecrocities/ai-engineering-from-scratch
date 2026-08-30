# المتجهات والمعادلات والعمليات

> كل شبكة عصبية هي مجرد مضاعفة المصفوفة مع خطوات إضافية.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## أهداف التعلم

- بناء فئة المصفوفة مع العمليات الحكيمة العنصر، مضاعفة المصفوفة، نقل، تحديد، والعكس
- تمييز مضاعفة العناصر من مضاعفة المصفوفة وتوضيح متى تنطبق كل واحدة
- تنفيذ طبقة واحدة من شبكة عصبية كثيفة (`relu(W @ x + b)`) باستخدام فئة ماتريكس فقط من الصفر
- شرح قواعد البث وكيف يعمل إضافة التحيزات في إطار شبكات الأعصاب

## المشكلة

إذا أردت بناء شبكة عصبية، فاقرأ الرمز وشاهد هذا:

```
output = activation(weights @ input + bias)
```

هذا`@`هو مضاعفة المصفوفة.`weights`هي المصفوفة.`input`إذا كنت لا تعرف ما تفعل هذه العمليات، هذا الخط هو السحر. إذا كنت تعرف، انها كامل المضي قدما من طبقة في ثلاث عمليات.

كل صورة يعالجها نموذجك هي ماتريكس من قيم البيكسل. كل كلمة تضمنها متجه. كل طبقة من كل شبكة عصبية هي تحول ماتريكس. لا يمكنك بناء أنظمة الذكاء الاصطناعي دون أن تكون متسلية في عمليات المصفوفة بنفس الطريقة التي لا يمكنك كتابة الشفرة دون فهم المتغيرات.

هذا الدروس يبني هذه السهولة من الصفر.

## المفهوم

### المتجهات: قوائم مرتبة للأرقام

المتجه هو قائمة بالأرقام ذات الاتجاه والحجم. في الذكاء الاصطناعي، يمثل المتجهون نقاط البيانات أو الخصائص أو المعلمات.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

متجه ثنائي الأبعاد`[3, 4]`تشير إلى إحداثيات (3, 4) على مستوى مسطح. طولها (الكبيرة) هو 5 (الثلاثي 3-4-5).

### المصفوفات: شبكات الأرقام

المصفوفة هي شبكة ثنائية الأبعاد. الصفوف والعمدة. المصفوفة m x n لديها m الصفوف و n العمدة.

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

في الشبكات العصبية، تقوم المصفوفات الوزنية بتحويل متجهات المدخلات إلى متجهات الخروج. تستخدم طبقة ذات 784 مدخلًا و 128 خروجاً ماتريساً وزناً 128 × 784.

### لماذا الأشكال مهمة

مضاعفة المصفوفة لديها قاعدة صارمة:`(m x n) @ (n x p) = (m x p)`الأبعاد الداخلية يجب أن تتطابق

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

إذا حصلت على خطأ عدم مطابقة الشكل في PyTorch، هذا هو السبب.

### خريطة العمليات

| Operation | What it does | Neural network use |
|-----------|-------------|-------------------|
| Addition | Element-wise combine | Adding bias to output |
| Scalar multiply | Scale every element | Learning rate * gradients |
| Matrix multiply | Transform vectors | Layer forward pass |
| Transpose | Flip rows and columns | Backpropagation |
| Determinant | Single number summary | Checking invertibility |
| Inverse | Undo a transformation | Solving linear systems |
| Identity | Do-nothing matrix | Initialization, residual connections |

### النسب العنصرية مقابل مضاعفة المصفوفة

هذا التمييز يثير الابتدائيين باستمرار

في العنصر: ضرب المواقع المتطابقة يجب أن تكون كلا المصفوفتين ذات الشكل

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

مضاعفة المصفوفة: منتجات النقاط من الصفوف والعمود. يجب أن تتطابق الأبعاد الداخلية.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

عمليات مختلفة، نتائج مختلفة، قواعد مختلفة.

### الإذاعة

عندما تضيف متجه التحيز إلى صفوف الخروج، فإن الشكول لا تتطابق. الإذاعة تمدد المجموعة الأصغر لتتناسب.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

كل إطار جديد يفعل هذا تلقائياً، فهمه يمنع الارتباك عندما تبدو الأشكال خاطئة ولكن الرمز يعمل.

```figure
vector-projection
```

## بناءها

### الخطوة الأولى: فئة المتجهات

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

### الخطوة الثانية: فئة المصفوفة مع العمليات الأساسية

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

### الخطوة الثالثة: شاهدوا أن الأمر يعمل

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

### الخطوة الرابعة: الاتصال بالشبكات العصبية

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

هذه طبقة كثيفة واحدة:`output = relu(W @ x + b)`كل طبقة كثيفة في كل شبكة عصبية تفعل هذا بالضبط.

## استخدمها

إنّ (نومبي) يقوم بكلّ شيء أعلاه في خطوط أقل وأوامر أكبر أسرع.

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

- نعم`@`عامل في مكالمات Python `__matmul__`ينفذها NumPy مع روتينات BLAS المثلى مكتوبة باللغة C و Fortran نفس الرياضيات، أسرع بنسبة 100 مرة

الإذاعة في NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

يُبث NumPy تلقائيًا التحيز 1D عبر كل من الصفوف. هكذا يعمل إضافة التحيز في كل إطار شبكة عصبية.

## أرسله

هذه الدروس تنتج طلب لتعليم عمليات المصفوفة من خلال الحدس الهندسي. انظر `outputs/prompt-matrix-operations.md`. . .

فصيلة المصفوفة التي بنيت هنا هي أساس إطار شبكة عصبية صغيرة بنيت في المرحلة 3، الدروس 10.

## التمارين

1. **Verify the inverse.**ضرب`A @ A.inverse_2x2()`و تأكد من الحصول على المصفوفة الهوية. حاولي ذلك مع ثلاث مصفوفات مختلفة 2x2. ماذا يحدث عندما يكون المحدد هو الصفر؟

2. **Implement 3x3 inverse.**تمديد فئة المصفوفة لحساب العكسات للمصفوفات 3x3 باستخدام طريقة الجمع. اختبرها ضد NumPy `np.linalg.inv`. . .

3. **Build a two-layer network.**باستخدام فئة المصفوفة الخاصة بك فقط (لا يوجد NumPy) ، قم بإنشاء شبكة عصبية ذات طبقتين: المدخل (3) -> الخفية (4) -> الخروج (2). قم بتشغيل الأوزان العشوائية ، و قم بتشغيل مرور إلى الأمام ، وتحقق من صحة جميع الأشكال.

## الشروط الرئيسية

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

## المزيد من القراءة

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)- البصرية للعمليات التي تم تغطيتها هنا
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)- القواعد الدقيقة التي تتبعها NumPy
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf)- إشارة موجزة للجزرية الخطية الخاصة بالجهاز
