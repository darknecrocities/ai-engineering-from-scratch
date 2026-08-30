# تحويلات المصفوفة

> المصفوفة هي آلة تقوم بإعادة تشكيل الفضاء تعلم ما تفعله لكل نقطة، وستفهم التحول بأكمله

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lessons 01-02 (Linear Algebra Intuition, Vectors & Matrices Operations)
**Time:** ~75 minutes

## أهداف التعلم

- بناء المصفوفات الدورانية والتنمية والقص والعكس وتطبيقها على النقاط 2D و 3D
- قم بتكوين تحويلات متعددة عن طريق مضاعفة المصفوفة وتحقق من أن النظام مهم
- احسب القيم الخاصة والمتجهات الخاصة للمصفوفات 2x2 من المعادلة المميزة
- شرح لماذا تقرر القيم الخاصة اتجاهات PCA، استقرار RNN، وسلوك التجميع الطيفي

## المشكلة

تقرأ عن PCA وترى "عثر على المتجهات الخاصة للمصفوفة التغيرات". تقرأ عن استقرار النموذج وترى "تحقق من أن جميع القيم الخاصة لها حجم أقل من 1." تقرأ عن زيادة البيانات وترى "تطبق دوران عشوائي". لا شيء من هذا منطقي حتى تفهم ما تفعله المصفوفات إلى الفضاء هندسيًا.

المصفوفات ليست مجرد شبكات الأرقام. إنها آلات مساحية. تقوم المصفوفة الدورانية بتدوير نقاط. تقوم المصفوفة التوسيعية بتمددها. تقوم المصفوفة الشاحنة بتدحرجها. كل تحول يطبقه شبكة عصبية على البيانات هو أحد هذه العمليات أو تركيب منها. هذا الدروس يجعل هذه العمليات ملموسة.

## المفهوم

### التحولات كالمصفوفات

كل تحول خطي في 2D يمكن كتابته كصفائح 2x2 . المصفوفة تخبرك بالضبط أين ينتهي المتجهات الأساسية [1, 0] و [0, 1]. كل شيء آخر يتبع.

```mermaid
graph LR
    subgraph Before["Standard Basis"]
        e1["e1 = [1, 0] (along x)"]
        e2["e2 = [0, 1] (along y)"]
    end
    subgraph Transform["Matrix M"]
        M["M = columns are new basis vectors"]
    end
    subgraph After["After Transformation M"]
        e1p["e1' = new x-basis"]
        e2p["e2' = new y-basis"]
    end
    e1 --> M --> e1p
    e2 --> M --> e2p
```

### التناوب

تدور ثنائي الأبعاد عن طريق الزاوية تثا تبقي المسافات والزوايا سليمة.

```mermaid
graph LR
    subgraph Before["Before Rotation"]
        A["A(2, 1)"]
        B["B(0, 2)"]
    end
    subgraph Rot["Rotate 45 degrees"]
        R["R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]"]
    end
    subgraph After["After Rotation"]
        Ap["A'(0.71, 2.12)"]
        Bp["B'(-1.41, 1.41)"]
    end
    A --> R --> Ap
    B --> R --> Bp
```

في 3D، تدور حول محور. كل محور لديه ماتريسك التناوب الخاص به:

```
Rz(theta) = | cos  -sin  0 |     Rotate around z-axis
            | sin   cos  0 |     (x-y plane spins, z stays)
            |  0     0   1 |

Rx(theta) = | 1   0     0    |   Rotate around x-axis
            | 0  cos  -sin   |   (y-z plane spins, x stays)
            | 0  sin   cos   |

Ry(theta) = |  cos  0  sin |     Rotate around y-axis
            |   0   1   0  |     (x-z plane spins, y stays)
            | -sin  0  cos |
```

### التوسع

تمدد أو تضغط على طول كل محور بشكل مستقل.

```mermaid
graph LR
    subgraph Before["Before Scaling"]
        A["A(2, 1)"]
        B["B(0, 2)"]
    end
    subgraph Scale["Scale sx=2, sy=0.5"]
        S["S = [[2, 0], [0, 0.5]]"]
    end
    subgraph After["After Scaling"]
        Ap["A'(4, 0.5)"]
        Bp["B'(0, 1)"]
    end
    A --> S --> Ap
    B --> S --> Bp
```

### القص

يلتحى الشفرة المحور الواحد بينما يبقى المحور الآخر ثابتًا.

```mermaid
graph LR
    subgraph Before["Before Shear"]
        A["A(1, 0)"]
        B["B(0, 1)"]
    end
    subgraph Shear["Shear in x, k=1"]
        Sh["Shx = [[1, k], [0, 1]]"]
    end
    subgraph After["After Shear"]
        Ap["A(1, 0) unchanged"]
        Bp["B'(1, 1) shifted"]
    end
    A --> Sh --> Ap
    B --> Sh --> Bp
```

المصفوفات المقطوعة:
- `Shx = [[1, k], [0, 1]]`تحويلات x من k * y
- `Shy = [[1, 0], [k, 1]]`تحويلات y من k * x

### التفكير

العكس يعكس نقاط عبر محور أو خط.

```mermaid
graph LR
    subgraph Before["Before Reflection"]
        A["A(2, 1)"]
    end
    subgraph Reflect["Reflect across y-axis"]
        R["[[-1, 0], [0, 1]]"]
    end
    subgraph After["After Reflection"]
        Ap["A'(-2, 1)"]
    end
    A --> R --> Ap
```

المصفوفات التأثيرية:
- انعكس عبر محور y: `[[-1, 0], [0, 1]]`
- انعكس عبر محور x: `[[1, 0], [0, -1]]`

### التكوين: تحولات السلاسل

تطبيق التحول A ثم B هو نفس ضرب ماتريسكهم: `result = B @ A @ point`النظام مهم، ثم تدور المقياس يعطي نتائج مختلفة عن المقياس ثم تدور.

```mermaid
graph LR
    subgraph Path1["Rotate 90 then Scale (2, 0.5)"]
        P1["(1, 0)"] -->|"Rotate 90"| P2["(0, 1)"] -->|"Scale"| P3["(0, 0.5)"]
    end
```

المكون: `S @ R = [[0, -2], [0.5, 0]]`

```mermaid
graph LR
    subgraph Path2["Scale (2, 0.5) then Rotate 90"]
        Q1["(1, 0)"] -->|"Scale"| Q2["(2, 0)"] -->|"Rotate 90"| Q3["(0, 2)"]
    end
```

المكون: `R @ S = [[0, -0.5], [2, 0]]`

نتائج مختلفة مضاعفة المصفوفة ليست محولة

### القيم الخاصة والمتجهات الخاصة

معظم المتجهات تغير الاتجاه عندما تضربها المصفوفة. المتجهات الخاصة خاصة: المصفوفة فقط يقياسها، لا يدورها أبدا. عامل التقياس هو القيمة الخاصة.

```
A @ v = lambda * v

v is the eigenvector (direction that survives)
lambda is the eigenvalue (how much it stretches)

Example: A = | 2  1 |
             | 1  2 |

Eigenvector [1, 1] with eigenvalue 3:
  A @ [1,1] = [3, 3] = 3 * [1, 1]     (same direction, scaled by 3)

Eigenvector [1, -1] with eigenvalue 1:
  A @ [1,-1] = [1, -1] = 1 * [1, -1]  (same direction, unchanged)
```

المصفوفة تمدد الفضاء بـ 3x على طول [1, 1] وتبقى [1, -1] دون تغيير. كل اتجاه آخر هو مزيج من هذين الاثنين.

### التكوين الخاص

إذا كانت المصفوفة لديها n متجهات خاصة مستقلة خطيا، يمكن تفكيكها:

```
A = V @ D @ V^(-1)

V = matrix whose columns are eigenvectors
D = diagonal matrix of eigenvalues
V^(-1) = inverse of V

This says: rotate into eigenvector coordinates, scale along each axis, rotate back.
```

### لماذا قيمها الخاصة مهمة

**PCA.**المتجهات الخاصة للمصفوفة التغيرات هي المكونات الرئيسية. القيم الخاصة تخبرك كم التباين كل مكون يلتقط. فرز حسب القيمة الخاصة، والحفاظ على الجزء العلوي k، ولديك خفض الأبعاد.

**Stability.**في الشبكات المتكررة والأنظمة الديناميكية، تسبب القيم الخاصة ذات الحجم > 1 انفجار المخرجات. الحجم < 1 يسبب اختفاءها. هذه هي مشكلة التهاب/انفجار التدفق الموضح في جملة واحدة.

**Spectral methods.**تستخدم شبكات العصبية الرسمية القيم الخاصة بالمصفوفة المجاورة. تستخدم التجميع الطيفي القيم الخاصة باللابلاسي. الكائنات الخاصة تكشف عن هيكل الرسم البياني.

### العامل القياسي كعامل تحديد حجم

ويقول لك معدل ماتريكس التحويل كم يقدر مساحة (2D) أو حجم (3D).

```
det = 1:   area preserved (rotation)
det = 2:   area doubled
det = 0:   space crushed to lower dimension (singular)
det = -1:  area preserved but orientation flipped (reflection)

| det(Rotation) | = 1        (always)
| det(Scale sx, sy) | = sx * sy
| det(Shear) | = 1           (area preserved)
| det(Reflection) | = -1     (orientation flipped)
```

```figure
matrix-transform
```

## بناءها

### الخطوة 1: المصفوفات التحولية من الصفر (بيتون)

```python
import math

def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]

def scaling_2d(sx, sy):
    return [[sx, 0], [0, sy]]

def shearing_2d(kx, ky):
    return [[1, kx], [ky, 1]]

def reflection_x():
    return [[1, 0], [0, -1]]

def reflection_y():
    return [[-1, 0], [0, 1]]

def mat_vec_mul(matrix, vector):
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]

def mat_mul(a, b):
    rows_a, cols_b = len(a), len(b[0])
    cols_a = len(a[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]

point = [1.0, 0.0]
angle = math.pi / 4

rotated = mat_vec_mul(rotation_2d(angle), point)
print(f"Rotate (1,0) by 45 deg: ({rotated[0]:.4f}, {rotated[1]:.4f})")

scaled = mat_vec_mul(scaling_2d(2, 3), [1.0, 1.0])
print(f"Scale (1,1) by (2,3): ({scaled[0]:.1f}, {scaled[1]:.1f})")

sheared = mat_vec_mul(shearing_2d(1, 0), [1.0, 1.0])
print(f"Shear (1,1) kx=1: ({sheared[0]:.1f}, {sheared[1]:.1f})")

reflected = mat_vec_mul(reflection_y(), [2.0, 1.0])
print(f"Reflect (2,1) across y: ({reflected[0]:.1f}, {reflected[1]:.1f})")
```

### الخطوة الثانية: تشكيل التحوّلات

```python
R = rotation_2d(math.pi / 2)
S = scaling_2d(2, 0.5)

rotate_then_scale = mat_mul(S, R)
scale_then_rotate = mat_mul(R, S)

point = [1.0, 0.0]
result1 = mat_vec_mul(rotate_then_scale, point)
result2 = mat_vec_mul(scale_then_rotate, point)

print(f"Rotate 90 then scale: ({result1[0]:.2f}, {result1[1]:.2f})")
print(f"Scale then rotate 90: ({result2[0]:.2f}, {result2[1]:.2f})")
print(f"Same? {result1 == result2}")
```

### الخطوة الثالثة: القيم الخاصة من الصفر (2x2)

لـ 2 × 2 المصفوفة`[[a, b], [c, d]]`، القيم الخاصة تحل المعادلة المميزة: `lambda^2 - (a+d)*lambda + (ad - bc) = 0`. . .

```python
def eigenvalues_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    det = a * d - b * c
    discriminant = trace ** 2 - 4 * det
    if discriminant < 0:
        real = trace / 2
        imag = (-discriminant) ** 0.5 / 2
        return (complex(real, imag), complex(real, -imag))
    sqrt_disc = discriminant ** 0.5
    return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)

def eigenvector_2x2(matrix, eigenvalue):
    a, b = matrix[0]
    c, d = matrix[1]
    if abs(b) > 1e-10:
        v = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        v = [eigenvalue - d, c]
    else:
        if abs(a - eigenvalue) < 1e-10:
            v = [1, 0]
        else:
            v = [0, 1]
    mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return [v[0] / mag, v[1] / mag]

A = [[2, 1], [1, 2]]
vals = eigenvalues_2x2(A)
print(f"Matrix: {A}")
print(f"Eigenvalues: {vals[0]:.4f}, {vals[1]:.4f}")

for val in vals:
    vec = eigenvector_2x2(A, val)
    result = mat_vec_mul(A, vec)
    scaled = [val * vec[0], val * vec[1]]
    print(f"  lambda={val:.1f}, v={[round(x,4) for x in vec]}")
    print(f"    A@v = {[round(x,4) for x in result]}")
    print(f"    l*v = {[round(x,4) for x in scaled]}")
```

### الخطوة 4: العامل القياسي كعامل تحديد حجم

```python
def det_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

print(f"det(rotation 45) = {det_2x2(rotation_2d(math.pi/4)):.4f}")
print(f"det(scale 2,3)   = {det_2x2(scaling_2d(2, 3)):.1f}")
print(f"det(shear kx=1)  = {det_2x2(shearing_2d(1, 0)):.1f}")
print(f"det(reflect y)   = {det_2x2(reflection_y()):.1f}")

singular = [[1, 2], [2, 4]]
print(f"det(singular)     = {det_2x2(singular):.1f}")
print("Singular: columns are proportional, space collapses to a line.")
```

## استخدمها

يقوم NumPy بمعالجة كل هذا مع روتينات محسنة.

```python
import numpy as np

theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

point = np.array([1.0, 0.0])
print(f"Rotate (1,0) by 45 deg: {R @ point}")

S = np.diag([2.0, 3.0])
composed = S @ R
print(f"Scale(2,3) after Rotate(45): {composed @ point}")

A = np.array([[2, 1], [1, 2]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"  A @ v{i} = {A @ v}, lambda * v{i} = {lam * v}")

print(f"\ndet(R) = {np.linalg.det(R):.4f}")
print(f"det(S) = {np.linalg.det(S):.1f}")

B = np.array([[3, 1], [0, 2]], dtype=float)
vals, vecs = np.linalg.eig(B)
D = np.diag(vals)
V = vecs
reconstructed = V @ D @ np.linalg.inv(V)
print(f"\nEigendecomposition A = V @ D @ V^-1:")
print(f"Original:\n{B}")
print(f"Reconstructed:\n{reconstructed}")
```

### الدوران ثلاثي الأبعاد مع NumPy

```python
def rotation_3d_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rotation_3d_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

point_3d = np.array([1.0, 0.0, 0.0])
rotated_z = rotation_3d_z(np.pi / 2) @ point_3d
rotated_x = rotation_3d_x(np.pi / 2) @ point_3d

print(f"\n3D point: {point_3d}")
print(f"Rotate 90 around z: {np.round(rotated_z, 4)}")
print(f"Rotate 90 around x: {np.round(rotated_x, 4)}")
```

## أرسله

يبنى هذا الدروس الأساس الهندسية لتحليل وزن الشبكة العصبية (Phase 2) والحسابات. يعد رمز القيمة الخاصة / الجهاز التنفيذي المبن هنا نفس الخوارزمية التي تشجع تقليل الأبعاد ، والتكسيم الطيفي ، وتحليل الاستقرار في أنظمة ML الإنتاجية.

## التمارين

1. تطبيق الدوران، وتحديد النطاق، والقص على مربع وحدة (الزوايا في [0,0، [1,0، [1,1]، [0,1]). طبع الزوايا المحوّلة لكل منها. تحقق من أن الدوران يحافظ على المسافات بين الزوايا.

2. ابحث عن القيم الخاصة للمصفوفة [4, 2] ، [1, 3]] يدوياً باستخدام المعادلة الخصائصية. ثم تحقق باستخدام وظيفة الصفر الخاصة بك ومع NumPy.

3. قم بإنشاء تركيبة من ثلاثة تحويلات (تحول 30 درجة، وتحقيق مقياس [1.5، 0.8، وتقطع مع kx=0.3) وتطبيقها على 8 نقاط مرتبة في دائرة. طبع قبل وبعد الإحداثيات. احسب معين المصفوفة المكونة وتحقق من أنها تساوي نسبة معينة من المعينات الفردية.

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Rotation matrix | "Spins things" | An orthogonal matrix that moves points along circular arcs while preserving distances and angles. Determinant is always 1. |
| Scaling matrix | "Makes things bigger" | A diagonal matrix that stretches or compresses independently along each axis. Determinant is the product of scale factors. |
| Shearing matrix | "Slants things" | A matrix that shifts one coordinate proportionally to another, turning rectangles into parallelograms. Determinant is 1. |
| Reflection | "Mirrors things" | A matrix that flips space across an axis or plane. Determinant is -1. |
| Composition | "Do two things" | Multiplying transformation matrices to chain operations. Order matters: B @ A means apply A first, then B. |
| Eigenvector | "Special direction" | A direction that the matrix only scales, never rotates. The transformation's fingerprint. |
| Eigenvalue | "How much it stretches" | The scalar factor by which the matrix scales its eigenvector. Can be negative (flip) or complex (rotation). |
| Eigendecomposition | "Break the matrix apart" | Writing a matrix as V @ D @ V^(-1), separating it into its fundamental scaling directions and magnitudes. |
| Determinant | "A single number from a matrix" | The factor by which the transformation scales area (2D) or volume (3D). Zero means the transformation is irreversible. |
| Characteristic equation | "Where eigenvalues come from" | det(A - lambda * I) = 0. The polynomial whose roots are the eigenvalues. |

## المزيد من القراءة

- [3Blue1Brown: Linear Transformations](https://www.3blue1brown.com/lessons/linear-transformations)-- البصرية للشكل الذي يعيد تشكيل المصفوفات الفضاء
- [3Blue1Brown: Eigenvectors and Eigenvalues](https://www.3blue1brown.com/lessons/eigenvalues)-- أفضل تفسير بصري لما تعنيه المتجهات ذاتية بشكل هندسي
- [MIT 18.06 Lecture 21: Eigenvalues and Eigenvectors](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)-علاج كلاسيكي (غيلبرت سترانغ)
