# قاعدة السلسلة والتفريق الآلي

> قاعدة السلسلة هي المحرك وراء كل شبكة عصبية تتعلم.

**Type:** Build
**Language:**بايثون
**Prerequisites:** Phase 1, Lesson 04 (Derivatives & Gradients)
**Time:** ~90 minutes

## أهداف التعلم

- بناء محرك أوتوغريد أدنى (فئة القيمة) الذي يسجل العمليات ويحسب التراجع عن طريق التشغيل الذاتي في وضع العكس
- تنفيذ خطوات للأمام والخلفية عبر الرسم البياني للحساب باستخدام الترتيب التوبولوجي
- بناء وتدريب جهاز الرؤية متعددة الطبقات على XOR باستخدام محرك Autograd فقط من الصفر
- التحقق من صحة التغيير الذاتي باستخدام التحقق من التراجع ضد الاختلافات النهائية الرقمية

## المشكلة

يمكنك حساب مشتقات الوظائف البسيطة. ولكن شبكة عصبية ليست وظيفة بسيطة. إنها مئات الوظائف المكونة معا: مضاعفة المصفوفة، إضافة التحيز، تطبيق التفعيل، مضاعفة المصفوفة مرة أخرى، softmax، خسارة الانتروبية المتقاطعة. الخروج هو وظيفة من وظيفة من وظيفة من وظيفة.

لتدريب الشبكة، تحتاج إلى تراجع الخسارة فيما يتعلق بكل وزن واحد. القيام بذلك يدويا مستحيل لملايين المعلمات. القيام بذلك عددا (الاختلافات المحدودة) بطيئة جدا.

قاعدة السلسلة تعطيك الرياضيات. التفريق الآلي يعطيك الخوارزمية. معاً يسمح لك بتحساب التراجع الدقيق من خلال تركيبات تعسفية من الوظائف في الوقت النسبي لمرور واحد للأمام.

هكذا تعمل PyTorch و TensorFlow و JAX ستقوم ببناء نسخة صغيرة من الصفر

## المفهوم

### قاعدة السلسلة

إذا`y = f(g(x))`، مشتقة من `y`فيما يتعلق بـ`x`هو:

```
dy/dx = dy/dg * dg/dx = f'(g(x)) * g'(x)
```

ضرب المشتقات على طول السلسلة كل وصلة تساهم بمشتقاتها المحلية

مثال:`y = sin(x^2)`

```
g(x) = x^2       g'(x) = 2x
f(g) = sin(g)     f'(g) = cos(g)

dy/dx = cos(x^2) * 2x
```

بالنسبة للمكونات العميقة، تمتد السلسلة:

```
y = f(g(h(x)))

dy/dx = f'(g(h(x))) * g'(h(x)) * h'(x)
```

كل طبقة في شبكة عصبية هي ربطة واحدة في هذه السلسلة.

### الرسومات الحسابية

الرسم البياني الحاسوبي يجعل قاعدة السلسلة مرئية كل عملية تصبح عقدة. البيانات تتدفق إلى الأمام عبر الرسم البياني. تدفق الدرجات إلى الوراء.

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

يطبق التسلسل للخلف في كل عقدة، وتتكاثر التدرج من الخروج إلى المدخلات.

### الوضع الأمامي مقابل الوضع العكسي

هناك طريقتان لتطبيق قاعدة السلسلة من خلال الرسم البياني.

**Forward mode**يبدأ في المدخلات ويضغط على المشتقات إلى الأمام.`dx/dx = 1`ويتكاثر خلال كل عملية، جيد عندما يكون لديك عدد قليل من المدخلات و الكثير من المخرجات

```
Forward mode: seed dx/dx = 1, propagate forward

  x = 2       (dx/dx = 1)
  a = x^2     (da/dx = 2x = 4)
  y = sin(a)  (dy/dx = cos(a) * da/dx = cos(4) * 4 = -2.615)
```

**Reverse mode**يبدأ في الخروج ويرفع التراجع إلى الوراء.`dy/dy = 1`و ينتشر من خلال كل عملية بالعكس. جيد عندما يكون لديك الكثير من المدخلات و عدد قليل من المخرجات.

```
Reverse mode: seed dy/dy = 1, propagate backward

  y = sin(a)  (dy/dy = 1)
  a = x^2     (dy/da = cos(a) = cos(4) = -0.654)
  x = 2       (dy/dx = dy/da * da/dx = -0.654 * 4 = -2.615)
```

شبكات الأعصاب لديها ملايين المدخلات (الوزن) ومخرج واحد (الخسارة). يحسب وضع العكس جميع التدفقات في مرور واحد للخلف. لهذا السبب يستخدم التنشر العكسي وضع العكس.

| Mode | Seed | Direction | Best when |
|------|------|-----------|-----------|
| Forward | `dx_i/dx_i = 1` | Input to output | Few inputs, many outputs |
| Reverse | `dy/dy = 1` | Output to input | Many inputs, few outputs (neural nets) |

### أرقام مزدوجة لنظام التقدم

يمكن تنفيذ وضع الأمام بشكل أروع مع أرقام مزدوجة.`a + b*epsilon`أين`epsilon^2 = 0`. . .

```
Dual number: (value, derivative)

(2, 1) means: value is 2, derivative w.r.t. x is 1

Arithmetic rules:
  (a, a') + (b, b') = (a+b, a'+b')
  (a, a') * (b, b') = (a*b, a'*b + a*b')
  sin(a, a')         = (sin(a), cos(a)*a')
```

زرع متغير المدخل مع مشتق 1. ينتشر المشتق تلقائيا خلال كل عملية.

### بناء محرك أوتوجراد

محركات الـ " أوتوغراد " تحتاج إلى ثلاثة أشياء:

1. **Value wrapping.**لف كل رقم في جسم يحتفظ بقيمةه و تراجعها
2. **Graph recording.**كل عملية تسجل مدخلاتها و وظيفة التراجع المحلية.
3. **Backward pass.**تصفية الترتيبات التوبولوجية الرسم البياني، ثم المشي بها العكس، تطبيق قاعدة السلسلة في كل عقدة.

هذا بالضبط ما هو PyTorch`autograd`نعم، نعم`torch.Tensor`الصف يحتوي على قيم، يسجل العمليات عندما `requires_grad=True`، ويحسب التراجع عندما تدعو`.backward()`. . .

### كيف تعمل الـ "بيتورش أوتوجراد" تحت الغطاء

عندما تكتب رمز PyTorch:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # 7.0 = 2*x + 3 = 2*2 + 3
```

(بيتورش) داخلياً:

1. يخلق`Tensor`العقدة`x`مع`requires_grad=True`
2. كل عملية (`**`،`*`،`+`) يخلق عقدة جديدة ويستسجل وظيفة الخلفية
3. `y.backward()`يُفيد التشغيل الذاتي في وضع العكس عبر الرسم البياني المسجل
4. كل عقدة`grad_fn`يحسب التراجع المحلي ويمرره إلى العقدة الأم
5. تتراكم الدرجات في`.grad`الصفات عن طريق الإضافة (ليس البداية)

الرسم البياني ديناميكي (محدد من خلال تشغيل). يتم بناء الرسم البياني الجديد على كل مرور إلى الأمام. لهذا السبب يدعم PyTorch تدفق التحكم (إذا / وإلا ، حلقات) داخل النماذج.

```figure
chain-rule
```

## بناءها

### الخطوة الأولى: فئة القيمة

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

كلّ شخص`Value`تخزين بياناتها الرقمية، وتحديدها (في البداية صفر) ، وظيفة تراجعية، وتشير إلى العقدة الطفولة التي أنتجتها.

### الخطوة الثانية: العمليات الحسابية مع تعقب التراجع

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

كل عملية تخلق إغلاق يعرف كيفية حساب التدرج المحلي وتضاعف بالتحدر فوق التدرج (`out.grad`)`+=`يتعامل مع الحالة التي تستخدم فيها قيمة في عمليات متعددة.

### الخطوة الثالثة: المرور للخلف

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

التنظيم التوبولوجي يضمن حساب تراجع كل عقدة بشكل كامل قبل أن ينتشر إلى أطفالها. تراجع البذور هو 1.0 (dy / dy = 1).

### الخطوة الرابعة: المزيد من العمليات لمحرك كامل

فصفة القيمة الأساسية تتعامل مع الجمع والضاعفة والتحرك. محرك أوتوجراد الحقيقي يحتاج إلى المزيد. إليك العمليات التي تحتاجها لبناء شبكات عصبية:

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

الجزء الذكي:`__sub__`و`__truediv__`يتم تعريفها من حيث العمليات الحالية. فإنها تحصل على تراجعات صحيحة مجانا لأن قاعدة السلسلة تتكون من خلال العمليات الأساسية إضافة / مول / بو.

### الخطوة 5: ميني MLP من الصفر

مع فئة القيم الكاملة، يمكنك بناء شبكة عصبية لا توارش، لا NumPy، فقط القيم وقاعدة السلسلة.

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

أ`Neuron`الحسابات`tanh(w1*x1 + w2*x2 + ... + b)`.`Layer`هو قائمة بالخلايا العصبية.`MLP`كل وزن هو`Value`، لذا اتصل`loss.backward()`يُنشر التدرج إلى كل مُعيار

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

هذا هو micrograd. حلقة تدريب كاملة للشبكات العصبية في Python النقي مع التمييز الآلي. كل إطار للتعلم العميق التجاري يفعل الشيء نفسه على نطاق واسع.

### الخطوة 6: التحقق من التدريج

كيف تعرف أن التشخيص الذاتي صحيح؟ قارنه بالمشتقات الرقمية. هذا هو التحقق من التراجع.

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

اختبرها على تعبير معقد:

```python
def expr(x):
    return (x ** 3 + x * 2 + 1).tanh()

ad, num, diff = gradient_check(expr, 0.5)
print(f"Autodiff:  {ad:.8f}")
print(f"Numerical: {num:.8f}")
print(f"Difference: {diff:.2e}")
# Difference should be < 1e-5
```

التحقق من التدرج ضروري عند تنفيذ عمليات جديدة. إذا كان لديك مرور خلفي لديه خطأ، فإن التحقق الرقمي يلتقطه. كل تنفيذ دراسة عميقة خطيرة يقوم بتحقق من التدرج أثناء التطوير.

**When to use gradient checking:**

| Situation | Do gradient check? |
|-----------|-------------------|
| Adding a new operation to your autograd | Yes, always |
| Debugging a training loop that won't converge | Yes, check gradients first |
| Production training | No, too slow (2x forward passes per parameter) |
| Unit tests for autograd code | Yes, automate it |

### الخطوة 7: التحقق من حساب يدوي

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

التحقق اليدوي: `y = relu(x1*x2 + 1)`منذ`x1*x2 + 1 = 7 > 0`"ريلو" هو الهوية
`dy/dx1 = x2 = 3`. .`dy/dx2 = x1 = 2`المحرك يطابق

## استخدمها

### التحقق من PyTorch

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

نفس التراجعات، محركك يحسب نفس النتيجة مثل PyTorch لأن الرياضيات هي نفسها: التشغيل الذاتي في وضع العكس عبر قاعدة السلسلة.

### تعبير أكثر تعقيداً

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

## أرسله

هذا الدرس ينتج عن:
- `outputs/skill-autodiff.md`-- مهارة لبناء وتحليل الاصلاحيات
- `code/autodiff.py`-- محرك بسيط من نوعه يمكنك تمديد

فئة القيمة التي تم بناؤها هنا هي الأساس لدورة تدريب شبكة الأعصاب في المرحلة 3.

## التمارين

1. إضافة`__pow__`إلى فئة القيمة حتى تتمكن من الحساب`x ** n`تأكد من ذلك`d/dx(x^3)`في`x=2`متساوية`12.0`. . .

2. إضافة`tanh`كعمل تنشيط. تحقق من ذلك`tanh'(0) = 1`و`tanh'(2) = 0.0707`(تقريب)

3. بناء الرسم البياني الحسابي لنورون واحد:`y = relu(w1*x1 + w2*x2 + b)`قم بحساب كلّ الخمسة من التدرج وتحقق من "بيتورش".

4. تنفيذ التشغيل الذاتي في وضع التقدم باستخدام أرقام مزدوجة.`Dual`و التحقق من أنه يعطي نفس المشتقات مثل محركك في وضع العكس.

## الشروط الرئيسية

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

## المزيد من القراءة

- [3Blue1Brown: Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8)-- تفسير بصري لقاعدة السلسلة في الشبكات العصبية
- [PyTorch Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)-- كيف يعمل النظام الحقيقي
- [Baydin et al., Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)-- إشارة شاملة
