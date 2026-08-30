# مقدمة لـ PyTorch

> لقد صنعت المحرك من محركات الحامل والحركات الآن تعلم ما يقوده الجميع

**Type:** Build
**Languages:** Python
**Prerequisites:** Lesson 03.10 (Build Your Own Mini Framework)
**Time:** ~75 minutes

## أهداف التعلم

- بناء وتدريب شبكات عصبية باستخدام PyTorch nn.Module، nn.Sequential، و autograd
- استخدموا مؤشرات PyTorch، وتسارع GPU، ودورة التدريب القياسية (zero_grad، للأمام، الخسارة، الخلف، الخطوة)
- حول مكونات الإطار الصغرى من الصفر إلى ما يعادلها بـ PyTorch
- الملف الشخصي ومقارنة سرعة التدريب بين إطار Python Pure- و PyTorch على نفس المهمة

## المشكلة

لديك إطار عمل صغير. طبقات خطية، ريلو، التخلي عن، معيار المجموعة، آدم، جهاز تحميل بيانات، حلقة تدريب. إنه تدرب شبكة 4 طبقات على مشكلة تصنيف الدائرة في بيثون النقي.

كما أنه بطيء 500x من PyTorch في نفس المشكلة.

يقوم PyTorch بإرسال نفس العمليات إلى أجزاء C ++ / CUDA المثلى التي تعمل على GPU. على NVIDIA A100 واحدة، يقوم PyTorch بتدريب ResNet-50 (25.6M ملامي الحدود) على ImageNet (1.28M الصور) في حوالي 6 ساعات. سيستغرق إطارك حوالي 3000 ساعة على نفس المهمة - إذا لم ينفذ من الذاكرة أولاً.

السرعة ليست الفجوة الوحيدة. لا يوجد دعم لـ GPU في إطار العمل الخاص بك. لا يوجد تمييز تلقائي -- كتبت يدوياً للخلف) لكل وحدات. لا توجد سلسلة. لا تدريب منتشر. لا توجد دقة مختلطة. لا توجد طريقة لإلغاء التدفق التدريجي دون بيانات الطباعة.

تقوم PyTorch بملء كل هذه الفجوات. وهي تفعل ذلك مع الحفاظ على نفس النموذج العقلي بالضبط الذي قمت بإنشائه بالفعل: الوحدة ، للأمام ((() ، المعلمات ((() ، والعودة ((() ، والتحسين. الخطوة ((().

## المفهوم

### لماذا فاز بيتورش

في عام 2015، طلب من TensorFlow تعريف الرسم البياني للحوسبة الدولية قبل تشغيل أي شيء. قمت ببناء الرسم البياني، ووضعها، ثم إرسال البيانات من خلالها. إعادة التشغيل يعني النظر في تصاميم الرسم البياني. تغيير الهندسة المعمارية يعني إعادة بناء الرسم البياني من الصفر.

بدأت PyTorch في عام 2017 مع فلسفة مختلفة: تنفيذ حريص. يمكنك كتابة Python.`y = model(x)`في الواقع يحسب y الآن، وليس "إضافة عقدة إلى الرسم البياني الذي سوف يحسب y في وقت لاحق". هذا يعني أدوات التحليل القياسية Python عملت. طباعة() عملت. pdb عملت. إذا /else في مرورك المقدمة عملت.

بحلول عام 2020 ، كان السوق قد تحدثت. ارتفع حصة PyTorch في ورق بحث ML من 7% (2017) إلى أكثر من 75% (2022). تستخدم Meta ، Google DeepMind ، OpenAI ، Anthropic ، و Hugging Face جميعًا PyTorch كإطار أساسي. تبنت TensorFlow 2.x تنفيذًا حريصًا رداً على ذلك - الاعتراف الصامت بأن تصميم PyTorch كان صحيحًا.

الدرس: تجربة المطورين المركبات إطار عمل بطيء بنسبة 10% ولكن أسرع بنسبة 50% لإزالة التشويق يفوز كل مرة

### الجهاز

الجهاز هو مجموعة متعددة الأبعاد مع ثلاثة خصائص حاسمة: الشكل، dtype، والجهاز.

```python
import torch

x = torch.zeros(3, 4)           # shape: (3, 4), dtype: float32, device: cpu
x = torch.randn(2, 3, 224, 224) # batch of 2 RGB images, 224x224
x = torch.tensor([1, 2, 3])     # from a Python list
```

**Shape**هو الامتعداد. وهو شكل (), وكتور هو (n,), ماتريكس هو (m, n), مجموعة من الصور هي (مجموعة, قنوات, ارتفاع, عرض).

**Dtype**يسيطر على الدقة والذاكرة

| dtype | Bits | Range | Use case |
|-------|------|-------|----------|
| float32 | 32 | ~7 decimal digits | Default training |
| float16 | 16 | ~3.3 decimal digits | Mixed precision |
| bfloat16 | 16 | Same range as float32, less precision | LLM training |
| int8 | 8 | -128 to 127 | Quantized inference |

**Device**يحدد أين يحدث الحساب.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(3, 4, device=device)
x = x.to("cuda")
x = x.cpu()
```

كل عملية تتطلب جميع الجهاز التنسورية نفسها. هذا هو الخطأ رقم 1 PyTorch المبتدئين ضرب: `RuntimeError: Expected all tensors to be on the same device`إصلاحها عن طريق نقل كل شيء إلى نفس الجهاز قبل الحساب.

**Reshaping**هو الوقت المستمر -- يغير البيانات المعدنية، وليس البيانات.

```python
x = torch.randn(2, 3, 4)
x.view(2, 12)      # reshape to (2, 12) -- must be contiguous
x.reshape(6, 4)    # reshape to (6, 4) -- works always
x.permute(2, 0, 1) # reorder dimensions
x.unsqueeze(0)     # add dimension: (1, 2, 3, 4)
x.squeeze()        # remove size-1 dimensions
```

### أوتوجراد

يتطلب منك إطار العمل الصغير تنفيذها للخلف (() لكل وحدات. لا يفعل PyTorch. يسجل كل عملية على العجلات في رسومة محمولة موجهة (الرسمة الحاسبية) ثم يمر عبر هذا الرسمة العكسي لحساب تراجعات تلقائيا.

```mermaid
graph LR
    x["x (leaf)"] --> mul["*"]
    w["w (leaf, requires_grad)"] --> mul
    mul --> add["+"]
    b["b (leaf, requires_grad)"] --> add
    add --> loss["loss"]
    loss --> |".backward()"| add
    add --> |"grad"| b
    add --> |"grad"| mul
    mul --> |"grad"| w
```

الفرق الرئيسي من إطارك: PyTorch يستخدم التشغيل الذاتي القائم على الشريط. كل عملية ترتبط بـ "الشريط" خلال المرور الأمامي. الاتصال `.backward()`يُعيد التسجيل في العكس

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2 + 3 * x
z = y.sum()
z.backward()
print(x.grad)  # dz/dx = 2x + 3
```

ثلاثة قواعد للدراسة الذاتية:

1. فقط الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " الـ " " الـ " " " " الـ " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "`requires_grad=True`تراكم التدرج
2. الدرجات تتراكم بطبيعة الحال -- دعوة `optimizer.zero_grad()`قبل كل مرور خلفي
3. `torch.no_grad()`تعطيل تتبع التراجع (استخدام أثناء التقييم)

### nn.مودول

`nn.Module`هي الصف الأساسي لكل مكون للشبكة العصبية في PyTorch. لقد بنيت هذا التجريد بالفعل في الدروس 10. إصدار PyTorch يضيف تسجيل المعلمات تلقائي، اكتشاف الوحدة التجاعدية، إدارة الجهاز، وتسلسل الحالة.

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

عندما تعين`nn.Module`أو`nn.Parameter`كصفت في `__init__`"بيتورش" تسجلها تلقائياً`model.parameters()`هذا هو السبب في أنك لا تحتاج أبدا إلى جمع الوزن يدويا كما فعلت في الإطار الصغير.

أساسيات البناء:

| Module | What it does | Parameters |
|--------|-------------|------------|
| nn.Linear(in, out) | Wx + b | in*out + out |
| nn.Conv2d(in_ch, out_ch, k) | 2D convolution | in_ch*out_ch*k*k + out_ch |
| nn.BatchNorm1d(features) | Normalize activations | 2 * features |
| nn.Dropout(p) | Random zeroing | 0 |
| nn.ReLU() | max(0, x) | 0 |
| nn.GELU() | Gaussian error linear | 0 |
| nn.Embedding(vocab, dim) | Lookup table | vocab * dim |
| nn.LayerNorm(dim) | Per-sample normalization | 2 * dim |

### أداءات الخسارة ومحسنات

(بايتورش) تسلم نسخ جاهزة للإنتاج من كل ما بنيته

**Loss functions**(من`torch.nn`):

| Loss | Task | Input |
|------|------|-------|
| nn.MSELoss() | Regression | Any shape |
| nn.CrossEntropyLoss() | Multi-class classification | Logits (not softmax) |
| nn.BCEWithLogitsLoss() | Binary classification | Logits (not sigmoid) |
| nn.L1Loss() | Regression (robust) | Any shape |
| nn.CTCLoss() | Sequence alignment | Log probabilities |

ملاحظة:`CrossEntropyLoss`يجمع`LogSoftmax`+ `NLLLoss`إضافة المواد الخام، وليس المخرجات المضمونة. هذا خطأ شائع ينتج التراجع الخاطئ بصمت.

**Optimizers**(من`torch.optim`):

| Optimizer | When to use | Typical LR |
|-----------|-------------|-----------|
| SGD(params, lr, momentum) | CNNs, well-tuned pipelines | 0.01--0.1 |
| Adam(params, lr) | Default starting point | 1e-3 |
| AdamW(params, lr, weight_decay) | Transformers, fine-tuning | 1e-4--1e-3 |
| LBFGS(params) | Small-scale, second-order | 1.0 |

### حلقة التدريب

كل حلقة تدريبية من PyTorch تتبع نفس النمط الخمس خطوات.

```mermaid
sequenceDiagram
    participant D as DataLoader
    participant M as Model
    participant L as Loss fn
    participant O as Optimizer

    loop Each Epoch
        D->>M: batch = next(dataloader)
        M->>L: predictions = model(batch)
        L->>L: loss = criterion(predictions, targets)
        L->>M: loss.backward()
        O->>M: optimizer.step()
        O->>O: optimizer.zero_grad()
    end
```

النمط القنوني:

```python
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

خمسة خطوط داخل حلقة اللحظة، خمسة خطوط تدرب GPT-4، Diffusion مستقر، و LLaMA. الهندسة المعمارية تتغير. البيانات تتغير. هذه الخطوط الخمسة لا.

### مجموعة البيانات و DataLoader

(بيتورش)`Dataset`هو فئة تجريدية مع طريقتين: `__len__`و`__getitem__`. .`DataLoader`يلفها مع إعادة التجميع، التدفق، وتحميل البيانات متعددة العمليات.

```python
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
```

`num_workers=4`يخلق 4 عمليات لحمل البيانات بالتوازي بينما تتدرب GPU على الحزمة الحالية. على أحمال عمل مرتبطة بالقرص (الصور الكبيرة، الصوت) ، وهذا وحده يمكن أن تضاعف سرعة التدريب.

### تدريبات الجيبو

نقل النموذج إلى GPU:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

هذا يتحرك بشكل متكرر كل معايير ومضخة إلى جهاز التلفزيون. ثم يحرك كل دفعة أثناء التدريب:

```python
inputs, targets = inputs.to(device), targets.to(device)
```

**Mixed precision**يقلل من نصف استخدام الذاكرة ويضاعف من خلالها النمو على أجهزة المعالجة المعالجة المعالجة المعالجة الحديثة (A100، H100، RTX 4090) عن طريق التشغيل للأمام/الظهر في float16 مع الحفاظ على الأوزان الرئيسية في float32:

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, targets in loader:
    with autocast(device_type="cuda"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### مقارنة: Mini Framework vs PyTorch vs JAX

| Feature | Mini Framework (L10) | PyTorch | JAX |
|---------|---------------------|---------|-----|
| Autodiff | Manual backward() | Tape-based autograd | Functional transforms |
| Execution | Eager (Python loops) | Eager (C++ kernels) | Traced + JIT compiled |
| GPU support | No | Yes (CUDA, ROCm, MPS) | Yes (CUDA, TPU) |
| Speed (MNIST MLP) | ~300s/epoch | ~0.5s/epoch | ~0.3s/epoch |
| Module system | Custom Module class | nn.Module | Stateless functions (Flax/Equinox) |
| Debugging | print() | print(), pdb, breakpoint() | Harder (JIT tracing breaks print) |
| Ecosystem | None | Hugging Face, Lightning, timm | Flax, Optax, Orbax |
| Learning curve | You built it | Moderate | Steep (functional paradigm) |
| Production use | Toy problems | Meta, OpenAI, Anthropic, HF | Google DeepMind, Midjourney |

```figure
dropout-mask
```

## بناءها

(ميلف) ثلاث طبقات تدرب على (منست) باستخدام (بيتورش) البدائية فقط، لا يوجد غلفات عالية المستوى`torchvision.datasets`نحن ننزل ونحلل البيانات الخام بأنفسنا

### الخطوة الأولى: تحميل MNIST من الملفات الخام

تم إرسال MNIST إلى البيانات بأربعة ملفات: صور التدريب (60,000 × 28 × 28) ، وملفات التدريب، وملفات الاختبار (10,000 × 28 × 28) ، وملفات الاختبار. ننزلها ونحلل النموذج الثنائي.

```python
import torch
import torch.nn as nn
import struct
import gzip
import urllib.request
import os

def download_mnist(path="./mnist_data"):
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    os.makedirs(path, exist_ok=True)
    for f in files:
        filepath = os.path.join(path, f)
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(base_url + f, filepath)

def load_images(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        images = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        images = images.reshape(num, rows * cols).float() / 255.0
    return images

def load_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
        labels = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    return labels
```

### الخطوة الثانية: حدد النموذج

3-طبقة MLP: 784 -> 256 -> 128 -> 10. تنشيط ReLU. التخلي عن التنظيم. لا يوجد معيار لتحقيق البطولة.

```python
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)
```

الطبقة الخارجة تنتج 10 logits خام (واحد لكل رقم). لا softmax -- `CrossEntropyLoss`يتعامل مع ذلك داخلياً

عدد المعايير: 784*256 + 256 + 256*128 + 128 + 128*10 + 10 = 235,146. صغير بمعايير حديثة. GPT-2 صغير لديه 124M. هذا القطار في ثوان.

### الخطوة الثالثة: حلقة التدريب

النمط القنوني للأمام-الخسارة-الخطوات الخلفية.

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total
```

ملاحظة`torch.no_grad()`هذا يعطى التحرر الذاتي، مما يقلل من استهلاك الذاكرة ويتسارع استنتاج. بدون ذلك، يقوم PyTorch ببناء الرسم البياني الحاسوبي الذي لا تستخدمه.

### الخطوة الرابعة: قم بتجميع كل شيء

```python
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    download_mnist()
    train_images = load_images("./mnist_data/train-images-idx3-ubyte.gz")
    train_labels = load_labels("./mnist_data/train-labels-idx1-ubyte.gz")
    test_images = load_images("./mnist_data/t10k-images-idx3-ubyte.gz")
    test_labels = load_labels("./mnist_data/t10k-labels-idx1-ubyte.gz")

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )

    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Parameters: {num_params:,}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    print()

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        print(
            f"Epoch {epoch+1:2d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    torch.save(model.state_dict(), "mnist_mlp.pt")
    print(f"\nModel saved to mnist_mlp.pt")
    print(f"Final test accuracy: {test_acc:.4f}")
```

الناتج المتوقع بعد 10 فترات: ~ 97.8٪ دقة الاختبار. وقت التدريب على المعالجة المركزية: ~ 30 ثانية. على GPU: ~ 5 ثانية. على الإطار الصغير مع نفس الهندسة المعمارية: ~ 45 دقيقة.

## استخدمها

### مقارنة سريعة: Mini Framework vs PyTorch

| Mini Framework (Lesson 10) | PyTorch |
|---------------------------|---------|
| `model = Sequential(Linear(784, 256), ReLU(), ...)` | `model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), ...)` |
| `pred = model.forward(x)` | `pred = model(x)` |
| `optimizer.zero_grad()` | `optimizer.zero_grad()` |
| `grad = criterion.backward()` then `model.backward(grad)` | `loss.backward()` |
| `optimizer.step()` | `optimizer.step()` |
| No GPU | `model.to("cuda")` |
| Manual backward for every module | Autograd handles everything |

المقابلة متطابقة تقريباً، الفرق هو كل شيء تحت الغطاء

### نموذجات حفظ وتحميل

```python
torch.save(model.state_dict(), "model.pt")

model = MNISTModel()
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()
```

دائماً أنقذ`state_dict()`(قامة المعلمات) ، وليس كائن النموذج. حفظ كائن النموذج يستخدم البرتقال، الذي يكسر عندما تقوم بتعديل الرمز.

### تخطيط معدلات التعلم

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10
)
for epoch in range(10):
    train_one_epoch(model, train_loader, criterion, optimizer, device)
    scheduler.step()
```

تقوم بيتورش بنقل 15 خطة: StepLR، ExponentialLR، CosineAnnealingLR، OneCycleLR، ReduceLROnPlateau. جميعها متصلة بنفس واجهة التحسين.

## أرسله

هذا الدروس يُنتج اثنين من الأثاث:

- `outputs/prompt-pytorch-debugger.md`-- تحذير لتشخيص فشل في تدريب PyTorch الشائع
- `outputs/skill-pytorch-patterns.md`-- مرجع مهارات لنمطات تدريب PyTorch

## التمارين

1. **Add batch normalization.**إدراج`nn.BatchNorm1d`بعد كل طبقة خطية (قبل التشغيل). مقارنة دقة الاختبار وسرعة التدريب مقابل النسخة التي يتم إطلاقها فقط. يجب أن يصل معايير الحزمة إلى 98% + في فترات أقل.

2. **Implement a learning rate finder.**تدريب لمدة فترة مع زيادة معدلية للتعلم بشكل متسارع (من 1e-7 إلى 1.0). خسارة المساحة مقابل LR. يكون LR الأمثل قبل أن تبدأ الخسارة في التسلق. استخدم هذا لتحديد LR أفضل لنموذج MNIST.

3. **Port to GPU with mixed precision.**إضافة`torch.amp.autocast`و`GradScaler`على حلقة التدريب. قياس التدفق (عينات/ثانية) مع ودون دقة مختلطة على GPU. على A100، توقع ~ 2x سرعة.

4. **Build a custom Dataset.**قم بتنزيل Fashion-MNIST (المثل في النموذج مثل MNIST ولكن مع أدوات الملابس). تنفيذ `FashionMNISTDataset(Dataset)`الفصل مع`__getitem__`و`__len__`. تدريب نفس MLP ومقارنة الدقة. الموضة-MNIST هو أصعب -- توقع ~ 88٪ مقابل ~ 98٪.

5. **Replace Adam with SGD + momentum.**القطار مع`SGD(params, lr=0.01, momentum=0.9)`. مقارنة منحنى التقارب. ثم أضف`CosineAnnealingLR`و نرى ما إذا كانت (إس جي دي) ستقبض على (آدم) بحلول العصر العاشر

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Tensor | "A multi-dimensional array" | A typed, device-aware array with automatic differentiation support baked into every operation |
| Autograd | "Automatic backprop" | A tape-based system that records operations during forward pass, then replays them in reverse to compute exact gradients |
| nn.Module | "A layer" | The base class for any differentiable computation block -- registers parameters, supports nesting, handles train/eval modes |
| state_dict | "The model weights" | An OrderedDict mapping parameter names to tensors -- the portable, serializable representation of a trained model |
| .backward() | "Compute gradients" | Traverse the computational graph in reverse, computing and accumulating gradients for every leaf tensor with requires_grad=True |
| .to(device) | "Move to GPU" | Recursively transfer all parameters and buffers to the specified device (CPU, CUDA, MPS) |
| DataLoader | "The data pipeline" | An iterator that batches, shuffles, and optionally parallelizes data loading from a Dataset |
| Mixed precision | "Use float16" | Train with float16 forward/backward for speed while keeping float32 master weights for numerical stability |
| Eager execution | "Run it now" | Operations execute immediately when called, not deferred to a later compilation step -- the core design choice that differentiates PyTorch from TF 1.x |
| zero_grad | "Reset gradients" | Set all parameter gradients to zero before the next backward pass, since PyTorch accumulates gradients by default |

## المزيد من القراءة

- پاسكيه وآخرون، "بيتورش: أسلوب إمبراطي، مكتبة التعلم العميق عالي الأداء" (2019) -- الورقة الأصلية التي تشرح تعادلات تصميم بيتورش
- دروس بيتورش: "تعلم بيتورش مع الأمثلة" (https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) -- المسار الرسمي من الجهاز التنسوري إلى
- دليل تحديد أداء PyTorch (https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) -- الدقة المختلطة، وموظفي DataLoader، والذاكرة المثبتة، وغيرها من التحسينات الإنتاجية
- (هوراس هيه) ، "جعل التعلم العميق يذهب إلى (برر) "https://horace.io/brrr_intro.html) -- لماذا تدريب GPU سريع، مع استراتيجيات التحسين الخاصة بـ PyTorch
