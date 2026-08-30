# पाइटॉर्च का परिचय

> आप पिस्टन और क्रैंकशाफ्ट से इंजन बनाया है. अब सीखना कि सभी वास्तव में चलाता है.

**Type:** Build
**Languages:** Python
**Prerequisites:** Lesson 03.10 (Build Your Own Mini Framework)
**Time:** ~75 minutes

## सीखने के लक्ष्य

- PyTorch के nn.Module, nn.Sequential और autograd का उपयोग करके तंत्रिका नेटवर्क का निर्माण और प्रशिक्षण
- PyTorch tensors, GPU त्वरण, और मानक प्रशिक्षण लूप (शून्य_ग्रेड, आगे, हानि, पीछे, कदम) का उपयोग करें
- अपने स्क्रैच से अपने मिनी फ्रेमवर्क घटकों को उनके PyTorch समकक्षों में परिवर्तित करें
- प्रोफ़ाइल और एक ही कार्य पर अपने शुद्ध-पायथन फ्रेमवर्क और PyTorch के बीच प्रशिक्षण गति की तुलना

## समस्या

आपके पास एक काम करने वाला मिनी फ्रेमवर्क है। रैखिक परतें, ReLU, ड्रॉपआउट, बैच मानदंड, एडम, एक डेटा लोडर, एक प्रशिक्षण लूप। यह शुद्ध पायथन में सर्कल वर्गीकरण समस्या पर चार परत नेटवर्क को प्रशिक्षित करता है।

यह भी 500 गुना धीमी है PyTorch पर एक ही समस्या पर.

आपका मिनी फ्रेमवर्क एक समय में एक नमूना को निहित पायथन लूप के साथ संसाधित करता है। पायटॉर्च GPU पर चलने वाले अनुकूलित C ++ / CUDA कर्नेल में समान संचालन भेजता है। एक एकल NVIDIA A100 पर, पायटॉर्च लगभग 6 घंटे में ImageNet पर ResNet-50 (25.6M पैरामीटर) को प्रशिक्षित करता है (1.28M छवियां) । आपके फ्रेमवर्क में लगभग 3,000 घंटे लगेंगे एक ही कार्य पर - अगर यह पहले मेमोरी से बाहर नहीं निकलता है।

गति एकमात्र अंतर नहीं है. आपके फ्रेमवर्क में कोई जीपीयू समर्थन नहीं है. कोई स्वचालित भेदभाव नहीं है - आपने प्रत्येक मॉड्यूल के लिए हाथ से पीछे लिखा है। कोई क्रमबद्धता नहीं है। कोई वितरित प्रशिक्षण नहीं है। कोई मिश्रित सटीकता नहीं है। प्रिंट स्टेटमेंट के बिना ग्रेडिएंट प्रवाह को डिबग करने का कोई तरीका नहीं है।

PyTorch इन सभी अंतराल को भरता है। और यह वही मानसिक मॉडल रखता है जिसे आपने पहले ही बनाया हैः मॉड्यूल, आगे(), पैरामीटर(), पीछे(), अनुकूलक। चरण() । अवधारणाएं एक-एक से स्थानांतरित होती हैं। वाक्यविन्यास लगभग समान है। अंतर यह है कि PyTorch एक दशक के सिस्टम इंजीनियरिंग को उसी इंटरफ़ेस के पीछे लपेटता है जिसे आपने खरोंच से डिज़ाइन किया है।

## अवधारणा

### क्यों पायटॉर्च ने जीत हासिल की

2015 में, TensorFlow ने आपको किसी भी चीज़ को चलाने से पहले एक स्थैतिक गणना ग्राफ को परिभाषित करने की आवश्यकता थी। आपने ग्राफ बनाया, इसे संकलित किया, फिर इसे डेटा के माध्यम से खिलाया। डिबगिंग का मतलब था ग्राफ विज़ुअलाइज़ेशन पर ध्यान देना। वास्तुकला को बदलना ग्राफ को खरोंच से पुनर्निर्माण करना था।

PyTorch 2017 में एक अलग दर्शन के साथ लॉन्च किया गया था: उत्सुक निष्पादन. आप पायथन लिखते हैं। यह तुरंत चलता है। `y = model(x)`वास्तव में अब y की गणना करता है, "एक ग्राफ में एक नोड जोड़ें जो बाद में y की गणना करेगा।" इसका मतलब है कि मानक पायथन डिबगिंग उपकरण काम किया। प्रिंट() काम किया। pdb काम किया। यदि / अन्य आपके फॉरवर्ड पास में काम किया।

2020 तक, बाजार ने बात की थी। एमएल शोध पत्रों में पायटॉर्च की हिस्सेदारी 7% (2017) से बढ़कर 75% (2022) हो गई। मेटा, गूगल डीपमाइंड, ओपनएआई, मानव विज्ञान और गले लगाना चेहरा सभी पायटॉर्च का उपयोग अपने प्राथमिक ढांचे के रूप में करते हैं। TensorFlow 2.x ने प्रतिक्रिया में उत्सुक निष्पादन को अपनाया - चुपचाप स्वीकार करना कि पायटॉर्च का डिजाइन सही था।

सबक: डेवलपर अनुभव यौगिकों. एक फ्रेमवर्क जो 10% धीमा है लेकिन 50% तेजी से डिबग करने के लिए हर बार जीतता है.

### टेंसर

एक टेन्सर तीन महत्वपूर्ण गुणों के साथ एक बहुआयामी सरणी हैः आकार, डीटाइप और डिवाइस।

```python
import torch

x = torch.zeros(3, 4)           # shape: (3, 4), dtype: float32, device: cpu
x = torch.randn(2, 3, 224, 224) # batch of 2 RGB images, 224x224
x = torch.tensor([1, 2, 3])     # from a Python list
```

**Shape**एक स्केलर आकार (), एक वेक्टर (n,), एक मैट्रिक्स (m, n), एक बैच छवियों (बैच, चैनल, ऊंचाई, चौड़ाई) है।

**Dtype**सटीकता और स्मृति को नियंत्रित करता है।

| dtype | Bits | Range | Use case |
|-------|------|-------|----------|
| float32 | 32 | ~7 decimal digits | Default training |
| float16 | 16 | ~3.3 decimal digits | Mixed precision |
| bfloat16 | 16 | Same range as float32, less precision | LLM training |
| int8 | 8 | -128 to 127 | Quantized inference |

**Device**यह निर्धारित करता है कि गणना कहाँ होती है।

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(3, 4, device=device)
x = x.to("cuda")
x = x.cpu()
```

प्रत्येक ऑपरेशन एक ही डिवाइस पर सभी Tensors की आवश्यकता होती है. यह है # 1 PyTorch त्रुटि शुरुआती हिटः `RuntimeError: Expected all tensors to be on the same device`गणना से पहले एक ही डिवाइस पर सब कुछ ले जाकर इसे ठीक करें।

**Reshaping**यह निरंतर समय है -- यह मेटाडेटा को बदलता है, डेटा को नहीं।

```python
x = torch.randn(2, 3, 4)
x.view(2, 12)      # reshape to (2, 12) -- must be contiguous
x.reshape(6, 4)    # reshape to (6, 4) -- works always
x.permute(2, 0, 1) # reorder dimensions
x.unsqueeze(0)     # add dimension: (1, 2, 3, 4)
x.squeeze()        # remove size-1 dimensions
```

### ऑटोग्राड

आपके मिनी फ्रेमवर्क में आपको प्रत्येक मॉड्यूल के लिए पीछे की ओर लागू करने की आवश्यकता होती है। PyTorch नहीं करता है। यह एक निर्देशित एसाइक्लिक ग्राफ (गणितीय ग्राफ) में टेन्सर पर प्रत्येक ऑपरेशन रिकॉर्ड करता है और फिर स्वचालित रूप से ग्रेडिएंट्स की गणना करने के लिए उस ग्राफ को उलटता है।

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

आपके फ्रेमवर्क से मुख्य अंतरः PyTorch टेप आधारित ऑटोडिफ का उपयोग करता है. प्रत्येक ऑपरेशन आगे के पास के दौरान एक "टेप" से जुड़ता है। कॉल `.backward()`टेप को उल्टा रीप्ले करता है।

```python
x = torch.randn(3, requires_grad=True)
y = x ** 2 + 3 * x
z = y.sum()
z.backward()
print(x.grad)  # dz/dx = 2x + 3
```

ऑटोग्रेड के तीन नियमः

1. केवल पत्ती टेन्सर के साथ `requires_grad=True`जमा gradients
2. ग्रेडिएंट डिफ़ॉल्ट रूप से जमा होते हैं -- कॉल `optimizer.zero_grad()`प्रत्येक पीछे की ओर जाने से पहले
3. `torch.no_grad()`ग्रेडिएंट ट्रैकिंग को अक्षम करता है (मूल्यांकन के दौरान उपयोग)

### nn.मॉड्यूल

`nn.Module`PyTorch में प्रत्येक तंत्रिका नेटवर्क घटक के लिए आधार वर्ग है. आपने पहले ही इस अमूर्तता को पाठ 10 में बनाया है। PyTorch के संस्करण में स्वचालित पैरामीटर पंजीकरण, पुनरावर्ती मॉड्यूल खोज, डिवाइस प्रबंधन, और राज्य डिक्ट सिरीलाइज़ेशन जोड़ता है।

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

जब आप एक असाइनमेंट`nn.Module`या `nn.Parameter` में एक विशेषता के रूप में`__init__`, PyTorch स्वचालित रूप से इसे पंजीकृत करता है।`model.parameters()`इस कारण से आपको कभी भी मैन्युअल रूप से वजन एकत्र करने की आवश्यकता नहीं है जैसा कि आपने मिनी फ्रेमवर्क में किया था।

मुख्य निर्माण सामग्रीः

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

### हानि कार्य और अनुकूलक

PyTorch उत्पादन के लिए तैयार संस्करणों जहाजों आप बनाया सब कुछ.

**Loss functions**(से `torch.nn`):

| Loss | Task | Input |
|------|------|-------|
| nn.MSELoss() | Regression | Any shape |
| nn.CrossEntropyLoss() | Multi-class classification | Logits (not softmax) |
| nn.BCEWithLogitsLoss() | Binary classification | Logits (not sigmoid) |
| nn.L1Loss() | Regression (robust) | Any shape |
| nn.CTCLoss() | Sequence alignment | Log probabilities |

नोटः `CrossEntropyLoss`संयोजन `LogSoftmax`+ `NLLLoss`आंतरिक रूप से. कच्चे लॉजिट पास करें, न कि सॉफ्टमैक्स आउटपुट. यह एक आम गलती है जो चुपचाप गलत ग्रेडिएंट का उत्पादन करती है।

**Optimizers**(से `torch.optim`):

| Optimizer | When to use | Typical LR |
|-----------|-------------|-----------|
| SGD(params, lr, momentum) | CNNs, well-tuned pipelines | 0.01--0.1 |
| Adam(params, lr) | Default starting point | 1e-3 |
| AdamW(params, lr, weight_decay) | Transformers, fine-tuning | 1e-4--1e-3 |
| LBFGS(params) | Small-scale, second-order | 1.0 |

### प्रशिक्षण चक्र

प्रत्येक PyTorch प्रशिक्षण लूप एक ही 5-चरण पैटर्न का पालन करता है. आप पहले से ही पाठ 10 से यह पता है.

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

कैनोनिक पैटर्नः

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

बैच लूप के अंदर पांच पंक्तियाँ जो जीपीटी-4, स्थिर विसारण और एलएएमए को प्रशिक्षित करती हैं। वास्तुकला बदलती है। डेटा बदलता है। ये पांच पंक्तियां नहीं करती हैं।

### डेटासेट और डेटा लोडर

पायटॉर्च की `Dataset`दो विधियों के साथ एक अमूर्त वर्ग हैः `__len__`और `__getitem__`. .`DataLoader`बैचिंग, मिक्सिंग और मल्टी-प्रोसेस डेटा लोडिंग के साथ इसे लपेटता है।

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

`num_workers=4`वर्तमान बैच पर GPU ट्रेन करते समय समानांतर में डेटा लोड करने के लिए 4 प्रक्रियाओं का उत्पादन करता है। डिस्क-बाउंड वर्कलोड (बड़े चित्र, ऑडियो) पर, अकेले यह प्रशिक्षण गति को दोगुना कर सकता है।

### जीपीयू प्रशिक्षण

GPU में मॉडल स्थानांतरित करनाः

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

यह प्रत्येक पैरामीटर और बफर को जीपीयू पर पुनरावर्ती रूप से स्थानांतरित करता है। फिर प्रशिक्षण के दौरान प्रत्येक बैच को स्थानांतरित करता हैः

```python
inputs, targets = inputs.to(device), targets.to(device)
```

**Mixed precision**फ्लोट16 में आगे/पीछे चलकर आधुनिक जीपीयू (A100, H100, RTX 4090) पर मेमोरी उपयोग को आधा और मुख्य भारों को फ्लोट32 में बनाए रखकर पारगमन को दोगुना करता हैः

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

### तुलनाः मिनी फ्रेमवर्क बनाम पायटॉर्च बनाम जैक्स

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

## इसे बनाओ

एक 3-परत MLP MNIST पर प्रशिक्षित केवल PyTorch आदिम प्रयोग कर. कोई उच्च स्तर के wrappers. नहीं.`torchvision.datasets`हम खुद कच्चे डेटा डाउनलोड और विश्लेषण करते हैं।

### चरण 1: कच्चे फ़ाइलों से MNIST लोड करें

MNIST 4 gziped फ़ाइलों के रूप में जहाजः प्रशिक्षण छवियों (60,000 x 28 x 28), प्रशिक्षण लेबल, परीक्षण छवियों (10,000 x 28 x 28), परीक्षण लेबल। हम उन्हें डाउनलोड और द्विआधारी प्रारूप विश्लेषण।

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

### चरण 2: मॉडल को परिभाषित करें

एक 3-परत MLP: 784 -> 256 -> 128 -> 10. ReLU सक्रियण. नियमितता के लिए ड्रॉप। यह सरल रखने के लिए कोई बैच मानक नहीं है।

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

आउटपुट परत 10 कच्चे लॉगिट (एक प्रति अंक) उत्पन्न करता है। कोई softmax -- `CrossEntropyLoss`आंतरिक रूप से संभालता है।

पैरामीटर की संख्याः 784*256 + 256 + 256*128 + 128 + 128*10 + 10 = 235,146. आधुनिक मानकों के अनुसार छोटा। GPT-2 छोटा 124M है। यह सेकंड में ट्रेन करता है।

### चरण 3: प्रशिक्षण लूप

आगे-हारे-पछाड़-चरण पैटर्न।

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

नोट `torch.no_grad()`यह ऑटोग्रेड को निष्क्रिय करता है, मेमोरी उपयोग को कम करता है और निष्कर्ष को तेज करता है। इसके बिना, PyTorch एक गणना ग्राफ बनाता है आप कभी उपयोग नहीं करते हैं।

### चरण 4: सब कुछ एक साथ जोड़ें

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

10 युगों के बाद अपेक्षित आउटपुटः ~ 97.8% परीक्षण सटीकता। सीपीयू पर प्रशिक्षण समयः ~ 30 सेकंड। जीपीयू परः ~ 5 सेकंड। एक ही वास्तुकला के साथ अपने मिनी फ्रेमवर्क परः ~ 45 मिनट।

## इसका प्रयोग करें

### त्वरित तुलनाः मिनी फ्रेमवर्क बनाम पायटॉर्च

| Mini Framework (Lesson 10) | PyTorch |
|---------------------------|---------|
| `model = Sequential(Linear(784, 256), ReLU(), ...)` | `model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), ...)` |
| `pred = model.forward(x)` | `pred = model(x)` |
| `optimizer.zero_grad()` | `optimizer.zero_grad()` |
| `grad = criterion.backward()` then `model.backward(grad)` | `loss.backward()` |
| `optimizer.step()` | `optimizer.step()` |
| No GPU | `model.to("cuda")` |
| Manual backward for every module | Autograd handles everything |

इंटरफ़ेस लगभग समान है. अंतर है सब कुछ हुड के नीचे है.

### बचत और लोड मॉडल

```python
torch.save(model.state_dict(), "model.pt")

model = MNISTModel()
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()
```

हमेशा बचत करें`state_dict()`(पैरामीटर शब्दकोश), मॉडल ऑब्जेक्ट नहीं। मॉडल ऑब्जेक्ट को सहेजने के लिए मक्खन का उपयोग किया जाता है, जो जब आप कोड को रीफ़ैक्टर करते हैं तो टूट जाता है। राज्य डिक्ट पोर्टेबल हैं।

### सीखने की दरों की योजना

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10
)
for epoch in range(10):
    train_one_epoch(model, train_loader, criterion, optimizer, device)
    scheduler.step()
```

पायटॉर्च 15 से अधिक शेड्यूलर भेजता हैः स्टेपएलआर, एक्सपोनेंशियलएलआर, कॉसीनएनेलिंगएलआर, वनसाइकलएलआर, रेड्यूसएलआरओएनप्लेट। सभी एक ही अनुकूलक इंटरफ़ेस में प्लग करते हैं।

## इसे भेजें

इस पाठ में दो कलाकृतियां उत्पन्न होती हैंः

- `outputs/prompt-pytorch-debugger.md`-- सामान्य PyTorch प्रशिक्षण विफलताओं का निदान करने के लिए एक संकेत
- `outputs/skill-pytorch-patterns.md`-- PyTorch प्रशिक्षण पैटर्न के लिए कौशल संदर्भ

## व्यायाम

1. **Add batch normalization.**सम्मिलित करें `nn.BatchNorm1d`प्रत्येक रैखिक परत के बाद (सक्रियता से पहले) परीक्षण सटीकता और प्रशिक्षण गति की तुलना केवल ड्रॉप-आउट संस्करण के साथ करें। बैच मानदंड कम समय में 98%+ तक पहुंचना चाहिए।

2. **Implement a learning rate finder.**एक युग के लिए प्रशिक्षण के साथ तेजी से बढ़ रही सीखने की दर (1e-7 से 1.0) । प्लॉट हानि बनाम LR। इष्टतम LR नुकसान चढ़ने से ठीक पहले है। MNIST मॉडल के लिए बेहतर LR चुनने के लिए इसका उपयोग करें।

3. **Port to GPU with mixed precision.**जोड़ें `torch.amp.autocast`और `GradScaler`एक ए 100 पर, ~ 2x गति की उम्मीद करें।

4. **Build a custom Dataset.**फैशन-MNIST (MNIST के समान प्रारूप लेकिन कपड़ों के साथ) डाउनलोड करें।`FashionMNISTDataset(Dataset)`कक्षा के साथ `__getitem__`और `__len__`. उसी MLP को अभ्यास करें और सटीकता की तुलना करें. फैशन-MNIST कठिन है - ~ 88% बनाम ~ 98% की उम्मीद करें।

5. **Replace Adam with SGD + momentum.**`SGD(params, lr=0.01, momentum=0.9)`. अभिसरण वक्रों की तुलना करें. फिर एक जोड़ें`CosineAnnealingLR`अनुसूची और देखें कि क्या SGD युग 10 द्वारा एडम को पकड़ने के लिए.

## प्रमुख शर्तें

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

## आगे पढ़ना

- पाज़्के एट अल., "पायटॉर्चः एक अनिवार्य शैली, उच्च प्रदर्शन गहरी शिक्षा पुस्तकालय" (2019) -- मूल पेपर जो पायटॉर्च के डिजाइन व्यापारों की व्याख्या करता है
- PyTorch Tutorials: "उदाहरणों के साथ PyTorch सीखना" (https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) -- टेन्सर से nn.Module तक का आधिकारिक मार्ग
- PyTorch प्रदर्शन ट्यूनिंग गाइड (https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) -- मिश्रित परिशुद्धता, डेटा लोडर श्रमिक, पिन मेमोरी, और अन्य उत्पादन अनुकूलन
- होरेस हे, "गहन सीखने को ब्रोरर बनाना" (https://horace.io/brrr_intro.html) -- क्यों GPU प्रशिक्षण तेजी से है, PyTorch विशिष्ट अनुकूलन रणनीतियों के साथ
