# Resim sınıflandırması

> Bir sınıflandırıcı, pikselden sınıflardaki olasılık dağılımına kadar bir fonksiyon.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 Lesson 09 (Model Evaluation), Phase 3 Lesson 10 (Mini Framework), Phase 4 Lesson 03 (CNNs)
**Time:** ~75 minutes

## Öğrenme Hedefleri

- CIFAR-10 üzerinde sonundan sonuna kadar bir görüntü sınıflandırma boru hattı oluşturmak: veri kümesi, artırma, model, eğitim döngüsü, değerlendirme
- Her bileşenin rolünü açıklayın (veriler yükleyici, kayb, optimizer, programlayıcı, artırma) ve kayb eğrisinde bunların herhangi birinin nasıl kırıldığını tahmin edin
- Karıştırma, kesme ve etiket düzeltme uygulaması sıfırdan başlayın ve her birinin eklenmeye değer olduğunu haklı çıkarın
- Toplam doğruluğundan daha fazla veri kümesi ve model hatalarını teşhis etmek için bir karıştırma matrisi ve sınıf başına doğru/içini geri çeken tablo okuyun

## Sorun

Gönderen her görüntü görevi bir düzeyde görüntü sınıflandırmasına düşürülür. Deteksiyon bölgeleri sınıflandırır. Segmentasyon pikselleri sınıflandırır. Arama sınıfı centroidlere benzerlik göstererek sıralamaktadır. sınıflandırma doğru elde etmek  veri kümesi döngüsü, artırma politikası, kaybı, değerlendirme  aşamada diğer tüm görevlere aktarılan beceri.

Çoğu sınıflandırma hataları modelde yok. Onlar bir boru hattında yaşıyorlar: bozuk bir normallaşma, bir düzenlenmemiş eğitim kümesi, etiketleri çarpıtan büyütme, eğitim verileri ile kirlenmiş bir doğrulama bölümü, çağ 30'dan sonra sessizce farklılaşan bir öğrenme oranı. Doğru bir ayarla CIFAR-10'da %93'e ulaşan bir CNN, genellikle kırık bir ayarla %70-75 puan alır ve kayıp eğri her zaman makul görünür.

Bu ders tüm boru hattını el ile kablolar böylece her parça kontrol edilebilir.`torchvision.datasets`Bu bir böcek saklayabilir.

## Anlaşım

### Sınıflandırma boru hattı

```mermaid
flowchart LR
    A["Dataset<br/>(images + labels)"] --> B["Augment<br/>(random transforms)"]
    B --> C["Normalise<br/>(mean/std)"]
    C --> D["DataLoader<br/>(batch + shuffle)"]
    D --> E["Model<br/>(CNN)"]
    E --> F["Logits<br/>(N, C)"]
    F --> G["Cross-entropy loss"]
    F --> H["Argmax<br/>at eval"]
    G --> I["Backward"]
    I --> J["Optimizer step"]
    J --> K["Scheduler step"]
    K --> E

    style A fill:#dbeafe,stroke:#2563eb
    style E fill:#fef3c7,stroke:#d97706
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#dcfce7,stroke:#16a34a
```

Bu döngünün her satırı bir böcek yaşayabileceği bir yer.`model(x).softmax()`Kayıp sessizce yanlış gradiyenti hesaplamadan önce. artışlar sadece girişlere uygulanır, etiketlere değil  ikisini de karıştıran karıştırmayı hariç. `optimizer.zero_grad()`Bu hatalar, öğrenme eğriğini bir hata yapmadan düzeltir.

### Çarşı entropi, logit ve softmax

Bir sınıflandırıcı üretir `C`Logit olarak adlandırılan bir görüntü başına sayı. Softmax uygulaması onları olasılık dağılımına dönüştürür:

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

Çarşı entropi doğru sınıfın negatif log olasılığını ölçer:

```
CE(z, y) = -log( softmax(z)_y )
        = -z_y + log( sum_j exp(z_j) )
```

Sağdaki form sayısal olarak sabit olan (log-sum-exp) PyTorch'ın `nn.CrossEntropyLoss`softmax + NLL'i bir çalışmada birleştirir ve doğrudan çiğ logitleri alır. softmax'ı önce kendiniz uygulayarak neredeyse her zaman bir hata  log(softmax(softmax(z)) hesaplarsınız), anlamsız bir miktar.

### Neden artış işe yarıyor?

Bir CNN'in çeviri için indüktif bir önyargısı vardır (vez paylaşımından), ancak ürünlere, atışlara, renk gerginliğine veya okluksiyona herhangi bir değişim yoktur. Bu değişimleri öğretmenin tek yolu, bunları uygulayan pikselleri göstermektir. Eğitim sırasında her rastgele dönüşüm: "bu iki görüntü aynı etiketlere sahiptir; farkı görmezden gelen özellikleri öğrenin".

```
Original crop:  "dog facing left"
Flip:           "dog facing right"       <- same label, different pixels
Rotate(+15):    "dog, slight tilt"
Colour jitter:  "dog in warmer light"
RandomErasing:  "dog with patch missing"
```

Kural: büyütme etiketini korumalıdır. Bir rakamdaki kesim ve dönüşüm "6" 'ı "9'e çevirir; bu veri kümesi için daha küçük dönüş aralıkları kullanır ve rakamlara özgü invariansaları saygı gösteren büyütmeleri seçersiniz.

### Karıştırma ve kesme

Normal artış pikselleri dönüştürür ama etiketleri tek sıcak tutar.**Mixup**ve **cutmix**Her ikisini de araştıracak şekilde bunu çözebilirsiniz.

```
Mixup:
  lambda ~ Beta(a, a)
  x = lambda * x_i + (1 - lambda) * x_j
  y = lambda * y_i + (1 - lambda) * y_j

Cutmix:
  paste a random rectangle of x_j into x_i
  y = area-weighted mix of y_i and y_j
```

Neden işe yarıyor: model, dikenli bir-sıcak hedefleri hatırlamayı bırakır ve sınıflar arasında interpolasyon yapmayı öğrenir. Eğitim kaybı artıyor, test doğruluğu artıyor. Bu, herhangi bir sınıflandırıcı için en ucuz dayanıklılık yükseltmesidir.

### Etiket düzeltme

Karışık bir kuzen.`[0, 0, 1, 0, 0]`, karşı tren`[eps/C, eps/C, 1-eps, eps/C, eps/C]`Küçük bir şey için .`eps`Bu, modelin keyfi keskinliklerle keskin bir şekilde keskin bir şekilde üretmesini engeller ve kalibrasyonu neredeyse hiç bir maliyetle iyileştirir.`nn.CrossEntropyLoss(label_smoothing=0.1)`PyTorch 1.10'dan beri.

### Düzgünlüğü aşan değerlendirme

Toplam doğruluk dengesizliği saklar. 90-10 ikili sınıflandırıcı her zaman çoğunluk sınıfının puanını tahmin eder.

- **Per-class accuracy** sınıf başına bir sayı; hemen düşük performanslı kategoriler ortaya çıkar.
- **Confusion matrix** C x C çizgi i col j = sınıf j olarak öngörülen gerçek sınıfın sayısı; diyagonal doğru, diyagonal dışı diyagonallar modelinizin yaşadığı yerdir.
- **Top-1 / Top-5** doğru sınıfın en iyi 1 veya en iyi 5 tahminte olup olmadığını; Top-5'ün ImageNet için önemli olduğu "Norwich terrier" vs "Norfolk terrier" gibi sınıflar gerçekten belirsiz olduğu için.
- **Calibration (ECE)** 0.8 güven tahminleri zamanın %80'inde doğru mu? Modern ağlar sistematik olarak aşırı güvenlidir; sıcaklık ölçeklendirilmesi veya etiket düzeltmesi ile düzeltin.

```figure
receptive-field
```

## Yapın

### Adım 1: Determinizm sentetik veri kümesi

CIFAR-10 disk üzerinde yaşıyor. Bu dersi tekrarlanabilir ve hızlı yapmak için modelin öğrenmesi gereken sınıf-specifik yapısı olan CIFAR  32x32 RGB görüntülerine benzeyen sentetik bir veri kümesi oluşturduk.

```python
import numpy as np
import torch
from torch.utils.data import Dataset


def synthetic_cifar(num_per_class=1000, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    Y = []
    for c in range(num_classes):
        centre = rng.uniform(0, 1, (3,))
        freq = 2 + c
        for _ in range(num_per_class):
            yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
            r = np.sin(xx * freq) * 0.5 + centre[0]
            g = np.cos(yy * freq) * 0.5 + centre[1]
            b = (xx + yy) * 0.5 * centre[2]
            img = np.stack([r, g, b], axis=-1)
            img += rng.normal(0, 0.08, img.shape)
            img = np.clip(img, 0, 1)
            X.append(img.astype(np.float32))
            Y.append(c)
    X = np.stack(X)
    Y = np.array(Y)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


class ArrayDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, int(self.Y[i])
```

Her sınıf kendi renk paleti ve frekans kalıbı, ayrıca modelin pixelleri ezberlemek yerine sinyali öğrenmesini zorlamak için Gaussian gürültüsü elde eder.

### Adım 2: Normalleştirme ve artırma

Her görme borusunun sahip olduğu iki dönüşüm.

```python
def standardize(mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    def _fn(img):
        return (img - mean) / std
    return _fn


def random_hflip(p=0.5):
    def _fn(img):
        if np.random.random() < p:
            return img[:, ::-1, :].copy()
        return img
    return _fn


def random_crop(pad=4):
    def _fn(img):
        h, w = img.shape[:2]
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = np.random.randint(0, 2 * pad)
        x = np.random.randint(0, 2 * pad)
        return padded[y:y + h, x:x + w, :]
    return _fn


def compose(*fns):
    def _fn(img):
        for fn in fns:
            img = fn(img)
        return img
    return _fn
```

Yemekten önce yansıtacak bir çubuğu, sıfır çubuğu değil, çünkü siyah sınırlar, modelin önemsiz bir şekilde görmezden gelmeyi öğrendiği bir sinyal.

### Adım 3: Karıştırma

Eğitim aşamasının içinde iki görüntü ve iki etiket karıştırır.

```python
def mixup_batch(x, y, num_classes, alpha=0.2):
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, num_classes).float()
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_onehot = torch.nn.functional.one_hot(y, num_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mixed, y_mixed


def soft_cross_entropy(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
```

`soft_cross_entropy`Hedef tam olarak bir sıcaktan sonra, normal bir sıcaktan aşağı düşer.

### 4. Adım: Eğitim döngüsü

Tam tarif: verileri bir kez geçmek, her partiye bir kez gradientler, zamanlama bir kez adım atmak.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, device, num_classes, use_mixup=True):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x_m, y_soft = mixup_batch(x, y, num_classes)
            logits = model(x_m)
            loss = soft_cross_entropy(logits, y_soft)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        # Training accuracy vs the un-mixed labels `y` is only an approximation
        # when mixup is on (the model saw soft targets, not y). Treat it as a
        # rough progress signal; rely on val accuracy for real performance.
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        for t, p in zip(y.cpu(), pred.cpu()):
            cm[t, p] += 1
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return loss_sum / total, correct / total, cm
```

Her antrenman döngüsünü yazdığında kontrol ettiğin beş değişkenlik:

1. `model.train()`Eğitimden önce,`model.eval()`değerlendirme yapmadan önce  düşüş ve seri norm davranışlarını tersine çevirir.
2. `.zero_grad()`Daha önce`.backward()`- Evet .
3. `.item()`Eğer bir hesaplama grafikini canlı tutmak için hiçbir şey yok.
4. `@torch.no_grad()`değerlendirme sırasında  hafıza ve zaman tasarrufu sağlar, ince kazaları önler.
5. Argmax çiğ logitlere karşı, softmax değil  aynı sonuç, bir operasyon daha az.

### Adım 5: Bir araya getirin

Kullanın `TinyResNet`Önceki dersden, birkaç dönem için eğitim, değerlendirme.

```python
from main import synthetic_cifar, ArrayDataset
from main import standardize, random_hflip, random_crop, compose
from main import mixup_batch, soft_cross_entropy
from main import train_one_epoch, evaluate
# TinyResNet comes from the previous lesson (03-cnns-lenet-to-resnet).
# Adjust the import path to wherever you stored the previous lesson's code.
from cnns_lenet_to_resnet import TinyResNet  # example placeholder

X, Y = synthetic_cifar(num_per_class=500)
split = int(0.9 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
train_tf = compose(random_hflip(), random_crop(pad=4), standardize(mean, std))
eval_tf = standardize(mean, std)

train_ds = ArrayDataset(X_train, Y_train, transform=train_tf)
val_ds = ArrayDataset(X_val, Y_val, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyResNet(num_classes=10).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, 10, use_mixup=True)
    va_loss, va_acc, _ = evaluate(model, val_loader, device, 10)
    scheduler.step()
    print(f"epoch {epoch:2d}  lr {scheduler.get_last_lr()[0]:.4f}  "
          f"train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
```

Sentetik veri kümesi üzerinde, bu beş dönem içinde neredeyse mükemmel bir doğrulama doğruluğuna ulaşır, bu da noktayı oluşturur: boru hattı doğru, model neyi öğrenebilir. Veri kümesini gerçek CIFAR-10 için değiştirin ve aynı döngü trenleri değişikliksiz olarak ~ 90%'e ulaşır.

### Adım 6: Kafas karışıklığı matrisi okuyun

Sadece doğruluk, modelin nerede başarısız olduğunu asla söylemez.

```python
def print_confusion(cm, labels=None):
    c = cm.shape[0]
    labels = labels or [str(i) for i in range(c)]
    print(f"{'':>6}" + "".join(f"{l:>5}" for l in labels))
    for i in range(c):
        row = cm[i].tolist()
        print(f"{labels[i]:>6}" + "".join(f"{v:>5}" for v in row))
    print()
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    prec = tp / (tp + fp).clamp_min(1)
    rec = tp / (tp + fn).clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
    for i in range(c):
        print(f"{labels[i]:>6}  prec {prec[i]:.3f}  rec {rec[i]:.3f}  f1 {f1[i]:.3f}")

_, _, cm = evaluate(model, val_loader, device, 10)
print_confusion(cm)
```

Satırlar gerçek sınıflardır, sütunlar tahminlerdir. 3 ve 5 sınıflar arasında diyagonal dışı sayılar bir kümesi model bu iki şeyi karıştırır ve size hedeflenen veri toplama veya sınıf-özel bir artış için bir başlangıç noktası verir.

## Kullan

`torchvision`Gerçek CIFAR-10 için tüm boru hattı dört satır ve bir eğitim döngüsü.

```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
train_tf = Compose([
    RandomCrop(32, padding=4, padding_mode="reflect"),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean, std),
])
eval_tf = Compose([ToTensor(), Normalize(mean, std)])

train_ds = CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
val_ds   = CIFAR10(root="./data", train=False, download=True, transform=eval_tf)
```

Dikkat edilmesi gereken iki şey var: ortalama/std **dataset-specific** CIFAR-10 eğitim kümesi üzerinde hesaplanmıştır, ImageNet değil  ve refleks padı topluluk-devay ürün politikasıdır.

## Gönder

Bu ders şunları ortaya çıkarır:

- `outputs/prompt-classifier-pipeline-auditor.md` yukarıdaki beş değişken için bir eğitim senaryosunu denetleyen ve ilk ihlal ortaya çıkan bir istek.
- `outputs/skill-classification-diagnostics.md` Bir karışıklık matrisi ve sınıf isimlerinin bir listesine göre sınıf başına başarısızlıkları özetleyen ve en etkili tek düzeltmeyi öneren bir beceri.

## Egzersizler

1. **(Easy)**Aynı modelin sentez veriler kümesinde beş dönem boyunca karışıklık ve karışıklık olmadan çalıştırılmasını sağlayın.
2. **(Medium)**Kürtme  her eğitim görüntüsünde rastgele 8x8 kare sıfırlayın  ve bir ablasyon vs hiçbir artış, hflip+crop, hflip+crop+cutout, hflip+crop+mixup çalıştırın.
3. **(Hard)**CIFAR-100 borusunu oluşturun (100 sınıf, aynı giriş boyutu) ve ResNet-34 eğitimini yayınlanan doğruluğun % 1'i içinde yeniden üretin. Ekstra: üç öğrenme oranını ve iki ağırlık kaybını tarayın, yerel bir CSV'ye giriş yapın, son karışıklık-matris-üstün karışıklık tablounu oluşturun.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Logits | "Raw outputs" | The pre-softmax vector of C numbers per image; cross-entropy expects these, not softmaxed values |
| Cross-entropy | "The loss" | Negative log-probability of the correct class; combines log-softmax and NLL in one stable op |
| DataLoader | "The batcher" | Wraps a dataset with shuffling, batching, and (optional) multi-worker loading; gets blamed for half of training bugs |
| Augmentation | "Random transforms" | Any pixel-level transform at training time that preserves the label; teaches invariances the CNN does not have natively |
| Mixup / Cutmix | "Mix two images" | Blend both inputs and labels so the classifier learns smooth interpolations instead of hard boundaries |
| Label smoothing | "Softer targets" | Replace one-hot with (1-eps, eps/(C-1), ...); improves calibration and slightly boosts accuracy |
| Top-k accuracy | "Top-5" | The correct class is in the k highest-probability predictions; used on datasets with genuinely ambiguous classes |
| Confusion matrix | "Where errors live" | C x C table where entry (i, j) counts images of true class i predicted as j; diagonal is right, off-diagonal tells you what to fix |

## Daha Fazla Okumak

- [CS231n: Training Neural Networks](https://cs231n.github.io/neural-networks-3/) Tek sayfada eğitim borusunun en net gezisi
- [Bag of Tricks for Image Classification (He et al., 2019)](https://arxiv.org/abs/1812.01187) Toplu olarak, ResNet'in doğruluğuna % 3-4 katkı sağlayan her küçük numara
- [mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)](https://arxiv.org/abs/1710.09412) orijinal karışıklık kağıdı; üç sayfa teorik ve ikna edici deneyler
- [Why temperature scaling matters (Guo et al., 2017)](https://arxiv.org/abs/1706.04599) modern ağların yanlış kalibrlendiğini kanıtlayan kağıt ve bir skalar parametresi ile sabitledi
