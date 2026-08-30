# Nöral ağları düzeltmek

> Ağınız oluşturuldu. Çalıştı. Bir sayı üretti. Sayı yanlış ve hiçbir şey çökmedi. En zor türde düzeltme için hoş geldiniz.

**Type:** Build
**Languages:** Python, PyTorch
**Prerequisites:** Phase 03 Lessons 01-10 (especially backpropagation, loss functions, optimizers)
**Time:** ~90 minutes

## Öğrenme Hedefleri

- Sistematik debugging stratejileri kullanarak yaygın sinir ağı hatalarını (NaN kaybı, düz kayb eğri, aşırı uyum, ossilasyon) teşhis etmek
- Model mimarlığınızın ve eğitim döngüsünün doğru olup olmadığını kontrol etmek için "bir partiye fazla uygun" tekniğini uygulayın
- Kayıp / patlayan kayma sorunlarını belirlemek için gradient büyüklüklerini, aktivasyon dağılımlarını ve ağırlık normlarını incelemek
- Veriler boru hattını, model mimarisini, kayıp işlevi, optimizer ve öğrenme oranı sorunlarını kapsayan bir defegleme kontrol listesini oluşturun

## Sorun

Geleneksel yazılım bozulduğunda çöker. Bir sıfır işaretçi bir istisna atır. Bir tür eşleşmezliği bir düzenleme sırasında başarısız olur. Bir tek hata açıkça yanlış bir çıkış üretir.

Sinir ağları size bu lüksü vermez.

Kırık bir sinir ağı tamamlanıp, kayıp değeri yazdırır ve tahminler yapar. Kayıplar azaltabilir. Tahminler makul görünebilir. Ama model sessizce yanlış -- kısayolları öğrenmek, gürültü ezberlemek, veya işe yaramaz bir yerel minimum'a yaklaşmak. Google araştırmacıları, ML debugging süresinin %60-70'inin hata üretmeyen, ancak model kalitesini düşüren "sessiz" hatalar üzerinde harcandığını tahmin etti.

İşlemli ve kırık bir model arasındaki fark genellikle tek bir yanlış çizgidir: eksik bir çizgi `zero_grad()`, bir geçiş boyutu, öğrenme oranı 10x. Kanonik "Nöral Ağ Eğitimi Tarifi" (2019) şöyle başlar: "En yaygın sinir ağ hataları çökmeyen hatalardır".

Bu ders size bu böcekleri bulmayı öğretiyor.

## Anlaşım

### Yanlış Düşünce

Baskı ve pray debugging'i unutun. Nöral ağ debugging sistematik bir yaklaşım gerektirir çünkü geri bildirim döngüsü yavaş (öğrenme koşusu başına dakikalar saatler) ve semptomlar belirsiz (kötü kayıp 20 farklı şeyi ifade edebilir).

Altın kural:**start simple, add complexity one piece at a time, and verify each piece independently.**

```mermaid
flowchart TD
    A["Loss not decreasing"] --> B{"Check learning rate"}
    B -->|"Too high"| C["Loss oscillates or explodes"]
    B -->|"Too low"| D["Loss barely moves"]
    B -->|"Reasonable"| E{"Check gradients"}
    E -->|"All zeros"| F["Dead ReLUs or vanishing gradients"]
    E -->|"NaN/Inf"| G["Exploding gradients"]
    E -->|"Normal"| H{"Check data pipeline"}
    H -->|"Labels shuffled"| I["Random-chance accuracy"]
    H -->|"Preprocessing bug"| J["Model learns noise"]
    H -->|"Data is fine"| K{"Check architecture"}
    K -->|"Too small"| L["Underfitting"]
    K -->|"Too deep"| M["Optimization difficulty"]
```

### Simptom 1: Kayıplar azalmıyor

Bu en yaygın şikayet. Ekipler geçiyor, kaybı düz kalıyor ya da vahşice dalgalanıyor.

**Wrong learning rate.**Çok yüksek: kayıp dalgalanır veya NaN'ye atlar. Çok düşük: kayıp çok yavaş azalır ve düz görünür. Adam için 1e-3'den başlayın. SGD için, 1e-1 veya 1e-2'den başlayın.

**Dead ReLUs.**Bir ReLU nöronu büyük bir negatif giriş alırsa, 0 çıkışı yapar ve gradiyenti 0 olur. Bir daha asla etkinleştirmez. Yeterince nöron ölürse, ağ öğrenemez. Kontrol: her ReLU katmanından sonra tam olarak 0 olan etkinliklerin bölümü basın.

**Vanishing gradients.**Sigmoid veya tanh etkinleştirmeleri olan derin ağlarda, gradientler geriye yayıldıkça eksponansiyel olarak küçülürler. İlk katmana ulaştıklarında, ~ 0'durlar. İlk katmanlar öğrenmeyi durdurur. Düzelt: ReLU/GELU kullanın, kalan bağlantılar ekleyin veya parti normallaşımı kullanın.

**Exploding gradients.**RNN'lerde ve çok derin ağlarda yaygın olan kayıplar NaN'e atlıyor.`torch.nn.utils.clip_grad_norm_`), öğrenme oranını düşürmek veya normalleşmeyi eklemek.

### Simptom 2: Kayıplar azalıyor ama model kötü

Kayıplar azalır. Eğitim doğruluğu %99'a ulaşır. Ama test doğruluğu %55'dir. Ya da model gerçek veriler üzerinde anlamsız sonuçlar üretir.

**Overfitting.**Model eğitim verilerini öğrenme kalıpları yerine ezberler. Eğitim ve doğrulama kaybı arasındaki boşluk zamanla büyür. Düzelt: daha fazla veri, düşüş, kilo kaybı, erken durma, veri artırımı.

**Data leakage.**Test verileri eğitimde sızdı.Düşünçlü bir doğruluk.Çık sık nedenler: bölmeden önce karıştırma, tüm veri kümesinden istatistiklerle önceden işleme, bölünmüştürler üzerinden kopyalama örnekleri.Düzeltme: ilk bölünme, ikinci preprocess, kopyalamaları kontrol et.

**Label errors.**Çoğu gerçek veri kümesindeki etiketlerin %5-10'u yanlış (Northcutt et al., 2021 -- "Test kümelerindeki yaygın etiket hataları"). Model gürültüyü öğrenir. Düzelt: yanlış etiketlenen örnekleri bulmak ve düzeltmek için güvenli öğrenmeyi kullanın veya yüksek kayıplı örnekleri görmezden gelmek için kayıp kesimini kullanın.

### Simptom 3: Kayıpta olan NaN veya Inf

Kayıp değeri `nan`veya `inf`Eğitim bitti.

**Learning rate too high.**Geliştirmeler ağırlıkların patlaması için aşırı ilerliyor.

**log(0) or log(negative).**Çaplak entropik kaybı hesaplamaları `log(p)`Eğer modeliniz tam olarak 0 veya negatif bir olasılık çıkarsa, kayıt patlar.`[eps, 1-eps]`nerede`eps=1e-7`- Evet .

**Division by zero.**Batch normallaşması standart sapma ile bölünür. Sürekli değerleri olan bir batch std=0'dur. Düzelt: isimlendiriciye epsilon ekleyin (PyTorch bunu varsayılan olarak yapar, ancak özel uygulamalar olmayabilir).

**Numerical overflow.**Büyük aktivasyonlar `exp()`Bu nedenle, bu işlemin en yüksek oranı, en yüksek oranı eksponansyalılaştırmadan önce çıkarmak için yapılması gerekir.

### Teknik 1: Gelişme Kontrolü

Analiz gradiyentilerinizi (backprop'dan) sayısal gradiyentilere (sınırlı farklılıklardan) karşılaştırın.

Parametre için sayısal gradient `w`- ...

```
grad_numerical = (loss(w + eps) - loss(w - eps)) / (2 * eps)
```

Anlaşmanın ölçüsü (sarefli fark):

```
rel_diff = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

- Eğer`rel_diff < 1e-5`Doğru.`rel_diff > 1e-3`- Kesinlikle bir böcek.

```mermaid
flowchart LR
    A["Parameter w"] --> B["w + eps"]
    A --> C["w - eps"]
    B --> D["Forward pass"]
    C --> E["Forward pass"]
    D --> F["loss+"]
    E --> G["loss-"]
    F --> H["(loss+ - loss-) / 2eps"]
    G --> H
    H --> I["Compare to backprop gradient"]
```

### Teknik 2: Aktifleştirme istatistikleri

Eğitim sırasında her katman sonrasında etkinliklerin ortalama ve standart sapmalarını izleyin. Sağlıklı ağlar, normalleşmeden sonra 0 ve std yakınlarında (normalleşmeden sonra) veya en az sınırlı olarak etkinliklerin ortalamasını sürdürür.

| Health indicator | Mean | Std | Diagnosis |
|-----------------|------|-----|-----------|
| Healthy | ~0 | ~1 | Network is learning normally |
| Saturated | >>0 or <<0 | ~0 | Activations stuck at extreme values |
| Dead | 0 | 0 | Neurons are dead (all zeros) |
| Exploding | >>10 | >>10 | Activations growing without bound |

### Teknik 3: Gradyent Akış Görüşü

Her katman için ortalama gradient büyüklüğünü çizin. Sağlıklı bir ağda, gradient büyüklükleri katmanlar arasında yaklaşık olarak benzer olmalıdır. Eğer ilk katmanların daha sonraki katmanlardan 1000 kat daha küçük gradientleri varsa, kaybolan gradientler vardır.

```mermaid
graph LR
    subgraph "Healthy Gradient Flow"
        L1["Layer 1<br/>grad: 0.05"] --- L2["Layer 2<br/>grad: 0.04"] --- L3["Layer 3<br/>grad: 0.06"] --- L4["Layer 4<br/>grad: 0.05"]
    end
```

```mermaid
graph LR
    subgraph "Vanishing Gradient Flow"
        V1["Layer 1<br/>grad: 0.0001"] --- V2["Layer 2<br/>grad: 0.003"] --- V3["Layer 3<br/>grad: 0.02"] --- V4["Layer 4<br/>grad: 0.08"]
    end
```

### Teknik 4: Bir Satırlık Üstü Uygulama

Derin öğrenmede en önemli debugging tekniği.

Bir küçük partiyi alın (8-32 örnek). 100'den fazla iterasyon için üzerinde çalışın. Kayıp neredeyse sıfıra ulaşmalı ve eğitim doğruluğu %100'e ulaşmalıdır.

Bu test:
- Kırık kayıp fonksiyonları
- Kırık geri geçişler
- Verileri temsil etmek için çok küçük bir mimarlık
- Model parametrelerine bağlı olmayan optimizer
- Veriler ve etiketler yanlış ayarlandı

Bu 30 saniye sürer ve saatlerce tam antrenman çalışmasını kurtarır.

### 5. Teknik: Öğrenme oranı arayan

Leslie Smith (2017) bir dönem boyunca öğrenme oranını çok küçükden (1e-7) çok büyük (10) bir süre boyunca kaydetmeyi önerdi.

```mermaid
graph TD
    subgraph "LR Finder Plot"
        direction LR
        A["1e-7: loss=2.3"] --> B["1e-5: loss=2.3"]
        B --> C["1e-3: loss=1.8"]
        C --> D["1e-2: loss=0.9 -- steepest"]
        D --> E["1e-1: loss=0.5"]
        E --> F["1.0: loss=NaN -- too high"]
    end
```

Bu örnekte en iyi LR: ~1e-3 (en dik noktanın öncesinde büyüklük bir sıralama).

### Yaygın PyTorch Böcekleri

PyTorch topluluğunda en kolektif saatleri harcadıkları böcekler bunlar:

| Bug | Symptom | Fix |
|-----|---------|-----|
| Forgetting `optimizer.zero_grad()` | Gradients accumulate across batches, loss oscillates | Add `optimizer.zero_grad()` before `loss.backward()` |
| Forgetting `model.eval()` at test time | Dropout and batch norm behave differently, test accuracy varies between runs | Add `model.eval()` and `torch.no_grad()` |
| Wrong tensor shapes | Silent broadcasting produces wrong results, no error | Print shapes after every operation during debugging |
| CPU/GPU mismatch | `RuntimeError: expected CUDA tensor` | Use `.to(device)` on model AND data |
| Not detaching tensors | Computation graph grows forever, OOM | Use `.detach()` or `with torch.no_grad()` |
| In-place operations breaking autograd | `RuntimeError: modified by in-place operation` | Replace `x += 1` with `x = x + 1` |
| Data not normalized | Loss stuck at random-chance level | Normalize inputs to mean=0, std=1 |
| Labels as wrong dtype | Cross-entropy expects `Long`, got `Float` | Cast labels: `labels.long()` |

### Başlıca Çözme Masası

| Symptom | Likely cause | First thing to try |
|---------|-------------|-------------------|
| Loss stuck at -log(1/num_classes) | Model predicting uniform distribution | Check data pipeline, verify labels match inputs |
| Loss NaN after a few steps | Learning rate too high | Reduce LR by 10x |
| Loss NaN immediately | log(0) or division by zero | Add epsilon to log/division operations |
| Loss oscillating wildly | LR too high or batch size too small | Reduce LR, increase batch size |
| Loss decreasing then plateaus | LR too high for fine-tuning phase | Add LR schedule (cosine or step decay) |
| Training acc high, test acc low | Overfitting | Add dropout, weight decay, more data |
| Training acc = test acc = chance | Model not learning anything | Run overfit-one-batch test |
| Training acc = test acc but both low | Underfitting | Bigger model, more layers, more features |
| Gradients all zero | Dead ReLUs or detached computation graph | Switch to LeakyReLU, check `.requires_grad` |
| Out of memory during training | Batch too large or graph not freed | Reduce batch size, use `torch.no_grad()` for eval |

```figure
learning-curves
```

## Yapın

Bir ağı kasıtlı olarak kırıp her sorunu teşhis etmek için araç kümesini kullanırsınız.

### Adım 1: NetworkDebugger Sınıfı

Bir katman başına aktivasyon ve gradient istatistiklerini kaydetmek için PyTorch modeli ile bağlanır.

```python
import torch
import torch.nn as nn
import math


class NetworkDebugger:
    def __init__(self, model):
        self.model = model
        self.activation_stats = {}
        self.gradient_stats = {}
        self.loss_history = []
        self.lr_losses = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.LeakyReLU)):
                hook = module.register_forward_hook(self._make_activation_hook(name))
                self.hooks.append(hook)
                hook = module.register_full_backward_hook(self._make_gradient_hook(name))
                self.hooks.append(hook)

    def _make_activation_hook(self, name):
        def hook(module, input, output):
            with torch.no_grad():
                out = output.detach().float()
                self.activation_stats[name] = {
                    "mean": out.mean().item(),
                    "std": out.std().item(),
                    "fraction_zero": (out == 0).float().mean().item(),
                    "min": out.min().item(),
                    "max": out.max().item(),
                }
        return hook

    def _make_gradient_hook(self, name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                with torch.no_grad():
                    grad = grad_output[0].detach().float()
                    self.gradient_stats[name] = {
                        "mean": grad.mean().item(),
                        "std": grad.std().item(),
                        "abs_mean": grad.abs().mean().item(),
                        "max": grad.abs().max().item(),
                    }
        return hook

    def record_loss(self, loss_value):
        self.loss_history.append(loss_value)

    def check_loss_health(self):
        if len(self.loss_history) < 2:
            return "NOT_ENOUGH_DATA"
        recent = self.loss_history[-10:]
        if any(math.isnan(v) or math.isinf(v) for v in recent):
            return "NAN_OR_INF"
        if len(self.loss_history) >= 20:
            first_half = sum(self.loss_history[:10]) / 10
            second_half = sum(self.loss_history[-10:]) / 10
            if second_half >= first_half * 0.99:
                return "NOT_DECREASING"
        if len(recent) >= 5:
            diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            if max(diffs) - min(diffs) > 2 * abs(sum(diffs) / len(diffs)):
                return "OSCILLATING"
        return "HEALTHY"

    def check_activations(self):
        issues = []
        for name, stats in self.activation_stats.items():
            if stats["fraction_zero"] > 0.5:
                issues.append(f"DEAD_NEURONS: {name} has {stats['fraction_zero']:.0%} zero activations")
            if abs(stats["mean"]) > 10:
                issues.append(f"EXPLODING_ACTIVATIONS: {name} mean={stats['mean']:.2f}")
            if stats["std"] < 1e-6:
                issues.append(f"COLLAPSED_ACTIVATIONS: {name} std={stats['std']:.2e}")
        return issues if issues else ["HEALTHY"]

    def check_gradients(self):
        issues = []
        grad_magnitudes = []
        for name, stats in self.gradient_stats.items():
            grad_magnitudes.append((name, stats["abs_mean"]))
            if stats["abs_mean"] < 1e-7:
                issues.append(f"VANISHING_GRADIENT: {name} abs_mean={stats['abs_mean']:.2e}")
            if stats["abs_mean"] > 100:
                issues.append(f"EXPLODING_GRADIENT: {name} abs_mean={stats['abs_mean']:.2e}")
        if len(grad_magnitudes) >= 2:
            first_mag = grad_magnitudes[0][1]
            last_mag = grad_magnitudes[-1][1]
            if last_mag > 0 and first_mag / last_mag > 100:
                issues.append(f"GRADIENT_RATIO: first/last = {first_mag/last_mag:.0f}x (vanishing)")
        return issues if issues else ["HEALTHY"]

    def print_report(self):
        print("\n=== NETWORK DEBUGGER REPORT ===")
        print(f"\nLoss health: {self.check_loss_health()}")
        if self.loss_history:
            print(f"  Last 5 losses: {[f'{v:.4f}' for v in self.loss_history[-5:]]}")
        print("\nActivation diagnostics:")
        for item in self.check_activations():
            print(f"  {item}")
        print("\nGradient diagnostics:")
        for item in self.check_gradients():
            print(f"  {item}")
        print("\nPer-layer activation stats:")
        for name, stats in self.activation_stats.items():
            print(f"  {name}: mean={stats['mean']:.4f} std={stats['std']:.4f} zero={stats['fraction_zero']:.1%}")
        print("\nPer-layer gradient stats:")
        for name, stats in self.gradient_stats.items():
            print(f"  {name}: abs_mean={stats['abs_mean']:.2e} max={stats['max']:.2e}")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
```

### İkinci Adım: Bir Satırlık Üstü Uygulama

```python
def overfit_one_batch(model, x_batch, y_batch, criterion, lr=0.01, steps=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    print("\n=== OVERFIT ONE BATCH TEST ===")
    print(f"Batch size: {x_batch.shape[0]}, Steps: {steps}")

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                preds = (output > 0).float() if output.shape[-1] == 1 else output.argmax(dim=1)
                targets = y_batch if y_batch.dim() == 1 else y_batch.squeeze()
                acc = (preds.squeeze() == targets).float().mean().item()
            print(f"  Step {step:3d} | Loss: {loss.item():.6f} | Accuracy: {acc:.1%}")

    final_loss = loss.item()
    if final_loss > 0.1:
        print(f"\n  FAIL: Loss did not converge ({final_loss:.4f}). Model or training loop is broken.")
        return False
    print(f"\n  PASS: Loss converged to {final_loss:.6f}")
    return True
```

### Adım 3: Öğrenme oranı

```python
def find_learning_rate(model, x_data, y_data, criterion, start_lr=1e-7, end_lr=10, steps=100):
    import copy
    original_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1 / steps)

    model.train()
    results = []
    best_loss = float("inf")
    current_lr = start_lr

    print("\n=== LEARNING RATE FINDER ===")

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_data)
        loss = criterion(output, y_data)

        if math.isnan(loss.item()) or loss.item() > best_loss * 10:
            break

        best_loss = min(best_loss, loss.item())
        results.append((current_lr, loss.item()))

        loss.backward()
        optimizer.step()

        current_lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

    model.load_state_dict(original_state)

    if len(results) < 10:
        print("  Could not complete LR sweep -- loss diverged too quickly")
        return results

    min_loss_idx = min(range(len(results)), key=lambda i: results[i][1])
    suggested_lr = results[max(0, min_loss_idx - 10)][0]

    print(f"  Swept {len(results)} steps from {start_lr:.0e} to {results[-1][0]:.0e}")
    print(f"  Minimum loss {results[min_loss_idx][1]:.4f} at lr={results[min_loss_idx][0]:.2e}")
    print(f"  Suggested learning rate: {suggested_lr:.2e}")

    return results
```

### 4. Adım: Gradyent Kontrolü

```python
def _flat_to_multi_index(flat_idx, shape):
    multi_idx = []
    remaining = flat_idx
    for dim in reversed(shape):
        multi_idx.insert(0, remaining % dim)
        remaining //= dim
    return tuple(multi_idx)


def gradient_check(model, x, y, criterion, eps=1e-4):
    model.train()
    x_double = x.double()
    y_double = y.double()
    model_double = model.double()

    print("\n=== GRADIENT CHECK ===")
    overall_max_diff = 0
    checked = 0

    for name, param in model_double.named_parameters():
        if not param.requires_grad:
            continue

        layer_max_diff = 0

        model_double.zero_grad()
        output = model_double(x_double)
        loss = criterion(output, y_double)
        loss.backward()
        analytical_grad = param.grad.clone()

        num_checks = min(5, param.numel())
        for i in range(num_checks):
            idx = _flat_to_multi_index(i, param.shape)
            original = param.data[idx].item()

            param.data[idx] = original + eps
            with torch.no_grad():
                loss_plus = criterion(model_double(x_double), y_double).item()

            param.data[idx] = original - eps
            with torch.no_grad():
                loss_minus = criterion(model_double(x_double), y_double).item()

            param.data[idx] = original

            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytical = analytical_grad[idx].item()

            denom = max(abs(numerical), abs(analytical), 1e-8)
            rel_diff = abs(numerical - analytical) / denom

            layer_max_diff = max(layer_max_diff, rel_diff)
            checked += 1

        overall_max_diff = max(overall_max_diff, layer_max_diff)
        status = "OK" if layer_max_diff < 1e-5 else "MISMATCH"
        print(f"  {name}: max_rel_diff={layer_max_diff:.2e} [{status}]")

    model.float()

    print(f"\n  Checked {checked} parameters")
    if overall_max_diff < 1e-5:
        print("  PASS: Gradients match (rel_diff < 1e-5)")
    elif overall_max_diff < 1e-3:
        print("  WARN: Small differences (1e-5 < rel_diff < 1e-3)")
    else:
        print("  FAIL: Gradient mismatch detected (rel_diff > 1e-3)")
    return overall_max_diff
```

### Adım 5: Kasten Kırık Ağlar

Şimdi araç kitini kırık ağlara uygulayın ve her birini teşhis edin.

```python
def demo_broken_networks():
    torch.manual_seed(42)
    x = torch.randn(64, 10)
    y = (x[:, 0] > 0).long()

    print("\n" + "=" * 60)
    print("BUG 1: Learning rate too high (lr=10)")
    print("=" * 60)
    model1 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger1 = NetworkDebugger(model1)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=10.0)
    criterion = nn.CrossEntropyLoss()
    for step in range(20):
        optimizer1.zero_grad()
        out = model1(x)
        loss = criterion(out, y)
        debugger1.record_loss(loss.item())
        loss.backward()
        optimizer1.step()
    debugger1.print_report()
    debugger1.remove_hooks()

    print("\n" + "=" * 60)
    print("BUG 2: Dead ReLUs from bad initialization")
    print("=" * 60)
    model2 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
    with torch.no_grad():
        for m in model2.modules():
            if isinstance(m, nn.Linear):
                m.weight.fill_(-1.0)
                m.bias.fill_(-5.0)
    debugger2 = NetworkDebugger(model2)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    for step in range(50):
        optimizer2.zero_grad()
        out = model2(x)
        loss = criterion(out, y)
        debugger2.record_loss(loss.item())
        loss.backward()
        optimizer2.step()
    debugger2.print_report()
    debugger2.remove_hooks()

    print("\n" + "=" * 60)
    print("BUG 3: Missing zero_grad (gradients accumulate)")
    print("=" * 60)
    model3 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger3 = NetworkDebugger(model3)
    optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.01)
    for step in range(50):
        out = model3(x)
        loss = criterion(out, y)
        debugger3.record_loss(loss.item())
        loss.backward()
        optimizer3.step()
    debugger3.print_report()
    debugger3.remove_hooks()

    print("\n" + "=" * 60)
    print("HEALTHY NETWORK: Correct setup for comparison")
    print("=" * 60)
    model_good = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger_good = NetworkDebugger(model_good)
    optimizer_good = torch.optim.Adam(model_good.parameters(), lr=1e-3)
    for step in range(50):
        optimizer_good.zero_grad()
        out = model_good(x)
        loss = criterion(out, y)
        debugger_good.record_loss(loss.item())
        loss.backward()
        optimizer_good.step()
    debugger_good.print_report()
    debugger_good.remove_hooks()

    print("\n" + "=" * 60)
    print("OVERFIT-ONE-BATCH TEST (healthy model)")
    print("=" * 60)
    model_test = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    overfit_one_batch(model_test, x[:8], y[:8], criterion)

    print("\n" + "=" * 60)
    print("LEARNING RATE FINDER")
    print("=" * 60)
    model_lr = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    find_learning_rate(model_lr, x, y, criterion)

    print("\n" + "=" * 60)
    print("GRADIENT CHECK")
    print("=" * 60)
    model_grad = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 2))
    gradient_check(model_grad, x[:4], y[:4], criterion)
```

## Kullan

### PyTorch İçine Ekli Aletler

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

with torch.autograd.detect_anomaly():
    output = model(input_tensor)
    loss = criterion(output, target)
    loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_mean={param.grad.abs().mean():.2e}")
```

### Ağırlık ve Tarafsızlık Entegre edilmesi

```python
import wandb

wandb.init(project="debug-training")

for epoch in range(100):
    loss = train_one_epoch()
    wandb.log({
        "loss": loss,
        "lr": optimizer.param_groups[0]["lr"],
        "grad_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")),
    })

    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"grad/{name}": wandb.Histogram(param.grad.cpu().numpy())})
```

### TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/debug_experiment")

for epoch in range(100):
    loss = train_one_epoch()
    writer.add_scalar("Loss/train", loss, epoch)

    for name, param in model.named_parameters():
        writer.add_histogram(f"weights/{name}", param, epoch)
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad, epoch)
```

### Çözümlülük Kontrol Listesini (Tüm Eğitimden Önce)

1. Bir seri test yapın, eğer başarısız olursa durun.
2. Baskı model özetini -- parametre sayısını doğrulamak makul.
3. Rastgele verilerle tek bir ileri geçiş yapın. Çıkış şeklini kontrol edin.
4. 5 dönem için tren... Kayıpların azalmasını kontrol edin.
5. Aktiflik istatistiklerini kontrol edin. Ölmüş katmanlar, patlamalar yok.
6. Değişken akışını kontrol edin. Yok yok, patlama yok.
7. Veri hattını doğrulayın. Etiketlerle 5 rastgele örnek yazdırın.

## Gönder

Bu ders şunları ortaya çıkarır:
- `outputs/prompt-nn-debugger.md`-- sinir ağı eğitimi başarısızlıklarını teşhis için bir uyarı
- `outputs/skill-debug-checklist.md`-- Debug eğitim sorunları için bir karar ağacı kontrol listesini

Çözümleme için ana dağıtım kalıpları:
- Üretim eğitim senaryolarına izleme kancalarını ekle
- Günlük aktivasyonu ve gradient istatistikleri W&B veya TensorBoard'a her N adımda
- NaN kaybı, ölü nöronlar (> 80% sıfır) veya gradient patlaması için otomatik uyarılar uygulanmalıdır.
- Arsitektirleri veya veri borularını değiştirirken her zaman bir serili overfit testini çalıştırın

## Egzersizler

1. **Add an exploding gradient detector.**Değiştir `NetworkDebugger`Bu, gradientlerin bir eşiği aştığını tespit etmek ve otomatik olarak gradient kesme değerini önermek için.

2. **Build a dead neuron resurrector.**Ölü ReLU nöronlarını tanımlayan (her zaman 0 çıkaran) ve gelen ağırlıklarını Kaiming başlangıcı ile yeniden başlatır bir işlev yazın. Bu, nöronların %70'inin öldüğü bir ağı geri aldığını gösterin.

3. **Implement the learning rate finder with plotting.**Uzaklaştırma`find_learning_rate`sonuçları bir CSV olarak kaydetmek ve CSV'yi okuyan ve matplotlib kullanarak LR vs kayb eğriyi görüntüleyen ayrı bir senaryo yazmak için.

4. **Create a data pipeline validator.**Tren/test bölükleri üzerinde ikili örnekler, etiket dağılım dengesizliği (> 10:1 oranı), giriş normalleşmesi (ortalama 0, std yakın 1), ve verilerdeki NaN/Inf değerlerini kontrol eden bir işlev yazın.

5. **Debug a real failure.**10. Dersten mini çerçeveyi alın, ince bir hata (örneğin ağırlık matrisini geriye doğru naklet) ve hangi parametre yanlış gradientlere sahip olduğunu tam olarak bulmak için gradient kontrolünü kullanın.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Silent bug | "It runs but gives bad results" | A bug that produces no error but degrades model quality -- the dominant failure mode in ML |
| Dead ReLU | "The neurons died" | A ReLU neuron whose input is always negative, so it outputs 0 and receives 0 gradient permanently |
| Vanishing gradients | "Early layers stop learning" | Gradients shrink exponentially through layers, making weights in early layers effectively frozen |
| Exploding gradients | "Loss went to NaN" | Gradients grow exponentially through layers, causing weight updates so large they overflow |
| Gradient checking | "Verify backprop is correct" | Comparing analytical gradients from backprop to numerical gradients from finite differences |
| Overfit-one-batch | "The most important debug test" | Training on a single small batch to verify the model CAN learn -- if it cannot, something is fundamentally broken |
| LR finder | "Sweep to find the right learning rate" | Exponentially increasing the learning rate over one epoch and picking the rate just before loss diverges |
| Data leakage | "Test data leaked into training" | When information from the test set contaminates training, producing artificially high accuracy |
| Activation statistics | "Monitor layer health" | Tracking mean, std, and zero-fraction of each layer's output to detect dead, saturated, or exploding neurons |
| Gradient clipping | "Cap the gradient magnitude" | Scaling gradients down when their norm exceeds a threshold, preventing exploding gradient updates |

## Daha Fazla Okumak

- Smith, "Trening Neural Networks için Siklik Öğrenme Sınıfları" (2017) - Öğrenme Sınıfı Testini Tanıtım Eden Kağıt (LR Finder)
- Northcutt et al., "Test Setlerinde Yaygın Etiket Hataları Makineler Öğrenme Benchmarks'i Durdurabilirleştirir" (2021) -- ImageNet, CIFAR-10 ve diğer büyük referans değerlerinin %3-6'unun yanlış olduğunu gösterir
- Zhang et al., "Deep Learning Understanding Requires Re-Thinking Generalization" (2017) - nöral ağların rastgele etiketleri hatırlayabileceğini gösteren makale, bu yüzden overfit-one-batch testinin işe yaraması
- PyTorch belgesi `torch.autograd.detect_anomaly`ve `torch.autograd.set_detect_anomaly`İçeriye yerleştirilmiş NaN/Inf tespit için
