# Aralıklı Kontrol Noktası ve Aktifleştirme Yeniden Hesaplama

> Backprop, her geçici etkinleştirmeyi tutar. 70B parametreleri ve 128K bağlamında bu, her sıra başına 3 TB etkinleştirme demektir. Kontrol noktası hafıza için FLOPs işlemleri: kaydetmek yerine yeniden hesaplamak. Sorun hangisi bölgeleri düşürmek ve cevap "bütünü" değil.

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## Sorun

Bir transformatör eğitimi, her katman için geriye ayrılmış her operasyonun girişlerini saklar: dikkat girişleri, Q/K/V projeksiyonları, softmax çıkışı, FFN girişleri, norm çıkışları ve kalan akım. Gizli boyutlu bir katman için `d`, dizi uzunluğu `L`, parti`B`Bu , emir üzerine .`12 * B * L * d`katman başına yüzen.

- Evet .`d=8192, L=8192, B=1`Bu, BF16'da katman başına 800 MB'dir. 64 katlı bir model 51 GB aktivasyonlara sahiptir.`L^2`(bkz: baş başına) ve tensor paralel kısmi kopyaları oluşturmadan önce.

İki taraflı fatur: BF16 ağırlıkları artı optimizer durumu 80GB'ye uygun olabilir, ancak etkinleştirmeler sizi öteye itebilir. Gradient kontrol noktası (aka activation recalculation) standart düzeltme. Çoğu etkinleştirmeyi bırakın; geriye dönerken ileriyi tekrar yapın.

Naifce yapıldığında, kontrol noktası, adım başına yaklaşık %33 daha fazla ileri geçiş FLOP maliyetindedir. İyi yapıldı  Korthikanti et al.'ın "akıllı seçimi" başına seçici kontrol noktası  5x hafıza tasarruf edersiniz.

## Anlaşım

### Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geriye Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri Geri

`output = layer(input)`Geriye dönmek istiyor .`grad_input`ve `grad_params`Onları hesaplamak için:

- `input`(Bilgilemek için `grad_params = input.T @ grad_output`Düzsel katmanlar için)
- bazı aktive derivatifler arası (ReLU/GELU/softmax'ın derivatifleri aktive değerine bağlıdır)

Ön geçit otomatik olarak otograd grafikinde depolanır.`tensor.retain_grad()`ve girişine ihtiyacı olan her operasyon bir referans tutar.

### Tam Kontrol Noktası Saçma

Ağı ikiye böl .`N`Önceki bölümler. Önceki bölümler sırasında, her bölüm için sadece * giriş * depolayın. Geriye geçişler gerektiğinde, segmentin önceki geçişini yeniden çalıştırın, sonra farklılaştırın.

Örnek: 32 katmanlı transformatör, her katman 1 katmanlı 32 bölüme ayrılmıştır.

- Hatıra: 32 katman giriş (küçük) vs 32 * (katman başına etkinleştirme hacmi) (çok büyük).
- Ekstra hesaplama: Segmente başına 1 ekstra ileri, yani %33 daha fazla ileri FLOP toplamı (geriye doğru 2x ileri olduğu için, tam adım 1 + 1 + 2 = 4 birim yerine 1 + 2 = 3 olur).

Bu Chen et al. 2016 tarihli orijinal tarifi: her bir kontrol noktası `sqrt(L)`L=64, 8 kontrol noktası.

### Seçimsel Kontrol Noktası (Korthikanti 2022)

Tüm etkinleştirmeler aynı maliyetli değil.`B*L*L*heads`FFN gizli etkinliği `B*L*4d`Uzun sekanslarda softmax hakimdir.

Seçimsel kontrol noktası, ucuz depolama aktivasyonlarını (lineer projeksiyonlar, kalıntılar) tutar ve sadece pahalı olanları (özen) yeniden hesaplar.

Megatron-Core bunu "seçici" etkinleştirme yeniden hesaplama olarak uyguluyor.

### Çıkarım

Yeniden hesaplama alternatifleri: ileri ve geriye doğru CPU RAM'a devreye aktarma. PCIe bant genişliği gerektirir; boş bant genişliği yeniden maddeleşme maliyetinden fazla olduğunda yararlıdır. Karışık stratejiler yaygın: bazı katmanları kontrol et, diğerlerini boşalt.

FSDP2 birinci sınıf bir seçenek olarak yükten çıkartır. GPU hafıza boğazında bulunduklarında yükten çıkartır.

### Ücret Modelini Yeniden Hesapla

Her adımda saf bir kontrol noktası ile FLOPs .`k`katmanları `L`- ...

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

Seçimsel kontrol noktası ile sadece dikkat çekirdeğini yeniden hesaplarsınız, tüm katmanı değil:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Hatıra Kaydetme Modülü

Katman başına etkinleştirme hacmi: `A`- Evet .`L`katmanlar, toplam aktivasyon hafızası: `L * A`- Evet .

Tam kontrol noktası (sektör boyutu 1): sadece depolama `L * input_volume`(~`L * 1/10 A`Standart bir transformatör için).`9 * L * A * 1/10`- Evet .

Kontrol noktası her zaman .`k`katmanlar: depolama `L/k * A`Ek olarak .`k-1`aktif segment içindeki katmanların değeri.

- Evet .`k = sqrt(L)`, bellek ve yeniden hesaplama maliyeti hem ölçekle `sqrt(L)` En iyi fiyat değişikliği.

### Kontrol Noktasına Ne Zaman Gitmemek

- Bir boru hattının en iç katmanları uçuşta zaten.
- Eğlence hesabına hükmeden ilk ve son katmanlar (transformatörlerde nadirdir).
- FlashAttention'ı kullanan dikkat çekirdekleri  Flash zaten softmax hızını yeniden hesaplar, bu yüzden ek katman seviyesindeki kontrol işaretlemeyi üstte biraz ekler.

### Uygulama Şekilleri

1. **Function wrapper:**Bir bölümü içine sarın `torch.utils.checkpoint.checkpoint(fn, input)`Sadece PyTorch mağazaları .`input`, geriye dönüp her şeyi yeniden hesaplar.

2. **Decorator-based:**Etiketlemenin kontrol noktası olarak yapılması gereken katmanlar; eğitmen, hangi bölümlerin toplanıp sarılacağına konfigürasyon zamanında karar verir.

3. **Manual explicit recompute:**Sıradan bir alışkanlık olarak geriye geçmeyi kendin yaz.`recompute_forward`Öncekiyi depolanan giriş ile çiftleştirir.

Üçü de aynı fonksiyonel sonuç verir.

### TP / PP / FP8 ile etkileşim

- **Tensor parallel:**Kontrol noktası girişleri yeniden hesaplama sırasında toplanmalı veya yeniden dağıtılmalıdır; iletişim maliyetini karşılamak.
- **Pipeline parallel:**Tipik bir örnektir. Her boru hattının aşamasının ileriye doğru kontrol edilmesi böylece geri sıra mikrobatçlar aktifleşme belleğini yeniden kullanabilmektedir.
- **FP8 recompute:**amax tarihleri yeniden hesaplama sırasında güncellenmiş orijinal ileri veya FP8 ölçek sürüşleri ile eşleşmelidir.

```figure
activation-recompute
```

## Yapın

### Adım 1: Bölümlerle Oyuncak Model

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### İkinci Adım: Geriye Alışmak İçin Tüm Aktivasyonlara İhtiyaç Var

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Adım 3: Kontrol Noktası-Her-k hafıza

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Dördüncü Adım: Maliyet modeli

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Adım 5: Hatıra Tahminici

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Adım 6: Optimal Bölüm Boyutu

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Adım 7: Seçimçi Kontrol Noktası Kararı

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## Kullan

- **torch.utils.checkpoint**- Evet .`from torch.utils.checkpoint import checkpoint`PyTorch'daki kanonik ambalaj. Bir fonksiyonu sarar; sadece girişleri saklar, geriye doğru yeniden hesaplar.
- **Megatron-Core activation recomputation**: destekler `selective`- Evet .`full`ve`block`2024+ sınır eğitiminde standart.
- **FSDP2 offload**- Evet .`module.to_empty(device="cpu")`- Evet .`offload_policy`FSDP2'de yeniden hesaplama yerine CPU'ya etkinleştirmelerini kısaltır.
- **DeepSpeed ZeRO-Offload**: Optimizer durumları ve etkinleştirmeleri için CPU yükü çıkartmak, kontrol noktasını tamamlamak.

## Gönder

Bu ders bize çok yararlı .`outputs/prompt-activation-recompute-policy.md` model yapılandırmasını (katmanlar, gizli, seq, parti) ve mevcut GPU belleğini alan ve katman başına yeniden hesaplama politikasını (hiçbir / seçici / tam / yüklenme) yayınlayan bir istek.

## Egzersizler

1. Doğru olduğunu kontrol et.`model_forward`+ `model_backward`(tam aktivasyon) vs `model_forward_checkpointed`+ `model_backward_checkpointed`Parametre gradiyenti makinenin hassasiyetine eşit olmalıdır.

2. Tarama bölümü boyutu `k`1 ' den `L`- FLOP'u ve hafızayı çiz.

3. Seçimsel kontrol işaretlemeyi uygulayın: dikkat modülünün girişini, ancak aralarını değil saklayın. 32 katlı bir model için FLOP üst üstlük vs tam katman kontrol işaretlemesini seq=8192'de ölçün.

4. Çıkarma ekleyin. Segment girişlerini simülasyonlu bir "CPU tamponu"na (ayrı bir liste) kaydetin. "PCIe bant genişliği" byte/zaman olarak ölçün ve çıkarma ve yeniden hesaplama arasındaki kesinti noktasını bulun.

5. Gerçek PyTorch transformatörünü , içinde ve dışında bir referans göster .`torch.utils.checkpoint`. hafıza ölçümleri (den`torch.cuda.max_memory_allocated`) ve adım zaman.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Gradient checkpointing | "Save memory by redoing forward" | Store segment inputs only; recompute intermediates during backward to get gradient-support tensors |
| Activation recomputation | "Same as checkpointing" | The HPC-flavored name for the same technique |
| Segment size (k) | "How many layers per checkpoint" | Number of layers whose intermediates are dropped and rematerialized together |
| Selective checkpointing | "Korthikanti's trick" | Recompute only expensive-to-store activations (attention softmax); keep cheap ones |
| Full checkpointing | "The naive version" | Recompute every layer's intermediates in every segment |
| Block checkpointing | "Coarse-grained" | Checkpoint whole transformer blocks; largest granularity |
| FLOP overhead | "The compute tax" | Extra FLOPs per step = (recompute FLOPs) / (fwd + bwd FLOPs); 33% naive, 5% selective |
| Activation offload | "Ship to CPU" | Move activations to CPU RAM across forward->backward; alternative to recompute |
| sqrt-L rule | "The classical optimum" | For uniform-cost layers, optimal checkpoint spacing is sqrt(L) layers |
| Attention-softmax volume | "The O(L^2) problem" | L^2 * heads * batch floats; dominates activation memory at long contexts |

## Daha Fazla Okumak

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174)- ...diğerleri kontrol etmek için resmileştirilen orijinal kağıt.
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198)-- Seçkin etkinleştirme yeniden hesaplama ve resmi maliyet analizi
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645)-- ters modunda yeniden maddeleşme yoluyla alternatif sabit hafıza yaklaşımı
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840)-- Ölçüsünde aktifleştirme yükü
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html)-- Standart API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html)-- Seçkin, tam ve blok modları
