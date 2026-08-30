# OCR ve Belge Anlaşması

> OCR üç aşamalı bir boru hattıdır.

**Type:** Learn + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06 (Detection), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~45 minutes

## Öğrenme Hedefleri

- Klasik OCR borusunu (detekte -> tanıma -> düzenleme) ve modern uçtan uç alternatiflerini (Donut, Qwen-VL-OCR) izle
- İlişkici Zamanlı Sınıflandırma (CTC) Kayıpları, OCR eğitiminin sırayla gerçekleşmesi için uygulanmalıdır
- Üretim belgeleri eğitimsiz analiz için PaddleOCR veya EasyOCR kullanın
- OCR, düzenleme analiz ve belge anlayışını ayırt edin  ve her görev için doğru aracı seçin

## Sorun

Metinlerle dolu görüntüler her yerde bulunur: makbuzlar, faturalar, kimlik kimlikleri, tarama kitapları, formlar, beyaz levhalar, işaretler, ekran görüntüleri.

Bu alan üç beceri katmanına ayrılmıştır:

1. **OCR proper**: pikselleri metne dönüştürün.
2. **Layout parsing**: grup OCR çıkışı bölgelere (şefkat, çerçeve, tablo, başlık)
3. **Document understanding**: düzenlenmiş alanları ("invoice_total = $42.50") çıkarın.

Her katman klasik ve modern yaklaşımlara sahiptir ve "Bir resmeden metin istiyorum" ve "Bu makbuzdan toplam miktarı istiyorum" arasındaki boşluk çoğu ekip fark ettiğinden daha büyüktür.

## Anlaşım

### Klasik boru hattı

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **Text detection**Satır başına veya kelime başına dörtgenler üretir.
- **Recognition**her bölgeyi sabit bir yüksekliğe çıkarır, bir karakter dizisini oluşturmak için CNN + BiLSTM + CTC çalıştırır.
- **Layout**okuma sırasını yeniden oluşturur (Latin için yukarıdan aşağıya, soldan sağa; Arapça için farklı, Japonca için).

### Tek paragrafdaki CTC

OCR tanıma, sabit uzunluklu bir özellik haritasından değişken uzunluklı bir dizini üretir. CTC (Graves et al., 2006) karakter düzeyde bir uyumsuzluk olmadan bunu eğitmenizi sağlar. Model her zaman adımında bir dağılım (sözlü + boş) çıkarır; CTC kaybı, tekrarları birleştirdikten ve boşlukları kaldırduktan sonra hedef metine indirgen tüm birimleri kenara çıkarır.

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

CTC, CRNN'in 2015 yılında çalışmasının ve hala 2026'da üretilen çoğu OCR modeli eğitmesinin nedeni.

### Modern uçtan uç modelleri

- **Donut**(Kim et al., 2022)  bir ViT kodlayıcı + bir metin dekodörü; bir görüntü okuyor ve doğrudan JSON yayar.
- **TrOCR** ViT + transformatör decoder, çizgi seviyesindeki OCR.
- **Qwen-VL-OCR / InternVL** OCR görevleri için ince ayarlanmış tam görme dil modelleri; karmaşık belgeler üzerinde 2026 yılında en iyi doğruluk.
- **PaddleOCR** Klasik DB + CRNN boru hattı olgun bir üretim paketi; hala açık kaynaklı iş atı.

Sonundan sonuna kadarki modeller daha fazla veri ve hesaplama gerektiriyor ancak çok aşamalı boru hattlarının hata birikimini atlatıyor.

### Yapılandırma analizleri

Yapılandırılmış belgeler için her bölgeyi etiketleyen bir düzenleme dedektörü (LayoutLMv3, DocLayNet) çalıştırın: Başlık, Paragraf, Şekil, Tablo, Ayağa Not. Okumak sırası daha sonra "örnekler boyunca düzenlemede tekrarlanır, bir zincir".

Formular için kullan **Key-Value extraction**Modeller (görünen zengin belgeler için donut, düz taramalar için LayoutLMv3) görüntü + tespit edilen metin + konumları alır ve yapılandırılmış anahtar-değer çiftlerini tahmin eder.

### Değerlendirme ölçümleri

- **Character Error Rate (CER)** Levenshtein mesafe / referans uzunluğu. Daha düşük daha iyi. Üretim hedefi: temiz taramalarda < 2%
- **Word Error Rate (WER)** kelime seviyesinde aynı.
- **F1 on structured fields** Ana değerli görevler için;`{invoice_total: 42.50}`Doğru görünüyor.
- **Edit distance on JSON** sonundan sonuna kadar belge analizleri için; Donut kağıdı, standart ağaç düzenleme mesafesini tanıttı.

```figure
cv3-ctc-collapse
```

## Yapın

### Adım 1: CTC kaybı + açgözlü dekodör

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss`Bu açgözlülükle decoder, bir ışın araması ile daha basit ve genellikle %1 CER'nin içinde.

### Adım 2: Küçük CRNN tanıtıcısı

Satır OCR için en az CNN + BiLSTM.

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

Sıkı yükseklik girimi (CNN maksimum yüksekliği 1'e kadar belirler). Genişlik, CTC için zaman boyutudur.

### Adım 3: Sentetik OCR

Sonundan sonuna duman testi için siyah beyaz parmaklık iplerini oluşturun.

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

Gerçek bir OCR veri kümesi şriftleri, gürültü, dönüm, bulanıklık ve renk ekler. Yukarıdaki boru hattı aynıdır.

### 4. Adım: Eğitim çizelgesi

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

Bu önemsiz sentetik veriler üzerinde 200 adımdan ~3'den ~0.2'ye düşmek gerekir.

## Kullan

Üç üretim yolu:

- **PaddleOCR** olgun, hızlı, çok dilli.`paddleocr.PaddleOCR(lang="en").ocr(image_path)`- Evet .
- **EasyOCR**Python-devli, çok dilli, PyTorch omurgası.
- **Tesseract** klasik; hala eski tarayılan belgeler için kullanışlı modeller mücadele ederken.

Sonundan sonuna kadar belge analizleri için Donut veya VLM kullanın:

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

Kitsler, faturalar ve tekrarlanabilir yapısı olan formlar için Donut'u ince ayarlayın. Bilinmeyen belgeler veya nedenlerle OCR için, Qwen-VL-OCR gibi bir VLM geçerli varsayımdır.

## Gönder

Bu ders şunları ortaya çıkarır:

- `outputs/prompt-ocr-stack-picker.md` Tesseract / PaddleOCR / Donut / VLM-OCR'i belirtilen belge türünü, dili ve yapısını seçen bir istek.
- `outputs/skill-ctc-decoder.md` uzunluk normallendirme dahil açgözlülük ve ışın araması CTC decoders sıfırdan yazma becerisi.

## Egzersizler

1. **(Easy)**TinyCRNN'i 500 adım boyunca 5 rakamlı rastgele sayısal iplere uygulayın.
2. **(Medium)**Açgözlü çözümü ışın araması ile değiştirin (beam_width=5). CER delta raporunu gönderin.
3. **(Hard)**20 risit, çizgi öğeleri çıkarmak ve {item_name, price} çiftleri için el etiketli zemin gerçeği karşı F1 hesaplamak için PaddleOCR kullanın.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| OCR | "Text from pixels" | Turning image regions into character sequences |
| CTC | "Alignment-free loss" | Loss that trains a sequence model without per-timestep labels; marginalises over alignments |
| CRNN | "Classic OCR model" | Conv feature extractor + BiLSTM + CTC; the 2015 baseline still used in production |
| Donut | "End-to-end OCR" | ViT encoder + text decoder; emits JSON directly from image |
| Layout parsing | "Find regions" | Detect and label Title/Table/Figure/Paragraph regions in a document |
| Reading order | "Text sequence" | Ordering of recognised regions into a sentence; trivial for Latin, non-trivial for mixed layouts |
| CER / WER | "Error rates" | Levenshtein distance / reference length at character or word granularity |
| VLM-OCR | "LLM that reads" | A vision-language model trained or prompted for OCR tasks; current SOTA on complex documents |

## Daha Fazla Okumak

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) orijinal CNN+RNN+CTC mimarisi
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) orijinal CTC kağıdı; algoritmik fikirlerle yoğun bir şekilde doldurulmuş
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664) OCR'den uzak bir belge anlayış transformörü
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) açık kaynaklı üretim OCR yığın
