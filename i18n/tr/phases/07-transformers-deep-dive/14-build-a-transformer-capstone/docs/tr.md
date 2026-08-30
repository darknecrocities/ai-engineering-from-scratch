# Bir Transformer'ı Baştan Yap  Capstone

> 13 ders, bir model, kısayol yok.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 minutes

## Sorun

Her makaleyi okudun. Dikkat, çok başlı bölünmeler, konum kodlamaları, kodlayıcı ve dekoder blokları, BERT ve GPT kaybı, MoE, KV önbelleği uyguladın. Şimdi gerçek bir görev üzerinde birlikte çalıştırın.

Baş taşı: karakter düzeyinde dil modelleme görevi için küçük bir decoder-tek transformatörü son-son eğit. Shakespeare'i okuyor. Yeni Shakespeare'i oluşturur. 10 dakikadan az bir süre içinde bir dizüstü bilgisayar üzerinde eğitilmek için yeterince küçük. Daha büyük bir veri kümesi ve daha uzun bir eğitimde değişmek size gerçek bir LM sağlar.

Bu ders "nanoGPT"i. Orijinal değil. Karpathy'nin 2023 nanoGPT öğretim kitabı her öğrencinin en az bir kez yazdığı referans uygulamasıdır. Şekilini kaldırıp kapadığımız şeyi yeniden düzenliyoruz.

## Anlaşım

![Transformer-from-scratch block diagram](../assets/capstone.svg)

Mimarlık, şöyle açıklandı:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### Neyi gönderiyoruz

- `GPTConfig` Tüm hiperparametreyi yapılandırmak için tek bir yer.
- `MultiHeadAttention` sebepçi, serili, seçeneği Flash tarzı yol (PyTorch's `scaled_dot_product_attention`)
- `SwiGLUFFN` Modern FFN.
- `Block` Normalden önce, kalan sarılmış dikkat + FFN.
- `GPT` yerleşim, yığılmış bloklar, LM başı, üretmek().
- AdamW, cosine LR, gradient kesimi ile eğitim döngüsü.
- Shakespeare'in metinleri için bir işaretlemeci.

### - Neyi göndermiyoruz?

- RoPE  Ders 04'te kavramsal olarak uygulanmıştır. Burada basitlik için öğrenilen pozisyonsal yerleşimleri kullanıyoruz.
- KV önbelleği, her nesil adımıyla tüm önbelleği yeniden hesaplar. Daha yavaş ama daha basit.
- Flash Dikkat  PyTorch 2.0+ otomatik gönderiler girdiler eşleşirse; biz kullanıyoruz `F.scaled_dot_product_attention`- Evet .
- MoE'yi blok başına tek FFN'den görüyorsunuz.

### Hedef ölçümleri

Mac M2 dizüstü bilgisayarında, 4 katlı, 4 başlı, d_model=128 GPT 2000 adım için eğitilmiş.`tinyshakespeare.txt`- ...

- Eğitim kaybı yaklaşık 6 dakika içinde ~4.2 (hassasi) ~1.5'e doğru ilerliyor.
- Örnek alınan ürün Shakespeare şeklinde görünüyor: eski kelimeler, çizgi kesintiler, "ROMEO:" gibi özel isimler ortaya çıkıyor.
- Val kaybı (teksten son 10%'i tutturulmuştur) eğitim kaybını yakından takip eder; bu boyutta/ bütçede fazla uygun değildir.

```figure
n5-block-stack
```

## Yapın

Bu ders PyTorch kullanıyor.`torch`(CPU yapı tamamdır).`code/main.py`Senaryo şu şekilde:

- İndirmek`tinyshakespeare.txt`Eğer eksikse (veya yerel bir kopyayı okuyabilirse).
- Byte seviyesindeki char tokenizer.
- Tren/val bölümü 90/10'da.
- Desteklenen donanım üzerinde bf16 otomatik olarak yayınlanan eğitim döngüsü.
- Eğitimden sonra örnekleme tamamlandı.

### Adım 1: Veriler

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 eşsiz karakter, küçük kelime birikimi, 4 baytlık bir kelime boyutuna uygun, BPE, tokenizer dramı yok.

### Adım 2: Model

Bakın .`code/main.py`.Blok, ders 05  pre-norm, RMSNorm, SwiGLU, nedenci MHA'dan ders kitabı. 4/4/128 için parametre sayısı: ~800K.

### Adım 3: Eğitim döngüsü

- 256 uzunluklu simge pencerelerinin rastgele bir seriyi alın.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 4. adım: örnek

Bir istek verildiğinde, tekrar tekrar ileriye doğru, en üst p logitlerinden örnek, ekle ve devam et. 500 token sonrasında dur.

### Adım 5: çıkış oku

2000 adımdan sonra:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

Shakespeare değil, Shakespeare şeklinde, 800 bin parameter ve 6 dakika bir dizüstü bilgisayarla kazanmış.

## Kullan

Bu temel taş, bir referans mimarisi.

1. **Swap the tokenizer.**BPE kullanın (örneğin `tiktoken.get_encoding("cl100k_base")`Sözcük boyutu 65'den ~ 50.000'e kadar atlıyor.
2. **Train on a bigger corpus.**Kullanım`OpenWebText`veya `fineweb-edu`Tek bir A100'de 10B token, 125M param GPT için yaklaşık 24 saat sürer.
3. **Add RoPE + KV cache + Flash Attention.**Aşağıdaki egzersizler her birinizi takip eder.

Bu, akıcı İngilizce üreten 125M parametresi GPT olarak sona erer. Sınır modeli değil. Ama aynı kod yolu  sadece daha büyük  Karpathy, EleutherAI ve Allen Enstitüsü tarafından 2026'da araştırma kontrol noktalarını eğitmek için kullanılır.

## Gönder

Bakın .`outputs/skill-transformer-review.md`. Yetenek, önceki 13 dersin tümünde doğruluk için sıfırdan dönüştürücü uygulamasını gözden geçirir.

## Egzersizler

1. **Easy.**Çık .`code/main.py`Eğitimli modelinizin son aşamada geçerlilik kaybının 2.0'den az olduğunu kontrol edin.`max_steps`2000'den 5000'e kadar olan val kaybı gelişmeye devam ediyor mu?
2. **Medium.**Öğrenilen pozisyonsal yerleşimleri RoPE ile değiştirin.`MultiHeadAttention`Tren ve kontrol val kaybı en az o kadar düşük.
3. **Medium.**Örnekleme döngüsünde bir KV önbelleği uygulayın. 500 token oluşturun.
4. **Hard.**Bir sonraki bir ek bir token (MTP  DeepSeek-V3'den Multi-Token Tahmini) öngören ikinci bir baş ekleyin.
5. **Hard.**Blok başına tek FFN'yi 4 uzman MoE ile değiştirin. Router + top-2 yönlendirme.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy's tutorial repo" | Minimal decoder-only transformer training code, ~300 LOC; the canonical reference. |
| tinyshakespeare | "The standard toy corpus" | ~1.1 MB of text; every character-LM tutorial since 2015 uses it. |
| Tied embeddings | "Share input/output matrix" | LM head weight = transpose of token embedding matrix; saves parameters, improves quality. |
| bf16 autocast | "Training precision trick" | Run forward/back in bf16, keep optimizer state in fp32; standard since 2021. |
| Gradient clipping | "Stops spikes" | Cap global grad norm at 1.0; prevents training blowups. |
| Cosine LR schedule | "The 2020+ default" | LR ramps up linearly (warmup) then decays cosine-shaped to 10% of peak. |
| MFU | "Model FLOP Utilization" | Achieved FLOPs / theoretical peak; 40% dense, 30% MoE is strong in 2026. |
| Val loss | "Held-out loss" | Cross-entropy on data the model never saw; overfit detector. |

## Daha Fazla Okumak

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) klasik notlı uygulanma.
