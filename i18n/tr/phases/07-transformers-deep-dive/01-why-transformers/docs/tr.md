# Neden Transformers  RNN'lerle Sorunlar

> RNN'ler bir seferde birer tokeni işliyor. Transformerler tüm tokeni bir seferde işliyor. Bu tek mimari bahis 2017'den sonra derin öğrenimdeki her ölçekleme eğriğini değiştirdi.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 minutes

## Sorun

2017'den önce, gezegen üzerindeki her en son sekans modeli  dil, çeviri, konuşma  bir geri dönüştürücü sinir ağıydı. LSTM ve GRU'lar yarım on yıl boyunca ImageNet eşdeğer çeviri referansları kazandı.

Üç ölümcül zayıflıkları vardı.`t+1`Gizli bir durumun belirtileri için gerekli .`t`1.024 tokenlik bir dizi, bir GPU'da 1.024 seri adım anlamına gelir.

Kayıp gradientler, 50 token geriye gelen bilginin zaten 50 doğrusal olmayan birimle sıkıştırıldığını gösterir. Gated recurrent units (LSTM, GRU) kırıklığı yumuşatır ama asla ortadan kaldırmaz. Uzun mesafeli bağımlılıklar  "Geçen yaz Kyoto'ya giden bir uçakta okuduğum kitap..."  rutin olarak başarısız oldu.

Sıkı genişliğin gizli durumları, kodlayıcı, kaynakın 5 token veya 500 olması önemli değil; şişe boynuzunun aynı şekli vardır.

2017'de yayınlanan "Eğer İhtiyacınız varsa Dikkat" makalesinde radikal bir şey önerildi: Tekrarlanmayı tamamen bırakın. Her pozisyonun diğer pozisyonlara paralel olarak dikkat etmesine izin verin.

Sonuç 2026 yılına kadar her modalite hakimdir. Dil (GPT-5, Claude 4, Llama 4), görme (ViT, DINOv2, SAM 3), ses (Sippeder), biyoloji (AlphaFold 3), robotik (RT-2). Aynı blok, farklı girişler.

## Anlaşım

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Recurrence as a bottleneck.**RNN hesaplar `h_t = f(h_{t-1}, x_t)`Her adım önceki adımdan bağlı.`h_5`Daha önce`h_4`10.000'den fazla paralel çekirdekli modern GPU'larda bu, uzun bir dizide silikonun %99'unu atıyor.

**Attention as a broadcast.**Kendine dikkat hesaplamaları `output_i = sum_j(a_ij * v_j)`Her çift için .`(i, j)`Tüm N×N dikkat matrisi bir partili matmul'e doldurulur.

**The speedup is not a constant.**Bu `O(N)`seri derinliği ve `O(1)`Serial derinliği. pratikte, transformatörler eşleşen donanımlarda N=512'de 510x daha hızlı çalıştırılır ve boşluk, `O(N²)`Akılda tutulan hafıza duvarı (Flash Attention'ın daha sonra düzelttiği  12. dersi gör).

**What transformers cost.**Dikkat hafıza ölçekleri `O(N²)`128K bağlamı için kaydırıcı pencereler, RoPE ekstrapolasyonu, Flash dikkat kapakları veya doğrusal dikkat çeşitleri gerekir.`O(N)`Zaman ve hafıza her iki tarafta; transformatörler zamanla hafıza alışverişinde bulunur ve paralellik yoluyla zamanı geri kazanır.

**The inductive bias shift.**RNN'ler yerellik ve yenilikçilik kabul eder. Transformatörler hiçbir şeyi kabul etmez.  her çift dikkat için bir adaydır. Bu nedenle transformatörler iyi eğitilmek için daha fazla veriye ihtiyaç duyarlar, ancak bir kez daha ölçeklendirirler. Chinchilla (2022) bunu resmileştirdi: yeterli token verildiğinde, bir transformatör her zaman eşit parametreler sayısının RNN'ini yener.

```figure
rnn-vs-parallel
```

## Yapın

Burada sinir ağı yok. Biz çekirdek boğazını sayısal olarak simüle ediyoruz. Böylece dizüstü bilgisayarınızda boşluğu hissedebilirsiniz.

### Adım 1: Seri derinliğini ölç

Bakın .`code/main.py`Birini bir dizi ekleme zinciri olarak kodlarız (serial, RNN gibi). Birini paralel bir azaltma olarak kodlarız (ekleyici, dikkat gibi). Aynı matematik, farklı bağımlılık grafiği.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

RNN sürümü O(N) ve tek bir CPU borusu. saf Python'da bile dikkat biçimindeki azaltma uzunluğu ≥ 1000'e geçiyor çünkü Python'un `sum()`C'de uygulanır ve adım başına tercümanlık ödemesi olmadan tekrarlanır.

### Adım 2: teorik işlemleri sayın

Her iki algoritma da N ekler. Fark * bağımlılık derinliği *: bir sonraki başlamadan önce kaç işlem sıradan gerçekleşmelidir. RNN derinliği = N. Dikkat derinliği = log(N) bir ağaç azaltma ile veya 1 paralel tarama ile.

### Adım 3: Uzun diziler üzerinde empiriyel ölçeklendirme

O  N) boşluğu görülebilen bir zamanlama tablosunu yazdırırırız. 2026 Mac dizüstü bilgisayarında, 1000 element altındaki diziler ölçülmek için çok hızlıdır. 100,000'in dizileri temiz bir çizgi tarama gösterir. Bunu 16 384 token transformatörüne 12 katman LSTM eşdeğerle ölçeyin ve eğitim duvar saati neden 2016'da bir engelleyici olduğunu göreceksiniz.

## Kullan

2026'da RNN'i ne zaman seçmeliyiz?

| Situation | Pick |
|-----------|------|
| Streaming inference, one token at a time, constant memory | RNN or state-space model (Mamba, RWKV) |
| Very long sequences (>1M tokens) where attention memory explodes | Linear attention, Mamba 2, Hyena |
| Edge device with no matmul accelerator | Depthwise-separable RNN still wins on FLOPs/watt |
| Anything else (training, batched inference, context up to 128K) | Transformer |

Mamba gibi devlet- uzay modelleri (SSM) esasen her ikisinin de en iyisini veren yapılandırılmış parametreleşme ile RNN'lerdir: `O(N)`Sıkıntılı bir şekilde, bir sürü farklı yöntemler kullanılır.

## Gönder

Bakın .`outputs/skill-architecture-picker.md`. Yetenek, uzunluk, geçiş gücü ve eğitim bütçesi kısıtlamaları göz önünde bulundurularak yeni bir dizi sorunu için bir mimari seçer.

## Egzersizler

1. **Easy.**Al .`rnn_style`-`code/main.py`Ve saklı durumların uzunluğu 64 vektörü ile değiştirmek.
2. **Medium.**Temiz Python'da paralel bir önbellek-sümayı (Hillis-Steele taraması) uygulayın.
3. **Hard.**Dikkat biçimindeki azaltmayı GPU'ya PyTorch'e aktarın.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Recurrence | "RNNs are sequential" | Computation where step `t` depends on step `t-1`, forcing serial execution along the time axis. |
| Serial depth | "How deep the graph is" | Longest chain of dependent ops; bounds wall-clock even on infinite hardware. |
| Attention | "Let tokens look at each other" | Weighted sum `sum_j a_ij v_j` where `a_ij` comes from a similarity score between positions i and j. |
| Context window | "How much the model sees" | Number of positions an attention layer can take as input; quadratic memory cost scales here. |
| Inductive bias | "Assumptions baked into the architecture" | Prior about what the data looks like; CNNs assume translation invariance, RNNs assume recency. |
| State-space model | "RNN with algebra behind it" | Recurrence parameterized for parallel training via structured state-space matrices. |
| Quadratic bottleneck | "Why context costs so much" | Attention memory = `O(N²)` in sequence length; Flash Attention hides the constants, not the scaling. |

## Daha Fazla Okumak

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) ana akım NLP'de tekrarlanmayı öldüren makale.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) dikkat doğduğu yer, RNN'ye bağlanmış.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) Kayıt için orijinal LSTM kağıdı.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) Transformatörlere modern tekrarlayıcı cevap.
