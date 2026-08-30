# Görsel Autoregressive Modeling (VAR): Sonraki Ölçekli Tahmin

> VAR örnekleri, 1x1 token, sonra 2x2, sonra 4x4'ü, son çözünürlüğüne kadar tahmin eder. Her ölçek önceki ile koşullanır. 2024 makalesinde VAR görüntü üretimi için GPT tarzında ölçekleme yasalarına uyandığını ve aynı hesaplama bütçesinde DiT'yi yendiğini göstermiştir. Bu ders temel mekanizmayı inşa eder.

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 minutes

## Sorun

Autoregressive nesil, dil modelleme'de baskınlık gösterdi çünkü tahmin edilebilir bir ölçeklendirme yapıyordu: daha fazla hesaplama, daha fazla parametre, daha düşük karmaşıklık, daha iyi çıkışlar. Resim jenerasyonu 2024'e kadar iki ana AR girişimine sahipti: PixelRNN/PixelCNN (pixel-by-pixel) ve DALL-E 1 / Parti / MuseGAN (VQ-VAE kodlarında token-by-token).

Her ikisi de nesil sorunuyla mücadele etti. Pikseller ve jetonlar 2 boyutlu bir şebeke içinde düzenlenir, ancak AR modeli onları 1 boyutlu bir raster sırasında ziyaret etmelidir. Bir erken köşede pixel, görüntünün sonunda ne olacağı hakkında hiçbir fikre sahip değildir.

VAR, üretilen şeyi değiştirerek jenerasyon sırası sorunu çözmektedir. VAR, görüntü tokenlerini uzayda birbiriyle tahmin etmek yerine, artan çözünürlüklerle tüm bir görüntüyi tahmin eder. Adım 1: 1x1 token (toplam görüntü "cadet") tahmin edin. Adım 2: 2x2 token şebekesini (kırmızı özellikler) tahmin edin. Adım 3: 4x4 şebekesini tahmin edin. Adım K: son (H/8) x(W/8) şebekesini tahmin edin.

Her ölçek, önceki tüm ölçeklere (kötü olarak " ölçek sırası") ve kendi ölçekinde paralel olarak bakmaktadır.

## Anlaşım

### VQ-VAE Çok Ölçekli Tokenizer

VAR ' ın bir **multi-scale discrete tokenizer**. Bir resim x için , gittikçe daha yüksek çözünürlüklü token ağlarının bir dizi üretir:

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

Her z_k aynı kod defterini kullanır (tipik boyut 4096-16384). Her ölçekte belirtilme bağımsız değildir  her ölçekte kalanların toplamı f yeniden oluşturur:

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

Bu bir **residual VQ**K-1'in kaçırdığı şeyleri yakalar.

Çok ölçekli VQ tokenizer bir kez (VQGAN gibi) eğitilir ve sonra dondurulur. Tüm üreticiler tarafından yapılır.

### Sonraki Ölçekli Tahmin

Geliştirici model, önceki tüm ölçeklerden belirtileri gören ve bir sonraki ölçekteki belirtileri tahmin eden bir dönüştürücüdür.

Giriş sırası yapısı:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Konum yerleşimleri, hem ölçek indeksi hem de ölçek içindeki uzay pozisyonunu kodlar. Dikkat ölçek sırasındaki sebepsel bir şeydir: ölçek k'deki token, pozisyon (i, j) ölçek 1..k'deki tüm tokenlere ve ölçek k'deki tokenlerin kendisinde kullanılan ölçek içi sırada daha önce gelen tokenlere bakabilir (VAR, ölçek içi nedensellik olmadan sabit konum dikkati kullanır  bir ölçek içindeki tüm pozisyonlar paralel olarak öngörülür).

Eğitim kaybı: her bir ölçek k'de, tüm önceki ölçekli işaretler verildiğinde z_k belirtilerini tahmin edin. Ayrı VQ kodlarında çapraz entropi kaybı. "sequence" hariç GPT ile aynı yapı şimdi ölçekli yapılandırılmıştır.

### Nesil

Sonuç olarak:
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

K = 10 ölçek için, jenerasyon 10 dönüştürücü ileri geçiştir. Her geçiş tüm ölçeğini paralel olarak  bir ölçek içinde bir belirti başına bir autoregresyon üretir. 256x256 görüntü için bu yaklaşık 10 geçiş vs DiT'in 28-50'si.

### Neden Bir sonraki ölçek, bir sonraki ölçekten daha üstün

Üç yapısal kazanç:
1. **Coarse-to-fine aligns with natural image statistics.**İnsan görsel algısı ve görüntü verileri her ikisi de ölçek bağımlı düzenlilikleri gösterir: düşük frekanslı yapı istikrarlıdır ve tahmin edilebilir; yüksek frekanslı detaylar düşük frekanslı içeriğe bağlıdır.
2. **Parallel generation within scale.**GPT tarzındaki token AR'den farklı olarak, VAR tüm tokenleri bir adım içinde bir ölçekte üretir.
3. **No generation order bias.**K ölçeğinde tokens tüm ölçek k-1'i görür; erken tokenslerin geç bağlamın kullanılabilir olmadan önce kendini öne sürmesine zorlayan "sol-of" veya "üstün" bir önyargı yoktur.

### Ölçekleme Kanunu

Tian et al. VAR'ın, ImageNet'de FID için karmaşıklık için yaptığı gibi, FID için de güç hukuku ölçekleme eğri takip ettiğini gösterdi. Parametreyi ya da hesaplamayı ikiye katlamak hatayı güvenilir bir şekilde yarıya çıkarır. Bu, bu tür ölçeklendirme davranışlarını dil modelleri kadar temiz bir şekilde sergileyen ilk görüntü üreticisi modeliydi. Sonuç olarak, VAR ölçeği tahminleri mimarlık başına empirik tahminler değil, hesaplamalardan öngörülebilir hale gelir.

### Yayılma ile İlgili İlişkiler

VAR ve difüzyon aynı veri sıkıştırma hikayesini paylaşır: her ikisi de üretim sorunu daha kolay alt sorunlar dizisine ayrılır.

- Yayılma: yavaş yavaş gürültü ekle, bir adım atmayı öğren.
- VAR: yavaş yavaş çözünürlük ekleyin, bir sonraki ölçeği tahmin etmeyi öğrenin.

Bu iki sistem de, sorun boyunca farklı eksilerdir. Her ikisi de kontrol edilebilir koşullu dağılımlar verir. Empirik olarak VAR, sonuçlandırma (çık geçişler, tümleri bir ölçek içinde paralel) ve sınıf koşullu ImageNet'de DiT ile eşleşir veya yenir. Metin koşullu VAR (VARclip, HART) aktif bir araştırma yönüdür.

```figure
gx-var-next-scale
```

## Yapın

İçeri`code/main.py`Sen:
1. Küçük bir tane yap .**multi-scale VQ tokenizer**sentetik "resim" verileri (2 boyutlu Gaussian yüzükleri).
2. Tren A**VAR-style transformer**Sonraki ölçekte belirtiler tahmin etmek için.
3. Transformatörü 4 kez (4 ölçek) arayarak ve kodlama yoluyla örnekleme yapın.
4. Ölçekli düzenli eğitimlerin bir ölçek içinde paralel üretimi sağladığını kontrol edin.

Bu bir oyuncak uygulaması. Konu, ölçekli yapılandırılmış dikkat maskesinin ve ölçek içindeki paralel jenerasyonun gerçekten çalışmasını görmek.

## Gönder

Bu ders bize çok yararlı .`outputs/skill-var-tokenizer-designer.md` Çok ölçekli bir tokenizer tasarlama becerisi: ölçek sayısı, ölçek oranları, kod defteri boyutu, kalan paylaşım, dekodör mimarisi.

## Egzersizler

1. **Scale count ablation.**4, 6, 8, 10 ölçekle VAR'ı eğit. Yeniden inşaat kalitesini ve otomatik atılım sayısını ölçün. Daha fazla ölçek = daha ince kalıntılar = daha iyi kalite ama daha fazla atılım.

2. **Codebook size.**512 4096, 16384 kod defterleri ile tren simgesellerini hazırlayın.

3. **Parallel-within-scale check.**Eğitimli bir VAR için dikkat kalıbını açıkça ölçün. k ölçeğinde, model çapraz ölçekli pozisyonlara dikkat ediyor mu ama ölçek içi değil? maskelerin uygulanmasını kontrol edin.

4. **VAR vs DiT scaling.**Aynı ImageNet sınıf koşullı görevi için, eşleşen parametre bütçelerinde VAR ve DiT'yi eğit (örneğin, 33M, 130M, 458M).

5. **Text conditioning.**AdaLN üzerinden bir ek koşullama giriş olarak bir metin gömülmesi (CLIP birleştirilmiş) almak için VAR'ı genişlet. Bu HART tarifi. FID metin uyumlu örnekleme konusunda ne kadar iyileştirir?

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| VAR | "Visual AutoRegressive" | Image generation by next-scale prediction over a pyramid of VQ token grids |
| Next-scale prediction | "Predict coarser, then finer" | The model predicts tokens at increasing resolution scales, conditioning on all previous scales |
| Multi-scale VQ tokenizer | "Residual VQ" | VQ-VAE that produces K token grids of increasing resolution, with decoder summing all scales |
| Scale k | "Pyramid level k" | One of K resolution levels, from 1x1 at k=1 up to (H/p)x(W/p) at k=K |
| Parallel-within-scale | "One forward per scale" | All tokens at scale k are predicted in one transformer pass, not autoregressively |
| Causal-across-scales | "Scale-ordered attention" | Token at scale k can attend to all of scales 1..k but not scales k+1..K |
| Residual VQ | "Additive tokenization" | Each scale's tokens encode the residual left by lower scales; decoder sums all scale embeddings |
| VAR scaling law | "Image GPT scaling" | FID follows a predictable power law in compute, like language models' perplexity |
| HART | "Hybrid VAR + text" | Text-conditional VAR variant combining MaskGIT-style iterative decoding with VAR's scale structure |
| Scale position embedding | "(scale, row, col) triple" | Positional encoding carries both the scale index and spatial coordinates within the scale |

## Daha Fazla Okumak

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) VAR kağıdı, kanonik referans
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) DiT, difüzyon karşılaştırma başlangıç
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841)VQGAN, tokenizer ailesinin VAR'ın çok ölçekli tokenizer uzantıları
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) VQ-VAE, ayrı görüntü işaretlemesinin temeli
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) metin şartlı VAR
