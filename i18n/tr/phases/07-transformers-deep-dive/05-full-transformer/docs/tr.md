# Tam Transformer  Kodlayıcı + Dekodör

> Dikkat yıldızdır. Diğer her şey  kalıntılar, normallaşma, ileriye aktarma, çapraz dikkat  derinlere yığılmasına izin veren bir heykel.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## Sorun

Tek bir dikkat katmanı bir özellik çıkarıcıdır, bir model değildir. Bir katman için bir matmul dil için yeterli kapasite değildir. Doğru tesisat olmadan derinlik  ve derinlik kesintiler gerekir.

2017 Vaswani kağıdı, bir dikkat katmanını yığılabilir bir blok haline getiren altı tasarım kararını paketledi.  sadece kodlayıcı (BERT), sadece dekodör (GPT), sadece kodlayıcı-dekodör (T5) 'den bu yana her transformatör aynı iskeleti miras alır. 2026 yılında bloklar (RMSNorm, SwiGLU, pre-norm, RoPE) geliştirildi, ancak iskelet aynıdır.

Bu ders iskelet. Sonraki dersler onu  06 kodlayıcılar için, 07 dekodörler için, 08 kodlayıcı-dekodör için uzmanlaştırır.

## Anlaşım

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### Altı parça

1. **Embedding + positional signal.**Tokens → vectors. RoPE (modern) veya sinusoidal (klasik) yoluyla enjekte edilen pozisyon.
2. **Self-attention.**Her pozisyon diğerine bakıyor, kodlayıcılarda maskeli.
3. **Feed-forward network (FFN).**Konum açısından iki katlı MLP: `W_2 · activation(W_1 · x)`- Default olarak 4× genişleme oranı.
4. **Residual connection.** `x + sublayer(x)`Bu olmadan, gradientler 6 katınlıktan sonra kaybolur.
5. **Layer normalization.** `LayerNorm`veya `RMSNorm`Geri kalan akışı istikrarlı hale getirir.
6. **Cross-attention (decoder only).**Sorgular dekodörden, anahtarlardan ve kodlayıcı çıkış değerlerinden gelir.

Bir blok boyunca bir vektör akışını izleyin: dikkat pozisyonlar arasında karışır, kalanı ileriye taşıyor, FFN onu dönüştürüyor ve norm akışı istikrarlı tutar.

```figure
transformer-block
```

### Kodlayıcı blok (BERT, T5 kodlayıcı tarafından kullanılır)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Kodlayıcı iki yönlü, maskeli değil.

### Dekodör blok (GPT, T5 dekodör tarafından kullanılır)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Dekoder blok başına üç alt katman vardır. Orta bir  çapraz dikkat  sadece bilgi kodlayıcıdan dekodöre akıyor. Saf bir dekoder-tek mimaride (GPT) çapraz dikkat atılır ve sadece kendini dikkat + FFN maskeli vardır.

### Normalden önce vs. normalden sonra

Orijinal kağıt: `x + sublayer(LN(x))`vs `LN(x + sublayer(x))`. Post-normal 2019 civarında favori kaybetti  dikkatli ısınma olmadan derin bir şekilde eğitmek daha zordur.`LN`* öncesinde* alt katman) 2026'da varsayılan: Llama, Qwen, GPT-3+, Mistral hepsi kullanıyor.

### 2026'da modernleştirilmiş blok

Vaswani 2017 LayerNorm + ReLU'yu gönderdi. Modern asmalar her ikisini de değiştirdi.

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices, total params match) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

RMSNorm, hesaplama tasarrufu sağlayan ve en azından empirik olarak aynı derecede sabit olan LayerNorm'un ortalama merkezini düşürür.`Swish(W1 x) ⊙ W3 x`) Llama, PaLM ve Qwen makalelerinde ReLU/GELU FFN'i yaklaşık 0,5 puan daha iyi performans göstermektedir.

### Parametre sayısı

Bir blok için `d_model = d`ve FFN genişlemesi `r`- ...

- MHA:`4 · d²`(Q, K, V, O projeleri)
- FFN (SwiGLU): `3 · d · (r · d)`- Evet .`3rd²`
- Normalar: önemsiz

- Evet .`d = 4096, r = 2.6, layers = 32`(yaklaşık Llama 3.8B), toplam: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`(daha yerleştirmeler ve baş) Yayınlanan eşleşmeler sayılır.

## Yapın

### Adım 1: yapı taşları

Küçük olanı kullanırdım .`Matrix`Ders 03'ten sınıflandırılmış (bağımsızlık için bu dosyaya kopyalandı):

- `layer_norm(x, eps=1e-5)` ortalama çıkar, std ile böl.
- `rms_norm(x, eps=1e-6)` RMS ile bölün.
- `gelu(x)`ve `silu(x) * W3 x`- Evet.
- `ffn_swiglu(x, W1, W2, W3)`- Evet .
- `encoder_block(x, params)`ve `decoder_block(x, enc_out, params)`- Evet .

Bakın .`code/main.py`Tam telüke için.

### Adım 2: 2 katmanlı bir kodlayıcı ve 2 katmanlı bir dekodör kablosu

Onları yığ. Kodlayıcı çıkışını her dekodere çapraz dikkatle aktar. Çıkış projesinden önce son bir LN ekle.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Adım 3: Oyuncak örneği üzerinde ileriye koş

6 simgelik bir kaynak ve 5 simgelik bir hedef gönderin.`(5, vocab)`Bu ders mimarlık hakkında, kayıp değil.

### Adım 4: RMSNorm + SwiGLU'da değişim

LayerNorm ve ReLU-FFN'i RMSNorm ve SwiGLU ile değiştirin. Şekillerin halen eşleşmesini onaylayın. Bu bir fonksiyon değiştirmesi ile 2026 modernizasyonu.

## Kullan

PyTorch/TF referans uygulamalar: `nn.TransformerEncoderLayer`- Evet .`nn.TransformerDecoderLayer`Ama 2026 üretim kodunun çoğu kendi blokunu kullanıyor çünkü:

- Flash Dikkat, dikkat içindeki dikkatle çağrılır, değil `nn.MultiheadAttention`- Evet .
- GQA / MLA'nın adı stdlib referansında bulunmuyor.
- RoPE, RMSNorm, SwiGLU PyTorch'ın öntanımlı özellikleri değil.

HF `transformers`okumalısınız: `modeling_llama.py`Bu sadece 2026'da kodlayıcı blok. 500 satırlık ve bir kez yürümek için değer.

**Encoder vs decoder vs encoder-decoder — when to pick:**

| Need | Pick | Example |
|------|------|---------|
| Classification, embeddings, QA over text | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

Sadece dekodörle kazanılan dil, en temiz ölçeklendirme ve hem anlayış hem de üretimi ile başa çıkıldığı için. Encoder-decoder hala giriş açık bir "kaynak dizisi" kimliğine sahip olduğunda en iyisidir (çevirim, konuşma tanıma, yapılandırılmış görevler).

## Gönder

Bakın .`outputs/skill-transformer-block-reviewer.md`. Yetenek, 2026'da varsayılan standartlara göre yeni bir transformatör blok uygulamasını gözden geçirir ve eksik parçaları (pre-norm, RoPE, RMSNorm, GQA, FFN genişleme oranı) işaretler.

## Egzersizler

1. **Easy.**Encoder_block'daki parametreleri  olarak say`d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. Bloku uygulayarak ve `sum(p.numel() for p in block.parameters())`- Evet .
2. **Medium.**Post-norm'dan pre-norm'a geçin. Her ikisini de başlatın ve 12 katmanlı bir şekilde rastgele giriş üzerine aktivasyon normunu ölçün. Post-norm'un aktivasyonları patlamalıdır; pre-norm'lar sınırlı kalmalıdır.
3. **Hard.**Oyuncak kopyası görevinde 4 katmanlı bir kodlayıcı-dekodör uygula (kopya `x`100 adım tren. Kayıp rapor. RMSNorm + SwiGLU + RoPE değişimi kayıp düşüyor mu?

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Block | "One transformer layer" | Stack of norm + attention + norm + FFN, wrapped in residual connections. |
| Residual | "Skip connection" | `x + f(x)` output; enables gradient flow through deep stacks. |
| Pre-norm | "Normalize before, not after" | Modern: `x + sublayer(LN(x))`. Trains deeper without warmup gymnastics. |
| RMSNorm | "LayerNorm without the mean" | Divide by RMS; one less op, same empirical stability. |
| SwiGLU | "The FFN everyone switched to" | `Swish(W1 x) ⊙ W3 x → W2`. Beats ReLU/GELU on LM ppl. |
| Cross-attention | "How the decoder sees the encoder" | MHA with Q from decoder, K/V from encoder outputs. |
| FFN expansion | "How wide the middle MLP is" | Ratio of hidden-size to d_model, usually 4 (LayerNorm) or 2.6 (SwiGLU). |
| Bias-free | "Drop the +b terms" | Modern stacks omit biases in linear layers; slight ppl improvement, smaller model. |

## Daha Fazla Okumak

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) orijinal blok özellikleri.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) neden pre-norm, post-norm'u derinden yendi.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) SwiGLU kağıdı.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) kanonik 2026 sadece dekodör blok.
