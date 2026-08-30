# Konut kodlaması  Sinusoidal, RoPE, ALiBi

> Dikkat, permutasyon-invariant. "Kedi çarşaf üzerinde oturdu" ve "mat üzerinde sat kedi" pozisyon sinyalsiz aynı çıkış üretir.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 minutes

## Sorun

Skalalı nokta ürün dikkat, sırayla kördür.`softmax(Q K^T / √d) V`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             `X`Dikkatin içinde hiçbir şey pozisyonu önemsemedi.

Bu bir kelime çantası modelinde bir hata değil. dil, kod, ses, video için  emir anlamı taşıyan her şey  ölümcül.

Bu çözüm, yerleşimlere bir şekilde yerleştirmek.

1. **Absolute sinusoidal**(Vaswani 2017) Ekle `sin/cos`Basit, öğrenilmez, eğitimli uzunluklardan kötü bir şekilde uzaklaştırılır.
2. **RoPE — Rotary Position Embeddings**(Su 2021). Q ve K vektörlerini pozisyonuna orantılı bir açıyla döndürün.
3. **ALiBi — Attention with Linear Biases**(Press 2022). Tüm yerleşimleri atlayın; mesafeye göre dikkat puanlarına baş başına bir çizgi ceza ekleyin. Mükemmel uzunluk ekstrapolasyonu.

2026 itibariyle, esasen her sınır açık modeli RoPE kullanır: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi.

## Anlaşım

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### Kesin sinusoidal

Bir sabit matris önceden hesaplayın `PE`şekli ile`(max_len, d_model)`- ...

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

O zaman ...`X' = X + PE[:N]`Her boyut farklı bir frekansta sinusoid.`max_len`: hiçbir şey modelin sadece pozisyon 02047'i gördüğünde 2048 pozisyonunda ne olduğunu söylemedi.

### RoPE

Bir çift boyut için Q ve K vektörlerini (kaynak değil) döndürün `(2i, 2i+1)`- ...

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

Aynı dönümsellik pozisyonlu anahtarlara uygulayın `pos_k`- Dots ürünü`q'_m · k'_n``(m - n)`- Yalnız.**the attention score depends only on the relative distance**- ...eğer de dönüşüm mutlak pozisyonlarda kilitlenmiş olsa.

RoPE'yi uzatmak: `base`Llama 3'ün 8K'den 128K'ye kadar uzandığı bu şekilde.

### ALiBi

Dikkatin doğrudan puanlarını ayır.

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

Nerede ?`m_h`başı için özel bir eğimdir (örneğin `1 / 2^(8·h/H)`) Yakınlıklı tokenler artırılır; uzak tokenler cezalandırılır. Eğitim zamanı maliyeti yoktur. Kağıt uzunluk ekstrapolasyonunun sinusoidal olduğunu ve RoPE'ye orijinal eğitim uzunluğunda eşleştiğini gösterir.

### 2026'da neyi seçmeliyiz?

| Variant | Extrapolation | Training cost | Used by |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | poor | free | original transformer, early BERT |
| Learned absolute | none | tiny | GPT-2, GPT-3 |
| RoPE | good with scaling | free | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | excellent | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | excellent | free | BLOOM, MPT, Baichuan |

RoPE, yapısını değiştirmeden dikkat çekerek, nispet konumunu kodlayarak ve `base`hiperparametre uzun bağlamlı ince ayarlama için temiz bir düğme verir.

```figure
rope-explorer
```

## Yapın

### Adım 1: Sinusoidal kodlama

Bakın .`code/main.py`- 4 satırlık hesaplama:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

İlk dikkat katmanından önce bunu yerleştirme matrisine ekleyin.

### Adım 2: Q, K'ye uygulanan RoPE

RoPE, Q ve K'de yerinde çalışır. Her çift dimmer için:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

Önemli: aynı işlevi pozisyonda Q ' e uygulayın `m`Ve K pozisyonda `n`- Dots ürünü bir `cos((m-n)·θ_i)`Dikkat, nispet konumunu ücretsiz öğrenir.

### Adım 3: ALiBi yamaçları ve tarafsızlığı

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

Ekle`bias[h]`- ...`(seq_len, seq_len)`dikkat puanı matrisi başı `h`, sonra softmax.

### Adım 4: RoPE'nin nispet mesafe özelliğini doğrulayın

İki rastgele vektör seçin .`a, b`- Dönüşüm .`(pos_a, pos_b)`- Sonra da .`(pos_a + k, pos_b + k)`Bu özellik RoPE'nin tüm noktasıdır  mutlak sıfırlama ile değişmez, sadece nispetik boşluk önemlidir.

## Kullan

PyTorch 2.5+ gemileri RoPE hizmetleri `torch.nn.functional`Çoğu üretim kodu kullanıyor.`flash_attn`veya `xformers`RoPE dikkat çekirdeğinin içinde uygulanır.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Long-context tricks in 2026:**

- **NTK-aware interpolation.**Yeniden ölçeklendirme`base`- ...`base * (scale_factor)^(d/(d-2))`4K'den 16K'ye kadar uzanırken.
- **YaRN.**Daha akıllı bir interpolasyon, uzun bağlamlarda dikkat entropiyi korur. Llama 3.1 128K kullanıyor.
- **LongRoPE.**Microsoft'un 2024 yöntemi, evrimsel arama kullanarak boyut boyutlarındaki faktörleri seçer.
- **Position interpolation + fine-tuning.**Sadece pozisyonları uzatma faktörü ile küçültüp 15B token için ince ayarlayın.

## Gönder

Bakın .`outputs/skill-positional-encoding-picker.md`. Bu beceri, hedef bağlam uzunluğu, ekstrapolasyon ihtiyaçları ve eğitim bütçesi göz önüne alındığında yeni bir model için bir kodlama stratejisini seçer.

## Egzersizler

1. **Easy.**Sinusoidal çizgiyi çiz .`PE` için bir ısı haritası olarak matris`max_len=512, d=128`"Dimension index büyüdükçe çizgiler daha genişleşiyor" şeklini onaylayın.
2. **Medium.**NTK-a karşı duyarlı RoPE ölçeklendirme uygulayın. 256 uzunluklı sekanslarda küçük bir LM çalıştırın, sonra 1024 uzunlukta ölçeklendirme ile ve olmaksızın test edin. Kafasını karışıklık ölçün.
3. **Hard.**ALiBi ve RoPE'yi aynı dikkat modülüne uygulayın. 4 katlı bir transformatörü bir kopyalama görevi üzerinde uzunluğu 512 olan dizilerle çalıştırın. Test sırasında 2048'e ekstrapolasyon yapın.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Positional encoding | "Tells attention about order" | Any signal added to embeddings or attention that encodes position. |
| Sinusoidal | "The original one" | `sin/cos` at geometric frequencies added to embeddings; doesn't extrapolate. |
| RoPE | "Rotary embeddings" | Rotate Q, K by position-dependent angle; dot product encodes relative distance. |
| ALiBi | "Linear bias trick" | Add `-m·\|i-j\|` to attention scores; no embedding needed, great extrapolation. |
| base | "RoPE's knob" | The frequency scaler in RoPE; increase to extend context at inference. |
| NTK-aware | "A RoPE scaling trick" | Rescale `base` so high-frequency dims aren't squeezed when context expands. |
| YaRN | "The fancy one" | Per-dimension interpolation+extrapolation that preserves attention entropy. |
| Extrapolation | "Works beyond trained length" | Can the position scheme serve correct output past `max_len` seen in training? |

## Daha Fazla Okumak

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) orijinal sinusoidal.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) RoPE kağıdı.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) ALiBi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) RoPE ölçeklendirme.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)Meta'nın Llama 2 uzun bağlamlı makalesi.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) Phi-3-Long tarafından kullanılan Microsoft yöntemi ve Use It bölümünde alıntılanmıştır.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) Her RoPE ölçekleme sisteminin üretim derecesi uygulanması (öntemli, doğrusal, dinamik, YaRN, LongRoPE, Llama-3).
