# Çok Başlı Dikkat

> Bir dikkat başı bir arada bir ilişkiyi öğrenir. Sekiz baş sekiz öğrenir.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 minutes

## Sorun

Tek bir kendi dikkat başı bir dikkat matrisi hesaplar. Bu matris bir tür ilişkiyi yakalar  genellikle eğitim sinyalleri ne olursa olsun kayıpları en aza indirgenir. Verilerinizde konu-ketim anlaşması, eş-referans, uzun mesafeli konuşma ve sentaksik parçalanma varsa, tek bir baş onları tek yumuşak maksimum dağılımına ayırır ve sinyalin yarısını kaybeder.

2017 Vaswani makalesinden alınan düzeltme: her biri kendi Q, K, V projeksiyonlarıyla paralel olarak birkaç dikkat fonksiyonu çalıştırır ve çıkışları birleştirir. Her baş daha küçük boyut alt alanında çalışır.`d_model / n_heads`Toplam parametreler aynı kalır.

Çok başlı dikkat, 2026 gemilerindeki her transformatörün varsayılanıdır. Tek argüman * kaç* başlı ve anahtarların ve değerlerin projeksiyonları paylaşıp paylaşmadığı hakkında (Grouped-Query Attention, Multi-Query Attention, Multi-head Latent Attention).

## Anlaşım

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Split.**Al .`X`şekli ile`(N, d_model)`- Her biri şekilinde Q, K, V'ye kadar.`(N, d_model)`- Yeniden değiştir .`(N, n_heads, d_head)`nerede`d_head = d_model / n_heads`- Transposer `(n_heads, N, d_head)`- Evet .

**Attend in parallel.**Her başın içinde bir nokta üretici dikkatini çalıştır.`(N, d_head)`Başlar yerleşimlerin farklı alt alanlarında çalışır ve dikkat hesaplama sırasında hiç konuşmazlar.

**Concatenate and project.**Yüklü başları geri dön .`(N, d_model)`ve öğrenilmiş bir çıkış matrisine çarpır `W_o`şekli ile`(d_model, d_model)`- Evet .`W_o`Başların karışması.

**Why it works.**Her baş, temsil bütçesi için diğerleriyle rekabet etmeden uzmanlaşabilmektedir. 2019  2024'ten kalma araştırma çalışmaları farklı baş rollerini göstermektedir: pozisyonel başlar, önceki simgeye katılan başlar, kopya başları, isimli varlık başları, indüksiyon başları (konekst içi öğrenmenin temeli olan).

**The 2026 lineage of variations:**

| Variant | Q heads | K/V heads | Used by |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA, modern varsayılan bir yöntemdir çünkü KV-cache belleğini `N/G`MLA, K/V'yi gizli bir alanlara sıkıştırarak daha ileri gider, sonra hesaplama zamanında geri projeksine geçerek  FLOP'ler maliyetini artırır, çok daha fazla bellek tasarruf eder.

```figure
multihead-split
```

## Yapın

### Adım 1: Tek başlı dikkatimizden başları ayırın

Alın .`SelfAttention`2. dersten sonra bir çift bölünme/konkat ile sarın.`code/main.py`bir numpy uygulaması için; mantık:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

Birini yeniden şekillendirip birini transpose et.`nn.MultiheadAttention`- Evet .

### Adım 2: Başlık başına dikkat çekilen ürünleri ölçeklendirin

Her baş kendi parçalarını alır. Dikkat bir matmul olur.

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

Gerçek donanımlı .`Qh @ Kh.transpose(...)`Bir tane .`bmm`GPU ' nun görebileceği tek bir parça şekil .`(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`Başları eklemek ücretsizdir.

### Adım 3: Gruplandırılmış Sorgu Dikkat Variansı

Sadece anahtar ve değer projeleri değişir.`n_heads`gruplar; K ve V get `n_kv_heads < n_heads`gruplar ve tekrar tekrarlar:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

Bu hafıza tasarrufu yapar çünkü sadece`n_kv_heads`KV önbelleğinde kopyalar var, değil `n_heads`Llama 3 70B 64 sorgu başlığı ile 8 KV başlığı  8× önbelleği küçültücü kullanıyor.

### Dördüncü adım: Her başın ne öğrendiklerini araştır

MHA'yı 4 başlı kısa cümle ile çalıştır.`(N, N)`Farklı başlar farklı yapıları seçerken rastgele başlangıç yaparak göreceksiniz. Bu kısmen sinyal, kısmen de alt uzaylarda dönüm simetrisidir.

## Kullan

PyTorch'de tek satırlı versiyon:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+'den itibaren GQA:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**How many heads?**2026'daki üretim modellerinden gelen basamak kuralları:

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head`Bu, bir başın "görme" yeteneğinin birimidir. 32'nin altına düşer ve başlar ölçekleme faktörüne karşı savaşmaya başlar.`sqrt(d_head)`256'den fazla bir iş yaparsanız "çok küçük uzman" avantajını kaybedeceksiniz.

## Gönder

Bakın .`outputs/skill-mha-configurator.md`. Bu beceri, yeni bir transformatör için baş sayısını, kv baş sayısını ve projeksiyon stratejisini, parametre bütçesi, dizi uzunluğu ve dağıtım hedefi vererek önerir.

## Egzersizler

1. **Easy.**MHA ' dan alın .`code/main.py`ve değişim .`n_heads`1 ila 16 arasında `d_model=64`Bir katmanlı modelin kaybını sentetik kopyalama işinde planlayın.
2. **Medium.**MQA uygulaması (tüm sorgu başlıkları arasında paylaşılan bir KV başlığı). Parametre sayısının ne kadar düşeceğini ölçmek vs. tam MHA. N=2048 için sonuçta KV-cache boyutunun ne kadar azaldığını hesaplayın.
3. **Hard.**Çok başlı Latent Dikkatin küçük bir versiyonunu uygulayın: K, V'yi bir sıralama ile sıkıştırın.`r`- KV'de saklanıp dikkat zamanı ile sıkıştır.`r`Kaş belleği tam MHA'nın 1/8'inden aşağı geçirken kalite doğrulama işleminin 1 bitinin içinde kalır mı?

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Head | "A single attention circuit" | One Q/K/V projection of dimension `d_head = d_model / n_heads` with its own attention matrix. |
| d_head | "Head dimension" | Per-head hidden width; almost always 64 or 128 in production. |
| Split / combine | "Reshape tricks" | `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose around attention. |
| W_o | "Output projection" | `(d_model, d_model)` matrix applied after concatenating heads; where heads mix. |
| MQA | "One KV head" | Multi-Query Attention: single shared K/V projection. Smallest KV cache, some quality loss. |
| GQA | "The default since Llama 2" | Grouped-Query Attention with `n_kv_heads < n_heads`; repeats to match Q. |
| MLA | "DeepSeek's trick" | Multi-head Latent Attention: K,V compressed to low-rank latent, decompressed at attend time. |
| Induction head | "The circuit behind in-context learning" | A pair of heads that detect previous occurrences and copy what followed them. |

## Daha Fazla Okumak

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) orijinal çok başlı özellik.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) MQA kağıdı.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) eğitimden sonra MHA'yı GQA'ya nasıl dönüştürülecek.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) MLA ve neden cache belleğinde MHA/GQA'yı yendi.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) Başların ne yaptığını mekanizma olarak gör.
