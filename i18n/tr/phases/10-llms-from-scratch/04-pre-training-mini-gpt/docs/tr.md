# Mini GPT'nin önceden eğitilmesi (124M Parametre)

> GPT-2 Small'ın 124 milyon parametre vardır. 12 transformatör katmanı, 12 dikkat başlığı ve 768 boyutlu yerleşim. Bir GPU'da birkaç saat içinde sıfırdan eğitilebilir. Çoğu insan bunu asla yapmaz. Önceden eğitilmiş kontrol noktalarını kullanırlar. Ama eğer kendiniz eğitilmezseniz, ürünlerini inşa ettiğiniz modelin içinde neler olup bittiğini anlamayacaksınız.

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10, Lessons 01-03 (Tokenizers, Building a Tokenizer, Data Pipelines)
**Time:** ~120 minutes

## Öğrenme Hedefleri

- GPT-2 mimarisi (124M parametresi) tamamını sıfırdan uygula: token yerleştirmeler, pozisyon yerleştirmeler, transformatör blokları ve dil model başlığı
- GPT modeli bir metin korpusunda, çapraz entropik kaybı ile bir sonraki belirti tahminini kullanarak çalıştır
- Temerate örneği ve üst-k/üst-p filtresi ile autoregressive metin oluşturulmasını uygula
- Eğitim kaybı eğriliklerini izle ve modelin tutarlı dil kalıplarını öğrendiğini doğrulay

## Sorun

Transformatörün ne olduğunu biliyorsunuz. Şekilleri okudunuz. "İlgilenmek tek ihtiyacınız var" diye okuyabilir ve bir tahtaya "Kendi Başlı Dikkat" ile etiketlenmiş kutular çizersiniz.

Bu hiçbir şey, bir model metin oluşturduğunda ne olduğunu anlamanızı anlamına gelmez.

GPT-2 Small'da 124.438.272 parametre bulunmaktadır. Her biri bir eğitim döngüsü ile ayarlandı: ileri geçiş, hesap kaybı, geri geçiş, güncelleme ağırlıkları. 12 transformatör blok. Her blokta 12 dikkat başlığı. 768 boyutlu bir yerleşim alanı. 50.257 tokenin bir kelime kaynağı. Model bir token oluşturduğunda, tüm 124 milyon parametre bir tek matris çarpma zincirine katılır ve bir dizi token kimliği alır ve bir sonraki token üzerinde olasılık dağılımını üretir.

Eğer bunu hiç kendi yapmadıyorsanız, kara kutuyla çalışıyorsunuz. API'yi kullanabilirsiniz. Düzeltme yapabilirsiniz. Ama bir şey ters gittiğinde -- model halüsinasyon yaparken, kendini tekrarlarken, talimatları izlemeyi reddederken -- neden için zihinsel bir modeliniz yok.

Bu ders GPT-2'yi sıfırdan küçük olarak inşa ediyor. PyTorch'de değil. Numpy'de. Her matris çarpımı görünür. Her gradient koduyla hesaplanır.

## Anlaşım

### GPT Mimarlığı

GPT, autoregressive dil modelidir. "Autoregressive" demek ki, bir seferde bir token oluşturur, her biri önceki tüm tokenlere koşullanmıştır.

İşte token ID'lerden sonraki token olasılıklarına kadar tüm hesaplama grafikleri:

1. İşaret kimlikleri geliyor. Şekil: (batch_size, seq_len).
2. İşaret gömülme araması. Her kimlik 768 boyutlu bir vektör haritası. Şekil: (batch_size, seq_len, 768).
3. Yer yerleştirme araması. Her konum (0, 1, 2, ...) 768 boyutlu bir vektöre haritası. Aynı şekil.
4. Token yerleştirmeler + pozisyon yerleştirmeler ekleyin.
5. 12 transformatör blokundan geç.
6. Son katman normallaştırma.
7. Süzlük boyutuna çizgi projeksiyon. Şekil: (batch_size, seq_len, vocab_size).
8. - Muhtemelen, yumuşaklık.

Bu tüm model. Yıkışlar yok. Tekrarlanmak yok. Sadece yerleşimler, dikkat, geri dönüş ağları ve katman normları 12 kez yığılmış.

```mermaid
graph TD
    A["Token IDs\n(batch, seq_len)"] --> B["Token Embeddings\n(batch, seq_len, 768)"]
    A --> C["Position Embeddings\n(batch, seq_len, 768)"]
    B --> D["Add"]
    C --> D
    D --> E["Transformer Block 1"]
    E --> F["Transformer Block 2"]
    F --> G["..."]
    G --> H["Transformer Block 12"]
    H --> I["Layer Norm"]
    I --> J["Linear Head\n(768 -> 50257)"]
    J --> K["Softmax\nNext-token probabilities"]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#0f3460,color:#fff
    style C fill:#1a1a2e,stroke:#0f3460,color:#fff
    style D fill:#1a1a2e,stroke:#16213e,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style H fill:#1a1a2e,stroke:#e94560,color:#fff
    style I fill:#1a1a2e,stroke:#16213e,color:#fff
    style J fill:#1a1a2e,stroke:#0f3460,color:#fff
    style K fill:#1a1a2e,stroke:#51cf66,color:#fff
```

### Transformer Blok

12 bloktan her biri aynı kalıpta. Pre-norm mimarisi (GPT-2 orijinal transformatör gibi pre-norm, post-norm kullanır):

1. LayerNorm
2. Çok Başlı Kendine Dikkat
3. Geri kalan bağlantı (gönüllü giriş geri ekle)
4. LayerNorm
5. İlaçlı İlaçlı Ağ (MLP)
6. Geri kalan bağlantı (gönüllü giriş geri ekle)

Geriye doğru yayılma sırasında 1 blok'a ulaştıkları zaman gradientler kaybolur. Onlarla gradientler kayıptan herhangi bir katman boyunca "çıkış" yoluyla doğrudan akışabilir. Bu nedenle 12, 32, hatta 96 blok yığılabilir (GPT-4'in 120 kullanması söyleniyor).

### Dikkat: Temel Mekanizma

Kendine dikkat etmek, her simgeyi önceki simgeye bakıp her birine ne kadar katılacağına karar verir.

Her bir simge pozisyonu için girişten üç vektörü hesaplayın:
- **Query (Q)**"Ne arıyorum?"
- **Key (K)**"Ne içermem?"
- **Value (V)**"Ne tür bilgiler taşıyorum?"

```
Q = input @ W_q    (768 -> 768)
K = input @ W_k    (768 -> 768)
V = input @ W_v    (768 -> 768)

attention_scores = Q @ K^T / sqrt(d_k)
attention_scores = mask(attention_scores)   # causal mask: -inf for future positions
attention_weights = softmax(attention_scores)
output = attention_weights @ V
```

GPT'yi otomatik olarak geriye dönük yapan sebep maskıdır. 5 pozisyon 0-5 pozisyonlarına bakabilir, ancak 6, 7, 8 ve benzeri pozisyonlara bakamaz. Bu, modelin eğitim sırasında gelecekteki jetonlara bakarak " aldatmasını" önler.

**Multi-head attention**768 boyutlu alanı 64 boyutlu 12 başlara ayırır. Her baş farklı bir dikkat tarzı öğrenir. Bir baş sintaksis ilişkileri (subject-verb anlaşması) izleyebilir. Bir başkası semantik benzerliği (sinoonimleri) izleyebilir. Bir başkası konum yakınlığı (yakın kelime) izleyebilir.

```mermaid
graph LR
    subgraph MultiHead["Multi-Head Attention (12 heads)"]
        direction TB
        I["Input (768)"] --> S1["Split into 12 heads"]
        S1 --> H1["Head 1\n(64 dims)"]
        S1 --> H2["Head 2\n(64 dims)"]
        S1 --> H3["..."]
        S1 --> H12["Head 12\n(64 dims)"]
        H1 --> C["Concat (768)"]
        H2 --> C
        H3 --> C
        H12 --> C
        C --> O["Output Projection\n(768 -> 768)"]
    end

    subgraph SingleHead["Each Head Computes"]
        direction TB
        Q["Q = X @ W_q"] --> A["scores = Q @ K^T / 8"]
        K["K = X @ W_k"] --> A
        A --> M["Apply causal mask"]
        M --> SM["Softmax"]
        SM --> MUL["weights @ V"]
        V["V = X @ W_v"] --> MUL
    end

    style I fill:#1a1a2e,stroke:#e94560,color:#fff
    style O fill:#1a1a2e,stroke:#e94560,color:#fff
    style Q fill:#1a1a2e,stroke:#0f3460,color:#fff
    style K fill:#1a1a2e,stroke:#0f3460,color:#fff
    style V fill:#1a1a2e,stroke:#0f3460,color:#fff
```

Sqrt(d_k) - sqrt(64) = 8 - ile bölünme ölçeklendiriliyor. Bu olmadan nokta ürünleri yüksek boyutlu vektörler için büyür ve softmax'ı gradientlerin neredeyse sıfır olduğu bölgelere doğru itirir. Bu orijinal "Dikkat Tek İhtiyacınız Var" makalesindeki ana bilgilerden biriydi.

### KV Cache: Neden Tahmin Hızlı

Eğitim sırasında tüm dizini bir anda işletiyorsunuz. Tahmin sırasında, bir seferde bir token oluşturursunuz. Optimize edilmeden, N tokenini oluşturmak, tüm N-1 önceki tokenleri için yeniden hesaplama dikkatini gerektirir. Bu, üretilen token başına O(N^2) veya uzunluğu N dizisi için toplam O(N^3)

KV Cache bunu çözer. Her bir token için K ve V hesapladıktan sonra, onları saklayın. N+1 simgesini oluştururken, yeni simgesini Q'yu hesaplamanız ve önceki tüm simgelerden önbelleğe alınan K ve V'yi aramanız gerekir. Bu, K ve V hesaplamaları için token başına maliyetin O(N) 'den O(1) 'ye düşmesini sağlar. Dikkat puanı hesaplaması hala O(N) çünkü önceki tüm pozisyonlara dikkat edersiniz, ancak giriş üzerinde redundant matris çarpımlarından kaçınırsınız.

12 katman ve 12 başlı GPT-2 için, KV önbelleği 2 (K + V) x 12 katman x 12 baş x 64 dims = 18.432 değerleri bir token için saklar. 1024-token dizisi için, bu FP32'de yaklaşık 75MB. 128 katmanlı Llama 3 405B için, tek bir dizide KV önbelleği 10GB'yi aştırabilir. Bu nedenle uzun bağlamlı sonuç belleğe bağlıdır.

### Ön doldurma vs. Dekodlama: İki aşama

Bir LLM'ye bir istek gönderdiğinde, sonuç iki farklı aşamada gerçekleşir.

**Prefill**Tüm işaretler bilinir, böylece model tüm pozisyonlar için dikkatini aynı anda hesaplayabilir. Bu aşama hesaplama bağlıdır - GPU tam throughput'ta matris çarpmalarını yapıyor. A100'de 1000 işaretli bir işaret için, ön doldurma yaklaşık 20-50 ms sürer.

**Decode**Tokenleri birer birer üretir. Her yeni token önceki tüm tokenlere bağlıdır. Bu aşama hafıza bağlanır -- şişe boynuzunda, GPU hafızasından model ağırlıklarını ve KV önbelleğini okuyoruz, matris matematikinin kendisi değil. GPU'nun hesap çekirdekleri çoğunlukla hafıza okumalarını beklerken hareketsiz kalır. GPT-2 için, her dekodlama adımı, matmuls'in kaç FLOP'ye ihtiyaç duyduğundan bağımsız olarak yaklaşık aynı zaman alır, çünkü hafıza bant genişliği kısıtlama.

Bu fark üretim sistemleri için önemlidir. GPU hesaplama ile geçiş ölçeklerini önceden doldur (daha fazla FLOPS = daha hızlı önceden doldur). Anıt bant genişliği ile geçiş ölçeklerini çöz (hızlıca bellek = daha hızlı dekode). Bu nedenle NVIDIA'nın H100, A100'ye kıyasla bellek bant genişliği geliştirmelerine odaklandı - doğrudan jeton üretimini hızlandırıyor.

```mermaid
graph LR
    subgraph Prefill["Phase 1: Prefill"]
        direction TB
        P1["Full prompt\n(all tokens known)"]
        P2["Parallel computation\n(compute-bound)"]
        P3["Builds KV Cache"]
        P1 --> P2 --> P3
    end

    subgraph Decode["Phase 2: Decode"]
        direction TB
        D1["Generate token N"]
        D2["Read KV Cache\n(memory-bound)"]
        D3["Append to KV Cache"]
        D4["Generate token N+1"]
        D1 --> D2 --> D3 --> D4
        D4 -.->|repeat| D1
    end

    Prefill --> Decode

    style P1 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style P2 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style P3 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style D4 fill:#1a1a2e,stroke:#e94560,color:#fff
```

### Eğitim Çelişkisi

Bir LLM eğitimi bir sonraki belirti tahminidir. Tokenler [0, 1, 2, ..., N-1] verildiğinde, belirti belirtilerini [1, 2, 3, ..., N] tahmin edin. Kayıp işlevi modelin öngörülen olasılık dağılımıyla gerçek bir sonraki belirti arasındaki çapraz entropi.

Bir eğitim adım:

1. **Forward pass**: Parçayı tüm 12 bloktan geçirin. Her pozisyon için logitler alın.
2. **Compute loss**: Logit ve hedef tokenler arasındaki çapraz entropi (gelenek bir pozisyonla değiştirilmiştir).
3. **Backward pass**: Geri yayılma kullanarak tüm 124M parametreleri için gradient hesaplayın.
4. **Optimizer step**GPT-2'de Adam'ın öğrenme hızının yükselmesi ve kosinus bozulması için kullanılıyor.

Öğrenme oranı programı beklediğinizden daha önemlidir. GPT-2 ilk 2.000 adım boyunca 0'dan en yüksek öğrenme oranına kadar ısınır, sonra bir kozin eğri ardından bozulur. Yüksek öğrenme oranıyla başlayan model farklılaşır. Sürekli yüksek oranı korumak daha sonraki eğitimde kayıplara neden olur. ısınma-sonra bozulma modeli her büyük LLM tarafından kullanılır.

### GPT-2 Küçük: Sayılar

| Component | Shape | Parameters |
|-----------|-------|------------|
| Token embeddings | (50257, 768) | 38,597,376 |
| Position embeddings | (1024, 768) | 786,432 |
| Per-block attention (W_q, W_k, W_v, W_out) | 4 x (768, 768) | 2,359,296 |
| Per-block FFN (up + down) | (768, 3072) + (3072, 768) | 4,718,592 |
| Per-block LayerNorms (2x) | 2 x 768 x 2 | 3,072 |
| Final LayerNorm | 768 x 2 | 1,536 |
| **Total per block** | | **7,080,960** |
| **Total (12 blocks)** | | **85,054,464 + 39,383,808 = 124,438,272** |

Çıktı projeksiyon (logits başı) simge gömülme matrisi ile ağırlıkları paylaşır. Buna ağırlık bağlaması denir. Parametre sayısını 38M'ye düşürür ve performansını artırır, çünkü modelin giriş ve çıkış için aynı temsil alanını kullanmasını zorlar.

## Yapın

### Adım 1: Katman Ekle

Token yerleşimleri, 50.257 olası tokenin her birini 768 boyutlu bir vektöre haritasıyor. Konum yerleşimleri, her tokenin sırada nerede yer aldığı hakkında bilgi ekliyor.

```python
import numpy as np

class Embedding:
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        self.token_embed = np.random.randn(vocab_size, embed_dim) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, embed_dim) * 0.02

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        tok_emb = self.token_embed[token_ids]
        pos_emb = self.pos_embed[:seq_len]
        return tok_emb + pos_emb
```

Başlangıç için 0.02 standart sapma GPT-2 kağıdından gelir. Çok büyük ve başlangıç ileri geçişleri eğitimyi istikrarsızlaştıran aşırı değerler üretir. Çok küçük ve başlangıç çıkışları tüm girişler için neredeyse aynıdır, bu da erken gradient sinyallerini işe yaramaz hale getirir.

### Adım 2: Sebep Maskası ile Kendine Dikkat

İlk önce tek başlı dikkat. Sebep maskası, her pozisyonun sadece kendisini ve önceki pozisyonları takip edebilmesini sağlayan, yumuşak maksimumtan önce gelecek pozisyonları negatif sonsuzluğa ayarlar.

```python
def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, -1, -2 if Q.ndim == 4 else 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ V
```

Softmax uygulaması, eksponenciye edilmeden önce maksimumı çıkarır. Bu olmadan, exp(large_number) sonsuzluğa akıyor. Bu, herhangi bir sabit c için softmax(x - c) = softmax(x) nedeniyle çıkışını değiştirmeyen bir sayısal istikrar hilesi.

### Üçüncü Adım: Çok Başlı Dikkat

768 boyutlu girişleri 64 boyutlu 12 başlara ayırın. Her baş dikkatini bağımsız olarak hesaplar. Sonuçları birleştirin ve 768 boyutlara geri gönderin.

```python
class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_out = np.random.randn(embed_dim, embed_dim) * 0.02

    def forward(self, x, mask=None):
        batch, seq_len, d = x.shape
        Q = (x @ self.W_q).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = (x @ self.W_k).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = (x @ self.W_v).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)
        attn_out = weights @ V

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d)
        return attn_out @ self.W_out
```

Yeniden şekillendirmek-transpose-yeniden şekillendirmek dansı, çok başlı ilgi için en kafa karıştırıcı bir bölümdür. İşte ne oluyor: (batch, seq_len, 768) tensörü (batch, seq_len, 12, 64), sonra (batch, 12, seq_len, 64) olur. Şimdi 12 başın her birinin dikkatini çekecek kendi (seq_len, 64) matrisi var. Dikkat ettikten sonra, süreci tersine çeviririz: (batch, 12, seq_len, 64) becomes (batch, seq_len, 12, 64) becomes (batch, seq_len, 768).

### Dördüncü Adım: Transformer Blok

Tek tam transformatör bloğu: LayerNorm, kalıntılı bir çok başlı dikkat, LayerNorm, kalıntılı bir feedforward.

```python
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class FeedForward:
    def __init__(self, embed_dim, ff_dim):
        self.W1 = np.random.randn(embed_dim, ff_dim) * 0.02
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, embed_dim) * 0.02
        self.b2 = np.zeros(embed_dim)

    def forward(self, x):
        h = x @ self.W1 + self.b1
        h = np.maximum(0, h)  # GELU approximation: ReLU for simplicity
        return h @ self.W2 + self.b2


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_dim):
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim)

    def forward(self, x, mask=None):
        x = x + self.attn.forward(self.ln1.forward(x), mask)
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x
```

Feedforward ağı 768 boyutlu girişini 3,072 boyutlara (4x) genişletir, bir çizgizliği uyguluyor, sonra 768'e geri proje eder. Bu genişleme-sıkıştırma örneği, modelin her pozisyonda çalışmak için "geniş" bir iç temsil sağlar. GPT-2 GELU etkinleştirmesini kullanır, ancak burada basitlik için ReLU kullanıyoruz - fark mimarisi anlamak için küçüktür.

### Adım 5: Tam GPT modeli

12 transformatör blokunu yığ. Ön tarafta yerleştirme katmanı ve arka tarafta çıkış projeksiyonu ekle.

```python
class MiniGPT:
    def __init__(self, vocab_size=50257, embed_dim=768, num_heads=12,
                 num_layers=12, max_seq_len=1024, ff_dim=3072):
        self.embedding = Embedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(embed_dim)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)

        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x, mask)
        x = self.ln_f.forward(x)

        logits = x @ self.embedding.token_embed.T
        return logits

    def count_parameters(self):
        total = 0
        total += self.embedding.token_embed.size
        total += self.embedding.pos_embed.size
        for block in self.blocks:
            total += block.attn.W_q.size + block.attn.W_k.size
            total += block.attn.W_v.size + block.attn.W_out.size
            total += block.ffn.W1.size + block.ffn.b1.size
            total += block.ffn.W2.size + block.ffn.b2.size
            total += block.ln1.gamma.size + block.ln1.beta.size
            total += block.ln2.gamma.size + block.ln2.beta.size
        total += self.ln_f.gamma.size + self.ln_f.beta.size
        return total
```

Ağırlık bağlamasına dikkat edin: `logits = x @ self.embedding.token_embed.T`. Çıkış projeksiyonu, simge gömülme matrisini yeniden kullanır (transposed). Bu sadece parametre tasarruf hilesi değildir. Bu, modelin simgeyi (sürükleme) anlamak ve tahmin etmek için aynı vektör alanı kullanması anlamına gelir (sürükleme).

### Adım 6: Eğitim Çubuğu

124M parametrelerinde gerçek bir eğitim çalışması için bir GPU ve PyTorch gerekir. Bu eğitim döngüsü, saf bir numpy'de çalışan küçük bir modelin mekaniğini gösterir.

```python
def cross_entropy_loss(logits, targets):
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    max_logits = logits_flat.max(axis=-1, keepdims=True)
    log_softmax = logits_flat - max_logits - np.log(
        np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
    )

    loss = -log_softmax[np.arange(len(targets_flat)), targets_flat].mean()
    return loss


def train_mini_gpt(text, vocab_size=256, embed_dim=128, num_heads=4,
                   num_layers=4, seq_len=64, num_steps=200, lr=3e-4):
    tokens = np.array(list(text.encode("utf-8")[:2048]))
    model = MiniGPT(
        vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_layers, max_seq_len=seq_len, ff_dim=embed_dim * 4
    )

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Training tokens: {len(tokens):,}")
    print(f"Config: {num_layers} layers, {num_heads} heads, {embed_dim} dims")
    print()

    for step in range(num_steps):
        start_idx = np.random.randint(0, max(1, len(tokens) - seq_len - 1))
        batch_tokens = tokens[start_idx:start_idx + seq_len + 1]

        input_ids = batch_tokens[:-1].reshape(1, -1)
        target_ids = batch_tokens[1:].reshape(1, -1)

        logits = model.forward(input_ids)
        loss = cross_entropy_loss(logits, target_ids)

        if step % 20 == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f}")

    return model
```

Kayıp, ln(vocab_size) yakınında başlar. 256 token bayt seviyesindeki sözlük için ln(256) = 5.55. Bir rastgele model her token'e eşit olasılık belirler. Eğitim ilerledikçe kayıp düşer çünkü model ortak desenleri tahmin etmeyi öğrenir: "th" "t" sonrası, bir dönem sonrası alan ve benzeri şeyler.

Üretim sırasında, Adam optimizer'i gradient birikimi, öğrenme hızı ısınması ve gradient kesimi ile kullanırsınız. Önümüze geçme-kayıp-geriye geri güncelleme döngüsü aynıdır. Optimizer daha gelişmiştir.

### 7 . Adım: Metin Oluşturma

Generasyon, eğitilmiş modelle bir seferde bir token öngörüyor. Her tahmin çıkış dağılımından örnek alınır (veya argmax olarak açgözlülükle alınır).

```python
def generate(model, prompt_tokens, max_new_tokens=100, temperature=0.8):
    tokens = list(prompt_tokens)
    seq_len = model.embedding.pos_embed.shape[0]

    for _ in range(max_new_tokens):
        context = np.array(tokens[-seq_len:]).reshape(1, -1)
        logits = model.forward(context)
        next_logits = logits[0, -1, :]

        next_logits = next_logits / temperature
        probs = np.exp(next_logits - next_logits.max())
        probs = probs / probs.sum()

        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(next_token)

    return tokens
```

Sıcaklık rastlantıyı kontrol eder. Sıcaklık 1.0 ham dağılım kullanır. Sıcaklık 0.5 onu keskinleştirir (öntemsel - model en iyi seçeneklerini daha sık seçer). Sıcaklık 1.5 onu düzeltir (sıcaklık daha fazla - düşük olasılık belirtiler daha büyük bir şans elde eder). Sıcaklık 0.0 açgözlü bir çözme (her zaman en yüksek olasılık belirtilerini seçin).

- Evet .`tokens[-seq_len:]`GPT-2 için maksimum bağlam uzunluğu (1024) olduğu için pencerenin olması gereklidir.

```figure
sampling-decoder
```

## Kullan

### Tam Eğitim ve Nesil Demo

```python
corpus = """The transformer architecture has revolutionized natural language processing.
Attention mechanisms allow the model to focus on relevant parts of the input.
Self-attention computes relationships between all pairs of positions in a sequence.
Multi-head attention splits the representation into multiple subspaces.
Each attention head can learn different types of relationships.
The feedforward network provides nonlinear transformations at each position.
Residual connections enable gradient flow through deep networks.
Layer normalization stabilizes training by normalizing activations.
Position embeddings give the model information about token ordering.
The causal mask ensures autoregressive generation during training.
Pre-training on large text corpora teaches the model general language understanding.
Fine-tuning adapts the pre-trained model to specific downstream tasks."""

model = train_mini_gpt(corpus, num_steps=200)

prompt = list("The transformer".encode("utf-8"))
output_tokens = generate(model, prompt, max_new_tokens=100, temperature=0.8)
generated_text = bytes(output_tokens).decode("utf-8", errors="replace")
print(f"\nGenerated: {generated_text}")
```

Küçük bir modelle küçük bir korpusta, oluşturulan metin en iyi durumda yarı tutarlı olacaktır. Eğitim metniyle bazı bayt seviyesindeki desenleri öğrenecek, ancak GPT-2'nin 40GB eğitim verisi ve tam 124M parametresi mimarisi ile yaptığı gibi genelleştiremez. Konu çıkış kalitesi değil. Önemli olan her adımı takip edebilmeniz: yerleştirme arama, dikkat hesaplama, geri dönüşüm, logit projeksiyonu, softmax ve örnekleme. Her operasyon görünür.

## Gönder

Bu ders bize çok yararlı .`outputs/prompt-gpt-architecture-analyzer.md`-- herhangi bir GPT tarzı modelindeki mimari seçeneklerini analiz eden bir istek. Ona bir model kartı veya teknik rapor vererek parametrelerin tahsisini, dikkat tasarımını ve ölçekleme kararlarını parçaladı.

## Egzersizler

1. Modelle 12/12 yerine 24 kat ve 16 baş kullanmak için değişiklik yapın. Parametreyi sayın.

2. GELU etkinleştirme fonksiyonunu uygulayın (GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))) ve ReLU'yu feedforward ağındaki değiştirin. Her etkinleştirme ile 500 adım boyunca eğitim uygulayın ve son kaybı karşılaştırın.

3. Generasyon fonksiyonuna bir KV önbelleği ekleyin. İlk ileri geçişten sonra her katman için K ve V tensörlerini saklayın ve sonraki jetonlar için tekrar kullanın. Hızlandırmayı ölçün: 200 jetonları önbelleği ile ve olmadan oluşturun ve duvar saatini karşılaştırın.

4. Üst-k örneklemeyi uygulayın (sadece en yüksek olasılıklı k simgeleri dikkate alın) ve üst-p örneklemeyi (nukleüs örnekleme: top-p=0.95 ile sıcaklık 0,8'de çıkış kalitesini karşılaştırın.

5. Bir eğitim kaybı eğri planı oluşturun. Modelini 1000 adım ve plan kaybı vs. adım için eğit. Üç aşamayı tanımlayın: hızlı başlangıç düşüşü (orta bayt öğrenme), daha yavaş orta aşama (biçim biçimleri öğrenme) ve plato (küçük korpus üzerinde üstü).

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Autoregressive | "It generates one word at a time" | Each output token is conditioned on all previous tokens -- the model predicts P(token_n \| token_0, ..., token_{n-1}) |
| Causal mask | "It can't see the future" | An upper-triangular matrix of -infinity values that prevents attention to future positions during training |
| Multi-head attention | "Multiple attention patterns" | Splitting Q, K, V into parallel heads (e.g., 12 heads of 64 dims each for GPT-2) so each head can learn different relationship types |
| KV Cache | "Caching for speed" | Storing computed Key and Value tensors from previous tokens to avoid redundant computation during autoregressive generation |
| Prefill | "Processing the prompt" | The first inference phase where all prompt tokens are processed in parallel -- compute-bound on GPU FLOPS |
| Decode | "Generating tokens" | The second inference phase where tokens are generated one at a time -- memory-bound on GPU bandwidth |
| Weight tying | "Sharing embeddings" | Using the same matrix for input token embeddings and the output projection head -- saves 38M params in GPT-2 |
| Residual connection | "Skip connection" | Adding the input directly to the output of a sublayer (x + sublayer(x)) -- enables gradient flow in deep networks |
| Layer normalization | "Normalizing activations" | Normalizing across the feature dimension to mean 0 and variance 1, with learnable scale and bias parameters |
| Cross-entropy loss | "How wrong the predictions are" | -log(probability assigned to the correct next token), averaged over all positions -- the standard LLM training objective |

## Daha Fazla Okumak

- [Radford et al., 2019 -- "Language Models are Unsupervised Multitask Learners" (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)-- 124M ile 1.5B parametreleri ailesini tanıtan GPT-2 kağıdı
- [Vaswani et al., 2017 -- "Attention Is All You Need"](https://arxiv.org/abs/1706.03762)-- orijinal transformatör kağıdı, ölçülü nokta ürün dikkat ve çok başlı dikkat ile
- [Llama 3 Technical Report](https://arxiv.org/abs/2407.21783)-- Meta 16K GPU ile GPT mimarisini 405B parametrelerine nasıl ölçeklendirdi
- [Pope et al., 2022 -- "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102)-- prefill vs decode ve KV cache analizini resmileştiren kağıt
