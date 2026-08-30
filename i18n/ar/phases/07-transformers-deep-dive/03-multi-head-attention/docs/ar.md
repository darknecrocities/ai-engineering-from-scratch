# الاهتمام متعدد الرؤوس

> رأس واحد يدرك علاقة واحدة في كل مرة ثمانية رؤوس تتعلم ثمانية رؤوس مجانية خذ المزيد منها

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 minutes

## المشكلة

رأس الانتباه الذاتي واحد يحسب ماتريسكة الانتباه. هذه المصفوفة تسجل نوعًا واحدًا من العلاقات  عادةً ما تكون تلك التي تقلل من الخسارة على أي إشارة تدريبية. إذا كان بياناتك تتضمن اتفاقًا موضوعيًا ومفعولًا ومشاركة إشارة ، وتحديثًا طويلًا ، وتقطيعًا نحويًا متصلبًا معًا ، فإن رأس واحد يضعهم في توزيع واحد ذو غاية ضعيفة ويخسر نصف الإشارة.

التحدي من ورقة Vaswani 2017: تشغيل العديد من وظائف الاهتمام بالتوازي ، كل منها مع توقعات Q ، K ، V الخاصة به ، وتحديد الخروج. تعمل كل رأس في فرعية صغيرة من الأبعاد `d_model / n_heads`.الفقرات الكلية تبقى نفسها .تزداد الطاقة التعبيرية

الاهتمام متعدد الرؤوس هو الاختيار الافتراضي لكل محول في السفن 2026. الحجة الوحيدة حول * كم عدد الرؤوس * وما إذا كانت المفاتيح والقيم تشارك التنبؤات (التأمل المجموعي، التأمل المتعدد، التأمل المتعدد الرؤوس).

## المفهوم

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Split.**خذ`X`من الشكل`(N, d_model)`. مشروع إلى Q، K، V كل من الشكل `(N, d_model)`. إعادة التغيير إلى`(N, n_heads, d_head)`أين`d_head = d_model / n_heads`. نقل إلى`(n_heads, N, d_head)`. . .

**Attend in parallel.**أطلقوا على نطاق واسع النقطة المنتجة الاهتمام داخل كل رأس.`(N, d_head)`.ال رؤوس تعمل على مختلف الفضاء الفرعي من الإدراج ولا تتحدث أبدا خلال حساب الاهتمام نفسه.

**Concatenate and project.**رأس السحب يعود إلى`(N, d_model)`و تكرر بمصفوفة الخروج المتعلمة `W_o`من الشكل`(d_model, d_model)`. .`W_o`حيث تخلط الرؤوس

**Why it works.**يمكن لكل رأس التخصص دون المنافسة مع الآخرين لميزانية تمثيلية. تظهر دراسات الاستقصاء من 20192024 أدوار رأس متميزة: رؤوس الموقف، الرأس الذي يشارك في الرمز السابق، رؤوس النسخ، رؤوس الكيانات المسمى، رؤوس الإدراج (التي تتمثل في التعلم في السياق).

**The 2026 lineage of variations:**

| Variant | Q heads | K/V heads | Used by |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA هي الاختيار الافتراضي الحديث لأنه يقلل من ذاكرة KV-Cache بمعدل `N/G`بينما تحافظ على الجودة الكاملة تقريبا. MLA يذهب أبعد من ذلك عن طريق ضغط K / V في مساحة غامضة، ثم التنبؤ مرة أخرى في وقت الحساب  تكلف FLOPs، ويحفظ الكثير من الذاكرة.

```figure
multihead-split
```

## بناءها

### الخطوة الأولى: تقسيم الرؤوس من الاهتمام المتوحد الذي لدينا بالفعل

خذوا`SelfAttention`من الدروس 02 و لفها بـ زوج من المزقين`code/main.py`لتنفيذ ضئيل؛ المنطق هو:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

واحد إعادة تشكيل و واحد نقل لا حلقة هذا بالضبط ما يفعله PyTorch تحت`nn.MultiheadAttention`. . .

### الخطوة الثانية: تشغيل نقطة-المنتج الاهتمام لكل شخص

كل رأس يحصل على شريحة خاصة من Q، K، V. الانتباه يصبح ململ المكتسب:

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

على الأجهزة الحقيقية`Qh @ Kh.transpose(...)`هو واحد`bmm`. الجيبو يرى شكل واحد من الشكل`(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`إضافة رؤوس مجانية

### الخطوة 3: مجموعة-سؤال الاهتمام المتغير

فقط تغير الأساس والقيمة التنبؤات.`n_heads`المجموعات: K و V الحصول`n_kv_heads < n_heads`المجموعات وتكرر لتطابق:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

في الاستنتاج هذا يُخفي الذاكرة لأن فقط`n_kv_heads`النسخة الحية في الجهاز التخزيني، لا `n_heads`. Llama 3 70B يستخدم 64 رأس استفسار مع 8 رؤوس كيف  8× مخفف الاحتفاظ.

### الخطوة الرابعة: قم بتحقيق ما تعلمته كل رأس

إضغط على الجملة القصيرة مع 4 رؤوس. لكل رأس، طبع`(N, N)`سترى رؤوس مختلفة تختار بنية مختلفة حتى مع البداية عشوائية

## استخدمها

في PyTorch، النسخة ذات الخط الواحد:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA اعتبارا من PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**How many heads?**قواعد الإبهام من نماذج الإنتاج في عام 2026:

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head`تقريباً دائماً يصل إلى 64 أو 128. إنها وحدة كمية رأس واحد يمكنه "رؤية". انخفض إلى أقل من 32 و تبدأ الرؤوس في محاربة عامل التوسع`sqrt(d_head)`إضافة إلى إضافة إضافية إلى 256 و تفقد ميزة "الكثير من المتخصصين الصغار".

## أرسله

انظر`outputs/skill-mha-configurator.md`. توصي المهارة بعد الرأس، و عدد الرأس، واستراتيجية التنبؤ لمحول جديد مع إعطاء ميزانية المعلمات، وطول التسلسل، و هدف الانتشار.

## التمارين

1. **Easy.**خذ المكتب من`code/main.py`وتغيير`n_heads`من 1 إلى 16 مع `d_model=64`تدوين خسارة نموذج صغير من طبقة واحدة على مهمة نسخ اصطناعية هل تساعد رؤوس أكثر أو تساعد أو تؤذي؟
2. **Medium.**تنفيذ MQA (رأس KV واحد مشترك بين جميع رؤوس الاستفسار). قياس كمية تراجع عدد المعايير مقابل MHA الكامل. حساب كمية تقلص حجم KV-Cache عند الاستنتاج ل N = 2048.
3. **Hard.**تنفيذ نسخة صغيرة من الاهتمام المتخفي متعدد الرؤوس: ضغط K، V إلى رتبة`r`الاختفاء، تخزن الاختفاء في الاحتفاظ الكهربائي، وتفكيك في وقت الاهتمام.`r`هل تتجاوز ذاكرة الاحتفاظ بالخزنة أقل من 1/8 من MHA الكامل بينما تبقى الجودة ضمن 1 بت من التحقق من التحقق؟

## الشروط الرئيسية

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

## المزيد من القراءة

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) المواصفات الأصلية متعددة الرؤوس.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)ورقة المراقبة
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) كيفية تحويل MHA إلى GQA بعد التدريب.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) MLA ولماذا تفوق MHA / GQA على ذاكرة التخزين.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)نظرة ميكانيكية على ما تفعله الرؤوس في الواقع.
