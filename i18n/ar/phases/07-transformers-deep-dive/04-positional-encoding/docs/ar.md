# التشفير الموضعي  الحاجز الشقري، ROPE، ALiBi

> الاهتمام هو محورية-غير متغير. "الجدال جلست على المفتاح" و "الجدال على القطال على المفتاح" تنتج نفس الخروج دون إشارة الموقف. ثلاثة خوارزميات تحللها  كل مع رهان مختلف على ما يعني "الموقف".

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 minutes

## المشكلة

الاهتمام المتوسط من النقطة هو أعمى من النظام.`softmax(Q K^T / √d) V`يتم حسابها من تشابهات زوجية.`X`لا شيء داخل الاهتمام يهتم بالموقع

هذا ليس خطأ في نموذج كيس الكلمات. بالنسبة للغة، الرمز، الصوت، الفيديو  أي شيء حيث النظام يحمل المعنى  هو قاتل.

الحل هو حقن الموقف في التوابل بطريقة ما ثلاث حقول من الإجابات:

1. **Absolute sinusoidal**(Vaswani 2017) إضافة `sin/cos`بسيط، خالي من التعلم، يستغرق بشكل ضعيف ما وراء الطول المدرب.
2. **RoPE — Rotary Position Embeddings**(Su 2021). قم بتدوير متجهات Q و K بزاوية متناسبة بالموقع. يقوم بتشفير *موقف النسبي* مباشرة في منتج النقاط. هيمننت في عام 2026.
3. **ALiBi — Attention with Linear Biases**(طباعة 2022). تخطي التوابل بالكامل؛ أضف عقوبة خطية لكل رأس إلى درجات الاهتمام بناءً على المسافة. استنتاج الطول ممتاز.

اعتبارا من عام 2026 ، تستخدم كل نموذج مفتوح حدودي بشكل أساسي RoPE: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. عدد قليل من نماذج السياق الطويل تستخدم ALiBi أو إصداراتها الحديثة. التخدير الحقيقى المطلق تاريخي.

## المفهوم

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### الحجرة الحجرية المطلقة

الحساب المسبق لمصفوفة ثابتة `PE`من الشكل`(max_len, d_model)`:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

إذن`X' = X + PE[:N]`كل بعد هو سينوسيد في تردد مختلف. يتعلم النموذج قراءة الموقف من نمط المرحلة. يفشل خارج`max_len`: لا شيء أخبر النموذج بما يحدث في الموقف 2048 عندما رأى فقط المواقع 02047.

### (ROPE)

قم بتدوير متجهات Q و K (ليس منشغلات) لزوج من الأبعاد `(2i, 2i+1)`:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

تطبيق نفس التناوب على المفاتيح مع الموقف `pos_k`. منتج النقطة`q'_m · k'_n`يصبح وظيفة `(m - n)`وحدها، هذا هو:**the attention score depends only on the relative distance**على الرغم من أن الدوران كان مُفتوحاً من المواقع المطلقة

تمديد (ROPE): `base`يمكن أن يتم قياسها (NTK-وعي، YaRN، LongRoPE) لاستقطابها إلى سياقات أطول دون إعادة التدريب. تمتد Llama 3 من 8K إلى 128K السياق بهذه الطريقة.

### (أليبي)

تفوت خدعة التثبيت، تحيز الاهتمام يسجل مباشرة:

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

أين`m_h`هو منحدر محدد للرأس (مثل `1 / 2^(8·h/H)`تُعزز الرموز القريبة؛ وتُعاقب الرموز البعيدة. لا تكلفة وقت التدريب. يظهر الورقة أن الاستخراج الطويل يتجاوز السينوسيدال ويطابق روبي على طوله المُدرب الأصلي.

### ماذا تختار في عام 2026

| Variant | Extrapolation | Training cost | Used by |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | poor | free | original transformer, early BERT |
| Learned absolute | none | tiny | GPT-2, GPT-3 |
| RoPE | good with scaling | free | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | excellent | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | excellent | free | BLOOM, MPT, Baichuan |

فازت (روبي) لأنها تثير الانتباه دون تغيير الهندسة المعمارية وتشفير الموقف النسبي`base`يقدم المعلم المضاد زر نظيف للتحسينات الدقيقة في السياق الطويل.

```figure
rope-explorer
```

## بناءها

### الخطوة الأولى: تشفير السينوسويدي

انظر`code/main.py`حساب أربعة خطوط:

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

أضف هذا إلى المصفوفة قبل طبقة الاهتمام الأولى.

### الخطوة الثانية: تطبيق RoPE على Q و K

تعمل RoPE في مكان Q و K. لكل زوج من المضغوطات:

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

الحاسم: تطبيق نفس الوظيفة على Q في الموقف `m`و (ك) في الموقف`n`إن منتجهم يلتقط`cos((m-n)·θ_i)`العامل على كل زوج من أزواج الإحداثيات

### الخطوة الثالثة: منحدرات و تحيزات ALiBi

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

إضافة`bias[h]`إلى`(seq_len, seq_len)`مديرية درجة الاهتمام من الرأس`h`ثم "سوف ماكس"

### الخطوة الرابعة: التحقق من خصائص المسافة النسبية للروبي

اختر متجهين عشوائيين`a, b`. تدور من خلال`(pos_a, pos_b)`- ثم من قبل`(pos_a + k, pos_b + k)`. يجب أن تتطابق كلا المنتجات النقطة داخل خطأ نقطة عائمة. هذه الخصية هي نقطة كاملة من RoPE  انها غير متغيرة للتعويض المطلق، فقط الفجوة النسبية مهمة.

## استخدمها

"بيتورش 2.5+" تسافر على مصادر "روبي" في "`torch.nn.functional`معظم أسلوب استخدامات رمز الإنتاج`flash_attn`أو`xformers`حيث يتم تطبيق RoPE داخل جوهر الاهتمام.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Long-context tricks in 2026:**

- **NTK-aware interpolation.**إعادة التوسع`base`إلى`base * (scale_factor)^(d/(d-2))`عندما تمتد من 4K إلى 16K +.
- **YaRN.**إنتربولات ذكية تحافظ على إنتروبيّة الاهتمام في سياقات طويلة.
- **LongRoPE.**طريقة مايكروسوفت لعام 2024 التي تستخدم البحث التطوري لتحديد عوامل على نطاق الأبعاد
- **Position interpolation + fine-tuning.**فقط تقلص المراكز بمعدل التوسع وتحسن معدلها على الرموز 15B

## أرسله

انظر`outputs/skill-positional-encoding-picker.md`. تتخذ المهارة استراتيجية تشفير لنموذج جديد نظراً لمدة السياق المستهدفة واحتياجات الاستخراج وميزانية التدريب.

## التمارين

1. **Easy.**رسم الحاجز الصناعي`PE`المصفوفة كخريطة حرارة ل`max_len=512, d=128`تأكيد نمط "الشرائط تصبح أوسع مع نمو مؤشر الأبعاد"
2. **Medium.**قم بتنفيذ مقياسات ROPE التي تدرك NTK. قم بتدريب LM الصغير على تسلسلات بطول 256 ، ثم اختبر على طول 1024 مع ودون مقياس. قم بتقييم الارتباك.
3. **Hard.**تنفيذ ALiBi و RoPE في نفس وحدات الاهتمام. تدريب محول 4 طبقات على مهمة نسخ مع تسلسلات طول 512. استنتاج إلى 2048 في وقت الاختبار. مقارنة التدهور.

## الشروط الرئيسية

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

## المزيد من القراءة

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762)-المنحدر الأصلي
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)ورق روبي
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)-أليبي
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) حالة من الفن RoPE مقياس.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) ورقة "إلاما 2" طويلة السياق
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) طريقة مايكروسوفت المستخدمة من قبل Phi-3-Long و التي نقلت عنها في قسم استخدامها.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) تنفيذات درجة الإنتاج لكل نظام قياس روبي (الرائعي، الخطي، الديناميكي، YaRN، LongRoPE، Llama-3).
