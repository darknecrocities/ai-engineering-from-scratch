# المحول الكامل  مرموز + مرموز

> الانتباه هو النجم. كل شيء آخر  بقايا، التطبيع، التغذية، الانتباه المتقاطع  هو الرفف الذي يسمح لك بتجميعها بعمق.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## المشكلة

طبقة انتباه واحدة هي مستخرج ميزة ، وليس نموذج. واحد من المتمولات في كل طبقة ليس قدرة كافية للغة. تحتاج إلى عمق  وقطع عمق دون الأنابيب الصحيحة.

أُحتوي ورقة فاسواني 2017 على ست قرارات تصميم حولت طبقة من الاهتمام إلى كتلة قابلة للتجميع. يتلقى كل محول منذ  إكودر فقط (BERT) ، و decoder فقط (GPT) ، و encoder-decoder (T5)  نفس العظم. في عام 2026 تمت إصلاح الكتل (RMSNorm ، SwiGLU ، pre-norm ، RoPE) ولكن العظم هو نفسه.

هذه الدروس هي العظم. الدروس التالية تخصيصه  06 للكودرات، 07 للكودرات، 08 للكودرات-الديكودرات.

## المفهوم

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### القطع الستة

1. **Embedding + positional signal.**الوهم → المتجهات. وضعية حقن عن طريق RoPE (حديث) أو السينوسيدل (كلاسيكي).
2. **Self-attention.**كل موقف يُعتني بالآخر مغرم في مُشفرات
3. **Feed-forward network (FFN).**الموضع من حيث الطبقة الثنائية من المعدات: `W_2 · activation(W_1 · x)`نسبة التوسع 4× حسب الاختيار
4. **Residual connection.** `x + sublayer(x)`بدون هذا، تتلاشى التدرج بعد حوالي 6 طبقات
5. **Layer normalization.** `LayerNorm`أو`RMSNorm`يُثبت التيار المتبقية
6. **Cross-attention (decoder only).**تظهر الأسئلة من المفكّر والفاتيح والقيم من خروج المفكّر.

شاهد تدفق المتجه عبر كتلة واحدة: الاهتمام يختلط عبر المواقع، والباقي يحمله إلى الأمام، FFN يغيره، والطبيعة تبقي التيار مستقرا.

```figure
transformer-block
```

### بلوك المشفير (المستخدم من قبل BERT، T5 encoder)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

إنّ المُشفّر مُتجهّيّ، لا تخفيض، كلّ المواقع تُرى كلّ المواقع.

### حلقة تشكيل (تستخدمها GPT، T5 decoder)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

يحتوي المفكّر على ثلاث طبقات فرعية لكل كتلة. الوسط  الاهتمام المتقاطع  هو المكان الوحيد الذي تتدفق فيه المعلومات من المفكّر إلى المفكّر. في بنية المفكّر فقط النقي (GPT) ، يتم حذف الاهتمام المتقاطع وتتمكن من تزويد الاهتمام الذاتي بمظهر + FFN.

### قبل القاعدة مقابل بعد القاعدة

الورق الأصلي:`x + sublayer(LN(x))`و`LN(x + sublayer(x))`. بعد الطبيعية فقدت النحو في عام 2019  من الصعب التدريب بعمق دون تدفئة دقيقة.`LN`* قبل * طبقة فرعية) هو 2026 الافتراضي: Llama، Qwen، GPT-3+، Mistral جميع استخدامها.

### الكتلة المتجددة لعام 2026

قام Vaswani 2017 بإرسال LayerNorm + ReLU. استبدلت كومات حديثة كلا. كيف تبدو كتلة الإنتاج في الواقع:

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices, total params match) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

يقلل RMSNorm من متوسط مركزية LayerNorm (حساب واحد أقل) ، والتي توفر الحساب وتستقر تجريبيا على الأقل.`Swish(W1 x) ⊙ W3 x`) يتجاوز بشكل متسق ReLU/GELU FFN بنحو 0.5 نقطة ppl في ورق Llama و PaLM و Qwen.

### عدد المعايير

لمكتبة واحدة مع`d_model = d`وتوسيع FFN `r`:

- (م.ه.إم):`4 · d²`(مقاطع Q، K، V، O)
- FFN (SwiGLU): `3 · d · (r · d)`- -`3rd²`
- المعايير: لا تُعتبر مهمّة

في`d = 4096, r = 2.6, layers = 32`(تقريبًا إلاما 3 8 ب) ، إجمالي: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`(بضافة التوابع والرأس)

## بناءها

### الخطوة الأولى: قطع البناء

باستخدام الصفحة الصغيرة`Matrix`الصف من الدروس 03 (منسخة إلى هذا الملف من أجل الاستقلال):

- `layer_norm(x, eps=1e-5)` انقطاع المتوسط، تقسيم ب std.
- `rms_norm(x, eps=1e-6)` تقسيم بواسطة RMS لا يوجد معدل منقطعة
- `gelu(x)`و`silu(x) * W3 x`(سويجلي)
- `ffn_swiglu(x, W1, W2, W3)`. . .
- `encoder_block(x, params)`و`decoder_block(x, enc_out, params)`. . .

انظر`code/main.py`للكابلات الكاملة

### الخطوة الثانية: سلك مُشفّر طبقة 2 و مُشفّر طبقة 2

قم بتجميعها، قم بإدخال خروج المُشفّر إلى كلّ إشعار مُشفّر، أضف إشارةً أخيرةً قبل إطلاق الخروج.

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

### الخطوة الثالثة: اجري إلى الأمام على مثال لعبة

إرسال مصدر 6 رموز و هدف 5 رموز عبر. التحقق من شكل الخروج هو `(5, vocab)`لا تدريب، هذا الدروس يتعلق بالهندسة المعمارية، وليس الخسارة.

### الخطوة الرابعة: التبادل في RMSNorm + SwiGLU

استبدل LayerNorm و ReLU-FFN بـ RMSNorm و SwiGLU. تأكد أن الأشكال لا تزال مطابقة. هذا هو تحديث 2026 مع استبدال وظيفة واحدة.

## استخدمها

تنفيذات مرجعية PyTorch/TF: `nn.TransformerEncoderLayer`،`nn.TransformerDecoderLayer`لكن معظم رمز الإنتاج 2026 يُسجل كتله الخاص لأن:

- الاهتمام الفلكي يُدعى داخل الاهتمام، وليس عبر `nn.MultiheadAttention`. . .
- المجلس الوطني للشؤون القانونية/المجلس الوطني للشؤون الوطنية (GQA/MLA) ليس في إشارة المجلس الوطني للشؤون الوطنية.
- (روبي) ، (رمس نورم) ، (سويغلو) ليست طرق (بيتورش) القابلة للتعيين.

HF `transformers`يحتوي على كتيب مرجعية نظيفة يجب عليك قراءتها: `modeling_llama.py`هو الكانونيكي 2026 مجرد الكتلة المفكّرة. انها حوالي 500 سطر و يستحق المشي من خلال مرة واحدة.

**Encoder vs decoder vs encoder-decoder — when to pick:**

| Need | Pick | Example |
|------|------|---------|
| Classification, embeddings, QA over text | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

لغة المفكّر فقط فازت لأنها تتحكم في النظافة الأكثر نظافة وتتعامل مع فهم وتوليد الكلمات. لا يزال المفكّر المفكّر أفضل عندما يكون لدى المدخل هوية واضحة "سلسلة المصدر" (ترجمة، التعرف على الكلام، مهام مهيكلة).

## أرسله

انظر`outputs/skill-transformer-block-reviewer.md`. تقوم المهارة بمراجعة تنفيذ كتلة محول جديدة ضد الاختلافات الافتراضية لعام 2026 وتعريض الأجزاء المفقودة (معدل توسع قبل المعيار، RoPE، RMSNorm، GQA، FFN).

## التمارين

1. **Easy.**احتساب المعايير في كودر_بلوكك في `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التحقق من التد`sum(p.numel() for p in block.parameters())`. . .
2. **Medium.**التحول من بعد القاعدة إلى قبل القاعدة. قم بتشغيل كل منهما وقياس معايير التفعيل بعد 12 طبقة متداخلة على إدخال عشوائي. يجب أن تنفجر تنشيطات بعد القاعدة؛ يجب أن تبقى التفعيلات قبل القاعدة محدودة.
3. **Hard.**تنفيذ تشفير- تشفير 4 طبقات على مهمة نسخة لعبة (نسخ `x`التدريب 100 خطوة. تقرير الخسارة. تبادل في RMSNorm + SwiGLU + RoPE  هل تنخفض الخسارة؟

## الشروط الرئيسية

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

## المزيد من القراءة

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) تحديدات الكتل الأصلية
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)لماذا قبل الطبيعية يضرب بعد الطبيعية بشكل عميق
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) RMSNorm
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)ورقة SwiGLU
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) الكانونيكل 2026 كتل فقط
