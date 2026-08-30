# بناء محول من الصفر  الحجر القصري

> ثلاث عشرة درساً، نموذج واحد، لا اختصار

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 minutes

## المشكلة

لقد قرأت كل ورقة، لقد وضعت الانتباه، شقق متعددة الرؤوس، تشفيرات المواقع، حواجز تشفير و تشفير، فقدان BERT و GPT، MoE، KV cache. الآن اجعلهم يعملون معا على مهمة حقيقية.

الحجر الرئيسي: تدريب محول صغير فقط للكشف من نهاية إلى نهاية على مهام نمذجة لغة على مستوى الشخصيات. يقرأ شكسبير. يولد شكسبير جديد. هو صغير بما يكفي لتدريب على جهاز كمبيوتر محمول في أقل من 10 دقائق. هو صحيح بما فيه الكفاية أن تبادل مجموعة بيانات أكبر وتدريب أطول يحصل لك LM الحقيقي.

هذا هو "nanoGPT" من الدورة. انها ليست أصلية  Karpathy 2023 دراسة nanoGPT هو التنفيذ المرجعية كل طالب يكتب مرة واحدة على الأقل. نحن رفع الشكل وإعادة تشكيل حول ما قمنا بتغطيته.

## المفهوم

![Transformer-from-scratch block diagram](../assets/capstone.svg)

الهندسة المعمارية، تلاحظ:

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

### ما نُرسل

- `GPTConfig` مكان واحد لتكوين جميع المعايير المضادة.
- `MultiHeadAttention` سببية، دفعة، مع طريق اختياري في نمط فلاش (PyTorch's `scaled_dot_product_attention`)
- `SwiGLUFFN` FFN الحديث.
- `Block` الاهتمام المسبق للطبيعة، والغلفة المتبقية + FFN.
- `GPT` التوابل، الكتل المكتبة، رأس LM، توليد (().
- حلقة تدريب مع آدم و، كوسين LR، تراجع تراجع.
- إشارة على مستوى "كار" على نص "شكسبير".

### ما لا نُرسل

- يستخدم الممارسة المميزة المميزة (ROPE) في الدروس 04. هنا نستخدم التوابل المميزة المكتسبة من أجل البساطة.
- كش كيف خلال الجيل  كل خطوة جيل يعيد حساب الاهتمام على المسبق الكامل. أبطأ ولكن أبسط. التمارين تطلب منك إضافة كش كيف.
- الاهتمام الفلاش  PyTorch 2.0+ إرسالات تلقائية إذا كانت المدخلات تتطابق ؛ نستخدم `F.scaled_dot_product_attention`. . .
- الـ (مـو)  واحد FFN لكل كتلة. لقد رأيت الـ (مـو) في الدروس 11.

### المقاييس المستهدفة

على جهاز كمبيوتر محمول Mac M2، 4 طبقات، 4 رأس، d_model=128 GPT تدرب على 2000 خطوة على `tinyshakespeare.txt`:

- فقدان التدريب يتقارب من ~ 4.2 (الصدفة) إلى ~ 1.5 في حوالي 6 دقائق.
- النتائج المختارة تبدو شكسبير: الكلمات القديمة، وقطع الخطوط، الأسماء الخاصة مثل "روميو:" تظهر.
- فقدان الفائدة (حفاظ على 10% الأخير من النص) تتبع خسارة التدريب عن كثب، لا يوجد أي مبالغ في هذا الحجم / الميزانية.

```figure
n5-block-stack
```

## بناءها

هذه الدروس تستخدم PyTorch.`torch`(إنشاء المركز المركزي جيد) انظر`code/main.py`السيناريو يدير:

- التنزيل`tinyshakespeare.txt`إذا لم تظهر (أو قرأت نسخة محلية).
- إشارة البايت على مستوى البايت
- التقسيم بين القطار والقطار عند 90/10.
- حلقة تدريب مع bf16 التلقائي على الأجهزة المدعومة.
- تمت اختبار العينات بعد التدريب

### الخطوة الأولى: البيانات

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 حرفاً فريدة، مخزون لغات صغير، يتناسب بحجم الكلمات البيتين، لا توجد حروف إضافية، لا توكنيزر درامي.

### الخطوة الثانية: النموذج

انظر`code/main.py`. الكتل هي كتاب دراسي من الدروس 05  قبل القاعدة، RMSNorm، SwiGLU، MHA السببية.

### الخطوة الثالثة: حلقة التدريب

الحصول على مجموعة عشوائية من طول 256 نافذة رمزية للأمام، التحولات المتقاطعة واحد، الخلفية، خطوة آدم و.سجل، تكرر.

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

### الخطوة الرابعة: العينة

عند إعطاء طلب، مراراً وتكراراً، عينة من علامات "أعلى" ، وإضافة، واستمر. توقف بعد 500 رمزاً.

### الخطوة 5: قراءة الخروج

بعد 2000 خطوة:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

ليس شكسبير، ولكن شكسبير، فوز واضح لـ 800 ألف برميل و 6 دقائق على جهاز كمبيوتر محمول

## استخدمها

هذه الحجر الرئيسي هو معمارة مرجعية ثلاثة امتدادات لنشرها إلى شيء حقيقي:

1. **Swap the tokenizer.**استخدم بيبي (مثل `tiktoken.get_encoding("cl100k_base")`يرتفع حجم الكلمات من 65 إلى ~ 50,000. تحتاج القدرة النموذجية إلى التوسع للتعويض.
2. **Train on a bigger corpus.**استخدام`OpenWebText`أو`fineweb-edu`10B رموز على A100 واحد يستغرق حوالي 24 ساعة ل 125M-param GPT.
3. **Add RoPE + KV cache + Flash Attention.**تمارين أدناه ستقوم بتجريبهما

ينتهي هذا الأمر باعتباره GPT بمعيار 125M الذي يولد اللغة الإنجليزية السريعة. ليس نموذجًا حدوديًا. ولكن نفس مسار الشفرة  أكبر فقط  هو ما يستخدمه كارباتي، إيليوثرااي، ومعهد ألين لتدريب نقاط التفتيش للبحوث في عام 2026.

## أرسله

انظر`outputs/skill-transformer-review.md`. المهارة تدرس تنفيذ محول من الصفر لتحقيق الدقة في جميع الدروس السابقة الثلاثة عشر.

## التمارين

1. **Easy.**أركض`code/main.py`تأكد من أن خسارة التحقق من التحقق من النموذج الذي تدرب عليه في الخطوة الأخيرة أقل من 2.0. تغيير `max_steps`من 2000 إلى 5000  هل يواصل فقدان القيمة التحسن؟
2. **Medium.**استبدل التوابل الموضعية المتعلمة بالروبي.`MultiHeadAttention`-توقفوا و تحققوا من انخفاض ضياع القيمة
3. **Medium.**تنفيذ كاش كيه في حلقة العينات. توليد 500 رمز مع ودون كاش. يجب أن تتحسن الساعة الحائطية 520 × على جهاز كمبيوتر محمول.
4. **Hard.**إضافة رأس ثان إلى النموذج الذي يتوقع التوقعات التالية زائد واحد (MTP  التوقعات متعددة التوكن من DeepSeek-V3). تدريب المشترك. هل يساعد؟
5. **Hard.**استبدل FFN الفردي لكل كتلة مع 4 خبراء MoE. راوتر + طريق 2 أعلى. انظر كيف تتغير فقدان val عند المماثلة المعلمات النشطة.

## الشروط الرئيسية

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

## المزيد من القراءة

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) التنفيذ الكلاسيكي الملاحظ.
