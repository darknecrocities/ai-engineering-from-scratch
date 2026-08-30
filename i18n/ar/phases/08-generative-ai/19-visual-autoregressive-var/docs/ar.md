# النمذجة المرئية السلبية (VAR): التنبؤ على النطاق التالي

> نموذجات التوزيع عينة تكراريا في الوقت (تخليل الخطوات). تمثيلات VAR تكراريا في المقياس  يتوقع رمز 1x1 ، ثم 2x2 ، ثم 4x4 ، حتى القرار النهائي ، كل مقياس تكييف على السابق. أظهرت ورقة 2024 أن VAR تتطابق مع قوانين تحليل النطاق على النمط GPT لتوليد الصور وتغلب على DiT في نفس ميزانية الحساب. هذا الدروس يبني الآلية الأساسية.

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 minutes

## المشكلة

كان هناك محاولتين رئيسيتين للاتصال بالواقع قبل عام 2024: PixelRNN/PixelCNN (بيكسل بيكسل) و DALL-E 1 / Parti / MuseGAN (علامات عن طريق العلامات على رموز VQ-VAE).

عانى كلا من هذه النقاط من مشكلة ترتيب الجيل. يتم ترتيب البيكسلات والرموز في شبكة 2D ، ولكن يجب على نموذج AR زيارةها في ترتيب 1D raster. لا يعرف البيكسل الزاوية المبكرة ما يصبح الصورة في النهاية. تحسنت جودة الجيل أسوأ من GPT-on-text ولم تصل أبدا إلى جودة نموذج التوزيع عند الحساب المماثل.

تقوم VAR بتحديد مشكلة ترتيب الجيل من خلال تغيير ما يتم توليده. بدلاً من توقع رموز الصورة واحدة تلو الأخرى في الفضاء، تتوقع VAR صورة كاملة عند ارتفاع القرارات. الخطوة 1: تتوقع رمزا 1x1 (الصورة العامة "التالي"). الخطوة 2: تتوقع شبكة رموز 2x2 (الميزات الأكثر قاسية). الخطوة 3: تتوقع شبكة 4x4. الخطوة K: تتوقع شبكة النهائية (H/8) x ((W/8)).

كل مقياس يحتفظ بجميع المقياس السابقة (بسبب "نظام المقياس") والموازية داخل مقياسه الخاص. تختفي مشكلة النظام: يتم إنتاج الصورة بأكملها في المقياس k في مرسلة تحول واحدة.

## المفهوم

### VQ-VAE Tokenizer متعدد النطاقات

VAR تحتاج إلى**multi-scale discrete tokenizer**. لصور x ، فإنه ينتج سلسلة من الشبكات الرمزية عالية الدقة تدريجيا:

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

كل z_k يستخدم نفس كتاب التعليمات (القياس النموذجي 4096-16384). لا يتم تعيين الرموز على كل مقياس مستقل  يتم تدريبها بحيث يتم إعادة جمع بقايا كل مقياس f:

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

هذا هو**residual VQ**تغير. مقياس k يلتقط ما المقياس 1..k-1 غاب. القيادة تأخذ مجموع جميع التوابل في المقياس ويجعل الصورة.

يتم تدريب رمز VQ متعدد النطاق مرة واحدة (مثل VQGAN) ثم تجميد. يتم القيام بكل العمل التوليدي من قبل النموذج السريع على الطرف العلوي.

### التنبؤات القادمة

النموذج التوليد هو محول يرى الرموز من جميع المقاييس السابقة ويتوقع الرموز في المقاييس التالية.

هيكل تسلسل المدخل:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

تضمنت إدخالات الموقف كل من مؤشر المقياس وموقع المكان داخل المقياس. الاهتمام سببي في الترتيب من النطاق: الوهم على النطاق k، الموقف (i، j) يمكن أن يشاهد جميع الرموز على النطاق 1..k والرموز على النطاق k نفسها التي تأتي في وقت سابق في أي ترتيب داخل النطاق يستخدم (VAR يستخدم الاهتمام الموقف الثابت دون وجود سببية داخل النطاق  يتم التنبؤ بالجميع من المواقع داخل النطاق بالتوازي).

خسارة التدريب: في كل مقياس k، توقع الرموز z_k مع جميع الرموز القياسية السابقة. فقدان الانتروبية المتقاطعة على رموز VQ المفصلة. نفس الهيكل مثل GPT باستثناء "الترتيب" هو الآن بنية مقياسية.

### الجيل

عند الاستنتاج:
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

بالنسبة إلى مقياس K = 10 ، فإن التوليد هو 10 مرسلات محولة للأمام. كل مرسلة تنتج مقياسها بأكمله بالتوازي  لا يوجد تراجع للطاقة في داخل مقياس. بالنسبة إلى صورة 256x256 ، هذا حوالي 10 مرسلات مقابل 28-50 من DiT.

### لماذا يربح النطاق التالي على النطاق التالي

ثلاثة انتصارات هيكلية:
1. **Coarse-to-fine aligns with natural image statistics.**يظهر بيانات الإدراك البصري البشري ومجموعات البيانات الصورة على حد سواء دقة تعتمد على المقياس: هيكل التردد المنخفض مستقرة وقابلة للتنبؤ؛ وتشكل التفاصيل عالية التردد مشروطة بمحتويات التردد المنخفض. تستغل التنبؤات القادمة هذا.
2. **Parallel generation within scale.**على عكس AR رمزية على النمط GPT ، تنتج VAR جميع الرموز على مقياس في خطوة واحدة. طول توليد فعال هو مقياس التسجيل بدلاً من خطي.
3. **No generation order bias.**تُرى الرموز في الحجم k جميع الحجم k-1؛ لا يوجد تعصب "أيسر" أو "أعلى" يفرض على الرموز المبكرة الالتزام قبل أن يكون السياق المتأخر متاحاً.

### قانون الحد الأقصى

(تيان) وآخرون أظهرت أن VAR تتبع منحنى تحديد النطاق القومي لـ FID على ImageNet  تماماً كما تفعل GPT بالنسبة للغموض. مضاعفة المعلمات أو الحساب بشكل موثوق يقلل من الأخطاء إلى النصف. كان هذا أول نموذج تصويري يظهر هذا النوع من السلوك التوسعي بشكل نظيف مثل نموذج اللغة. النتيجة هي أن التنبؤات على مقياس VAR تصبح قابلة للتنبؤ من الحسابات ، وليس التخمينات التجريبية لكل معماري.

### العلاقة مع الانتشار

يشترك VAR والانشطاع في نفس قصة ضغط البيانات: كلاهما ينفصل مشكلة التوليد إلى سلسلة من المشاكل الفرعية الأسهل.

- التنشر: إضافة ضجيج تدريجيًا، تعلم إعادة خطوة واحدة.
- إضافة القرار تدريجياً، تعلم التنبؤ بالمقياس التالي.

فهي محور مختلفة من خلال المشكلة. كلاهما يعطي توزيعات مشروطة قابلة للتعامل. تجريبيا VAR أسرع في الاستنتاج (أقل مرورات، جميعها متوازية داخل مقياس) وتطابق أو تضرب DiT على ImageNet المشروط للصف. VAR مشروط النص (VARclip، HART) هو اتجاه بحث نشط.

```figure
gx-var-next-scale
```

## بناءها

في`code/main.py`ستقوم:
1. بناء صغير **multi-scale VQ tokenizer**على بيانات "الصورة" الصناعية (2 حلقات غوسية بعمر).
2. - إقترب**VAR-style transformer**للتنبؤ بالرموز على النطاق التالي
3. عينة من خلال استدعاء المحول 4 مرات (4 مقياس) و فك تشفير.
4. التحقق من أن التدريب على نطاق مرتب يجعل التوليد متوازياً داخل النطاق.

هذا هو تنفيذ لعبة. النقطة هي رؤية قناع الاهتمام المُهيّن على النطاق والإنتاج المتوازي داخل النطاق يعمل فعلاً.

## أرسله

هذا الدرس يُنتج`outputs/skill-var-tokenizer-designer.md` مهارة لتصميم رمز متعدد النطاقات: عدد النطاقات، نسبة النطاقات، حجم دفتر التعليمات، مشاركة النفايات، بنية القيادة.

## التمارين

1. **Scale count ablation.**قم بتدريب VAR مع 4, 6, 8, 10 مقياس. قياس جودة إعادة الإعمار مقابل عدد الممرات السريعة. المزيد من المقاييس = بقايا أكثر دقة = جودة أفضل ولكن أكثر الممرات.

2. **Codebook size.**تعديل الـ "توكينيزر" مع أرقام الكود 512، 4096، 16384 أرقام الكود الكبيرة تعطي إعادة بناء أفضل ولكن التنبؤ أصعب.

3. **Parallel-within-scale check.**بالنسبة لـ VAR المدرب، قم بقياس نمط الاهتمام صراحة. داخل مقياس k، هل يحتفل النموذج بالمواقع المتعددة المقياسات ولكن ليس داخل المقياس؟ تحقق من تنفيذ القناع.

4. **VAR vs DiT scaling.**بالنسبة لمهمة نفسها في مجال ImageNet ، قم بتدريب VAR و DiT بميزانيات المعايير المقابلة (مثل 33M ، 130M ، 458M). قم بتحديد FID مقابل الحساب. يجب أن يسحب VAR من قبل DiT في كل حجم  إعادة إنتاج نتيجة الورقة على نطاق صغير.

5. **Text conditioning.**توسيع VAR لتأخذ إضافة نصية (CLIP جمع) كإدخال إضافي للتعويض عبر adaLN. هذه وصفة HART. كم تحسن FID على أخذ العينات المتحالفة مع النص؟

## الشروط الرئيسية

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

## المزيد من القراءة

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) ورقة VAR، مرجع القنوني
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) DiT، نقطة قياس التوزيع
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) VQGAN، عائلة الوسائط VAR الوسائط المتعددة
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) VQ-VAE، أساس رمزية الصورة المفصلة
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) VAR مشروطة بالنص
