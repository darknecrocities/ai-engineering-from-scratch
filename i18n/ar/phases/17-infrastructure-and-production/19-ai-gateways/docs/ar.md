# بوابات الذكاء الاصطناعي  LiteLLM، Portkey، Kong AI Gateway، Bifrost

> يقع بوابة بين تطبيقاتك ومقدمي النموذج. الميزات الأساسية هي توجيه المقدم، والعودة إلى الوراء، والجربات المرتددة، ومحدود السعر، والإشارات السرية، واللاحظة، والحواجز. انقسام السوق في 2026: **LiteLLM**هو MIT OSS مع 100 + مزود، متوافق مع OpenAI، ولكن ينفصل حول ~ 2000 RPS (8 GB ذاكرة، فشل في التقييمات المنشورة) ؛ أفضل لـ Python، <500 RPS، تطوير / النماذج الأولية. **Portkey**هو جهاز التحكم الموضحة (حراسة، تحرير PII، اكتشاف jailbreak، مسارات التدقيق) ، ذهب Apache 2.0 مفتوح المصدر مارس 2026, 20-40 ms تأخر فوق، $49/mo production tier. **Kong AI Gateway** built on Kong Gateway — Kong's own benchmark on same 12 CPUs: 228% faster than Portkey, 859% faster than LiteLLM; $100/نموذج/شهر (ماكس 5 على مستوى Plus) ؛ مناسبة للشركات إذا كنت بالفعل على Kong. **Bifrost**(ماكسيم AI)  التجربة التلقائية مع إعادة التشغيل قابلة للتكوين، والعودة إلى الأنثروبيك على OpenAI 429. **Cloudflare / Vercel AI Gateways** إدارة، صفر عمليات، إعادة المحاولة الأساسية. إقامة البيانات تدفع قرار مضيف الذاتي. يجلس بورتكي وكونغ في الوسط مع إدارة OSS + اختياريًا.

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 minutes

## أهداف التعلم

- قم بإدراج الخصائص الستة الرئيسية للبوابة (التوجيه والإرجاع والإعادة المحاولة والحدات السريعة والسرات واللاحظة والحواجز).
- خريطة أربعة بوابات 2026 (LiteLLM ، Portkey ، Kong AI ، Bifrost) لتنمية السقف وقضايا الاستخدام.
- أذكر مؤشر كونغ (228٪ مقابل بورتكي، 859٪ مقابل لايت إل إم) وشرح لماذا يهم بالنسبة > 500 ريبس.
- اختر المضيف الذاتي مقابل المدار مع إعطاء إقامة البيانات وميزانية العمليات.

## المشكلة

إن منتجك يدعى OpenAI، Anthropic، و Llama المضيفة الذاتية. لكل مزود SDK مختلفة، نموذج الخطأ، حد السعر، وخطة التأليف. تريد الإفشل (إذا OpenAI 429s، جرب Anthropic) ، متجر إثبات واحد، قابلية الملاحظة الموحدة، ومحدود السعر لكل مستأجر.

إعادة اختراع هذا في طبقة التطبيق يربط كل خدمة مع كل مزود. طبقة البوابة تعززها في عملية واحدة مع API واحدة (عادة متوافقة مع OpenAI) التي تعزز إلى مزودي.

## المفهوم

### ستة ميزات أساسية

1. **Provider routing** OpenAI، Anthropic، Gemini، self-hosted، إلخ. وراء API واحدة.
2. **Fallback**على 429، 5xx، أو فشل في الجودة، حاول مرة أخرى في مكان آخر.
3. **Retries** التخلف المُتَعَدّد، المحاولات المُحدّدة.
4. **Rate limits** لكل مستأجر، لكل مفتاح، لكل نموذج
5. **Secret references** سحب إثباتات من الخزنة في وقت تشغيل (لا في التطبيق).
6. **Observability** سمات OTel + GenAI (مرحلة 17 · 13) + تعريف التكاليف.
7. **Guardrails** تحرير المعلومات الشخصية، اكتشاف الإختراقات السرية، مرشحات المواضيع المسموح بها.

### LiteLLM  MIT OSS، بيثون

- 100 مزود، متوافق مع OpenAI، إعداد الجهاز التوجيهي، الخلفية، قابلية للملاحظة الأساسية.
- يختفي حوالي 2000 ريبس في مقياس كونغ؛ 8 جيجابايت من ذاكرة الوصول، فشل في التقييم تحت الحمل المستمر.
- أفضل تناسب: تطبيق Python، <500 RPS، بوابات تطوير/موجبة، توجيه تجريبي.
- التكلفة: 0 دولار لـ OSS؛ مستوى مجاني من السحابة موجود.

### المفتاح  وضع طائرة التحكم

- أباتشي 2.0 أوسس اعتبارا من مارس 2026، الحراسة، تحرير المعلومات الشخصية، اكتشاف الإختراقات القبضية، مسارات التدقيق.
- 20-40 ms لكل طلب تأخير التكلفة العليا.
- 49 دولارًا في الشهر لمستوى الإنتاج مع الاحتفاظ + SLA.
- أفضل تناسب: الصناعات المنظمة التي تحتاج إلى الحواجز + قابلية الملاحظة المجمعة.

### كونغ إيه إيه بيتواي

- بنيت على بوابة كونغ (نتج بوابة API الناضج ، lua+OpenResty).
- مقياس "كونغ" الخاص على ما يعادل 12 وحدات مركزية: 228% أسرع من "بورتكي"، 859% أسرع من "ليتيلم".
- السعر: 100 دولار للنموذج / الشهر، ماكسب 5 على مستوى زائد.
- أفضل تناسب: بالفعل على Kong؛ > 1000 RPS؛ مستعدة للحصول على الترخيص.

### الثلج الثنائي (ماكسيم AI)

- محاولات إعادة تلقائيّة مع إعادة التشغيل القابلة للتشغيل
- العودة إلى الأنثروبيك على OpenAI 429 هي وصفة تقليدية.
- المُدخل الجديد، التجاري.

### مدخل Cloudflare AI / مدخل Vercel AI

- تم إدارة عمليات صفر، محاولة إعادة أساسية و قابلية للملاحظة
- أفضل تناسب: تطبيقات جاوا سكريبت الخدمة الحافة على Cloudflare / Vercel.
- محدود مقارنة بـ"كونج"/"بورتكي" على الحواجز وحدود السعر

### المضيفة الذاتية مقابل المدارة

إقامة البيانات هي وظيفة الإجبار. الرعاية الصحية والمالية المضيفة الذاتية الافتراضية (LiteLLM أو Portkey OSS أو Kong). منتجات المستهلك المضيفة الافتراضية المدارة (Cloudflare AI Gateway) أو المتوسطة المستوى (Portkey managed). الهجين: المضيفة الذاتية للمستأجر المنظم ، المدارة للآخرين.

### ميزانية التأخير

- التلفة المضافة: 5-15 ms عادة
- 20-40 ميس فوق
- 3-8 ميس فوق
- Cloudflare/Vercel: 1-3 ms التكلفة العامة (ميزة الحافة).

تأخير البوابة يضيف مباشرة إلى TTFT. بالنسبة إلى TTFT P99 < 100 ms SLA ، Kong أو Cloudflare. بالنسبة إلى P99 < 500 ms ، أي.

### المادة التعريفية المحددة

يعمل بطانية رمز بسيطة حتى نطاق معتدل. يتطلب المتداولين متعددة نافذة زلقة + مساعدة انفجار + تطبيقات لكل مستأجر. LiteLLM يركب بطانية رمزية؛ يركب Kong سفينة زلقة؛ يركب Portkey سفينة.

### البوابة + قابلية للملاحظة + توجيه

المرحلة 17 · 13 (التلاحظ) + 16 (التوجيه النموذجي) + 19 (البوابات) هي نفس الطبقة في الإنتاج. اختر أداة واحدة تغطي كل ثلاثة أو قم بتشغيلها بعناية: معظم عمليات نشر 2026 تجمع بين هليكون (التلاحظ) أو بورتكي (المنصات) مع كونغ (المقياس) لدورات منفصلة.

### أرقام يجب أن تتذكر

- ليتللم: انقطاع عند 2000 ريبس، ذاكرة 8 جيجابايت.
- مفتاح البورط: 20-40 ms فوق التكلفة؛ Apache 2.0 منذ مارس 2026.
- كونغ: 228% أسرع من بورتكي، 859% أسرع من ليتيللم.
- أسعار كونغ: 100 دولار/نموذج/شهر، 5 أقصى على مستوى زائد.
- Cloudflare/Vercel: 1-3 ms فوق الحافة.

```figure
mx-gateway-fallback
```

## استخدمها

`code/main.py`يحاكي توجيه البوابة مع تعطل عبر 3 مزودي تحت حقن 429/5xx. يبلغ عن تأخر، معدل إعادة المحاولة، ومعدل ضربات تعطل.

## أرسله

هذا الدرس يُنتج`outputs/skill-gateway-picker.md`بالنظر إلى النطاق، وضع العمليات، الامتثال، ميزانية التأخير، يختار البوابة.

## التمارين

1. أركض`code/main.py`. إعداد الخلف من OpenAI→Anthropic→مضيفة ذاتية. ما هو معدل الإصابة المتوقع عند معدل خطأ المزود 5٪؟
2. إنّ إمكانية التسجيل الخاصة بك هي TTFT P99 < 200 ms على خط أساسي 300 ms. أيّ بوابات تبقى ضمن الميزانية؟
3. العميل في الرعاية الصحية يحتاج إلى إضافة شخصية ذاتية + إصدار المعلومات الشخصية + مراجعة.
4. مقارنة LiteLLM vs Kong: عند أي سقف RPS يجب أن ينتقل فريق؟
5. تصميم سياسة الحد من الأسعار لمجموعة متعددة المستأجرين SaaS: مستوى مجاني، مستوى تجريبي، مستوى مدفوع. علبة الوهم أو نافذة زلقة؟

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Gateway | "API broker" | Process sitting between apps and providers |
| LiteLLM | "the MIT one" | Python OSS, 100+ providers, breaks at 2K RPS |
| Portkey | "guardrails gateway" | Control plane + observability, Apache 2.0 |
| Kong AI Gateway | "the scale one" | Built on Kong Gateway, benchmark leader |
| Bifrost | "Maxim's gateway" | Retries + Anthropic fallback recipe |
| Cloudflare AI Gateway | "edge managed" | Edge-deployed managed gateway, zero-ops |
| PII redaction | "data scrub" | Regex + NER mask before sending to model |
| Jailbreak detection | "prompt injection guard" | Classifier on user input |
| Audit trail | "regulated log" | Immutable record of every LLM call |
| Token-bucket | "simple rate limit" | Refill-based rate limiter |
| Sliding-window | "precise rate limit" | Time-windowed rate limiter; better fairness |

## المزيد من القراءة

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
