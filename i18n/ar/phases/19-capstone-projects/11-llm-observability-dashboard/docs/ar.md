# كابستون 11  LLM الملاحظة و Eval Dashboard

> لانغفوز) أصبح مفتوحاً) أرسلت أريز فينيكس خرائط 2026 GenAI semconv. هليكون وبرينترست كلتا تضاعفت في تخصيص التكاليف لكل مستخدم أصبح OpenLLMetry من Traceloop أداة SDK الفعلية. شكل الإنتاج هو ClickHouse للآثار، Postgres للبيانات المعدنية، Next.js للUI، وجيش صغير من وظائف التقييم (DeepEval، RAGAS، LLM-قاضي) التي تمر على أثر العينات. قم ببناء واحدة مضيفة ذاتية، واستوعب من أربعة عائلات على الأقل من SDK، وأظهر التقاط رجعة الحقن في أقل من خمس دقائق.

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:**P11 · P13 · P17 · P18
**Time:** 25 hours

## المشكلة

كل فريق من أفراد الذكاء الاصطناعي الذين يديرون حركة الإنتاج في عام 2026 يحافظون على مستوى قابل للملاحظة جنبا إلى جنب مع النموذج. إخصاب التكاليف اكتشاف الهلوسة مراقبة التدفق إشارة إختراق السجن لوحة التحكم SLO. إنذارات تسرب المعلومات الإشارات مفتوحة المصدر  لانغفوز، فينيكس، OpenLLMetry  تجمع على OpenTelemetry GenAI الاتفاقيات الترجمية كخطط استهلاك. يمكنك الآن استخدام أدوات OpenAI، Anthropic، Google، LangChain، LlamaIndex، و vLLM مع SDK واحد ونقل الموافقات.

ستقوم ببناء لوحة تحكمية مضيفة ذاتية تستخدم أربع أسرة على الأقل من أجهزة SDK ، وتشغيل مجموعة صغيرة من أعمال تقييم على آثار عينات ، وتكتشف التنحدر ، والإشعارات. شريط القياس: بالنظر إلى تراجع مدرك (إشارة تبدأ في إنتاج PII) ، فإن لوحة التحكم تسلمها وتطلق إشعارا في أقل من خمس دقائق.

## المفهوم

إنجست هو OTLP HTTP. إن SDK ينتج التسعيرات GenAI-semconv: `gen_ai.system`،`gen_ai.request.model`،`gen_ai.usage.input_tokens`،`gen_ai.response.id`،`llm.prompts`،`llm.completions`. يمتد الأرض في ClickHouse لتحليلات العمود؛ البيانات المعدنية (المستخدمين والجلسات والتطبيقات) الأرض في Postgres.

تعمل إيفال على أنها عمل تجميعي على آثار عينات. تسجل DeepEval الولاءة والسمية والسوابية. تسجل RAGAS مقاييس الاسترداد عندما يحمل المسار سياق الاسترداد. يقوم قضاة LLM المخصصون بتشغيل عمليات التحقق من المجال المحدد (فشرة PII ، استجابة خارج السياسة). تسجل إيفال مرة أخرى إلى نفس ClickHouse كما تتصل فترات التقييم المرتبطة بالمسار الأولي.

تراقب أوقات الكشف عن التدفق توزيعات الفضاء المضمنة على مر الزمن (اختلاف PSI أو KL على التدفقات السريعة) بالإضافة إلى اتجاهات درجة التقييم. تغذى التحذيرات على Prometheus Alertmanager ثم Slack / PagerDuty. واجهة المستخدم هي Next.js 15 مع Recharts.

## الهندسة المعمارية

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## الـ"كثيرة"

- إحتمال: OpenTelemetry SDKs + GenAI الاتفاقيات الترجمية؛ نقل HTTP OTLP
- المجموعة: OpenTelemetry المجموعة مع معالج عينة الذيل (لسيطرة على التكاليف)
- التخزين: ClickHouse لفترات، Postgres للوفائض المعلومات، S3 لملف الأحداث الخام
- Evals: DeepEval، RAGAS 0.2، Arize Phoenix تقييم حزمة، المخصصة ماجستير في القانون-قضاة
- التدفق: PSI / KL على التوابل المشتركة السريعة (محولات الجملة) أسبوعيا
- الإشعار: Prometheus Alertmanager -> Slack / PagerDuty
- واجهة المستخدم: Next.js 15 App Router + Recharts + أعمال الخادم
- SDKs مدعومة خارج الصندوق: OpenAI، انتروبيك، جوجل جناي، لانج تشين، لاماينديكس، vLLM

```figure
ce-otel-drift
```

## بناءها

1. **Collector config.**OpenTelemetry Collector مع جهاز استقبال HTTP OTLP ، ومعينة الذيل التي تحافظ على 100٪ من آثار الخطأ و 10٪ من النجاحات ، والمصدرين إلى ClickHouse و S3.

2. **ClickHouse schema.**الجدول `spans`مع عمود تعكس GenAI semconv: `gen_ai_system`،`gen_ai_request_model`،`input_tokens`،`output_tokens`،`latency_ms`،`prompt_hash`،`trace_id`،`parent_span_id`، بالإضافة إلى حقيبة JSON لتحملات فائدة طويلة. إضافة مؤشرات ثانوية بواسطة user_id و app_id.

3. **SDK coverage test.**اكتب تطبيقًا عميلًا صغيرًا باستخدام كل SDK (OpenAI ، Anthropic ، Google ، LangChain ، LlamaIndex ، vLLM) مع OpenLLMetry الآلي. تحقق من أن كل واحد ينتج امتدادات GenAI القنونية التي تهبط في ClickHouse.

4. **Eval jobs.**يقرأ الوظيفة المخططة آثار العينات التي تمت في آخر 15 دقيقة ويقوم بتشغيل وفاء DeepEval، والسمومية، وارتباط الإجابة.

5. **Custom LLM-judge.**قاضي تسريب المعلومات الشخصية: إذا تم إعطاء رد، اتصل بمحافظ قانونية للحراسة لتحديد احتمال تسريب المعلومات الشخصية.

6. **Drift detection.**العمل الأسبوعي يحسب PSI بين هذا الأسبوع جمع التوابل السريعة وخط الأساسية الـ4 الأسابيع المتبقية.

7. **Dashboard.**Next.js 15 مع صفحات: المحة العامة (مدة/ ثانية، تكلفة/ مستخدم، تأخير p95) ، والسجلات (البحث + شلال السيل) ، والتحقيقات (اتجاه الوفاء، السمومية) ، والانتقال (PSI مع مرور الوقت) ، والإشعارات.

8. **Alerting chain.**يقرأ مصدر Prometheus مجموعات درجة التقييم وفرص المدة المتأخرة؛ أورترمانجر طرق إلى Slack للحصول على التحذيرات و PagerDuty للانتهاكات الحرجة.

9. **Regression probe.**حقن خطأ: يبدأ الروبوت المكتشف في تسريب SSNs مزيفة في 1% من الأحيان. قياس MTTR: من خطأ نشر إلى تنبيه Slack.

## استخدمها

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## أرسله

`outputs/skill-llm-observability.md`هو المنتج. في إطار تطبيق LLM، تتناول لوحة التحكم آثارها، وتقوم بتقييمات، وإشعارات عن التجول، وتظهر تقسيم التكلفة/المستخدم في Next.js.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Trace-schema coverage | Number of SDK families producing canonical GenAI spans (target: 6+) |
| 20 | Eval correctness | DeepEval / RAGAS scores vs hand-labeled set |
| 20 | Dashboard UX | MTTR on injected regression (under 5 minutes target) |
| 20 | Cost / scale | Sustained ingest at 1k spans/sec without backlog |
| 15 | Alerting + drift detection | Prometheus/Alertmanager chain exercised end to end |
| **100** | | |

## التمارين

1. إضافة أدوات مخصصة لإطار Haystack. التحقق من أن المدى القنوني يصل في ClickHouse مع المخلصين `gen_ai.*`الصفات

2. تغيير DeepEval للمقيّمين في فينيكس على نفس المسارات، وتحديد التذبذب بين محركات تقييم.

3. تحدّد جهاز الكشف عن التجرف: حساب PSI لكل هوية التطبيق بدلاً من العالميّة.

4. إضافة صفحة "أثر المستخدم": تكلفة لكل مستخدم ومعدل الفشل لكل مستخدم مع خطوط إشعاع.

5. بناء سياسة أخذ عينات الذيل التي تحتفظ بنسبة 100٪ من الأثر مع السمومة > 0.5 + عينة طبقة من البقية بنسبة 10٪.

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM attributes" | 2025 OpenTelemetry spec for LLM span attributes (system, model, tokens) |
| Tail sampling | "Post-trace sample" | Collector decides to keep or drop a trace after it completes (can peek errors) |
| PSI | "Population stability index" | Drift metric comparing two distributions; > 0.2 typically signals meaningful drift |
| LLM-judge | "Eval as model" | An LLM scoring another LLM's output on a rubric (faithfulness, toxicity, PII) |
| Tail-sampling policy | "Keep-rule" | Rule that decides which traces to persist vs drop; errored + sample-rate |
| Eval span | "Linked eval trace" | Child span carrying an eval score linked to the original LLM call span |
| Cost per user | "Unit economics" | Dollar cost attributed to a user_id over a window; key product metric |

## المزيد من القراءة

- [Langfuse](https://github.com/langfuse/langfuse) منصة الملاحظة المفتوحة للقوى المرجعية
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) مرجع بديل مع دعم قوي للتحرك
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) عائلة SDK ذاتية الاستعمال
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) مخطط الإستهلاك
- [Helicone](https://www.helicone.ai) قابلية مراقبة مضيفة بديلة
- [Braintrust](https://www.braintrust.dev) منصة التقييم الأولى البديلة
- [ClickHouse documentation](https://clickhouse.com/docs) مخزن المدة العمودية
- [DeepEval](https://github.com/confident-ai/deepeval) مكتبة المقيّمين
