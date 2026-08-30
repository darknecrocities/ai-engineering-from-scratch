# A2A  بروتوكول وكيل إلى وكيل

> المكتب العمومي هو العميل إلى الأداة. A2A (Agent2Agent) هو وكيل إلى وكيل بروتوكول مفتوح للسماح وكلاء غير مرئية بنيت على إطار مختلف التعاون. أصدرتها جوجل في أبريل 2025 ، وتبرعت إلى مؤسسة لينكس في يونيو 2025 ، ووصلت إلى v1.0 في أبريل 2026 مع 150 + مؤيدين بما في ذلك AWS و Cisco و Microsoft و Salesforce و SAP و ServiceNow. لقد استوعب نظام إيك بي ام ACP و أضاف تمديد المدفوعات AP2. هذا الدروس يتناول بطاقة العميل، دورة حياة المهمة، والترابطين النقلين.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## أهداف التعلم

- تمييز بين حالة استخدام وكيل إلى أداة (MCP) من حالة استخدام وكيل إلى وكيل (A2A).
- نشر بطاقة العميل في `/.well-known/agent.json`مع المهارات و البيانات المعدنية النهائية.
- إرسال دورة حياة المهمة (المقدم → العمل → مطلوب المدخل → اكتمل / فشل / إلغاء / رفض).
- استخدم رسائل مع الأجزاء (النص، الملف، البيانات) والقطع الأثرية كمخرجات.

## المشكلة

وكيل خدمة العملاء يحتاج إلى تفويض كتابة التقرير إلى وكيل كاتب متخصص. الخيارات قبل A2A:

- إطار إطار إعادة التأهيل المخصص يعمل لكن كل إزواج هو مرة واحدة
- قاعدة شفرة مشتركة تتطلب من العملاء أن يستخدموا نفس الإطار
- المكسبين: لا يناسب: المكسبين هو للدعوة الأدوات، وليس لعاملين يتعاونونون مع الحفاظ على كل عامل من التفكير الداخلي غير الشفاف.

A2A تملأ الفجوة. فإنه ينمثل التفاعل عندما يرسل وكيل واحد مهمة إلى آخر، مع دورة حياة، والرسائل، والقطع الأثرية. يبقى حالة الداخلية للوكيل المطلق غير واضحة  يرى المدعو فقط عمليات انتقال حالة المهمة والمخرجات النهائية.

A2A هو بروتوكول "دع العملاء عبر الإطار يتحدثون مع بعضهم البعض". لا يحل محل MCP؛ والثنين يكملون.

## المفهوم

### وكيل بطاقة

كل وكيل متوافق مع A2A ينشر بطاقة في `/.well-known/agent.json`:

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

الاكتشاف يستند إلى عناوين URL: احضر البطاقة، تعلم عناوين URL نقطة نهاية A2A، إعداد المهارات.

### بطاقات العميل الموقعة (AP2)

إضافة AP2 (سبتمبر 2025) توقيعات رمزية إلى بطاقات وكيل. ناشر يوقع بطاقته الخاصة مع JWT؛ المستهلكين التحقق. يمنع التمثيل.

### دورة حياة المهمة

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

العملاء يبدأون مع `tasks/send`. يتحول العميل المُتصل عبر الدول؛ يشترك العملاء في تحديثات الدولة عبر SSE أو استطلاع.

### الرسائل والأجزاء

رسالة تحمل واحدة أو أكثر من الأجزاء:

- `text` محتوى واضح
- `file` قاعدة64 البقع مع mimeType.
- `data` كتب تحميل JSON (إدخال بنية للوكيل المطلوب).

مثال:

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### الأثاث

المخرجات هي الأثاث، وليس السلاسل الخام. الأثاث هو المخرج المسمى، المخطط:

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

يمكن نشر الأثاث كجزء من الكائنات

### إرتباطان في النقل

1. **JSON-RPC over HTTP.** `/a2a`نقطة نهاية، POST لطلبات، SSE اختياري للتدفق. التزامن الافتراضي.
2. **gRPC.**للبيئات المؤسسية حيث تكون gRPC محلية.

كلتا الارتباطات تحمل نفس شكل الرسالة المنطقية.

### الحفاظ على الكفاءة

مبدأ تصميم رئيسي: حالة العميل الداخلي المطلق غير شفافة. يرى المدعو حالة المهمة والقطع الأثرية. سلسلة الأفكار للعميل المطلق، ودعوات أدائه، وفوضيته الفرعية للعميل غير مرئية. هذا يختلف عن MCP، حيث الدعوات الأداة شفافة.

المنطق: A2A تمكن المنافسين من التعاون دون الكشف عن الداخلية. A2A يمكن أن يكون "تصل هذه وكيل خدمة العملاء" دون أن يتعلم المتصل كيفية تنفيذ هذا وكيل الخدمة.

### خط زمني

- **2025-04-09.**جوجل تعلن عن A2A
- **2025-06-23.**تبرع لمؤسسة لينكس
- **2025-08.**يمتص جهاز "إيك بي إم"
- **2025-09.**سفن AP2 تمديد (دفع العملاء).
- **2026-04.**إصدار v1.0 مع 150 + منظمة دعم.

### العلاقة مع المؤسسة

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | JWT-signed Agent Cards (AP2) |
| Transport | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

استخدم MCP عندما تريد استدعاء أداة محددة. استخدم A2A عندما تريد تفويض مهمة كاملة إلى وكيل آخر. العديد من أنظمة الإنتاج تستخدم كليهما: وكيل يستخدم MCP لتطبيق أداة و A2A لتطبيق تعاونه.

```figure
a2a-task-lifecycle
```

## استخدمها

`code/main.py`يطبق قاعدة A2A الحد الأدنى: وكيل البحث ينشر بطاقته، وكيل الكتاب يحصل على `tasks/send`مع أجزاء بما في ذلك PDF و تعليمات نصية، الانتقالات من خلال العمل → input_required → working → اكتمل، وتعيد عنصر نصي. جميع stdlib؛ يستخدم نقل في الذاكرة للتركيز على أشكال الرسائل.

ما الذي يجب أن ننظر إليه:

- شكل بطاقة العميل JSON.
- تحديد الهوية المهام والتحولات الحكومية
- رسائل ذات أجزاء مختلطة
- الإدخال المطلوب في منتصف المهمة
- العودة الفنية عند الانتهاء.

## أرسله

هذا الدرس يُنتج`outputs/skill-a2a-agent-spec.md`. بالنظر إلى وكيل جديد يجب أن يكون قابلاً للدعوة من قبل وكلاء آخرين، فإن المهارة تنتج بطاقة وكيل JSON، مخطط مهارات، ورسم خط النهاية.

## التمارين

1. أركض`code/main.py`. تتبع دورة حياة المهمة بأكملها، بما في ذلك الموقف المطلوب من المدخلات عندما يطلب الوكيل المشارك توضيحا.

2. إضافة بطاقة وكيل موقعة وقع مع HMAC على بطاقة JSON القنوني كتابة مؤكدة وتأكيد أنها تفشل على بطاقة متحولة

3. تنفيذ سلسلة المهام: يقوم وكيل الكتاب بإصدار ثلاث قطع متزايدة من الأدوات على SSE ويتراكمها المدعو.

4. تصميم وكيل A2A الذي يلف خادم MCP. رسم كل أداة MCP إلى مهارة A2A. لاحظ التنازلات  ما هو الضموضة التي فقدت؟

5. اقرأ إعلان A2A v1.0 وتحدد الميزة الوحيدة التي لم يتم تنفيذها بعد من قبل أي إطار اعتبارا من أبريل 2026. (لمحة: يتعلق الأمر بتفويض المهام متعددة المكالمات).

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | "Agent-to-Agent protocol" | Open protocol for opaque agent collaboration |
| Agent Card | "`.well-known/agent.json`" | Published metadata describing an agent's skills and endpoint |
| Skill | "A callable unit" | A named operation the agent supports (analog to MCP tool) |
| Task | "Unit of delegation" | A work item with a lifecycle and final artifact |
| Message | "Task input" | Carries Parts (text, file, data) |
| Part | "Typed chunk" | `text` / `file` / `data` element of a message |
| Artifact | "Task output" | Named, typed output returned on completion |
| AP2 | "Agent Payments Protocol" | Signed Agent Cards extension for trust and payments |
| Opacity | "Black-box collaboration" | Called agent's internals are hidden from caller |
| Input-required | "Task pause" | Lifecycle state when the agent needs more info |

## المزيد من القراءة

- [a2a-protocol.org](https://a2a-protocol.org/latest/) مواصفات A2A القنونية
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) تنفيذات مرجعية و SDKs
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) يونيو 2025 تحويل الحكم
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade)خريطة الطريق وسرعة الشركاء
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) إصدارات إصدارات v1.0 وتوجيهات تراجعية
