# بناء خادم MCP: بيثون بلا حالة و TypeScript

> لا يتذكر خادم MCP الحديث ضغط اليد. فإنه يؤكد بيانات الأساسية على كل طلب، ويقوم بتشغيل عامل واحد، ويرد نتيجة واحدة من المكتوبات.

**Type:** Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 13, Lesson 06
**Time:** ~85 minutes

## أهداف التعلم

- تنفيذ إلزاميا `server/discover`لـ MCP `2026-07-28`. . .
- تأكيد إصدار البروتوكول وإمكانيات العميل على كل طلب.
- تعرض الأدوات والموارد والطلبات بتنظيم القائمة المحددة.
- العودة`resultType`، هوية الخادم، وتلميحات التخزين على النتائج الصحيحة.
- خدمة نفس العقد بدون ولاية على استوديو محدد خط جديد في Python و TypeScript.

## المشكلة

خادم يخزن قدرات العميل بعد الرسالة الأولى سهلة البناء وصعبة التشغيل. قد تخدم نفس العملية العملاء المتسلسلين. قد يصل طلب بعيد إلى عامل مختلف. يمكن أن تسرب إعلان قدرة قديمة السلوك عبر حدود التفويض.

المفوضية`2026-07-28`يحل بروتوكول جزء من هذه المشكلة عن طريق جعل كل طلب وصف الذاتي. لا يزال تطبيقك يمكن أن تحتفظ بقوائم دائمة، وظائف، أو التعاملات الحالة الصريحة. ما لا يمكن أن تحتفظ به هو حالة بروتوكول مخفية التي تغير كيفية فك طلب لاحقا.

هذه الدروس تبني خادم ملاحظات مرتين. تستخدم إصدارات Python و TypeScript فقط مكتباتها القياسية للبرنامج الأساسي. كلاهما يعرض نفس الطرق ويفرض نفس العقد السلكي.

## المفهوم

### حلقة الرسائل الحديثة

```text
read one JSON-RPC line
parse the envelope
if it is a notification, do not respond
validate params._meta for this request
route by method
wrap success with resultType and serverInfo
write one JSON-RPC response line
forget request-scoped metadata
```

ثلاثة قواعد استوديو لا تزال مهمة:

- اكتب فقط رسائل JSON-RPC إلى stdout أرسل التشخيص إلى stderr
- حدد الرسائل مع خط جديد وفرز كل رد
- اخرج فوراً عندما يصل القائم إلى مركز الطيران

عمر العملية هو حياة النقل. إنه ليس جلسة MCP الحديثة.

### التحقق من التحقق من التحقق من التحقق من التحقق

كل طلب يجب أن يكون:

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

الحقول الأولى والثانية مطلوبة.`clientInfo`يوصى به. تأكيد شكل هوية موجود، ولكن لا تعامله كتحقق.

إذا كان الإصدار غير مدعوم، عودة الرمز `-32022`مع`requested`و`supported`. المعلومات المفقودة عن الطلب غير صالحة`-32602`لا تملأ أبداً الحقول المفقودة من مكالمة سابقة

### الاكتشاف المفروض

يجب أن تنفيذ الخوادم الحديثة`server/discover`. نتيجة اكتشاف كاملة تشمل الإصدارات الحديثة المدعومة والإمكانيات والإرشادات الاختيارية ، وتلميحات التخزين الآلي ، و هوية الخادم في النتيجة `_meta`:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

لا يفتح "ديسكوير" الخادم، قد يتصل العميل`tools/list`دون أن نسمي اكتشافاً لأن`tools/list`يحتوي بالفعل على نفس البيانات المعدنية للمطالبة.

### الأدوات

`tools/list`يعود قائمة تحديدية من وصفات الأدوات. يحسن ترتيب الاستجابة من تخزين الاستجابة ويبقي سياق النموذج مستقرا. النتيجة تتطلب أيضا `ttlMs`و`cacheScope`. . .

`tools/call`يعيد كتلة المحتوى و`isError`. استخدم خطأ JSON-RPC عندما تكون غلاف البروتوكول أو معايير الطريقة غير صالحة. استخدم `isError: true`عندما يتم تشغيل دعوة أداة صالحة ولكن الأداة نفسها تفشل.

تعليقات الأدوات لا تزال إشارات، وليس إجراءات:

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

يجب على المضيف استخدامها للتأكيد والعرض. يجب على الخادم لا يزال ينفذ الإذن الحقيقي.

### الموارد

`resources/list`يعود وصفات URI مستقرة. `resources/read`يعود المحتوى المكتوب. كليهما قابلة للتخزين في `2026-07-28`، لذا كلاهما يتضمن`ttlMs`و`cacheScope`. . .

استخدام`cacheScope: "private"`لا يجوز لإعادة استخدام الاحتياطي المشترك للردود الخاصة عبر سياقات الائتمان.

التغيير الحديث لا يستخدم `resources/subscribe`. العميل يفتح`subscriptions/listen`و الطلبات`resourceSubscriptions`أو فئات تغيير القائمة. الدروس 10 تبني هذا التدفق.

### الإشارات

`prompts/list`هو قابلة للتخفيض و تحديد.`prompts/get`يعطي عرضًا مسمّىً مع حجج. النتيجة المسمّمة للمساعدة كاملة، ولكنها ليست من القائمة القابلة للتخزين أو نتائج القراءة التي تتطلب إشارات التخزين.

### كل نتيجة ناجحة يتم كتابتها

تستخدم الأمثلة لفاحة واحدة لكل نجاح:

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

قائمة، قراءة، ومعالجة اكتشاف إضافة `ttlMs`بالإضافة`cacheScope`. مركزية هذه الملفات تمنع أحد المستخدمين من حذف صامتة حقل النتائج الحديثة.

### لا يوجد طلبات من الخادم

قد يرسل الخادم الحديث إشعارات تتعلق بطلب العميل ، أو إشعارات على خادم مفتوح `subscriptions/listen`لا يجب أن يرسل طلب JSON-RPC الخاص به.

عندما يحتاج المدير إلى أخذ العينات أو إثارة أو إدخال الجذور ، فإنه يعيد `input_required`النتيجة. يقوم العميل بتلبية طلبات المدخلات المضمنة ومحاولة الطريقة الأصلية مع معرف طلب جديد. يتناول الدروس 11 نمط طلبات متعددة رحلة.

### التوافق الصريح مع التراث

يمكن أن ينفذ خادم عصر مزدوج أيضاً`2025-11-25`يضغط على فرع إرث منفصل بشكل واضح. يختار السلوك الحديث عندما يتطلب الحديث`_meta`الحقول موجودة وتتراث السلوك عندما يتلقى `initialize`. . .

لا تضع`2026-07-28`الطلب من خلال المسار التمسك اليدوي التراثي. لا تملأ الحديث `resultType`النتائج التبني المترتبة على التبني المترتبة على النتائج المترتبة على التبني المترتبة على التبني المترتبة على النتائج المترتبة على التبني المترتبة على النتائج المترتبة على التبني المترتبة على النتائج المترتبة على التبني المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتائج المترتبة على النتبة

```figure
t3-dispatch-loop
```

## استخدمها

تشغيل عرض التجربة المحدودة لخادم Python واختبارات:

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

تشغيل منفذ TypeScript مع متشغل TypeScript:

```bash
npx tsx main.ts --demo
```

الظهور يرسله`server/discover`يردّ كل طلب حديث بيانات أساسية. كل نجاح يتضمن هوية الخادم.

## أرسله

هذه الدروس تُسافر`outputs/skill-mcp-server-scaffolder.md`. إنّه ينتج خطة خادم حديثة مع عقد اكتشاف، وتصديق حسب الطلب، وقوائم تحديدية قابلة للتخزين، ومعدّل إرث معزول اختياري.

## التمارين

1. إزالة القدرات من طلب واحد وإثبات أن الخادم لا يستخدم إعلان الطلب السابق مرة أخرى.
2. عكس الـ`TOOLS`،`PROMPTS`تأكد من أن جميع نتائج القائمة تظل مستقرة
3. إضافة مدمرة `notes_delete`أداة و تتطلب فحص الإذن داخل المنفذ.`destructiveHint`كلمحة عن تجربة التأثير فقط
4. إضافة`resources/templates/list`مع`ttlMs`،`cacheScope`، و التنظيم المحدد
5. قم ببناء مُعدل إرث منفصل لـ `2025-11-25`إضافة اختبارات تثبت أن طلب حديث لا يدخل

## الشروط الرئيسية

| Term | Meaning |
|------|---------|
| Stateless server | Handles each request from its own metadata without protocol-session memory |
| `server/discover` | Mandatory modern method that advertises versions and capabilities |
| Complete result | Successful modern result with `resultType: "complete"` |
| Cacheable result | Discovery, list, or resource-read result with `ttlMs` and `cacheScope` |
| Deterministic list | Same logical registry produces the same item order |
| Server identity | Recommended `io.modelcontextprotocol/serverInfo` in result `_meta` |
| Tool error | Valid tool call that returns content with `isError: true` |
| Protocol error | Invalid JSON-RPC or MCP request returned through `error` |

## المزيد من القراءة

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
