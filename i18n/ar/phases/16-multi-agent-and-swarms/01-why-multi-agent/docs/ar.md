# لماذا "العامل المتعدد"؟

> عميل واحد يضرب جدار، الحركة الذكية ليست عميل أكبر، بل أكثر عملاء.

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## أهداف التعلم

- تحديد السقف المتعلق بالوكيل الواحد (تجاوز السياق، الخبرة المختلطة، ضغوط الزجاجة المتسلسلة) وشرح متى يكون الانقسام إلى عدة عوامل الخطوة الصحيحة
- مقارنة أنماط التنسيق (الأنبوب، المروحة المتوازية، المشرف، التسلسل) واختيار المناسب لنظام المهمة المعين
- تصميم نظام متعدد الوكلاء مع حدود واضحة للدور، والحالة المشتركة، وعقد الاتصالات
- تحليل التنازلات بين تعقيدات متعددة الوكلاء (التأخير والتكلفة وصعوبة التحليل) مقابل سهولة الوكيل الواحد

## المشكلة

لقد بنيت وكيل واحد في المرحلة 14، يعمل. يمكنه قراءة الملفات، تشغيل الأوامر، الاتصال بالمنصات التنفيذية، والحكم حول النتائج. ثم تقوم بتوجيهها إلى قاعدة رمزية حقيقية: 200 ملف، ثلاث لغات، اختبارات تعتمد على البنية التحتية، ومتطلب للبحث في المنصات التنفيذية الخارجية قبل كتابة الرمز.

الوكيل يختنق. ليس لأن ماجستير التدريس غبي، ولكن لأن المهمة تتجاوز ما يمكن لمركز واحد حلول التعامل معه. نافذة السياق يملأ بمحتوى الملفات. الوكيل ينسى ما قرأه قبل 40 مكالمة أداة. يحاول أن يكون باحثا، ومدمج، ومراجعا في وقت واحد، و يفعل كل ثلاثة بشكل سيء.

هذا هو السقف الذي يستخدم عامل واحد، تضربه كلما تطلب مهمة:

- **More context than fits in one window**- قراءة 50 ملف ينفجر أكثر من 200 ألف رمز
- **Different expertise at different stages**- البحث يتطلب تحفيز مختلف عن توليد الرموز
- **Work that can happen in parallel**لماذا تقرأ ثلاث ملفات متتالية عندما يمكنك قراءتها في وقت واحد؟

## المفهوم

### السقف الذي يعمل بمُعامل واحد

وكيل واحد هو حلقة واحدة، نافذة سياق واحدة، عرض نظام واحد. تخيل ذلك:

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

ثلاثة أشياء تكسرها:

1. **Context saturation**وبحلول الجولة الثلاثين، استهلك العميل 150 ألف رمز من محتويات الملفات، ومخرجات الأوامر، والحكمة السابقة، وتضيع التفاصيل الحرجة من الجولة الخامسة.

2. **Role confusion**- إشارة نظامية تقول "أنت باحث، مُبرم، مراجعة، ومختبر" تنتج وكيل يبحث نصف، نصف مُبرم، ولا ينهي مراجعة.

3. **Sequential bottleneck**العميل يقرأ الملف "أ"، ثم الملف "ب"، ثم الملف "ج"، ثلاث مكالمات متسلسلة لـ"مدرسة التدريس"

### الحل المتعدد الوكلاء

تقسّم العمل، أعط كل عميل وظيفة واحدة، نافذة سياقية واحدة، وطلب نظام واحد مصمم لهذا العمل:

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

كل عميل لديه:
- طلب نظام مركز ("أنت مراجع رمز. وظيفتك الوحيدة هي إيجاد الأخطاء".)
- نافذة سياقها الخاصة (غير ملوثة من عمل الوكلاء الآخرين)
- عقد واضح للدخول/الخروج (يتلقى ملاحظات البحث، ومدونة الخروقات)

### أنظمة حقيقية تقوم بهذا

**Claude Code subagents**- عندما يولد (كلود كود) " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "`Task`يخلق عميل طفلي مع مهمة محددة والآباء يبقون سياقها نظيفاً الطفل يقوم بعمل مركز يعيد ملخصاً

**Devin**يدير وكيل المخطط، وكيل المبرمج، وكيل المتصفح. ينفصل المخطط العمل إلى خطوات. الكودر يكتب الرمز. المتصفح يبحث في الوثائق. كل منها له سياق منفصل.

**Multi-agent coding teams (SWE-bench)**- أنظمة ذات الأداء الأعلى على مقعد SWE تستخدم باحثًا يقرأ قاعدة البرمجيات، ومخططًا يصمم الإصلاح، ومبرمجًا ينفذها.

**ChatGPT Deep Research**- يخلق عدة عوامل بحث متوازية، كل منها يستكشف زاوية مختلفة، ثم يجمع النتائج.

### الطيف

الوكيل المتعدد ليس ثنائيًا إنه طيف:

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**Single agent**-حلقة واحدة، إرسال واحد، جيد للمهام البسيطة

**Subagents**والداً يُولد الأطفال لأعمال فرعية مركزة والآباء يحافظون على الخطة الأطفال يُبلاغون عن ذلك هذا ما يفعله كلود كود

**Pipeline**- العملاء يعملون بالتسلسل. إنتاج العميل A يصبح مدخل العميل B. جيد لأجراء العمل المرتبطة: البحث -> الرمز -> مراجعة -> اختبار.

**Team**-الوكلاء يعملون بالتوازي مع حافلة رسائل مشتركة لكل منهم دور، منسق ينسق، جيد عندما تكون هناك حاجة إلى مهارات مختلفة في نفس الوقت

**Swarm**لا يوجد مسجل ثابت، الوكلاء يحصلون على العمل من صف، جيد لمهام متوازية عالية التكامل.

### النماذج الأربعة متعددة الوكلاء

#### النمط الأول: خط أنابيب

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

كل وكيل يغير البيانات ويمررها إلى الأمام، سهل التفكير، الفشل في مرحلة واحدة يمنع البقية.

#### النمط الثاني: التشغيل / التشغيل

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

تقسيم العمل عبر وكلاء متوازين، ثم دمج النتائج. جيد للمهام التي تتحلل إلى مهام فرعية مستقلة.

#### النمط الثالث: عامل الموسيقى

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

الموسيقي الذكي يقرر ما يجب القيام به، ويمنح العمال، ويتولى النتائج. الموسيقي هو نفسه وكيل مع الأدوات للعمال التناسلية.

#### النمط الرابع: مجموعة من الأقران

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

لا يوجد مسجل مركزي، الوكلاء يتواصلون من ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذوي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذي ذ

### متى لا تستخدم العاملات المتعددة

الوكيل المتعدد يضيف التعقيد. كل رسالة بين الوكلاء هي نقطة فشل محتملة. إزالة الخطأ يذهب من "قراءة محادثة واحدة" إلى "تتبع الرسائل عبر خمسة وكلاء".

**Stay single-agent when:**
- تناسب المهمة في نافذة سياق واحدة (أقل من ~ 100k رموز من البيانات العاملة)
- لا تحتاج إلى إشارات مختلفة للمرحلة المختلفة
- الإعدام التسلسلي سريع بما فيه الكفاية
- المهمة بسيطة بما فيه الكفاية أن تقسيمها يضيف المزيد من التكاليف العامة من القيمة

**The complexity cost:**
- كل حدود العميل هي خطوة ضغط ضائعة: يتم تلخيص السياق الكامل للعميل A في رسالة للعميل B
- منطق التنسيق (من يفعل ماذا، متى، في أي ترتيب) هو مصدر الخاص به من الأخطاء
- زيادة التأخير: N العاملين يعني N سلسلة LLM الدعوات الحد الأدنى، أكثر إذا كانوا بحاجة إلى التحدث ذهابا وإيابا
- مضاعفات التكلفة: كل وكيل يحرق الرموز بشكل مستقل

قاعدة عامة: إذا كانت مهمة تستغرق أقل من 20 مكالمة أداة وتتناسب مع 100 ألف رمز، فبقي الأمر وكيل واحد.

```figure
swarm-messages
```

## بناءها

### الخطوة الأولى: العميل الوحيد المفرط

هنا وكيل واحد يحاول القيام بكل شيء. لديه طلب نظام واحد ضخم ونفذة سياق واحدة تحتوي على البحث، والرمز، والإستعراض:

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

مشاكل هذا النهج:
- نافذة السياق تنمو مع كل مرحلة، وبحلول مرحلة المراجعة، تحتوي على ملاحظات بحثية وكود واعتبارات سابقة.
- إن طلب النظام عام لا يمكن ضبطه لكل مرحلة
- لا شيء يسير بالتوازي

### الخطوة الثانية: وكلاء متخصصون

الآن تقسّمها كل عميل يحصل على وظيفة واحدة

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

كل متخصص لديه طلب مركز. كل واحد يحصل على نافذة سياق نظيفة مع المدخلات التي يحتاجها فقط.

### الخطوة الثالثة: تنسيق من خلال الرسائل

إرسال رسالة معينة إلى المتخصصين

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

كل وكيل يتلقى فقط الرسائل الموجزة إليه لا تلوث السياق. 50 ألف رمز من الباحث من قراءة الوثائق لا تدخل أبداً سياق المراجع.

### الخطوة الرابعة: مقارنة

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

يستخدم إصدار متعدد الوكلاء أكثر من مجموع الرموز (ثلاث وكلاء، ثلاثة مكالمات LLM منفصلة) ولكن سياق كل وكيل يبقى نظيفاً. تحسن جودة كل مرحلة لأن طلب النظام متخصص.

## استخدمها

هذه الدروس تنتج طلب قابلة للاستعمال مرة أخرى لتحديد متى يجب أن تذهب إلى وكيل متعدد. انظر `outputs/prompt-multi-agent-decision.md`. . .

## التمارين

1. أضف خبير رابع: وكيل "مختبر" يتلقى الرمز من المبرمج ومراجعة ردود الفعل من المراجعة، ثم يكتب الاختبارات
2. تعديل خط الأنابيب حتى يتمكن المراجع من إرسال تعليقات مرة أخرى إلى المبرمج لدورة مراجعة (ماكس 2 جولات)
3. تحويل خط الأنابيب المتسلسل إلى مروحة: تشغيل الباحث ووكيل "تحليل المتطلبات" بالتوازي ، ثم دمج نتائجهم قبل الانتقال إلى المبرمج

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Swarm | "A hive mind of AI agents" | A set of peer agents with shared state and no fixed leader. Behavior emerges from local interactions. |
| Orchestrator | "The boss agent" | An agent whose tools include spawning and managing other agents. It plans and delegates but may not do the actual work. |
| Coordinator | "The traffic cop" | A non-agent component (often just code, not an LLM) that routes messages between agents based on rules. |
| Consensus | "The agents agree" | A protocol where multiple agents must reach agreement before proceeding. Used when conflicting outputs need resolution. |
| Emergent behavior | "The agents figured it out themselves" | System-level patterns that arise from agent interactions but were not explicitly programmed. Can be useful or harmful. |
| Fan-out / fan-in | "Map-reduce for agents" | Splitting a task across parallel agents (fan-out), then combining their results (fan-in). |
| Message passing | "Agents talk to each other" | The communication mechanism between agents: structured data sent from one agent to another, replacing shared context windows. |

## المزيد من القراءة

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977)- مسح أنماط متعددة الوكلاء
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155)- إطار محادثة Microsoft متعدد الوكلاء
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code)- كيف يوفّر كلود كود مع مهمة
- [CrewAI documentation](https://docs.crewai.com/)- إطار متعدد الوكلاء القائم على الأدوار
