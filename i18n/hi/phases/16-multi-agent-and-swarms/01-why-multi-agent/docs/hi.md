# क्यों मल्टी-एजेंट?

> एक एजेंट दीवार पर धक्का मारता है। स्मार्ट कदम एक बड़ा एजेंट नहीं है - यह अधिक एजेंट है।

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## सीखने के लक्ष्य

- एकल एजेंट की सीमा की पहचान करें (संदर्भ अधिशेष, मिश्रित विशेषज्ञता, अनुक्रमिक बोतल गला) और समझाएं कि कई एजेंटों में विभाजन सही कदम कब है
- ऑर्केस्ट्रेशन पैटर्न (पाइपलाइन, समानांतर फैन-आउट, पर्यवेक्षक, पदानुक्रमिक) की तुलना करें और किसी दिए गए कार्य संरचना के लिए सही एक का चयन करें
- स्पष्ट भूमिका सीमाओं, साझा स्थिति और संचार अनुबंध के साथ एक बहु-एजेंट प्रणाली का डिजाइन करें
- बहु-एजेंट जटिलता (लैटेंसी, लागत, डिबगिंग कठिनाई) के मुकाबले एकल-एजेंट सादगी के व्यापारों का विश्लेषण करें

## समस्या

आप चरण 14 में एक एजेंट बनाया है. यह काम करता है. यह फ़ाइलें पढ़ सकता है, कमांड चला सकता है, एपीआई कॉल कर सकते हैं, और परिणामों के बारे में तर्क. फिर आप इसे एक वास्तविक कोडबेस पर इंगित करते हैंः 200 फ़ाइलें, तीन भाषाओं, बुनियादी ढांचे पर निर्भर परीक्षण, और कोड लिखने से पहले बाहरी एपीआई की जांच करने की आवश्यकता।

एजेंट डूब जाता है. यह इसलिए नहीं है क्योंकि एलएलएम मूर्ख है, बल्कि इसलिए कि कार्य एक एजेंट लूप से अधिक है। संदर्भ विंडो फ़ाइल सामग्री से भर जाती है। एजेंट भूल जाता है कि उसने 40 टूल कॉल से पहले क्या पढ़ा था। यह एक ही समय में एक शोधकर्ता, एक कोडर और एक समीक्षक होने की कोशिश करता है, और तीनों को खराब करता है।

यह एकल एजेंट की छत है. आप इसे हर बार जब एक कार्य की आवश्यकता होती हैः

- **More context than fits in one window**- 50 फ़ाइलें पढ़ने 200k टोकन से अधिक उड़ा
- **Different expertise at different stages**- अनुसंधान को कोड जनरेशन से अलग प्रेरणा की आवश्यकता होती है
- **Work that can happen in parallel**- आप तीन फ़ाइलों को एक साथ पढ़ सकते हैं जब आप एक साथ तीन फ़ाइलों को पढ़ने के लिए क्यों?

## अवधारणा

### एकल एजेंट छत

एक एकल एजेंट एक लूप, एक संदर्भ विंडो, एक सिस्टम प्रॉम्प्ट है। इसे कल्पना करेंः

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

तीन चीजें टूटती हैंः

1. **Context saturation**टर्न 30 तक एजेंट ने फ़ाइल सामग्री, कमांड आउटपुट और पूर्व तर्क के 150k टोकन का उपभोग किया है।

2. **Role confusion**- एक प्रणाली प्रोंपट जो कहता है "आप एक शोधकर्ता, कोडर, समीक्षक और परीक्षक हैं" एक एजेंट का उत्पादन करता है जो आधा शोध करता है, आधा कोड, और कभी समीक्षा समाप्त नहीं करता है।

3. **Sequential bottleneck**- एजेंट फ़ाइल A पढ़ता है, फिर फ़ाइल B, फिर फ़ाइल C. तीन सीरियल LLM कॉल. तीन सीरियल उपकरण निष्पादन. कोई समानांतर.

### बहु-एजेंट समाधान

प्रत्येक एजेंट को एक काम, एक संदर्भ विंडो और उस काम के लिए एक सिस्टम संकेत देंः

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

प्रत्येक एजेंट के पास हैः
- एक केंद्रित प्रणाली प्रलोभन ("आप एक कोड रिवीजर हैं। आपका एकमात्र काम बग ढूंढना है।")
- अपनी संदर्भ खिड़की (अन्य एजेंटों के काम से प्रदूषित नहीं)
- एक स्पष्ट इनपुट/आउटपुट अनुबंध (अनुसंधान नोट प्राप्त करता है, आउटपुट कोड)

### वास्तविक प्रणालियाँ जो ऐसा करती हैं

**Claude Code subagents**- जब क्लाउड कोड एक उप-बॉगेंट पैदा करता है`Task`माता पिता अपने संदर्भ को साफ रखता है। बच्चा केंद्रित काम करता है और एक सारांश देता है।

**Devin**- एक योजनाकार एजेंट, एक कोडर एजेंट, और एक ब्राउज़र एजेंट चलाता है. योजनाकार चरणों में काम को तोड़ता है. कोडर कोड लिखता है. ब्राउज़र प्रलेखन की जांच करता है. प्रत्येक अलग संदर्भ है.

**Multi-agent coding teams (SWE-bench)**- SWE-बेंच पर शीर्ष प्रदर्शन वाले सिस्टम एक शोधकर्ता का उपयोग करते हैं जो कोडबेस को पढ़ता है, एक योजनाकार जो फिक्स को डिजाइन करता है, और एक कोडर जो इसे लागू करता है।

**ChatGPT Deep Research**- समानांतर में कई खोज एजेंट उत्पन्न करता है, प्रत्येक एक अलग कोण की खोज करता है, फिर परिणामों को संश्लेषित करता है।

### स्पेक्ट्रम

मल्टी-एजेंट बाइनरी नहीं है। यह एक स्पेक्ट्रम हैः

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

**Single agent**सरल कार्यों के लिए अच्छा है।

**Subagents**एक माता-पिता ध्यान केंद्रित उप-कार्य के लिए बच्चों को पैदा करता है। माता-पिता योजना बनाए रखता है। बच्चे रिपोर्ट वापस। यह क्लाउड कोड करता है।

**Pipeline**एजेंट ए के आउटपुट एजेंट बी के इनपुट बन जाते हैं। चरणबद्ध कार्यप्रवाहों के लिए अच्छा हैः अनुसंधान -> कोड -> समीक्षा -> परीक्षण।

**Team**- एजेंट एक साझा संदेश बस के साथ समानांतर चल रहे हैं. प्रत्येक एक भूमिका है. एक संग्राहक समन्वयक. अच्छा जब एक ही समय में विभिन्न कौशल की जरूरत है.

**Swarm**- कई समान या लगभग समान एजेंटों के साथ साझा राज्य. कोई निश्चित ऑर्केस्ट्रेटर नहीं. एजेंटों को एक कतार से काम लेने. उच्च-प्रोफ्यूट समानांतर कार्यों के लिए अच्छा.

### चार बहु-एजेंट पैटर्न

#### पैटर्न 1: पाइपलाइन

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

प्रत्येक एजेंट डेटा को बदल देता है और उसे आगे बढ़ाता है, तर्क करने के लिए सरल। एक चरण में विफलता बाकी को ब्लॉक करती है।

#### पैटर्न 2: फैन आउट / फैन इन

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

समानांतर एजेंटों के बीच काम को विभाजित करें, फिर परिणामों को मिलाएं। स्वतंत्र उप-कार्य में विघटित होने वाले कार्यों के लिए अच्छा।

#### नमूना 3: ऑर्केस्ट्रेटर-वर्कर

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

एक स्मार्ट ऑर्केस्ट्रेटर तय करता है कि क्या करना है, कार्यकर्ताओं को सौंपता है, और परिणामों को संश्लेषित करता है। ऑर्केस्ट्रेटर स्वयं कार्यकर्ताओं को कुपोषण करने के लिए उपकरण के साथ एक एजेंट है।

#### पैटर्न 4: समकक्षों का झुंड

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

कोई केंद्रीय ऑर्केस्ट्रेटर नहीं है, एजेंट एक-दूसरे से संवाद करते हैं, निर्णय बातचीत से निकलते हैं, डिबग करना कठिन है, लेकिन कई एजेंटों के लिए स्केल।

### मल्टी एजेंट का उपयोग कब नहीं करना चाहिए

मल्टी-एजेंट जटिलता जोड़ता है। एजेंटों के बीच प्रत्येक संदेश एक संभावित विफलता बिंदु है। डिबगिंग "एक बातचीत पढ़ें" से "पांच एजेंटों के बीच संदेशों का पता लगाने" तक जाता है।

**Stay single-agent when:**
- कार्य एक संदर्भ विंडो में फिट बैठता है (काम डेटा के ~ 100k टोकन के तहत)
- आपको विभिन्न चरणों के लिए अलग-अलग सिस्टम संकेतों की आवश्यकता नहीं है
- क्रमिक निष्पादन काफी तेजी से है
- यह कार्य इतना सरल है कि इसे विभाजित करने से मूल्य से अधिक ओवरहेड होता है

**The complexity cost:**
- प्रत्येक एजेंट सीमा एक घाटा संपीड़न कदम हैः एजेंट ए का पूरा संदर्भ एजेंट बी के लिए एक संदेश में सारांशित हो जाता है
- समन्वय तर्क (किसे क्या करता है, कब, किस क्रम में) बग का अपना स्रोत है
- लटेंसी बढ़ जाती हैः N एजेंट का मतलब है N सीरियल LLM कॉल न्यूनतम, अधिक अगर उन्हें आगे और पीछे बात करने की जरूरत है
- लागत गुणा: प्रत्येक एजेंट स्वतंत्र रूप से टोकन जलाता है

अंगूठे का नियमः यदि किसी कार्य में 20 से कम उपकरण कॉल होते हैं और 100k टोकन में फिट होते हैं, तो इसे एकल एजेंट रखें।

```figure
swarm-messages
```

## इसे बनाओ

### चरण 1: अतिभारित एकल एजेंट

यहाँ एक ही एजेंट सब कुछ करने की कोशिश कर रहा है. यह एक विशाल प्रणाली प्रॉम्प्ट और एक संदर्भ विंडो है अनुसंधान, कोड, और समीक्षाओं को पकड़ने के लिए हैः

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

इस दृष्टिकोण के साथ समस्याएंः
- संदर्भ विंडो प्रत्येक चरण के साथ बढ़ जाती है। समीक्षा चरण द्वारा, इसमें शोध नोट्स और कोड और पूर्व तर्क शामिल हैं।
- सिस्टम प्रॉम्प्ट सामान्य है. इसे प्रत्येक चरण के लिए ट्यून नहीं किया जा सकता है।
- समानांतर में कुछ भी नहीं चलता है।

### चरण 2: विशेषज्ञ एजेंट

अब इसे विभाजित करें. प्रत्येक एजेंट एक नौकरी प्राप्त करता हैः

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

प्रत्येक विशेषज्ञ के पास एक केंद्रित संकेत होता है। प्रत्येक को केवल आवश्यक इनपुट के साथ एक साफ संदर्भ विंडो मिलती है।

### चरण 3: संदेशों के माध्यम से समन्वय करें

विशेषज्ञों को एक साथ वायर करें स्पष्ट संदेश पारित करने के साथः

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

प्रत्येक एजेंट को केवल संदेश प्राप्त होते हैं जो उसे संबोधित किए जाते हैं। कोई संदर्भ प्रदूषण नहीं। शोधकर्ता के 50 हजार दस्तावेज पढ़ने के टोकन कभी भी समीक्षक के संदर्भ में प्रवेश नहीं करते हैं।

### चरण 4: तुलना करें

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

मल्टी-एजेंट संस्करण में अधिक कुल टोकन (तीन एजेंट, तीन अलग-अलग एलएलएम कॉल) का उपयोग किया जाता है, लेकिन प्रत्येक एजेंट का संदर्भ साफ रहता है। प्रत्येक चरण की गुणवत्ता में सुधार होता है क्योंकि सिस्टम प्रॉम्प्ट विशेष है।

## इसका प्रयोग करें

यह सबक बहु-एजेंट में जाने के समय का निर्णय लेने के लिए एक पुनः प्रयोज्य संकेत देता है।`outputs/prompt-multi-agent-decision.md`. .

## व्यायाम

1. एक चौथा विशेषज्ञ जोड़ेंः एक "टेस्टर" एजेंट जो कोडर से कोड प्राप्त करता है और समीक्षाकर्ता से प्रतिक्रिया की समीक्षा करता है, फिर परीक्षण लिखता है
2. पाइपलाइन को संशोधित करें ताकि समीक्षक एक संशोधन लूप के लिए कोडर को प्रतिक्रिया भेज सके (अधिकतम 2 राउंड)
3. पंक्तिबद्ध पाइपलाइन को एक फैन-आउट में परिवर्तित करेंः शोधकर्ता और "आवश्यकता विश्लेषक" एजेंट को समानांतर में चलाएं, फिर कोडर पर जाने से पहले उनके आउटपुट को मिलाएं

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Swarm | "A hive mind of AI agents" | A set of peer agents with shared state and no fixed leader. Behavior emerges from local interactions. |
| Orchestrator | "The boss agent" | An agent whose tools include spawning and managing other agents. It plans and delegates but may not do the actual work. |
| Coordinator | "The traffic cop" | A non-agent component (often just code, not an LLM) that routes messages between agents based on rules. |
| Consensus | "The agents agree" | A protocol where multiple agents must reach agreement before proceeding. Used when conflicting outputs need resolution. |
| Emergent behavior | "The agents figured it out themselves" | System-level patterns that arise from agent interactions but were not explicitly programmed. Can be useful or harmful. |
| Fan-out / fan-in | "Map-reduce for agents" | Splitting a task across parallel agents (fan-out), then combining their results (fan-in). |
| Message passing | "Agents talk to each other" | The communication mechanism between agents: structured data sent from one agent to another, replacing shared context windows. |

## आगे पढ़ना

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977)- बहु-एजेंट पैटर्न का सर्वेक्षण
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155)- माइक्रोसॉफ्ट के मल्टी एजेंट वार्तालाप ढांचे
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code)- कैसे क्लाउड कोड कार्य के साथ प्रतिनिधि
- [CrewAI documentation](https://docs.crewai.com/)- भूमिका आधारित बहु एजेंट ढांचा
