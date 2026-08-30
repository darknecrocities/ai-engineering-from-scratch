# Neden çok ajan?

> Bir ajan duvara çarptı, akıllı hareket daha büyük bir ajan değil, daha fazla ajan.

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Öğrenme Hedefleri

- Tek ajanlı tavanı belirleyin (koneks aşırılığı, karışık uzmanlık, sıralı şişek boynuz) ve birden fazla ajanlılığa bölünmenin doğru hareket olduğunu açıklayın
- Orkestralama desenlerini karşılaştırın (pipeline, paralel fan-out, yönetici, hiyerarşik) ve verilen görev yapısı için doğru olanı seçin
- Açık rol sınırları, paylaşılan durum ve iletişim sözleşmesi ile çoklu ajanlı bir sistem tasarlayın
- Çoklu ajan karmaşıklığının (kenaklık, maliyet, hata işleminde zorluk) karşılaştırmalarını tek ajan basitliği ile analiz edin

## Sorun

14. aşamada tek bir ajan oluşturdun. İşliyor. Dosyaları okuyabilir, komutları çalıştırır, API'leri arayabilir ve sonuçları düşünüyor. Sonra gerçek bir kod tabanına yönlendiriyorsun: 200 dosya, üç dil, altyapıya bağlı testler ve kod yazmadan önce dış API'leri araştırmak için bir gereksinim.

Bu yüzden, bu iş, bir ajanın yapabileceği işten fazlasını yapar. Bu iş, bir ajanın yapabileceği işten fazlasını yapar. Bu iş, bir ajanın yapabileceği işten fazlasını yapar. Bu iş, bir ajanın yapabileceği işten fazlasını yapar. Bu iş, bir ajanın yapabileceği işten fazlasını yapar. Bu iş, bir ajanın yapabileceği işten fazlasını yapar. Bu iş, bir ajanın yapabildiği işten sonra, bir ajanın yapabildiği işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu işten sonra, bu, bu işten sonra, bu, bu, bu, bu, bu, bu, bu, bu, bu, bu, bir işten, bu, bu, bu, bir işten, bu, bu, bir işten, bu, bu, bu, bir işten, bu, bu, bu, bu, bu, bu, bu, bu, bu, bu, bu, bu, bir şey, bu, bu, bu, bu, bu, bu, bu, bu, bu, bu, bu, bu, bu, için, bu, bu, bu, bu, bu, bu, bu, bu, bu, bu, bu, bu, için, bu, bu, bu, bu, bu, bu, bu, bu, için, bu, bu, bu, bu, bu, bu, bu, bu, bu, bu, için, bu, bu, bu, bu, bu, bu, bu

Bu tek ajanlı tavan. Bir görev için her zaman vurulur.

- **More context than fits in one window**- 50 dosya okuyarak 200 bin tokeni geçiyor
- **Different expertise at different stages**- araştırma kod üretiminden farklı bir teşvik gerektirir
- **Work that can happen in parallel**- ...sadece aynı anda okuyabilirseniz neden üç dosyayı sıradan okuyun?

## Anlaşım

### Tek Ajanlı Tavan

Tek bir ajan bir döngü, bir bağlam penceresi, bir sistem uyarısı.

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

Üç şey kırılır:

1. **Context saturation**30. turda ajan 150 bin token dosya içeriği, komut çıkışı ve önceki akıl yürütme tüketti.

2. **Role confusion**- "Sen bir araştırmacı, kodlayıcı, inceleyicisin ve testçi" diyen bir sistem uyarısı yarı araştırma yapan, yarı kodlayan ve incelemeyi asla bitirmeyen bir ajan üretir.

3. **Sequential bottleneck**- ajan A dosyasını okuyor, sonra B dosyasını, sonra C dosyasını.

### Çok Ajanlı Çözüm

Her ajanın bir görevi, bir bağlam penceresi ve bu göreve uygun bir sistem uyarısı yapın:

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

Her ajanın:
- odaklanmış bir sistem sorgulaması ("Kodu inceleyicisiniz. Tek işiniz hata bulmak. ")
- Kendi bağlam penceresi (diğer ajanların çalışmaları tarafından kirlenmemiş)
- Açık bir giriş/çıktı sözleşmesi ( Araştırma notları alır, çıkış kodu alır)

### Bunu Yapacak Gerçek Sistemler

**Claude Code subagents**- Claude Code bir subagen doğururken .`Task`Bu, bir çocuk ajanı oluşturur ve bir görev belirlenir. Ebeveyn bağlamını temiz tutar. Çocuk odaklı bir iş yapar ve bir özet gönderir.

**Devin**- planlayıcı ajanı, kodlayıcı ajanı ve tarayıcı ajanı çalıştırır. Planlayıcı işi adımlara ayırır. kodlayıcı kod yazar. tarayıcı belgeleri araştırır. Her birinin ayrı bağlamı vardır.

**Multi-agent coding teams (SWE-bench)**- SWE-benç'teki en iyi performanslı sistemler, kod tabanını okuyan bir araştırmacı, düzeltmeyi tasarlayan bir planlayıcı ve uygulayan bir kodleyiciden yararlanır.

**ChatGPT Deep Research**- her biri farklı açılardan araştırma yapan, paralel olarak birden fazla arama ajanı üretir, sonra sonuçları sentezler.

### Spektrum

Çoklu ajan ikili değil, bir spektrum:

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

**Single agent**- Bir döngü, bir uyarı.

**Subagents**- bir ebeveyn çocuklarını odaklı alt görevler için doğurur. ebeveyn planı korur. çocuklar rapor verir. Claude Code böyle yapar.

**Pipeline**- ajanlar sırayla çalışırlar. A ajanın çıkışı A ajanın girişine dönüşür.

**Team**- ajanlar ortak mesaj otobüsü ile paralel olarak çalışırlar. her birinin bir rolü vardır. bir orkeströr koordinatör.

**Swarm**- ortak durumlu çok sayıda aynı veya neredeyse aynı ajan. sabit orkeströr yok. ajanlar sıradan iş alırlar. yüksek performanslı paralel görevler için iyi.

### Dört Çok Ajanlı Örnek

#### Şekil 1: Pipeline

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

Her ajan verileri dönüştürür ve öne aktarır.

#### Şekil 2: Fan Out / Fan In

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

Paralel ajanlar arasında çalışmayı bölün, sonra sonuçları birleştir.

#### Model 3: Orkestratör-işçi

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

Akıllı orkeströr ne yapacağını belirler, işçilere görevlendirir ve sonuçları sentezler.

#### Dörtüncü örnek: Arkadaşlar

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

Merkez orkeströrü yok, ajanlar birbiriyle iletişim kurar, kararlar etkileşimden kaynaklanır, hataları düzeltmek daha zor ama birçok ajan için ölçeklendirilir.

### Çoklu Ajanlar Ne Zaman Kullanılmamalı

Multi-agent karmaşıklığı artırır. ajanlar arasındaki her mesaj potansiyel bir başarısızlık noktasıdır. Debug "bir konuşmayı oku"dan "beş ajan arasındaki mesajları izlemek"e kadar gider.

**Stay single-agent when:**
- Görev bir bağlam penceresine (iş verisi tokenlerinin ~ 100k altında) uymaktadır.
- Farklı aşamalarda farklı sistem isteklerine ihtiyacınız yok .
- İletişim yeteri kadar hızlı .
- Görev yeterince basit ki bölmek değerden daha fazla genel maliyet ekler.

**The complexity cost:**
- Her ajan sınırı bir kayblı sıkıştırma adımıdır: A ajanının tüm bağlamı A ajan için bir mesaj olarak özetlenir
- Koordinasyon mantığı (kim ne yapar, ne zaman, hangi sırada) kendi hata kaynağıdır
- Gecikme artışları: N ajanlar N seri LLM çağrıları minimum anlamına gelir, ileri geri konuşmaları gerekiyorsa daha fazla
- Maliyet katılaştırıcıları: her ajan jetonları bağımsız olarak yakar

Basamak kural: Bir görev 20'den az araç çağrısı alır ve 100k tokene uyarsa, tek ajanlı tutun.

```figure
swarm-messages
```

## Yapın

### Adım 1: Aşırı yüklü tek bir ajan

Burada her şeyi yapmaya çalışan tek bir ajan var. Bu büyük bir sistem sorgulaması ve bir bağlam penceresi araştırma, kod ve incelemeleri tutan:

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

Bu yaklaşımdaki sorunlar:
- Konekst penceresi her aşamada büyüyor. Değerlendirme aşamasında araştırma notları ve kod ve önceden akıl yürütme içerir.
- Sistem uyarısı geneldir. Her aşama için ayarlanamaz.
- Hiçbir şey paralel olarak yürümüyor.

### İkinci Adım: Uzman Ajanlar

Şimdi paylaşıp her ajan bir iş alır:

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

Her uzmanın odaklı bir ipucu vardır. Her biri sadece ihtiyaç duyduğu girişlerle temiz bir bağlam penceresi alır.

### Üçüncü Adım: Mesajlar Göndererek Birlikte İşlem Yapın

Uzmanlara açık mesajla haber verin:

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

Her ajan sadece kendisine gönderilen mesajları alır.Kontext kirliliği yoktur. Araştırmacıların 50 bin belge okuyuşu asla değerlendirici'nin bağlamına girmez.

### Dördüncü adım: karşılaştır

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

Çoklu ajan sürümü daha fazla toplam token kullanır (üç ajan, üç ayrı LLM çağrısı), ancak her ajanın bağlamı temiz kalır.

## Kullan

Bu ders, ne zaman çoklu ajanlık yapılması gerektiği konusunda tekrar kullanılabilir bir ipucu üretir.`outputs/prompt-multi-agent-decision.md`- Evet .

## Egzersizler

1. Dördüncü bir uzman ekleyin: kodlayıcıdan kod alan ve inceleyiciden geri bildirimleri inceleyen ve ardından testler yazan bir "tester" ajanı
2. Tüzükleyi değiştirin böylece inceleyiciler bir inceleme döngüsü için kodlayıcıya geri bildirim gönderebilir (maksimum 2 tur)
3. Düzsel boru hattını bir fan-out'a dönüştürün: araştırmacıyı ve "gereklilik analizatörü" ajanını paralel olarak çalıştırın, sonra kodlamacıya geçmeden önce çıkışlarını birleştirin

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Swarm | "A hive mind of AI agents" | A set of peer agents with shared state and no fixed leader. Behavior emerges from local interactions. |
| Orchestrator | "The boss agent" | An agent whose tools include spawning and managing other agents. It plans and delegates but may not do the actual work. |
| Coordinator | "The traffic cop" | A non-agent component (often just code, not an LLM) that routes messages between agents based on rules. |
| Consensus | "The agents agree" | A protocol where multiple agents must reach agreement before proceeding. Used when conflicting outputs need resolution. |
| Emergent behavior | "The agents figured it out themselves" | System-level patterns that arise from agent interactions but were not explicitly programmed. Can be useful or harmful. |
| Fan-out / fan-in | "Map-reduce for agents" | Splitting a task across parallel agents (fan-out), then combining their results (fan-in). |
| Message passing | "Agents talk to each other" | The communication mechanism between agents: structured data sent from one agent to another, replacing shared context windows. |

## Daha Fazla Okumak

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977)- çoklu ajanlı modellerin araştırılması
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155)- Microsoft'un çok ajanlı konuşma çerçevesini
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code)- Claude Code'un görevle nasıl temsil ettiği
- [CrewAI documentation](https://docs.crewai.com/)- Rol tabanlı çoklu ajan çerçevesini
