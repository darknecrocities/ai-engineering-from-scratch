# RLHF: نموذج الجائزة + PPO

> يدرس SFT النموذج اتباع التعليمات. لكنه لا يعلم النموذج أي رد أفضل. يمكن أن تختلف إجابتان صحيحتان عن الناحية الجهامية ودقيقة من الناحية الفعلية بشكل كبير في المفيدية. RLHF هو كيفية تشفير الحكم البشري في سلوك النموذج. هذا هو ما يجعل كلود مفيدًا وGPT مهذبًا.

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10, Lesson 06 (Instruction Tuning / SFT)
**Time:** ~90 minutes

## أهداف التعلم

- بناء نموذج مكافأة يسجل جودة الاستجابة من أزواج تفضيلات البشر (المتختر مقابل الرفض)
- تنفيذ حلقة تدريبية لـ PPO التي تحسن سياسة نموذج اللغة مقابل نموذج الجائزة مع عقوبة KL
- شرح لماذا يطلب RLHF ثلاثة نماذج (SFT، مكافأة، سياسة) وكيف يمنع قيود KL اختراق مكافأة
- تقييم تأثير RLHF عن طريق مقارنة جودة الاستجابة قبل وبعد تحسين الاختيارات

## المشكلة

اسأل نموذج "شرح الحوسبة الكمية" و قد ينتج:

**Response A:**"الحوسبة الكمومية تستخدم الكوبيتات التي يمكن أن تكون موجودة في التنظيم، مما يعني أنها يمكن أن تكون 0، 1 أو كليهما في وقت واحد. وهذا يسمح للكمبيوترات الكمومية لعملية حسابات معينة بشكل متسارع بشكل متكامل من الكمبيوتر الكلاسيكية. وتشمل الخوارزميات الرئيسية خوارزمية شور لتحديد العدد الكبير وخوارزمية غروفر للبحث في قواعد البيانات غير المرتبة".

**Response B:**"الحوسبة الكمومية هي نوع من الحوسبة التي تستخدم الظواهر الميكانيكية الكمومية. تم اقتراحها لأول مرة في الثمانينيات. اقترح ريتشارد فيينمان أن النظم الكمومية يمكن محاكيتها بواسطة أجهزة الكموم. النمو كبير منذ ذلك الحين. تعمل العديد من الشركات الآن على أجهزة الكموم. تقدم IBM وGoogle وغيرها. دعمت Google في عام 2019 على سيادة الكموم".

كلتا الردتين صحيحتان في الواقع. كلاهما صائب من الناحية الجهازية. كلاهما يتبع التعليمات. ولكن الرد A هو أفضل بوضوح. هو أكثر حدة، أكثر إعلاما، وأفضل هيكلة. إنسان يختار A في كل مرة.

لا يمكن لـ SFT أن يلتقط هذا التمييز. فإنه يدرب النموذج على الاستجابات "الصحيحة"، لكنه ليس لديه آلية للقول "هذا الاستجابة أفضل من ذلك". فإنه يعامل كل مثال على التدريب على قدم المساواة. إذا ظهرت كل من A و B في مجموعة بيانات SFT، فإن النموذج سيتعلم من كليهما على قدم المساواة.

(الـ (ريل هف) يحلّ هذا إنه يدرب نموذج مكافأة للتنبؤ بأي رد فعل يفضل الإنسان، ثم يستخدم تلك الإشارة للمكافأة لدفع نموذج اللغة نحو نتائج عالية الجودة. استخدم InstructGPT (مسبق ChatGPT) RLHF لتحسين فعالية GPT-3، وصادقيتها، وعدم تضررها بشكل كبير. يفضل المقيمون الداخليون لـ OpenAI نتائج InstructGPT على نتائج GPT-3 في 85% من الأحيان، على الرغم من أن InstructGPT أصغر 135 مرة (1.3B مقابل 175B).

## المفهوم

### المراحل الثلاثة

(الـ (ر.إل.ه.إف) ليست دورة تدريبية واحدة إنها خط أنابيب من ثلاث مراحل متسلسلة، كل منها يبني على المرحلة السابقة

**Stage 1: SFT.**تدريب نموذج أساسي على أزواج التعليمات والردود (الدرس 06) ، وهذا يعطيك نموذج يمكنه اتباع التعليمات ولكن لا يعرف أي ردود فعل أفضل من الآخرين.

**Stage 2: Reward Model.**جمع بيانات تفضيلات الإنسان: أظهر للملاحنين ردود فعل اثنتين على نفس الاستعلامات وتسأل "أيهما أفضل؟" قم بتدريب نموذج للتنبؤ بهذه التفضيلات. يأخذ نموذج الجائزة (الاستعلامات، الاستجابة) كمدخول ويخرج نتيجة متدنية.

**Stage 3: PPO.**استخدم نموذج الجائزة لتوليد إشارة تدريبية لنموذج اللغة. يقوم نموذج اللغة بتوليد الردود ، ويمتد نموذج الجائزة على ذلك ، ويقوم PPO بتحديث نموذج اللغة لإنتاج ردود فعل ذات درجة أعلى. عقوبة تباين KL تمنع نموذج اللغة من الابتعاد عن نقطة التفتيش SFT.

```mermaid
graph TD
    subgraph Stage1["Stage 1: SFT"]
        B["Base Model"] --> S["SFT Model"]
        D["Instruction Data\n(27K examples)"] --> S
    end

    subgraph Stage2["Stage 2: Reward Model"]
        S --> |"Generate responses"| P["Preference Pairs\n(prompt, winner, loser)"]
        H["Human Annotators"] --> P
        P --> R["Reward Model\nR(prompt, response) → score"]
    end

    subgraph Stage3["Stage 3: PPO"]
        S --> |"Initialize policy"| PI["Policy Model\n(being optimized)"]
        S --> |"Freeze as reference"| REF["Reference Model\n(frozen SFT)"]
        PI --> |"Generate"| RESP["Response"]
        RESP --> R
        R --> |"Reward signal"| PPO["PPO Update"]
        REF --> |"KL penalty"| PPO
        PPO --> |"Update"| PI
    end

    style S fill:#1a1a2e,stroke:#51cf66,color:#fff
    style R fill:#1a1a2e,stroke:#e94560,color:#fff
    style PI fill:#1a1a2e,stroke:#0f3460,color:#fff
    style REF fill:#1a1a2e,stroke:#0f3460,color:#fff
    style PPO fill:#1a1a2e,stroke:#e94560,color:#fff
```

### نموذج الجائزة

نموذج الجائزة هو نموذج لغوي يتم إعادة تطبيقه كجهاز تسجيل. خذ نموذج SFT ، واستبدل رأس نموذج اللغة (الذي يخرج توزيعًا على المفردات) برأس متزايد (الذي يخرج رقمًا واحدًا). الهندسة المعمارية متطابقة حتى الطبقة النهائية.

المدخل: طلب متواصل مع رد. الخروج: نقطة مكافأة واسكال واحدة.

بيانات التدريب هي أزواج تفضيلات الإنسان. لكل طلب، يرى الملاحظون ردوداً وتختارون أفضل واحد. هذا يخلق ثلاثة أزواج للتدريب: (الطلب، الرد المفضل، الرد الرفض).

وظيفة الخسارة تستخدم نموذج برادلي تيري من تفضيلات الزوجية:

```
loss = -log(sigmoid(reward(preferred) - reward(rejected)))
```

هذه هي المعادلة الرئيسية`sigmoid(reward(A) - reward(B))`يعطي احتمال تفضيل الاستجابة A على الاستجابة B. الضرر يدفع نموذج الجائزة لتعيين درجة أعلى للرد المفضل.

لماذا المقارنات بالتزاوج بدلاً من النتائج المطلقة؟ لأن البشر فظيعون في تخصيص النتائج الجودة المطلقة ("هل هذا الاستجابة 7.3 أو 7.5 من 10؟") ولكن جيدين جداً في المقارنات النسبية ("هل A أفضل من B؟"). يقوم نموذج برادلي تيري بتحويل المقارنات النسبية إلى نظام مستمر للاستجابة المطلقة.

**InstructGPT numbers:**جمع "أوبن آي آي" 33 ألف زوج مقارنة من 40 مقاولًا. كل مقارنة استغرقت حوالي 5 دقائق. هذا يبلغ 2750 ساعة من العمل البشري لبيانات تدريب نموذج الجائزة.

### PPO: تحسين السياسة القريبة

PPO هو خوارزمية تعليمي للتعزيز. في RLHF، "البيئة" هي نموذج الجائزة، و "الوكيل" هو نموذج اللغة، و "الفعال" هو توليد رمز.

الهدف:

```
maximize: E[R(prompt, response)] - beta * KL(policy || reference)
```

يضغط المفهوم الأول على النموذج لتوليد استجابات مكافأة عالية. يمنع المفهوم الثاني (عقوبة الانحراف من KL) النموذج من الانحراف بعيداً جداً عن نقطة التفتيش SFT.

لماذا عقوبة KL؟ بدونها، يجد النموذج حلول مهينة. يتم تدريب نموذج الجائزة على مجموعة محدودة من البيانات من تفضيلات الإنسان. لديها نقاط عمياء. نموذج اللغة سوف يستغل تلك البقع العمياء - العثور على نتائج عالية في نموذج الجائزة ولكن في الواقع غير منطقية. أمثلة كلاسيكية:

- تكرار "أنا مفيدة جداً وغير ضارة!" يسجل درجات عالية على نماذج مكافأة المساعدة / الخفية
- إنتاج ردود فعل صريحة، بصوت رسمي ولكن فارغة تتطابق مع نمط "جودة عالية"
- استغلال عبارات محددة حدثت أن تتصل مع مكافأة عالية في بيانات التدريب

عقوبة كيل تقول: يمكنك التحسين، ولكن لا يمكنك أن تصبح نموذج مختلف تماما. إبقى بالقرب من النسخة SFT، الذي كان معقولا بالفعل.

**InstructGPT numbers:**استخدم تدريب PPO lr = 1.5e-5 ، معدل KL beta = 0.02 ، و 256K حلقات (أزواج الاستجابة السريعة) ، و 4 دورات PPO لكل دفعة. استغرق خط أنابيب RLHF بأكمله عدة أيام على مجموعة من GPUs.

```mermaid
graph LR
    subgraph PPO["PPO Training Loop"]
        direction TB
        PROMPT["Sample prompt\nfrom dataset"] --> GEN["Policy generates\nresponse"]
        GEN --> SCORE["Reward model\nscores response"]
        GEN --> KL["Compute KL divergence\nvs reference model"]
        SCORE --> OBJ["Objective:\nreward - beta * KL"]
        KL --> OBJ
        OBJ --> UPDATE["PPO gradient update\n(clipped surrogate loss)"]
        UPDATE --> |"repeat"| PROMPT
    end

    style PROMPT fill:#1a1a2e,stroke:#0f3460,color:#fff
    style SCORE fill:#1a1a2e,stroke:#51cf66,color:#fff
    style KL fill:#1a1a2e,stroke:#e94560,color:#fff
    style OBJ fill:#1a1a2e,stroke:#e94560,color:#fff
```

### هدف المنظمة التنفيذية

يستخدم PPO "هدف بديل مقطوع" لمنع التحديثات الكبيرة المفرطة. يتم تقليص النسبة بين السياسة الجديدة واحتمالات السياسة القديمة إلى النطاق [1 - epsilon ، 1 + epsilon ] ، حيث يكون epsilon عادة 0.2.

```
ratio = pi_new(action | state) / pi_old(action | state)
clipped_ratio = clip(ratio, 1 - epsilon, 1 + epsilon)
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

تقدر وظيفة الميزة كم هو أفضل من الرد الحالي مقارنة بالجودة المتوقعة. في RLHF:

```
advantage = reward(prompt, response) - baseline
```

غالباً ما تكون الخط الأساسي متوسط الجائزة على الاستجابات الأخيرة. يعني الميزة الإيجابية أن الاستجابة كانت أفضل من المتوسط؛ وميزة السلبية يعني أنها كانت أسوأ. يزيد PPO من احتمالات الاستجابات فوق المتوسط ويقلل من احتمالات أقل من المتوسط.

يمنع التقطيع تحديثات كارثية. إذا حصل استجابة واحدة على مكافأة عالية بشكل غير عادي، فإن النسبة غير المقطوعة قد تكون كبيرة جداً، مما يسبب في تحول النموذج بشكل كبير نحو هذا الاستجابة. يحدد التقطيع التحديث، مما يحافظ على استقرار التدريب.

### مكافأة التسلل

الجانب المظلم من RLHF. نموذج اللغة هو التحسين ضد نموذج الجائزة، وهو وكيل غير كامل لتحسب تفضيلات الإنسان. كما نموذج اللغة يحصل على أفضل في تعظيم الجائزة، فإنه يبدأ استغلال نقاط ضعف نموذج الجائزة.

أساليب الفشل الشائعة:

| Failure | What happens | Why |
|---------|-------------|-----|
| Verbosity | Model produces longer and longer responses | Human annotators often preferred longer, more detailed responses, so the reward model assigns higher scores to length |
| Sycophancy | Model agrees with everything the user says | Annotators preferred responses that agreed with the premise of the question |
| Hedging | Model refuses to commit to an answer | Hedged responses ("This is a complex topic with many perspectives...") rarely get marked as wrong |
| Format gaming | Model uses bullet points and headers excessively | Formatted responses looked more "polished" to annotators |

استراتيجيات التخفيف: عقوبة KL أقوى (تمنع النموذج من الابتعاد بعيدا بما فيه الكفاية لاستغلال نقاط الضعف) ، تدريب نموذج الجائزة على أمثلة معارضة (أوضاع الفشل المعروفة) ، واستخدام نماذج الجائزة المتعددة مع بنيات مختلفة (أكثر صعوبة في اختراق كل ذلك في وقت واحد).

### خطوط أنابيب RLHF الحقيقية

| Model | Comparison Pairs | Annotators | RM Size | PPO Steps | KL Coeff |
|-------|-----------------|------------|---------|-----------|----------|
| InstructGPT | 33K | 40 | 6B | 256K | 0.02 |
| Llama 2 Chat | ~1M | undisclosed | 70B | undisclosed | 0.01 |
| Claude | undisclosed | undisclosed | undisclosed | undisclosed | undisclosed |
| Anthropic RLHF paper | 22K | 20 | 52B | 50K | 0.001 |

دراسة 2022 من أنثروبيك تدرب نموذج مكافأة 52B على 22000 مقارنة. نموذج مكافأة أكبر تنتج إشارات أكثر موثوقية، مما يجعل تدريب PPO أكثر استقرارا. استخدام نموذج مكافأة صغير لتدريب نموذج لغة كبيرة هو مخاطر - نموذج مكافأة ليس لديه قدرة كافية لالتقاط اللونات من الاستجابات الجيدة مقابل السيئة.

```figure
rlhf-pipeline
```

## بناءها

### الخطوة الأولى: بيانات تفضيل اصطناعية

في الإنتاج، يقوم الملاحظون البشريون بإنشاء بيانات الاختيارات. سنقوم بإنشاء أزواج اصطناعية حيث يكون الاستجابة "التي تفضل" أفضل بشكل موضوعي (أكثر حدة، أكثر دقة، أكثر مفيداً).

```python
import numpy as np

PREFERENCE_DATA = [
    {
        "prompt": "What is the capital of France?",
        "preferred": "The capital of France is Paris.",
        "rejected": "France is a country in Europe. It has many cities. The capital is Paris. Paris is known for the Eiffel Tower.",
    },
    {
        "prompt": "Explain gravity in one sentence.",
        "preferred": "Gravity is the force that attracts objects with mass toward each other.",
        "rejected": "Gravity is something that makes things fall down when you drop them.",
    },
    {
        "prompt": "What is 15 times 7?",
        "preferred": "15 times 7 is 105.",
        "rejected": "Let me think about this. 15 times 7. Well, 10 times 7 is 70, and 5 times 7 is 35, so the answer might be around 105.",
    },
    {
        "prompt": "Name three programming languages.",
        "preferred": "Python, Rust, and TypeScript.",
        "rejected": "There are many programming languages. Some popular ones include various languages like Python and others.",
    },
    {
        "prompt": "What year did World War II end?",
        "preferred": "World War II ended in 1945.",
        "rejected": "World War II was a major global conflict. It involved many countries. The war ended in the mid-1940s, specifically in 1945.",
    },
    {
        "prompt": "Define machine learning.",
        "preferred": "Machine learning is a field where algorithms learn patterns from data to make predictions without being explicitly programmed.",
        "rejected": "Machine learning is a type of AI. AI stands for artificial intelligence. Machine learning uses data to learn.",
    },
]
```

الردود المفضلة هي موجزة ومباشرة. الردود التي رفضت تظهر أنماط فشل شائعة: التغطية غير الضرورية، التحوط، التفسير الزائد، وعدم الدقة. هذا هو بالضبط نوع التمييز الذي لا يمكن أن تستقطبه SFT ولكن RLHF يمكن.

### الخطوة الثانية: تعديل المثالية

يستخدم نموذج الجائزة مجدداً بنية المحول من GPT الصغيرة ، لكنه يحل محل رأس الخروج بحجم المفردات بمقاس مستوى واحد.

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "04-pre-training-mini-gpt", "code"))
from main import MiniGPT, LayerNorm, Embedding, TransformerBlock


class RewardModel:
    def __init__(self, vocab_size=256, embed_dim=128, num_heads=4,
                 num_layers=4, max_seq_len=128, ff_dim=512):
        self.embedding = Embedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(embed_dim)
        self.reward_head = np.random.randn(embed_dim) * 0.02

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)

        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x, mask)
        x = self.ln_f.forward(x)

        last_hidden = x[:, -1, :]
        reward = last_hidden @ self.reward_head

        return reward
```

يأخذ نموذج الجائزة الحالة الخفية في موقف الرمز * الأخير* ويمبنىها إلى رمز مقياس. لماذا الرمز الأخير؟ لأن قناع الاهتمام السببية يعني أن الموقف الأخير قد حضر كل الرمز السابق. لديه التمثيل الأكثر اكتمالا لسلسلة كاملة (السرعة، الاستجابة).

### الخطوة الثالثة: خسارة برادلي تيري

تدريب نموذج الجائزة على أزواج التفضيل باستخدام خسر برادلي تيري أزواجية.

```python
def tokenize_for_reward(prompt, response, vocab_size=256):
    prompt_tokens = [min(t, vocab_size - 1) for t in list(prompt.encode("utf-8"))]
    response_tokens = [min(t, vocab_size - 1) for t in list(response.encode("utf-8"))]
    return prompt_tokens + [0] + response_tokens


def sigmoid(x):
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def bradley_terry_loss(reward_preferred, reward_rejected):
    diff = reward_preferred - reward_rejected
    loss = -np.log(sigmoid(diff) + 1e-8)
    return loss


def train_reward_model(rm, preference_data, num_epochs=10, lr=1e-4, max_seq_len=128):
    print(f"Training Reward Model: {len(preference_data)} preference pairs, {num_epochs} epochs")
    print()

    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        num_pairs = 0

        indices = np.random.permutation(len(preference_data))

        for idx in indices:
            pair = preference_data[idx]

            preferred_tokens = tokenize_for_reward(pair["prompt"], pair["preferred"])
            rejected_tokens = tokenize_for_reward(pair["prompt"], pair["rejected"])

            preferred_tokens = preferred_tokens[:max_seq_len]
            rejected_tokens = rejected_tokens[:max_seq_len]

            preferred_ids = np.array(preferred_tokens).reshape(1, -1)
            rejected_ids = np.array(rejected_tokens).reshape(1, -1)

            r_preferred = rm.forward(preferred_ids)[0]
            r_rejected = rm.forward(rejected_ids)[0]

            loss = bradley_terry_loss(r_preferred, r_rejected)

            if r_preferred > r_rejected:
                epoch_correct += 1

            diff = r_preferred - r_rejected
            grad = sigmoid(diff) - 1.0

            rm.reward_head -= lr * grad * rm.ln_f.forward(
                rm.embedding.forward(preferred_ids)
            )[:, -1, :].flatten()

            epoch_loss += loss
            num_pairs += 1

        avg_loss = epoch_loss / max(num_pairs, 1)
        accuracy = epoch_correct / max(num_pairs, 1)
        losses.append(avg_loss)
        accuracies.append(accuracy)

        if epoch % 2 == 0:
            print(f"  Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1%}")

    return rm, losses, accuracies
```

مقياس الدقة بسيط: ما هو الجزء من أزواج التفضيلات التي يصنفها نموذج الجائزة بشكل صحيح؟ النموذج العشوائي يحصل على 50٪ يجب أن يتجاوز نموذج مكافأة مدرب جيدًا على البيانات النظيفة 70٪. نموذج مكافأة InstructGPT حقق دقة حوالي 72٪ على المقارنات المتبقية، والتي تبدو منخفضة ولكن في الواقع جيدة - العديد من أزواج الاختيارات غير واضحة حتى بالنسبة للبشر (اتفاق بين الملاحنين كان حوالي 73٪).

### الخطوة الرابعة: حلقة PPO مبسطة

إن التنفيذ الكامل لـ PPO معقد. هذا التنفيذ يحتوي على الآلية الأساسية: إنتاج الردود، وتسجيلها، وحساب الميزة، وتحديث السياسة مع عقوبة KL.

```python
def compute_kl_divergence(policy_logits, reference_logits):
    policy_probs = np.exp(policy_logits - policy_logits.max(axis=-1, keepdims=True))
    policy_probs = policy_probs / policy_probs.sum(axis=-1, keepdims=True)
    policy_probs = np.clip(policy_probs, 1e-10, 1.0)

    ref_probs = np.exp(reference_logits - reference_logits.max(axis=-1, keepdims=True))
    ref_probs = ref_probs / ref_probs.sum(axis=-1, keepdims=True)
    ref_probs = np.clip(ref_probs, 1e-10, 1.0)

    kl = np.sum(policy_probs * np.log(policy_probs / ref_probs), axis=-1)
    return kl.mean()


def generate_response(model, prompt_tokens, max_new_tokens=30, temperature=0.8, max_seq_len=128):
    tokens = list(prompt_tokens)

    for _ in range(max_new_tokens):
        context = np.array(tokens[-max_seq_len:]).reshape(1, -1)
        logits = model.forward(context)
        next_logits = logits[0, -1, :]

        next_logits = next_logits / max(temperature, 1e-8)
        probs = np.exp(next_logits - next_logits.max())
        probs = probs / probs.sum()
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / probs.sum()

        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(int(next_token))

    return tokens


def copy_model_weights(source, target):
    target.embedding.token_embed = source.embedding.token_embed.copy()
    target.embedding.pos_embed = source.embedding.pos_embed.copy()
    target.ln_f.gamma = source.ln_f.gamma.copy()
    target.ln_f.beta = source.ln_f.beta.copy()
    for s_block, t_block in zip(source.blocks, target.blocks):
        t_block.attn.W_q = s_block.attn.W_q.copy()
        t_block.attn.W_k = s_block.attn.W_k.copy()
        t_block.attn.W_v = s_block.attn.W_v.copy()
        t_block.attn.W_out = s_block.attn.W_out.copy()
        t_block.ffn.W1 = s_block.ffn.W1.copy()
        t_block.ffn.W2 = s_block.ffn.W2.copy()
        t_block.ffn.b1 = s_block.ffn.b1.copy()
        t_block.ffn.b2 = s_block.ffn.b2.copy()
        t_block.ln1.gamma = s_block.ln1.gamma.copy()
        t_block.ln1.beta = s_block.ln1.beta.copy()
        t_block.ln2.gamma = s_block.ln2.gamma.copy()
        t_block.ln2.beta = s_block.ln2.beta.copy()


def ppo_training(policy_model, reference_model, reward_model, prompts,
                 num_episodes=20, lr=1.5e-5, kl_coeff=0.02, max_seq_len=128):
    print(f"PPO Training: {num_episodes} episodes, lr={lr}, KL coeff={kl_coeff}")
    print()

    rewards_history = []
    kl_history = []

    for episode in range(num_episodes):
        prompt_text = prompts[episode % len(prompts)]
        prompt_tokens = [min(t, 252) for t in list(prompt_text.encode("utf-8"))]

        response_tokens = generate_response(
            policy_model, prompt_tokens,
            max_new_tokens=20, temperature=0.8, max_seq_len=max_seq_len
        )

        response_ids = np.array(response_tokens[:max_seq_len]).reshape(1, -1)
        reward = reward_model.forward(response_ids)[0]

        policy_logits = policy_model.forward(response_ids)
        ref_logits = reference_model.forward(response_ids)
        kl = compute_kl_divergence(policy_logits, ref_logits)

        total_reward = reward - kl_coeff * kl

        rewards_history.append(float(reward))
        kl_history.append(float(kl))

        for block in policy_model.blocks:
            update_scale = lr * total_reward
            block.ffn.W1 += update_scale * np.random.randn(*block.ffn.W1.shape) * 0.01
            block.ffn.W2 += update_scale * np.random.randn(*block.ffn.W2.shape) * 0.01

        if episode % 5 == 0:
            avg_reward = np.mean(rewards_history[-5:]) if rewards_history else 0
            avg_kl = np.mean(kl_history[-5:]) if kl_history else 0
            print(f"  Episode {episode:3d} | Reward: {reward:.4f} | KL: {kl:.4f} | "
                  f"Avg Reward: {avg_reward:.4f}")

    return policy_model, rewards_history, kl_history
```

الحلقة الأساسية: (1) أخذ عينات من طلب، (2) توليد استجابة، (3) تسجيله مع نموذج المكافأة، (4) حساب تباين KL مقابل المرجع المجمد، (5) حساب المكافأة المعدلة (المكافأة ناقص عقوبة KL) ، (6) تحديث السياسة. تنمو عقوبة KL مع تباين السياسة من المرجع، مما يمنع تلقائيًا اختراق المكافأة.

### الخطوة 5: مقارنة النتائج

بعد RLHF، يجب أن تكون ردود النموذج السياسي أعلى من ردود النموذج الأصلي SFT على نموذج الجائزة.

```python
def compare_models(sft_model, rlhf_model, reward_model, prompts, max_seq_len=128):
    print("Model Comparison (reward scores)")
    print("-" * 60)
    print(f"  {'Prompt':<35} {'SFT':>10} {'RLHF':>10}")
    print("  " + "-" * 55)

    sft_total = 0.0
    rlhf_total = 0.0

    for prompt in prompts:
        prompt_tokens = [min(t, 252) for t in list(prompt.encode("utf-8"))]

        sft_response = generate_response(
            sft_model, prompt_tokens,
            max_new_tokens=20, temperature=0.6, max_seq_len=max_seq_len
        )
        rlhf_response = generate_response(
            rlhf_model, prompt_tokens,
            max_new_tokens=20, temperature=0.6, max_seq_len=max_seq_len
        )

        sft_ids = np.array(sft_response[:max_seq_len]).reshape(1, -1)
        rlhf_ids = np.array(rlhf_response[:max_seq_len]).reshape(1, -1)

        sft_reward = reward_model.forward(sft_ids)[0]
        rlhf_reward = reward_model.forward(rlhf_ids)[0]

        sft_total += sft_reward
        rlhf_total += rlhf_reward

        truncated_prompt = prompt[:33] + ".." if len(prompt) > 35 else prompt
        print(f"  {truncated_prompt:<35} {sft_reward:>10.4f} {rlhf_reward:>10.4f}")

    n = len(prompts)
    print("  " + "-" * 55)
    print(f"  {'Average':<35} {sft_total/n:>10.4f} {rlhf_total/n:>10.4f}")

    return sft_total / n, rlhf_total / n
```

## استخدمها

### التجربة الكاملة لخط أنابيب RLHF

```python
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("RLHF PIPELINE: REWARD MODEL + PPO")
    print("=" * 70)
    print()

    print("STAGE 1: SFT Model (from Lesson 06)")
    print("-" * 40)
    sft_model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    print(f"  Parameters: {sft_model.count_parameters():,}")
    print()

    print("STAGE 2: Train Reward Model")
    print("-" * 40)
    rm = RewardModel(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )

    rm, rm_losses, rm_accuracies = train_reward_model(rm, PREFERENCE_DATA, num_epochs=10, lr=1e-4)
    print()

    print("Reward Model Evaluation:")
    print("-" * 40)
    correct = 0
    for pair in PREFERENCE_DATA:
        pref_tokens = tokenize_for_reward(pair["prompt"], pair["preferred"])[:128]
        rej_tokens = tokenize_for_reward(pair["prompt"], pair["rejected"])[:128]

        r_pref = rm.forward(np.array(pref_tokens).reshape(1, -1))[0]
        r_rej = rm.forward(np.array(rej_tokens).reshape(1, -1))[0]

        if r_pref > r_rej:
            correct += 1
        print(f"  Preferred: {r_pref:+.4f} | Rejected: {r_rej:+.4f} | {'Correct' if r_pref > r_rej else 'Wrong'}")

    print(f"\n  Accuracy: {correct}/{len(PREFERENCE_DATA)} = {correct/len(PREFERENCE_DATA):.1%}")
    print()

    print("STAGE 3: PPO Training")
    print("-" * 40)

    policy_model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    reference_model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )

    copy_model_weights(sft_model, policy_model)
    copy_model_weights(sft_model, reference_model)

    train_prompts = [pair["prompt"] for pair in PREFERENCE_DATA]

    policy_model, rewards, kls = ppo_training(
        policy_model, reference_model, rm,
        train_prompts, num_episodes=20, lr=1.5e-5, kl_coeff=0.02
    )
    print()

    print("=" * 70)
    print("COMPARISON: SFT vs RLHF")
    print("=" * 70)
    print()

    eval_prompts = [
        "What is the capital of France?",
        "Explain gravity.",
        "Name three programming languages.",
    ]

    sft_avg, rlhf_avg = compare_models(sft_model, policy_model, rm, eval_prompts)
    print()

    print("=" * 70)
    print("KL DIVERGENCE ANALYSIS")
    print("=" * 70)
    print()

    if kls:
        print(f"  Initial KL: {kls[0]:.4f}")
        print(f"  Final KL:   {kls[-1]:.4f}")
        print(f"  Max KL:     {max(kls):.4f}")
        kl_threshold = 0.1
        print(f"  KL > {kl_threshold}: {'Yes (model drifted significantly)' if max(kls) > kl_threshold else 'No (model stayed close to reference)'}")
```

## أرسله

هذا الدرس يُنتج`outputs/prompt-reward-model-designer.md`-- تحذير لتصميم خطوط تدريب نموذج الجائزة. بالنظر إلى السلوك المستهدف (المساعدة، القدرة على التشفير، السلامة) ، فإنه ينتج بروتوكول جمع البيانات، وإرشادات الملاحظين، ومعايير تقييم نموذج الجائزة.

## التمارين

1. تعديل نموذج الجائزة لاستخدام متوسط جميع الحالات الخفية بدلاً من الموقف الأخير فقط. مقارنة الدقة. يمنح نهج المجموعة المتوسط كل رمز وزن متساو ، في حين يعتمد نهج الموقف الأخير على الاهتمام السببي للمعلومات الإجمالية. اختبار على 6 أزواج الاختيارات وتقرير أي نهج يحصل على درجات أعلى من الدقة.

2. تنفيذ تصنيف نموذج الجائزة. بعد التدريب، قم بتشغيل جميع أزواج التفضيلات عبر نموذج الجائزة وحساب: (أ) متوسط الجائزة للردود المفضلة، (ب) متوسط الجائزة للردود المرفوضة، (ج) الهامش (المفضلة - المرفوضة). يجب أن يكون لدى نموذج مقياس جيد هامش واضح. ثم أضف 4 أزواج تفضيلات جديدة وتحقق ما إذا كانت الهامش تحتوي على بيانات غير مرئية.

3. قم بتحاكي اختراق المكافآت. قم بإنشاء نموذج مكافأة يعطي درجات عالية للردود الطويلة (الرد = len(رد) / 100). قم بتشغيل PPO مع نموذج المكافأة المعيب ولاحظ نموذج السياسة الذي يولد نتائج متكررة طويلة بشكل متزايد. ثم أضف عقوبة KL من 0.1 وظهر أنه يمنع السلوك المتدهور.

4. تنفيذ مكافأة متعددة الأهداف. تدريب نموذجين من المكافآت - واحد للمساعدة والآخر للمكافأة. مزجهم على أنه R = 0.7 * R_helpful + 0.3 * R_concise. أظهر أن الهدف المشترك ينتج استجابات مفيدة ومكافأة، وتجنب فخ الفصائحية من مكافأة مفيدة واحدة.

5. مقارنة معايير KL المختلفة. تشغيل PPO مع beta=0.001 (منخفض جداً، اختراق المكافآت) ، beta=0.02 (معياري) ، و beta=0.5 (عالي جداً، لا تعلم). رسم منحنى المكافأة و منحنى KL لكل منهما. يجب أن يظهر beta=0.02 تشغيل تحسن ثابت في المكافأة مع KL المحدود.

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| RLHF | "Training with human feedback" | Reinforcement Learning from Human Feedback: a three-stage pipeline (SFT, reward model, PPO) that optimizes language model outputs using human preference signals |
| Reward model | "A model that scores responses" | A transformer with a scalar output head, trained on pairwise human preferences using the Bradley-Terry loss |
| Bradley-Terry | "The comparison model" | A probabilistic model where P(A > B) = sigmoid(score(A) - score(B)), converting pairwise preferences into a consistent scoring function |
| PPO | "The RL algorithm" | Proximal Policy Optimization: updates the policy to maximize reward while clipping the update magnitude to prevent instability |
| KL divergence | "How different two distributions are" | A measure of the difference between the policy model's token distribution and the reference model's -- used as a penalty to prevent reward hacking |
| KL penalty | "The leash on the model" | Beta * KL(policy \|\| reference) subtracted from the reward signal -- prevents the policy from diverging too far from the SFT checkpoint |
| Reward hacking | "Gaming the reward" | When the policy finds degenerate high-reward outputs by exploiting weaknesses in the reward model instead of genuinely improving |
| Preference pair | "Which is better, A or B?" | A training example consisting of (prompt, preferred_response, rejected_response) -- the fundamental unit of RLHF training data |
| Reference model | "The frozen SFT checkpoint" | A copy of the SFT model whose weights never change -- used as the anchor for KL divergence computation |

## المزيد من القراءة

- [Ouyang et al., 2022 -- "Training language models to follow instructions with human feedback" (InstructGPT)](https://arxiv.org/abs/2203.02155)-- الورقة التي جعلت RLHF عملية لنماذج اللغة الكبيرة
- [Schulman et al., 2017 -- "Proximal Policy Optimization Algorithms"](https://arxiv.org/abs/1707.06347)-- ورقة PPO الأصلية من OpenAI
- [Bai et al., 2022 -- "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback"](https://arxiv.org/abs/2204.05862)-- ورقة "أنثروبيك" "التي تضم تحليلًا مفصلًا لـ "الإنكشاف المكافئ" وعقوبات "كيل"
- [Stiennon et al., 2020 -- "Learning to summarize with human feedback"](https://arxiv.org/abs/2009.01325)-- RLHF تطبيق على التجميع، والتي تظهر أن نماذج الجائزة يمكن أن تستقطب حكمات نوعية مختلفة
- [Christiano et al., 2017 -- "Deep reinforcement learning from human preferences"](https://arxiv.org/abs/1706.03741)-- العمل الأساسي على تعلم وظائف مكافأة من مقارنات البشر
