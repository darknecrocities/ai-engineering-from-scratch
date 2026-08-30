# التنسيق الدقيق مع LoRA & QLoRA

> التنسيق الكامل لنموذج 7B يتطلب 56 جيجابايت من VRAM. ليس لديك ذلك. ولا معظم الشركات تفعل ذلك. LoRA يسمح لك بتنسيق النموذج نفسه في 6 جيجابايت عن طريق تدريب أقل من 1% من المعلمات. هذا ليس تعزيزا - انها تتطابق مع جودة التنسيق الكامل في معظم المهام. النظام الإيكولوجي الكامل مفتوح المصدر التنسيق الكامل يعمل على هذه الحيلة واحدة.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 06 (Instruction Tuning / SFT)
**Time:** ~75 minutes
**Related:**المرحلة 10 تغطي حلقات SFT / DPO من الصفر. هذه الدروس تربط تلك في مجموعات أدوات PEFT 2026 (PEFT ، TRL ، Unsloth ، Axolotl ، LLaMA-Factory).

## أهداف التعلم

- تنفيذ LoRA عن طريق حقن مصفوفات المعدلات منخفضة الصف (A و B) في طبقات الاهتمام الخاصة بالنموذج المُدرب مسبقاً
- حساب وفورات المعلمات من LoRA مقابل التنظيم الدقيق الكامل: ترتيب r مع d_مثل الأبعاد القطارات 2*r*d المعلمات بدلا من d^2
- ضبط النموذج باستخدام QLoRA (4 بت قاعدة كمية + مكيّفات LoRA) لتناسب مع ذاكرة GPU المستهلك
- دمج وزنات LoRA مرة أخرى في النموذج الأساسي لتنفيذ وتقارن سرعة الاستنتاج مع ودون مُعدات

## المشكلة

لديك نموذج أساسي. إلاما 3 8 ب. تريد أن يجيب على تذاكر دعم العملاء بصوت شركتك. SFT هو الحل. ولكن SFT لديها مشكلة تكلفة.

تحديثات التنسيق الدقيق الكاملة لكل پیرامتر في النموذج. Llama 3 8B لديها 8 مليارات پیرامتر. في fp16 ، كل پیرامتر يتطلب 2 بايت. هذا هو 16 جيجا غاي فقط لتحميل الوزن. أثناء التدريب ، تحتاج أيضًا إلى تراجع (16 جيجا غاي) ، وحالات التحسين لـ آدم (32 جيجا غاي لتحرك + تغير) ، وتفعيلات. إجمالي: حوالي 56 جيجا غاي لـ VRAM لنموذج واحد من 8B.

A100 80GB بالكاد يمكن أن تستوعب هذا.$3-4/hour on cloud providers. Training for 3 epochs on 50,000 examples takes 6-10 hours. That's $30-40 لكل تجربة، إنجز 10 تجارب لتصل إلى المعايير المضادة بشكل صحيح و إنفقت 400 دولار قبل نشر أي شيء

إذا قمت بتقييم هذا إلى Llama 3 70B، ستصبح الأرقام سخيفة، 140 جيجابايت للأوزان وحدها، ستحتاج إلى مجموعة، 100 دولار+ لكل تجربة.

هناك مشكلة أعمق أيضاً. التنسيق الكامل يغير كل وزن في النموذج. إذا قمت بتنسيق كل بيانات دعم العملاء، قد تدهور القدرات العامة للنموذج. هذا يسمى النسيان الكارثي. النموذج يتحسن في مهمتك وأسوأ في كل شيء آخر.

تحتاج إلى طريقة تدرب أقل ملامح، تستخدم أقل ذاكرة، ولا تدمر المعرفة الموجودة في النموذج.

## المفهوم

### لورا: التكيف منخفض الرتب

نشر إدوارد هو وزملاؤه في مايكروسوفت LoRA في يونيو 2021. نظرة ورقة: تحديثات الوزن أثناء ضبط الدقة لديها رتبة داخلية منخفضة. لا تحتاج إلى تحديث جميع المعلمات 16.7 مليون في ماتريكس الوزن 4096x4096. يمكن التقاط المعلومات المفيدة في التحديث بواسطة ماتريكس من رتبة 16 أو 32.

هذه هي الرياضيات، طبقة خطية قياسية تحسب:

```
y = Wx
```

حيث W هو d_out x d_in المصفوفة. بالنسبة لمبادرة 4096x4096 الاهتمام، هذا هو 16،777،216 ملامح.

لورا تجميد W و يضيف تدمير منخفض الدرجة:

```
y = Wx + BAx
```

حيث B هو (d_out x r) و A هو (r x d_in). صفوف r أصغر بكثير من d -- عادة 8, 16 أو 32.

ل r=16 على طبقة 4096x4096:
- المعلمات الأصلية: 4096 × 4096 = 16،777،216
- معايير لورا: (4096 × 16) + (16 × 4096) = 65536 + 65536 = 131 072
- الخصم: 131,072 / 16,777,216 = 0.78%

أنت تدرب 0.78% من المعلمات وتحصل على 95-100% من الجودة.

```mermaid
graph LR
    X["Input x"] --> W["Frozen W (d x d)"]
    X --> A["A (r x d)"]
    A --> B["B (d x r)"]
    W --> Plus["+ (merge)"]
    B --> Plus
    Plus --> Y["Output y"]

    style W fill:#1a1a2e,stroke:#e94560,color:#fff
    style A fill:#0f3460,stroke:#16213e,color:#fff
    style B fill:#0f3460,stroke:#16213e,color:#fff
```

A يتم تشغيله مع غوسيان عشوائي. B يتم تشغيله إلى الصفر. وهذا يعني أن مساهمة LoRA تبدأ من الصفر -- يبدأ النموذج التدريب من سلوكها الأصلي وتتعلم تدريجيا التكيف.

### عامل الحجم: ألفا

LoRA يقدم عامل مقياس ألفا الذي يتحكم في مدى تأثير التحديث المنخفض على الخروج:

```
y = Wx + (alpha / r) * BAx
```

عندما يكون ألفا = r، فإن التوسع هو 1x. عندما يكون ألفا = 2r (المتباعدة المشتركة) ، فإن التوسع هو 2x. هذه المعلمة المفرطة تحكم معدل التعلم من مسار LoRA بشكل مستقل عن معدل التعلم الأساسي.

الإرشادات العملية:
- الف = 2 * رتبة هو اتفاقية مجتمعية شائعة (الورقة الأصلية المستخدمة الف = رتبة في معظم التجارب)
- الفا = رتبة يعطي 1x مقياس، المحافظة ولكن مستقرة
- الفا العليا تعني تحديثات أكبر لكل خطوة، والتي يمكن أن تسريع التقارب أو تسبب عدم الاستقرار

### أين تطبيق LoRA

المحول لديه العديد من الطبقات الخطية. لا تحتاج إلى إضافة لورا إلى جميعها. الورق الأصلي اختبر مزيجات مختلفة:

| Target Layers | Trainable Params (7B) | Quality |
|--------------|----------------------|---------|
| q_proj only | 4.7M | Good |
| q_proj + v_proj | 9.4M | Better |
| q_proj + k_proj + v_proj + o_proj | 18.9M | Best for attention |
| All linear (attention + MLP) | 37.7M | Marginal gain, 2x params |

نقطة الحلوة لمعظم المهام: q_proj + v_proj. هذا يستهدف استفسار وتقديرات القيمة في الانتباه الذاتي ، والتي تحكم ما يشتري النموذج وما المعلومات التي يستخرجها. إضافة طبقات MLP تساعد في المهام المعقدة مثل توليد الشفرة ولكن تضاعف عدد المعايير لخفض العائدات على المهام البسيطة.

### اختيار الرتب

الدرجة r تحكم التعبير من التكيف:

| Rank | Trainable Params (per layer) | Best For |
|------|---------------------------|----------|
| 4 | 32,768 | Simple classification, sentiment |
| 8 | 65,536 | Single-domain Q&A, summarization |
| 16 | 131,072 | Multi-domain tasks, instruction following |
| 32 | 262,144 | Complex reasoning, code generation |
| 64 | 524,288 | Diminishing returns for most tasks |
| 128 | 1,048,576 | Rarely justified |

أظهرت هوو وآخرون أن r=4 يحتوي بالفعل على معظم التكيف للمهام البسيطة. r=8 و r=16 هي الخيارات الأكثر شيوعا في الممارسة. تجاوز r=64 نادرا ما يحسن الجودة ويبدأ في فقدان ميزة الذاكرة للورا.

### QLoRA: 4-bit كمية + LoRA

نشر تيم ديتمرز وزملاؤه في جامعة واشنطن QLoRA في مايو 2023. فكرة: تحديد النموذج الأساسي المجمد إلى دقة 4-بيت، ثم توصيل مكيّفات LoRA في fp16 في الأعلى.

هذا يغير معادلة الذاكرة بشكل كبير:

| Method | Weight Memory (7B) | Training Memory (7B) | GPU Required |
|--------|-------------------|---------------------|-------------|
| Full fine-tune (fp16) | 14GB | ~56GB | 1x A100 80GB |
| LoRA (fp16 base) | 14GB | ~18GB | 1x A100 40GB |
| QLoRA (4-bit base) | 3.5GB | ~6GB | 1x RTX 3090 24GB |

تقوم QLoRA بثلاث مساهمات فنية:

**NF4 (Normal Float 4-bit)**: نوع بيانات جديد مصمم خصيصًا لوزن الشبكات العصبية. يتابع وزن الشبكات العصبية توزيعًا طبيعيًا تقريبًا. يضع NF4 مستويات تعريفها الكمية 16 في كوانتيلات توزيع طبيعي قياسي. هذا مثاليًا من الناحية النظرية للمعلومات للمعلومات الموزعة عادة. يفقد أقل معلومات من التعريف الكملي الموحد بـ 4 بتات (INT4) أو Float4 القياسي.

**Double quantization**: استمرارات الكمية نفسها تأخذ الذاكرة. كل كتلة من 64 وزنها تحتاج إلى عامل مقياس fp32 (4 بايت). بالنسبة لنموذج 7B، هذا هو 0.4GB إضافي. الكمية المزدوجة تحكم هذه المستمرات إلى fp8, وتقلل من التكلفة الجوية إلى 0.1GB. صغير ولكن يجمع.

**Paged optimizers**خلال التدريب، يمكن أن تتجاوز حالة المحفز (دفع وأزمة آدم) ذاكرة GPU على تسلسلات طويلة. يستخدم المحفز المصفحات ذاكرة موحدة NVIDIA لتصفح حالة المحفز تلقائيًا إلى ذاكرة CPU عندما يتم استنفاد ذاكرة GPU، ويعيد الصفحة مرة أخرى عند الحاجة. هذا يمنع انهيار OOM على حساب بعض التنفيذ.

### مسألة الجودة

هل تقليل المعلمات أو تقييم القاعدة يؤثر على الجودة؟

| Method | MMLU (5-shot) | MT-Bench | HumanEval |
|--------|--------------|----------|-----------|
| Full fine-tune (Llama 2 7B) | 48.3 | 6.72 | 14.6 |
| LoRA r=16 | 47.9 | 6.68 | 14.0 |
| QLoRA r=16 (NF4) | 47.5 | 6.61 | 13.4 |
| QLoRA r=64 (NF4) | 48.1 | 6.70 | 14.2 |

يقع مقياس LoRA عند r=16 ضمن 1% من التنسيق الدقيق الكامل على معظم المعايير. يفقد QLoRA عند r=16 جزءًا آخر من المئة. يطابق QLoRA عند r=64 في الأساس التنسيق الدقيق الكامل مع استخدام 90٪ أقل من الذاكرة.

### التكاليف الحقيقية

التنسيق الدقيق للاما 3 8B على 50،000 مثال (3 حقول):

| Method | GPU | Time | Cost |
|--------|-----|------|------|
| Full fine-tune | 2x A100 80GB | 8 hours | ~$32 |
| LoRA r=16 | 1x A100 40GB | 4 hours | ~$8 |
| QLoRA r=16 | 1x RTX 4090 24GB | 6 hours | ~$5 |
| QLoRA r=16 (Unsloth) | 1x RTX 4090 24GB | 2.5 hours | ~$2 |
| QLoRA r=16 | 1x T4 16GB | 12 hours | ~$4 |

تكلفة QLoRA على جهاز GPU لمستهلك واحد أقل من غداء. هذا هو السبب في انفجار مجتمع ضبط الدقة المفتوحة في عام 2023 ولماذا كل إطار تدريب أقل من QLoRA بطبيعة الحال في عام 2026.

### كومة PEFT 2026

| Framework | What it is | Pick when |
|-----------|-----------|-----------|
| **Hugging Face PEFT** | The canonical LoRA/QLoRA/DoRA/IA3 library | You want raw control and your training loop is already on `transformers.Trainer` |
| **TRL** | HF's reinforcement-from-feedback trainers (SFT, DPO, GRPO, PPO, ORPO) | You need DPO/GRPO after SFT; built on top of PEFT |
| **Unsloth** | Triton-kernel rewrite of the forward/backward pass | You want 2-5x speedup + half the VRAM with no accuracy loss; Llama/Mistral/Qwen family |
| **Axolotl** | YAML-config wrapper over PEFT + TRL + DeepSpeed + Unsloth | You want reproducible, version-controlled training runs |
| **LLaMA-Factory** | GUI/CLI/API over PEFT + TRL | You want zero-code fine-tuning; 100+ model families supported |
| **torchtune** | Native PyTorch recipes, no `transformers` dep | You want minimal deps and your org already standardizes on PyTorch |

قاعدة البصمة: استخدام البحث أو تجربة فريدة → PEFT. خط أنابيب الإنتاج المتكرر → أكسولوتل مع أجزاء Unsloth تمكينها. تصميم النماذج الأولية القذيفة → LLaMA-Factory.

### إندماج المعدات

بعد التدريب، لديك شيئين: نموذج الأساس المجمد ومعدل LoRA الصغير (عادة 10-100MB).

1. **Keep them separate**: تحميل النموذج الأساسي، تحميل المعدل فوق. تبادل المعدلات للمهام المختلفة. هكذا يمكنك تقديم العديد من التغيرات المعدلة من النموذج الأساسي واحد.

2. **Merge them permanently**: حساب W' = W + (ألفا / ر) * BA و حفظ النتيجة كنموذج جديد كامل. النموذج المدمج هو نفس الحجم من الأصلي. لا تكلفة استنتاجية. لا يوجد مكيّف لإدارة.

لخدمة مهام متعددة (معدل دعم العملاء، معدل رمز، معدل ترجمة) ، حافظ على تفاصيلها. لتنفيذ نموذج متخصص واحد، دمج.

تقنيات الاندماج المتقدمة لمجتمع العديد من المعدات:

- **TIES-Merging**(يادف وزملاء 2023): يقطع معايير الحجم الصغير، يحل الصراعات الإشارية، ثم يدمج. يقلل من التداخل بين المكيفات.
- **DARE**(Yu et al. 2023): يُسقط عشوائياً معايير المعدل قبل دمجها ويعيد التقييم على الباقي. فعال بشكل مفاجئ في دمج القدرات.
- **Task arithmetic**: ببساطة إضافة أو خصم وزن المعدل. إضافة معدل "رمز" ومعدل "رياضيات" غالبا ما ينتج نموذج جيد في كليهما.

### عندما لا يجب أن تكون على ما يرام

التنسيق الدقيق هو الخيار الثالث، وليس الأول.

**First: prompt engineering.**اكتب طلب نظام أفضل، أضف بعض الأمثلة القليلة، استخدم سلسلة التفكير، هذا لا يكلف شيئاً و يستغرق دقائق، إذا كان التسجيل يصل لك إلى 80٪ من الطريق، فربما لا تحتاج إلى ضبطها.

**Second: RAG.**إذا كان النموذج بحاجة إلى معرفة بياناتك المحددة (وثائق، قاعدة المعرفة، كتالوج المنتجات) ، فإن الاسترداد أرخص وأكثر صيانة من خبزها في الوزن. انظر الدروس 06.

**Third: fine-tuning.**استخدم هذا عندما تحتاج إلى النموذج لتبني نمط أو شكل أو نمط معين لا يمكن تحقيقه من خلال الاستفسار. عندما تحتاج إلى خروج منظمة ثابتة. عندما تحتاج إلى تصنيف نموذج أكبر إلى أصغر. عندما يهم التأخير ولا يمكنك تحمل الرموز الإضافية من استفسار بضع أوراق.

```mermaid
graph TD
    Start["Need better model behavior?"] --> PE["Try prompt engineering"]
    PE -->|"Works"| Done["Ship it"]
    PE -->|"Not enough"| RAG["Need external knowledge?"]
    RAG -->|"Yes"| RAGBuild["Build RAG pipeline"]
    RAG -->|"No, need style/format change"| FT["Fine-tune with LoRA/QLoRA"]
    RAGBuild -->|"Works"| Done
    RAGBuild -->|"Also need style change"| FT
    FT --> Done

    style Start fill:#1a1a2e,stroke:#e94560,color:#fff
    style Done fill:#0f3460,stroke:#16213e,color:#fff
```

```figure
lora-params
```

## بناءها

نطبق "لورا" من الصفر في "بيتورش" النقي، لا مكتبات، لا سحر، ستبني طبقة "لورا" وتحقنها في نموذج، وتدربها، وتدمج الوزن مرة أخرى.

### الخطوة الأولى: طبقة لورا

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.randn(in_features, rank) * (1 / math.sqrt(rank)))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling
```

يتم تشغيل A مع قيم عشوائية مقياسية. يتم تشغيل B إلى الصفر. يبدأ المنتج BA من الصفر، لذلك يبدأ النموذج بسلوكها الأصلي.

### الخطوة الثانية: طبقة خطية ملفوفة بـ "لورا"

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

يتم تجميد الطبقة الخطية الأصلية. يمكن تدريبها فقط لمعايير LoRA (A و B).

### الخطوة الثالثة: حقن لورا في النموذج

```python
def inject_lora(model, target_modules, rank=8, alpha=16):
    for param in model.parameters():
        param.requires_grad = False

    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]
                lora_linear = LinearWithLoRA(module, rank, alpha)
                setattr(parent, child_name, lora_linear)
                lora_layers[name] = lora_linear
    return lora_layers
```

أولاً، قم بتجميد كل معايير في النموذج. ثم قم بالتمرير على شجرة النموذج، واكتشف طبقات خطية تطابق أسماء المستهدف الخاصة بك، واستبدلها بإصدارات LoRA الملفوفة. هي المصفوفات A و B LoRA هي المعايير الوحيدة التي يمكن تدريبها في النموذج بأكمله.

### الخطوة الرابعة: احتساب المعايير

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }
```

### الخطوة 5: إندماج الوزن مرة أخرى

```python
def merge_lora_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            with torch.no_grad():
                merged = (
                    module.lora.A @ module.lora.B
                ) * module.lora.scaling
                module.linear.weight.data += merged.T
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, module.linear)
```

بعد الاندماج، تختفي طبقات LoRA، النموذج هو نفس الحجم من الأصلي مع التكيف مخبز في الوزن. لا يوجد استنتاج على التكلفة العليا.

### الخطوة 6: محاكاة كمية QLoRA

```python
def quantize_to_nf4(tensor, block_size=64):
    blocks = tensor.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
    scales = torch.clamp(scales, min=1e-8)
    quantized = torch.round(blocks / scales).clamp(-8, 7).to(torch.int8)
    return quantized, scales

def dequantize_from_nf4(quantized, scales, original_shape):
    dequantized = quantized.float() * scales
    return dequantized.reshape(original_shape)
```

هذا يحاكي تحديد الكميات من 4 بتات عن طريق رسم خرائط للأوزان إلى 16 مستوى منفصلة داخل كتلة من 64 .

### الخطوة السابعة: حلقة التدريب

```python
def train_lora(model, data, epochs=5, lr=1e-3, batch_size=4):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(len(data["inputs"]))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x = data["inputs"][batch_idx]
            y = data["targets"][batch_idx]

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

    return losses
```

### الخطوة الثامنة: التجربة الكاملة

```python
def demo():
    torch.manual_seed(42)
    d_model = 256
    n_classes = 10

    model = nn.Sequential(
        nn.Linear(d_model, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    )

    n_samples = 500
    x = torch.randn(n_samples, d_model)
    y = torch.randint(0, n_classes, (n_samples,))
    y_onehot = torch.zeros(n_samples, n_classes).scatter_(1, y.unsqueeze(1), 1.0)

    data = {"inputs": x, "targets": y_onehot}

    params_before = count_parameters(model)

    lora_layers = inject_lora(
        model, target_modules=["0", "2"], rank=8, alpha=16
    )

    params_after = count_parameters(model)

    losses = train_lora(model, data, epochs=20, lr=1e-3)

    merge_lora_weights(model)
    params_merged = count_parameters(model)

    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_merged": params_merged,
        "losses": losses,
    }
```

يخلق التجربة نموذجًا صغيرًا ، ويُحقق LoRA في طبقتين ، ويعمل عليه ، ويهضم الوزن مرة أخرى. ينخفض عدد المعايير من التدريب الكامل إلى ~ 1% قابل للتدريب أثناء تدريب LoRA ، ثم يعود إلى الهندسة المعمارية الأصلية بعد الاندماج.

## استخدمها

مع نظام "الوجهه المقبلة" ، يأخذ "لورا" على نموذج حقيقي حوالي 20 سطر:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

بالنسبة لـ QLoRA، أضف بيتساند بايتس كمية:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

model = get_peft_model(model, lora_config)
```

هذا هو، نفس حلقة التدريب نفس خط البيانات النموذج الأساسي يعيش الآن في 4 بتات، مُعايضات LoRA تدرب في Fp16, والشيء كله يناسب في 6 جيجابايت.

للتدريب مع مدرب الوجه المقبض:

```python
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./lora-adapter")
```

المعدل المحفوظ هو 10-100 ميغابايت. النموذج الأساسي يبقى غير متأثر. يمكنك مشاركة المعدلات على حلقة Hugging Face دون إعادة توزيع النموذج الكامل.

## أرسله

هذا الدرس ينتج عن:
- `outputs/prompt-lora-advisor.md`-- طلب يساعدك على تحديد درجة LoRA، وحدات الهدف، والعلامات العالية لمهمتك المحددة
- `outputs/skill-fine-tuning-guide.md`-- مهارة تعلّم العملاء شجرة القرارات متى وكيفية ضبط

## التمارين

1. **Rank ablation study.**قم بتشغيل التجربة التجريبية مع الصفوف 2، 4، 8، 16، 32، و 64. قم بتخطيط الخسارة النهائية مقابل الصف. ابحث عن نقطة العائد المتناقص حيث لا يزيد تضاعف الصف عن النصف الخسارة. بالنسبة لمهمة تصنيف بسيطة على ميزات 256 بعد، يجب أن يكون هذا حوالي r = 8-16 .

2. **Target module comparison.**تعديل inject_lora لتحديد الارض فقط طبقة "0"، طبقة "2"، طبقة "4" وحدها، والثلاثة. تدريب كل فارقة لمدة 20 حقبة. مقارنة سرعة التقارب والخسارة النهائية. هذا يعكس القرار الحقيقي لتحديد الارض q_proj مقابل v_proj مقابل جميع الطبقات الخطية.

3. **Quantization error analysis.**خذ ماتريصات الوزن التي تم تدريبها قبل وبعد quantize_to_nf4 / dequantize_from_nf4. احسب متوسط الخطأ التربيعي، أقصى خطأ مطلق، والارتباط بين الوزن الأصلي والإعادة بناء. التجربة مع قيم حجم الكتل من 32, 64, 128 و 256.

4. **Multi-adapter serving.**قم بتدريب مُعايير LoRA على مجموعتين فرعية مختلفة من البيانات (حتى المؤشرات مقابل المؤشرات الغريبة). حفظ كلا المُعايير. قم بتحميل النموذج الأساسي مرة واحدة، ثم قم بتبادل المُعايير وتحقق من أن كل منها ينتج نتائج مختلفة على نفس المدخل. هكذا تعمل أنظمة الإنتاج على عدة نماذج دقيقة من قاعدة واحدة.

5. **Merge vs. unmerged inference.**مقارنة خروج نموذج LoRA قبل وبعد merge_lora_weights على نفس 100 مدخل. التحقق من أن الخروج هي متطابقة (في حدود تحمل نقطة عائمة من 1e-5). ثم استنتاج سرعة مقارنة لكلا - دمج يجب أن يكون أسرع قليلا لأنه هو المصفوفة واحدة مضاعفة بدلا من اثنين.

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| LoRA | "Efficient fine-tuning" | Low-Rank Adaptation: freeze base weights, train two small matrices A and B whose product approximates the full weight update |
| QLoRA | "Fine-tune on a laptop" | Quantized LoRA: load the base model in 4-bit NF4, train LoRA adapters in fp16 on top, enabling 7B fine-tuning in 6GB VRAM |
| Rank (r) | "How much the model can learn" | The inner dimension of the A and B matrices; controls expressiveness vs. parameter count |
| Alpha | "LoRA learning rate" | Scaling factor applied to the LoRA output; alpha/r scales the adaptation's contribution to the final output |
| NF4 | "4-bit quantization" | Normal Float 4: a 4-bit data type with quantization levels at normal distribution quantiles, optimal for neural network weights |
| Adapter | "The small trained part" | The LoRA A and B matrices saved as a separate file (10-100MB), loadable on top of any copy of the base model |
| Target modules | "Which layers to LoRA" | The specific linear layers (q_proj, v_proj, etc.) where LoRA adapters are injected |
| Merging | "Bake it in" | Computing W + (alpha/r) * BA and replacing the original weight, eliminating the adapter overhead at inference |
| Paged optimizers | "Don't OOM during training" | Offloading optimizer states (Adam momentum, variance) to CPU when GPU memory is exhausted |
| Catastrophic forgetting | "Fine-tuning broke everything else" | When updating all weights causes the model to lose previously learned capabilities |

## المزيد من القراءة

- هوو وآخرون، "لورا: تكييف درجة منخفضة لنماذج اللغة الكبيرة" (2021) -- الورقة الأصلية التي تقدم طريقة التفكك منخفضة الدرجة، اختبرت على GPT-3 175B مع درجة منخفضة 4
- ديتمرز وغيرهم، "QLoRA: التنسيق الجيد لنموذجات اللغة المعدلة" (2023) -- يقدم NF4 ، التعريف المعدل المزدوج ، ومحفزات الصفحات ، مما يتيح 65B التنسيق الجيد على GPU واحد 48GB
- وثائق مكتبة PEFT (huggingface.co/docs/peft) -- المكتبة القياسية لـ LoRA و QLoRA ، وغيرها من الطرق الفعالة بالبرامج في النظام البيئي Hugging Face
- ياداف وغيره، "TIES-Merging: Solving Interference When Merging Models" (2023) -- تقنيات لمزيج العديد من مُعدات LoRA دون تدهور الجودة
- [Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (NeurIPS 2023)](https://arxiv.org/abs/2305.18290)-- استنتاج DPO؛ مرحلة تحسين الاختيارات التي تأتي بعد SFT، لا يوجد نموذج مكافأة مطلوب.
- [TRL documentation](https://huggingface.co/docs/trl/)-- المرجع الرسمي ل`SFTTrainer`،`DPOTrainer`،`KTOTrainer`، و سطح التكامل مع PEFT/bitsandbytes/Unsloth.
- [Unsloth documentation](https://docs.unsloth.ai/)-- الأجزاء المدمجة التي تضاعف التناسب الدقيق وتقلل من الذاكرة إلى النصف؛ طبقة الأداء تحت TRL.
- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/)-- مدرب متعدد GPU SFT / DPO / QLoRA الذي تم تكوينه بواسطة YAML ؛ بديل للتكوين كرمز للخطوط المكتوبة يدوياً.
