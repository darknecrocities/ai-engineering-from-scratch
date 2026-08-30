# لماذا تحولات  المشاكل مع RNNs

> تقوم RNNs بمعالجة الرموز الواحدة في كل مرة. تقوم المحولات بمعالجة جميع الرموز في وقت واحد. هذا الرهان المعماري الوحيد غير كل منحنى التوسع في التعلم العميق بعد عام 2017.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 minutes

## المشكلة

قبل عام 2017، كان كل نموذج تسلسل حديث على الكوكب  لغة، ترجمة، خطاب  شبكة عصبية متكررة. حصلت LSTMs و GRU على مقياس ترجمة معادل ImageNet لمدة نصف عقد. كانت الأداة الوحيدة التي كان لدى أي شخص.

كان لديهم ثلاثة نقاط ضعف قاتلة، الحسابات المتسلسلة يعني أنه لا يمكنك التوازي على طول محور الزمن:`t+1`يحتاج إلى الحالة الخفية من الوسائل`t`. تسلسل 1,024 رمز يعني 1,024 خطوة تسلسلية على GPU التي يمكن أن تفعل 1,000,000 نقطة عائمة العمليات في كل دورة. تدريب الوقت الحائط-الساعة مقياس خطيا مع طول تسلسل على الأجهزة المصممة للموازاة.

يعني التراجع المتلاشى أن المعلومات 50 رمزاً إلى الوراء قد ضُغطت بالفعل من خلال 50 غير خطية. وحدات متكررة المُغلقة (LSTM، GRU) أرتدت إلى التغلب ولكن لم تُزيل ذلك. الاعتمادات على المدى الطويل  "كان الكتاب الذي قرأته في الصيف الماضي في طائرة إلى كيوتو..."  فشل بشكل روتيني.

الحالات الخفية ذات الواسعة الثابتة يعني أن المُشفّر ضغط على تسلسل المصدر بأكمله إلى متجه واحد قبل أن يرى المُشفّر أي شيء. لا يهم ما إذا كان المصدر 5 رموز أو 500؛ عقد الزجاجة هو نفس الشكل.

مقالة 2017 "الاهتمام هو كل ما تحتاجه" اقترحت شيئا جذريا: انزل التكرار تماما. دع كل موقف يشاهد كل موقف آخر بالتوازي. تدريب في مضاعفة واحدة كبيرة المصفوفة بدلا من 1,024 متتالية.

النتيجة تهيمن على كل طريقة بحلول عام 2026. اللغة (GPT-5 ، كلود 4 ، لامة 4) ، الرؤية (ViT ، DINOv2 ، SAM 3) ، الصوت (سوسر) ، البيولوجيا (ألفافولد 3) ، الروبوتات (RT-2). نفس الكتلة ، مدخلات مختلفة.

## المفهوم

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Recurrence as a bottleneck.**الحسابات الـ RNN `h_t = f(h_{t-1}, x_t)`كل خطوة تعتمد على الخطوة السابقة لا يمكنك الحساب`h_5`قبل ذلك`h_4`على أجهزة المعالجة المعالجة المعالجة الحديثة مع أكثر من 10،000 نواة متوازية، هذا يضيع 99٪ من السيليكون على تسلسل طويل.

**Attention as a broadcast.**حسابات الاهتمام الذاتي`output_i = sum_j(a_ij * v_j)`لكل زوج`(i, j)`المصفوفة الاهتمامية بأكملها تعتمد على المصفوفة المكتسبة واحدة لا تتوقف أي خطوة على الخطوة الأخرى

**The speedup is not a constant.**هذا هو الفرق بين`O(N)`عمق تسلسلي و`O(1)`عمق سلسلة. في الممارسة العملية، المحولات تدرب 510 × أسرع في كل عصر على الأجهزة المقابلة عند N = 512, والفجوة تتوسع مع طول التسلسل حتى تضرب `O(N²)`جدار الذاكرة للاهتمام (الذي صارته Flash Attention لاحقاً  انظر الدروس 12).

**What transformers cost.**نطاقات الذاكرة الاهتمام مثل`O(N²)`بالنسبة للسياق 2K، حسناً. بالنسبة للسياق 128K، تحتاج إلى نوافذ زلقة، استنتاجات RoPE، فلاش انتباه الطلاء، أو المتغيرات الانتباه الخطية.`O(N)`في كل من الوقت والذاكرة، المحولات تتداول الوقت مقابل الذاكرة ثم تستعيد الوقت من خلال التوازي.

**The inductive bias shift.**يفترض RNNs الموقع والجدد. يتصور المحولون شيئاً  كل زوج مرشح للاهتمام. لهذا السبب يحتاج المحولون إلى المزيد من البيانات للتدريب بشكل جيد ولكن يتوسع بعد أن يحصلوا عليه. تشينشيلا (2022) رسمياً هذا: مع إعطاء رموز كافية ، يتغلب المحول دائمًا على RNN من عدد المعلمات المتساوية.

```figure
rnn-vs-parallel
```

## بناءها

لا شبكة عصبية هنا نقوم بتقريب عقدة الزجاجة الرقمية حتى تشعر الفجوة على جهاز الكمبيوتر المحمول الخاص بك.

### الخطوة الأولى: قياس عمق التسلسل

انظر`code/main.py`نُبني وظيفتين. أحدهما يرمز تسلسل كسلسلة من الإضافات (سلسلة، مثل RNN). أحدهما يرمزها كخفض متوازي (بث، مثل الاهتمام). نفس الرياضيات، رسوم بيانية مختلفة.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

نحن نقوم بتوقيت كل من على تسلسلات تصل إلى 100،000 عنصر. نسخة RNN هي O(N) ومركز واحد من نظام التشغيل المركزي. حتى في بيثون النقي، فإن تخفيض النمط الاهتمام يضرب على طول ≥ 1000 لأن بيثون `sum()`يتم تنفيذها في C وتكرارها دون تكلفة المترجم لكل خطوة.

### الخطوة الثانية: احتساب العمليات النظرية

كلتا الخوارزميات تضيف N. الفرق هو * عمق الاعتماد *: كم من العمليات يجب أن تحدث متتالية قبل أن يتم البدء التالي. عمق RNN = N. عمق الاهتمام = log(N) مع تخفيض شجرة ، أو 1 مع مسح متوازي. العمق ، وليس العدد العملي ، يحدد وقت GPU.

### الخطوة الثالثة: التوسع التجريبي على تسلسلات طويلة

نقوم بطبع جدول زمني يجعل الفجوة O(N) مرئية. على جهاز كمبيوتر محمول 2026 Mac، التسلسلات تحت 1000 عنصر سريعة جدًا لقياسها. تسلسلات من 100,000 تظهر مسحًا خطيًا نظيفًا. قم بتقييم ذلك إلى محول 16,384 رمز مع مساوية LSTM 12 طبقة وسترى لماذا كان ساعة الجدار التدريبية مقصوداً في عام 2016.

## استخدمها

متى لا يزال لاختيار RNN في 2026:

| Situation | Pick |
|-----------|------|
| Streaming inference, one token at a time, constant memory | RNN or state-space model (Mamba, RWKV) |
| Very long sequences (>1M tokens) where attention memory explodes | Linear attention, Mamba 2, Hyena |
| Edge device with no matmul accelerator | Depthwise-separable RNN still wins on FLOPs/watt |
| Anything else (training, batched inference, context up to 128K) | Transformer |

النماذج الفضائية للدولة (SSM) مثل Mamba هي أساسا RNNs مع المعلمات المهيكلة التي تعطيهم أفضل من كليهما: `O(N)`دراسة الذاكرة التشخيصية، التدريب الموازي عن طريق التشخيص الانتقائي. يعيدون 90% من جودة المحول مع تحسين مقياس السياق الطويل. في عام 2026 تعالج معظم المختبرات الحدودية نماذج محولة SSM + محولات (مثل جامبا، سامبا)  لا تموت التكرار، إنه مكون.

## أرسله

انظر`outputs/skill-architecture-picker.md`. يختار المهارة بنية لمشكلة تسلسل جديدة نظراً لقيود الطول والعبور والمدفوعات على ميزانية التدريب. يجب أن ترفض دائمًا توصية RNN نقية للتدريب فوق رموز 1B دون ذكر التنازل.

## التمارين

1. **Easy.**خذ`rnn_style`من`code/main.py`وبدل الحالة المخفية المتعددة بالمتجه الطويل-64 من الحالات المخفية. إعادة قياس. كم تنمو التكلفة السريرية مع بعد الحالة المخفية؟
2. **Medium.**قم بتنفيذ جمع مُسبق متوازي (مسح هيليس-ستيل) في بيثون النقي. تحقق من أنّه ينتج نفس الخروج الرقمي مثل المسح المتسلسل على طول 1024. احتساب العمق.
3. **Hard.**نقل التقليل على طريقة الاهتمام إلى PyTorch على GPU. وقت كل من اثنين كما تسبح طول التسلسل من 64 إلى 65،536.

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Recurrence | "RNNs are sequential" | Computation where step `t` depends on step `t-1`, forcing serial execution along the time axis. |
| Serial depth | "How deep the graph is" | Longest chain of dependent ops; bounds wall-clock even on infinite hardware. |
| Attention | "Let tokens look at each other" | Weighted sum `sum_j a_ij v_j` where `a_ij` comes from a similarity score between positions i and j. |
| Context window | "How much the model sees" | Number of positions an attention layer can take as input; quadratic memory cost scales here. |
| Inductive bias | "Assumptions baked into the architecture" | Prior about what the data looks like; CNNs assume translation invariance, RNNs assume recency. |
| State-space model | "RNN with algebra behind it" | Recurrence parameterized for parallel training via structured state-space matrices. |
| Quadratic bottleneck | "Why context costs so much" | Attention memory = `O(N²)` in sequence length; Flash Attention hides the constants, not the scaling. |

## المزيد من القراءة

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762)-الورقة التي قتلت التكرار في النظام الرئيسي للاستكشاف النفسي
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)حيث ولدت الاهتمام، وترتبط على RNN.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)ورقة LSTM الأصلية، فقط للمسجلة.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) الإجابة المتكررة الحديثة للمتحولات.
