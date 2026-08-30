# تفاهم في المعلومات والوثائق

> OCR هو خط أنابيب ثلاث مراحل  اكتشاف صناديق النص، وتعرف على الأحرف، ثم وضعها. كل نظام OCR الحديثة إعادة ترتيب هذه المراحل أو دمجها.

**Type:** Learn + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06 (Detection), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~45 minutes

## أهداف التعلم

- تتبع خط الأنابيب الكلاسيكية لـ OCR (اكتشاف -> التعرف -> التخطيط) والبدائل الحديثة من النهاية إلى النهاية (دونوت ، Qwen-VL-OCR)
- تنفيذ خسارة CTC (التصنيف الزمني للاتصال) لتدريب OCR المتسلسل إلى المتسلسل
- استخدام PaddleOCR أو EasyOCR لتحليل وثائق الإنتاج دون تدريب
- تمييز OCR، تحليل التخطيط، وفهم الوثائق  واختيار الأداة المناسبة لكل مهمة

## المشكلة

الصور المملئة بالنص موجودة في كل مكان: الإيصالات والفواتير والهوية والكتب المسحرة والنماذج واللواح البيضاء والرسومات والصور الشاشية. استخرج البيانات المهيكلة منها ليس فقط الحروف، ولكن "هذه هي المبلغ الإجمالي" هي واحدة من أعلى القيمة مشكلة الرؤية التطبيقية.

المجال ينقسم إلى ثلاث طبقات من المهارات:

1. **OCR proper**تحويل البيكسلات إلى نص
2. **Layout parsing**: إنتاج المجموعة OCR إلى مناطق (اللقب، الجسم، الجدول، العنوان).
3. **Document understanding**: استخراج الحقول المهيكلة ("فواتير_جميع = 42.50 دولار") من التخطيط.

كل طبقة لديها نهج كلاسيكي ومحاصر، والفرق بين "أريد نص من صورة" و "أريد المبلغ الإجمالي من هذا الإيصالات" أكبر مما تدرك معظم الفرق.

## المفهوم

### خط الأنابيب الكلاسيكي

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **Text detection**ينتج أرباعي خط أو كلمة.
- **Recognition**يصل كل منطقة إلى ارتفاع ثابت، ويقوم بإدارة جهاز CNN + BiLSTM + CTC لإنتاج تسلسل شخصية.
- **Layout**يعيد بناء ترتيب القراءة (من الأعلى إلى الأسفل، من اليسار إلى اليمين بالنسبة إلى اللاتينية؛ مختلفة بالنسبة إلى العربية واليابانية).

### المعلومات المتاحة في الفقرة الواحدة

إن التعرف على OCR ينتج تسلسلًا متغيرًا من خريطة ميزة ذات طول ثابت. يسمح لك CTC (Graves et al., 2006) بتدريب هذا دون توازن مستوى الأحرف. يقوم النموذج بإخراج توزيع على (الكلمات + الفراغ) في كل خطوة زمنية. تخسر CTC على هامش جميع التوصلات التي تقل إلى النص المستهدف بعد دمج التكرارات وإزالة الفراغات.

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

ويعتبر CTC هو السبب في عمل CRNN في عام 2015 وما زال يدرّب معظم نماذج OCR الإنتاجية في عام 2026.

### النماذج الحديثة من نهاية إلى نهاية

- **Donut**(كيم وزملاء، 2022)  مرموز ViT + مرموز نص؛ يقرأ صورة ويصدر JSON مباشرة. لا يوجد كشف نص، لا يوجد وحدات التخطيط.
- **TrOCR** جهاز تشخيص المتحول ViT + لـ OCR على مستوى الخط.
- **Qwen-VL-OCR / InternVL** نماذج كاملة للغة الرؤية مُعدّلة لمهام OCR؛ أفضل دقة في عام 2026 على الوثائق المعقدة.
- **PaddleOCR** خط أنابيب DB + CRNN الكلاسيكي في حزمة إنتاج ناضجة؛ لا يزال حزمة العمل مفتوحة المصدر.

تحتاج النماذج من نهاية إلى نهاية إلى المزيد من البيانات والحسابات ولكن تجنب تراكم الأخطاء في خطوط الأنابيب متعددة المراحل.

### تحليل التخطيط

بالنسبة للوثائق المهيكلة، قم بتشغيل كاشف التخطيط (LayoutLMv3 ، DocLayNet) الذي يضع علامة على كل منطقة: عنوان ، الفقرة ، الرسم ، الجدول ، ملاحظة أسفل. يصبح ترتيب القراءة بعد ذلك "تكرر عبر المناطق في ترتيب التخطيط ، المزدوج".

في الاستمارات، استخدم **Key-Value extraction**النماذج (دونوت للوثائق الغنية بالبصر، LayoutLMv3 للتسحينات العادية).

### مقاييس التقييم

- **Character Error Rate (CER)** مسافة ليفينشتاين / طول مرجع. أقل أفضل. هدف الإنتاج: < 2% على المسحات النظيفة.
- **Word Error Rate (WER)** نفس الشيء على مستوى الكلمات.
- **F1 on structured fields** للمهمات ذات القيمة الرئيسية؛ تدابير ما إذا كان `{invoice_total: 42.50}`يبدو صحيحاً
- **Edit distance on JSON** لتحليل الوثائق من نهاية إلى نهاية. أدخلت ورقة دونوت مسافة تعديل شجرة معايرة.

```figure
cv3-ctc-collapse
```

## بناءها

### الخطوة الأولى: فقدان CTC + مفكّر طموح

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss`يستخدم تنفيذ CuDNN الفعال عندما يكون متاحًا. إن المفكّر البشع أبسط من البحث عن الشعاع وعادة ما يكون ضمن 1% من إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إطار إ

### الخطوة الثانية: جهاز التعرف على CRNN الصغير

الحد الأدنى من CNN + BiLSTM لخط OCR.

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

المدخلات ذات الارتفاع الثابت (إن إن أقصى مستوى من الارتفاع إلى 1).

### الخطوة الثالثة: OCR الاصطناعي

توليد سلسلة من الأرقام السوداء إلى البيضاء لاختبار الدخان من النهاية إلى النهاية.

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

مجموعة بيانات OCR الحقيقية تضيف الخطوط والضوضاء والدورة والضباب واللون.

### الخطوة الرابعة: رسم التدريب

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

يجب أن تنخفض الخسارة من ~ 3 إلى ~ 0.2 على 200 خطوة على هذه البيانات الاصطناعية البسيطة.

## استخدمها

ثلاثة طرق إنتاج:

- **PaddleOCR** بالغة، سريعة، متعددة اللغات. استخدام خط واحد: `paddleocr.PaddleOCR(lang="en").ocr(image_path)`. . .
- **EasyOCR** Python-أصلي، متعددة اللغات، PyTorch العمود الفقري.
- **Tesseract** الكلاسيكية؛ لا تزال مفيدة للمستندات القديمة المسحرة عندما تتعرض النماذج للصراع.

لتحليل الوثائق من النهاية إلى النهاية، استخدم Donut أو VLM:

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

بالنسبة للإيصالات والفواتير والأوراق ذات الهيكل المتكرر، قم بتحسين دونوت. بالنسبة للمستندات التعسفية أو OCR مع التفكير، فإن VLM مثل Qwen-VL-OCR هو الافتراض الحالي.

## أرسله

هذا الدرس ينتج عن:

- `outputs/prompt-ocr-stack-picker.md` طلب يختار Tesseract / PaddleOCR / Donut / VLM-OCR نوع وثيقة معينة، اللغة، والهيكل.
- `outputs/skill-ctc-decoder.md` مهارة تكتب طموحة و البحث عن الشعاع CTC مبرمجة من الصفر، بما في ذلك التطبيع على الطول.

## التمارين

1. **(Easy)**تدريب TinyCRNN على سلسلة رقمية عشوائية 5 أرقام لمدة 500 خطوة. تقرير CER على مجموعة مدعومة.
2. **(Medium)**استبدل تشفير الشعاع ببحث الشعاع (beam_width=5). تقرير CER delta. على أي مدخلات يربح بحث الشعاع؟
3. **(Hard)**استخدم PaddleOCR على مجموعة من 20 إيصال، استخراج عناصر الخط، وحساب F1 ضد الحقيقة الأرضية المسموحة يدويا ل {item_name، سعر} أزواج.

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| OCR | "Text from pixels" | Turning image regions into character sequences |
| CTC | "Alignment-free loss" | Loss that trains a sequence model without per-timestep labels; marginalises over alignments |
| CRNN | "Classic OCR model" | Conv feature extractor + BiLSTM + CTC; the 2015 baseline still used in production |
| Donut | "End-to-end OCR" | ViT encoder + text decoder; emits JSON directly from image |
| Layout parsing | "Find regions" | Detect and label Title/Table/Figure/Paragraph regions in a document |
| Reading order | "Text sequence" | Ordering of recognised regions into a sentence; trivial for Latin, non-trivial for mixed layouts |
| CER / WER | "Error rates" | Levenshtein distance / reference length at character or word granularity |
| VLM-OCR | "LLM that reads" | A vision-language model trained or prompted for OCR tasks; current SOTA on complex documents |

## المزيد من القراءة

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) بنية CNN+RNN+CTC الأصلية
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) ورقة CTC الأصلية؛ مليئة بالكثير من الأفكار الخوارزمية
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664) محول لفهم الوثائق خالي من OCR
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) كومة OCR الإنتاج مفتوح المصدر
