# نقل التعلم وتحسينه

> شخص آخر قضى مليون ساعة في معالجة المعالجة المعالجة المعالجة (الذي يُعلم الشبكة كيف تبدو الحواف والنسيج وأجزاء الأشياء

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 4 Lesson 04 (Image Classification)
**Time:** ~75 minutes

## أهداف التعلم

- تمييز استيراد الميزات من التنسيق الدقيق واختيار الميزة المناسبة بناءً على حجم مجموعة البيانات ومسافة النطاق وال ميزانية الحسابية
- تحميل العمود الفقري المُدرب مسبقاً، واستبدال رأس المُصنف، وتدريب رأس فقط إلى خط أساسي العمل في أقل من 20 خط
- تخفيف طبقات مع معدلات التعلم التمييزية تدريجياً حتى يتمكن الميزات الجينية المبكرة من الحصول على تحديثات أصغر من تلك المتأخرة المحددة للمهمة
- تشخيص ثلاثة أخطاء شائعة: التنحيل من الميزات عالية جدا LR على الكتل غير المجمدة، انهيار إحصاءات BN على مجموعات بيانات صغيرة، ونسيان كارثي

## المشكلة

تدريب ResNet-50 على ImageNet يكلف حوالي 2000 ساعة GPU. عدد قليل جدا من الفرق لديها هذا الميزانية لكل مهمة يتم شحنها. ما تقريبا كل فريق في الواقع شحن هو العمود الفقري المدرب مسبقا مع رأس جديد مدرب على بضع مئات أو بضعة آلاف الصور المحددة للمهمة.

هذا ليس طريق مختصر أول كتلة من أي قناة CNN تدرب على ImageNet تعلم الحواف والمرشحات مثل Gabor. في الأقسام القليلة التالية تتعلم النسخ والحافظات البسيطة الكتل الوسطى تتعلم أجزاء الكائنات الكتل الأخيرة تتعلم مزيجات تبدأ في أن تبدو مثل الفئات 1000 ImageNet. النسبة الـ 90% الأولى من هذه الهرمية تنتقل تقريباً دون تغيير إلى التصوير الطبي، والتفتيش الصناعي، بيانات الأقمار الصناعية، وكل مهمة رؤية أخرى  لأن الطبيعة لديها مفرد محدود من الحواف والنسائح. الـ10٪ الأخيرة هي ما تدرب عليه

الحصول على حق نقل لديه ثلاثة أخطاء تنتظرك: تدمير الميزات المسبقة تدريب مع معدل التعلم مرتفع جدا، وجوع نموذج المعلومات عن طريق تجميد أكثر من اللازم، والسماح BatchNorm الإحصاءات الجارية تتحرك نحو مجموعة بيانات صغيرة التي لم تتعلم بقية الشبكة من. هذا الدرس يذهب كل منهم عن قصد.

## المفهوم

### استخرج الميزات مقابل ضبطها

نظامين، يتم اختيارهم من خلال مدى ثقتك في الميزات المسبقة والتدريبات التي لديك

```mermaid
flowchart TB
    subgraph FE["Feature extraction — backbone frozen"]
        FE1["Pretrained backbone<br/>(no gradient)"] --> FE2["New head<br/>(trained)"]
    end
    subgraph FT["Fine-tuning — end-to-end"]
        FT1["Pretrained backbone<br/>(tiny LR)"] --> FT2["New head<br/>(normal LR)"]
    end

    style FE1 fill:#e5e7eb,stroke:#6b7280
    style FE2 fill:#dcfce7,stroke:#16a34a
    style FT1 fill:#fef3c7,stroke:#d97706
    style FT2 fill:#dcfce7,stroke:#16a34a
```

قواعد الإبهام:

| Dataset size | Domain distance | Recipe |
|--------------|-----------------|--------|
| < 1k images | close to ImageNet | Freeze backbone, train head only |
| 1k-10k | close | Freeze first 2-3 stages, fine-tune the rest |
| 10k-100k | any | Fine-tune end-to-end with discriminative LR |
| 100k+ | far | Fine-tune everything; consider training from scratch if domain is far enough |

"قرب من ImageNet" يعني تقريبا صور RGB طبيعية مع محتوى يشبه الكائنات. المسحات المعلوماتية المعلوماتية الطبية، الصور الأقمار الصناعية الجوية، والنظارية هي مجالات بعيدة  الميزات لا تزال تساعد، ولكن سوف تحتاج إلى السماح المزيد من الطبقات للتكيف.

### لماذا التجمد يعمل على الإطلاق

يكتشف موقع "ميكنيت" أنّه ليس متخصصًا في الفئات الـ1.000. وهي متخصصة في إحصاءات الصور الطبيعية: الحواف عند التوجهات المحددة، والتركيبات، وأنماط التناقض، والشكل البدائي. هذه الإحصاءات مستقرة على جميع المجالات البصرية التي يمكن أن يسميها الإنسان تقريباً. لهذا السبب، فإن نموذج تم تدريبه على ImageNet وتقييم الصفر الصاروخ على CIFAR-10 مع رأس خطي جديد فقط (لا توضيح جيد للعمود الفقري) يصل إلى 80٪ + دقة. الرأس يتعلم أي من الميزات المكتسبة بالفعل لتنسق لهذا المهمة.

### معدلات التعلم التمييزية

عندما تقوم بتفكيك التجميد، يجب أن تتدرب الطبقات المبكرة بطيئة أكثر من الطبقات المتأخرة. الطبقات المبكرة ترميز الميزات العامة التي تريد الحفاظ عليها؛ الطبقات المتأخرة ترميز بنية محددة للمهمة التي تحتاج إلى التحرك كثيرا.

```
Typical recipe:

  stage 0 (stem + first group): lr = base_lr / 100    (mostly fixed)
  stage 1:                       lr = base_lr / 10
  stage 2:                       lr = base_lr / 3
  stage 3 (last backbone group): lr = base_lr
  head:                          lr = base_lr  (or slightly higher)
```

في PyTorch هذه ليست سوى قائمة من مجموعات المعلمات تم تمرير إلى المحافظ. نموذج واحد، خمسة معدلات التعلم، صفر رمز إضافي.

### مشكلة "بيتش نورم"

طبقات BN تحمل`running_mean`و`running_var`المضخات التي تم تحديدها على ImageNet. إذا كانت مهمتك لديها توزيع مختلف للبيكسل  إضاءة مختلفة، جهاز استشعار مختلفة، مساحة مختلفة لون  تلك المضخات خاطئة. ثلاثة خيارات في ترتيب الاختيارات:

1. **Fine-tune with BN in train mode.**دع BN تحديث إحصائيات التشغيل مع كل شيء آخر. اختيار افتراضي عندما تكون مجموعة البيانات الخاصة بالمهمة متوسطة الحجم (>= 5k أمثلة).
2. **Freeze BN in eval mode.**حافظ على إحصاءات ImageNet وتدريب فقط الوزن. صحيح عندما مجموعة البيانات الخاصة بك صغيرة بما فيه الكفاية
3. **Replace BN with GroupNorm.**يزيل مشكلة المتوسط المتحرك بالكامل. يستخدم في الكشف والشقوق التقسيم حيث حجم اللحظة لكل GPU صغير.

إن أخطأت في هذا الأمر، فإن الدقة تتراوح بين 5 و15 بالمئة

### تصميم الرأس

رأس التصنيف هو 1-3 طبقة خطية زائد إلقاء اختياري كل مشعل رؤية العمود الفقري يرسل رأس افتراضي الذي يمكنك استبدال:

```
backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)          # ResNet
backbone.classifier[1] = nn.Linear(..., num_classes)                    # EfficientNet, MobileNet
backbone.heads.head = nn.Linear(..., num_classes)                       # torchvision ViT
```

بالنسبة لمجموعات بيانات صغيرة، تكون طبقة خطية واحدة كافية عادة. إضافة طبقة مخفية (خطية -> ReLU -> إيقاف -> خطية) تساعد عندما يكون توزيع المهام أبعد من توزيع التدريب في العمود الفقري.

### التهالك المتعدد للطبقات

نسخة أكثر سلاسة من LR التمييزية المستخدمة في التنسيقات الدقيقة الحديثة (BEiT، DINOv2، ViT-B). بدلاً من تجميع الطبقات إلى مراحل، أعط كل طبقة LR أصغر قليلاً من تلك فوقها:

```
lr_layer_k = base_lr * decay^(L - k)
```

مع التهالك = 0.75 و L = 12 كتلة محول، القطارات الأولى كتلة في `0.75^11 ≈ 0.04x`و هو أكثر أهمية لتحويل الموسيقى الدقيقة من لسي إن إن، حيث تكون الموسيقى الدقيقة المجموعة المرحلة عادة كافية.

### ما الذي يجب تقييمه

عمليات التعلم النقل تحتاج إلى رقمين لن تتبعها في عملية الخدش:

- **Pretrained-only accuracy**دقة الرأس مع تم تجميد العمود الفقري هذه هي الأرض
- **Fine-tuned accuracy**نفس النموذج بعد التدريب من نهايتها إلى نهايتها هذا هو سقفك

إذا كان المنسق الدقيق أقل من المميز فقط، لديك معدل التعلم أو خطأ BN.

```figure
transfer-learning
```

## بناءها

### الخطوة الأولى: قم بتحميل العمود الفقري المُدرب مسبقاً وتفتيشه

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print(backbone)
print()
print("classifier head:", backbone.fc)
print("feature dim:", backbone.fc.in_features)
```

`ResNet18`يحتوي على أربع مراحل (`layer1..layer4`) بالإضافة إلى جذع و`fc`كل قاعدة نخاع تصنيف مشعل رؤية لديها هيكل مماثل.

### الخطوة الثانية: استخراج الميزة

```python
def make_feature_extractor(num_classes=10):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model = make_feature_extractor(num_classes=10)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"trainable: {trainable:>10,}")
print(f"frozen:    {frozen:>10,}")
```

فقط`model.fc`العمود الفقري هو مستخرج مُجمد

### الخطوة الثالثة: التأقلم الدقيق التمييزي

أداة تُبني مجموعات المعلمات مع معدلات التعلم المحددة للمرحلة.

```python
def discriminative_param_groups(model, base_lr=1e-3, decay=0.3):
    stages = [
        ["conv1", "bn1"],
        ["layer1"],
        ["layer2"],
        ["layer3"],
        ["layer4"],
        ["fc"],
    ]
    groups = []
    for i, names in enumerate(stages):
        lr = base_lr * (decay ** (len(stages) - 1 - i))
        params = [p for n, p in model.named_parameters()
                  if any(n.startswith(k) for k in names)]
        if params:
            groups.append({"params": params, "lr": lr, "name": "_".join(names)})
    return groups

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 10)
for p in model.parameters():
    p.requires_grad = True

groups = discriminative_param_groups(model)
for g in groups:
    print(f"{g['name']:>10s}  lr={g['lr']:.2e}  params={sum(p.numel() for p in g['params']):>8,}")
```

`decay=0.3`يعني كل خطة قطار بنسبة 30% من سرعة الخطوة التالية. `fc`يصل`base_lr`،`layer4`يصل`0.3 * base_lr`،`conv1`يصل`0.3^5 * base_lr ≈ 0.00243 * base_lr`صوت متطرف، و من الناحية التجريبية يعمل

### الخطوة الرابعة: التعامل مع المجموعة

مساعدة لتجميد بيانات BN بدون تجميد أوزانها

```python
def freeze_bn_stats(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
    return model
```

اتصل بها بعد أن تستقر`model.train()`في بداية كل عصر`model.train()`يُحول كل شيء إلى وضع التدريب، وهذا يعكسها فقط لطبقات BN.

### الخطوة 5: حلقة تحسينية من نهاية إلى نهاية

```python
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

def fine_tune(model, train_loader, val_loader, device, epochs=5, base_lr=1e-3, freeze_bn=False):
    model = model.to(device)
    groups = discriminative_param_groups(model, base_lr=base_lr)
    optimizer = SGD(groups, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        if freeze_bn:
            freeze_bn_stats(model)
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            tr_total += x.size(0)
            tr_correct += (logits.argmax(-1) == y).sum().item()
        scheduler.step()

        model.eval()
        va_total, va_correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(-1)
                va_total += x.size(0)
                va_correct += (pred == y).sum().item()
        print(f"epoch {epoch}  train {tr_loss/tr_total:.3f}/{tr_correct/tr_total:.3f}  "
              f"val {va_correct/va_total:.3f}")
    return model
```

خمس فترات مع وصفة CIFAR-10 أعلاه تأخذ `ResNet18-IMAGENET1K_V1`من 70٪ دقة الصور الخطية الصفراء إلى 93٪ دقة دقيقة. الرأس وحده سوف تتحرك نحو 86٪ دون أن تلمس العظم الفقري.

### الخطوة السادسة: تخفيف التجميد تدريجياً

جدول زمني يفتح مرحلة واحدة في كل عصر من النهاية إلى البداية.

```python
def progressive_unfreeze_schedule(model):
    stages = ["layer4", "layer3", "layer2", "layer1"]
    yielded = set()

    def start():
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    def unfreeze(epoch):
        if epoch < len(stages):
            name = stages[epoch]
            yielded.add(name)
            for n, p in model.named_parameters():
                if n.startswith(name):
                    p.requires_grad = True
            return name
        return None

    return start, unfreeze
```

اتصل`start()`مرة قبل العصر الأول.`unfreeze(epoch)`إعادة بناء المحفز كلما تغير مجموعة المعايير التي يمكن تدريبها، وإلا فإن الحدود المتجمدة لا تزال تحتوي على لحظات مخفية تحملها.

## استخدمها

معظم المهام الحقيقية`torchvision.models`+ ثلاثة خطوط كافية. الآلات الثقيلة فوق مهمة عندما تلتقي بمشاكل التي لا يمكن إصلاحها من قبل المكتبة.

```python
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, num_classes)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

اثنين آخرين من الاختلالات في درجة الإنتاج:

- `timm`السفن ~ 800 قاعدة رأس مرئية مسبقة مع API متسقة (`timm.create_model("resnet50", pretrained=True, num_classes=10)`(لجميع الموسيقى الخفيفة خارج حديقة الحيوانات، إنها المعيار
- بالنسبة للمتحولات`transformers.AutoModelForImageClassification.from_pretrained(name, num_labels=N)`يعطيك ViT / BEiT / DeiT مع نفس تعبيرات التحميل مثل نماذج النص.

## أرسله

هذا الدرس ينتج عن:

- `outputs/prompt-fine-tune-planner.md` عرض يختار استخراج الميزات مقابل التقدم مقابل ضبط الدقيق من النهاية إلى النهاية بناءً على حجم مجموعة البيانات ، ومسافة النطاق ، وميزانية الحساب.
- `outputs/skill-freeze-inspector.md` مهارة التي، نظرا لنموذج PyTorch، تقرير ما هي المعايير التي يمكن تدريبها، والتي هي طبقات BatchNorm في وضع تقييم، وما إذا كان المتحسن يتم إعطاء المعايير التي يمكن تدريبها في الواقع.

## التمارين

1. **(Easy)**- إقترب`ResNet18`كمسبار خطي (مجمّد على العمود الفقري) وكمساعدة كاملة على مجموعة بيانات CIFAR المختلفة. قم بتقرير كل من الدقة جنبا إلى جنب. شرح أي فجوة تخبرك بتحويل الميزات بشكل جيد والتي تخبرك أنها لا تفعل ذلك.
2. **(Medium)**إدخال خطأ عمدا: set `base_lr = 1e-1`على مرحلة العمود الفقري بدلا من الرأس.`discriminative_param_groups`المساعد، سجل المرحلة التي تبدأ فيها كل مرحلة بالتباين.
3. **(Hard)**خذ مجموعة بيانات التصوير الطبي (مثل CheXpert-small أو PatchCamelyon أو HAM10000) وقارن ثلاثة أنظمة: (أ) ImageNet-pretrained frozen backbone + linear head؛ (ب) ImageNet-pretrained fine-tune end-to-end؛ (ج) تدريب الرمض. تقرير الدقة والتكلفة الحسابية لكل منها. في أي حجم مجموعة بيانات يصبح تدريب الرمض منافساً؟

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Feature extraction | "Freeze and train head" | Backbone parameters frozen, only the new classifier head receives gradient |
| Fine-tuning | "Retrain end-to-end" | All parameters trainable, usually with much smaller LR than scratch training |
| Discriminative LR | "Smaller LR for early layers" | Optimizer parameter groups where early-stage LR is a fraction of late-stage LR |
| Layer-wise LR decay | "Smooth LR gradient" | Per-layer LR multiplied by decay^(L - k); common in transformer fine-tunes |
| Catastrophic forgetting | "The model lost ImageNet" | A too-high LR overwrites pretrained features before the new task signal is learnt |
| BN statistics drift | "Running mean is wrong" | BatchNorm running_mean/var computed on a different distribution than the current task, silently hurting accuracy |
| Linear probe | "Frozen backbone + linear head" | Evaluation of pretrained features — accuracy of the best linear classifier on top of the frozen representation |
| Catastrophic collapse | "Everything predicts one class" | Happens when fine-tuning with an LR high enough to destroy features before gradients from the head can stabilise |

## المزيد من القراءة

- [How transferable are features in deep neural networks? (Yosinski et al., 2014)](https://arxiv.org/abs/1411.1792) الورقة التي تحدد كمية قابلية نقل الميزات عبر الطبقات
- [Universal Language Model Fine-tuning (ULMFiT, Howard & Ruder, 2018)](https://arxiv.org/abs/1801.06146) وصفة التمييز الأصلية لـ LR / التجمد التدريجي ؛ تنقل الأفكار مباشرة إلى الرؤية
- [timm documentation](https://huggingface.co/docs/timm) الإشارة إلى العظام الفقرية الحديثة للرؤية والخطوات الدقيقة التي تم تدريبها عليها
- [A Simple Framework for Linear-Probe Evaluation (Kornblith et al., 2019)](https://arxiv.org/abs/1805.08974) لماذا تُهم دقة الصفحة الخطية وكيفية الإبلاغ عنها بشكل صحيح
