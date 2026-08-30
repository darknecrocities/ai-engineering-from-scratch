# إنتاج الصور  GANs

> "جان" هي شبكتين عصبيتين في لعبة ثابتة، واحد يسحب، واحد ينتقد، يزدادون قدرة على التواصل مع بعضهم البعض حتى تخدع الرسوم النقاد.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 3 Lesson 06 (Optimizers), Phase 3 Lesson 07 (Regularization)
**Time:** ~75 minutes

## أهداف التعلم

- شرح اللعبة الحد الأدنى بين المولد والتمييز و لماذا يتوافق التوازن مع p_model = p_data
- تنفيذ DCGAN في PyTorch وجعله يخلق صور اصطناعية متماسكة 32x32 في أقل من 60 سطر
- استقرار تدريب GAN مع ثلاث حيل قياسية: فقدان غير المشبعة، المعيار الطيفي، TTUR (قاعدة تحديث مرتين)
- قراءة منحنى التدريب التي تميز التقارب الصحي من انهيار النظام، التذبذب، والتمييز-الفوز-كامل

## المشكلة

التصنيف يعلم الشبكة كيفية رسم صور إلى العلامات. يُعكس الجيل المشكلة: عينة صور جديدة تبدو وكأنها جاءت من نفس التوزيع. لا توجد إنتاج "صحيح" يمكنك التفاوت معه؛ هناك فقط توزيع تريد محاكته.

لا يمكن لـ (MSE، التنقل المتقاطع) قياس "هل جاءت هذه العينة من التوزيع الحقيقي". تقليل خطأ لكل بيكسل ينتج متوسطات مغمورة، وليس عينات واقعية. كان الانفجار هو معرفة الخسارة: تدريب شبكة ثانية وظيفتها هي معرفة حقيقة من مزيفة، واستخدام حكمها لدفع مولد.

وقد وضعت GANs (Goodfellow et al., 2014) هذا الإطار. بحلول عام 2018، كانت StyleGAN تنتج 1024 × 1024 وجهًا لا يمكن التمييز بينهما من الصور. وقد استولى نماذج التنشر على العرش على الجودة والتحكم فيها منذ ذلك الحين، ولكن كل حيلة تجعل التنشر عملية  خيارات التطبيع، الفضاء الخفيف، فقدان الميزات  تم فهمها لأول مرة على GANs.

## المفهوم

### الشبكات

```mermaid
flowchart LR
    Z["z ~ N(0, I)<br/>noise"] --> G["Generator<br/>transposed convs"]
    G --> FAKE["Fake image"]
    REAL["Real image"] --> D["Discriminator<br/>conv classifier"]
    FAKE --> D
    D --> OUT["P(real)"]

    style G fill:#dbeafe,stroke:#2563eb
    style D fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- نعم**generator**G يأخذ متجه للضوضاء `z`و يخرج صورة.**discriminator**D يأخذ صورة وتخرج مقياس واحد: احتمال أن الصورة حقيقية.

### اللعبة

(ج) يريد أن يكون (د) خاطئاً (د) يريد أن يكون محقاً

```
min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

اقرأ من اليمين إلى اليسار: D هو زيادة دقة على الحقيقي (`log D(real)`(و) وهمية`log (1 - D(fake))`(G) يقلل من دقة D على المزيفات`D(G(z))`أن تكون عالقاً

أثبت (غودفيلو) أن هذا الحد الأدنى له توازن عالمي حيث`p_G = p_data`, D يخرج 0.5 في كل مكان، و الانفصال بين النموات المولدة والواقعية هو صفر. الجزء الصعب هو الحصول على ذلك.

### الخسارة غير المشبعة

النموذج أعلاه غير مستقرة عدداً في بداية التدريب`D(G(z))`هو قريب من الصفر لكل مزيف، لذلك `log(1 - D(G(z)))`لديه تراجعات تختفي فيما يتعلق بـ G. الإصلاح: ضياع فليب G.

```
L_D = -E_x[log D(x)] - E_z[log(1 - D(G(z)))]
L_G = -E_z[log D(G(z))]                          # non-saturating
```

الآن متى`D(G(z))`إنّه قريب من الصفر، فقدان G كبير و تراجعها معلوميّ.

### قواعد معماري DCGAN

رادفورد، ميتز، تشينتالا (2015) تمزق سنوات من التجارب الفاشلة إلى خمس قواعد تجعل تدريب GAN مستقرة:

1. استبدل التجميع بالمركبات المتقدمة (كلتا الشباك).
2. استخدم معايير اللحظة في كل من المولد والتمييز، باستثناء خروج G ومدخل D.
3. إزالة الطبقات المتصلة بالكامل على المعماريات العميقة.
4. G يستخدم ReLU على جميع الطبقات باستثناء الخروج (tanh للخروج في [-1, 1]).
5. D يستخدم LeakyReLU (منفي_التنحدر=0.2) على جميع الطبقات.

كل GAN الحديث القائم على المجموعات (StyleGAN، BigGAN، GigaGAN) لا يزال يبدأ من هذه القواعد ويستبدل الأجزاء واحدة في كل مرة.

### أساليب الفشل وتوقيعاتها

```mermaid
flowchart LR
    M1["Mode collapse<br/>G produces a narrow<br/>set of outputs"] --> S1["D loss low,<br/>G loss oscillating,<br/>sample variety drops"]
    M2["Vanishing gradients<br/>D wins completely"] --> S2["D accuracy ~100%,<br/>G loss huge and static"]
    M3["Oscillation<br/>G and D keep trading<br/>wins forever"] --> S3["Both losses swing<br/>wildly with no downward trend"]

    style M1 fill:#fecaca,stroke:#dc2626
    style M2 fill:#fecaca,stroke:#dc2626
    style M3 fill:#fecaca,stroke:#dc2626
```

- **Mode collapse**: G يجد صورة واحدة تحكم D وينتج ذلك فقط.
- **Discriminator wins**D يصبح قوياً جداً بسرعة كبيرة، وتختفي تراجعيات G. إصلاح: D أصغر، معدل تعلم D أقل، أو تطبيق تسهيل اللوحة على اللوحات الحقيقية.
- **Oscillation**: الفوز في التجارة دون الاقتراب من التوازن. تحديد: TTUR (تعلم D أسرع من G بمعدل 2-4) ، أو الانتقال إلى خسارة Wasserstein.

### التقييم

لا يوجد حقيقة في الأراضي، لذا كيف تعرف أنهم يعملون؟

- **Sample inspection**فقط انظر إلى 64 عينة في نهاية كل عصر. غير قابل للتفاوض.
- **FID (Fréchet Inception Distance)** المسافة بين التوزيعات المميزة في Inception-v3 للجماعات الحقيقية والمتولدة. أقل هو أفضل. معيار اجتماعي.
- **Inception Score** أكبر سناً، أكثر هشاشة؛ تفضل FID.
- **Precision/Recall for generative models** تقيس الجودة (الدقة) والغطاء (التذكير) بشكل منفصل.

لتحقيق بيانات صناعية صغيرة، فان التفتيش العينوي كافيا.

```figure
cv-gan-image
```

## بناءها

### الخطوة الأولى: مولد

مولد DCGAN صغير يأخذ ضجيج 64 أبعاد وينتج صورة 32x32.

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=64, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feat * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 4, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 2, feat, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))
```

أربعة قوائم نقل، كل منها مع`kernel_size=4, stride=2, padding=1`لذا فهي تضاعف حجم المكان بشكل نظيف. تنشيط الخروج في [-1, 1] عبر tanh.

### الخطوة الثانية: التمييز

مرآة المولّد، "المركبات المتدفقة" المتدفقة، تنتهي بـ "المناسبة"

```python
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feat, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 2, feat * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 4, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x).view(-1)
```

آخر مقعد يقلل من`4x4`خريطة الميزات إلى `1x1`. إنتاج واحد من مستوى الكمبيوتر لكل صورة، تطبيق sigmoid فقط أثناء حساب الخسارة.

### الخطوة الثالثة: خطوة التدريب

بديل: تحديث D مرة واحدة، ثم G مرة واحدة، كل دفعة.

```python
import torch.nn.functional as F

def train_step(G, D, real, z, opt_g, opt_d, device):
    real = real.to(device)
    bs = real.size(0)

    # D step
    opt_d.zero_grad()
    d_real = D(real)
    d_fake = D(G(z).detach())
    loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
              + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
    loss_d.backward()
    opt_d.step()

    # G step
    opt_g.zero_grad()
    d_fake = D(G(z))
    loss_g = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
    loss_g.backward()
    opt_g.step()

    return loss_d.item(), loss_g.item()
```

`G(z).detach()`في الخطوة D أمر حاسم: نحن لا نريد أن تدفق التراجعات إلى G خلال تحديثها. نسيان ذلك هو خطأ المبتدئين الكلاسيكي.

### الخطوة الرابعة: حلقة تدريب كاملة على الأشكال الاصطناعية

```python
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def synthetic_images(num=2000, size=32, seed=0):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((num, 3, size, size), dtype=np.float32) - 1.0
    for i in range(num):
        r = rng.uniform(6, 12)
        cx, cy = rng.uniform(r, size - r, size=2)
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        color = rng.uniform(-0.5, 1.0, size=3)
        for c in range(3):
            imgs[i, c][mask] = color[c]
    return torch.from_numpy(imgs)

device = "cuda" if torch.cuda.is_available() else "cpu"
data = synthetic_images()
loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=True)

G = Generator(z_dim=64, img_channels=3, feat=32).to(device)
D = Discriminator(img_channels=3, feat=32).to(device)
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(10):
    for (batch,) in loader:
        z = torch.randn(batch.size(0), 64, device=device)
        ld, lg = train_step(G, D, batch, z, opt_g, opt_d, device)
    print(f"epoch {epoch}  D {ld:.3f}  G {lg:.3f}")
```

`Adam(lr=2e-4, betas=(0.5, 0.999))`إذا كانت الوضع الاختيارية DCGAN  البيتا1 المنخفضة تبقي فترة الزخم من استقرار اللعبة المعارضة كثيرا.

### الخطوة 5: أخذ العينات

```python
@torch.no_grad()
def sample(G, n=16, z_dim=64, device="cpu"):
    G.eval()
    z = torch.randn(n, z_dim, device=device)
    imgs = G(z)
    imgs = (imgs + 1) / 2
    return imgs.clamp(0, 1)
```

دائماً انقل إلى وضع تقييم قبل أخذ العينات. بالنسبة لـ DCGAN هذا مهم لأن إحصاءات تشغيل معايير البطولة تستخدم بدلاً من إحصاءات البطولة.

### الخطوة 6: التطبيع الطيفي

بديل للـ BN في المتميز الذي يضمن الشبكة هو 1-Lipschitz. يصلح معظم فشل "D يفوز بقوة كبيرة".

```python
from torch.nn.utils import spectral_norm

def build_sn_discriminator(img_channels=3, feat=64):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(img_channels, feat, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat, feat * 2, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 2, feat * 4, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 4, 1, 4, 1, 0)),
    )
```

التغيير`Discriminator`لـ`build_sn_discriminator()`وغالباً ما لا تحتاج إلى خدعة TTUR. المعيار الطيفي هو أسهل تحديث واحد للقوة يمكنك تطبيقه.

## استخدمها

بالنسبة للإنتاج الجيد، استخدم الأوزان المُدربة مسبقاً أو أبدل التوزيع.

- `torch_fidelity`يحسب FID / IS على مولد الخاص بك دون كتابة رمز تقييم مخصص.
- `pytorch-gan-zoo`(الورث) و `StudioGAN`تم اختبار عمليات تنفيذ DCGAN و WGAN-GP و SN-GAN و StyleGAN و BigGAN على متن السفينة.

في عام 2026، لا تزال GANs هي أفضل خيار ل: إنتاج الصور في الوقت الحقيقي (التأخير <10 ms) ، ونقل الأسلوب، ترجمة الصورة إلى الصورة مع التحكم الدقيق (Pix2Pix، CycleGAN). ينتصر التوزيع على التصوير الصوري وتكييف النص.

## أرسله

هذا الدرس ينتج عن:

- `outputs/prompt-gan-training-triage.md` طلب يقرأ وصف منحنى التدريب ويحدد وضع الفشل (انهيار الوضع، D-فوز، التذبذب) بالإضافة إلى الإصلاح الموصى به.
- `outputs/skill-dcgan-scaffold.md`مهارة تكتب منصة DCGAN من`z_dim`، الهدف`image_size`و`num_channels`، بما في ذلك حلقة تدريب ومدفوعة العينات.

## التمارين

1. **(Easy)**قم بتدريب DCGAN أعلاه على مجموعة بيانات الدورات الاصطناعية وتحفظ شبكة من 16 عينة في نهاية كل عصر.
2. **(Medium)**استبدل قاعدة اللحظة من المتمييز مع قاعدة الطيف. تدريب كلا النسخين جنبا إلى جنب. أي واحد يتقارب أسرع؟ أي واحد لديه اختلاف أقل بين ثلاث بذور؟
3. **(Hard)**تنفيذ DCGAN مشروط: إدخال علامة الفئة في كل من G و D (تقاطع واحد حار إلى الضوضاء في G، وتقاطع قناة تضمين الفئة في D). تدريب على مجموعة بيانات "الدوائر مقابل المربع" الاصطناعية من الدروس 7 ووضح أن تطبيق التأهيل الفئة يعمل عن طريق أخذ العينات مع علامات محددة.

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Generator (G) | "The draws-stuff net" | Maps noise to images; trained to fool the discriminator |
| Discriminator (D) | "The critic" | Binary classifier; trained to distinguish real from generated images |
| Minimax | "The game" | min over G, max over D of an adversarial loss; equilibrium is p_G = p_data |
| Non-saturating loss | "The numerically sane version" | G's loss is -log(D(G(z))) instead of log(1 - D(G(z))) to avoid vanishing gradients early in training |
| Mode collapse | "Generator makes one thing" | G produces only a small subset of the data distribution; fix with SN, minibatch discrimination, or larger batch |
| TTUR | "Two learning rates" | D learns faster than G, typically by a factor of 2-4; stabilises training |
| Spectral norm | "1-Lipschitz layer" | A weight-normalisation that bounds each layer's Lipschitz constant; stops D from becoming arbitrarily steep |
| FID | "Fréchet Inception Distance" | Distance between Inception-v3 feature distributions of real and generated sets; the standard evaluation metric |

## المزيد من القراءة

- [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)-الورقة التي بدأت كل شيء
- [DCGAN (Radford, Metz, Chintala, 2015)](https://arxiv.org/abs/1511.06434)قواعد الهندسة المعمارية التي جعلت GAN تدريبية
- [Spectral Normalization for GANs (Miyato et al., 2018)](https://arxiv.org/abs/1802.05957) خدعة الاستقرار الوحيدة الأكثر فائدة
- [StyleGAN3 (Karras et al., 2021)](https://arxiv.org/abs/2106.12423)" سوتا غان " ، يقرأ مثل ألبوم أفضل أغاني من كل خدعة من العقد الماضي
