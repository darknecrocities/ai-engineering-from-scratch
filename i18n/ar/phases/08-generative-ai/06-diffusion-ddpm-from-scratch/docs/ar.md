# نماذج التوزيع  DDPM من الصفر

> أعطى هو، جين، أبيبيل (2020) المجال وصفة لم يستطع التوقف عنها. تدمير البيانات مع الضوضاء على أكثر من ألف خطوة صغيرة. تدرب شبكة عصبية واحدة للتنبؤ بالضوضاء. عكس العملية عند الاستنتاج. اليوم كل صورة رئيسية، فيديو، 3D، ونموذج الموسيقى يعمل على هذا الحلقة، ربما مع مطابقة التدفق أو خدوش التوافق على القمة.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 8 · 02 (VAE)
**Time:** ~75 minutes

## المشكلة

هل تريد عينة ل`p_data(x)`. يلعب GANs لعبة minimax التي تختلف غالباً. تنتج VAEs عينات مشمسة من جهاز تشخيص غوس. ما تريد حقاً هو هدف تدريب هو (أ) خسارة مستقرة واحدة (لا نقطة مقعد، لا حد أدنى)`log p(x)`(حيث لديك احتمالات) ، و (ج) عينات تتطابق مع جودة SOTA.

Sohl-Dickstein et al. (2015) كان لديه إجابة نظرية: تحديد سلسلة ماركوف `q(x_t | x_{t-1})`الذي يضيف تدريجيا ضجيج غوسيان، وتدريب سلسلة عكسية`p_θ(x_{t-1} | x_t)`لتنفيذ التخريب. هو، جين، أبيل (2020) أظهر أن الخسارة يمكن تبسيطها إلى سطر واحد  توقع الضوضاء  ونظف الرياضيات. في عام 2020 كانت هذه فضولية. في عام 2021 أنتجت عينات متطورة. في عام 2022 أصبحت Diffusion مستقرة. في عام 2026 هي الأساس.

## المفهوم

![DDPM: forward noise, reverse denoise](../assets/ddpm.svg)

**Forward process `q`.**إضافة ضجيج غوسيان `T`الخطوات الصغيرة. النموذج المغلق  السبب في أن الرياضيات قابلة للتحكم  هو أن الخطوة التراكمية هي أيضا غوسية:

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

أين`α̅_t = ∏_{s=1..t} (1 - β_s)`لجدول زمني `β_t`. اختر`β_t`من 1e-4 إلى 0.02 خطيا على T=1000 خطوات و `x_T`هو تقريبا`N(0, I)`. . .

**Reverse process `p_θ`.**تعلم شبكة عصبية`ε_θ(x_t, t)`الذي يتوقع الضجيج الذي تم إضافة.`x_t`، يُعَلّم بـ:

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

أين`σ_t`هو إما`sqrt(β_t)`أو التباين المكتشف. التعبير قبيح ولكن هو مجرد الجبر  الحل ل `x_{t-1}`نظراً للخلفية`q(x_{t-1} | x_t, x_0)`وبدل`x_0`مع تقديرها المتوقع للضوضاء.

**Training loss.**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

العينة`x_0`من البيانات، اختر عشوائي `t`، عينة`ε ~ N(0, I)`، حساب الضوضاء`x_t`في إطلاق واحد عبر الشكل المغلق، والعودة على الضجيج. خسارة واحدة، لا تقليص، لا KL، لا خدوش إعادة التقييم.

**Sampling.**ابدأ`x_T ~ N(0, I)`. أعيد الخطوة العكسية من`t = T`إلى`1`-أجري

## لماذا يعمل

ثلاث بديهيات:

1. **Denoising is easy; generating is hard.**في`t=T`، البيانات هي ضجيج خالص  الشبكة لديها لحل مشكلة بسيطة.`t=0`، الشبكة فقط تحتاج إلى تنظيف بضع بكسلات.`t`المشكلة صعبة ولكن الشبكة لديها العديد من التدرج التي تتدفق من خلال الوزن نفسه من كل مستوى الضوضاء.

2. **Score matching in disguise.**(فنسنت) (2011) أثبت أن التنبؤ بالضوضاء يعادل التقدير`∇_x log q(x_t | x_0)`يستخدم SDE العكسية هذه النتيجة للمشي فوق تراجع الكثافة  المشي العشوائي الموجّه نحو مناطق احتمالية عالية.

3. **The ELBO reduces to simple MSE.**الحد السفلي المتغير الكامل لديه مصطلح KL لكل خطوة زمنية. مع تخصيص DDPM ، تبسيط هذه المصطلحات KL إلى MSE على تنبؤ الضوضاء مع معايير محددة. هوه خفض المعايير (سيدعوها "سوائل" الخسارة) والجودة *تحسن*.

```figure
diffusion-denoise
```

## بناءها

`code/main.py`يطبق DDPM 1D. البيانات هي مزيج من وضعين. "الشبكة" هي MLP صغيرة التي تأخذ`(x_t, t)`التدريب هو الخسارة ذات الخط واحد.

### الخطوة الأولى: الجدول الزمني المسبق (الشكل المغلق)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### الخطوة الثانية: العينة`x_t`في إطلاق واحد

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### الخطوة الثالثة: خطوة تدريبية واحدة

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### الخطوة الرابعة: أخذ العينات العكسية

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

بالنسبة لمشكلة 1-D مع 40 خطوة زمنية و 24 وحدة MLP، هذا يتعلم خليط الحالة الثنائية في ~ 200 عصر.

## تكييف الوقت

الشبكة تحتاج إلى معرفة الخطوة الزمنية التي تقوم بتخفيفها. خيارين قياسيين:

- **Sinusoidal embedding.**مثل ترانسفورمير التشفير الموقع.`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`-مر عبر جهاز "ميل لوين" ، وترسل إلى الشبكة
- **Film / group-norm conditioning.**إضافة المشروع إلى نطاق/تجاه في كل قناة (FiLM) في كل كتلة.

رمز اللعب لدينا يستخدم مخططات الحبل الصناعي → مخططات الإنتاج U-Nets استخدام FiLM.

## الفخاخ

- **Schedule matters a lot.**خطية`β`هو DDPM الافتراضي ولكن جدول كوسين (نيكل و دريوال، 2021) يعطي أفضل FID لنفس الحساب.
- **Timestep embedding is fragile.**يمرون بالخام`t`كما يعمل العائم لعبة 1-D ولكن يفشل في الصور؛ استخدم دائما إضافة مناسبة.
- **V-prediction vs ε-prediction.**بالنسبة للأنظمة الضيقة (ت صغيرة جدا أو كبيرة جدا) ،`ε`لديه ضعف في إشارة إلى الضجيج.`v = α·ε - σ·x`) أكثر استقراراً؛ تستخدمها SDXL، SD3 وFlux.
- **Classifier-free guidance.**عند الاستنتاج، حساب كل من المشروط وغير المشروط `ε`، ثم`ε_cfg = (1 + w) · ε_cond - w · ε_uncond`مع`w ≈ 3-7`. تم تغطيتها في الدروس 08.
- **1000 steps is a lot.**يستخدم الإنتاج DDIM (20-50 خطوة) ، DPM-Solver (10-20 خطوة) ، أو التقطير (1-4 خطوة). انظر الدروس 12.

## استخدمها

| Role | Typical stack in 2026 |
|------|-----------------------|
| Image pixel-space diffusion (small, toy) | DDPM + U-Net |
| Image latent diffusion | VAE encoder + U-Net or DiT (Lesson 07) |
| Video latent diffusion | Spatiotemporal DiT (Sora, Veo, WAN) |
| Audio latent diffusion | Encodec + diffusion transformer |
| Science (molecules, proteins, physics) | Equivariant diffusion (EDM, RFdiffusion, AlphaFold3) |

التوزيع هو العمود الفقري التوليدي العالمي. تطابق التدفق (الدرس 13) هو منافس 2024-2026 الذي يفوز عادةً بسرعة الاستنتاج لنفس الجودة.

## أرسله

إنقاذ`outputs/skill-diffusion-trainer.md`. تتخذ مهارة مجموعة بيانات + ميزانية الحساب والخروج: الجدول الزمني (الخطوي / الكوسين / السجمايد) ، والهدف التنبؤ (ε / v / x) ، وعدد الخطوات ، ومقياس التوجيه ، وأسرة العينات ، وبروتوكول تقييم.

## التمارين

1. **Easy.**تغيير T من 40 إلى 10 `code/main.py`كيف تتدهور جودة العينة (الهيستومغرام المرئي للمخرجات) ؟
2. **Medium.**انقل من التنبؤ إلى التنبؤ إلى التنبؤ إلى التنبؤ
3. **Hard.**إضافة إرشادات خالية من المصنفات.`c ∈ {0, 1}`، انخفضها 10٪ من الوقت أثناء التدريب، وفي وقت أخذ العينات استخدام `ε = (1+w)·ε_cond - w·ε_uncond`. قياس معدل ضربات الوضع المشروط عند `w = 0, 1, 3, 7`. . .

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Forward process | "Adding noise" | Fixed Markov chain `q(x_t \| x_{t-1})` that destroys the data. |
| Reverse process | "Denoising" | Learned chain `p_θ(x_{t-1} \| x_t)` that reconstructs the data. |
| β schedule | "The noise ladder" | Per-step variance; linear, cosine, or sigmoid. |
| α̅ | "Alpha bar" | Cumulative product `∏(1 - β)`; gives closed-form `x_t` from `x_0`. |
| Simple loss | "MSE on noise" | `\|\|ε - ε_θ(x_t, t)\|\|²`; all variational derivations collapse to this. |
| ε-prediction | "Predict noise" | Output is the noise added; standard DDPM. |
| V-prediction | "Predict velocity" | Output is `α·ε - σ·x`; better conditioning across t. |
| DDPM | "The paper" | Ho et al. 2020; linear β, 1000 steps, U-Net. |
| DDIM | "Deterministic sampler" | Non-Markov sampler, 20-50 steps, same training objective. |
| Classifier-free guidance | "CFG" | Mix conditional and unconditional noise predictions to amplify conditioning. |

## ملاحظة الإنتاج: استنتاج التوزيع هو مشكلة الحد الأدنى

ورقة DDPM تعمل T=1000 خطوات عكسية. لا أحد يرسل ذلك في الإنتاج. كل كومة استنتاج حقيقية تختار واحدة من ثلاثة استراتيجيات  وكل خريطة نظيفة إلى إطار الإنتاج من "من أين يأتي التأخير":

1. **Faster sampler, same model.**DDIM (20-50 خطوة) ، DPM-Solver++ (10-20) ، UniPC (8-16). استبدال القفز في الحلقة العكسية؛ المدربين `ε_θ`الوزن غير متأثر، يقلل من التأخير 20 إلى 50 مرة
2. **Distillation.**تدريب الطالب على مطابقة المعلم في أقل خطوات: التملية التدريجية (2 → 1) ، نماذج التوافق (شخصية → 1-4) ، LCM ، SDXL-Turbo ، SD3-Turbo. يقلل من التأخير 5-10 × آخر ، يتطلب إعادة التدريب.
3. **Caching and compilation.** `torch.compile(unet, mode="reduce-overhead")`، خلفيات انتشار TensorRT-LLM ،`xformers`الاهتمام SDPA، وزن bf16. يقلل من التأخير في كل خطوة ~ 2 ×. يجمع مع (1) و (2).

بالنسبة لخادم انتشار الإنتاج، فإن محادثة الميزانية هي نفس ما وصفته الأدب الإنتاجية لـ LLM:`num_steps × step_cost + VAE_decode`، التدفق هو`batch_size × (num_steps × step_cost)^-1`. TTFT صغير (خطوة واحدة) ؛ TPOT يعادل وقت الاستجابة الكاملة لأن إنتاج الصورة "كل مرة واحدة" من منظور المستخدم.

## المزيد من القراءة

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)ورقة التوزيع، قبل وقتها.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) DDPM
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)-الـ DDIM، أقل خطوات
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672)جدول الكوسين، التباين المتعلم.
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) إرشادات المصنف
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) CFG
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364)-ملاحظة موحدة، وصفة نظيفة
