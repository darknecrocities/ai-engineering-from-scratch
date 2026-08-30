# المُحَرِّفات الذاتية والمتغيرات المُحَرِّفات الذاتية (VAE)

> يقوم جهاز تشفير السيارات العادي بتقليص ثم بإعادة تشكيل. يتذكر. لا يولد. أضف خدعة واحدة  اضغط على الرمز ليظهر غوسيان  وستحصل على عينة. هذه الخدعة الوحيدة، إعادة تشكيل`z = μ + σ·ε`ولهذا السبب كل نموذج تصوير متساوٍ للتنشر الخفيف والتي تستخدمها في عام 2026 لديه VAE عند المدخل.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 07 (CNNs), Phase 8 · 01 (Taxonomy)
**Time:** ~75 minutes

## المشكلة

قم بتضغط رقم من 784 بكسل على رمز 16 رقماً ثم قم بإعادة تشكيل. سوف يقوم جهاز تشكيل السيارات العادي بإعادة تشكيل MSE ولكن مساحة الرمز هي فوضى ضخمة. اختر نقطة عشوائية في مساحة الرمز، فكرها، وستحصل على ضجيج. ليس لديها عينة. إنه نموذج ضغط مرتدياً.

ما تريد حقا هو: (أ) مساحة الرمز هو صافي، وتوزيع سلس يمكنك أن تأخذ عينات من  قل غوسيانية متوازية `N(0, I)`(ب) تشفير أي عينة ينتج رقم معقول، و (ج) تشفير و تشفير لا يزال الضغط جيد. ثلاثة أهداف، بنية واحدة، خسارة واحدة.

حل VAE 2013 Kingma هذا عن طريق تدريب المُرمّد لإخراج * توزيع*`q(z|x) = N(μ(x), σ(x)²)`، سحب هذا التوزيع نحو البار`N(0, I)`عبر عقوبة KL، ثم أخذ العينات`z`من`q(z|x)`قبل فك الشفرة. في وقت الاستنتاج، إسقط المشفر، العينة `z ~ N(0, I)`عقوبة ك.إل هي ما يفرض على مساحة الرمز أن تكون مهيكلة.

في عام 2026 نادرًا ما تقوم VAEs بالشحن بمفردها  تم تفوقها من خلال التوزيع لجودة الصورة الخام  ولكنها هي مرموزة الاختيار لكل نموذج للتوزيع الخاطئ (SD 1/2/XL/3, Flux, AudioCraft). تعلم VAE وتتعلم الطبقة الأولى غير المرئية لكل خط أنابيب الصورة التي تستخدمها.

## المفهوم

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**Autoencoder.** `z = encoder(x)`،`x̂ = decoder(z)`، الخسارة = `||x - x̂||²`-المكان غير المهيكلي

**VAE encoder.**النتائج اثنين من المتجهات: `μ(x)`و`log σ²(x)`هذه تحدد`q(z|x) = N(μ, diag(σ²))`. . .

**Reparameterization trick.**أخذ العينات من`q(z|x)`لا يمكن التمييز. أعيد كتابة العينة على النحو`z = μ + σ·ε`أين`ε ~ N(0, I)`الآن`z`هو وظيفة تحديدية من`(μ, σ)`بالإضافة إلى ضجيج غير معين  تدفق التدرج `μ`و`σ`. . .

**Loss.**دليل (الربط السفلي) ، شروطتان:

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

إعادة الإعمار تدفع`x̂`نحو`x`. كيل يدفع`q(z|x)`في حين أن النقاط التجارية في البيانات الأولى من النقاط التجارية، تُصبح النقاط التجارية أكثر وضوحاً. تتداول بينها. β (<1) = عينات أكثر وضوحاً، فضاء رمز أقل غوسياً. β (>1) = مساحة رمز أكثر نظافة، عينات أكثر وضوحاً. β-VAE (Higgins 2017) جعلت هذه الزر مشهورة وبدأت بحوث التفريق.

**Sampling.**عند الاستنتاج: رسم`z ~ N(0, I)`لا يوجد عينة متكررة مثل التوزيع

```figure
vae-latent-grid
```

## بناءها

`code/main.py`يطبق VAE صغير دون نومبي أو مشعل. المدخل هو بيانات اصطناعية 8-dimensional استخدمت من مزيج غوسيان 2-كون في 8-D. Encoder و decoder MLPs واحد مخفي الطبقة. نطبق تنح تفعيل، المضي قدما، الخسارة، والمركبة الخلفية مكتوبة يدويا. ليس الإنتاج  التربية.

### الخطوة 1: المُشفّر إلى الأمام

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

`log σ²`بدلاً من`σ`لذلك إنتاج الشبكة غير مقيد (المضافة اللينة من σ هي فخ  تتوقف تراجعات في σ ≈ 0).

### الخطوة الثانية: إعادة تشكيل وتحديد المعدلات

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### الخطوة الثالثة: الـ ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

كيل بالضبط في شكل مغلق لأن كلا التوزيعين غوسيان. لا تتكامل عددا. الناس لا يزالون يرسلون الرمز مع مونت كارلو تقديرات كيل في 2026  هو بطيء 3x دون سبب.

### الخطوة الرابعة: توليد

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

هذا هو النموذج التوليد خمسة خطوط

## الفخاخ

- **Posterior collapse.**محركات قيادة المدة`q(z|x) → N(0, I)`بفعالية كبيرة`z`لا يحمل أي معلومات عن`x`. إصلاح: التخلّص من β (بدء β=0، الرّامب إلى 1), التخلّص من البط أو تخطي KL في الأبعاد غير النشطة.
- **Blurry samples.**احتمالات القيادة الجاوسي تشير إلى إعادة بناء MSE ، وهو مطلوب بايز ل L2 (المتوسط)  متوسط مجموعة من الأرقام المثيرة للصدق هو رقم غامض. تحديد: القيادة المفصلة (VQ-VAE ، NVAE) ، أو استخدام VAE فقط كمُخترع وتوزيع كومة على الاختفاء (هذا ما يفعله Diffusion Stable).
- **β too large, too early.**انظر الانهيار الخلفي، ابدأ عند β≈0.01 ومرحلة
- **Latent dim too small.**يعمل 16-D لـ MNIST ، 256-D لـ ImageNet 2562 ، 2048-D لـ ImageNet 10242. VAE من Stable Diffusion يضغط على 512 × 512 × 3 → 64 × 64 × 4 (32x عامل النموذج إلى أسفل في المساحة الفضائية ، 32x في القنوات).

## استخدمها

مجموعة "2026" في إيران:

| Situation | Pick |
|-----------|------|
| Image-latent encoder for diffusion | Stable Diffusion VAE (`sd-vae-ft-ema`) or Flux VAE |
| Audio-latent encoder | Encodec (Meta), SoundStream, or DAC (Descript) |
| Video latents | Sora's spatiotemporal patches, Latte VAE, WAN VAE |
| Disentangled representation learning | β-VAE, FactorVAE, TCVAE |
| Discrete latents (for transformer modelling) | VQ-VAE, RVQ (ResidualVQ) |
| Continuous latents for generation | Plain VAE, then condition a flow/diffusion model in that latent space |

نموذج الانتشار الخفيف هو VAE مع نموذج الانتشار يعيش بين المُشفّر والمنشط. يقوم VAE بالضغط الخام، يقوم نموذج الانتشار بالرفع الثقيل. نفس النموذج للفيديو (VAE + DiT) والصوت (Encodec + MusicGen Transformer).

## أرسله

إنقاذ`outputs/skill-vae-trainer.md`. . .

المهارات: ملف مجموعة البيانات + هدف التخفيض المتخفيض + استخدامها في المستقبل (إعادة الإعمار أو أخذ العينات أو مدخل التوزيع المتخفيض) والخروج: اختيار الهندسة المعمارية (بطنة / β / VQ / RVQ) ، جدول β ، التخفيض المتخفيض ، احتمالات المقاطع (غوسيان مقابل فئة) ، وخطة التقييم (تعريف MSE ، KL لكل ضوء ، مسافة Fréchet بين `q(z|x)`و`N(0, I)`)

## التمارين

1. **Easy.**التغيير`β`في`code/main.py`إلى`0.01`،`0.1`،`1.0`،`5.0`.سجل إعادة الإعمار النهائية MSE و KL. أي β هو أفضل Pareto لبياناتك الاصطناعية؟
2. **Medium.**استبدل احتمالات القيادة الجاوسي بعقيدة بيرنولي (خسارة الانتروبيا المتقاطعة). مقارنة نوعية العينة على نسخة ثنائية من نفس البيانات الاصطناعية.
3. **Hard.**التمديد`code/main.py`إلى VQ-VAE صغير: استبدال المستمر `z`مع البحث عن أقرب جيران في كتاب كود من إدخالات K=32. مقارنة إعادة الإعمار MSE وتقرير عدد إدخالات كودك استخدام (انهيار كودكوب حقيقي).

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Autoencoder | Encode-decode network | `x → z → x̂`, learn MSE. Not generative. |
| VAE | AE with a sampler | Encoder outputs a distribution, KL penalty shapes code space. |
| ELBO | Evidence lower bound | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`; tight when `q = p(z\|x)`. |
| Reparameterization | `z = μ + σ·ε` | Rewrites stochastic node as deterministic + pure noise. Enables backprop through sampling. |
| Prior | `p(z)` | Target distribution for the latent, typically `N(0, I)`. |
| Posterior collapse | "KL term wins" | Encoder ignores `x`, outputs the prior; decoder must hallucinate. |
| β-VAE | Tunable KL weight | `loss = recon + β·KL`. Higher β = more disentangled but blurrier. |
| VQ-VAE | Discrete latent | Replace continuous `z` with nearest codebook vector; enables transformer modelling. |

## ملاحظة الإنتاج: الممر المكثف هو أكثر مسار حرارة في خادم التوزيع

في خط أنابيب Diffusion / Flux / SD3 المستقر يتم استدعاء VAE مرتين لكل طلب  مرة واحدة لترميز (إذا كنت تفعل img2img / inpainting) ومرة واحدة لترميز. في 10242 مرسلة المقرر غالبا ما تكون أكبر ذروة تنشيط ذاكرة واحدة في خط الأنابيب كله لأنه يظهر`128×128×16`الاختفاءات العودة إلى`1024×1024×3`اثنين من العواقب العملية:

- **Slice or tile the decode.** `diffusers`يُعرض`pipe.vae.enable_slicing()`و`pipe.vae.enable_tiling()`تيلينغ تجارية صغيرة القطع الأثرية للخياطة`O(tile²)`الذاكرة بدلاً من`O(H·W)`. ضروري لـ 10242+ على أجهزة البيانات المعالجة البيانية المستهلكة
- **bf16 decoder, fp32 numerics for the final resize.**تم إطلاق SD 1.x VAE في fp32 و * ينتج بصمت NaNs * عند إلقاء إلى fp16 في 10242 + سفن SDXL `madebyollin/sdxl-vae-fp16-fix` تفضل دائماً الإصدار المثبت fp16-fix أو استخدم bf16.

## المزيد من القراءة

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) ورقة VAE
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) التفريق بين β- VAE
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) VQ-VAE
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) صورة حديثة في الفن
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) انتشار مستقيم؛ VAE كمُخترف.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)إنكودك، معيار الصوت في إيه إيه
