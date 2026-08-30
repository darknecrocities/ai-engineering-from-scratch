# التوزيع المستقر  الهندسة المعمارية والتنظيم

> Diffusion Stable هو DDPM الذي يعمل في الفضاء الخفيف ل VAE المدرب مسبقا، مشروط على النص عن طريق الانتباه المتقاطع، ومعينة مع حلول ODE تحديدية سريعة، وتحكم من خلال توجيه خالية من المصنف.

**Type:** Learn + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 10 (Diffusion), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~75 minutes

## أهداف التعلم

- تتبع خمسة قطع من خط الأنابيب المتواصلة: VAE، مرموز النص، U-Net، المجدول، فحص السلامة  وما تفعله كل منها في الواقع
- شرح التوزيع المتخفي ولماذا تدريب في مساحة متخفية 4x64x64 (بدلاً من صورة 3x512x512) يقلل من الحساب بنسبة 48x دون فقدان الجودة
- استخدام`diffusers`لتوليد الصور، وتشغيل الصورة إلى الصورة، والرسومات الداخلية، وتوليد القيادة من ControlNet
- التنسيق الدقيق التنفيض المستقر مع LoRA على مجموعة بيانات مخصصة صغيرة و تحميل مكيّف LoRA عند الاستنتاج

## المشكلة

تدريب DDPM مباشرة على صور RGB 512x512 مكلف. كل خطوة تدريبية تراجع من خلال شبكة U-Net التي ترى قيم المدخل 3x512x512 = 786,432، وتأخذ العينات 50+ إلى الأمام تمر عبر نفس شبكة U-Net. في مستوى جودة Stable Diffusion 1.5 (صادر في 2022) ، فإن انتشار الفضاء البيكسل يحتاج إلى حوالي 256 شهرًا من التدريب في GPU و 10-30 ثانية لكل صورة على GPU المستهلك.

الخدعة التي جعلت النص الصور المفتوحة مميزة كانت**latent diffusion**(Rombach et al., CVPR 2022). تدريب VAE الذي يرسم صورة 3x512x512 إلى 10x64x64 مضغوطة متخفية وراء، ثم القيام بالانشطاعات في هذا الفضاء المتخفية.`(3*512*512)/(4*64*64) = 48x`.تراجع معينة من عشرات الثواني إلى أقل من ثواني على نفس المصفوفة المعالجة

تقريبا كل نموذج جديد لتوليد الصور  SDXL، SD3، FLUX، HunyuanDiT، Wan-Video  هو نموذج انتشار متخفي مع اختلافات على المُحافظ الذاتي، والمنطق (U-Net أو DiT) ، وتكييف النص. تعلم Diffusion مستقر وقد تعلمت القالب.

## المفهوم

### خط الأنابيب

```mermaid
flowchart LR
    TXT["Text prompt"] --> TE["Text encoder<br/>(CLIP-L or T5)"]
    TE --> CT["Text<br/>embedding"]

    NOISE["Noise<br/>4x64x64"] --> UNET["UNet<br/>(denoiser with<br/>cross-attention<br/>to text)"]
    CT --> UNET

    UNET --> SCHED["Scheduler<br/>(DPM-Solver++,<br/>Euler)"]
    SCHED --> LATENT["Clean latent<br/>4x64x64"]
    LATENT --> VAE["VAE decoder"]
    VAE --> IMG["512x512<br/>RGB image"]

    style TE fill:#dbeafe,stroke:#2563eb
    style UNET fill:#fef3c7,stroke:#d97706
    style SCHED fill:#fecaca,stroke:#dc2626
    style IMG fill:#dcfce7,stroke:#16a34a
```

- **VAE** مُجمدة مُصطدرة ذاتية. مُصطدرة تحوّل الصورة إلى مخططات (مستخدمة في img2img والتدريب). مُصطدرة تحوّل المخططات إلى صورة.
- **Text encoder** رمز نص CLIP (SD 1.x/2.x) ، CLIP-L + CLIP-G (SDXL) ، أو T5-XXL (SD3/FLUX). ينتج سلسلة من التوابل الرمزية.
- **U-Net** المُعَلِّم. لديه طبقات إنتباه متقاطعة تُحضر من اللاتنت إلى النص المُضمن في كل مستوى قرار.
- **Scheduler**خوارزمية أخذ العينات (DDIM، Euler، DPM-Solver++). يختار sigmas، يجمع الضوضاء المتوقعة مرة أخرى إلى اللاتين.
- **Safety checker** مرشح NSFW / محتويات غير قانونية اختياري على الصورة المصدرة.

### الإرشادات الخالية من التصنيف (CFG)

التعلم في تكوين النص البسيط `epsilon_theta(x_t, t, c)`لكل طلب`c`. CFG تدرب نفس الشبكة مع `c`انخفضت 10% من الوقت (حل محلها بتوابل فارغة) ، مما يعطي نموذج واحد يتوقع كل من الضجيج المشروط وغير المشروط.

```
eps = eps_uncond + w * (eps_cond - eps_uncond)
```

`w`هو مقياس التوجيه`w=0`غير مشروط`w=1`هو مشروط`w>1`يدفع الإنتاج نحو "أكثر مشروطة على الإستعراض" على حساب التنوع.`w=7.5`. . .

CFG هو السبب في أن النص إلى الصورة يعمل على جودة الإنتاج. بدونها، تحذير المخرج ضعيفًا؛ مع ذلك، تحكم التحذيرات.

### هندسة الفضاء المتخفية

الرقم المتخفي لـ 4 قنوات في إيه اي ليس مجرد صورة مضغوطة إنها مجموعة متنوعة حيث تتوافق الحسابات تقريبًا مع التحريرات التفاصلية (هناك هندسة السرعة + التقاطع كلاهما يعيش هنا) ، حيث تم تدريب شبكة U-Net للتنشر على إنفاق ميزانية النمذجة بأكملها. لا ينتج تشخيص 4x64x64 latente عشوائية صورة تبدو عشوائية  فإنه ينتج القمامة، لأن فقط مجموعة معينة من الاختفاءات تشخيص الصور الصالحة.

عواقب:

1. **Img2img**= تشفير الصورة إلى غامضة، إضافة ضجيج جزئي، تشغيل المُعبر، فك تشفير. ينجو بنية الصورة لأن التشفير شبه قابلة للتحويل. يتغير المحتوى بناءً على الإشارة.
2. **Inpainting**= نفس img2img ولكن المحدد يحدد فقط المناطق المخفية؛ المناطق غير المخفية يتم الاحتفاظ بها في الخرق المشفر.

### بنية شبكة الإنترنت

إن شبكة SD U-Net هي نسخة كبيرة من TinyUNet من الدروس 10 مع ثلاثة إضافات:

- **Transformer blocks**في كل قرار مساحي، يحتوي على الاهتمام الذاتي + الاهتمام المتبادل بالنص المضمن.
- **Time embedding**عبر MLP على تشفير السينوسويدي
- **Skip connections**بين المُرمّد والمُرمّد عند قرارات متطابقة.

مجموع المعلمات في SD 1.5: ~ 860M. SDXL: ~ 2.6B. FLUX: ~ 12B. قفزة في المعلمات هي في الغالب في طبقات الاهتمام.

### تحديدات الحجم

يحتاج التنسيق الكامل للتسريب المستقر إلى 20+ جيجابايت من VRAM ويحديث 860 مليون برمجة. LoRA (التكيف منخفض الرتب) يبقي النموذج الأساسي مجمدًا ويضرب matrices صغيرة من الدرجة التفكيك في طبقات الاهتمام. يعتبر مكيّف LoRA لـ SD عادةً 10-50 MB ، يتدرب في 10-60 دقيقة على GPU لمستهلك واحد ، ويتحمّل في وقت الاستنتاج كتحديث يضعف.

```
Original: W_q : (d_in, d_out)   frozen
LoRA:     W_q + alpha * (A @ B)   where A : (d_in, r), B : (r, d_out)

r is typically 4-32.
```

(لورا) هي الطريقة التي يتم بها توزيع كل الموسيقى المحددة في المجتمع تقريباً (سيفيتاي) و (تقبيل الوجه) يستضيفون ملايين منها

### المواعيد التي ستراها

- **DDIM** تحديد، ~ 50 خطوة، بسيطة.
- **Euler ancestral** مستويات مستويات 30 إلى 50 خطوة، عينات أكثر إبداعاً قليلاً
- **DPM-Solver++ 2M Karras** تحديد، 20 إلى 30 خطوة، افتراض الإنتاج.
- **LCM / TCD / Turbo** نماذج التواصل والفرازات المقطوعة؛ 1-4 خطوات على حساب نوعية.

تغيير المخططات هو تغيير واحد خط في `diffusers`وأحياناً تصحيح مشاكل العينات دون أي إعادة التدريب.

```figure
cv3-latent-compression
```

## بناءها

هذا الدرس يستخدم`diffusers`نهاية إلى نهاية بدلا من إعادة بناء Diffusion Stable من الصفر. الأجزاء التي ستحتاج إلى إعادة بناءها (VAE ، مرموز النص ، U-Net ، المجدول) هي مواضيع دروسهم الخاصة. هنا الهدف هو السهولة مع API الإنتاج.

### الخطوة الأولى: النص إلى الصورة

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe(
    prompt="a dog riding a skateboard in tokyo, studio ghibli style",
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("dog.png")
```

`float16`يقلل من نصف الـ VRAM دون فقدان نوعية مرئي. `num_inference_steps=25`مع تطابقات DPM-Solver++ الافتراضية `num_inference_steps=50`مع DDIM.

### الخطوة الثانية: تغيير الموعد

```python
from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
```

حالة المخطط منفصلة عن أوزان شبكة U. يمكنك التدريب على DDPM ومعينة مع أي المخطط.

### الخطوة الثالثة: صورة إلى صورة

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

init_image = Image.open("dog.png").convert("RGB").resize((512, 512))
out = img2img(
    prompt="a dog riding a skateboard, oil painting",
    image=init_image,
    strength=0.6,
    guidance_scale=7.5,
).images[0]
```

`strength`كمية الضوضاء التي يجب إضافةها قبل التخلص (0.0 = غير متغيرة، 1.0 = إعادة التأهيل الكاملة).

### الخطوة الرابعة: إدلاء الطلاء

```python
from diffusers import StableDiffusionInpaintPipeline

inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
).to("cuda")

image = Image.open("dog.png").convert("RGB").resize((512, 512))
mask = Image.open("dog_mask.png").convert("L").resize((512, 512))

out = inpaint(
    prompt="a cat",
    image=image,
    mask_image=mask,
    guidance_scale=7.5,
).images[0]
```

البيكسلات البيضاء في القناع هي المنطقة التي يجب تجديدها. البيكسلات السوداء يتم الحفاظ عليها.

### الخطوة 5: تحميل لورا

```python
pipe.load_lora_weights("sayakpaul/sd-lora-ghibli")
pipe.fuse_lora(lora_scale=0.8)

image = pipe(prompt="a village square in ghibli style").images[0]
```

`lora_scale`يحدد قوة؛ 0.0 = لا تأثير، 1.0 = تأثير كامل. `fuse_lora`يخبز المعدل إلى الأوزان الموضحة لسرعة، ولكن يمنع التبادل.`pipe.unfuse_lora()`قبل تحميل جهاز تعديل مختلف

### الخطوة 6: تدريب لوري (رسم)

تدريب حقيقي لـ " لورا " يعيش في`peft`أو`diffusers.training`. المخطط:

```python
# Pseudocode
for step, batch in enumerate(dataloader):
    images, prompts = batch
    latents = vae.encode(images).latent_dist.sample() * 0.18215

    t = torch.randint(0, num_train_timesteps, (batch_size,))
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, t)

    text_emb = text_encoder(tokenizer(prompts))

    pred_noise = unet(noisy_latents, t, text_emb)  # LoRA weights injected here

    loss = F.mse_loss(pred_noise, noise)
    loss.backward()
    optimizer.step()
```

تتلقى المصفوفات LoRA فقط تراجعًا؛ يتم تجميد U-Net و VAE ومدفوع النص. مع حجم اللحظة 1 ومراقبة تراجعية يتناسب مع 8 جيجابايت من VRAM.

## استخدمها

في الإنتاج، القرارات التي تتخذها فعلياً:

- **Model family**: SD 1.5 لموسيقى مجتمع مفتوحة المصدر ، SDXL لتحقيق أعلى ، SD3 / FLUX للحصول على أحدث وتطلبات الترخيص الصارمة.
- **Scheduler**: DPM-Solver++ 2M Karras لمدة 20-30 خطوة، LCM-LoRA عندما يكون التأخير أقل من 1 ثانية.
- **Precision**: `float16`في 4080/4090`bfloat16`على A100 وأحدث،`int8`(بـ (`bitsandbytes`أو`compel`) عندما تكون VRAM ضيقة.
- **Conditioning**: يعمل النص البسيط؛ لإضافة ControlNet (قوة، عمق، وضع) على رأس خط الأنابيب الأساسي لتحسين التحكم.

لإنتاج اللحوم`AUTO1111`- لا ، لا`ComfyUI`هي أدوات المجتمع؛ لخدمات الإنتاج،`diffusers`+ `accelerate`أو`optimum-nvidia`مع تجميع TensorRT.

## أرسله

هذا الدرس ينتج عن:

- `outputs/prompt-sd-pipeline-planner.md` استشارة تختار SD 1.5 / SDXL / SD3 / FLUX بالإضافة إلى الجدول المحدد والدقة بالنظر إلى ميزانية التأخير والهدف الأمني والقيود المفروضة على الترخيص.
- `outputs/skill-lora-training-setup.md` مهارة تكتب إعداد تدريب كامل لـ LoRA لمجموعة بيانات مخصصة بما في ذلك العناوين الرئيسية والرتبة وحجم اللحظة وتيرة التعلم.

## التمارين

1. **(Easy)**أخلق نفس الإشارة مع `guidance_scale`في`[1, 3, 5, 7.5, 10, 15]`.أوصف كيف تتغير الصورة .بأي قيمة توجيهية تظهر الأثاث ؟
2. **(Medium)**خذ أي صورة حقيقية، فحصها`StableDiffusionImg2ImgPipeline`في`strength`في`[0.2, 0.4, 0.6, 0.8, 1.0]`ما هي القوة التي تحافظ على التركيب مع تغيير النمط؟ لماذا يتجاهل 1.0 المدخل بالكامل؟
3. **(Hard)**قم بتدريب LoRA على 10-20 صورة لموضوع واحد (حيوان أليف أو شعار أو شخصية) وخلق مشاهد جديدة مع هذا الموضوع فيها. قم بتقرير ترتيب LoRA وخطوات التدريب التي أنتجت أفضل الحفاظ على الهوية دون إعادة التكيف مع الصور المدخنة.

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Latent diffusion | "Diffuse in latents" | Run the entire DDPM in the VAE latent space (4x64x64) instead of pixel space (3x512x512); 48x compute saving |
| VAE scale factor | "0.18215" | Constant that rescales the VAE's raw latent to roughly unit variance; hardcoded in every SD pipeline |
| Classifier-free guidance | "CFG" | Mix conditional and unconditional noise predictions; the single most impactful inference knob |
| Scheduler | "Sampler" | The algorithm that turns noise + model predictions into a denoised latent trajectory |
| LoRA | "Low-rank adapter" | Small rank-decomposition matrices that fine-tune attention layers without touching base weights |
| Cross-attention | "Text-image attention" | Attention from latent tokens to text tokens; injects prompt information at every U-Net level |
| ControlNet | "Structure conditioning" | A separately-trained adapter that steers SD with an extra input (canny, depth, pose, segmentation) |
| DPM-Solver++ | "The default scheduler" | Second-order deterministic ODE solver; best quality at low step counts (20-30) in 2026 |

## المزيد من القراءة

- [High-Resolution Image Synthesis with Latent Diffusion (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752) ورقة التفريق المستقر؛ يتضمن كل إزالة تبرير التصميم
- [Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)](https://arxiv.org/abs/2207.12598) ورقة CFG
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) كانت لورا أول من قام بتطبيق النظام النووي؛ تم نقلها إلى SD دون أي تغيير تقريباً
- [diffusers documentation](https://huggingface.co/docs/diffusers) الإشارة لكل خط أنابيب SD / SDXL / SD3 / FLUX
