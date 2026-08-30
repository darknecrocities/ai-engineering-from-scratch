# أزياء العالم وتوزيع الفيديو

> نموذج فيديو يتنبأ بالثواني القادمة من المشهد هو محاكاة العالم. تحديد ذلك التنبؤ على الإجراءات و لديك محرك لعبة تعلم.

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 10 (Diffusion), Phase 4 Lesson 12 (Video Understanding), Phase 4 Lesson 23 (DiT + Rectified Flow)
**Time:** ~75 minutes

## أهداف التعلم

- شرح الفرق بين نموذج توليد الفيديو النقي (Sora 2) ونموذج العالم المتحكم في العمل (Genie 3 ، DreamerV3)
- وصف فيديو DiT: معطيات الفضاء-الوقت، تشفير الموقف ثلاثي الأبعاد، الاهتمام المشترك عبر رموز (T، H، W)
- تتبع كيفية توصيل نموذج عالمي إلى الروبوتات: خطط VLM → نموذج الفيديو يحاكي → ديناميكيات عكسية تنبعث عن أفعال
- اختيار بين سورا 2، جني 3، رينواي GWM-1 العالمات، وان فيديو، وحنيوانفيديو للحصول على حالة استخدام معينة (فيديو إبداعي، سيم التفاعلي، التوليد الحكم الذاتي القيادة)

## المشكلة

إنتاج الفيديو و نموذج العالم يتقارب في عام 2026. نموذج يمكنه توليد دقيقة متماسكة من الفيديو، تعلم، بطريقة ما، كيف يتحرك العالم: استمرارية الأشياء، الجاذبية، السببية، النمط. إذا قمت بتشريط ذلك التنبؤ على الإجراءات (مشي يساراً، فتح الباب) ، يصبح نموذج الفيديو محاكاة قابلة للتعلم يمكن استبدال محرك اللعبة، محاكاة القيادة، أو بيئة الروبوتات.

المراهنات ملموسة جيني 3 تولد بيئات قابلة للعب من صورة واحدة. "المساحة المقصودة "جي.إم.إيه-1 العالمات تقوم بتوليد مشاهد لا نهاية لها تنتج Sora 2 مقاطع فيديو طويلة الدقيقة مع صوت متزامن وطبيعة نموذجية. إنفيديا كوسمو-درايف ووايف غايا-2 و تسلا دريفينغ وورلد تولد مقاطع فيديو قيادة واقعية لمعلومات تدريب السيارات الذاتية. نموذج العالم يتولى بشكل هادئ التمثيل الفعلي للروبوتات

هذه الدروس هي درسة "الصورة الكبيرة" للمرحلة 4. إنها تربط إنتاج الصور وفهم الفيديو والحكمة العاملة مع نمط الهندسة المعمارية التي يتحرك إليها البحث السائد.

## المفهوم

### ثلاث عائلات من النموذج العالمي

```mermaid
flowchart LR
    subgraph GEN["Pure video generation"]
        G1["Text / image prompt"] --> G2["Video DiT"] --> G3["Video frames"]
    end
    subgraph ACTION["Action-conditioned world model"]
        A1["Past frames + action"] --> A2["Latent-action video DiT"] --> A3["Next frames"]
        A3 --> A1
    end
    subgraph RL["World models for RL (DreamerV3)"]
        R1["State + action"] --> R2["Latent transition model"] --> R3["Next latent + reward"]
        R3 --> R1
    end

    style GEN fill:#dbeafe,stroke:#2563eb
    style ACTION fill:#fef3c7,stroke:#d97706
    style RL fill:#dcfce7,stroke:#16a34a
```

- **Sora 2**هو إنتاج الفيديو النقيّة المُشروط على الإشارات، لا يوجد واجهة عمل. لا يمكنك "توجيه"ها في منتصف التنفيذ.
- **Genie 3**،**GWM-1 Worlds**،**Mirage / Magica**النماذج العالمية المتحركة. إضافة الإجراءات الخفية من الفيديو الملاحظ، ثم تحديد التنبؤات الإطار المستقبلية على الإجراءات. التفاعلية  تضغط على المفاتيح أو تحريك الكاميرا والسينة تستجيب.
- **DreamerV3**وتتوقع عائلة النموذج العالمي الكلاسيكي RL في مساحة خفية مع تكييف العمل الصريح ، مدربة على إشارة مكافأة. أقل بصرية ؛ أكثر فائدة ل RL فعالة العينات.

### معمارة الفيديو

```
Video latent:          (C, T, H, W)
Patchify (spatial):    grid of P_h x P_w patches per frame
Patchify (temporal):   group P_t frames into a temporal patch
Resulting tokens:      (T / P_t) * (H / P_h) * (W / P_w) tokens
```

التشفير الموضعي هو 3D: إدخال مدير أو تعلم لكل إحداثيات (t، h، w).

- **Full joint** جميع الرموز الاحتياطية على جميع الرموز الاحتياطية. O ((N^2) مع الرموز الاحتياطية. محظور على مقاطع الفيديو الطويلة.
- **Divided** التناوب التناوبي (الموقف الفضائي نفسه عبر الزمن: `(H*W) * T^2`) والاهتمام بالمساحة (المساوية في الوقت، عبر الفضاء: `T * (H*W)^2`يستخدمها TimeSformer ومعظم المشاهد الفيديو
- **Window** نوافذ محلية في (t, h, w). يستخدمها فيديو سوين.

كل نموذج للتوزيع الفيديو لعام 2026 يستخدم أحد هذه الأنماط الثلاثة بالإضافة إلى تكييف AdaLN (الدرس 23) وتدفق المصلح.

### تكييف الإجراءات: نماذج عمل غامضة

العبقري يتعلم**latent action**في كل إطار من خلال التنبؤ التمييزي بالعمل بين زوج من الإطارات المتتالية. ثم يقوم مُشعّر النموذج بتشريع الإجراء المتخفّض المستنير  وليس على مفاتيح لوحة المفاتيح الصريحة. عند الإستنتاج، يمكن للمستخدم تحديد إجراء متخفّض (أو أخذ عينة من سابقة جديدة) ويولد النموذج الإطار التالي المتوافق مع هذا الإجراء.

سورا تخطي واجهة العمل بالكامل. مقياسها يتنبأ بالتعليمات التالية من رموز الزمن الفضائي السابقة. تحدد السرعة البدء؛ لا شيء يديرها في منتصف الجيل.

### الموافقة الجسدية

إصدار Sora 2 لعام 2026 يعلن صراحة **physical plausibility**: الوزن، والتوازن، استمرارية الكائن، السبب والنتيجة. يقاسه الفريق عن طريق درجات المثقلة المحددة يدوياً؛ يحسن النموذج بشكل واضح على الكائنات التي سقطت، والحروف التي اصطدمت، والفشل في الغرض (قفز مفقود) مقابل Sora 1.

لا تزال الموافقة هي النظام المهيمن للخلف. أشرطة الفيديو 2024-2025 التي تظهر أشخاص يأكلون سباجيتي أو يشربون من النظارات كشفت عن عدم وجود تمثيل كائن مستمر في النموذج. أشرطة 2026 (سورا 2 ، رينواي جين 5 ، HunyuanVideo) تقلل من هذه المواد ولكن لا تخلص منها.

### نماذج عالمية للقيادة الذاتية

نموذجات العالم القيادة تولد مشاهد طريق واقعية مشروطة على المسارات أو صناديق الحدود أو خرائط الملاحة. الاستخدام:

- **Cosmos-Drive-Dreams**(NVIDIA)  تولد دقائق من الفيديو القيادة للتدريب RL.
- **Gaia-2**(موجة)  تركيب المشهد المحدد عن المسار لتقييم السياسات.
- **DrivingWorld**(تيسلا)  يحاكي مختلف الطقس، وقت اليوم، ظروف المرور.
- **Vista**(بايت دانس)  رد فعل تركيب مشهد القيادة.

إنها تحل محل جمع بيانات عالية الثمن في العالم الحقيقي لحالات الزاوية  مشوار المشاة في الليل، التقاطعات الجليدية، أنواع المركبات غير العادية  التي تتطلب في غير ذلك ملايين الأميال من القيادة.

### كومة الروبوتات: VLM + نموذج الفيديو + ديناميكية عكسية

حلقة الروبوتات الناشئة من ثلاثة مكونات:

1. **VLM**يُحلل الهدف ("تقط الكأس الحمراء") ، ويحدد تسلسلًا للعمل على مستوى عال.
2. **Video generation model**يحاكي كيفية تنفيذ كل عمل  يتوقع الملاحظات N الإطارات في الأمام.
3. **Inverse dynamics model**يستخرج القيود المتحركة الملموسة التي ستنتج هذه الملاحظات.

هذا يحل محل تشكيل الجائزة و RL ثقيلة العينات. يقوم نموذج العالم بالتخيل؛ وتغلق الديناميكيات العكسية الحلقة على التفعيل. Genie Envisioner هي مثال واحد؛ العديد من مجموعات البحوث تتحرك على هذا الهيكل.

### التقييم

- **Visual quality** FVD (Fréchet Video Distance) ، دراسات المستخدمين.
- **Prompt alignment** درجة كليبس لكل إطار، تقييم على شكل VQA.
- **Physical plausibility** تصنيف يدوي على مجموعة مقياسية (المقياس المقياس الداخلي لـ Sora 2، VBench).
- **Controllability**(لنموذجات العالم التفاعلية)  العمل → التواصل الملاحظ؛ هل يمكنك العودة إلى حالة سابقة؟

### نموذج المناظر الطبيعية في عام 2026

| Model | Use | Parameters | Output | License |
|-------|-----|------------|--------|---------|
| Sora 2 | text-to-video, audio | — | 1-min 1080p + audio | API only |
| Runway Gen-5 | text/image-to-video | — | 10s clips | API |
| Runway GWM-1 Worlds | interactive world | — | infinite 3D rollout | API |
| Genie 3 | interactive world from image | 11B+ | playable frames | research preview |
| Wan-Video 2.1 | open text-to-video | 14B | high-quality clips | non-commercial |
| HunyuanVideo | open text-to-video | 13B | 10s clips | permissive |
| Cosmos / Cosmos-Drive | autonomous driving sim | 7-14B | driving scenes | NVIDIA open |
| Magica / Mirage 2 | AI-native game engine | — | modifiable worlds | product |

```figure
v4-world-rollout
```

## بناءها

### الخطوة 1: تصفح الفيديو ثلاثية الأبعاد

```python
import torch
import torch.nn as nn


class VideoPatch3D(nn.Module):
    def __init__(self, in_channels=4, dim=64, patch_t=2, patch_h=2, patch_w=2):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, dim,
            kernel_size=(patch_t, patch_h, patch_w),
            stride=(patch_t, patch_h, patch_w),
        )
        self.patch_t = patch_t
        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, x):
        # x: (N, C, T, H, W)
        x = self.proj(x)
        n, c, t, h, w = x.shape
        tokens = x.reshape(n, c, t * h * w).transpose(1, 2)
        return tokens, (t, h, w)
```

محفظة ثلاثية الأبعاد مع خطوة مساوية للنواة تعمل كمسحوق فضائي-زمني. `(T, H, W) -> (T/2, H/2, W/2)`شبكة الرموز

### الخطوة 2: تشفير الموقف الدوري 3D

إرسال الموقف المتحرك (RoPE) يتم تطبيقه بشكل منفصل على طول `t`،`h`،`w`المحاور:

```python
def rope_3d(tokens, t_dim, h_dim, w_dim, grid):
    """
    tokens: (N, T*H*W, D)
    grid: (T, H, W) sizes
    t_dim + h_dim + w_dim == D
    """
    T, H, W = grid
    n, seq, d = tokens.shape
    if t_dim + h_dim + w_dim != d:
        raise ValueError(f"t_dim+h_dim+w_dim ({t_dim}+{h_dim}+{w_dim}) must equal D={d}")
    assert seq == T * H * W
    t_idx = torch.arange(T, device=tokens.device).repeat_interleave(H * W)
    h_idx = torch.arange(H, device=tokens.device).repeat_interleave(W).repeat(T)
    w_idx = torch.arange(W, device=tokens.device).repeat(T * H)
    # Simplified: just scale channels by frequencies. Real RoPE rotates pairs.
    freqs_t = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(t_dim // 2, device=tokens.device) / (t_dim // 2))
    freqs_h = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(h_dim // 2, device=tokens.device) / (h_dim // 2))
    freqs_w = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(w_dim // 2, device=tokens.device) / (w_dim // 2))
    emb_t = torch.cat([torch.sin(t_idx[:, None] * freqs_t), torch.cos(t_idx[:, None] * freqs_t)], dim=-1)
    emb_h = torch.cat([torch.sin(h_idx[:, None] * freqs_h), torch.cos(h_idx[:, None] * freqs_h)], dim=-1)
    emb_w = torch.cat([torch.sin(w_idx[:, None] * freqs_w), torch.cos(w_idx[:, None] * freqs_w)], dim=-1)
    return tokens + torch.cat([emb_t, emb_h, emb_w], dim=-1)
```

شكل مضيف مبسط. يدور ROPE الحقيقي القنوات المزدوجة عند الترددات. المعلومات الموضعية هي نفسها.

### الخطوة الثالثة: حظر الانتباه

```python
class DividedAttentionBlock(nn.Module):
    def __init__(self, dim=64, heads=2):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.space_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, grid):
        T, H, W = grid
        n, seq, d = x.shape
        # time attention: same (h, w), across t
        xt = x.view(n, T, H * W, d).permute(0, 2, 1, 3).reshape(n * H * W, T, d)
        a, _ = self.time_attn(self.ln1(xt), self.ln1(xt), self.ln1(xt), need_weights=False)
        xt = (xt + a).reshape(n, H * W, T, d).permute(0, 2, 1, 3).reshape(n, seq, d)
        # space attention: same t, across (h, w)
        xs = xt.view(n, T, H * W, d).reshape(n * T, H * W, d)
        a, _ = self.space_attn(self.ln2(xs), self.ln2(xs), self.ln2(xs), need_weights=False)
        xs = (xs + a).reshape(n, T, H * W, d).reshape(n, seq, d)
        xs = xs + self.mlp(self.ln3(xs))
        return xs
```

الاهتمام بالوقت يصل داخل كل موقف فضائي عبر الزمن. الاهتمام بالمكان يصل داخل كل إطار عبر المواقع. عمليات O  T ^ 2 + (HW) ^ 2) بدلاً من O  THW) ^ 2). هذا هو جوهر TimeSformer وكل فيديو حديث DiT.

### الخطوة الرابعة: قم بتأليف مقطع فيديو صغير

```python
class TinyVideoDiT(nn.Module):
    def __init__(self, in_channels=4, dim=64, depth=2, heads=2):
        super().__init__()
        self.patch = VideoPatch3D(in_channels=in_channels, dim=dim, patch_t=2, patch_h=2, patch_w=2)
        self.blocks = nn.ModuleList([DividedAttentionBlock(dim, heads) for _ in range(depth)])
        self.out = nn.Linear(dim, in_channels * 2 * 2 * 2)

    def forward(self, x):
        tokens, grid = self.patch(x)
        for blk in self.blocks:
            tokens = blk(tokens, grid)
        return self.out(tokens), grid
```

ليس مولد فيديو يعمل، إنه عرض هيكلي يشكّل كل قطعة بشكل صحيح.

### الخطوة 5: تحقق من الأشكال

```python
vid = torch.randn(1, 4, 8, 16, 16)  # (N, C, T, H, W)
model = TinyVideoDiT()
out, grid = model(vid)
print(f"input  {tuple(vid.shape)}")
print(f"tokens grid {grid}")
print(f"output {tuple(out.shape)}")
```

انتظر`grid = (4, 8, 8)`و`out = (1, 256, 32)`بعد التشغيل، يُنشر الرأس بعد ذلك إلى تقطيعات مساحية زمنية محددة، جاهزة لتكون غير مُشغّلة مرة أخرى في فيديو.

## استخدمها

أنماط وصول الإنتاج لعام 2026:

- **Sora 2 API**(OpenAI)  نص إلى الفيديو، صوت متزامن. أسعار مكافأة.
- **Runway Gen-5 / GWM-1**(مشاركة) الصورة إلى الفيديو، العوالم التفاعلية.
- **Wan-Video 2.1 / HunyuanVideo** مفتوح المصدر المضيف الذاتي.
- **Cosmos / Cosmos-Drive**(نفيديا)  قيادة محاكاة الوزن المفتوحة.
- **Genie 3** عرض مقدم للبحوث، طلب الوصول.

لبناء نموذج ديمو عالمي تفاعلي: ابدأ مع وان فيديو للجودة، وضع على مكيّف عمل غامض للتفاعل. لمحاكاة القيادة الذاتية: كوسموس-درايو هو مرجع مفتوح 2026 .

للروبوتات، الكبيرة في البرية:

1. هدف اللغة -> VLM (Qwen3-VL) -> خطة مستوى عالية.
2. خطة -> نموذج فيديو عمل غامض -> التنفيذ المتخيل.
3. الإطلاق -> نموذج الديناميكية المعاكسة -> إجراءات منخفضة المستوى.
4. الإجراءات التي تم تنفيذها -> الملاحظة التي تم إعادة إدخالها إلى الخطوة 1.

## أرسله

هذا الدرس ينتج عن:

- `outputs/prompt-video-model-picker.md` اختيارات بين Sora 2 / Runway / Wan / HunyuanVideo / Cosmos مع إعطاء المهمة والترخيص والتمدد.
- `outputs/skill-physical-plausibility-checks.md` مهارة تحدد التحققات الآلية (استمرارية الكائن والجاذبية والاستمرار) للعمل على أي فيديو تم إنشاؤه قبل الشحن.

## التمارين

1. **(Easy)**احسب عدد الرموز لفيديو 360p لمدة 5 ثوان عند اللمسة t=2, اللمسة h=8, اللمسة w=8.
2. **(Medium)**قم بتغيير كتلة الاهتمام المقسمة أعلاه للحصول على كتلة الاهتمام المشتركة الكاملة وقياس شكل وعدد المعلمات. شرح لماذا الاهتمام المقسم ضروري لنماذج الفيديو الحقيقية.
3. **(Hard)**بناء نموذج فيديو عمل غامض ضئيل: خذ مجموعة بيانات من (frame_t، action_t، frame_{t+1}) ثلاثية (أي لعبة 2D بسيطة) ، تدريب فيديو DiT صغير مشروط على إضافة العمل، وأظهر أن الإجراءات المختلفة تنتج إطارات مختلفة التالية.

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| World model | "Learned simulator" | A model that predicts future observations given state and action |
| Video DiT | "Spacetime transformer" | Diffusion transformer with 3D patchification and divided attention |
| Latent action | "Inferred control" | Discrete or continuous action latent inferred from frame pairs; used to condition next-frame generation |
| Divided attention | "Time then space" | Two attention operations per block — across time then across space — to keep O(N^2) manageable |
| Object permanence | "Things stay real" | Scene property that video models must learn; classic failure mode on food, glassware |
| FVD | "Fréchet Video Distance" | Video equivalent of FID; primary visual quality metric |
| Inverse dynamics model | "Observations to actions" | Given (state, next state), output the action that connects them; closes robotics loop |
| Cosmos-Drive | "NVIDIA driving sim" | Open-weights autonomous-driving world model for RL and evaluation |

## المزيد من القراءة

- [Sora technical report (OpenAI)](https://openai.com/index/video-generation-models-as-world-simulators/)
- [Genie: Generative Interactive Environments (Bruce et al., 2024)](https://arxiv.org/abs/2402.15391) نماذج عالم العمل الخفية
- [TimeSformer (Bertasius et al., 2021)](https://arxiv.org/abs/2102.05095) الاهتمام الموزع لتحولات الفيديو
- [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104) نماذج عالمية لـ RL
- [Cosmos-Drive-Dreams (NVIDIA, 2025)](https://research.nvidia.com/labs/toronto-ai/cosmos-drive-dreams/) نموذج العالم للقيادة
- [Top 10 Video Generation Models 2026 (DataCamp)](https://www.datacamp.com/blog/top-video-generation-models)
- [From Video Generation to World Model — survey repo](https://github.com/ziqihuangg/Awesome-From-Video-Generation-to-World-Model/)
