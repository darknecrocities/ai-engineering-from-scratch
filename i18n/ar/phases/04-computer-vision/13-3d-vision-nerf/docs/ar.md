# رؤية ثلاثية الأبعاد  غيول نقطة & NeRFs

> رؤية 3D تأتي في طعمين. سحابة نقطة هي الخروج الخام لمستشعر. NeRFs هي المجال الحجمي المكتمل. كلاهما يجيب "ما هو أين في الفضاء".

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 1 Lesson 12 (Tensor Operations)
**Time:** ~45 minutes

## أهداف التعلم

- تمييز التمثيلات الثلاثية الأبعاد الصريحة (سحابة النقاط، شبكة، فوكسل) والخاطئة (حقل المسافة الموقعة، NeRF) وعندما يتم استخدام كل منها
- فهم خدعة الوظيفة التناظرية من PointNet التي تجعل شبكة عصبية محول-غير متغير على مجموعة غير مرتبة من النقاط
- تتبع مرور NeRF إلى الأمام: إلقاء الأشعة، التصوير الحجمي، التشفير الموقع، كثافة MLP + رأس اللون
- استخدام`nerfstudio`أو`instant-ngp`لإعادة بناء ثلاثي الأبعاد المسبقة من مجموعة صغيرة من الصور الموضحة

## المشكلة

تنتج الكاميرا صورة ثنائية الأبعاد. تنتج LIDAR مجموعة من النقاط الثلاثية الأبعاد دون ترتيب. تنتج خط الأنابيب من الحركة بنية من نقطة مفتاح ثلاثية الأبعاد. تقوم NeRF بإعادة إنشاء مشهد ثلاثي الأبعاد بأكمله من حفنة من الصور الموضحة. كل هذه "رؤية" ولكن لا تبدو أي منها مثل الجهاز الكثيف الذي تريده CNN.

يعتبر رؤية 3D مهمة لأن كل مهمة روبوتية ذات قيمة عالية تقريباً تعمل في 3D: التقاط، تجنب العقبات، الملاحة، حجب AR، التقاط محتوى 3D. يتم منع مهندس الرؤية الذي يفهم الصور 2D فقط من أسرع قطعة نمو في المجال (محتويات AR / VR، الروبوتات، كومات القيادة الذاتية، إعادة بناء 3D بناءً على NeRF للعقارات أو البناء).

تمثل التمثيلين بشكل مختلف لأسباب مختلفة. السحب النقطية هي ما يعطيه الاستشعارات مجانا. النواة العصبية والخلفاء منها (3D Gaussian splatting، SDFs العصبية) هي ما تحصل عليه عندما تطلب من شبكة عصبية أن تتعلم مشهدًا.

## المفهوم

### غيول نقطة

سحابة النقاط هي مجموعة غير مرتبة من N نقاط في R^3، اختياريًا كل منها مع ميزات (اللون، والكثافة، والطبيعية).

```
cloud = [
  (x1, y1, z1, r1, g1, b1),
  (x2, y2, z2, r2, g2, b2),
  ...
  (xN, yN, zN, rN, gN, bN),
]
```

لا شبكة، لا اتصال، هناك خصائص تجعل ذلك صعباً للشبكات العصبية:

- **Permutation invariance** لا يجب أن تعتمد الناتجة على ترتيب النقاط.
- **Variable N** يجب أن يتعامل نموذج واحد مع السحب ذات الأحجام المختلفة.

حل PointNet (Qi et al., 2017) كلتا هذه الأفكار بفكرة واحدة: تطبيق MLP مشتركة على كل نقطة، ثم جمعها مع وظيفة متماثلة (مجموعة أقصى). النتيجة هي متجه مقياس ثابت لا يعتمد على النظام.

```
f(P) = max_{p in P} MLP(p)
```

هذه هي جوهر PointNet بأكمله. الإختلافات العميقة (PointNet+++، Point Transformer) تضيف العينات الهرمية والجمع المحلي ولكن خدعة الوظيفة التناظرية لا تتغير.

### بنية نقطة الشبكة

```mermaid
flowchart LR
    PTS["N points<br/>(x, y, z)"] --> MLP1["shared MLP<br/>(64, 64)"]
    MLP1 --> MLP2["shared MLP<br/>(64, 128, 1024)"]
    MLP2 --> MAX["max pool<br/>(symmetric)"]
    MAX --> FEAT["global feature<br/>(1024,)"]
    FEAT --> FC["MLP classifier"]
    FC --> CLS["class logits"]

    style MLP1 fill:#dbeafe,stroke:#2563eb
    style MAX fill:#fef3c7,stroke:#d97706
    style CLS fill:#dcfce7,stroke:#16a34a
```

"المشاركة MLP" تعني نفس MLP تعمل على كل نقطة بشكل مستقل. يتم تنفيذها كـ 1x1 conv على بعد النقطة لتحقيق الكفاءة.

### حقل الإشعاع العصبي (Neural Radiance Fields)

أخذ NeRFs (Mildenhall et al., 2020) السؤال "هل يمكننا إعادة بناء مشهد 3D من N صور؟" وأجابوا بشبكة عصبية هي المشهد. خرائط الشبكة `(x, y, z, viewing_direction)`إلى`(density, colour)`إعطاء رؤية جديدة هو حلقة إرسال الأشعة على هذه الشبكة

```
NeRF MLP:  (x, y, z, theta, phi) -> (sigma, r, g, b)

To render a pixel (u, v) of a new view:
  1. Cast a ray from the camera through pixel (u, v)
  2. Sample points along the ray at distances t_1, t_2, ..., t_N
  3. Query the MLP at each point
  4. Composite the colours weighted by (1 - exp(-sigma * dt))
  5. The sum is the rendered pixel colour
```

يُقارن الخسارة البيكسل المُسجّل بـ بيكسل الحقيقة الأرضية في صور التدريب. يقوم الخلفية من خلال خطوة التصوير بتحديث MLP. لا يوجد حقيقة الأرض 3D، لا يوجد هندسة صريحة  يتم تخزين المشهد في أوزان MLP.

### تشفير الموقف في NeRF

فانيلا ملفا على`(x, y, z)`لا يمكن أن تمثل تفاصيل عالية التردد لأن MLPs متحيزة بشكل طيفي نحو ترددات منخفضة. يصلح NeRF هذا عن طريق ترميز كل إحداثيات إلى متجه ميزة فوري قبل MLP:

```
gamma(p) = (sin(2^0 pi p), cos(2^0 pi p), sin(2^1 pi p), cos(2^1 pi p), ...)
```

ما يصل إلى مستويات التردد L=10. هذه هي نفس الحيلة التي يستخدمها المحولون للمواقع، ويظهر مرة أخرى في تكييف الوقت التفشي (الدرس 10). بدونها، تبدو NeRFs ضبابية.

### التسجيل الكمي

```
C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i

T_i  = exp(- sum_{j<i} sigma_j * delta_j)
delta_i = t_{i+1} - t_i
```

`T_i`هو الانتقال  كم من الضوء ينجو من النقطة i. `(1 - exp(-sigma_i * delta_i))`هو التضليل في النقطة i. `c_i`هو اللون. البيكسل النهائي هو المبلغ الموزن على طول الشعاع.

### ما الذي استبدل NRFs

إن النواة النقية للنواة النقدية بطيئة في التدريب (الساعات) والبطيئة في الإصدار (الثوان لكل صورة).

- **Instant-NGP**(2022)  تشفير شبكة الهاش يحل محل مدخل الموقف من MLP؛ القطارات في الثواني.
- **Mip-NeRF 360** يتعامل مع مشاهد غير محدودة ومكافحة التخفيف
- **3D Gaussian Splatting**(2023)  يحل محل الحقل الحجمي بملايين غوسيانات 3D؛ القطارات في دقائق، يعطي في الوقت الحقيقي.

تقريبا كل منتج حقيقي من نيرف في عام 2026 هو في الواقع 3D غوسيانة البث. النموذج العقلي لا يزال نيرف.

### مجموعات البيانات ومعايير الموازنة

- **ShapeNet** تصنيف وتقسيم نماذج CAD ثلاثية الأبعاد كغواص نقطة.
- **ScanNet** مسحات داخلية حقيقية لتقسيم
- **KITTI** سحابة نقطة LIDAR للقيادة الذاتية في الهواء الطلق.
- **NeRF Synthetic**- لا ، لا**Blended MVS** مجموعات بيانات الصور الموضحة لتركيب الرؤية.
- **Mip-NeRF 360**مجموعة بيانات  مشاهد حقيقية غير محدودة.

```figure
nerf-rays
```

## بناءها

### الخطوة الأولى: تصنيف PointNet

```python
import torch
import torch.nn as nn

class PointNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),    nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),   nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),  nn.BatchNorm1d(128),  nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 512),   nn.BatchNorm1d(512),  nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),    nn.BatchNorm1d(256),  nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (N, 3, num_points) — transposed for Conv1d
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = torch.max(x, dim=-1)[0]       # (N, 1024)
        return self.head(x)

pts = torch.randn(4, 3, 1024)
net = PointNet(num_classes=10)
print(f"output: {net(pts).shape}")
print(f"params: {sum(p.numel() for p in net.parameters()):,}")
```

حوالي 1.6 مليون مبرمير، يعمل على 1,024 نقطة لكل سحابة.

### الخطوة الثانية: تشفير المواقع

```python
def positional_encoding(x, L=10):
    """
    x: (..., D) -> (..., D * 2 * L)
    """
    freqs = 2.0 ** torch.arange(L, dtype=x.dtype, device=x.device)
    args = x.unsqueeze(-1) * freqs * 3.141592653589793
    sinc = torch.cat([args.sin(), args.cos()], dim=-1)
    return sinc.reshape(*x.shape[:-1], -1)

x = torch.randn(5, 3)
y = positional_encoding(x, L=10)
print(f"input:  {x.shape}")
print(f"encoded: {y.shape}     # (5, 60)")
```

مضاعفة بـ `2^l * pi`يعطي ترددات أعلى تدريجياً.

### الخطوة الثالثة: نيك نيف ملف

```python
class TinyNeRF(nn.Module):
    def __init__(self, L_pos=10, L_dir=4, hidden=128):
        super().__init__()
        self.L_pos = L_pos
        self.L_dir = L_dir
        pos_dim = 3 * 2 * L_pos
        dir_dim = 3 * 2 * L_dir
        self.trunk = nn.Sequential(
            nn.Linear(pos_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
        )
        self.sigma = nn.Linear(hidden, 1)
        self.color = nn.Sequential(
            nn.Linear(hidden + dir_dim, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 3), nn.Sigmoid(),
        )

    def forward(self, x, d):
        x_enc = positional_encoding(x, self.L_pos)
        d_enc = positional_encoding(d, self.L_dir)
        h = self.trunk(x_enc)
        sigma = torch.relu(self.sigma(h)).squeeze(-1)
        rgb = self.color(torch.cat([h, d_enc], dim=-1))
        return sigma, rgb

nerf = TinyNeRF()
x = torch.randn(128, 3)
d = torch.randn(128, 3)
s, c = nerf(x, d)
print(f"sigma: {s.shape}   rgb: {c.shape}")
```

ضئيلة مقارنة مع NeRF الأصلية (التي لديها 2 قذائف MLP من عمق 8). يكفي لإظهار الهندسة المعمارية.

### الخطوة الرابعة: التصوير الكمي على طول شعاع

```python
def volumetric_render(sigma, rgb, t_vals):
    """
    sigma: (..., N_samples)
    rgb:   (..., N_samples, 3)
    t_vals: (N_samples,) distances along the ray
    """
    delta = torch.cat([t_vals[1:] - t_vals[:-1], torch.full_like(t_vals[:1], 1e10)])
    alpha = 1.0 - torch.exp(-sigma * delta)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = alpha * trans
    rendered = (weights.unsqueeze(-1) * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)
    return rendered, depth, weights


N = 64
t_vals = torch.linspace(2.0, 6.0, N)
sigma = torch.rand(N) * 0.5
rgb = torch.rand(N, 3)
rendered, depth, weights = volumetric_render(sigma, rgb, t_vals)
print(f"rendered colour: {rendered.tolist()}")
print(f"depth:           {depth.item():.2f}")
```

شعاع واحد، 64 عينة، مركبة إلى بكسل RGB واحد وعمق.

## استخدمها

للعمل الحقيقي:

- `nerfstudio`(Tancik et al.)  مكتبة المرجعية الحالية لـ NeRF / Instant-NGP / Gaussian Splatting. خط الأوامر بالإضافة إلى متصفح الويب.
- `pytorch3d`(ميتا)  التعبير المميز، وسائل تسجيل السحابة النقطة، عمليات الشبكة.
- `open3d` معالجة السحابة النقطية، التسجيل، التصور.

لتنفيذها، استبدل التصفيق الثلاثي الأبعاد غوسي في معظم الأحيان النوافذ النظيفة النظيفة لأنّه يجعل 100 مرة أسرع. جودة إعادة الإعمار مماثلة.

## أرسله

هذا الدرس ينتج عن:

- `outputs/prompt-3d-task-router.md` عرض يتوجه إلى التمثيل الثلاثي الأبعاد الصحيح (سحابة النقاط، شبكة، فوكسل، NeRF، غوسيان) بناء على المهام والبيانات المدخولية.
- `outputs/skill-point-cloud-loader.md`مهارة كتب بيتورش`Dataset`للفيديات .ply / .pcd / .xyz مع التطبيع الصحيح ، والمركزية ، ومعينة النقاط.

## التمارين

1. **(Easy)**أظهر أن نقطة النت غير متغيرة: قم بتشغيل نفس السحابة مرتين، مرة واحدة مع تضلط النقاط. تحقق من أن المخرجات متطابقة حتى ضجيج نقطة عائمة.
2. **(Medium)**تنفيذ وظيفة توليد الأشعة الحد الأدنى التي، بالنظر إلى أساسيات الكاميرا وموقفها، تنتج أصول الأشعة واتجاهات لكل بكسل من صورة H x W.
3. **(Hard)**تدريب TinyNeRF على مجموعة بيانات اصطناعية من مشاهدات عرضة لمكعب ملونة (تولد عن طريق التصوير المميز أو متبع شعاع بسيط). تقرير الخسارة التصوير في الفترات 1 ، 10 و 100. في أي عصر ينتج النموذج مشاهدات قابلة للتعرف؟

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Point cloud | "3D points from LIDAR" | Unordered set of (x, y, z) + optional features per point |
| PointNet | "First neural net on point clouds" | Shared MLP per point + symmetric (max) pool; permutation-invariant by construction |
| NeRF | "MLP that is the scene" | Network mapping (x, y, z, dir) to (density, colour); rendered by ray casting |
| Positional encoding | "Fourier features" | Encode each coordinate into sin/cos at multiple frequencies to overcome MLP low-frequency bias |
| Volumetric rendering | "Ray integration" | Composite samples along a ray into a single pixel using transmittance and alpha |
| Instant-NGP | "Hash-grid NeRF" | Replaces NeRF's coordinate MLP with a multi-resolution hash grid; 100-1000x faster |
| 3D Gaussian splatting | "Millions of Gaussians" | Scene = collection of 3D Gaussians; renders in real time, trains in minutes |
| SDF | "Signed distance field" | Function returning signed distance to the nearest surface; another implicit representation |

## المزيد من القراءة

- [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593) تصنيف المتغيرات المتغيرة
- [NeRF (Mildenhall et al., 2020)](https://arxiv.org/abs/2003.08934)الورقة التي جعلت إعادة بناء 3D من الصور مشكلة شبكة عصبية
- [Instant-NGP (Müller et al., 2022)](https://arxiv.org/abs/2201.05989)شبكات الهاشيش، 1000x السرعة
- [3D Gaussian Splatting (Kerbl et al., 2023)](https://arxiv.org/abs/2308.04079) الهندسة المعمارية التي استبدلت النفط النووية في الإنتاج
