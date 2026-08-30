# Resim Yükleme  Yayınlama Modelleri

> Bir difüzyon modeli, denosiyon yapmayı öğrenir. Şımarık bir görüntüden küçük bir gürültü çıkarmak için eğitilir, bunu bin kez geriye tekrarlar ve bir görüntü jeneratörüne sahip olursunuz.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 07 (U-Net), Phase 1 Lesson 06 (Probability), Phase 3 Lesson 06 (Optimizers)
**Time:** ~75 minutes

## Öğrenme Hedefleri

- Önüne gelen gürültü sürecini çıkar `x_0 -> x_1 -> ... -> x_T`Ve neden kapalı biçim olduğunu açıklayın.`q(x_t | x_0)`t için geçerli
- Her adımda eklenen gürültüyü geriye çeviren ve saf gürültüden görüntüye geriye giden bir örnekleme hedefi uygulayın.
- Zaman koşullu bir U-Net oluşturun (CPU'da eğitilmek için yeterince küçük) herhangi bir zaman aşamasında gürültüyi tahmin eder
- DDPM ve DDIM örneklemesi arasındaki farkı ve her biri uygun olduğunda açıklayın (Leçon 23 akış eşleşmesini ve derinlikteki düzeltilmiş akışı kapsar)

## Sorun

GAN'lar tek çekim oluşturur: gürültü içeri, görüntü dışarı, bir ileri geçiş. Hızlı ve eğitilmesi zor. Diffüzyon modelleri tekrar tekrar üretir: saf gürültüden başlayarak, küçük adımlarla denetleştirir, görüntü ortaya çıkar. Yavaş ve kolay eğitilmektedirler. Son beş yıldır son özellik baskın olmuştur: küçük bir ekip bir difüzyon modeli eğitime ve makul örnekler alabilir; GAN eğitimi yıllarca başarısız koşularda öğrenilen bir meslektir.

Eğitim istikrarının ötesinde, difüzyonun tekrarlayıcı yapısı modern görüntü üretimi yapan her şeyi açar: metin koşullandırması, boyalama, görüntü düzenleme, süper çözünürlük, kontrol edilebilir stil. Örnekleme döngüsünün her aşaması yeni bir kısıtlama enjeksiyonu için bir yer. Bu kanca, Stable Diffusion, Imagen, DALL-E 3, Midjourney ve kullanacağınız her kontrol edilebilir görüntü modeli'nin nedenini gösteriyor.

Bu ders minimal DDPM'yi oluşturur: ileri gürültü, geriye denoizing, eğitim döngüsü. Bir sonraki ders (Stable Diffusion) onu bir VAE, bir metin kodlayıcı ve sınıflandırıcısız rehberlik ile bir üretim sistemine bağlar.

## Anlaşım

### İlerleme süreci

Bir resim çek .`x_0`Biraz Gaussian gürültüsü ekle .`x_1`Biraz daha ekleyip alacağız .`x_2`T adımlarını devam ettir .`x_T`Bu, saf Gaussian gürültüsünden neredeyse ayırt edilemez.

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1},  beta_t * I)
```

`beta_t`T=1000 adım üzerinde tipik olarak 0.0001'den 0.02'e kadar doğrusal küçük bir varyansa programıdır.

### Kapalı atlama

Bir adım sonra gürültü eklemek Markov zinciri, ama matematik katlanır: örnekleyebilirsiniz `x_t`Doğrudan `x_0`Bir adımdan sonra.

```
Define alpha_t = 1 - beta_t
Define alpha_bar_t = prod_{s=1..t} alpha_s

Then:
  q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0,  (1 - alpha_bar_t) * I)

Equivalently:
  x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
  where epsilon ~ N(0, I)
```

Bu tek denklem, yaymanın pratik olmasının tek nedeni.`t`, örnek`x_t`Doğrudan `x_0`, ve tek bir adımla tren  tüm Markov zincirinin simülasyonu gerekmiyor.

### Geri dönüşüm süreci

Önüme doğru ilerleme süreci sabit.`p(x_{t-1} | x_t)`Bu, sinir ağının öğrendiği şeydir.`x_{t-1}`Doğrudan; gürültüyü tahmin ediyorlar `epsilon`T adımında eklenir ve matematik çıkarır `x_{t-1}`- Bu yüzden.

```mermaid
flowchart LR
    X0["x_0<br/>(clean image)"] --> Q1["q(x_t|x_0)<br/>add noise"]
    Q1 --> XT["x_t<br/>(noisy)"]
    XT --> MODEL["model(x_t, t)"]
    MODEL --> EPS["predicted epsilon"]
    EPS --> LOSS["MSE against<br/>true epsilon"]

    XT -.->|sampling| STEP["p(x_{t-1}|x_t)"]
    STEP -.-> XT1["x_{t-1}"]
    XT1 -.->|repeat 1000x| X0S["x_0 (sampled)"]

    style X0 fill:#dcfce7,stroke:#16a34a
    style MODEL fill:#fef3c7,stroke:#d97706
    style LOSS fill:#fecaca,stroke:#dc2626
    style X0S fill:#dbeafe,stroke:#2563eb
```

### Eğitim kaybı

Her eğitim aşaması için:

1. Gerçek bir görüntü örneklemesi `x_0`- Evet .
2. Zaman aşamasını örnekleyin `t`[1, T'den] eşit olarak.
3. Örnek gürültü`epsilon ~ N(0, I)`- Evet .
4. Hesaplama`x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`- Evet .
5. Önceden tahmin et .`epsilon_theta(x_t, t)`Ağla.
6. En azı `|| epsilon - epsilon_theta(x_t, t) ||^2`- Evet .

Neural ağ her zamanki gürültüyü tahmin etmeyi öğrenir. Kayıp MSE.

### Örnekleme (DDPM)

Yaratmak için: `x_T ~ N(0, I)`Ve bir adım sonra geriye doğru yürüyüşe devam et.

```
for t = T, T-1, ..., 1:
    eps = model(x_t, t)
    x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps) + sqrt(beta_t) * z
    where z ~ N(0, I) if t > 1, else 0
return x_0
```

Anahtar şu ki, ters koşul genel olarak kapalı biçimde bilinmese de, bu özel Gaussian ileri süreci için öyle.

### Neden 1000 adım atıyorsun?

Önceki gürültü programı seçilir, böylece her adım, geri adım neredeyse Gaussian olması için yeterli gürültü ekler. Çok az adım ve geri adım Gaussian'dan uzak, ağ iyi bir şekilde modelleyebilir. Çok fazla adım ve örnekleme giderek azalır kazanç ile pahalı hale gelir. T = 1000 bir çizgi programı DDPM varsayılanıdır.

### DDIM: 20 kat daha hızlı örnekleme

DIM (Song et al., 2020) yeniden eğitim almadan zaman adımlarını atlayan bir belirleyici ters süreç tanımlar. DDIM ile 50 adımdan örnek almak yaklaşık 1000 adım DDPM kalitesini verir. Her üretim sistemi DDIM veya daha hızlı bir variansı kullanır (DPM-Solver, Euler ataları).

### Zaman şartlandırması

Ağ `epsilon_theta(x_t, t)`Modern difüzyon modelleri enjekte eder.`t`Sinusoidal zaman gömülmeleri (transformatörlerde konum kodlaması gibi aynı fikir) ile, her U-Net seviyesindeki özellik haritalarına eklenir.

```
t_embedding = sinusoidal(t)
feature_map += MLP(t_embedding)
```

Zaman koşullandırılmadan ağ, işlevsel ancak daha az örnek verimli olan ses seviyesini görüntüden tahmin etmek zorunda.

```figure
cv-diffusion-image
```

## Yapın

### Adım 1: Ses programı

```python
import torch

def linear_beta_schedule(T=1000, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


def precompute_schedule(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
    }

schedule = precompute_schedule(linear_beta_schedule(T=1000))
```

Bir kez önceden hesaplayın, eğitim ve örnekleme sırasında indeksle toplayın.

### Adım 2: Önceki yayılma (q_sampl)

```python
def q_sample(x0, t, noise, schedule):
    sqrt_a = schedule["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    return sqrt_a * x0 + sqrt_one_minus_a * noise
```

Tek satırlı kapalı form.`t`Zaman aşamaları bir seri, partideki her görüntü için bir tane.

### Adım 3: Küçük bir zaman koşullu U-Net

```python
import torch.nn as nn
import torch.nn.functional as F
import math

def timestep_embedding(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return emb


class TinyUNet(nn.Module):
    def __init__(self, img_channels=3, base=32, t_dim=64):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, base * 4),
            nn.SiLU(),
            nn.Linear(base * 4, base * 4),
        )
        self.t_dim = t_dim
        self.enc1 = nn.Conv2d(img_channels, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.mid = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(base * 2, img_channels, 3, padding=1)
        self.time_proj = nn.Linear(base * 4, base * 2)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.t_mlp(t_emb)
        t_proj = self.time_proj(t_emb)[:, :, None, None]

        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1)) + t_proj
        h3 = F.silu(self.mid(h2))
        d1 = F.silu(self.dec1(h3))
        d2 = torch.cat([d1, h1], dim=1)
        return self.dec2(d2)
```

İki seviye U-Net, zaman koşullandırması ile şişe boynunda enjekte edilir.

### 4. Adım: Eğitim döngüsü

```python
def train_step(model, x0, schedule, optimizer, device, T=1000):
    model.train()
    x0 = x0.to(device)
    bs = x0.size(0)
    t = torch.randint(0, T, (bs,), device=device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, schedule)
    pred = model(x_t, t)
    loss = F.mse_loss(pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

Bu eğitim döngüsü, hiçbir GAN oyunu, özel bir kayıp, tek bir MSE çağrısı.

### Adım 5: Örnekleme (DDPM)

```python
@torch.no_grad()
def sample(model, schedule, shape, T=1000, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    betas = schedule["betas"].to(device)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"].to(device)
    sqrt_recip_alphas = schedule["sqrt_recip_alphas"].to(device)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        coef = betas[t] / sqrt_one_minus_a[t]
        mean = sqrt_recip_alphas[t] * (x - coef * eps)
        if t > 0:
            x = mean + torch.sqrt(betas[t]) * torch.randn_like(x)
        else:
            x = mean
    return x
```

Gerçek kodda bunu DDIM 50 adımlı örnekleme cihazı ile değiştirirsin.

### Adım 6: DDIM örnekleme (deterministik, ~ 20 kat daha hızlı)

```python
@torch.no_grad()
def sample_ddim(model, schedule, shape, steps=50, T=1000, device="cpu", eta=0.0):
    model.eval()
    x = torch.randn(shape, device=device)
    alphas_cumprod = schedule["alphas_cumprod"].to(device)

    ts = torch.linspace(T - 1, 0, steps + 1).long()
    for i in range(steps):
        t = ts[i]
        t_prev = ts[i + 1]
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        a_t = alphas_cumprod[t]
        a_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        dir_xt = torch.sqrt(1 - a_prev - sigma ** 2) * eps
        noise = sigma * torch.randn_like(x) if eta > 0 else 0
        x = torch.sqrt(a_prev) * x0_pred + dir_xt + noise
    return x
```

`eta=0`tam olarak belirleyici (aynı gürültü giriş her zaman aynı çıkış üretir). `eta=1`DDPM'yi kurtarır.

## Kullan

Üretim işlerinde kullanın `diffusers`- ...

```python
from diffusers import DDPMScheduler, UNet2DModel

unet = UNet2DModel(sample_size=32, in_channels=3, out_channels=3, layers_per_block=2)
scheduler = DDPMScheduler(num_train_timesteps=1000)
```

Kütüphane hazır programlayıcıları (DDPM, DDIM, DPM-Solver, Euler, Heun), yapılandırılabilir U-Nets, metin-resim ve resim-resim için boru hattları ve LoRA ince ayarlama yardımcıları gönderir.

Araştırma için.`k-diffusion`(Katherine Crowson) en sadık referans uygulamalar ve en iyi örnekleme varianlarına sahiptir.

## Gönder

Bu ders şunları ortaya çıkarır:

- `outputs/prompt-diffusion-sampler-picker.md` kalite hedefine, gecikme bütçesine ve şartlandırma türüne göre DDPM / DDIM / DPM-Solver / Euler'i seçen bir istek.
- `outputs/skill-noise-schedule-designer.md` T ve hedef bozukluk seviyesini vererek, bir çizgi, cosine veya sigmoid beta programı üreten bir beceri, ayrıca zaman içinde sinyal-gürültü oranının teşhis planları.

## Egzersizler

1. **(Easy)**Önümüzdeki süreci görselleştirin: Bir görüntü ve bir çizgi alın `x_t`- ...`t in [0, 100, 250, 500, 750, 1000]`- Bunu kontrol et .`x_1000`saf Gaussian sesi gibi görünüyor.
2. **(Medium)**TinyUNet'i 20 dönem boyunca sentetik döngüler verisi üzerinde eğit ve 16 döngü örnekleyin. DDPM (1000 adım) ve DDIM (50 adım) örneklemesini karşılaştırın.
3. **(Hard)**Kosine gürültü programını uygula (Nichol & Dhariwal, 2021):`alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi / 2)`Aynı modeli çizgi ve kozin çizelgeleri ile çalıştırın ve kozin düşük adım sayısında daha iyi örnekler verdiğini gösterin.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Forward process | "Add noise over time" | Fixed Markov chain that corrupts an image into Gaussian noise over T steps |
| Reverse process | "Denoise step by step" | Learned distribution that walks back from noise to image |
| Epsilon prediction | "Predict the noise" | The training target: `epsilon_theta(x_t, t)` predicts the noise added at step t |
| Beta schedule | "Noise amounts" | Sequence of T small variances that define how much noise enters per step |
| alpha_bar_t | "Cumulative retain factor" | Product of (1 - beta_s) up to time t; bigger t means less signal left |
| DDPM sampler | "Ancestral, stochastic" | Samples each x_{t-1} from its conditional Gaussian; 1000 steps |
| DDIM sampler | "Deterministic, fast" | Rewrites sampling as a deterministic ODE; 20-100 steps with similar quality |
| Time conditioning | "Tell the model which t" | Sinusoidal embedding of t injected into the U-Net so it knows the noise level |

## Daha Fazla Okumak

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) yayımı pratik yapan ve FID'de GAN'ları yenen kağıt
- [Improved DDPM (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672) Kosinus programı ve v parametreleme
- [DDIM (Song, Meng, Ermon, 2020)](https://arxiv.org/abs/2010.02502) gerçek zamanlı sonuçlandırmayı mümkün kılan belirleyici örnekleme
- [Elucidating the Design Space of Diffusion (Karras et al., 2022)](https://arxiv.org/abs/2206.00364) her difüzyon tasarım seçeneğinin tek bir görünümü; mevcut en iyi referans
