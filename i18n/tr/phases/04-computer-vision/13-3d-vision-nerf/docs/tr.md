# 3D Görüş  Nokta Bulutları & NeRF'ler

> 3 boyutlu görüntü iki farklı şekilde ortaya çıkar. Nokta bulutları sensörün çiğ çıkışıdır. NeRF'ler öğrenilen boyutsal alandır. Her ikisi de "uzayda nerede" cevabını verir.

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 1 Lesson 12 (Tensor Operations)
**Time:** ~45 minutes

## Öğrenme Hedefleri

- Açık (nokta bulutu, ağ, voksel) ve içerikli (işaretlenmiş mesafe alanı, NeRF) 3D temsillerini ayırt edin ve her biri ne zaman kullanılır
- PointNet'in simetrik fonksiyon hilesini anlayın. Bu da sinir ağının düzensiz bir nokta kümesi üzerinde değişken bir değişkenlik gösterir.
- NeRF ileri geçişini izleyin: ışın dökümleri, volumetrik renderi, konum kodlaması, MLP yoğunluğu+ renk başlığı
- Kullanım`nerfstudio`veya `instant-ngp`Küçük bir dizi poz görüntüden önceden eğitilmiş 3D rekonstrüksiyon için

## Sorun

Bir kamera 2 boyutlu bir görüntü üretir. Bir LIDAR, düzenlenmeden 3 boyutlu noktaların bir dizi üretir. Bir yapı-hareket boru hattı, 3 boyutlu anahtar noktaların nadir bir bulutunu üretir. Bir NeRF, bir avuç poz görüntüden tüm 3 boyutlu bir sahneyi yeniden oluşturur. Bunların hepsi "görüş" dir, ancak hiçbirleri CNN'nin istediği yoğun tensöre benzememiştir.

3D görüntü önemli çünkü neredeyse her yüksek değerli robot görevi 3D'de çalışır: yakalama, engellerden kaçınma, navigasyon, AR gizleme, 3D içeriği yakalama. Sadece 2D görüntüleri anlayan bir görme mühendisi alanın en hızlı büyüyen kesiminden (AR / VR içeriği, robotik, otonom sürüş yığınları, emlak veya inşaat için NeRF tabanlı 3D yeniden inşaat) uzaklaştırılır.

İki temsil farklı nedenlerle baskın. Nokta bulutları sensörlerin size ücretsiz olarak verdiği şeydir. NeRF'ler ve onların ardıcılleri (3D Gaussian splating, sinirsel SDF'ler) bir nöro ağından bir sahneyi öğrenmesini istediğinizde elde ettiğiniz şeydir.

## Anlaşım

### Bıçak bulutları

Bir nokta bulutu, R^3'te N noktaların düzensiz bir kümesidir, seçeneği olarak her biri özelliklere (renk, yoğunluk, normal) sahiptir.

```
cloud = [
  (x1, y1, z1, r1, g1, b1),
  (x2, y2, z2, r2, g2, b2),
  ...
  (xN, yN, zN, rN, gN, bN),
]
```

İki özellik sinir ağları için zorlaştırıyor:

- **Permutation invariance** çıkış nokta sırasından bağımsız olmalıdır.
- **Variable N** tek bir model farklı boyutlu bulutları ele almalıdır.

PointNet (Qi et al., 2017) her ikisini de bir fikirle çözdü: her noktaya ortak bir MLP uygulayın, sonra simetrik bir fonksiyonla (maksimum bir havuz) toplayın. Sonuç, sıraya bağlı olmayan sabit boyutlu bir vektördür.

```
f(P) = max_{p in P} MLP(p)
```

Bu, PointNet'in tüm çekirdeğidir. Daha derin çeşitler (PointNet++, Point Transformer) hiyerarşik örnekleme ve yerel birleştirme ekler, ancak simetrik fonksiyon hilesi değişmez.

### PointNet mimarisi

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

"Bağışlanmış MLP", her noktada bağımsız olarak aynı MLP çalıştırılır.

### Nöral Radyans Alanları (NeRF)

NeRF (Mildenhall et al., 2020) "N fotoğraflardan 3D sahneyi yeniden yapılandırabilir miyiz?" sorusunu aldı ve sahne olan bir nöron ağıyla cevap verdi.`(x, y, z, viewing_direction)`- ...`(density, colour)`Yeni bir görüntü vermek, bu ağ üzerinde ışın yayım döngüsüdür.

```
NeRF MLP:  (x, y, z, theta, phi) -> (sigma, r, g, b)

To render a pixel (u, v) of a new view:
  1. Cast a ray from the camera through pixel (u, v)
  2. Sample points along the ray at distances t_1, t_2, ..., t_N
  3. Query the MLP at each point
  4. Composite the colours weighted by (1 - exp(-sigma * dt))
  5. The sum is the rendered pixel colour
```

Bir kayıp, render edilmiş pikselle eğitim fotoğraflarında yerçekimsel gerçeklik pikseline karşılaştırılır. Renderleme adımları aracılığıyla arka plan MLP'yi güncelleyebilir. 3D yerçekimsel gerçeklik, açık bir jeometri yoktur.

### NeRF'de pozisyon kodlaması

Vanilya MLP ' de .`(x, y, z)`NeRF, bu durumu, her koordinatın MLP'den önce bir Fourier özelliği vektörüne kodlanarak düzeltir:

```
gamma(p) = (sin(2^0 pi p), cos(2^0 pi p), sin(2^1 pi p), cos(2^1 pi p), ...)
```

L=10 frekans seviyelerine kadar. Bu, pozisyonlar için kullanılan aynı hile transformörleridir ve difüzyon zaman şartlandırmasında tekrar ortaya çıkar (Desin 10).

### Volumetrik görüntüleme

```
C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i

T_i  = exp(- sum_{j<i} sigma_j * delta_j)
delta_i = t_{i+1} - t_i
```

`T_i`i.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.I.`(1 - exp(-sigma_i * delta_i))`i. noktada bulanıklık.`c_i`Son piksel ışın boyunca ağırlanan bir toplamdır.

### NeRF'lerin yerini ne aldı?

Saf NeRF'ler eğitiminde (saatler) ve görüntüleme sırasında (bir görüntü başına saniyeler) yavaş.

- **Instant-NGP**(2022)  hash-grid kodlaması MLP'nin pozisyon girişini değiştirir; saniyeler içinde trenler.
- **Mip-NeRF 360** sınırsız sahne ve anti-aliasing ile başa çıkıyor.
- **3D Gaussian Splatting**(2023) , volumetrik alanı milyonlarca 3D Gaussians ile değiştirir; trenler dakikalarda, gerçek zamanlı olarak rendere eder.

2026'da neredeyse tüm gerçek NeRF ürünleri aslında 3 boyutlu Gaussian splattering.

### Verim kümeleri ve referans değerleri

- **ShapeNet** 3D CAD modellerinin nokta bulutları olarak sınıflandırılması ve bölünmesi.
- **ScanNet** bölümleşme için gerçek iç taramalar.
- **KITTI** Özerk sürücü için açık LIDAR nokta bulutları.
- **NeRF Synthetic**- Ne ?**Blended MVS** Görünüm sentezi için poz görüntü verileri.
- **Mip-NeRF 360**Veri kümesi  sınırsız gerçek sahne.

```figure
nerf-rays
```

## Yapın

### Adım 1: PointNet sınıflandırıcısı

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

1.6M'lik bir parametre, bulut başına 1.024 puan.

### Adım 2: Pozisyon kodlaması

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

 ile çarpma`2^l * pi`Bu da sürekli olarak daha yüksek frekanslar verir.

### Adım 3: Küçük NeRF MLP

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

NeRF'nin orijinaline kıyasla küçük (Dikamet 8'e sahip olan 2 MLP gövdesi)

### Adım 4: Bir ışın boyunca boyutsal görüntüleme

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

Bir ışın, 64 örnek, tek bir RGB piksel ve derinlik ile birleşik.

## Kullan

Gerçek iş için:

- `nerfstudio`(Tancik et al.)  NeRF / Instant-NGP / Gaussian Splatting için mevcut referans kütüphanesi. Komut satırı ve bir web izleyicisi.
- `pytorch3d`(Meta)  Farklı gösterim, nokta bulut yardımları, ağ operasyonları.
- `open3d` nokta bulut işleme, kayıt, görselleştirme.

3D Gaussian splating, saf NeRF'leri büyük ölçüde değiştirdi çünkü 100 kat daha hızlı hale getirdi.

## Gönder

Bu ders şunları ortaya çıkarır:

- `outputs/prompt-3d-task-router.md` görev ve giriş verilerine dayalı doğru 3D temsiline (nokta bulut, ağ, voksel, NeRF, Gaussian splat) yönlendiren bir istek.
- `outputs/skill-point-cloud-loader.md`Bir PyTorch yazma yeteneği.`Dataset`Doğru normallaştırma, merkezleme ve nokta örneklemesi ile .ply / .pcd / .xyz dosyaları için.

## Egzersizler

1. **(Easy)**PointNet'in permutasyon değişikliğinden farklı olduğunu gösterin: aynı bulutu iki kez, bir kez noktaları karıştırarak çalıştırın.
2. **(Medium)**Kamera içsellikleri ve pozları göz önüne alındığında, H x W görüntüsünün her pikseli için ışın kökenleri ve yönlerini üreten minimal bir ışın üretimi fonksiyonu uygulayın.
3. **(Hard)**TinyNeRF'yi renk küpünün (differensiyal rendering veya basit bir ışın izleyicisi ile oluşturulan) gösterilen görüntülerin sentetik bir veri kümesine uygulayın. 1, 10 ve 100 dönemlerde gösterilen kayıp raporlarını bildirin.

## Anahtar Terimler

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

## Daha Fazla Okumak

- [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593) Permutasyon değişken sınıflandırıcısı
- [NeRF (Mildenhall et al., 2020)](https://arxiv.org/abs/2003.08934)Fotoğraflardan 3 boyutlu rekonstrüksiyonu sinir ağı sorunu haline getiren kağıt
- [Instant-NGP (Müller et al., 2022)](https://arxiv.org/abs/2201.05989) Haş şebekeleri, 1000 kat hızlandırma
- [3D Gaussian Splatting (Kerbl et al., 2023)](https://arxiv.org/abs/2308.04079) üretimdeki NeRF'leri değiştiren mimarlık
