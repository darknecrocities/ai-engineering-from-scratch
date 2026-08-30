# Resim Temellikleri  Pikseller, Kanallar, Renk Alanları

> Bir görüntü, ışık örneklerinin bir tensörüdür. Kullandığınız her görüntü modeli bu tek gerçekten başlar.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1 Lesson 12 (Tensor Operations), Phase 3 Lesson 11 (Intro to PyTorch)
**Time:** ~45 minutes

## Öğrenme Hedefleri

- Sürekli bir sahne nasıl piksellere ayrılır ve örnekleme/kvantisalatma kararlarının her aşağı akımlı modelde tavan belirlemesini açıklayın.
- NumPy dizileri olarak görüntüleri okuyun, kesin ve kontrol edin ve HWC ve CHW düzenleri arasında akıcı olarak geçin
- RGB, gri ölçek, HSV ve YCbCr arasında dönüştürün ve her renk alanının neden var olduğunu açıklayın
- Piksel düzeyinde önceden işleme uygulayın (normalleştir, standartlaştır, boyutlandır, kanal önce) önceden eğitilmiş PyTorch görme modelleri beklediği gibi

## Sorun

Okuduğunuz her makale, indirdiğiniz her önceden eğitilmiş ağırlık, aradığınız her görüntü API'si girenin belirli bir kodlamasını varsayıyor.`uint8`Model istediği görüntü`float32`Bu da bir hata oluşturmaz. RGB'de eğitimli bir ağ için BGR'yi besleyin ve doğruluk on puan düşer. Bir model kanallar - kanalları beklediğinde son giriş - ilk ve ilk konfor katmanının yüksekliği bir özellik kanalı olarak ele alması. Bunların hiçbirisi hata atmaz. Sadece ölçümlerinizi mahvediyor ve dosyayı nasıl yüklediğinizde yaşayan bir hata için bir hafta avlarsınız.

Bir konvulsiyon, neyin kaydı olduğunu bildiğinizde karmaşık değildir. Zor kısmı, "bir görüntü" nin bir kamera, JPEG dekodörü, PIL, OpenCV, torchvision ve CUDA çekirdeği için farklı şeyler anlamına gelmesidir. Her yığının kendi eksis sırası, bayt aralığı ve kanal konvansiyonu vardır.

Bu ders temelini sabitler, böylece kalan aşama da buna dayanır. Sonuna kadar bir piksel ne olduğunu, bir piksel yerine neden üç sayı olduğunu, "ImageNet istatistikleriyle normalleştirmek"in aslında ne yaptığını ve bu aşamada diğer her dersinin öngördüğü iki veya üç düzen arasında nasıl hareket edeceğini bileceksiniz.

## Anlaşım

### Tüm önceden işleme boru hattı bir bakışta

Her üretim görme sistemi aynı dönüşümlü dönüşümler sırasıdır. Bir adım yanlış atarsan model eğitiminden farklı bir giriş görür.

```mermaid
flowchart LR
    A["Image file<br/>(JPEG/PNG)"] --> B["Decode<br/>uint8 HWC"]
    B --> C["Convert<br/>colorspace<br/>(RGB/BGR/YCbCr)"]
    C --> D["Resize<br/>shorter side"]
    D --> E["Center crop<br/>model size"]
    E --> F["Divide by 255<br/>float32 [0,1]"]
    F --> G["Subtract mean<br/>Divide by std"]
    G --> H["Transpose<br/>HWC → CHW"]
    H --> I["Batch<br/>CHW → NCHW"]
    I --> J["Model"]

    style A fill:#fef3c7,stroke:#d97706
    style J fill:#ddd6fe,stroke:#7c3aed
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#bfdbfe,stroke:#2563eb
```

İki kırmızı ve mavi kutuda sessiz başarısızlıkların %80'inin yaşadığı yerler: standartlaşmanın eksikliği ve yanlış düzenlemeler.

### Bir piksel bir örnek, bir kare değil.

Kamera sensörü, küçük detektörler arasında yer alan fotonları sayır. Her detektör saniyenin bir kısmında ışığı birleştirir ve ona çarpan foton sayısına orantılı bir voltaj yayar.

```
Continuous scene                 Sensor grid                     Digital image
(infinite detail)                (H x W detectors)               (H x W integers)

    ~~~~~                        +--+--+--+--+--+                 210 198 180 155 120
   # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  - ...> +--+--+--+--+--+----> 200 190 175 150 115
   ~~~~~                         |  |  |  |  |  |                 195 185 170 148 112
                                 +--+--+--+--+--+                 188 180 165 145 108
```

Bu aşamada iki seçenek vardır ve aşağıdaki her şeyin tavanını düzeltirler:

- **Spatial sampling**Bu, bir sahne derecesine göre kaç detektör belirler. Çok az, kenarları bozulur. Çok fazla, depolama ve hesaplama patlar.
- **Intensity quantization**8 bit 256 seviye verir ve görüntüleme için standarttır. 10, 12, 16 bit tıbbi görüntüleme, HDR ve ham sensör boru hattları için daha düzgün bir kaydırma ve madde verir.

Bir piksel, alanı olan renkli bir kare değil. Tek bir ölçümdür. Boyut değiştirdiğinizde veya döndüğünüzde, ölçüm şebekesini yeniden örneklersiniz.

### Neden üç kanal?

Bir detektör, tüm görünür spektrumda foton sayır  yani gri ölçek. Renk elde etmek için, sensör, ağı kırmızı, yeşil ve mavi filtrelerin mozaikleriyle kaplar. Demosaik yapıldıktan sonra, her uzay konumunun üç tam sayı vardır: kırmızı filtreli detektörün tepkisi, yeşil filtreli ve mavi filtreli yakınlarda. Bu üç tam sayı bir piksel RGB üçlüdür.

```
One pixel in memory:

    (R, G, B) = (210, 140, 30)   <- reddish-orange

An H x W RGB image:

    shape (H, W, 3)     stored as   H rows of W pixels of 3 values
                                    each in [0, 255] for uint8
```

Üç sihirli değildir. Derinlik kameraları bir Z kanalı ekler. Uydular kızılötesi ve ultraviyole bantlar ekler. Tıbbi taramalar genellikle bir kanal (X-ışını, CT) veya birçok (hiperspektral) vardır. Kanal sayısı son eksendir; konv katmanları karışmayı öğrenir.

### İki düzenleme konferansı: HWC ve CHW

Aynı tenzor, iki sıralama.

```
HWC (height, width, channels)           CHW (channels, height, width)

   W ->                                    H ->
  +-----+-----+-----+                     +-----+-----+
H |R G B|R G B|R G B|                   C |R R R R R R|
| +-----+-----+-----+                   | +-----+-----+
v |R G B|R G B|R G B|                   v |G G G G G G|
  +-----+-----+-----+                     +-----+-----+
                                          |B B B B B B|
                                          +-----+-----+

   PIL, OpenCV, matplotlib,              PyTorch, most deep learning
   almost every image file on disk       frameworks, cuDNN kernels
```

CHW, konvulsiyon çekirdeklerinin H ve W'ye kaydırılması nedeniyle var. Kanal eksisini önce tutmak, her çekirdekin her kanal için bir bitişik 2 boyutlu düzlem görmesini, temiz bir şekilde vektörleştirmesini sağlar. Disk formatları HWC'yi tutar çünkü bu bir sensörden tarama çizgilerinin nasıl çıktığına uyuyor.

Tek satırlı dönüşümü bin kez yazacaksınız:

```
img_chw = img_hwc.transpose(2, 0, 1)      # NumPy
img_chw = img_hwc.permute(2, 0, 1)        # PyTorch tensor
```

Anıtlama düzenlemesi, görselleştirilmiş:

```mermaid
flowchart TB
    subgraph HWC["HWC — pixels stored interleaved (PIL, OpenCV, JPEG)"]
        H1["row 0: R G B | R G B | R G B ..."]
        H2["row 1: R G B | R G B | R G B ..."]
        H3["row 2: R G B | R G B | R G B ..."]
    end
    subgraph CHW["CHW — channels stored as stacked planes (PyTorch, cuDNN)"]
        C1["plane R: entire H x W of red values"]
        C2["plane G: entire H x W of green values"]
        C3["plane B: entire H x W of blue values"]
    end
    HWC -->|"transpose(2, 0, 1)"| CHW
    CHW -->|"transpose(1, 2, 0)"| HWC
```

### Bayt aralıkları ve dtype

Üç büyük ibadetin baskısı:

| Convention | dtype | Range | Where you see it |
|------------|-------|-------|------------------|
| Raw | `uint8` | [0, 255] | Files on disk, PIL, OpenCV output |
| Normalized | `float32` | [0.0, 1.0] | After `img.astype('float32') / 255` |
| Standardized | `float32` | roughly [-2, +2] | After subtracting mean and dividing by std |

Konvülsiyonel ağlar standart girişler üzerinde eğitildi.`mean=[0.485, 0.456, 0.406]`- Evet .`std=[0.229, 0.224, 0.225]`Bu, üç kanalın tüm ImageNet eğitim kümesi üzerinde normalleştirilmiş piksellerle hesaplanan aritmetik ortalaması ve standart sapmasıdır.`uint8`Standartlaşmış yüzerlik bekleyen bir modelde uygulanan görmede en yaygın sessiz başarısızlık tekdir.

### Renk alanları ve neden var oldukları

RGB, yakalama biçimidir ancak bir model için her zaman en yararlı temsil değildir.

```
 RGB               HSV                       YCbCr / YUV

 R red             H hue (angle 0-360)       Y luminance (brightness)
 G green           S saturation (0-1)        Cb chroma blue-yellow
 B blue            V value/brightness (0-1)  Cr chroma red-green

 Linear to         Separates color from      Separates brightness from
 sensor output     brightness. Useful for    color. JPEG and most video
                   color thresholding, UI    codecs compress the chroma
                   sliders, simple filters   channels harder because the
                                             human eye is less sensitive
                                             to chroma detail than to Y.
```

Çoğu CNN'de RGB'yi beslerken diğer alanlarla karşılaşırken:

- **HSV** klasik CV kodu, renk tabanlı segmentasyon, beyaz dengeleme.
- **YCbCr** JPEG içsellerini, video boru hattlarını, sadece Y üzerinde çalışan süper çözünürlüklü modellerini okumak.
- **Grayscale** OCR, belge modelleri, renk sinyal yerine rahatsızlık değişken olduğu her durum.

RGB'den gri ölçek, ortalama değil, ağırlıklı bir toplamdır, çünkü insan gözü kırmızı veya maviye göre yeşilye daha duyarlıdır:

```
Y = 0.299 R + 0.587 G + 0.114 B       (ITU-R BT.601, the classic weights)
```

### Göreç oranı, boyut değiştirme ve interpolasyon

Her modelde sabit bir giriş boyutu vardır (224x224 çoğu ImageNet sınıflandırıcıları için, 384x384 veya 512x512 modern dedektörler için).

- **Resize shorter side, then center crop** Standart ImageNet tarifi. Görüntü oranını korur, kenar piksel çizgisini atır.
- **Resize and pad** boyut oranını ve her pikselini korur, siyah çubuklar ekler.
- **Resize directly to target**Bilgiyi uzatır, ucuz, geometriyi çarpıtır, birçok sınıflandırma görevi için iyi.

Interpolasyon yöntemi, yeni şebekenin eski şebekeninle uyumlu olmadığı zaman ara pikselilerin nasıl hesaplandığını belirler:

```
Nearest neighbour     fastest, blocky, only choice for masks/labels
Bilinear              fast, smooth, default for most image resizing
Bicubic               slower, sharper on upscaling
Lanczos               slowest, best quality, used for final display
```

Basamak kural: eğitim için binlinear, bakılacak varlık için iki küp veya lanczos, tam sayı sınıfı kimlikleri içeren herhangi bir şey için en yakın.

```figure
conv-output-size
```

## Yapın

### Adım 1: Bir görüntü tensörü oluşturun ve şeklini kontrol edin

Deterministik sentetik bir görüntü ile başlayın, böylece ilk laboratuvar sadece NumPy ile çevrimdışı çalıştırılır. Dosya dekodlaması ayrı bir sınırdır: bir JPEG veya PNG dekodörü RGB baytları döndürdükten sonra, aşağıdaki her tensor işlem aynıdır.

```python
import numpy as np

def synthetic_rgb(h=128, w=192, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    r = (np.sin(xx * 6) * 0.5 + 0.5) * 255
    g = yy * 255
    b = (1 - yy) * xx * 255
    rgb = np.stack([r, g, b], axis=-1) + rng.normal(0, 6, (h, w, 3))
    return np.clip(rgb, 0, 255).astype(np.uint8)

arr = synthetic_rgb()

print(f"type:   {type(arr).__name__}")
print(f"dtype:  {arr.dtype}")
print(f"shape:  {arr.shape}     # (H, W, C)")
print(f"min:    {arr.min()}")
print(f"max:    {arr.max()}")
print(f"pixel at (0, 0): {arr[0, 0]}")
```

Beklenen üretim: `shape: (H, W, 3)`- Evet .`dtype: uint8`, aralığı `[0, 255]`Bu, baytların bir kamera, bir görüntü dekodörü veya bu sentetik jeneratörden geldiği için kanonik olarak çözülmüş bir temsil.

### Adım 2: Çanakları bölün ve yeniden düzenlen

R, G, B'yi ayrı ayrı çıkarın ve sonra PyTorch için HWC'den CHW'ye dönüştürün.

```python
R = arr[:, :, 0]
G = arr[:, :, 1]
B = arr[:, :, 2]
print(f"R shape: {R.shape}, mean: {R.mean():.1f}")
print(f"G shape: {G.shape}, mean: {G.mean():.1f}")
print(f"B shape: {B.shape}, mean: {B.mean():.1f}")

arr_chw = arr.transpose(2, 0, 1)
print(f"\nHWC shape: {arr.shape}")
print(f"CHW shape: {arr_chw.shape}")
```

CHW sadece ekseleri yeniden düzenler; hafıza düzenlemesi izin verdiğinde hiçbir veri kopyası zorunlu olarak gerekmez.

### Adım 3: Gri ölçek ve HSV dönüşümleri

Ağırlıklı toplam gri ölçek, sonra da RGB-HSV manuel.

```python
def rgb_to_grayscale(rgb):
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (rgb.astype(np.float32) @ weights).astype(np.uint8)

def rgb_to_hsv(rgb):
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    cmax = np.max(rgb_f, axis=-1)
    cmin = np.min(rgb_f, axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask = delta > 0
    argmax = np.argmax(rgb_f, axis=-1)
    rmax = mask & (argmax == 0)
    gmax = mask & (argmax == 1)
    bmax = mask & (argmax == 2)
    h[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6
    h[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2
    h[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4
    h = h * 60.0

    s = np.divide(delta, cmax, out=np.zeros_like(delta), where=cmax > 0)
    v = cmax
    return np.stack([h, s, v], axis=-1)

gray = rgb_to_grayscale(arr)
hsv = rgb_to_hsv(arr)
print(f"gray shape: {gray.shape}, range: [{gray.min()}, {gray.max()}]")
print(f"hsv   shape: {hsv.shape}")
print(f"hue range: [{hsv[..., 0].min():.1f}, {hsv[..., 0].max():.1f}] degrees")
print(f"sat range: [{hsv[..., 1].min():.2f}, {hsv[..., 1].max():.2f}]")
print(f"val range: [{hsv[..., 2].min():.2f}, {hsv[..., 2].max():.2f}]")
```

Hue dereceler, doymak ve değer olarak çıkar.`hsv_full`Anlaşma.

### Dördüncü adım: Normalleştir, standartlaştır ve tersine çevir

Çizil baytlardan, önceden eğitilmiş bir ImageNet modeli beklediği tam tensora git ve sonra geri dön.

```python
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_imagenet(rgb_uint8):
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)
    return x

def deprocess_imagenet(chw_float32):
    x = chw_float32.transpose(1, 2, 0)
    x = x * std + mean
    x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return x

x = preprocess_imagenet(arr)
print(f"preprocessed shape: {x.shape}     # (C, H, W)")
print(f"preprocessed dtype: {x.dtype}")
print(f"preprocessed mean per channel:  {x.mean(axis=(1, 2)).round(3)}")
print(f"preprocessed std  per channel:  {x.std(axis=(1, 2)).round(3)}")

roundtrip = deprocess_imagenet(x)
max_diff = np.abs(roundtrip.astype(int) - arr.astype(int)).max()
print(f"roundtrip max pixel diff: {max_diff}    # should be 0 or 1")
```

Kanal başına ortalama sıfır, std bir yakın olmalıdır.`transforms.Normalize`Telefon, kapuk altında.

### Adım 5: Baştan yeniden boyutlandır

En yakın komşu çevreler, her çıkış koordinatını bir kaynak pikseline çevirir. İkizli interpolasyon, çevresindeki dört pikselyi bulur ve onları mesafe ile birleştirir. Aşağıdaki her iki uygulamada son noktalara uyumlu koordinatlar kullanılır, böylece ilk ve son kaynak pikselleri sabit kalır.

```python
def resize_coordinates(source_length, target_length):
    if target_length == 1:
        return np.zeros(1, dtype=np.float32)
    return np.linspace(0, source_length - 1, target_length, dtype=np.float32)

def nearest_resize(image, target_height, target_width):
    y = np.rint(resize_coordinates(image.shape[0], target_height)).astype(int)
    x = np.rint(resize_coordinates(image.shape[1], target_width)).astype(int)
    return image[y[:, None], x[None, :]]

def bilinear_resize(image, target_height, target_width):
    y = resize_coordinates(image.shape[0], target_height)
    x = resize_coordinates(image.shape[1], target_width)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, image.shape[0] - 1)
    x1 = np.minimum(x0 + 1, image.shape[1] - 1)
    wy = (y - y0)[:, None, None]
    wx = (x - x0)[None, :, None]

    source = image.astype(np.float32)
    top = source[y0[:, None], x0[None, :]] * (1 - wx)
    top += source[y0[:, None], x1[None, :]] * wx
    bottom = source[y1[:, None], x0[None, :]] * (1 - wx)
    bottom += source[y1[:, None], x1[None, :]] * wx
    result = top * (1 - wy) + bottom * wy
    return np.clip(np.rint(result), 0, 255).astype(image.dtype)

target_height = arr.shape[0] * 3
target_width = arr.shape[1] * 3
nearest = nearest_resize(arr, target_height, target_width)
bilinear = bilinear_resize(arr, target_height, target_width)

def local_roughness(x):
    gy = np.diff(x.astype(float), axis=0)
    gx = np.diff(x.astype(float), axis=1)
    return float(np.abs(gy).mean() + np.abs(gx).mean())

for name, out in [("nearest", nearest), ("bilinear", bilinear)]:
    print(f"{name:>8}  shape={out.shape}  roughness={local_roughness(out):6.2f}")
```

En yakınları sert kenarları koruduğu için kabalık konusunda en yüksek puanlar verir. Bilinear daha düzgüntür çünkü her yeni piksel her eksede iki pozisyonu birleştirir. Çekilebilir eş aynı ayrılabilir fikri bir eksede dört komşuya Catmull-Rom küp çekirdeği ile uzattır, sonra resim kütüphanesi olmadan tüm üç sonucu da yazdırır.

## Kullan

PyTorch, toplu, cihaz farkındalıklı tensörlerde aynı işlemleri yapar. Aşağıdaki kod daha kısa tarafı boyutlandırır, bir merkez biçimini alır, her kanalı standartlaştırır ve önceden eğitilmiş bir model beklediği NCHW tensörü üretir.

```python
import torch
import torch.nn.functional as F

image_hwc = torch.from_numpy(synthetic_rgb(256, 320))
batch = image_hwc.permute(2, 0, 1).unsqueeze(0).float() / 255.0

height, width = batch.shape[-2:]
scale = 256 / min(height, width)
resized_height = round(height * scale)
resized_width = round(width * scale)
batch = F.interpolate(
    batch,
    size=(resized_height, resized_width),
    mode="bilinear",
    align_corners=False,
    antialias=True,
)

top = (resized_height - 224) // 2
left = (resized_width - 224) // 2
batch = batch[:, :, top:top + 224, left:left + 224]

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
batch = (batch - mean) / std

print(f"tensor dtype: {batch.dtype}")
print(f"batched shape: {tuple(batch.shape)}")
print(f"per-channel mean: {batch.mean(dim=(0, 2, 3)).tolist()}")
print(f"per-channel std:  {batch.std(dim=(0, 2, 3)).tolist()}")
```

Dört adım, bu tam sırada: baytları yüzerek değiştirin ve HWC'yi NCHW'ye değiştirin, daha kısa tarafın boyutunu 256'ya değiştirin, 224x224 merkez biçimini alın, ardından ImageNet ortalamasını çıkarın ve standart sapma ile bölün. Bu sırayı ters çevirmek, modelin ulaştığını sessizçe değiştirir.

## Gönder

Bu ders şunları ortaya çıkarır:

- `outputs/prompt-vision-preprocessing-audit.md` herhangi bir model kartı veya veri kümesi kartını bir takımın yerine getirmesi gereken tam önceden işleme değişkenlerinin kontrol listesine dönüştüren bir istek.
- `outputs/skill-image-tensor-inspector.md` herhangi bir görüntü şeklinde tensor veya dizi verildiğinde dtip, düzen, aralığı ve çiğ, normalleştirilmiş veya standartlaştırılmış görünüşü rapor eden bir beceri.

## Egzersizler

1. **(Easy)**2x2 RGB oluştur`uint8`HWC'yi CHW'ye çevir ve geriye, her iki şeklini de yazdır ve dönüş yolculuğunun her değerini koruduğunu kanıtla.
2. **(Medium)**Yazmın .`standardize(img, mean, std)`ve bunun tersine birlikte bir `roundtrip_max_diff <= 1`Uint8 görüntülerini test et. fonksiyonlarınız HWC'deki tek bir görüntüde ve aynı çağrı ile NCHW'deki bir partide çalışmalıdır.
3. **(Hard)**3 kanallı ImageNet standartlaştırılmış bir tensör alın ve RGB'nin ağırlıklı karışımını bir gri ölçekli kanal olarak öğrenen 1x1 konforu üzerinden çalıştırın.`[0.299, 0.587, 0.114]`, dondur ve çıkışın el kitabına uyuyor olduğunu kontrol et.`rgb_to_grayscale`Hangi klasik renk- uzay dönüşümleri 1x1 dönüşümleri olarak yazılabilir?

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Pixel | "A coloured square" | One sample of light intensity at one grid location — three numbers for colour, one for grayscale |
| Channel | "The colour" | One of the parallel spatial grids stacked into an image tensor; last axis in HWC, first in CHW |
| HWC / CHW | "The shape" | Axis orderings for an image tensor; disk and PIL use HWC, PyTorch and cuDNN use CHW |
| Normalize | "Scale the image" | Divide by 255 so pixels live in [0, 1] — necessary but not sufficient |
| Standardize | "Zero-center" | Subtract mean and divide by std per channel so the input distribution matches what the model was trained on |
| Grayscale conversion | "Average the channels" | A weighted sum with coefficients 0.299/0.587/0.114 that matches human luminance perception |
| Interpolation | "How resize picks pixels" | The rule that decides output values when the new grid does not align with the old one — nearest for labels, bilinear for training, bicubic for display |
| Aspect ratio | "Width over height" | The ratio that distinguishes "resize and pad" from "resize and stretch" |

## Daha Fazla Okumak

- [Charles Poynton — A Guided Tour of Color Space](https://poynton.ca/PDFs/Guided_tour.pdf) rengin neden bu kadar çok olduğunu ve her birinin ne zaman önemli olduğunu açık bir teknik tedavi
- [PyTorch Vision Transforms Docs](https://pytorch.org/vision/stable/transforms.html) üretim sırasında oluşturduğunuz transformasyonların tüm hattı
- [How JPEG Works (Colt McAnlis)](https://www.youtube.com/watch?v=F1kYBnY6mwg) krom alt örnekleme, DCT'nin keskin bir görsel turunu ve JPEG'nin RGB yerine YCbCr'yi neden kodlaması
- [ImageNet Preprocessing Conventions (torchvision models)](https://pytorch.org/vision/stable/models.html) gerçeğin kaynağı`mean=[0.485, 0.456, 0.406]`ve hayvanat bahçesindeki her model neden bunu bekliyor?
