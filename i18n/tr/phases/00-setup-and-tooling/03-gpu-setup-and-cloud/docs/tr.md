# GPU Kurulum ve Bulut

> CPU'da eğitim öğrenmek için iyidir.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## Öğrenme Hedefleri

- Yerel GPU kullanımı doğrulanması `nvidia-smi`Ve PyTorch'in CUDA API'si
- Google Colab'ı ücretsiz bulut tabanlı deneyler için T4 GPU ile yapılandır
- CPU vs GPU'da standart matris çarpımı göster ve hızlandırmayı ölç
- VRAM'a en büyük modelin fp16 basamak kuralını kullanarak uygun olduğunu tahmin edin

## Sorun

1-3 aşamada yapılan derslerin çoğu CPU'da iyi çalışır. Ancak CNN'ler, transformatörler veya LLM'ler (vazesi 4+) eğitime başladıktan sonra, GPU hızlandırmasına ihtiyacınız var. CPU'da 8 saat süren bir eğitim süreci GPU'da 10 dakika sürer.

Üç seçenekiniz var: yerel GPU, bulut GPU veya Google Colab (ücretsiz).

## Anlaşım

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

```figure
s0-gpu-dispatch
```

## Yapın

### Seçenek 1: Yerel NVIDIA GPU

Bir tane var mı diye kontrol et.

```bash
nvidia-smi
```

PyTorch' i CUDA ile yükle:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Seçenek 2: Google Colab

1. Git .[colab.research.google.com](https://colab.research.google.com)
2. Çalışma Zamanı > Çalışma Zamanı Tipini Değiştir > T4 GPU
3. Çık .`!nvidia-smi`doğrulama

Bu kursdan defterleri doğrudan Colab'a yükle.

### Seçenek 3: Bulut GPU

Lambda Labs, RunPod veya Vast.ai için:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### - GPU yok mu?

Çoğu ders CPU'da çalışır. GPU'ya ihtiyaç duyanlar da öyle söyler ve Colab bağlantıları içerir.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Yap: GPU vs CPU referans

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Egzersizler

1. Yukarıdaki referans değerini çalıştırın ve CPU vs GPU sürelerini karşılaştırın
2. Eğer bir GPU'unuz yoksa, Google Colab'da çalıştırın ve karşılaştırın
3. Ne kadar GPU belleği olduğunuzu kontrol edin ve yerleştirilebilecek en büyük modelin tahmin edilmesini sağlayın (barmak kuralı: fp16 için parametresi başına 2 byte)

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| CUDA | "GPU programming" | NVIDIA's parallel computing platform that lets you run code on the GPU |
| VRAM | "GPU memory" | Video RAM on the GPU, separate from system RAM. Limits model size. |
| fp16 | "Half precision" | 16-bit floating point, uses half the memory of fp32 with minimal accuracy loss |
| Tensor Core | "Fast matrix hardware" | Specialized GPU cores for matrix multiplication, 4-8x faster than regular cores |
