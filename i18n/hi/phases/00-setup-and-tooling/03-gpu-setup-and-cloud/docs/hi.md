# GPU सेटअप और क्लाउड

> सीपीयू पर प्रशिक्षण सीखने के लिए ठीक है। वास्तविक के लिए प्रशिक्षण एक GPU की जरूरत है।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## सीखने के लक्ष्य

- स्थानीय GPU उपलब्धता का सत्यापन `nvidia-smi`और PyTorch के CUDA API
- मुफ्त क्लाउड आधारित प्रयोगों के लिए T4 GPU के साथ Google Colab को कॉन्फ़िगर करें
- CPU बनाम GPU पर बेंचमार्क मैट्रिक्स गुणन और गति को मापें
- अपने VRAM में फिट बैठता है कि सबसे बड़ा मॉडल का अनुमान लगाने के लिए fp16 अंगूठे के नियम का उपयोग

## समस्या

सीपीयू पर 8 घंटे का प्रशिक्षण 10 मिनट का होता है, लेकिन सीएनएन, ट्रांसफार्मर या एलएलएम (चरण 4+) को प्रशिक्षित करने के बाद आपको जीपीयू त्वरण की आवश्यकता होती है।

आपके पास तीन विकल्प हैंः स्थानीय जीपीयू, क्लाउड जीपीयू, या गूगल कोलब (मुफ्त) ।

## अवधारणा

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

## इसे बनाओ

### विकल्प 1: स्थानीय NVIDIA GPU

जाँचें कि क्या आपके पास एक हैः

```bash
nvidia-smi
```

CUDA के साथ PyTorch स्थापित करें:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### विकल्प 2: गूगल कोलाब

1. जाओ [colab.research.google.com](https://colab.research.google.com)
2. रनटाइम > रनटाइम प्रकार बदलें > T4 GPU
3. दौड़ें`!nvidia-smi`सत्यापित करने के लिए

इस कोर्स से नोटबुक सीधे कोलाब में अपलोड करें।

### विकल्प 3: क्लाउड जीपीयू

लैम्ब्डा लैब्स, रनपॉड या वास्ट.एआई के लिएः

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### कोई GPU नहीं?

अधिकांश पाठ CPU पर काम करते हैं. जिन लोगों को GPU की जरूरत है वे ऐसा कहेंगे और Colab लिंक शामिल करेंगे.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## इसे बनाएंः GPU बनाम CPU बेंचमार्क

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

## व्यायाम

1. ऊपर बेंचमार्क चलाएं और सीपीयू बनाम जीपीयू समय की तुलना करें
2. यदि आपके पास एक GPU नहीं है, तो इसे Google Colab पर चलाएं और तुलना करें
3. जांचें कि आपके पास कितनी GPU मेमोरी है और अनुमान लगाएं कि आप सबसे बड़ा मॉडल फिट कर सकते हैं (आंगूठे का नियमः fp16 के लिए प्रति पैरामीटर 2 बाइट्स)

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| CUDA | "GPU programming" | NVIDIA's parallel computing platform that lets you run code on the GPU |
| VRAM | "GPU memory" | Video RAM on the GPU, separate from system RAM. Limits model size. |
| fp16 | "Half precision" | 16-bit floating point, uses half the memory of fp32 with minimal accuracy loss |
| Tensor Core | "Fast matrix hardware" | Specialized GPU cores for matrix multiplication, 4-8x faster than regular cores |
