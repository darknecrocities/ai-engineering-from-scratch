# إعدادات GPU و السحابة

> التدريب على المعالجة المركزية جيد للتعلم التدريب الحقيقي يحتاج إلى GPU

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## أهداف التعلم

- التحقق من توافر GPU المحلي باستخدام `nvidia-smi`و API CUDA من PyTorch
- قم بتهيئة Google Colab مع GPU T4 للتجارب المستندة إلى السحابة مجانا
- قم بتحديد مضاعفة المصفوفة المرجعية على CPU مقابل GPU وقياس السرعة
- تقدير أكبر نموذج يناسب في VRAM الخاص بك باستخدام قاعدة fp16 الإبهام

## المشكلة

معظم الدروس في المراحل 1-3 تعمل بشكل جيد على المعالجة المركزية. ولكن بمجرد البدء في تدريب سي إن إن، المحولات، أو LLM (المراحل 4+) ، تحتاج إلى تسريع GPU. تشغيل التدريب الذي يستغرق 8 ساعات على المعالجة المركزية يستغرق 10 دقائق على GPU.

لديك ثلاثة خيارات: GPU محلية، GPU سحرية، أو Google Colab (مجانية).

## المفهوم

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

## بناءها

### الخيار 1: GPU NVIDIA المحلي

تحقق من وجودك

```bash
nvidia-smi
```

قم بتثبيت PyTorch مع CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### الخيار الثاني: Google Colab

1. إذهب إلى[colab.research.google.com](https://colab.research.google.com)
2. وقت تشغيل > تغيير نوع وقت تشغيل > GPU T4
3. أركض`!nvidia-smi`للتحقق

قم بتحميل الملاحظات من هذه الدورة مباشرة إلى (كولاب)

### الخيار الثالث: GPU السحاب

لـ Lambda Labs، RunPod، أو Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### لا مشكلة

معظم الدروس تعمل على المعالجة المركزية. أولئك الذين يحتاجون إلى GPU سوف يقول ذلك وتشمل روابط Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## بناءه: GPU مقابل CPU مقياس

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

## التمارين

1. قم بتشغيل المعيار المرجعي أعلاه ومقارنة CPU مقابل GPU
2. إذا لم يكن لديك جهاز GPU، تشغيله على Google Colab ومقارنة
3. تحقق من كمية ذاكرة GPU لديك وتقدير أكبر نموذج يمكنك إطلاقا (قاعدة الإبهام: 2 بايت لكل مبرمير ل fp16)

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| CUDA | "GPU programming" | NVIDIA's parallel computing platform that lets you run code on the GPU |
| VRAM | "GPU memory" | Video RAM on the GPU, separate from system RAM. Limits model size. |
| fp16 | "Half precision" | 16-bit floating point, uses half the memory of fp32 with minimal accuracy loss |
| Tensor Core | "Fast matrix hardware" | Specialized GPU cores for matrix multiplication, 4-8x faster than regular cores |
