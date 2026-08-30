#  GPU 设置和云

> 实用训练需要一个GPU.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## 学习目标

- 使用 `nvidia-smi`并且PyTorch的CUDAAPI
- 使用T4GPU配置Google Colab,可进行免费的基于云的实验
- 测量CPU与GPU的基数乘法,测量加快速度
- 根据fp16指纹,估计适合VRAM的最大模型

## 问题

在1-3阶段的大部分课程都在CPU上运行得很好.但是一旦你开始训练CNN,变压器或LLM (阶段4+),你需要GPU加速.一个8小时的训练运行在CPU上需要10分钟的GPU.

你有三个选择:本地GPU,云GPU或谷歌Collab (免费).

## 概念

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

## 建立它

### 选择1:本地NVIDIA GPU

检查你是否有:

```bash
nvidia-smi
```

安装PyTorch与CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 选择2:谷歌协作

1. 走去[colab.research.google.com](https://colab.research.google.com)
2. 运行时间 > 改变运行时间类型 > T4 GPU
3. 跑步`!nvidia-smi`检查

直接将课程的笔记本上传到科拉布.

### 选择3:云GPU

对于Lambda Labs,RunPod或Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有GPU?

需要GPU的人会说,并包括Colab链接.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 构建它:GPU与CPU基准

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

## 运动

1. 运行上述基准,并比较CPU与GPU时间
2. 如果没有GPU,请在Google Colab上运行,然后比较
3. 检查您有多少GPU内存,并估计您可以安装的最大模型 (指公规则:fp16的每个参数为2字节)

## 关键词

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| CUDA | "GPU programming" | NVIDIA's parallel computing platform that lets you run code on the GPU |
| VRAM | "GPU memory" | Video RAM on the GPU, separate from system RAM. Limits model size. |
| fp16 | "Half precision" | 16-bit floating point, uses half the memory of fp32 with minimal accuracy loss |
| Tensor Core | "Fast matrix hardware" | Specialized GPU cores for matrix multiplication, 4-8x faster than regular cores |
