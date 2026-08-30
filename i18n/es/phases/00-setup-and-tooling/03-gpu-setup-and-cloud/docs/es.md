# Configuración de GPU y nube

> El entrenamiento en CPU es bueno para aprender.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## Objetivos de aprendizaje

- Verifique la disponibilidad de GPU local utilizando `nvidia-smi`y la API CUDA de PyTorch
- Configurar Google Colab con una GPU T4 para experimentos gratuitos basados en la nube
- Indique la multiplicación de matriz de referencia en CPU vs GPU y mide la aceleración
- Estima el modelo más grande que se ajusta a su VRAM usando la regla de pulgar fp16

## El problema

La mayoría de las clases en las fases 1-3 funcionan bien en CPU. Pero una vez que comienzas a entrenar CNNs, transformadores o LLM (fase 4+), necesitas aceleración de GPU. Una carrera de entrenamiento que dura 8 horas en CPU toma 10 minutos en GPU.

Tienes tres opciones: GPU local, GPU en la nube o Google Colab (gratuito).

## El concepto

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

## Construye el mismo

### Opción 1: GPU local de NVIDIA

Compruebe si tiene uno:

```bash
nvidia-smi
```

Instalar PyTorch con CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Opción 2: Colab de Google

1. ¡ Vamos ![colab.research.google.com](https://colab.research.google.com)
2. Tiempo de ejecución > Cambia el tipo de tiempo de ejecución > GPU T4
3. - ¿ Qué ?`!nvidia-smi`para verificar

Cargue los cuadernos de este curso directamente a Colab.

### Opción 3: GPU en la nube

Para Lambda Labs, RunPod o Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### ¿No hay GPU?

La mayoría de las clases funcionan en CPU. Los que necesitan GPU dirán eso e incluirán enlaces Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Construir: GPU vs CPU referencia

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

## Los ejercicios

1. Ejecutar el índice de referencia de arriba y comparar CPU vs GPU veces
2. Si no tienes una GPU, ejecuta en Google Colab y compara
3. Compruebe la cantidad de memoria de GPU que tiene y estima el modelo más grande que pueda caber (regla de pulgar: 2 bytes por parámetro para fp16)

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| CUDA | "GPU programming" | NVIDIA's parallel computing platform that lets you run code on the GPU |
| VRAM | "GPU memory" | Video RAM on the GPU, separate from system RAM. Limits model size. |
| fp16 | "Half precision" | 16-bit floating point, uses half the memory of fp32 with minimal accuracy loss |
| Tensor Core | "Fast matrix hardware" | Specialized GPU cores for matrix multiplication, 4-8x faster than regular cores |
