# Visión 3D  Nube de punto y NeRF

> La visión 3D viene en dos sabores. las nubes de punto son la salida bruta del sensor. las NeRF son el campo volumétrico aprendido. Ambas respuestas son "lo que es dónde en el espacio".

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 1 Lesson 12 (Tensor Operations)
**Time:** ~45 minutes

## Objetivos de aprendizaje

- Distinguir entre representaciones 3D explícitas (nube de punto, malla, voxel) e implícitas (campo de distancia firmado, NeRF) y cuándo se utilizan cada una
- Comprender el truco de la función simétrica de PointNet que hace que una red neuronal permutation-invariant sobre un conjunto desordenado de puntos
- Trazar un paso hacia adelante de NeRF: fundición de rayos, renderización volumétrica, codificación posicional, densidad de MLP+tema de color
- Usar`nerfstudio`o `instant-ngp`para la reconstrucción 3D pre-entrenada a partir de un pequeño conjunto de imágenes de puesta

## El problema

Una cámara produce una imagen en 2D. Una LIDAR produce un conjunto de puntos en 3D sin orden. Una estructura de la tubería de movimiento produce una escasa nube de puntos clave en 3D. Una NeRF reconstruye una escena en 3D completa a partir de un puñado de imágenes de poses. Todas estas son "visión" pero ninguna de ellas se parece al tensor denso que quiere una CNN.

La visión 3D es importante porque casi todas las tareas de alto valor de los robots se ejecutan en 3D: captura, evitación de obstáculos, navegación, oclusión de AR, captura de contenido 3D. Un ingeniero de visión que sólo entiende imágenes 2D está excluido de la parte de campo de más rápido crecimiento (contenido AR / VR, robótica, pilas de conducción autónoma, reconstrucción 3D basada en NeRF para bienes raíces o construcción).

Las dos representaciones dominan por razones diferentes. Las nubes de puntos son lo que los sensores te dan de forma gratuita. NeRF y sus sucesores (3D Gaussian splatting, neural SDF) son lo que obtienes cuando pides a una red neuronal que aprenda una escena.

## El concepto

### Nube de punto

Una nube de puntos es un conjunto desordenado de N puntos en R^3, opcionalmente cada uno con características (color, intensidad, normal).

```
cloud = [
  (x1, y1, z1, r1, g1, b1),
  (x2, y2, z2, r2, g2, b2),
  ...
  (xN, yN, zN, rN, gN, bN),
]
```

No hay red, no hay conectividad. Dos propiedades hacen que esto sea difícil para las redes neuronales:

- **Permutation invariance** la salida no debe depender del orden de los puntos.
- **Variable N** un modelo único debe manejar nubes de diferentes tamaños.

PointNet (Qi et al., 2017) resolvió ambos con una idea: aplicar un MLP compartido a cada punto, luego agregar con una función simétrica (pool máximo).

```
f(P) = max_{p in P} MLP(p)
```

Este es todo el núcleo de PointNet. Las variantes más profundas (PointNet++, Point Transformer) añaden muestreo jerárquico y agregación local, pero el truco de función simétrica no ha cambiado.

### La arquitectura de PointNet

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

"MLP compartido" significa que el mismo MLP se ejecuta en cada punto de forma independiente.

### Los campos de radiación neuronal (NeRF)

NeRFs (Mildenhall et al., 2020) tomó la pregunta "¿Podemos reconstruir una escena 3D a partir de N fotos?" y respondió con una red neuronal que es la escena.`(x, y, z, viewing_direction)`¿ Qué ?`(density, colour)`Dar una nueva vista es un circuito de radiodifusión a través de esta red.

```
NeRF MLP:  (x, y, z, theta, phi) -> (sigma, r, g, b)

To render a pixel (u, v) of a new view:
  1. Cast a ray from the camera through pixel (u, v)
  2. Sample points along the ray at distances t_1, t_2, ..., t_N
  3. Query the MLP at each point
  4. Composite the colours weighted by (1 - exp(-sigma * dt))
  5. The sum is the rendered pixel colour
```

Una pérdida compara el píxel renderizado con el píxel de verdad en la tierra en las fotos de entrenamiento. Backprop a través del paso de renderización actualiza el MLP. No hay verdad en el suelo 3D, no hay geometría explícita  la escena se almacena en los pesos de MLP.

### Codificación de posición en NeRF

Una vanilla con MLP en .`(x, y, z)`No puede representar detalles de alta frecuencia porque los MLP están espectralmente sesgados hacia las frecuencias bajas. NeRF corrige esto codificando cada coordenada en un vector de características de Fourier antes del MLP:

```
gamma(p) = (sin(2^0 pi p), cos(2^0 pi p), sin(2^1 pi p), cos(2^1 pi p), ...)
```

Hasta niveles de frecuencia L=10. Este es el mismo truco que usan los transformadores para posiciones, y aparece de nuevo en el acondicionamiento del tiempo de difusión (lección 10).

### Renderamiento volumétrico

```
C(r) = sum_i T_i * (1 - exp(-sigma_i * delta_i)) * c_i

T_i  = exp(- sum_{j<i} sigma_j * delta_j)
delta_i = t_{i+1} - t_i
```

`T_i`es la transmisión  cuánto luz sobrevive al punto i. `(1 - exp(-sigma_i * delta_i))`es la opacidad en el punto i. `c_i`El píxel final es una suma ponderada a lo largo del rayo.

### Lo que sustituyó a las NERF

Las NeRF puras son lentas en entrenamiento (hora) y lentas en renderización (segundos por imagen).

- **Instant-NGP**(2022)  codificación de red de hash reemplaza la entrada de posición del MLP; trenes en segundos.
- **Mip-NeRF 360** maneja escenas ilimitadas y antialiasing.
- **3D Gaussian Splatting**(2023)  reemplaza el campo volumétrico con millones de Gaussians 3D; trenes en minutos, renderiza en tiempo real.

Casi todos los productos reales de NeRF en 2026 son en realidad 3D Gaussian splatting.

### Datos y referencias

- **ShapeNet** Clasificación y segmentación de modelos CAD 3D como nubes de puntos.
- **ScanNet** escaneos interiores reales para segmentación.
- **KITTI** Nube de punto LIDAR para conducción autónoma.
- **NeRF Synthetic**- ¿ Qué ?**Blended MVS** conjuntos de datos de imágenes presentadas para la síntesis de visualización.
- **Mip-NeRF 360**conjunto de datos  escenas reales ilimitadas.

```figure
nerf-rays
```

## Construye el mismo

### Paso 1: Clasificador de PointNet

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

Un parámetro de 1,6 millones, y se ejecuta en 1.024 puntos por nube.

### Paso 2: codificación de posición

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

Multiplicando por `2^l * pi`El sistema de radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la radio de la

### Paso 3: MLP de NeRF pequeño

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

Pequeño en comparación con el original NeRF (que tiene 2 troncos de profundidad MLP 8).

### Paso 4: Renderización volumétrica a lo largo de un rayo

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

Un rayo, 64 muestras, compuestas a un solo píxel RGB y una profundidad.

## Usalo

Para el trabajo real:

- `nerfstudio`(Tancik et al.)  la biblioteca de referencia actual para NeRF / Instant-NGP / Gaussian Splatting.
- `pytorch3d`(Meta)  renderización diferenciable, utilidades de nube de punto, operaciones de malla.
- `open3d` procesamiento en la nube de puntos, registro, visualización.

Para el despliegue, el 3D Gaussian splatting ha reemplazado en gran medida a los NeRF puros porque hace 100 veces más rápido.

## Envío

Esta lección produce:

- `outputs/prompt-3d-task-router.md` un prompt que se dirige a la representación 3D correcta (nube de punto, malla, voxel, NeRF, espacio de Gaussian) basado en la tarea y los datos de entrada.
- `outputs/skill-point-cloud-loader.md` una habilidad que escribe un PyTorch `Dataset`para archivos .ply / .pcd / .xyz con normalización correcta, centrar y muestreo de puntos.

## Los ejercicios

1. **(Easy)**Muestre que PointNet es invariable en permutación: ejecuta la misma nube dos veces, una vez con puntos mezclados. Verifique que las salidas son idénticas hasta el ruido de punto flotante.
2. **(Medium)**Implemente una función de generación de rayos mínima que, dada la intrínseca y la pose de la cámara, produzca los orígenes y direcciones de rayos para cada píxel de una imagen H x W.
3. **(Hard)**Entrenar un TinyNeRF en un conjunto de datos sintético de vistas renderizadas de un cubo de color (generado a través de una representación diferenciable o un simple rastreador de rayos).

## Términos clave

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

## Leer más

- [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593) el clasificador de permutación-invariante
- [NeRF (Mildenhall et al., 2020)](https://arxiv.org/abs/2003.08934) el papel que hizo de la reconstrucción 3D de fotos un problema de red neuronal
- [Instant-NGP (Müller et al., 2022)](https://arxiv.org/abs/2201.05989) redes de hash, 1000x de velocidad
- [3D Gaussian Splatting (Kerbl et al., 2023)](https://arxiv.org/abs/2308.04079) la arquitectura que sustituyó a los NeRF en la producción
