# Generación de imágenes  GAN

> Un GAN es dos redes neuronales en un juego fijo, uno empate, otro critique, se vuelven mejores juntos hasta que los dibujos engañan al crítico.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 3 Lesson 06 (Optimizers), Phase 3 Lesson 07 (Regularization)
**Time:** ~75 minutes

## Objetivos de aprendizaje

- Explica el juego de mínimas entre generador y discriminador y por qué el equilibrio corresponde a p_modelo = p_datos
- Implemente un DCGAN en PyTorch y haga que genere imágenes sintéticas coherentes 32x32 en menos de 60 líneas
- Estabilizar el entrenamiento en GAN con los tres trucos estándar: pérdida no saturante, norma espectral, TTUR (regla de actualización en dos escalas)
- Lea curvas de entrenamiento que distinguen la convergencia saludable del colapso de modo, la oscilación y el discriminador-ganas-completamente

## El problema

La clasificación enseña a una red a mapear imágenes a etiquetas. La generación invierte el problema: muestra nuevas imágenes que parecen provenir de la misma distribución. No hay salida "correcta" que pueda diferenciar; solo hay una distribución que desea imitar.

Las funciones de pérdida estándar (MSE, entropía cruzada) no pueden medir "¿esta muestra proviene de la distribución real?" Minimizando el error por píxel se producen promedios borrosos, no muestras realistas.

Los modelos de difusión han tomado el trono en cuanto a calidad y control, pero cada truco que hace que la difusión sea práctica  opciones de normalización, espacios latentes, pérdidas de características  fue entendido por primera vez en los GAN.

## El concepto

### Las dos redes

```mermaid
flowchart LR
    Z["z ~ N(0, I)<br/>noise"] --> G["Generator<br/>transposed convs"]
    G --> FAKE["Fake image"]
    REAL["Real image"] --> D["Discriminator<br/>conv classifier"]
    FAKE --> D
    D --> OUT["P(real)"]

    style G fill:#dbeafe,stroke:#2563eb
    style D fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

El **generator**G toma un vector de ruido `z`y saca una imagen.**discriminator**D toma una imagen y saca un solo escalar: la probabilidad de que la imagen sea real.

### El juego

G quiere que D esté equivocado, D quiere que tenga razón.

```
min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

Leer de derecha a izquierda: D maximiza la precisión en real (`log D(real)`) y falsificados (`log (1 - D(fake))`G está minimizando la precisión de D en las falsificaciones  quiere `D(G(z))`para estar drogado.

Goodfellow demostró que este mínimo tiene un equilibrio global donde `p_G = p_data`, D produce 0.5 en todas partes, y la divergencia Jensen-Shannon entre las distribuciones generales y reales es cero.

### Las pérdidas no saturantes

La forma anterior es numéricamente inestable.`D(G(z))`es casi cero para cada falso, así que `log(1 - D(G(z)))`tiene gradientes desapareciendo con respecto a G. La solución: la pérdida de G.

```
L_D = -E_x[log D(x)] - E_z[log(1 - D(G(z)))]
L_G = -E_z[log D(G(z))]                          # non-saturating
```

Ahora , ¿ cuándo ?`D(G(z))`El tren GAN moderno tiene esta variante.

### Reglas de arquitectura DCGAN

Radford, Metz, Chintala (2015) destilaron años de experimentos fallidos en cinco reglas que hacen que el entrenamiento GAN sea estable:

1. Reemplazar la agrupación con condes de paso (ambas redes).
2. Utilice la norma de lote tanto en el generador como en el discriminador, excepto la salida de G y la entrada de D.
3. Elimine las capas completamente conectadas en arquitecturas más profundas.
4. G utiliza ReLU en todas las capas excepto en la salida (tanh para la salida en [-1, 1]).
5. D utiliza LeakyReLU (negativo_inclinación=0.2) en todas las capas.

Cada GAN moderno basado en conchas (StyleGAN, BigGAN, GigaGAN) todavía comienza con estas reglas y reemplaza las piezas una a la vez.

### Modo de falla y sus firmas

```mermaid
flowchart LR
    M1["Mode collapse<br/>G produces a narrow<br/>set of outputs"] --> S1["D loss low,<br/>G loss oscillating,<br/>sample variety drops"]
    M2["Vanishing gradients<br/>D wins completely"] --> S2["D accuracy ~100%,<br/>G loss huge and static"]
    M3["Oscillation<br/>G and D keep trading<br/>wins forever"] --> S3["Both losses swing<br/>wildly with no downward trend"]

    style M1 fill:#fecaca,stroke:#dc2626
    style M2 fill:#fecaca,stroke:#dc2626
    style M3 fill:#fecaca,stroke:#dc2626
```

- **Mode collapse**G encuentra una imagen que engaña a D y produce sólo eso.
- **Discriminator wins**Se puede ver que la etiqueta de la etiqueta es más pequeña, o se puede aplicar un suavización de etiqueta en las etiquetas reales.
- **Oscillation**Las operaciones de las dos redes ganan sin acercarse nunca al equilibrio.

### Evaluación

Los GAN no tienen verdad, ¿cómo sabes que funcionan?

- **Sample inspection** sólo miren 64 muestras al final de cada época.
- **FID (Fréchet Inception Distance)** distancia entre las distribuciones de los conjuntos reales y generados de las características de Inception-v3.
- **Inception Score** mayor, más frágil; prefiere FID.
- **Precision/Recall for generative models** mide la calidad (precisión) y la cobertura (recall) por separado.

Para una pequeña prueba de datos sintéticos, basta con la inspección de muestras.

```figure
cv-gan-image
```

## Construye el mismo

### Paso 1: Generador

Un pequeño generador DCGAN que toma ruido de 64 dimensiones y produce una imagen de 32x32.

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=64, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feat * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 4, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 2, feat, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))
```

Cuatro convoyes transpuestas, cada una con `kernel_size=4, stride=2, padding=1`Así que duplican el tamaño espacial.

### Paso 2: Discriminación

Espejo del generador.

```python
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feat, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 2, feat * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 4, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x).view(-1)
```

El último conv reduce un `4x4`mapa de características a `1x1`. La salida es un solo escalar por imagen; aplicar sigmoid sólo durante el cálculo de pérdidas.

### Paso 3: Paso de formación

Alternativa: actualizar D una vez, luego G una vez, cada lote.

```python
import torch.nn.functional as F

def train_step(G, D, real, z, opt_g, opt_d, device):
    real = real.to(device)
    bs = real.size(0)

    # D step
    opt_d.zero_grad()
    d_real = D(real)
    d_fake = D(G(z).detach())
    loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
              + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
    loss_d.backward()
    opt_d.step()

    # G step
    opt_g.zero_grad()
    d_fake = D(G(z))
    loss_g = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
    loss_g.backward()
    opt_g.step()

    return loss_d.item(), loss_g.item()
```

`G(z).detach()`En el paso D es crítico: no queremos que los gradientes fluyan a G durante su actualización.

### Paso 4: Ciclo de entrenamiento completo en formas sintéticas

```python
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def synthetic_images(num=2000, size=32, seed=0):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((num, 3, size, size), dtype=np.float32) - 1.0
    for i in range(num):
        r = rng.uniform(6, 12)
        cx, cy = rng.uniform(r, size - r, size=2)
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        color = rng.uniform(-0.5, 1.0, size=3)
        for c in range(3):
            imgs[i, c][mask] = color[c]
    return torch.from_numpy(imgs)

device = "cuda" if torch.cuda.is_available() else "cpu"
data = synthetic_images()
loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=True)

G = Generator(z_dim=64, img_channels=3, feat=32).to(device)
D = Discriminator(img_channels=3, feat=32).to(device)
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(10):
    for (batch,) in loader:
        z = torch.randn(batch.size(0), 64, device=device)
        ld, lg = train_step(G, D, batch, z, opt_g, opt_d, device)
    print(f"epoch {epoch}  D {ld:.3f}  G {lg:.3f}")
```

`Adam(lr=2e-4, betas=(0.5, 0.999))`es el DCGAN por defecto  la baja beta1 mantiene el tiempo de impulso de estabilizar demasiado el juego adversario.

### Paso 5: Muestreo

```python
@torch.no_grad()
def sample(G, n=16, z_dim=64, device="cpu"):
    G.eval()
    z = torch.randn(n, z_dim, device=device)
    imgs = G(z)
    imgs = (imgs + 1) / 2
    return imgs.clamp(0, 1)
```

Siempre cambia al modo de evaluación antes de la muestreo. para DCGAN esto importa porque se utilizan estadísticas de ejecución de la norma del lote en lugar de las estadísticas del lote.

### Paso 6: Normalización espectral

Un reemplazo de BN en el discriminador que garantiza la red es 1-Lipschitz.

```python
from torch.nn.utils import spectral_norm

def build_sn_discriminator(img_channels=3, feat=64):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(img_channels, feat, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat, feat * 2, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 2, feat * 4, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 4, 1, 4, 1, 0)),
    )
```

Cambiar`Discriminator`por`build_sn_discriminator()`La norma espectral es la mejor actualización de robustez única que puede aplicar.

## Usalo

Para generación seria, utilice pesas preentrenadas o cambie a difusión.

- `torch_fidelity`computa FID / IS en su generador sin escribir código de evaluación personalizado.
- `pytorch-gan-zoo`(legado) y `StudioGAN`el buque ha probado las implementaciones de DCGAN, WGAN-GP, SN-GAN, StyleGAN y BigGAN.

En 2026, los GAN siguen siendo la mejor opción para: generación de imágenes en tiempo real (latencia <10 ms), transferencia de estilo, traducción de imagen a imagen con control preciso (Pix2Pix, CycleGAN).

## Envío

Esta lección produce:

- `outputs/prompt-gan-training-triage.md` una instrucción que lee una descripción de la curva de entrenamiento y selecciona el modo de falla (colapso de modo, D-win, oscilación) más la única solución recomendada.
- `outputs/skill-dcgan-scaffold.md` una habilidad que escribe un andamio DCGAN de `z_dim`, objetivo`image_size`, y `num_channels`, incluyendo el bucle de entrenamiento y el salvador de muestras.

## Los ejercicios

1. **(Easy)**Entrenar el DCGAN anterior en el conjunto de datos del círculo sintético y guardar una cuadrícula de 16 muestras al final de cada época.
2. **(Medium)**Replace la norma de lote del discriminador con la norma espectral. Entrenar ambas versiones lado a lado. ¿Cuál converge más rápido? ¿Cuál tiene menor variación entre tres semillas?
3. **(Hard)**Implementar un DCGAN condicional: introducir la etiqueta de clase en G y D (concertar un solo calor al ruido en G, concat un canal de incorporación de clase en D). Entrenar el conjunto de datos sintético "círculos vs cuadrados" de la lección 7 y demostrar que el acondicionamiento de clase funciona mediante muestreo con etiquetas específicas.

## Términos clave

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Generator (G) | "The draws-stuff net" | Maps noise to images; trained to fool the discriminator |
| Discriminator (D) | "The critic" | Binary classifier; trained to distinguish real from generated images |
| Minimax | "The game" | min over G, max over D of an adversarial loss; equilibrium is p_G = p_data |
| Non-saturating loss | "The numerically sane version" | G's loss is -log(D(G(z))) instead of log(1 - D(G(z))) to avoid vanishing gradients early in training |
| Mode collapse | "Generator makes one thing" | G produces only a small subset of the data distribution; fix with SN, minibatch discrimination, or larger batch |
| TTUR | "Two learning rates" | D learns faster than G, typically by a factor of 2-4; stabilises training |
| Spectral norm | "1-Lipschitz layer" | A weight-normalisation that bounds each layer's Lipschitz constant; stops D from becoming arbitrarily steep |
| FID | "Fréchet Inception Distance" | Distance between Inception-v3 feature distributions of real and generated sets; the standard evaluation metric |

## Leer más

- [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661) el periódico que comenzó todo
- [DCGAN (Radford, Metz, Chintala, 2015)](https://arxiv.org/abs/1511.06434) las reglas de arquitectura que hicieron que los GAN pudieran ser entrenados
- [Spectral Normalization for GANs (Miyato et al., 2018)](https://arxiv.org/abs/1802.05957) el truco de estabilización más útil
- [StyleGAN3 (Karras et al., 2021)](https://arxiv.org/abs/2106.12423) el SOTA GAN; se lee como un álbum de los mejores éxitos de cada truco de la última década
