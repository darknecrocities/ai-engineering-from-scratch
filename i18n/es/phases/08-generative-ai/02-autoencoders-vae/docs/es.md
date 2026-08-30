# Autoencodadores y autoencodadores variativos (VAE)

> Un autoencoder simple comprime y luego reconstruye. Memora. No genera. Añade un truco  fuerza el código para que parezca gaussiano  y obtienes un muestreo. Ese truco único, la reparameterización de `z = μ + σ·ε`, es por eso que cada modelo de difusión latente y de coincidencia de flujo de imagen que utilices en 2026 tiene un VAE en la entrada.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 07 (CNNs), Phase 8 · 01 (Taxonomy)
**Time:** ~75 minutes

## El problema

Comprimir un dígito MNIST de 784 píxeles a un código de 16 números, luego reconstruir. Un autoencoder simple hará la reconstrucción de MSE pero el espacio de código es un lío agudo. Elige un punto aleatorio en el espacio de código, decodifica, y obtienes ruido. No tiene muestreo. Es un modelo de compresión disfrazado.

Lo que realmente quieres es: (a) el espacio de código es una distribución limpia y suave que puedes tomar de un isotrópico gaussiano`N(0, I)`, (b) la descifrada de cualquier muestra produce un dígito plausible, y (c) el codificador y el decodificador todavía comprimen bien.

El VAE de Kingma 2013 resuelve esto entrenando al codificador para emitir una *distribución* `q(z|x) = N(μ(x), σ(x)²)`, tirando esa distribución hacia el prior`N(0, I)`a través de una penalización KL, y luego muestreo `z`de la`q(z|x)`En el momento de la inferencia, deja caer el codificador, muestra`z ~ N(0, I)`La pena KL es lo que obliga a estructurar el espacio de código.

En 2026 los VAEs rara vez envían de forma independiente  han sido superados por difusión por calidad de imagen en bruto  pero son el codificador de elección para cada modelo de difusión latente (SD 1/2/XL/3, Flux, AudioCraft). Aprende el VAE y aprendes la primera capa invisible de cada pipeline de imágenes que utilizas.

## El concepto

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**Autoencoder.** `z = encoder(x)`¿ Qué ?`x̂ = decoder(z)`, pérdida = `||x - x̂||²`- El espacio de código no está estructurado.

**VAE encoder.**Salidas de dos vectores: `μ(x)`y `log σ²(x)`- Estos definen .`q(z|x) = N(μ, diag(σ²))`¿ Qué ?

**Reparameterization trick.**Muestreo de `q(z|x)`No es diferenciable.`z = μ + σ·ε`donde`ε ~ N(0, I)`Ahora .`z`es una función determinista de `(μ, σ)`más un ruido no parámetro  flujo de gradientes `μ`y `σ`¿ Qué ?

**Loss.**Evidencia Bando inferior (ELBO), dos términos:

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

La reconstrucción impulsa`x̂`hacia`x`KL empuja .`q(z|x)`El primer tipo de muestras de desintegración de los átomos de la base de datos de la base de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos de datos

**Sampling.**En la inferencia: dibujar `z ~ N(0, I)`Una pasada hacia adelante, sin muestreo iterativo como la difusión.

```figure
vae-latent-grid
```

## Construye el mismo

`code/main.py`Implementa una pequeña VAE sin numpy o antorcha. La entrada es un dato sintético de 8 dimensiones extraído de una mezcla gaussiana de 2 componentes en 8D. El codificador y el decodificador son MLPs de capa oculta única. Implementamos activación tanh, pase hacia adelante, pérdida y un pase hacia atrás escrito a mano. No la producción  pedagogía.

### Paso 1: codificador hacia adelante

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

`log σ²`en lugar de`σ`Así que la salida de la red no está limitada (softplus de σ es una trampa  los gradientes mueren en σ ≈ 0).

### Paso 2: reparametrizar y decodificar

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### Paso 3: El ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

La gente todavía envía código con las estimaciones de Monte-Carlo KL en 2026  es 3 veces más lento sin razón.

### Paso 4: generar

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

Es el modelo generativo. Cinco líneas.

## Las trampas

- **Posterior collapse.**Dispositivos de término KL `q(z|x) → N(0, I)`tan agresivamente que`z`No lleva información sobre `x`. Corrección: β-annealing (inicio β=0, rampa a 1), bits libres, o saltar el KL en dimensiones inactivas.
- **Blurry samples.**La probabilidad del decodificador gaussiano implica la reconstrucción de MSE, que es Bayes-óptima para L2 (la media)  la media de un conjunto de dígitos plausibles es un dígito borroso.
- **β too large, too early.**Ver colapso posterior. Comienza en β≈0.01 y rampa.
- **Latent dim too small.**16-D funciona para MNIST, 256-D para ImageNet 2562, 2048-D para ImageNet 10242. La VAE de la difusión estable comprime 512×512×3 → 64×64×4 (32x factor de muestra baja en área espacial, 32x en canales).

## Usalo

La pila de VAE 2026:

| Situation | Pick |
|-----------|------|
| Image-latent encoder for diffusion | Stable Diffusion VAE (`sd-vae-ft-ema`) or Flux VAE |
| Audio-latent encoder | Encodec (Meta), SoundStream, or DAC (Descript) |
| Video latents | Sora's spatiotemporal patches, Latte VAE, WAN VAE |
| Disentangled representation learning | β-VAE, FactorVAE, TCVAE |
| Discrete latents (for transformer modelling) | VQ-VAE, RVQ (ResidualVQ) |
| Continuous latents for generation | Plain VAE, then condition a flow/diffusion model in that latent space |

Un modelo de difusión latente es un modelo de difusión que vive entre un codificador y un decodificador. El modelo de difusión hace la compresión gruesa, el modelo de difusión hace el levantamiento pesado.

## Envío

Salva .`outputs/skill-vae-trainer.md`¿ Qué ?

Tome habilidades: perfil de conjunto de datos + objetivo latente-dim + uso en aguas subterráneas (reconstrucción, muestreo o entrada de difusión latente) y resultados: elección de arquitectura (plan/β/VQ/RVQ), programa β, latente dim, probabilidad de decodificación (Gaussian vs categorical), y plan de evaluación (recon MSE, KL por dim, distancia Fréchet entre `q(z|x)`y `N(0, I)`¿Qué es lo que se hace?

## Los ejercicios

1. **Easy.**Cambiar`β`En el`code/main.py`¿ Qué ?`0.01`¿ Qué ?`0.1`¿ Qué ?`1.0`¿ Qué ?`5.0`. Graba la reconstrucción final de MSE y KL. ¿Cuál β es mejor para sus datos sintéticos?
2. **Medium.**Reemplazar la probabilidad de descodificación gaussiana con una probabilidad de Bernoulli (pérdida de entropía cruzada). Comparar la calidad de la muestra en una versión binaria de los mismos datos sintéticos.
3. **Hard.**Extenderse`code/main.py`en un mini VQ-VAE: sustituir el continuo `z`Comparar la reconstrucción de MSE y informar cuántas entradas de código se utilizan (el colapso del código es real).

## Términos clave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Autoencoder | Encode-decode network | `x → z → x̂`, learn MSE. Not generative. |
| VAE | AE with a sampler | Encoder outputs a distribution, KL penalty shapes code space. |
| ELBO | Evidence lower bound | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`; tight when `q = p(z\|x)`. |
| Reparameterization | `z = μ + σ·ε` | Rewrites stochastic node as deterministic + pure noise. Enables backprop through sampling. |
| Prior | `p(z)` | Target distribution for the latent, typically `N(0, I)`. |
| Posterior collapse | "KL term wins" | Encoder ignores `x`, outputs the prior; decoder must hallucinate. |
| β-VAE | Tunable KL weight | `loss = recon + β·KL`. Higher β = more disentangled but blurrier. |
| VQ-VAE | Discrete latent | Replace continuous `z` with nearest codebook vector; enables transformer modelling. |

## Nota de producción: el VAE es el camino más caliente en un servidor de difusión

En una línea de flujo / flujo / SD3 estable, el VAE se llama dos veces por solicitud  una vez para codificar (si se hace img2img / inpainting) y una vez para decodificar. En 10242 el pase de decodificador es a menudo el pico de memoria de activación más grande en toda la línea porque muestra`128×128×16`Los latentes de vuelta a `1024×1024×3`Dos consecuencias prácticas:

- **Slice or tile the decode.** `diffusers`expone `pipe.vae.enable_slicing()`y `pipe.vae.enable_tiling()`. Tiling comercializa un pequeño artefacto de costura para`O(tile²)`memoria en lugar de `O(H·W)`Es esencial para 10242+ en GPUs de consumo.
- **bf16 decoder, fp32 numerics for the final resize.**El SD 1.x VAE fue lanzado en fp32 y *produce silenciosamente NaNs* cuando se lanza a fp16 en 10242+. buques SDXL `madebyollin/sdxl-vae-fp16-fix` siempre prefiere la variante fp16-fix o utilizar bf16.

## Leer más

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) el documento de la AEV.
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) desentrañada β-VAE.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) VQ-VAE.
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) Imagen de última generación de VAE.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) Difusión estable; VAE como codificador.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) Encodec, el estándar de audio VAE.
