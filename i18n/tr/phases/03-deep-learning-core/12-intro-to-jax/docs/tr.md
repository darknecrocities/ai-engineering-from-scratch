# JAX'e Giriş

> PyTorch tensorları mutasyonlandırır. TensorFlow grafikler oluşturur. JAX saf fonksiyonları oluşturur. Sonuncusu derin öğrenme hakkında düşüncelerinizi değiştirir.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## Öğrenme Hedefleri

- JAX'in işlevsel API'si (jax.numpy, jax.grad, jax.jit, jax.vmap) kullanarak saf işlevli sinir ağ kodu yazın
- PyTorch'ın coşkulu mutasyonu ile JAX'in fonksiyonel bir komile modeli arasındaki temel tasarım farkını açıklayın.
- Naif Python'a kıyasla eğitim döngüslerini hızlandırmak için jit kompiliasyonu ve vmap vektörleştirmesini uygulayın
- JAX'de basit bir ağ eğit ve açık durum yönetimini PyTorch'ın nesne odaklı yaklaşımı ile karşılaştır

## Sorun

PyTorch'te sinir ağlarını nasıl inşa edeceğinizi biliyorsunuz.`nn.Module`- Arayın .`.backward()`Milyonlarca insan kullanıyor.

Ama PyTorch'in DNA'sına bir kısıtlama girdi: Python'da operasyonları birer birer seve izliyor.`tensor + tensor`Bu, 540 milyar parametrelik bir modelin 2.048 TPU'da eğitilmesi gerektiği sürece iyi çalışır.

Google DeepMind, Gemini'yi JAX'de eğitir. Anthropic Claude'u JAX'de eğitmiştir. Bunlar küçük operasyonlar değil, bunlar Dünya'daki en büyük sinir ağı eğitim sürümleri. JAX'i seçtiler çünkü eğitim döngüsünü Python çağrılarının bir dizi değil, bir yapılandırılabilir program olarak değerlendirir.

JAX, NumPy'yi üç süper güçle oluşturur: otomatik farklılaşma, JIT'in XLA'ya birleştirilmesi ve otomatik vektörleştirme. Bir örneği işleyen bir işlevi yazarsınız. JAX size bir partiyi işleyen, gradientleri hesaplayan, makine koduna birleştiren ve birden fazla cihazda çalıştırılan bir işlevi verir.

## Anlaşım

### JAX Felsefesi

JAX, işlevsel bir çerçeve.`.backward()`- Bu yöntem yerine:

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

Bu bir stil tercih değil. Bu bir kompilatör kısıtlaması. JIT komisyonu saf işlevler gerektirir - aynı girişler her zaman aynı çıkışlar üretir, yan etkileri yoktur. Bu kısıtlama 100 kat hızlandırmayı mümkün kılan şeydir.

### Jax.numpy: Tanıdık Yüzey

JAX NumPy API'sini hızlandırıcılarda yeniden uyguluyor:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

Aynı işlev isimleri, aynı yayın kuralları, aynı kesim semantikası ama diziler GPU/TPU'da canlı ve her işlem kompiliör tarafından izlenebilir.

Tek önemli fark: JAX dizileri değişmez.`a[0] = 5`Bunun yerine:`a = a.at[0].set(5)`Bu bir hafta boyunca garip geliyor, sonra da basıyor-- değişmezlik dönüşümleri böyle yapar.`grad`- Evet .`jit`ve`vmap`- Düzgün.

### jax.grad: Fonksiyonel Otodiff

PyTorch gradientleri tenzorlara bağlar (`.grad`JAX fonksiyonlara gradient bağlar.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`Bir fonksiyonu alır ve gradiyenti hesaplayan yeni bir fonksiyonu gönderir.`.backward()`Tansörlerde depolanmış hesaplama grafikleri yok. gradient sadece bir başka fonksiyon, çağırabilir, yazabilir veya JIT-kompile edebilirsiniz.

Bu, keyfi bir şekilde oluşturuyor:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

İkinci türevler, üçüncü türevler, Yakobyanlar, Hessyanlar, hepsi de bir araya gelerek.`grad`PyTorch bunu da yapabilir.`torch.autograd.functional.hessian`JAX'de, temel oluşturur.

Zorluk:`grad`Bu işlemler sadece saf fonksiyonlarda çalışır. İçinde baskı açıklamaları yoktur (içinde izleme sırasında çalıştırılır, çalıştırılmaz). Dış durum mutasyonları yoktur. Açık bir anahtar yönetimi olmadan rastgele sayı üretimi yoktur.

### jit: XLA'ya kompile

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

İlk çağrıda JAX fonksiyonu izler. Hangi işlemlerin gerçekleştiğini, uygulanmadan kaydeder. Sonra bu izleri Google'ın TPU ve GPU'lar için kompiler olan XLA'ya verir. XLA işlemleri birleştirir, fazladan hafıza kopyalarını ortadan kaldırır ve optimize edilmiş makine kodu üretir.

Sonraki aramalar Python'ı tamamen atlatır.

JIT yardımcı olduğunda:
- Eğitim adımları (aynı hesaplama binlerce kez tekrarlanır)
- İndirim (aynı model, farklı girişler)
- Benzer şekilli girişlerle birden fazla kez çağrılan herhangi bir fonksiyon

JIT ağrıyınca:
- Python kontrol akışıyla birlikte değerlere bağlı fonksiyonlar (`if x > 0`x izlenmiş bir dizidir)
- Tek çekim hesaplamaları (kompile overhead runtime'den fazla)
- Debugging (içinden izleme gerçek çalıştırmayı gizler)

Kontrol akışı kısıtlaması gerçek.`jax.lax.cond`yerine getirir .`if/else`- Evet .`jax.lax.scan`yerine getirir .`for`Bunlar seçmeli değiller - birleştirme fiyatıdır.

### vmap: Otomatik vektörleştirme

Bir örneği işleyen bir işlevi yazıyorsunuz:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`bir partiyi işlemek için kaldırır:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`Yöntem: toplanmayın `params`(tükeltilmiş),    0      `x`- Yönlük yok .`for`Bu yüzden, bu birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, birim, bir, birim, bir, birim, bir, bir, bir, birim, bir, bir, birim, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir, bir

Bu sintaksik şeker değil.`vmap`Python döngüsünden 10-100 kat daha hızlı çalışan birleşik vektörlü kod üretir.`jit`ve `grad`- ...

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Bir çizgi, PyTorch'de hack olmadan neredeyse imkansız.

### pmap: Cihazlardaki Veri Paralelisi

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`işlevi tüm mevcut cihazlarda (GPU/TPU) kopyalar ve partiyi bölür.`jax.lax.pmean`ve `jax.lax.psum`cihazlar arasında gradientleri senkronize etmek.

Google Gemini ' yi binlerce TPU v5e çipten kullanıyor .`pmap`(ve onun varisi)`shard_map`Programlama modeli: tek cihazlı versiyonu yazın, `pmap`- Tamam.

### Pytrees: Evrensel Veriler Yapısı

JAX, "pytrees" üzerinde çalışır. Liste, tuples, dicts ve arraylerin yuva içindeki kombinasyonları.

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

JAX ' in her dönüşümü ...`grad`- Evet .`jit`- Evet .`vmap`- Pytrees'i nasıl geçeceğini biliyor.`jax.tree.map(f, tree)`uygulanır .`f`Optimizeciler tüm parametreleri aynı anda bu şekilde güncelleyebilir:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

Hayır .`.parameters()`Metod. Parametre kayıt yok. Ağaç yapısı model.

### İşlevsel vs. Nesne odaklı

PyTorch depoları nesnelerin içinde şunları belirtir:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX açık durumlu saf fonksiyonları kullanır:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Parameler aktarılır. Hiçbir şey saklanmaz. Hiçbir şey mutasyonlanmaz. Bu her işlevi test edilebilir, yapılandırılabilir ve kompile edilebilir hale getirir. Bu aynı zamanda paramları kendiniz yönetmeniz anlamına gelir.

### JAX Ekosistemi

JAX size ilkellikler verir.

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | `nn.Module` with explicit state |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

Optax standart optimizer kütüphanesi. Bu, gradient dönüşümünü (Adam, SGD, kesim) parametreler güncelleme ile ayırır ve bunları oluşturmak önemsiz hale gelir:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### JAX vs PyTorch ne zaman kullanılır

| Factor | JAX | PyTorch |
|--------|-----|---------|
| TPU support | First-class (Google built both) | Community-maintained (torch_xla) |
| GPU support | Good (CUDA via XLA) | Best-in-class (native CUDA) |
| Debugging | Hard (tracing + compilation) | Easy (eager, line-by-line) |
| Ecosystem | Research-focused (Flax, Equinox) | Massive (HuggingFace, torchvision, etc.) |
| Hiring | Niche (Google/DeepMind/Anthropic) | Mainstream (everywhere) |
| Large-scale training | Superior (XLA, pmap, mesh) | Good (FSDP, DeepSpeed) |
| Prototyping speed | Slower (functional overhead) | Faster (mutate and go) |
| Production inference | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Who uses it | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

Dürüstçe cevap: JAX'i kullanmak için belirli bir nedeniniz yoksa PyTorch kullanın. Bu nedenler TPU erişim, örnek başına gradient gereksinimleri, çok cihazlı eğitim büyük ölçekte veya Google/DeepMind/Anthropic'te çalışmak.

### JAX'de rastgele sayılar

JAX'in küresel bir rastgele durumu yoktur. Her rastgele işlem açık bir PRNG anahtarı gerektirir:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

Bu ilk başta sinir bozucu ama PyTorch'un ürettiği bir özellik olan cihazlar ve komileler arasında yeniden üretilebilirliği garanti ediyor.`torch.manual_seed`Çoklu GPU ayarlarında garanti yapamaz.

```figure
batchnorm-effect
```

## Yapın

### Adım 1: Kurulum ve Veriler

JAX ve Optax kullanarak MNIST'de 3 katlı bir MLP eğitime geçeceğiz. 784 giriş, 256 ve 128 nöronun iki gizli katmanı, 10 çıkış sınıfı.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Adım 2: Parametreyi başlat

Bir sınıf yok, sadece bir Pytree'yi geri veren bir fonksiyon:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

Bir tohumdan ayrı üç PRNG anahtarı, her ağırlık, bir yastık dikte bir değişmez dizidir.

### Üçüncü Adım: Önceden Geç

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Temiz fonksiyonlar, paramlar, tahminler.`self`, hiç depolama durumu yok. `loss_fn`sıfırdan çapraz entropi hesaplar -- softmax, log, negatif ortalama.

### Dördüncü Adım: JIT'den oluşan Eğitim Adımı

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad`Bir geçişle hem kayıp değerini hem de eğilimi geri verir.`@jax.jit`Bu programın ilk çağrısı sonrasında, her eğitim adımı Python'a dokunmadan çalışır.

### Adım 5: Eğitim Çubuğu

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 dönem. ~ 97% test doğruluğu. İlk dönem yavaş (JIT kompili) Epochs 2-10 hızlı.

Ne eksik olduğunu fark et: hayır `.zero_grad()`Hayır .`.backward()`Hayır .`.step()`Tüm güncelleme bir fonksiyon çağrısıdır. Gradiyentler hesaplanır, Adam tarafından dönüştürülür ve parametrelere uygulanır.`train_step`- Evet .

## Kullan

### İpek: Google Standartı

Flax, JAX sinir ağı kütüphanesi en yaygın kütüphanesi.`nn.Module`Geriye, ama açık bir devlet yönetimiyle:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

PyTorch'la aynı yapı, ama `params`modelden ayrıdır. `model.init()`Param yaratıyor.`model.apply(params, x)`Model nesne durumsuz.

### Dönem: Pythonik Alternatif

Ekvinoks (Patrick Kidger tarafından) modeller pytrees olarak temsil edilir:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

Modelin kendisi bir pytree.`.apply()`Bu, JAX'in düşüncesine daha yakındır.

### Optax: Yapılandırılabilir Optimizerler

Optax, gradient dönüşümünü güncelleme ile ayırır:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Gradient kesimi, öğrenme hızının yükselmesi, ağırlık kaybı - hepsi bir değişim zinciri olarak oluşur. Her değişim gradientleri görür, değiştirir ve bir sonrakiye aktarır. Monolit optimizer sınıfı yoktur.

## Gönder

**Installation:**

```bash
pip install jax jaxlib optax flax
```

GPU desteği için:

```bash
pip install jax[cuda12]
```

TPU için (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Performance gotchas:**

- İlk JIT çağrısı yavaş (tümlenme).
- JIT'in içindeki JAX dizinleri üzerinde Python döngüslerinden kaçının.`jax.lax.scan`veya `jax.lax.fori_loop`- Evet .
- `jax.debug.print()`JIT'de çalışmaktadır.`print()`- Hayır.
- Profil `jax.profiler`XLA komisyonu şişek boğazlarını saklayabilir.
- JAX, öntanımlı olarak GPU belleğinin %75'ini önceden ayırır.`XLA_PYTHON_CLIENT_PREALLOCATE=false`- İptal etmek için.

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**This lesson produces:**
- `outputs/prompt-jax-optimizer.md`-- doğru JAX optimizer yapılandırmasını seçmek için bir ipucu
- `outputs/skill-jax-patterns.md`-- JAX'de fonksiyonel kalıpları kapsayan bir beceri

## Egzersizler

1. MLP'ye düşüş ekleyin. JAX'de düşüş bir PRNG anahtarı gerektirir. Bir anahtarı ileri geçişten geçerek her düşüş katman için bölün. Test doğruluğunu ile ve dışına karşılaştırın.

2. Kullanım`jax.vmap`32 MNIST görüntüler için örnek başına gradient hesaplamak için. Her örnek için gradient normunu hesaplayın. Hangi örneklerde en büyük gradient var ve neden?

3. Elci önleme fonksiyonunu genel bir fonksiyonla değiştirin `mlp_forward(params, x)`Bu, herhangi bir katman için çalışır.`jax.tree.leaves`derinliği otomatik olarak belirlemek için.

4. Eğitim adımını birlikte ve olmadan göster `@jax.jit`Her seferinde 100 adım yapın. Hardware'daki hızlanmanın ne kadarı var?

5. Yapıştırarak gradient kesimi uygulayın `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`Eğitimle ve kesmeden, etkisini görmek için eğitiş üzerinde eğilime normunu çiz.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| XLA | "The thing that makes JAX fast" | Accelerated Linear Algebra -- a compiler that fuses operations and generates optimized GPU/TPU kernels from a computation graph |
| JIT | "Just-in-time compilation" | JAX traces the function on first call, compiles to XLA, then runs the compiled version on subsequent calls |
| Pure function | "No side effects" | A function where the output depends only on inputs -- no global state, no mutation, no randomness without explicit keys |
| vmap | "Auto-batching" | Transforms a function that processes one example into one that processes a batch, without rewriting |
| pmap | "Auto-parallelism" | Replicates a function across multiple devices and splits the input batch |
| Pytree | "Nested dict of arrays" | Any nested structure of lists, tuples, dicts, and arrays that JAX can traverse and transform |
| Tracing | "Recording the computation" | JAX executes the function with abstract values to build a computation graph, without computing real results |
| Functional autodiff | "grad of a function" | Computing derivatives by transforming functions, not by attaching gradient storage to tensors |
| Optax | "JAX's optimizer library" | A composable library of gradient transformations -- Adam, SGD, clipping, scheduling -- that chain together |
| Flax | "JAX's nn.Module" | Google's neural network library for JAX, adding layer abstractions while keeping state explicit |

## Daha Fazla Okumak

- JAX belgesi: https://jax.readthedocs.io/- ... resmi doktorlar, yüksek lisans, JIT ve Vmap hakkında mükemmel dersler.
- "JAX: Python+NumPy programlarının yapılandırılabilir dönüşümleri" (Bradbury et al., 2018) - tasarım felsefesini açıklayan orijinal makale
- İpek belgesi: https://flax.readthedocs.io/-- JAX için Google'ın sinir ağ kütüphanesi
- Patrick Kidger, "Equinox: JAX'deki sinir ağları, çağrılabilir PyTrees ve filtreli dönüşümler aracılığıyla" (2021) -- Flax'in Pythonik alternatifidir
- DeepMind, "Optax: Composable gradient transformation and optimization" -- standart optimizer kütüphanesi
- "You Don't Know JAX" (Colin Raffel, 2020) - T5 yazarlarından birinin JAX gotchas ve kalıplarına yönelik pratik bir rehber
