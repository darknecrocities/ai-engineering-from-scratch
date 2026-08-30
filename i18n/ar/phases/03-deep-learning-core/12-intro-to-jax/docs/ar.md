# مقدمة للـ JAX

> (بيتورش) يُطفر الجهازات العصبية، (تنسور فلو) يُبني الرسومات، (جاكس) يُجمع وظائف نقية، وهذا الأخير يُغير طريقة تفكيرك بشأن التعلم العميق.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## أهداف التعلم

- كتابة رمز شبكة عصبية ذات وظيفة نقية باستخدام API الوظيفية JAX (jax.numpy، jax.grad، jax.jit، jax.vmap)
- شرح الفرق الرئيسية في التصميم بين طفرة PyTorch المهتمة ونموذج تجميع وظيفي JAX
- تطبيق تجميع jit و vmap التنقل لتسريع حلقات التدريب مقارنة مع بيثون ساذجة
- تدريب شبكة بسيطة في JAX وتقارن إدارة الحالة الصريحة مع نهج PyTorch المتحرك على الكائن

## المشكلة

أنت تعرف كيفية بناء شبكات عصبية في PyTorch.`nn.Module`، اتصل`.backward()`،إضغط على المحفّظ، إنه يعمل، ملايين الناس يستخدمونها

لكن (بيتورش) لديه قيود في حمض النووي: يتتبع العمليات بحماس، واحدة في كل مرة، في (بايتون).`tensor + tensor`كل خطوة تدريبية تعيد تفسير نفس رمز Python هذا يعمل بشكل جيد حتى تحتاج إلى تدريب نموذج بمليارات 540 على 2048 TPU ثم التكلفة العليا تقتلك

غوجل ديب ميند تدرب جيمين على جياكس. علمت أنتروبيك كلود على جياكس. هذه ليست عمليات صغيرة - إنها أكبر عمليات تدريب شبكة عصبية على الأرض. اختاروا جياكس لأنه يعامل حلقة التدريب الخاصة بك كبرنامج قابل للتجميع، وليس تسلسل من مكالمات بيثون.

JAX هو NumPy مع ثلاثة قوى خارقة: التمييز التلقائي، تجميع JIT إلى XLA، والمتصفح التلقائي. تكتب وظيفة التي تعالج مثال واحد. JAX يعطيك وظيفة التي تعالج مجموعة، وحسب التراجع، وتقوم بتجميعها إلى رمز الآلة، وتشغيل عبر أجهزة متعددة. كل ذلك دون تغيير الوظيفة الأصلية.

## المفهوم

### فلسفة جياكس

JAX هو إطار عمل لا فئات ولا حالة متغيرة لا`.backward()`الطريقة بدلاً من ذلك:

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

هذه ليست تفضيل الأسلوب. إنها قيود للمؤلف. تجميع JIT يتطلب وظائف نقية -- نفس المدخلات دائما تنتج نفس المخرجات، لا آثار جانبية. هذا القيود هو ما يجعل 100x السرعة ممكنة.

### jax.numpy: السطح المألوف

يقوم JAX بإعادة تنفيذ API NumPy على المسارع:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

نفس أسماء الوظائف نفس قواعد البث نفس النقشة الزمنية ولكن المجموعات تعيش على GPU / TPU وكل عملية تتبع بواسطة المُحْتَرِك

الفرق الحاسم واحد: صفوف JAX غير قابلة للتغيير.`a[0] = 5`بدلاً من ذلك:`a = a.at[0].set(5)`هذا يشعرني بالشعور الحرج لمدة أسبوع، ثم يضغط -- عدم التغيير هو ما يجعل التحولات مثل`grad`،`jit`و`vmap`قابلة للتحكم

### jax.grad: التشغيل الذاتي الوظيفي

يربط PyTorch المراجع إلى الجهاز (`.grad`JAX يربط التراجع بالعملات

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`يأخذ وظيفة ويرد وظيفة جديدة تحسب التراجع. لا `.backward()`لا يوجد مخطط حساب مخزن على الجهاز التنسوري. التراجع هو مجرد وظيفة أخرى يمكنك الاتصال، أو إعداد، أو JIT-إعداد.

هذا يشكّل بشكل تعسفي:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

المشتقات الثانية المشتقات الثالثة الجيكوبيون الهيسيون كل ذلك من خلال التركيب`grad`(بيتورش) يمكنه فعل هذا أيضاً`torch.autograd.functional.hessian`في JAX، هو الأساس.

القيود:`grad`لا توجد بيانات مطبوعة داخل (تتم تشغيلها أثناء التتبع وليس التنفيذ). لا توجد طفرة في الحالة الخارجية. لا توجد إنتاج أرقام عشوائية دون إدارة مفتاح صريحة.

### jit: قم بتجميع XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

في المكالمة الأولى، تتبع JAX الوظيفة -- تسجل العمليات التي تحدث، دون تنفيذها. ثم تمنح ذلك التتبع إلى XLA (الجيبر الخطوي المتسارع) ، محفز جوجل لـ TPUs و GPUs. XLA يدمج العمليات، يزيل نسخة ذاكرة زائدة، ويولد رمز آلة محسن.

المكالمات اللاحقة تتجاوز Python بالكامل. يتم تشغيل الكود المجمّع على المسرّع بسرعة C ++.

عندما يساعد JIT:
- خطوات التدريب (تكرر نفس الحساب آلاف المرات)
- الإستدلال (الموديل نفسه، المدخلات المختلفة)
- أي وظيفة تُدعى أكثر من مرة مع مدخلات ذات شكل مماثل

عندما يؤلم الجيت:
- وظائف مع تدفق التحكم في Python التي تعتمد على القيم (`if x > 0`حيث x هو صف متتبع)
- الحسابات المقطوعة الواحدة (تكلفة التجميع العلوية تتجاوز وقت التشغيل)
- إزالة الأخطاء (التتبع يخفي التنفيذ الفعلي)

قيود تدفق التحكم حقيقية`jax.lax.cond`يُستبدل`if/else`. .`jax.lax.scan`يُستبدل`for`هذه ليست اختياريّة، إنها ثمن التجميع.

### vmap: التنقل الآلي

تكتب وظيفة تعالج مثال واحد:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`يرفعه لتحويل دفعة:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`الوسائل: لا تتعب`params`(مشتركة) ، اللحظة فوق محور 0 من `x`لا دليل`for`لا تشكيل إعادة تشكيل، لا خيوط بعدة دفعة، يحدد JAX بعدة دفعة ويتم تحديد الحساب بأكمله

هذا ليس سكرًا صيغيًا`vmap`يخلق رمز متجه مزامج يعمل 10-100 مرة أسرع من حلقة Python. ويتكون من`jit`و`grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

على سبيل المثال، خط واحد، هذا مستحيل تقريباً في (بيتورش) بدون حوادث

### pmap: مواقعة البيانات عبر الأجهزة

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`يكرر الوظيفة عبر جميع الأجهزة المتاحة (GPUs / TPUs) ويقسّم المجموعة. داخل الوظيفة، `jax.lax.pmean`و`jax.lax.psum`التزامن بين التراجع بين الأجهزة.

جوجل تدرب التوأم عبر آلاف رقائق TPU v5e باستخدام `pmap`(و خليفته)`shard_map`النموذج التخطيطي: كتابة نسخة جهاز واحد، وترتيبها`pmap`لقد انتهى

### (بيتريس) - بنية البيانات العالمية

يعمل JAX على "شجرة البيترا" -- مزيجات متجمعة من القوائم والجماعات والقوائم والصفوف.

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

كل تحول من (جاكس)`grad`،`jit`،`vmap`-يعرف كيف يعبر الشجرة`jax.tree.map(f, tree)`يطبق`f`هكذا يقوم المحسنون بتحديث جميع المعلمات في وقت واحد:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

لا , لا`.parameters()`لا يوجد تسجيل لمعلمات، بنية الشجرة هي النموذج

### المهام مقابل المستهدفة للأشياء

مخازن PyTorch تصريح داخل الأشياء:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX يستخدم وظائف نقية مع حالة صريحة:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

يتم نقل المعلمات. لا يتم تخزين أي شيء. لا يوجد شيء يتغير. وهذا يجعل كل وظيفة قابلة للتحقق، قابلة للتكوين، ويمكن تجميعها. وهذا يعني أيضًا أنك تدير المعلمات بنفسك - أو تستخدم مكتبة مثل فلانكس أو إيكوينوكس.

### النظام البيئي JAX

"جاكس" يعطيك البدائية، والمكتبات تعطيك التجربة

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | `nn.Module` with explicit state |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

أوبتاكس هو مكتبة المحفز القياسي. فإنه يفصل تحويل التراجع (أدام، SGD، قطع) من تحديث المعلمات، مما يجعل من غير المهم أن تكوين:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### متى تستخدم JAX مقابل PyTorch

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

الجواب الصادق: استخدم PyTorch ما لم يكن لديك سبب محدد لاستخدام JAX. هذه الأسباب هي -- وصول TPU، الحاجة إلى تراجع لكل مثال، تدريب متعدد الأجهزة على نطاق واسع، أو العمل في Google / DeepMind / Anthropic.

### أرقام عشوائية في JAX

لا يوجد في JAX حالة عشوائية عالمية. كل عملية عشوائية تتطلب مفتاح PRNG صريح:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

هذا مزعج في البداية، لكنه يضمن إعادة التأهيل عبر الأجهزة والجمعات -- خاصية التي يستخدمها PyTorch`torch.manual_seed`لا يمكن أن تضمن في إعدادات متعددة GPU.

```figure
batchnorm-effect
```

## بناءها

### الخطوة الأولى: الإعداد والبيانات

سنقوم بتدريب 3 طبقات من المعلمين على نظام التعلم المختص بالمنظمة باستخدام JAX و Optax 784 مدخلات، طبقتين مخفيتين من 256 و 128 عصبية، 10 فصول خروج.

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

### الخطوة الثانية: إعادة تشغيل المعايير

لا يوجد فئة، مجرد وظيفة تعيد ثمرة:

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

يبدأ بإعداد المفاتيح اليدوية ثلاثة مفاتيح PRNG منفصلة من بذرة واحدة كل وزن هو صف غير قابل للتغيير في إصدار مقيم

### الخطوة الثالثة: التقدم

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

وظائف نقية، إدخال المعلمات، تنبؤ خارج.`self`لا يوجد حالة تخزين`loss_fn`يحسب الانتروبيا المتقاطعة من الصفر -- softmax، log، متوسط سلبي.

### الخطوة الرابعة: خطوة تدريبية مرتبة من JIT

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

`jax.value_and_grad`يعيد قيمة الخسارة والتحركات في مرور واحد.`@jax.jit`يقوم المعدل بتجميع كل من الوظائف إلى XLA. بعد الدعوة الأولى، يتم تشغيل كل خطوة تدريبية دون لمس Python.

### الخطوة 5: حلقة التدريب

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

10 أوقات. ~ 97٪ دقة الاختبار. الأوقات الأولى بطيئة (جميع JIT). أوقات 2-10 سريعة.

لاحظ ما يفتقد: لا`.zero_grad()`لا , لا`.backward()`لا , لا`.step()`التحديث كله هو دعوة عمل واحد مركب. يتم حساب المعدلات، وتحوّلها من قبل آدم، وتطبيقها على المعلمات -- كل شيء داخل`train_step`. . .

## استخدمها

### اللون: معيار جوجل

فلانكس هو المكتبة الأكثر شيوعا للشبكة العصبية JAX.`nn.Module`ولكن مع إدارة دولية صريحة:

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

نفس الهيكل مثل (بيتورش) ، لكن`params`هو منفصل عن النموذج. `model.init()`يخلق "بارام".`model.apply(params, x)`يُجري المُسلّم الأمامي، وكلّ هذا المُسلّم لا يملك حالة.

### التوالي: البديل البايتوني

يمثل الإيكنوكس (من قبل باتريك كيدجر) النماذج كثلاثة أشجار:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

النموذج نفسه هو شجرة`.apply()`هذه أقرب إلى طريقة تفكير جياكس

### أوتتاكس: أوتيتيمات قابلة للتكوين

أوتتاكس يقطع صيغة التحول من التحديث:

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

قطع التدرج، ارتفاع معدل التعلم، تدهور الوزن -- كل ذلك مركب كسلسلة من التحولات. كل تحويل يرى التدرج، يغيره، ويمر به إلى التالي. لا يوجد فئة تحسينات واحدة.

## أرسله

**Installation:**

```bash
pip install jax jaxlib optax flax
```

لدعم GPU:

```bash
pip install jax[cuda12]
```

بالنسبة لـ TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Performance gotchas:**

- أول مكالمة JIT بطيئة (التجميع) ، احترق قبل تقييم الموازنة.
- تجنب حلقات Python على صفوف JAX داخل JIT. استخدم `jax.lax.scan`أو`jax.lax.fori_loop`. . .
- `jax.debug.print()`يعمل داخل JIT.`print()`لا يُمكنك ذلك
- الملف الشخصي مع `jax.profiler`أو TensorBoard. تجميع XLA يمكن أن تخفي ضباب الزجاجة.
- JAX يخصص 75٪ من ذاكرة GPU من قبل افتراضي.`XLA_PYTHON_CLIENT_PREALLOCATE=false`لتعطيل.

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**This lesson produces:**
- `outputs/prompt-jax-optimizer.md`-- طلب لتحديد التكوين المناسب لشركة JAX
- `outputs/skill-jax-patterns.md`-- مهارة تغطي أنماط وظيفية في JAX

## التمارين

1. إضافة التوقف إلى MLP. في JAX، يتطلب التوقف مفتاح PRNG -- خيط مفتاح من خلال الممر الأمامي وتقسيمه لكل طبقة التوقف. مقارنة دقة الاختبار مع وخارج.

2. استخدام`jax.vmap`لتحديد تراجع لكل مثال لـ 32 صورة من MNIST. احسب معايير تراجع لكل مثال. أي أمثلة لديها أكبر تراجع، ولماذا؟

3. استبدل وظيفة الإستعراض اليدوية بالظروف العامة`mlp_forward(params, x)`يعمل على أي عدد من الطبقات. استخدم `jax.tree.leaves`لتحديد العمق تلقائيًا.

4. قم بتحديد خطوة التدريب مع و بدون`@jax.jit`كم عدد التسريع على أجهزةك؟ ما هو تكلفة التجميع على المكالمة الأولى؟

5. تنفيذ قطع التراجع عن طريق التركيب `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`تدريب مع و بدون قطع رسم معايير التراجع فوق التدريب لمعرفة التأثير

## الشروط الرئيسية

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

## المزيد من القراءة

- وثائق JAX: https://jax.readthedocs.io/- الأدب الرسمي، مع دروس ممتازة على درجة، jit، و vmap
- "جاكس: تحولات قابلة للتكوين من برامج بايثون+نومبي" (برادبري وآخرون، 2018) -- الورقة الأصلية التي تشرح فلسفة التصميم
- وثائق من الكتان: https://flax.readthedocs.io/-- مكتبة شبكة عصبية جوجل لل JAX
- باتريك كيدجر، "الإنكوانوكس: شبكات عصبية في JAX عبر PyTrees المزعومة والتحولات المصفاة" (2021) -- البديل البايتوني للفني
- ديب ميند، "أوبتاكس: تحويل وتحسين التراجع المتكامل" -- مكتبة المحسن القياسي
- "أنت لا تعرف جياكس" (كولين رافيل، 2020) -- دليل عملي على جاكس gotchas والأنماط، من أحد مؤلفي T5
