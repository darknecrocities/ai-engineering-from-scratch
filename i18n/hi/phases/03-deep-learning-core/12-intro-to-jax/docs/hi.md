# जैक्स का परिचय

> पायटॉर्च टेन्सर को उत्परिवर्तन करता है. टेन्सरफ्लो ग्राफ बनाता है. जैक्स शुद्ध कार्यों को संकलित करता है. अंतिम एक गहराई से सीखने के बारे में आपके विचार को बदलता है.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## सीखने के लक्ष्य

- JAX के कार्यात्मक API का उपयोग करके शुद्ध-कार्य तंत्रिका नेटवर्क कोड लिखें (jax.numpy, jax.grad, jax.jit, jax.vmap)
- PyTorch के उत्सुक उत्परिवर्तन और JAX के कार्यात्मक संकलन मॉडल के बीच मुख्य डिजाइन अंतर समझाएं
- निष्क्रिय पायथन की तुलना में प्रशिक्षण लूपों को तेज करने के लिए jit संकलन और vmap वेक्टरिज़ेशन लागू करें
- JAX में एक सरल नेटवर्क को प्रशिक्षित करें और PyTorch के ऑब्जेक्ट-उन्मुख दृष्टिकोण के साथ स्पष्ट राज्य प्रबंधन का विपरीत करें

## समस्या

आप जानते हैं कि कैसे PyTorch में तंत्रिका नेटवर्क बनाने के लिए. आप एक परिभाषित करते हैं`nn.Module`, कॉल `.backward()`यह काम करता है, लाखों लोग इसका इस्तेमाल करते हैं।

लेकिन PyTorch के पास एक प्रतिबंध है जो उसके डीएनए में है: यह पायथन में ऑपरेशन को उत्सुकता से ट्रैक करता है, एक-एक करके।`tensor + tensor`यह ठीक काम करता है जब तक आप एक 540 अरब पैरामीटर मॉडल को प्रशिक्षित करने की जरूरत है 2,048 TPUs पर. तो ओवरहेड आप मार देता है.

गूगल डीपमाइंड जेएएक्स पर जुड़वां प्रशिक्षित करता है। मानव ने क्लाउड को जेएएक्स पर प्रशिक्षित किया। ये छोटे ऑपरेशन नहीं हैं - ये पृथ्वी पर सबसे बड़े तंत्रिका नेटवर्क प्रशिक्षण रन हैं। उन्होंने जेएएक्स का चयन किया क्योंकि यह आपके प्रशिक्षण लूप को एक संकलित प्रोग्राम के रूप में व्यवहार करता है, पायथन कॉल के अनुक्रम के रूप में नहीं।

JAX तीन सुपर पावर के साथ NumPy हैः स्वचालित भिन्नता, JIT संकलन XLA, और स्वचालित वेक्टरिज़ेशन। आप एक फ़ंक्शन लिखते हैं जो एक उदाहरण को संसाधित करता है। JAX आपको एक फ़ंक्शन देता है जो एक बैच को संसाधित करता है, ग्रेडिएंट की गणना करता है, मशीन कोड के लिए संकलित करता है, और कई उपकरणों पर चलता है। मूल फ़ंक्शन को बदलने के बिना।

## अवधारणा

### जैक्स दर्शन

JAX एक कार्यात्मक ढांचा है. कोई वर्गों, कोई परिवर्तनशील राज्य, कोई नहीं.`.backward()`विधि के बजायः

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

यह एक शैली प्राथमिकता नहीं है. यह एक संकलक प्रतिबंध है. JIT संकलन के लिए शुद्ध कार्यों की आवश्यकता होती है - एक ही इनपुट हमेशा एक ही आउटपुट का उत्पादन करता है, कोई दुष्प्रभाव नहीं. यह प्रतिबंध है कि 100x गति संभव बनाता है.

### jax.numpy: परिचित सतह

JAX त्वरक पर NumPy API को पुनः लागू करता हैः

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

एक ही कार्य नाम, एक ही प्रसारण नियम, एक ही स्लाइसिंग अर्थशास्त्र, लेकिन सरणी जीपीयू / टीपीयू पर रहते हैं, और प्रत्येक ऑपरेशन संकलक द्वारा ट्रैक किया जा सकता है.

एक महत्वपूर्ण अंतरः JAX सरणी अपरिवर्तनीय हैं।`a[0] = 5`. इसके बजाय:`a = a.at[0].set(5)`यह एक सप्ताह के लिए अजीब लगता है, फिर यह क्लिक करता है -- अपरिवर्तनीयता है जो परिवर्तनों को जैसा बनाता है`grad`,`jit`और `vmap`संगत।

### jax.grad: कार्यात्मक ऑटोडिफ़

पायटोरच टेन्सरों के लिए ग्रेडिएंट्स संलग्न करता है (`.grad`) JAX फ़ंक्शन पर ग्रेडिएंट लगाता है।

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`एक फ़ंक्शन लेता है और एक नया फ़ंक्शन देता है जो ग्रेडिएंट की गणना करता है।`.backward()`कोई गणना ग्राफ tensors पर संग्रहीत नहीं है. ग्रेडिएंट सिर्फ एक और समारोह है आप कॉल कर सकते हैं, रचना, या JIT-संकलित.

यह मनमाने ढंग से बना हैः

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

दूसरे व्युत्पन्न, तीसरे व्युत्पन्न, जैकोबियन, हेसियन, सभी रचना द्वारा`grad`. PyTorch भी यह कर सकते हैं (`torch.autograd.functional.hessian`), लेकिन यह पर bolted है. JAX में, यह नींव है.

प्रतिबंध: `grad`केवल शुद्ध कार्यों पर काम करता है. कोई प्रिंट स्टेटमेंट अंदर नहीं (वे ट्रैकिंग के दौरान चल रहे हैं, निष्पादन नहीं). कोई बाहरी राज्य का उत्परिवर्तन नहीं। कोई स्पष्ट कुंजी प्रबंधन के बिना यादृच्छिक संख्या उत्पन्न नहीं।

### jit: XLA पर संकलित करें

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

पहले कॉल पर, JAX फ़ंक्शन को ट्रैक करता है - यह रिकॉर्ड करता है कि कौन से ऑपरेशन होते हैं, उन्हें निष्पादित किए बिना। फिर यह उस ट्रैक को XLA (एक्सेलेरेटेड रैखिक बीजगणित) को सौंपता है, Google के TPU और GPU के लिए संकलक। XLA ऑपरेशन को मिलाता है, रिक्त स्मृति प्रतियों को समाप्त करता है, और अनुकूलित मशीन कोड उत्पन्न करता है।

बाद के कॉल पूरी तरह से पायथन को छोड़ देते हैं। संकलित कोड सी ++ गति से त्वरक पर चलता है।

जब JIT मदद करता हैः
- प्रशिक्षण चरण (एक ही गणना हजारों बार दोहराई जाती है)
- इन्फरेंस (एक ही मॉडल, अलग-अलग इनपुट)
- समान आकार के इनपुट के साथ एक से अधिक बार बुलाया गया कोई भी फ़ंक्शन

जब JIT दर्द होता हैः
- पायथन नियंत्रण प्रवाह के साथ कार्य जो मानों पर निर्भर करता है (`if x > 0`जहां x एक पता लगाया सरणी है)
- एक-शॉट गणना (संकलित ओवरहेड रनटाइम से अधिक है)
- डिबगिंग (ट्रैकिंग वास्तविक निष्पादन को छिपाता है)

नियंत्रण प्रवाह प्रतिबंध वास्तविक है।`jax.lax.cond`प्रतिस्थापन`if/else`. .`jax.lax.scan`प्रतिस्थापन`for`लूप्स. ये वैकल्पिक नहीं हैं -- वे संकलन की कीमत हैं.

### vmap: स्वचालित वेक्टरिज़ेशन

आप एक फ़ंक्शन लिखते हैं जो एक उदाहरण को संसाधित करता हैः

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`उसे बैच को संसाधित करने के लिए उठाता हैः

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`साधन: बैच न करें `params`(साझा),  के अक्ष 0 पर बैच`x`. कोई मैनुअल नहीं .`for`लूप. कोई पुनर्व्यवस्थापन नहीं. कोई बैच आयाम थ्रेडिंग. JAX बैच आयाम का पता लगाता है और पूरे गणना वेक्टरलाइज करता है.

यह संश्लेषण चीनी नहीं है।`vmap`यह एक पायथन लूप की तुलना में 10-100 गुना तेजी से चल रहा है जो विलय वेक्टर कोड उत्पन्न करता है. और यह के साथ बना है `jit`और `grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

उदाहरण प्रति ग्रेडिएंट. एक पंक्ति. यह हैक के बिना PyTorch में लगभग असंभव है.

### pmap: डिवाइसों के बीच डेटा समानांतर

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`सभी उपलब्ध उपकरणों (GPU/TPU) पर फ़ंक्शन को दोहराता है और बैच को विभाजित करता है। फ़ंक्शन के अंदर, `jax.lax.pmean`और `jax.lax.psum`उपकरणों के बीच ग्रेडिएंट समक्रमण।

गूगल Gemini के माध्यम से ट्रेनों हजारों TPU v5e चिप्स का उपयोग कर `pmap`(और इसके उत्तराधिकारी)`shard_map`) प्रोग्रामिंग मॉडलः एकल डिवाइस संस्करण लिखें, `pmap`, किया गया है.

### पाइट्रीजः यूनिवर्सल डेटा स्ट्रक्चर

JAX "पाइट्री" पर काम करता है - सूची, टूपल्स, डिक्ट्स और सरणी के घोंसले संयोजन। आपके मॉडल पैरामीटर एक पिट्री हैंः

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

प्रत्येक जैक्स परिवर्तन--`grad`,`jit`,`vmap`- जानता है कि कैसे pytrees पार करने के लिए.`jax.tree.map(f, tree)`लागू होता है `f`इस तरह ऑप्टिमाइज़र एक ही बार में सभी मापदंडों को अपडेट करते हैंः

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

नहीं`.parameters()`कोई पैरामीटर पंजीकरण नहीं है। पेड़ संरचना मॉडल है।

### कार्यशील बनाम वस्तु-उन्मुख

PyTorch स्टोर वस्तुओं के अंदर की स्थिति बताता हैः

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX स्पष्ट स्थिति के साथ शुद्ध कार्यों का उपयोग करता हैः

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

पैरामेड्स को पास किया जाता है. कुछ भी संग्रहीत नहीं किया जाता है. कुछ भी उत्परिवर्तित नहीं किया जाता है. यह प्रत्येक फ़ंक्शन को परीक्षण योग्य, संगठनीय और संकलित करता है. इसका मतलब यह भी है कि आप स्वयं पैरामेड्स का प्रबंधन करते हैं - या फ्लेक्स या इक्विनॉक्स जैसे पुस्तकालय का उपयोग करते हैं।

### जैक्स पारिस्थितिकी तंत्र

जैक्स आपको आदिमता देता है। पुस्तकालय आपको एर्गोनॉमिक्स देता हैः

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | `nn.Module` with explicit state |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

ऑप्टैक्स मानक अनुकूलक पुस्तकालय है। यह पैरामीटर अपडेट से ग्रेडिएंट परिवर्तन (आदम, एसजीडी, क्लिपिंग) को अलग करता है, जिससे यह रचना करना तुच्छ होता हैः

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### JAX बनाम PyTorch का उपयोग कब करें

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

ईमानदार जवाबः PyTorch का उपयोग करें जब तक कि आपके पास JAX का उपयोग करने का कोई विशिष्ट कारण न हो। ये कारण हैं - TPU पहुंच, प्रति उदाहरण ग्रेडिएंट की आवश्यकता, बड़े पैमाने पर मल्टी-डिवाइस प्रशिक्षण, या Google / डीपमाइंड / एंथ्रोपिक में काम करना।

### JAX में यादृच्छिक संख्याएँ

JAX में वैश्विक यादृच्छिक स्थिति नहीं है। प्रत्येक यादृच्छिक ऑपरेशन के लिए स्पष्ट PRNG कुंजी की आवश्यकता होती हैः

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

यह पहले परेशान करने वाला है, लेकिन यह उपकरणों और संकलनों के बीच पुनरुत्पादनशीलता की गारंटी देता है -- एक गुण जो PyTorch के द्वारा`torch.manual_seed`बहु-जीपीयू सेटिंग्स में गारंटी नहीं दे सकता।

```figure
batchnorm-effect
```

## इसे बनाओ

### चरण 1: सेटअप और डेटा

हम JAX और Optax का उपयोग करके MNIST पर एक 3-परत MLP को प्रशिक्षित करेंगे. 784 इनपुट, 256 और 128 न्यूरॉन्स की दो छिपी हुई परतें, 10 आउटपुट कक्षाएं।

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

### चरण 2: पैरामीटर को आरंभ करें

कोई वर्ग नहीं. बस एक समारोह है कि एक pytree लौटाता हैः

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

एक बीज से विभाजित तीन PRNG कुंजी प्रत्येक वजन एक घोंसले हुए डिक्ट में एक अपरिवर्तनीय सरणी है।

### चरण 3: आगे की यात्रा

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

शुद्ध कार्य, पैराम्स में, पूर्वानुमान बाहर.`self`, कोई संग्रहीत राज्य नहीं। `loss_fn`शून्य से क्रॉस-एंट्रोपी की गणना करता है -- softmax, लॉग, नकारात्मक औसत।

### चरण 4: जेआईटी द्वारा संकलित प्रशिक्षण चरण

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

`jax.value_and_grad`एक पास में हानि मूल्य और ग्रेडिएंट दोनों को लौटाता है।`@jax.jit`डेकोरेटर दोनों कार्यों को XLA में संकलित करता है। पहले कॉल के बाद, प्रत्येक प्रशिक्षण चरण पायथन को छूने के बिना चलता है।

### चरण 5: प्रशिक्षण लूप

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

10 युगों. ~ 97% परीक्षण सटीकता. पहला युग धीमा है (जेआईटी संकलन) । 2-10 युग तेज हैं।

ध्यान दें कि क्या गायब हैः नहीं `.zero_grad()`नहीं , नहीं`.backward()`नहीं , नहीं`.step()`. पूरे अद्यतन एक रचना समारोह कॉल है. ग्रेडिएंट्स की गणना की जाती है, एडम द्वारा परिवर्तित, और मापदंडों पर लागू किया जाता है - सभी अंदर`train_step`. .

## इसका प्रयोग करें

### फ्लेक्सः गूगल मानक

फ्लेक्स सबसे आम JAX तंत्रिका नेटवर्क पुस्तकालय है. यह जोड़ता है`nn.Module`वापस, लेकिन स्पष्ट राज्य प्रबंधन के साथः

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

PyTorch के समान संरचना, लेकिन `params`मॉडल से अलग है। `model.init()`Params बनाता है। `model.apply(params, x)`आगे पास चलाता है. मॉडल वस्तु कोई राज्य नहीं है.

### समोच्च: पायथोनिक विकल्प

इक्विनॉक्स (पैट्रिक किडर द्वारा) मॉडल को पिट्री के रूप में दर्शाता हैः

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

मॉडल स्वयं एक पिट्री है.`.apply()`पैरामीटर सिर्फ मॉडल के पत्ते हैं. यह जैक्स के सोच के करीब है.

### ऑप्टैक्सः कम्पोजेबल ऑप्टिमाइज़र

ऑप्टाक्स अद्यतन से ग्रेडिएंट परिवर्तन को अलग करता हैः

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

ग्रेडिएंट क्लिपिंग, सीखने की दर में वार्मिंग, वजन घटाना - सब परिवर्तनों की एक श्रृंखला के रूप में गठित। प्रत्येक परिवर्तन ग्रेडिएंट देखता है, उन्हें संशोधित करता है, और उन्हें अगले में पारित करता है. कोई मोनोलिथिक अनुकूलक वर्ग नहीं।

## इसे भेजें

**Installation:**

```bash
pip install jax jaxlib optax flax
```

GPU समर्थन के लिएः

```bash
pip install jax[cuda12]
```

TPU (Google Cloud) के लिएः

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Performance gotchas:**

- पहली JIT कॉल धीमी है (संकलित) बेंचमार्किंग से पहले गर्म करें।
- JIT के अंदर JAX सरणी पर पायथन लूप से बचें। उपयोग `jax.lax.scan`या `jax.lax.fori_loop`. .
- `jax.debug.print()`JIT के भीतर काम करता है।`print()`नहीं है।
-  के साथ प्रोफ़ाइल`jax.profiler`XLA संकलन bottlenecks छिपा सकते हैं।
- JAX पूर्वनिर्धारित रूप से जीपीयू मेमोरी का 75% आवंटित करता है। सेट `XLA_PYTHON_CLIENT_PREALLOCATE=false`निष्क्रिय करने के लिए।

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**This lesson produces:**
- `outputs/prompt-jax-optimizer.md`-- सही JAX अनुकूलक विन्यास चुनने के लिए एक संकेत
- `outputs/skill-jax-patterns.md`-- JAX में कार्यात्मक पैटर्न को कवर करने की एक कौशल

## व्यायाम

1. MLP में ड्रॉपआउट जोड़ें. JAX में, ड्रॉपआउट के लिए एक PRNG कुंजी की आवश्यकता होती है - आगे के पास के माध्यम से एक कुंजी को जोड़ें और इसे प्रत्येक ड्रॉपआउट परत के लिए विभाजित करें. परीक्षण सटीकता की तुलना करें और बाहर।

2. उपयोग करें`jax.vmap`प्रत्येक उदाहरण के लिए ग्रेडिएंट मानदंड की गणना करें। किस उदाहरण में सबसे बड़ा ग्रेडिएंट है, और क्यों?

3. मैनुअल फॉरवर्ड फ़ंक्शन को एक सामान्य फ़ंक्शन से बदलें `mlp_forward(params, x)`जो किसी भी स्तर के लिए काम करता है। उपयोग `jax.tree.leaves`गहराई का स्वचालित रूप से निर्धारण करने के लिए।

4. प्रशिक्षण चरण के साथ और बिना बेंचमार्क करें `@jax.jit`आपके हार्डवेयर पर स्पीडअप कितना है? पहले कॉल पर संकलन ओवरहेड कितना है?

5. रचना करके ग्रेडिएंट क्लिपिंग को लागू करें `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. क्लिपिंग के साथ और बिना प्रशिक्षण. प्रभाव देखने के लिए प्रशिक्षण पर ग्रेडिएंट मानदंड का पता लगाएं.

## प्रमुख शर्तें

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

## आगे पढ़ना

- JAX प्रलेखन: https://jax.readthedocs.io/-- आधिकारिक डॉक्स, ग्रेड, जीआईटी, और वीएमएपी पर उत्कृष्ट ट्यूटोरियल के साथ
- "जाक्सः पायथन+नमपी कार्यक्रमों के संगत परिवर्तन" (ब्रैडबरी और अन्य, 2018) -- डिजाइन दर्शन की व्याख्या करने वाला मूल पेपर
- लिनन प्रलेखन: https://flax.readthedocs.io/-- गूगल के तंत्रिका नेटवर्क लाइब्रेरी JAX के लिए
- पैट्रिक किडर, "इक्विनोक्सः जॅक्स में न्यूरल नेटवर्क कॉल करने योग्य पाइट्री और फ़िल्टर किए गए परिवर्तनों के माध्यम से" (2021) -- फ्लेक्स के लिए पायथनिक विकल्प
- डीपमाइंड, "ओप्टैक्सः कम्पोजिबल ग्रेडिएंट ट्रांसफॉर्मेशन और ऑप्टिमाइज़ेशन" -- मानक ऑप्टिमाइज़र लाइब्रेरी
- "आपको जैक्स नहीं पता" (कोलिन राफेल, 2020) - एक व्यावहारिक गाइड जैक्स गॉच और पैटर्न, T5 के एक लेखक से
