# सक्रियण कार्य

> गैर-रेखागतता के बिना, आपका 100 परत नेटवर्क एक शानदार मैट्रिक्स गुणा है। सक्रियण वे गेट हैं जो तंत्रिका नेटवर्क को वक्र में सोचने देते हैं।

**Type:** Build
**Languages:** Python
**Prerequisites:** Lesson 03.03 (Backpropagation)
**Time:** ~75 minutes

## सीखने के लक्ष्य

- सिग्मोइड, टैन, रेलू, लीकी रेलू, गेलू, स्विश और सॉफ्टमैक्स को उनके व्युत्पन्नों के साथ खरोंच से लागू करें
- विभिन्न सक्रियणों के साथ 10+ परतों के माध्यम से सक्रियण परिमाणों को मापकर गायब होने वाली ग्रेडिएंट समस्या का निदान करें
- एक ReLU नेटवर्क में मृत न्यूरॉन्स का पता लगाएं और समझाएं कि GELU इस विफलता मोड से क्यों बचता है
- किसी दिए गए वास्तुकला के लिए सही सक्रियण फ़ंक्शन का चयन करें (ट्रांसफार्मर, सीएनएन, आरएनएन, आउटपुट परत)

## समस्या

दो रैखिक परिवर्तनों को ढेर करेंः y = W2(W1x + b1) + b2. इसे विस्तारित करेंः y = W2W1x + W2b1 + b2. यह सिर्फ y = Ax + c है - एक एकल रैखिक परिवर्तन। कोई फर्क नहीं पड़ता कि आप कितने रैखिक परतों को ढेर करते हैं, परिणाम एक मैट्रिक्स गुणा करने में गिर जाता है। आपके 100 परत नेटवर्क में एक एकल परत के समान प्रतिनिधित्व शक्ति है।

यह एक सैद्धांतिक जिज्ञासा नहीं है. इसका मतलब है कि एक गहरी रैखिक नेटवर्क सचमुच XOR नहीं सीख सकता है, एक सर्पिल डेटासेट को वर्गीकृत नहीं कर सकता है, एक चेहरे को पहचान नहीं सकता है। सक्रियण कार्यों के बिना, गहराई एक भ्रम है।

सक्रियण फ़ंक्शंस रैखिकता को तोड़ते हैं। वे एक गैर-रेखीय फ़ंक्शन के माध्यम से प्रत्येक परत के आउटपुट को विकृत करते हैं, जिससे नेटवर्क को निर्णय सीमाओं को मोड़ने, मनमानी कार्यों को करीब लाने और वास्तव में सीखने की क्षमता मिलती है। लेकिन गलत सक्रियण चुनें और आपके ग्रेडिएंट शून्य (गहरे नेटवर्क में सिग्मोइड) तक गायब हो जाते हैं, अनंत तक विस्फोट (गहन आरंभ के बिना असीमित सक्रियण) या आपके न्यूरॉन्स स्थायी रूप से मर जाते हैं (बड़े नकारात्मक पक्षपात के साथ रिलू) । सक्रियण फ़ंक्शन का चयन सीधे यह निर्धारित करता है कि क्या आपका नेटवर्क सीखता है।

## अवधारणा

### क्यों नॉनलाइनरता आवश्यक है

मैट्रिक्स गुणात्मक है. मैट्रिक्स A से एक वेक्टर गुणा करना, फिर मैट्रिक्स B से गुणा करना AB से गुणा करने के समान है. इसका मतलब है कि 10 रैखिक परतों को ढेर करना गणितीय रूप से एक बड़े मैट्रिक्स के साथ एक रैखिक परत के बराबर है. उन सभी मापदंडों, उस सभी गहराई को बर्बाद किया गया है. आपको श्रृंखला को तोड़ने के लिए कुछ चाहिए। यह सक्रियण कार्यों का काम है।

यहाँ सबूत है. एक रैखिक परत गणना करता है f ((x) = Wx + b. स्टैक दोः

```
Layer 1: h = W1 * x + b1
Layer 2: y = W2 * h + b2
```

प्रतिस्थापनः

```
y = W2 * (W1 * x + b1) + b2
y = (W2 * W1) * x + (W2 * b1 + b2)
y = A * x + c
```

एक परत। परतों के बीच एक गैर-रेखीय सक्रियण g (() डालेंः

```
h = g(W1 * x + b1)
y = W2 * h + b2
```

अब प्रतिस्थापन टूट जाता है. W2 * g(W1 * x + b1) + b2 को एक ही रैखिक परिवर्तन में कम नहीं किया जा सकता है। नेटवर्क गैर-रेखीय कार्यों का प्रतिनिधित्व कर सकता है। सक्रियण के साथ प्रत्येक अतिरिक्त परत में प्रतिनिधित्व क्षमता जोड़ती है।

### सिग्मोइड

तंत्रिका नेटवर्क के लिए मूल सक्रियण कार्य।

```
sigmoid(x) = 1 / (1 + e^(-x))
```

आउटपुट रेंजः (0,1) चिकनी, विभेदक, किसी भी वास्तविक संख्या को एक संभावना-जैसे मूल्य पर मैप करता है।

व्युत्पन्नः

```
sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
```

इस व्युत्पन्न का अधिकतम मूल्य 0.25 है, जो x = 0 पर होता है। बैकप्रॉपेगेशन में, ग्रेडिएंट परतों के माध्यम से गुणा होता है। सिग्मोइड की दस परतों का मतलब है कि ग्रेडिएंट अधिकतम 0.25 से दस गुना गुणा होता हैः

```
0.25^10 = 0.000000953674
```

मूल संकेत का एक मिलियनवें से भी कम। यह गायब हो रहा ग्रेडिएंट समस्या है। शुरुआती परतों में ग्रेडिएंट इतने छोटे हो जाते हैं कि वजन शायद ही अपडेट हो जाते हैं। नेटवर्क सीखता है - बाद के परतों में नुकसान कम होता है - लेकिन पहले परतें जमे हुए हैं। गहरे सिग्मोइड नेटवर्क बस प्रशिक्षण नहीं करते हैं।

अतिरिक्त समस्याः सिग्मोइड आउटपुट हमेशा सकारात्मक (0 से 1), जिसका अर्थ है कि वजन पर ग्रेडिएंट हमेशा एक ही संकेत होता है। यह ग्रेडिएंट अवतरण के दौरान ज़िग-जागिंग का कारण बनता है।

### तान

सिग्मोइड का केंद्रित संस्करण।

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

आउटपुट रेंजः (-1, 1) शून्य-केंद्रित, जो ज़िग-जाग समस्या को समाप्त करता है।

व्युत्पन्नः

```
tanh'(x) = 1 - tanh(x)^2
```

अधिकतम व्युत्पन्न 1.0 है x = 0 - सिग्मोइड से चार गुना बेहतर है. लेकिन गायब हो रहा ग्रेडिएंट समस्या अभी भी मौजूद है. बड़े सकारात्मक या नकारात्मक इनपुट के लिए, व्युत्पन्न शून्य के करीब है. दस परतें अभी भी ग्रेडिएंट को कुचलती हैं, केवल कम आक्रामक रूप से।

### रिलूः सफलता

सुधारित रैखिक इकाई. 2010 में नायर और हिंटन द्वारा गहन सीखने के लिए लोकप्रिय (फंक्शन स्वयं फुकुशिमा के 1969 के काम से जुड़ा हुआ है), यह सब कुछ बदल गया।

```
relu(x) = max(0, x)
```

आउटपुट रेंजः [0, अनंत) व्युत्पन्न तुच्छ सरल हैः

```
relu'(x) = 1  if x > 0
            0  if x <= 0
```

सकारात्मक इनपुट के लिए कोई गायब हो रहा ग्रेडिएंट नहीं है. ग्रेडिएंट ठीक से 1 है, सीधे से गुजरता है. यही कारण है कि गहरे नेटवर्क को प्रशिक्षित किया गया है - ReLU परतों पर ग्रेडिएंट परिमाण को संरक्षित करता है।

लेकिन एक विफलता मोड हैः मृत न्यूरॉन समस्या। यदि एक न्यूरॉन का वजन इनपुट हमेशा नकारात्मक होता है (बड़ी नकारात्मक पूर्वाग्रह या दुर्भाग्यपूर्ण वजन आरंभिकरण के कारण), तो इसका आउटपुट हमेशा शून्य होता है, इसका ग्रेडिएंट हमेशा शून्य होता है, और यह कभी अपडेट नहीं होता है। यह स्थायी रूप से मृत होता है। व्यवहार में, एक ReLU नेटवर्क में 10-40% न्यूरॉन प्रशिक्षण के दौरान मर सकते हैं।

### रिलू लीक

मृत न्यूरॉन्स के लिए सबसे सरल उपाय।

```
leaky_relu(x) = x        if x > 0
                alpha * x if x <= 0
```

जहां अल्फा एक छोटा सा स्थिर है, आमतौर पर 0.01। नकारात्मक पक्ष में शून्य के बजाय एक छोटा ढलान है, इसलिए मृत न्यूरॉन्स अभी भी एक ग्रेडिएंट संकेत प्राप्त करते हैं और ठीक हो सकते हैं।

### जीलू: आधुनिक डिफ़ॉल्ट

गैसियन त्रुटि रैखिक इकाई। 2016 में हेंड्रिक्स और जिम्पेल द्वारा पेश किया गया। BERT, GPT और अधिकांश आधुनिक ट्रांसफार्मर में डिफ़ॉल्ट सक्रियण।

```
gelu(x) = x * Phi(x)
```

जहां Phi ((x) मानक सामान्य वितरण का संचयी वितरण कार्य है। व्यवहार में प्रयोग किया जाने वाला अनुमानः

```
gelu(x) ~= 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

GELU हर जगह चिकनी है, छोटे नकारात्मक मानों की अनुमति देता है (रैलू के विपरीत जो शून्य तक हार्ड-क्लिप करता है), और एक संभाव्य व्याख्या हैः यह एक गौशियन वितरण के तहत सकारात्मक होने की संभावना के आधार पर प्रत्येक इनपुट का वजन करता है। यह चिकनी गेटिंग ट्रांसफार्मर वास्तुकला में रैलू से बेहतर प्रदर्शन करती है क्योंकि यह बेहतर ग्रेडिएंट प्रवाह प्रदान करती है और मृत न्यूरॉन समस्या से पूरी तरह से बचती है।

### स्विस / सिलू

रामाचंद्रन और अन्य द्वारा 2017 में स्वचालित खोज के माध्यम से स्वयं-गॉट सक्रियण की खोज की गई।

```
swish(x) = x * sigmoid(x)
```

स्विश औपचारिक रूप से x * sigmoid है। गूगल ने इसे सक्रियण फ़ंक्शन स्पेस पर स्वचालित खोज के माध्यम से खोजा -- एक तंत्रिका नेटवर्क जो तंत्रिका नेटवर्क के कुछ हिस्सों को डिजाइन करता है।

GELU की तरह, यह चिकनी, गैर-एकतरफा है, और छोटे नकारात्मक मानों की अनुमति देता है। अंतर सूक्ष्म हैः स्विश गेटिंग के लिए सिग्मोइड का उपयोग करता है जबकि GELU गॉसियन सीडीएफ का उपयोग करता है। व्यवहार में, प्रदर्शन लगभग समान है। स्विश का उपयोग EfficientNet और कुछ दृष्टि मॉडल में किया जाता है। GELU भाषा मॉडल में हावी है।

### सॉफ्टमैक्सः आउटपुट सक्रियण

छिपे हुए परतों में उपयोग नहीं किया जाता है। सॉफ्टमैक्स कच्चे स्कोर (लॉजिट्स) के वेक्टर को एक संभावना वितरण में परिवर्तित करता है।

```
softmax(x_i) = e^(x_i) / sum(e^(x_j) for all j)
```

प्रत्येक आउटपुट 0 से 1 के बीच होता है। सभी आउटपुटों का योग 1 तक होता है। यह इसे बहु-वर्ग वर्गीकरण के लिए मानक अंतिम सक्रियण बनाता है। सबसे बड़ा लॉजिट उच्चतम संभावना प्राप्त करता है, लेकिन argmax के विपरीत, softmax अंतर योग्य है और सापेक्ष विश्वास के बारे में जानकारी को संरक्षित करता है।

### आकारों की तुलना

```mermaid
graph LR
    subgraph "Activation Functions"
        S["Sigmoid<br/>Range: (0,1)<br/>Saturates both ends"]
        T["Tanh<br/>Range: (-1,1)<br/>Zero-centered"]
        R["ReLU<br/>Range: [0,inf)<br/>Dead neurons"]
        G["GELU<br/>Range: ~(-0.17,inf)<br/>Smooth gating"]
    end
    S -->|"Vanishing gradient"| Problem["Deep networks<br/>don't train"]
    T -->|"Less severe but<br/>still vanishes"| Problem
    R -->|"Gradient = 1<br/>for x > 0"| Solution["Deep networks<br/>train fast"]
    G -->|"Smooth gradient<br/>everywhere"| Solution
```

### क्रमिक प्रवाह तुलना

```mermaid
graph TD
    Input["Input Signal"] --> L1["Layer 1"]
    L1 --> L5["Layer 5"]
    L5 --> L10["Layer 10"]
    L10 --> Output["Output"]

    subgraph "Gradient at Layer 1"
        SigGrad["Sigmoid: ~0.000001"]
        TanhGrad["Tanh: ~0.001"]
        ReluGrad["ReLU: ~1.0"]
        GeluGrad["GELU: ~0.8"]
    end
```

### किस सक्रियण के समय

```mermaid
flowchart TD
    Start["What are you building?"] --> Hidden{"Hidden layers<br/>or output?"}

    Hidden -->|"Hidden layers"| Arch{"Architecture?"}
    Hidden -->|"Output layer"| Task{"Task type?"}

    Arch -->|"Transformer / NLP"| GELU["Use GELU"]
    Arch -->|"CNN / Vision"| ReLU["Use ReLU or Swish"]
    Arch -->|"RNN / LSTM"| Tanh["Use Tanh"]
    Arch -->|"Simple MLP"| ReLU2["Use ReLU"]

    Task -->|"Binary classification"| Sigmoid["Use Sigmoid"]
    Task -->|"Multi-class classification"| Softmax["Use Softmax"]
    Task -->|"Regression"| Linear["Use Linear (no activation)"]
```

```figure
softmax-temperature
```

## इसे बनाओ

### चरण 1: डेरिवेटिव के साथ सभी सक्रियण कार्यों को लागू करें

प्रत्येक फ़ंक्शन एक एकल फ्लोट लेता है और एक फ्लोट लौटाता है। प्रत्येक व्युत्पन्न फ़ंक्शन एक ही इनपुट लेता है और ग्रेडिएंट लौटाता है।

```python
import math

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_act(x):
    return math.tanh(x)

def tanh_derivative(x):
    t = math.tanh(x)
    return 1 - t * t

def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def leaky_relu_derivative(x, alpha=0.01):
    return 1.0 if x > 0 else alpha

def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

def gelu_derivative(x):
    phi = 0.5 * (1 + math.erf(x / math.sqrt(2)))
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return phi + x * pdf

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

def softmax(xs):
    max_x = max(xs)
    exps = [math.exp(x - max_x) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]
```

### चरण 2: कल्पना करें कि ग्रेडिएंट कहां मरते हैं

-5 से 5 तक 100 समान-स्थानित बिंदुओं पर ग्रेडिएंट की गणना करें। प्रत्येक सक्रियण के ग्रेडिएंट को शून्य के निकट दिखाने वाला एक पाठ हिस्टोग्राम प्रिंट करें।

```python
def gradient_scan(name, derivative_fn, start=-5, end=5, n=100):
    step = (end - start) / n
    near_zero = 0
    healthy = 0
    for i in range(n):
        x = start + i * step
        g = derivative_fn(x)
        if abs(g) < 0.01:
            near_zero += 1
        else:
            healthy += 1
    pct_dead = near_zero / n * 100
    print(f"{name:15s}: {healthy:3d} healthy, {near_zero:3d} near-zero ({pct_dead:.0f}% dead zone)")

gradient_scan("Sigmoid", sigmoid_derivative)
gradient_scan("Tanh", tanh_derivative)
gradient_scan("ReLU", relu_derivative)
gradient_scan("Leaky ReLU", leaky_relu_derivative)
gradient_scan("GELU", gelu_derivative)
gradient_scan("Swish", swish_derivative)
```

### चरण 3: ग्रिडिएंट प्रयोग गायब

सिग्मोइड बनाम रिलू का उपयोग करके एन परतों के माध्यम से संकेत को आगे-पास करें। सक्रियण परिमाण कैसे बदलता है, इसका मापन करें।

```python
import random

def vanishing_gradient_experiment(activation_fn, name, n_layers=10, n_inputs=5):
    random.seed(42)
    values = [random.gauss(0, 1) for _ in range(n_inputs)]

    print(f"\n{name} through {n_layers} layers:")
    for layer in range(n_layers):
        weights = [random.gauss(0, 1) for _ in range(n_inputs)]
        z = sum(w * v for w, v in zip(weights, values))
        activated = activation_fn(z)
        magnitude = abs(activated)
        bar = "#" * int(magnitude * 20)
        print(f"  Layer {layer+1:2d}: magnitude = {magnitude:.6f} {bar}")
        values = [activated] * n_inputs

vanishing_gradient_experiment(sigmoid, "Sigmoid")
vanishing_gradient_experiment(relu, "ReLU")
vanishing_gradient_experiment(gelu, "GELU")
```

### चरण 4: मृत न्यूरॉन डिटेक्टर

एक ReLU नेटवर्क बनाएं, उसके माध्यम से यादृच्छिक इनपुट पारित करें, गिनें कि कितने न्यूरॉन्स कभी नहीं चले जाते हैं।

```python
def dead_neuron_detector(n_inputs=5, hidden_size=20, n_samples=1000):
    random.seed(0)
    weights = [[random.gauss(0, 1) for _ in range(n_inputs)] for _ in range(hidden_size)]
    biases = [random.gauss(0, 1) for _ in range(hidden_size)]

    fire_counts = [0] * hidden_size

    for _ in range(n_samples):
        inputs = [random.gauss(0, 1) for _ in range(n_inputs)]
        for neuron_idx in range(hidden_size):
            z = sum(w * x for w, x in zip(weights[neuron_idx], inputs)) + biases[neuron_idx]
            if relu(z) > 0:
                fire_counts[neuron_idx] += 1

    dead = sum(1 for c in fire_counts if c == 0)
    rarely_fire = sum(1 for c in fire_counts if 0 < c < n_samples * 0.05)
    healthy = hidden_size - dead - rarely_fire

    print(f"\nDead Neuron Report ({hidden_size} neurons, {n_samples} samples):")
    print(f"  Dead (never fired):     {dead}")
    print(f"  Barely alive (<5%):     {rarely_fire}")
    print(f"  Healthy:                {healthy}")
    print(f"  Dead neuron rate:       {dead/hidden_size*100:.1f}%")

    for i, c in enumerate(fire_counts):
        status = "DEAD" if c == 0 else "WEAK" if c < n_samples * 0.05 else "OK"
        bar = "#" * (c * 40 // n_samples)
        print(f"  Neuron {i:2d}: {c:4d}/{n_samples} fires [{status:4s}] {bar}")

dead_neuron_detector()
```

### चरण 5: प्रशिक्षण तुलना - सिग्मोइड बनाम रिलू बनाम जेलू

सर्कल डेटासेट पर एक ही दो-परत नेटवर्क को तीन अलग-अलग सक्रियणों के साथ प्रशिक्षित करें (सर्किल के अंदर बिंदु = वर्ग 1, बाहर = वर्ग 0) । अभिसरण गति की तुलना करें।

```python
def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class ActivationNetwork:
    def __init__(self, activation_fn, activation_deriv, hidden_size=8, lr=0.1):
        random.seed(0)
        self.act = activation_fn
        self.act_d = activation_deriv
        self.lr = lr
        self.hidden_size = hidden_size

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(self.act(z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        error = self.out - target
        d_out = error * self.out * (1 - self.out)

        for i in range(self.hidden_size):
            d_h = d_out * self.w2[i] * self.act_d(self.z1[i])
            self.w2[i] -= self.lr * d_out * self.h[i]
            for j in range(2):
                self.w1[i][j] -= self.lr * d_h * self.x[j]
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def train(self, data, epochs=200):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                self.backward(y)
                total_loss += (pred - y) ** 2
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append(avg_loss)
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses


data = make_circle_data()

configs = [
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("ReLU", relu, relu_derivative),
    ("GELU", gelu, gelu_derivative),
]

results = {}
for name, act_fn, act_d_fn in configs:
    print(f"\n=== Training with {name} ===")
    net = ActivationNetwork(act_fn, act_d_fn, hidden_size=8, lr=0.1)
    losses = net.train(data, epochs=200)
    results[name] = losses

print("\n=== Final Loss Comparison ===")
for name, losses in results.items():
    print(f"  {name:10s}: start={losses[0]:.4f} -> end={losses[-1]:.4f} (improvement: {(1 - losses[-1]/losses[0])*100:.1f}%)")
```

## इसका प्रयोग करें

PyTorch इन सभी को कार्यात्मक और मॉड्यूल दोनों रूपों में प्रदान करता हैः

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(4, 10)

relu_out = F.relu(x)
gelu_out = F.gelu(x)
sigmoid_out = torch.sigmoid(x)
swish_out = F.silu(x)

logits = torch.randn(4, 5)
probs = F.softmax(logits, dim=1)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.GELU(),
    nn.Linear(64, 32),
    nn.GELU(),
    nn.Linear(32, 5),
)
```

ट्रांसफार्मर में छिपे हुए परतेंः GELU. सीएनएन में छिपे हुए परतें: ReLU. वर्गीकरण के लिए आउटपुट परतः सॉफ्टमैक्स. रिग्रेशन के लिए आउटपुट परतः कोई (रेखीय) । संभावनाओं के लिए आउटपुट परतः सिग्मोइड. यही है. इन डिफ़ॉल्ट से शुरू करें. जब आपके पास सबूत हों तो उन्हें बदलें।

RNNs और LSTMs छिपे हुए राज्य के लिए tanh और गेट के लिए sigmoid का उपयोग करते हैं, लेकिन अगर आप आज खरोंच से निर्माण कर रहे हैं, तो आप शायद RNNs का उपयोग नहीं कर रहे हैं. यदि आपके ReLU नेटवर्क में न्यूरॉन्स मर रहे हैं, GELU पर स्विच करें. जब तक आपके पास एक विशिष्ट कारण नहीं है, तब तक लीक ReLU तक न पहुंचें - GELU मृत न्यूरॉन्स की समस्या को हल करता है और बेहतर ग्रेडिएंट प्रवाह देता है।

## इसे भेजें

इस पाठ से उत्पन्न होता हैः
- `outputs/prompt-activation-selector.md`-- एक पुनः प्रयोज्य संकेत है कि आप किसी भी वास्तुकला के लिए सही सक्रियण समारोह चुनने में मदद करता है

## व्यायाम

1. पैरामीटरिक रिलू (PReLU) को लागू करें जहां नकारात्मक ढलान अल्फा एक सीखने योग्य पैरामीटर है। इसे सर्कल डेटासेट पर प्रशिक्षित करें और निश्चित लीक रिलू की तुलना करें।

2. 10. सिग्मोइड, टैन, रिलू और जीईएलयू के लिए प्रत्येक परत पर परिमाण का ग्राफ करें। प्रत्येक सक्रियण का संकेत प्रभावी रूप से किस परत पर शून्य तक पहुंचता है?

3. ELU (उत्पादक रैखिक इकाई) को लागू करेंः elu(x) = x यदि x > 0, अल्फा * (e^x - 1) यदि x <= 0. उसी नेटवर्क पर इसकी मृत न्यूरॉन दर की तुलना ReLU से करें।

4. प्रशिक्षण के दौरान चलने वाले "ग्रेडिएंट हेल्थ मॉनिटर" का निर्माण करेंः प्रत्येक युग में, प्रत्येक परत पर औसत ग्रेडिएंट ग्रेडिएंट की गणना करें। किसी भी परत का ग्रेडिएंट 0.001 से नीचे या 100 से अधिक होने पर चेतावनी प्रिंट करें।

5. कक्षा 01 से XOR डेटासेट का उपयोग करने के लिए प्रशिक्षण तुलना को संशोधित करें. किस सक्रियण को XOR पर सबसे तेजी से परिवर्तित किया जाता है? यह सर्कल परिणामों से क्यों भिन्न है?

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Activation function | "The nonlinear part" | A function applied to each neuron's output that breaks linearity, enabling the network to learn nonlinear mappings |
| Vanishing gradient | "Gradients disappear in deep networks" | Gradients shrink exponentially through layers when the activation's derivative is less than 1, making early layers untrainable |
| Exploding gradient | "Gradients blow up" | Gradients grow exponentially through layers when the effective multiplier exceeds 1, causing unstable training |
| Dead neuron | "A neuron that stopped learning" | A ReLU neuron whose input is permanently negative, producing zero output and zero gradient |
| Sigmoid | "Squishes values to 0-1" | The logistic function 1/(1+e^-x), historically important but causes vanishing gradients in deep networks |
| ReLU | "Clips negatives to zero" | max(0, x) -- the activation that made deep learning practical by preserving gradient magnitude |
| GELU | "The transformer activation" | Gaussian Error Linear Unit, a smooth activation that weights inputs by their probability of being positive |
| Swish/SiLU | "Self-gated ReLU" | x * sigmoid(x), discovered through automated search, used in EfficientNet |
| Softmax | "Turns scores into probabilities" | Normalizes a vector of logits into a probability distribution where all values are in (0,1) and sum to 1 |
| Leaky ReLU | "ReLU that doesn't die" | max(alpha*x, x) where alpha is small (0.01), preventing dead neurons by allowing small negative gradients |
| Saturation | "The flat part of sigmoid" | Regions where an activation's derivative approaches zero, blocking gradient flow |
| Logit | "The raw score before softmax" | The unnormalized output of the final layer before applying softmax or sigmoid |

## आगे पढ़ना

- नायर एंड हिंटन, "संशोधित रैखिक इकाइयां सीमित बोल्टज़मैन मशीनों में सुधार करती हैं" (2010) - पेपर जिसने ReLU को पेश किया और गहरे नेटवर्क के प्रशिक्षण को सक्षम किया
- हेंड्रिक्स और गिम्पल, "गॉसियन त्रुटि रैखिक इकाइयां (जीईएलयू) " (2016) -- सक्रियण फ़ंक्शन पेश किया जो ट्रांसफार्मर के लिए डिफ़ॉल्ट बन गया
- रामाचंद्रन और अन्य, "सक्रियता कार्यों की खोज" (2017) - स्विश की खोज के लिए स्वचालित खोज का उपयोग किया, यह दिखाते हुए कि सक्रियण डिजाइन स्वचालित किया जा सकता है
- ग्लोरोट और बेन्गियो, "गहरे फीडफॉरवर्ड तंत्रिका नेटवर्क को प्रशिक्षित करने की कठिनाई को समझना" (2010) - पेपर जो गायब हो रहे / विस्फोटक ग्रेडिएंट का निदान करता है और जेवियर आरंभिकरण का प्रस्ताव करता है
- गुडफ़ेलो, बेन्गियो, Courville, "डीप लर्निंग" अध्याय 6.3 (https://www.deeplearningbook.org/) -- छिपे हुए इकाइयों और सक्रियण कार्यों का कठोर उपचार
