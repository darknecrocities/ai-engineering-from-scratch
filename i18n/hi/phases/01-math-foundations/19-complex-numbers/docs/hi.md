# एआई के लिए जटिल संख्याएँ

> -1 की वर्गमूल कल्पना नहीं है, यह घूर्णन, आवृत्तियों और सिग्नल प्रसंस्करण के आधे भाग की कुंजी है।

**Type:** Learn
**Language:**पायथन
**Prerequisites:** Phase 1, Lessons 01-04 (linear algebra, calculus)
**Time:** ~60 minutes

## सीखने के लक्ष्य

- आयताकार और ध्रुवीय दोनों रूपों में जटिल अंकगणित (जोड़ें, गुणा करें, विभाजित करें, जोड़ें) करें
- जटिल एक्सपोनेंशियल और त्रिकोणमिति कार्यों के बीच परिवर्तित करने के लिए एयूलर के सूत्र का उपयोग करें
- एकता की जटिल जड़ों का उपयोग करके डिस्क्रिट फ़ूरियर ट्रांसफॉर्म को लागू करें
- ट्रांसफार्मर में रोपी और सिनोसाइडल पोजिशनिंग कोडिंग के जटिल रोटेशन कैसे अंतर्निहित हैं, इसकी व्याख्या करें

## समस्या

आप फ़ूरियर परिवर्तन पर एक पेपर खोलें और वहाँ है`i`आप ट्रांसफार्मर स्थिति कोडिंग देख सकते हैं और देख सकते हैं`sin`और `cos`आप क्वांटम कंप्यूटिंग के बारे में पढ़ते हैं और जटिल वेक्टर स्पेस में व्यक्त सब कुछ पा सकते हैं।

जटिल संख्याएं अमूर्त लगती हैं। -1 की वर्गमूल पर निर्मित संख्या प्रणाली एक गणितीय चाल की तरह लगती है। लेकिन यह कोई चाल नहीं है। यह घूर्णन और घर्षण की प्राकृतिक भाषा है। जब भी कुछ घूमता है, कंपन करता है या घर्षण करता है, तो जटिल संख्याएं सही उपकरण हैं।

जटिल संख्याओं को समझने के बिना, आप डिस्क्रिट फ़ूरियर ट्रांसफॉर्म को नहीं समझ सकते हैं। आप एफएफटी को नहीं समझ सकते हैं। आप नहीं समझ सकते कि आधुनिक भाषा मॉडल में रोपी (रोटरी पोजीशन एम्बेडिंग) कैसे काम करता है। आप नहीं समझ सकते कि मूल ट्रांसफार्मर पेपर में सिनोसाइडल पोजीशनल एन्कोडिंग्स ने वे आवृत्तियों का उपयोग क्यों किया है।

यह पाठ जटिल अंकगणित को खरोंच से बनाता है, इसे ज्यामिति से जोड़ता है, और आपको दिखाता है कि मशीन लर्निंग में जटिल संख्याएं कहां दिखाई देती हैं।

## अवधारणा

### जटिल संख्या क्या है?

एक जटिल संख्या में दो भाग होते हैंः एक वास्तविक भाग और एक कल्पनात्मक भाग।

```
z = a + bi

where:
  a is the real part
  b is the imaginary part
  i is the imaginary unit, defined by i^2 = -1
```

यह है. आप संख्या रेखा को एक विमान में विस्तारित करते हैं. वास्तविक संख्या एक अक्ष पर बैठती है. कल्पनात्मक संख्या दूसरी पर बैठती है. प्रत्येक जटिल संख्या इस विमान में एक बिंदु है.

### जटिल अंकगणित

**Addition.**वास्तविक भागों को एक साथ जोड़ें, कल्पनात्मक भागों को एक साथ जोड़ें।

```
(a + bi) + (c + di) = (a + c) + (b + d)i

Example: (3 + 2i) + (1 + 4i) = 4 + 6i
```

**Multiplication.**वितरण नियम का उपयोग करें और याद रखें कि i^2 = -1.

```
(a + bi)(c + di) = ac + adi + bci + bdi^2
                 = ac + adi + bci - bd
                 = (ac - bd) + (ad + bc)i

Example: (3 + 2i)(1 + 4i) = 3 + 12i + 2i + 8i^2
                            = 3 + 14i - 8
                            = -5 + 14i
```

**Conjugate.**कल्पनात्मक भाग के संकेत को मोड़ें।

```
conjugate of (a + bi) = a - bi
```

एक जटिल संख्या और उसके संयोग का गुणक हमेशा वास्तविक होता हैः

```
(a + bi)(a - bi) = a^2 + b^2
```

**Division.**संकेतक के संयोग से संख्या और संज्ञाकार को गुणा करें।

```
(a + bi) / (c + di) = (a + bi)(c - di) / (c^2 + d^2)
```

इससे नामकरण से कल्पनात्मक भाग को हटा दिया जाता है, जिससे आपको एक शुद्ध जटिल संख्या मिलती है।

### जटिल विमान

जटिल विमान प्रत्येक जटिल संख्या को 2D बिंदु पर मैप करता है क्षैतिज अक्ष वास्तविक अक्ष है, ऊर्ध्वाधर अक्ष काल्पनिक अक्ष है।

```
z = 3 + 2i  corresponds to the point (3, 2)
z = -1 + 0i corresponds to the point (-1, 0) on the real axis
z = 0 + 4i  corresponds to the point (0, 4) on the imaginary axis
```

एक जटिल संख्या एक ही समय में मूल से एक बिंदु और एक वेक्टर है। यह दोहरी व्याख्या ही जटिल संख्याओं को ज्यामिति के लिए उपयोगी बनाती है।

### ध्रुवीय रूप

विमान में किसी भी बिंदु को मूल से इसकी दूरी और सकारात्मक वास्तविक अक्ष से उसके कोण द्वारा वर्णित किया जा सकता है।

```
z = r * (cos(theta) + i*sin(theta))

where:
  r = |z| = sqrt(a^2 + b^2)     (magnitude, or modulus)
  theta = atan2(b, a)             (phase, or argument)
```

आयताकार रूप (a + bi) जोड़ने के लिए अच्छा है। ध्रुवीय रूप (r, theta) गुणा के लिए अच्छा है।

**Multiplication in polar form.**परिमाणों को गुणा करें, कोण जोड़ें।

```
z1 = r1 * e^(i*theta1)
z2 = r2 * e^(i*theta2)

z1 * z2 = (r1 * r2) * e^(i*(theta1 + theta2))
```

यही कारण है कि जटिल संख्याएं घूर्णन के लिए एकदम सही हैं। परिमाण 1 के साथ जटिल संख्या से गुणा करना शुद्ध घूर्णन है।

### यूलर का सूत्र

जटिल एक्सपोनेंशियल और त्रिकोणमिति के बीच पुलः

```
e^(i*theta) = cos(theta) + i*sin(theta)
```

यह इस पाठ में सबसे महत्वपूर्ण सूत्र है. जब theta = pi:

```
e^(i*pi) = cos(pi) + i*sin(pi) = -1 + 0i = -1

Therefore: e^(i*pi) + 1 = 0
```

पांच मूलभूत स्थिर (ई, आई, पीआई, 1, 0) एक समीकरण में जुड़े हुए हैं।

### एमएल के लिए ईलर का सूत्र क्यों मायने रखता है

एयूलर के सूत्र में कहा गया है कि`e^(i*theta)`थीटा = 0 पर आप (1, 0) पर हैं। थीटा = पई/2, पर आप (0, 1) पर हैं। थीटा = पई पर आप (-1, 0) पर हैं। थीटा = 3*पी/2, पर आप (0, -1) पर हैं। एक पूर्ण घूर्णन थीटा = 2*पी है।

इसका मतलब है कि जटिल एक्सपोनेंटल ARE घूर्णन हैं और घूर्णन संकेत प्रसंस्करण और ML में हर जगह हैं।

### 2D घूर्णन से कनेक्शन

जटिल संख्या (x + yi) को e^(i*theta से गुणा करके मूल के चारों ओर कोण theta द्वारा बिंदु (x, y) को घुमाया जाता है।

```
Rotation via complex multiplication:
  (x + yi) * (cos(theta) + i*sin(theta))
  = (x*cos(theta) - y*sin(theta)) + (x*sin(theta) + y*cos(theta))i

Rotation via matrix multiplication:
  [cos(theta)  -sin(theta)] [x]   [x*cos(theta) - y*sin(theta)]
  [sin(theta)   cos(theta)] [y] = [x*sin(theta) + y*cos(theta)]
```

वे समान परिणाम देते हैं. जटिल गुणा 2D घूर्णन है. घूर्णन मैट्रिक्स सिर्फ मैट्रिक्स संकेतन में लिखा जटिल गुणा है.

```mermaid
graph TD
    subgraph "Complex Multiplication = 2D Rotation"
        A["z = x + yi<br/>Point (x, y)"] -->|"multiply by e^(i*theta)"| B["z' = z * e^(i*theta)<br/>Point rotated by theta"]
    end
    subgraph "Equivalent Matrix Form"
        C["vector [x, y]"] -->|"multiply by rotation matrix"| D["[x cos theta - y sin theta,<br/> x sin theta + y cos theta]"]
    end
    B -.->|"same result"| D
```

### फासर और घूर्णन संकेत

एक जटिल घातीय e^(i*omega*t) एक इकाई वृत्त के चारों ओर कोणीय आवृत्ति ओमेगा पर घूमते बिंदु है। जैसे-जैसे t बढ़ता है, बिंदु वृत्त का पता चलता है।

इस घूर्णन बिंदु का वास्तविक हिस्सा cos(omega*t है। काल्पनिक हिस्सा sin(omega*t है। एक sinusidal संकेत घूर्णन जटिल संख्या की छाया है।

```
e^(i*omega*t) = cos(omega*t) + i*sin(omega*t)

Real part:      cos(omega*t)    -- a cosine wave
Imaginary part: sin(omega*t)    -- a sine wave
```

यह चरण का प्रतिनिधित्व है. एक घुमावदार सिनेस तरंग को ट्रैक करने के बजाय, आप एक सुचारू रूप से घूमते तीर को ट्रैक करते हैं। चरण शिफ्ट कोण ऑफसेट बन जाते हैं। परिमाण परिवर्तन परिमाण परिवर्तन बन जाते हैं। संकेतों का जोड़ वेक्टर योग बन जाता है।

### एकता की जड़ें

इकाई वृत्त पर समान रूप से अंतरित N बिंदुओं की इकाई की N- वीं जड़ें हैंः

```
w_k = e^(2*pi*i*k/N)    for k = 0, 1, 2, ..., N-1
```

N = 4 के लिए जड़ें हैंः 1, i, -1, -i (चार कम्पास बिंदु) ।
N = 8 के लिए, आप चार कम्पास बिंदुओं प्लस चार विघात प्राप्त करते हैं।

एकता की जड़ें डिस्क्रिट फ़ूरियर ट्रांसफॉर्म की नींव हैं। डीएफटी इन एन समान-स्थानित आवृत्तियों पर संकेत को घटकों में विघटित करता है।

### डीएफटी से कनेक्शन

सिग्नल x[0], x[1], ..., x[N-1] का डिस्क्रिट फ़ूरियर ट्रांसफॉर्म हैः

```
X[k] = sum_{n=0}^{N-1} x[n] * e^(-2*pi*i*k*n/N)
```

प्रत्येक एक्स [के] मापता है कि सिग्नल एकता की k-th जड़ के साथ कितना सहसंबंधित है - आवृत्ति k पर एक जटिल सिनोसाइड। डीएफटी एक सिग्नल को N घूर्णन चरणों में तोड़ता है और आपको प्रत्येक के परिमाण और चरण बताता है।

### मैं कल्पना क्यों नहीं कर रहा हूँ

"कल्पनात्मक" शब्द एक ऐतिहासिक दुर्घटना है. डेसकार्ट ने इसका उपयोग अस्वीकार करने के लिए किया। लेकिन मैं नकारात्मक संख्याओं की तुलना में अधिक कल्पनात्मक नहीं हूं जब लोग पहली बार उन्हें अस्वीकार कर रहे थे। नकारात्मक संख्याओं का जवाब है "आप 3 से 5 को क्या घटा सकते हैं? "कल्पनात्मक इकाई का जवाब है "आप -1 प्राप्त करने के लिए क्या वर्ग करते हैं? "

अधिक उपयोगी रूप सेः i एक 90 डिग्री घूर्णन ऑपरेटर है. एक बार i द्वारा एक वास्तविक संख्या को गुणा करें, आप 90 डिग्री को कल्पना की अक्ष पर घूमते हैं. फिर से i द्वारा गुणा करें (i ^ 2) आप एक और 90 डिग्री घूमते हैं - अब आप नकारात्मक वास्तविक दिशा में इंगित कर रहे हैं. यही कारण है कि i ^ 2 = -1. यह रहस्यमय नहीं है। यह दो चौथाई मोड़ से बनाया गया आधा मोड़ है।

यही कारण है कि इंजीनियरिंग में जटिल संख्याएं हर जगह हैं. जो भी घूमती है -- विद्युत चुम्बकीय तरंगें, क्वांटम अवस्थाएँ, संकेत घिसान, स्थिति कोडिंग -- स्वाभाविक रूप से जटिल संख्याओं द्वारा वर्णित होती है।

### जटिल एक्सपोनेंटियल बनाम त्रिकोणमितिक फ़ंक्शन

यूलर के सूत्र से पहले, इंजीनियरों ने संकेतों को ए * कॉस * ओमेगा * टी + फि) - परिमाण ए, आवृत्ति ओमेगा, चरण फि के रूप में लिखा था। यह काम करता है लेकिन अंकगणित को दर्दनाक बनाता है। विभिन्न चरणों वाले दो कोसिन जोड़ने के लिए त्रिकोणमित पहचान की आवश्यकता होती है।

जटिल एक्सपोनेंशियल के साथ, एक ही संकेत A*e^(i*(omega*t + phi)) है। दो संकेत जोड़ना सिर्फ दो जटिल संख्याओं को जोड़ना है। गुणा (मॉड्यूलेशन) केवल परिमाणों को गुणा करना और कोण जोड़ना है। चरण शिफ्ट कोण जोड़ने में बदल जाते हैं। आवृत्ति शिफ्ट फ़ैसर द्वारा गुणा हो जाते हैं।

सिग्नल प्रोसेसिंग का पूरा क्षेत्र जटिल एक्सपोनेंशियल नोटेशन में बदल गया क्योंकि गणित स्वच्छ है। "वास्तविक संकेत" हमेशा जटिल प्रतिनिधित्व का वास्तविक हिस्सा होता है। कल्पनात्मक भाग को लेखांकन के रूप में साथ ले जाया जाता है, जिससे सभी बीजगणित स्वाभाविक रूप से काम करते हैं।

### ट्रांसफार्मर से कनेक्शन

**Sinusoidal positional encodings**(मूल ट्रांसफार्मर पेपर):

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

पाप और कॉस जोड़े विभिन्न आवृत्तियों पर जटिल एक्सपोनेंटियल के वास्तविक और कल्पनात्मक भाग हैं। प्रत्येक आवृत्ति कोडिंग स्थिति के लिए एक अलग "रिज़ॉल्यूशन" प्रदान करती है। कम आवृत्तियां धीरे-धीरे बदलती हैं (गहरे स्थिति) । उच्च आवृत्तियां तेजी से बदलती हैं (अच्छी स्थिति) । साथ में वे प्रत्येक स्थिति को एक अद्वितीय आवृत्ति फिंगरप्रिंट देते हैं।

**RoPE (Rotary Position Embedding)**यह स्पष्ट रूप से जटिल घूर्णन मैट्रिक्स द्वारा क्वेरी और कुंजी वेक्टर को गुणा करता है। दो टोकन के बीच सापेक्ष स्थिति घूर्णन कोण बन जाती है। इन घूर्णन वेक्टरों का उपयोग करके ध्यान की गणना की जाती है, जिससे मॉडल जटिल गुणा के माध्यम से सापेक्ष स्थिति के प्रति संवेदनशील होता है।

| Operation | Algebraic Form | Geometric Meaning |
|-----------|---------------|-------------------|
| Addition | (a+c) + (b+d)i | Vector addition in the plane |
| Multiplication | (ac-bd) + (ad+bc)i | Rotate and scale |
| Conjugate | a - bi | Reflect over real axis |
| Magnitude | sqrt(a^2 + b^2) | Distance from origin |
| Phase | atan2(b, a) | Angle from positive real axis |
| Division | multiply by conjugate | Reverse rotation and rescale |
| Power | r^n * e^(i*n*theta) | Rotate n times, scale by r^n |

```mermaid
graph LR
    subgraph "Unit Circle"
        direction TB
        U1["e^(i*0) = 1"] -.-> U2["e^(i*pi/2) = i"]
        U2 -.-> U3["e^(i*pi) = -1"]
        U3 -.-> U4["e^(i*3pi/2) = -i"]
        U4 -.-> U1
    end
    subgraph "Applications"
        A1["Euler's formula:<br/>e^(i*theta) = cos + i*sin"]
        A2["DFT uses roots of unity:<br/>e^(2*pi*i*k/N)"]
        A3["RoPE uses rotation:<br/>q * e^(i*m*theta)"]
    end
    U1 --> A1
    U1 --> A2
    U1 --> A3
```

```figure
roots-of-unity
```

## इसे बनाओ

### चरण 1: जटिल कक्षा

एक जटिल संख्या वर्ग का निर्माण करें जो आयताकार और ध्रुवीय रूपों के बीच अंकगणित, परिमाण, चरण और रूपांतरण का समर्थन करता है।

```python
import math

class Complex:
    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return Complex(r, i)

    def __truediv__(self, other):
        denom = other.real ** 2 + other.imag ** 2
        r = (self.real * other.real + self.imag * other.imag) / denom
        i = (self.imag * other.real - self.real * other.imag) / denom
        return Complex(r, i)

    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self):
        return math.atan2(self.imag, self.real)

    def conjugate(self):
        return Complex(self.real, -self.imag)
```

### चरण 2: ध्रुवीय रूपांतरण और एयूलर का सूत्र

```python
def to_polar(z):
    return z.magnitude(), z.phase()

def from_polar(r, theta):
    return Complex(r * math.cos(theta), r * math.sin(theta))

def euler(theta):
    return Complex(math.cos(theta), math.sin(theta))
```

सत्यापित करेंः `euler(theta).magnitude()`हमेशा 1.0 होना चाहिए।`euler(0)`देना चाहिए (1, 0). `euler(pi)`(-1, 0) देना चाहिए।

### चरण 3: घूर्णन

कोण से एक बिंदु (x, y) को घूमना एक जटिल गुणा हैः

```python
point = Complex(3, 4)
rotated = point * euler(math.pi / 4)
```

केवल कोण बदलता है।

### चरण 4: जटिल अंकगणित से डीएफटी

```python
def dft(signal):
    N = len(signal)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            total = total + Complex(signal[n], 0) * euler(angle)
        result.append(total)
    return result
```

यह O(N^2) DFT है। प्रत्येक आउटपुट X[k] सिग्नल नमूनों का योग है जो एकता की जड़ों से गुणा होता है।

### चरण 5: उल्टा डीएफटी

उल्टा डीएफटी अपने स्पेक्ट्रम से मूल संकेत को पुनर्निर्माण करता है। आगे के डीएफटी से केवल परिवर्तनः एक्सपोनेंट में संकेत को पलटकर एन से विभाजित करें।

```python
def idft(spectrum):
    N = len(spectrum)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            total = total + spectrum[k] * euler(angle)
        result.append(Complex(total.real / N, total.imag / N))
    return result
```

यह आपको सही पुनर्निर्माण देता है. DFT लागू करें, फिर IDFT, और आप मशीन सटीकता के लिए मूल संकेत वापस मिलता है. कोई जानकारी खो नहीं है.

### चरण 6: एकता की जड़ें

```python
def roots_of_unity(N):
    return [euler(2 * math.pi * k / N) for k in range(N)]
```

दो गुणों की जांच करेंः
- प्रत्येक जड़ का परिमाण 1 होता है।
- सभी N जड़ों का योग शून्य है (वे सममितता द्वारा रद्द होते हैं) ।

ये गुण हैं जो डीएफटी को परिवर्तनीय बनाते हैं। एकता की जड़ें आवृत्ति क्षेत्र के लिए एक ऑर्थोगनल आधार बनाते हैं।

## इसका प्रयोग करें

पायथन में अंतर्निहित जटिल संख्या समर्थन है।`j`कल्पना की इकाई का प्रतिनिधित्व करता है।

```python
z = 3 + 2j
w = 1 + 4j

print(z + w)
print(z * w)
print(abs(z))

import cmath
print(cmath.phase(z))
print(cmath.exp(1j * cmath.pi))
```

सरणी के लिए, नम्पी जटिल संख्याओं को मूल रूप से संभालती हैः

```python
import numpy as np

z = np.array([1+2j, 3+4j, 5+6j])
print(np.abs(z))
print(np.angle(z))
print(np.conj(z))
print(np.real(z))
print(np.imag(z))

signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 128))
spectrum = np.fft.fft(signal)
freqs = np.fft.fftfreq(128, d=1/128)
```

## इसे भेजें

दौड़ें`code/complex_numbers.py`उत्पन्न करने के लिए `outputs/skill-complex-arithmetic.md`. .

## व्यायाम

1. **Complex arithmetic by hand.**गणना (2 + 3i) * (4 - i) और कोड के साथ सत्यापित करें। फिर गणना (5 + 2i) / (1 - 3i) । जटिल विमान पर दोनों परिणाम खींचें और जांचें कि गुणा घूमा और पहले संख्या को स्केल किया।

2. **Rotation sequence.**बिंदु (1, 0) से शुरू करें। e^(i*pi/6) से बारह बार गुणा करें। 12 गुणा के बाद यह सत्यापित करें कि आप (1, 0) पर लौटते हैं। प्रत्येक चरण में निर्देशांक प्रिंट करें और पुष्टि करें कि वे एक नियमित 12-गोन का पता लगाते हैं।

3. **DFT of a known signal.**एक संकेत बनाएं जो 32 बिंदुओं पर नमूने किए गए पाप के योगफल है ((2 * pi * 3 * t) और 0.5 * sin ((2 * pi * 7 * t) । अपने डीएफटी चलाएं। जांचें कि परिमाण स्पेक्ट्रम में आवृत्तियों 3 और 7 पर शिखर है, जिसमें शिखर 7 पर शिखर की ऊंचाई का आधा है।

4. **Roots of unity visualization.**इकाई की आठवीं जड़ की गणना करें. जांचें कि वे शून्य तक योग हैं. जांचें कि किसी भी जड़ को आदिम जड़ e^(2*pi*i/8) से गुणा करने से अगली जड़ मिलती है।

5. **Rotation matrix equivalence.**10 यादृच्छिक कोणों और 10 यादृच्छिक बिंदुओं के लिए, यह सत्यापित करें कि जटिल गुणा 2x2 घूर्णन मैट्रिक्स के साथ मैट्रिक्स-वेक्टर गुणा के समान परिणाम देता है। अधिकतम संख्यात्मक अंतर प्रिंट करें।

## प्रमुख शर्तें

| Term | What it means |
|------|---------------|
| Complex number | A number a + bi where a is the real part, b is the imaginary part, and i^2 = -1 |
| Imaginary unit | The number i, defined by i^2 = -1. Not imaginary in the philosophical sense -- it is a rotation operator |
| Complex plane | The 2D plane where the x-axis is real and the y-axis is imaginary. Also called the Argand plane |
| Magnitude (modulus) | The distance from the origin: sqrt(a^2 + b^2). Written as \|z\| |
| Phase (argument) | The angle from the positive real axis: atan2(b, a). Written as arg(z) |
| Conjugate | The mirror image across the real axis: conjugate of a + bi is a - bi |
| Polar form | Expressing z as r * e^(i*theta) instead of a + bi. Makes multiplication easy |
| Euler's formula | e^(i*theta) = cos(theta) + i*sin(theta). Connects exponentials to trigonometry |
| Phasor | A rotating complex number e^(i*omega*t) representing a sinusoidal signal |
| Roots of unity | The N complex numbers e^(2*pi*i*k/N) for k = 0 to N-1. N equally spaced points on the unit circle |
| DFT | Discrete Fourier Transform. Decomposes a signal into complex sinusoidal components using roots of unity |
| RoPE | Rotary Position Embedding. Uses complex multiplication to encode relative position in transformer attention |

## आगे पढ़ना

- [Visual Introduction to Euler's Formula](https://betterexplained.com/articles/intuitive-understanding-of-eulers-formula/)- भारी नोटेशन के बिना ज्यामितीय अंतर्ज्ञान का निर्माण करता है
- [Su et al.: RoFormer (2021)](https://arxiv.org/abs/2104.09864)- जटिल रोटेशन का उपयोग करके रोटरी पोजीशन एम्बेडिंग की शुरूआत करने वाला पेपर
- [Vaswani et al.: Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)- मूळ ट्रांसफार्मर पेपर विसोनसियल पोजिशनिंग कोडिंग्स के साथ
- [3Blue1Brown: Euler's formula with introductory group theory](https://www.youtube.com/watch?v=mvmuCPvRoWQ)- दृश्य स्पष्टीकरण क्यों e^(i*pi) = -1
- [Needham: Visual Complex Analysis](https://global.oup.com/academic/product/visual-complex-analysis-9780198534464)- जटिल संख्याओं का सर्वोत्तम दृश्य उपचार, ज्यामितीय अंतर्दृष्टि से भरा
- [Strang: Introduction to Linear Algebra, Ch. 10](https://math.mit.edu/~gs/linearalgebra/)- रैखिक बीजगणित और स्वमान के संदर्भ में जटिल संख्याएं
