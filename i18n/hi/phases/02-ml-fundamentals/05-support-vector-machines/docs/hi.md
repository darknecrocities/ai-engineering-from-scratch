# समर्थन वेक्टर मशीनें

> दो वर्गों के बीच सबसे चौड़ी सड़क खोजें. यही पूरा विचार है.

**Type:** Build
**Language:**पायथन
**Prerequisites:** Phase 1 (Lessons 08 Optimization, 14 Norms and Distances, 18 Convex Optimization)
**Time:** ~90 minutes

## सीखने के लक्ष्य

- मूल सूत्र पर शर्ट हानि और ग्रेडिएंट गिरावट का उपयोग करके खरोंच से एक रैखिक एसवीएम लागू करें
- अधिकतम मार्जिन सिद्धांत की व्याख्या करें और प्रशिक्षित मॉडल से समर्थन वेक्टरों की पहचान करें
- रैखिक, बहुपद और आरबीएफ कर्नेल की तुलना करें और समझाएं कि कर्नेल ट्रिक स्पष्ट उच्च-आयामी मानचित्रण से कैसे बचता है
- मार्जिन चौड़ाई और वर्गीकरण त्रुटियों के बीच सी पैरामीटर द्वारा नियंत्रित ट्रेडऑफ का मूल्यांकन करें

## समस्या

आपके पास दो वर्ग के डेटा बिंदु हैं और आपको उन्हें अलग करने के लिए एक रेखा (या हाइपरप्लेन) खींचने की आवश्यकता है। अनंत रूप से कई रेखाएं काम कर सकती हैं। आपको कौन सी चुननी चाहिए?

सबसे बड़ा मार्जिन वाला. मार्जिन निर्णय सीमा और प्रत्येक तरफ के निकटतम डेटा बिंदुओं के बीच की दूरी है. एक व्यापक मार्जिन का मतलब है कि वर्गीकरणकर्ता अधिक आत्मविश्वास है और अदृश्य डेटा को बेहतर सामान्य करता है।

यह अंतर्ज्ञान सपोर्ट वेक्टर मशीनों के लिए नेतृत्व करता है, जो एमएल में सबसे गणितीय रूप से सुरुचिपूर्ण एल्गोरिदम में से एक है। एसवीएम गहन सीखने से पहले प्रमुख वर्गीकरण विधि थी और छोटे डेटा सेट, उच्च आयामी डेटा और समस्याओं के लिए सबसे अच्छा विकल्प है जहां आपको सिद्धांतों के साथ एक अच्छी तरह से समझने वाले मॉडल की आवश्यकता होती है। सैद्धांतिक गारंटी।

एसवीएम सीधे चरण 1 से जुड़ते हैंः अनुकूलन संकुचित है (पढ़ना 18) , मार्जिन मानदंडों (पढ़ना 14) के साथ मापा जाता है, और कर्नेल ट्रिक उच्च-आयामी स्थान में कभी भी कंप्यूटिंग के बिना गैर-रैखिक सीमाओं को संभालने के लिए डॉट उत्पादों का उपयोग करता है।

## अवधारणा

### अधिकतम मार्जिन वर्गीकरण

{-1, +1} में y_i लेबल और विशेषता वेक्टर x_i के साथ रैखिक रूप से अलग डेटा को देखते हुए, हम एक हाइपरप्लेन w^T x + b = 0 चाहते हैं जो कक्षाओं को अलग करता है।

हाइपरप्लेन से बिंदु x_i से दूरी हैः

```
distance = |w^T x_i + b| / ||w||
```

सही ढंग से वर्गीकृत बिंदु के लिएः y_i * (w^T x_i + b) > 0. मार्जिन दोनों तरफ हाइपरप्लेन से निकटतम बिंदु तक की दूरी का दोगुना है।

```mermaid
graph LR
    subgraph Margin
        direction TB
        A["w^T x + b = +1"] ~~~ B["w^T x + b = 0"] ~~~ C["w^T x + b = -1"]
    end
    D["+ class points"] --> A
    E["- class points"] --> C
    B --- F["Decision boundary"]
```

अनुकूलन समस्याः

```
maximize    2 / ||w||     (the margin width)
subject to  y_i * (w^T x_i + b) >= 1  for all i
```

इसी तरह (मिमीमाइज़िंग नेशन्स को अनुकूलित करने में आसान है):

```
minimize    (1/2) ||w||^2
subject to  y_i * (w^T x_i + b) >= 1  for all i
```

यह एक घुमावदार चतुर्भुज कार्यक्रम है। इसमें एक अद्वितीय वैश्विक समाधान है। मार्जिन सीमाओं पर ठीक बैठने वाले डेटा बिंदु (जहां y_i * (w^T x_i + b) = 1) समर्थन वेक्टर हैं। वे एकमात्र बिंदु हैं जो निर्णय सीमा निर्धारित करते हैं। किसी भी गैर-समर्थन वेक्टर बिंदु को स्थानांतरित या हटा दें, और सीमा नहीं बदलती है।

### समर्थन वेक्टरः महत्वपूर्ण कुछ

```mermaid
graph TD
    subgraph Classification
        SV1["Support Vector (+ class)<br>y(w'x+b) = 1"] --- DB["Decision Boundary<br>w'x+b = 0"]
        DB --- SV2["Support Vector (- class)<br>y(w'x+b) = 1"]
    end
    O1["Other + points<br>(do not affect boundary)"] -.-> SV1
    O2["Other - points<br>(do not affect boundary)"] -.-> SV2
```

अधिकांश प्रशिक्षण बिंदुओं से कोई संबंध नहीं है। केवल समर्थन वेक्टर महत्वपूर्ण हैं। यही कारण है कि भविष्यवाणी के समय एसवीएम स्मृति-कुशल हैंः आपको केवल समर्थन वेक्टरों को संग्रहीत करने की आवश्यकता है, पूरे प्रशिक्षण सेट को नहीं।

समर्थन वेक्टरों की संख्या सामान्यीकरण त्रुटि पर भी एक सीमा देती है। डेटासेट आकार के सापेक्ष कम समर्थन वेक्टरों का अर्थ बेहतर सामान्यीकरण है।

### नरम मार्जिनः सी पैरामीटर के साथ शोर संभाल

वास्तविक डेटा शायद ही कभी पूरी तरह से अलग किया जा सकता है। कुछ बिंदु सीमा के गलत पक्ष पर या मार्जिन के अंदर हो सकते हैं। नरम मार्जिन सूत्र लचीले चर को पेश करके उल्लंघन की अनुमति देता है।

```
minimize    (1/2) ||w||^2 + C * sum(xi_i)
subject to  y_i * (w^T x_i + b) >= 1 - xi_i
            xi_i >= 0  for all i
```

लैक चर xi_i मापता है कि कितना बिंदु i मार्जिन का उल्लंघन करता है। C व्यापार-बंद को नियंत्रित करता हैः

| C value | Behavior |
|---------|----------|
| Large C | Penalizes violations heavily. Narrow margin, fewer misclassifications. Overfits |
| Small C | Allows more violations. Wide margin, more misclassifications. Underfits |

C नियमितता बल है, उल्टा। बड़ा C = कम नियमितता। छोटा C = अधिक नियमितता।

### झिल्ली हानिः एसवीएम हानि फ़ंक्शन

सॉफ्ट मार्जिन एसवीएम को एक निर्बाध अनुकूलन के रूप में फिर से लिखा जा सकता हैः

```
minimize    (1/2) ||w||^2 + C * sum(max(0, 1 - y_i * (w^T x_i + b)))
```

max(0, 1 - y_i * f(x_i)) शब्द ही झिल्ली हानि है। यह शून्य है जब बिंदु को सही ढंग से वर्गीकृत किया जाता है और मार्जिन से परे होता है। यह रैखिक है जब बिंदु मार्जिन के अंदर होता है या गलत वर्गीकृत होता है।

```
Hinge loss for a single point:

loss
  |
  | \
  |  \
  |   \
  |    \
  |     \_______________
  |
  +-----|-----|-------->  y * f(x)
       0     1

Zero loss when y*f(x) >= 1 (correctly classified, outside margin).
Linear penalty when y*f(x) < 1.
```

लॉजिस्टिक हानि (लॉजिस्टिक प्रतिगमन) की तुलना करेंः

```
Hinge:     max(0, 1 - y*f(x))          Hard cutoff at margin
Logistic:  log(1 + exp(-y*f(x)))        Smooth, never exactly zero
```

हिंज हानि दुर्लभ समाधान पैदा करती है (केवल समर्थन वेक्टरों में गैर-शून्य योगदान होता है) । लॉजिस्टिक हानि सभी डेटा बिंदुओं का उपयोग करती है। यह भविष्यवाणी के समय एसवीएम को अधिक स्मृति-कुशल बनाता है।

### ग्रेडिएंट अवतरण के साथ रैखिक एसवीएम का प्रशिक्षण

आप लिनेयर एसवीएम को शिंज हानि और एल 2 नियमितता पर ग्रेडिएंट गिरावट का उपयोग करके प्रशिक्षित कर सकते हैं, बिना प्रतिबंधित क्यूपी को हल किएः

```
L(w, b) = (lambda/2) * ||w||^2 + (1/n) * sum(max(0, 1 - y_i * (w^T x_i + b)))

Gradient with respect to w:
  If y_i * (w^T x_i + b) >= 1:  dL/dw = lambda * w
  If y_i * (w^T x_i + b) < 1:   dL/dw = lambda * w - y_i * x_i

Gradient with respect to b:
  If y_i * (w^T x_i + b) >= 1:  dL/db = 0
  If y_i * (w^T x_i + b) < 1:   dL/db = -y_i
```

इसे प्राथमिक सूत्र कहा जाता है। यह O ((n * d) प्रति युग में चलता है, जहां n नमूने की संख्या है और d विशेषताओं की संख्या है। बड़े, दुर्लभ, उच्च आयामी डेटा (पाठ वर्गीकरण) के लिए, यह तेजी से है।

### दोहरे सूत्र और कर्नेल ट्रिक

एसवीएम समस्या का लैग्रेंजियन डुअल (चरण 1 पाठ 18 से, केकेटी स्थितियों से) हैः

```
maximize    sum(alpha_i) - (1/2) * sum_ij(alpha_i * alpha_j * y_i * y_j * (x_i . x_j))
subject to  0 <= alpha_i <= C
            sum(alpha_i * y_i) = 0
```

डुअल केवल डेटा बिंदुओं के बीच डॉट उत्पादों x_i . x_j को शामिल करता है। यह मुख्य अंतर्दृष्टि है। प्रत्येक डॉट उत्पाद को एक कर्नेल फ़ंक्शन K(x_i, x_j) के साथ बदलें और SVM स्पष्ट रूप से परिवर्तन की गणना किए बिना गैर-रैखिक सीमाएं सीख सकता है।

```
Linear kernel:      K(x, z) = x . z
Polynomial kernel:  K(x, z) = (x . z + c)^d
RBF (Gaussian):     K(x, z) = exp(-gamma * ||x - z||^2)
```

आरबीएफ कर्नेल डेटा को अनंत आयामी अंतरिक्ष में मानचित्रित करता है। इनपुट अंतरिक्ष में निकट बिंदुओं का कर्नेल मूल्य 1 के करीब होता है। दूर के बिंदुओं का कर्नेल मूल्य 0 के करीब होता है। यह किसी भी चिकनी निर्णय सीमा को सीख सकता है।

```mermaid
graph LR
    subgraph "Input Space (not separable)"
        A["Data points in 2D<br>circular boundary"]
    end
    subgraph "Feature Space (separable)"
        B["Data points in higher dim<br>linear boundary"]
    end
    A -->|"Kernel trick<br>K(x,z) = phi(x).phi(z)"| B
```

कर्नेल ट्रिक उच्च-आयामी अंतरिक्ष में बिंदु उत्पाद की गणना करता है, कभी भी वहां जाने के बिना। D आयामों में डिग्री डी के बहुपद कर्नेल के लिए, स्पष्ट विशेषता अंतरिक्ष में O(D^d) आयाम हैं। लेकिन K(x, z) को O(D) समय में गणना की जाती है।

### प्रतिगमन के लिए एसवीएम (एसवीआर)

समर्थन वेक्टर रिग्रेशन डेटा के चारों ओर चौड़ाई के एक ट्यूब को फिट करता है। ट्यूब के अंदर के बिंदुओं में शून्य हानि होती है। ट्यूब के बाहर के बिंदुओं को रैखिक रूप से दंडित किया जाता है।

```
minimize    (1/2) ||w||^2 + C * sum(xi_i + xi_i*)
subject to  y_i - (w^T x_i + b) <= epsilon + xi_i
            (w^T x_i + b) - y_i <= epsilon + xi_i*
            xi_i, xi_i* >= 0
```

ईप्सिलन पैरामीटर ट्यूब चौड़ाई को नियंत्रित करता है। चौड़ा ट्यूब = कम समर्थन वेक्टर = चिकनी फिट। संकीर्ण ट्यूब = अधिक समर्थन वेक्टर = तंग फिट।

### क्यों एसवीएम गहन सीखने के लिए हार गए (और जब वे अभी भी जीतते हैं)

एसवीएम ने 1990 के दशक के अंत से 2010 के दशक की शुरुआत तक एमएल पर हावी रहा। गहरे सीखने ने उन्हें कई कारणों से पार कर लियाः

| Factor | SVMs | Deep learning |
|--------|------|---------------|
| Feature engineering | Requires it | Learns features |
| Scalability | O(n^2) to O(n^3) for kernel | O(n) per epoch with SGD |
| Image/text/audio | Needs handcrafted features | Learns from raw data |
| Large datasets (>100k) | Slow | Scales well |
| GPU acceleration | Limited benefit | Massive speedup |

एसवीएम अभी भी इन स्थितियों में जीतते हैंः
- छोटे डेटा सेट (सैकड़ों से लेकर हजारों नमूनों तक)
- उच्च आयामी दुर्लभ डेटा (TF-IDF सुविधाओं के साथ पाठ)
- जब आपको गणितीय गारंटी (मार्जिन सीमा) की आवश्यकता होती है
- जब प्रशिक्षण समय न्यूनतम होना चाहिए (रेखीय एसवीएम बहुत तेज है)
- स्पष्ट मार्जिन संरचना के साथ द्विआधारी वर्गीकरण
- विसंगतियों का पता लगाना (एक वर्ग का एसवीएम)

```figure
svm-margin
```

## इसे बनाओ

### चरण 1: झिल्ली हानि और ढलान

आधार, बैच और उसके ग्रेडिएंट के लिए शंकु हानि की गणना करें।

```python
def hinge_loss(X, y, w, b):
    n = len(X)
    total_loss = 0.0
    for i in range(n):
        margin = y[i] * (dot(w, X[i]) + b)
        total_loss += max(0.0, 1.0 - margin)
    return total_loss / n
```

### चरण 2: ग्रेडिएंट अवतरण के माध्यम से रैखिक एसवीएम

नियमित रूप से जंजीर हानि को कम करके प्रशिक्षित करें। कोई QP समाधान की आवश्यकता नहीं है।

```python
class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0

        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                margin = y[i] * (dot(self.w, X[i]) + self.b)
                if margin >= 1:
                    self.w = [wj - self.lr * self.lambda_param * wj
                              for wj in self.w]
                else:
                    self.w = [wj - self.lr * (self.lambda_param * wj - y[i] * X[i][j])
                              for j, wj in enumerate(self.w)]
                    self.b -= self.lr * (-y[i])

    def predict(self, X):
        return [1 if dot(self.w, x) + self.b >= 0 else -1 for x in X]
```

### चरण 3: कर्नेल फ़ंक्शन

रैखिक, बहुपद और आरबीएफ कर्नल को लागू करें।

```python
def linear_kernel(x, z):
    return dot(x, z)

def polynomial_kernel(x, z, degree=3, c=1.0):
    return (dot(x, z) + c) ** degree

def rbf_kernel(x, z, gamma=0.5):
    diff = [xi - zi for xi, zi in zip(x, z)]
    return math.exp(-gamma * dot(diff, diff))
```

### चरण 4: मार्जिन और समर्थन वेक्टर पहचान

प्रशिक्षण के बाद, पहचानें कि कौन से बिंदु समर्थन वेक्टर हैं और मार्जिन चौड़ाई की गणना करें।

```python
def find_support_vectors(X, y, w, b, tol=1e-3):
    support_vectors = []
    for i in range(len(X)):
        margin = y[i] * (dot(w, X[i]) + b)
        if abs(margin - 1.0) < tol:
            support_vectors.append(i)
    return support_vectors
```

देखो`code/svm.py`सभी डेमो के साथ पूर्ण कार्यान्वयन के लिए।

## इसका प्रयोग करें

स्किट-लर्न के साथः

```python
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
])
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
print(f"Support vectors: {clf['svm'].n_support_}")
```

महत्वपूर्णः एसवीएम को प्रशिक्षित करने से पहले हमेशा अपनी विशेषताएं स्केल करें। एसवीएम विशेषता परिमाणों के प्रति संवेदनशील होते हैं क्योंकि मार्जिन अनस्केल किए गए विशेषताओं पर निर्भर करता है, और ज्यामिति को विकृत करता है।

बड़े डेटा सेट के लिए, उपयोग करें `LinearSVC`(प्राथमिक सूत्र, O(n) प्रति युग)`SVC`(दोहरी सूत्र, O(n^2) से O(n^3)):

```python
from sklearn.svm import LinearSVC

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=1.0, max_iter=10000)),
])
```

## व्यायाम

1. एक 2D रैखिक रूप से अलग करने योग्य डेटासेट उत्पन्न करें। अपने रैखिक एसवीएम को प्रशिक्षित करें और समर्थन वेक्टरों की पहचान करें। सत्यापित करें कि समर्थन वेक्टर निर्णय सीमा के निकटतम बिंदु हैं।

2. शोर डेटासेट पर 0.001 से 1000 तक भिन्न C। प्रत्येक C मान के लिए निर्णय सीमा का पता लगाएं। चौड़े मार्जिन (अंडरफिटिंग) से संकीर्ण मार्जिन (ओवरफिटिंग) में संक्रमण का अवलोकन करें।

3. एक डेटासेट बनाएं जहां कक्षा सीमाएं गोल (रैखिक नहीं) हैं। दिखाएं कि एक रैखिक एसवीएम विफल है। आरबीएफ कर्नेल मैट्रिक्स की गणना करें और दिखाएं कि कक्षाएं कर्नेल-प्रेरित सुविधा स्थान में अलग हो जाती हैं।

4. एक ही डेटासेट पर हिंज हानि बनाम लॉजिस्टिक हानि की तुलना करें। एक रैखिक एसवीएम और लॉजिस्टिक प्रतिगमन को प्रशिक्षित करें। गणना करें कि प्रत्येक मॉडल के निर्णय सीमा में कितने प्रशिक्षण बिंदु योगदान करते हैं (सपोर्ट वेक्टर बनाम सभी बिंदु) ।

5. एसवीआर (एप्सिलन-असंवेदनशील हानि) लागू करें। इसे y = sin(x) + शोर पर समायोजित करें। भविष्यवाणियों के आसपास एप्सिलन ट्यूब को रेखांकित करें और समर्थन वेक्टरों (ट्यूब के बाहर बिंदुओं) को उजागर करें।

## प्रमुख शर्तें

| Term | What it actually means |
|------|----------------------|
| Support vectors | The training points closest to the decision boundary. The only points that determine the hyperplane |
| Margin | The distance between the decision boundary and the nearest support vectors. SVMs maximize this |
| Hinge loss | max(0, 1 - y*f(x)). Zero when correctly classified and outside the margin. Linear penalty otherwise |
| C parameter | Trade-off between margin width and classification errors. Large C = narrow margin, small C = wide margin |
| Soft margin | SVM formulation that allows margin violations via slack variables. Handles non-separable data |
| Kernel trick | Computing dot products in a high-dimensional feature space without explicitly mapping to that space |
| Linear kernel | K(x, z) = x . z. Equivalent to standard dot product. For linearly separable data |
| RBF kernel | K(x, z) = exp(-gamma * \|\|x-z\|\|^2). Maps to infinite dimensions. Learns any smooth boundary |
| Polynomial kernel | K(x, z) = (x . z + c)^d. Maps to a feature space of polynomial combinations |
| Dual formulation | Reformulation of the SVM problem that depends only on dot products between data points. Enables kernels |
| SVR | Support Vector Regression. Fits an epsilon-tube around the data. Points inside the tube have zero loss |
| Slack variables | xi_i: measures how much a point violates the margin. Zero for correctly classified points outside margin |
| Maximum margin | The principle of choosing the hyperplane that maximizes the distance to the nearest points of each class |

## आगे पढ़ना

- [Vapnik: The Nature of Statistical Learning Theory (1995)](https://link.springer.com/book/10.1007/978-1-4757-3264-1)- एसवीएम और सांख्यिकीय शिक्षा पर मूल पाठ
- [Cortes & Vapnik: Support-vector networks (1995)](https://link.springer.com/article/10.1007/BF00994018)- मूल एसवीएम पेपर
- [Platt: Sequential Minimal Optimization (1998)](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/)- एसएमओ एल्गोरिथ्म जिसने एसएमओ प्रशिक्षण को व्यावहारिक बनाया
- [scikit-learn SVM documentation](https://scikit-learn.org/stable/modules/svm.html)- कार्यान्वयन के विवरण के साथ व्यावहारिक मार्गदर्शिका
- [LIBSVM: A Library for Support Vector Machines](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)- अधिकांश एसवीएम कार्यान्वयन के पीछे सी ++ पुस्तकालय
