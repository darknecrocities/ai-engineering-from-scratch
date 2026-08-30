# बेयज़ का प्रमेय

> संभावना के बारे में आप क्या उम्मीद है. बेयज़ का प्रमेय के बारे में आप क्या सीखते हैं.

**Type:** Build
**Language:**पायथन
**Prerequisites:** Phase 1, Lesson 06 (Probability Fundamentals)
**Time:** ~75 minutes

## सीखने के लक्ष्य

- पूर्व, संभावनाओं और साक्ष्य से पिछली संभावनाओं की गणना करने के लिए बेयज़ के प्रमेय का उपयोग करें
- लैपलेस चिकनाई और लॉग-स्पेस गणना के साथ खरोंच से एक भोले बेयस पाठ वर्गीकरण बनाएं
- MLE और MAP अनुमान की तुलना करें और समझाएं कि MAP L2 नियमितता से कैसे मेल खाता है
- ए/बी परीक्षण के लिए बीटा-बाइनोमीअल संयुग्मित पूर्ववर्ती का उपयोग करके अनुक्रमिक बेयसियन अद्यतन लागू करें

## समस्या

एक मेडिकल टेस्ट 99% सटीक है. आप सकारात्मक परीक्षण करते हैं. आपके पास वास्तव में बीमारी होने की संभावना क्या है?

अधिकांश लोग कहते हैं कि 99%। वास्तविक उत्तर इस बात पर निर्भर करता है कि बीमारी कितनी दुर्लभ है। यदि हर 10,000 में से 1 व्यक्ति में यह है, तो सकारात्मक परिणाम आपको बीमार होने की संभावना केवल 1% देता है। शेष 99% सकारात्मक परिणाम स्वस्थ लोगों से झूठी अलार्म हैं।

यह एक चालाक सवाल नहीं है. यह बेयज़ का सिद्धांत है. हर स्पैम फ़िल्टर, हर चिकित्सा निदान, हर मशीन लर्निंग मॉडल जो अनिश्चितता को मात्राबद्ध करता है इस सटीक तर्क का उपयोग करता है. आप एक विश्वास से शुरू करते हैं. आप सबूत देखते हैं. आप अद्यतन करते हैं.

यदि आप इसे समझते हुए ML सिस्टम बनाते हैं, तो आप मॉडल आउटपुट को गलत तरीके से समझेंगे, खराब सीमाएं निर्धारित करेंगे, और अत्यधिक आत्मविश्वास वाली भविष्यवाणियां भेजेंगे।

## अवधारणा

### संयुक्त संभावना से बेयज़ तक

आप पहले से ही पाठ 06 से जानते हैं कि सशर्त संभावना हैः

```
P(A|B) = P(A and B) / P(B)
```

और सममित रूप सेः

```
P(B|A) = P(A and B) / P(A)
```

दोनों अभिव्यक्ति एक ही संख्यात्मक साझा करते हैंः P(A और B। उन्हें बराबर सेट करें और पुनर्गठन करेंः

```
P(A and B) = P(A|B) * P(B) = P(B|A) * P(A)

Therefore:

P(A|B) = P(B|A) * P(A) / P(B)
```

यह बेयज़ का प्रमेय है चार मात्राओं, एक समीकरण।

### चार भागों

| Part | Name | What it means |
|------|------|---------------|
| P(A\|B) | Posterior | Your updated belief about A after seeing evidence B |
| P(B\|A) | Likelihood | How probable the evidence B is if A is true |
| P(A) | Prior | Your belief about A before seeing any evidence |
| P(B) | Evidence | Total probability of seeing B under all possibilities |

साक्ष्य शब्द P(B) एक सामान्यीकरण के रूप में कार्य करता है. आप कुल संभावना के कानून का उपयोग करके इसे बढ़ा सकते हैंः

```
P(B) = P(B|A) * P(A) + P(B|not A) * P(not A)
```

### चिकित्सा परीक्षण का उदाहरण

एक बीमारी 10,000 में से 1 व्यक्ति को प्रभावित करती है। परीक्षण 99% सटीक है (बीमार लोगों का 99% पकड़ता है, समय के 1% में झूठे सकारात्मकता देता है) ।

```
P(sick)          = 0.0001     (prior: disease is rare)
P(positive|sick) = 0.99       (likelihood: test catches it)
P(positive|healthy) = 0.01    (false positive rate)

P(positive) = P(positive|sick) * P(sick) + P(positive|healthy) * P(healthy)
            = 0.99 * 0.0001 + 0.01 * 0.9999
            = 0.000099 + 0.009999
            = 0.010098

P(sick|positive) = P(positive|sick) * P(sick) / P(positive)
                 = 0.99 * 0.0001 / 0.010098
                 = 0.0098
                 = 0.98%
```

एक बार जब कोई बीमारी होती है तो सटीक परीक्षण भी अक्सर झूठे सकारात्मक परिणाम देते हैं यही कारण है कि डॉक्टर पुष्टि परीक्षणों का आदेश देते हैं।

### स्पैम फ़िल्टर उदाहरण

आपको एक ईमेल मिलता है जिसमें "लॉटरी" शब्द होता है। क्या यह स्पैम है?

```
P(spam)                = 0.3      (30% of email is spam)
P("lottery"|spam)      = 0.05     (5% of spam emails contain "lottery")
P("lottery"|not spam)  = 0.001    (0.1% of legitimate emails contain "lottery")

P("lottery") = 0.05 * 0.3 + 0.001 * 0.7
             = 0.015 + 0.0007
             = 0.0157

P(spam|"lottery") = 0.05 * 0.3 / 0.0157
                  = 0.955
                  = 95.5%
```

एक शब्द 30% से 95.5% तक संभावना को स्थानांतरित करता है। एक वास्तविक स्पैम फ़िल्टर एक साथ सैकड़ों शब्दों पर बेयज़ लागू करता है।

### साफ़ बेयसः स्वतंत्रता की धारणा

Naive Bayes यह वर्ग को देखते हुए सभी सुविधाओं को सशर्त रूप से स्वतंत्र मानते हुए कई विशेषताओं तक विस्तारित करता हैः

```
P(class | feature_1, feature_2, ..., feature_n)
  = P(class) * P(feature_1|class) * P(feature_2|class) * ... * P(feature_n|class)
    / P(feature_1, feature_2, ..., feature_n)
```

"नाईव" हिस्सा स्वतंत्रता परिकल्पना है। पाठ में, शब्द घटनाएं स्वतंत्र नहीं हैं ("नई" और "यॉर्क" संबद्ध हैं) लेकिन यह परिकल्पना अभ्यास में आश्चर्यजनक रूप से अच्छी तरह से काम करती है क्योंकि वर्गीकरणकर्ता को केवल वर्गों को रैंक करने की आवश्यकता है, न कि कैलिब्रेटेड संभावनाओं का उत्पादन करना है।

चूंकि सभी वर्गों के लिए नामक एक ही है, आप इसे छोड़ सकते हैं और बस संख्याओं की तुलना कर सकते हैंः

```
score(class) = P(class) * product of P(feature_i | class)
```

उच्चतम स्कोर वाला वर्ग चुनें।

### अधिकतम संभावना अनुमान (MLE)

प्रशिक्षण डेटा से आप P  विशेषताएं (Figure) कैसे प्राप्त करते हैं?

```
P("free"|spam) = (number of spam emails containing "free") / (total spam emails)
```

यह MLE है: उन पैरामीटर मानों का चयन करें जो अवलोकन किए गए डेटा को सबसे अधिक संभावना बनाते हैं। आप संभावना फ़ंक्शन को अधिकतम कर रहे हैं, जो डिस्क्रिट गिनती के लिए सापेक्ष आवृत्ति में कम हो जाता है।

समस्याः यदि एक शब्द प्रशिक्षण के दौरान स्पैम में कभी नहीं दिखाई देता है, तो MLE उसे शून्य संभावना देता है। एक अदृश्य शब्द पूरे उत्पाद को मार देता है। इसे लैपलेस चिकनाई के साथ ठीक करेंः

```
P(word|class) = (count(word, class) + 1) / (total_words_in_class + vocabulary_size)
```

प्रत्येक गणना में 1 जोड़ने से कोई संभावना कभी शून्य नहीं होती है।

### अधिकतम एक बाद में (MAP)

MLE पूछता हैः कौन से पैरामीटर P\ डेटा पैरामीटर को अधिकतम करते हैं?

MAP पूछता हैः P  पैरामीटर डेटा में अधिकतम कौन से पैरामीटर हैं?

बेयज़ के प्रमेय द्वाराः

```
P(parameters|data) proportional to P(data|parameters) * P(parameters)
```

MAP स्वयं मापदंडों पर एक पूर्ववर्ती जोड़ता है। यदि आपको लगता है कि मापदंड छोटे होने चाहिए, तो आप इसे एक पूर्ववर्ती के रूप में एन्कोड करते हैं जो बड़े मानों को दंडित करता है। यह एमएल में एल 2 नियमितकरण के समान है। रिग रेग्रिशन में "रिज" दंड सचमुच भार पर गौशियन पूर्ववर्ती है।

| Estimation | Optimizes | ML equivalent |
|------------|-----------|---------------|
| MLE | P(data\|params) | Unregularized training |
| MAP | P(data\|params) * P(params) | L2 / L1 regularization |

### बेयिसियन बनाम फ्रिक्वेन्टिस्टः व्यावहारिक अंतर

अक्सरता के वैज्ञानिक पैरामीटर को अपरिचित के रूप में देखते हैं। वे पूछते हैं, "अगर मैं इस प्रयोग को कई बार दोहराता तो क्या होता?"

बेयिसियन पैरामीटर को वितरण के रूप में देखते हैं। वे पूछते हैं, "मैंने जो देखा है, उसके आधार पर मैं पैरामीटर के बारे में क्या मानता हूं?"

एमएल प्रणालियों के निर्माण के लिए, व्यावहारिक अंतरः

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| Output | Point estimate | Distribution over values |
| Uncertainty | Confidence intervals (about procedure) | Credible intervals (about parameter) |
| Small data | Can overfit | Prior acts as regularization |
| Computation | Usually faster | Often requires sampling (MCMC) |

अधिकांश उत्पादन एमएल आवृत्तिवादी है (एसजीडी, बिंदु अनुमान) बेयिसियन तरीकों चमक जब आप एक माप अनिश्चितता (चिकित्सा निर्णय, सुरक्षा-महत्वपूर्ण प्रणालियों) की जरूरत है या जब डेटा दुर्लभ है (कुछ शॉट सीखने, ठंड शुरू) ।

### एमएल के लिए बेयसियन सोच महत्वपूर्ण क्यों है

यह संबंध समानता से भी गहरा हैः

**Priors are regularization.**भार पर गौसीयन पूर्वक्रम L2 नियमितता है। लैपलेस पूर्वक्रम L1 है। जब भी आप एक नियमितता शब्द जोड़ते हैं, तो आप एक बेयसीयन कथन बना रहे हैं कि आप किन पैरामीटर मानों की उम्मीद करते हैं।

**Posteriors are uncertainty.**एक ही भविष्यवाणी की गई संभावना आपको बताती है कि मॉडल उस अनुमान में कितना आश्वस्त है। बेयिसियन विधियों आपको एक वितरण देती हैः "मुझे लगता है कि P(स्पैम) 0.8 से 0.95 के बीच है।"

**Bayes updates are online learning.**आज का पीछे का हिस्सा कल का पूर्व बन जाता है जब आपका मॉडल नए डेटा देखता है, तो यह स्क्रैच से पुनः प्रशिक्षण के बजाय अपने विश्वासों को क्रमिक रूप से अपडेट करता है।

**Model comparison is Bayesian.**बेयिस सूचना मानदंड (बीआईसी), मार्जिनल प्रॉबलिटी और बेयिस कारक सभी बेयिस तर्क का उपयोग बिना ओवरफिटिंग के मॉडल के बीच चयन करने के लिए करते हैं।

```figure
bayes-update
```

## इसे बनाओ

### चरण 1: बेयज़ प्रमेय फ़ंक्शन

```python
def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence
    return posterior

result = bayes(prior=0.0001, likelihood=0.99, false_positive_rate=0.01)
print(f"P(sick|positive) = {result:.4f}")
```

### चरण 2: साफ़ बेयिस वर्गीकरण

```python
import math
from collections import defaultdict

class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_word_totals = defaultdict(int)
        self.vocab = set()

    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = doc.lower().split()
            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

    def predict(self, document):
        words = document.lower().split()
        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocab)
        best_class = None
        best_score = float("-inf")
        for cls in self.class_counts:
            score = math.log(self.class_counts[cls] / total_docs)
            for word in words:
                count = self.word_counts[cls].get(word, 0)
                total = self.class_word_totals[cls]
                score += math.log((count + self.smoothing) / (total + self.smoothing * vocab_size))
            if score > best_score:
                best_score = score
                best_class = cls
        return best_class
```

लॉग-प्रोबेबिलिटीज अंडरफ्लो को रोकती हैं। कई छोटी संभावनाओं को गुणा करने से फ्लोटिंग प्वाइंट के लिए संख्याएं बहुत छोटी होती हैं। लॉग-प्रोबेबिलिटीज को योग करना संख्यात्मक रूप से स्थिर और गणितीय रूप से समतुल्य है।

### चरण 3: स्पैम डेटा पर प्रशिक्षण

```python
train_docs = [
    "win free money now",
    "free lottery ticket winner",
    "claim your prize today free",
    "urgent offer free cash",
    "congratulations you won free",
    "meeting tomorrow at noon",
    "project update attached",
    "can we schedule a call",
    "quarterly report review",
    "lunch on thursday sounds good",
    "team standup notes attached",
    "please review the pull request",
]

train_labels = [
    "spam", "spam", "spam", "spam", "spam",
    "ham", "ham", "ham", "ham", "ham", "ham", "ham",
]

classifier = NaiveBayes()
classifier.train(train_docs, train_labels)

test_messages = [
    "free money waiting for you",
    "meeting rescheduled to friday",
    "you won a free prize",
    "please review the attached report",
]

for msg in test_messages:
    print(f"  '{msg}' -> {classifier.predict(msg)}")
```

### चरण 4: सीखी गई संभावनाओं की जांच करें

```python
def show_top_words(classifier, cls, n=5):
    vocab_size = len(classifier.vocab)
    total = classifier.class_word_totals[cls]
    probs = {}
    for word in classifier.vocab:
        count = classifier.word_counts[cls].get(word, 0)
        probs[word] = (count + classifier.smoothing) / (total + classifier.smoothing * vocab_size)
    sorted_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for word, prob in sorted_words[:n]:
        print(f"    {word}: {prob:.4f}")

print("\nTop spam words:")
show_top_words(classifier, "spam")
print("\nTop ham words:")
show_top_words(classifier, "ham")
```

## इसका प्रयोग करें

स्किट-लर्न जहाज उत्पादन के लिए तैयार भोले बेयज़ कार्यान्वयनः

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_docs)
clf = MultinomialNB()
clf.fit(X_train, train_labels)

X_test = vectorizer.transform(test_messages)
predictions = clf.predict(X_test)
for msg, pred in zip(test_messages, predictions):
    print(f"  '{msg}' -> {pred}")
```

एक ही एल्गोरिथ्म. CountVectorizer टोकनाइजेशन और शब्दावली निर्माण को संभालता है. MultinomialNB आंतरिक रूप से चिकनाई और लॉग-संभाव्यता को संभालता है. आपके खरोंच संस्करण 40 पंक्तियों में एक ही बात करता है।

## इसे भेजें

यहाँ निर्मित NaiveBayes वर्ग पूरी पाइपलाइन का प्रदर्शन करता हैः टोकनकरण, लैपलेस चिकनाई के साथ संभावना अनुमान, लॉग-स्पेस भविष्यवाणी।`code/bayes.py`पायथन के मानक पुस्तकालय से परे कोई निर्भरता के साथ अंत-से-अंत चलाता है।

### पूर्ववर्ती विवाह

जब पूर्व और पश्च में वितरण के एक ही परिवार से संबंधित हैं, पूर्व को "संयुक्त" कहा जाता है. यह बेयिसियन अद्यतन बीजगणित साफ बनाता है - आप संख्यात्मक एकीकरण के बिना एक बंद-रूप के पश्च प्राप्त करते हैं.

| Likelihood | Conjugate Prior | Posterior | Example |
|-----------|----------------|-----------|---------|
| Bernoulli | Beta(a, b) | Beta(a + successes, b + failures) | Coin flip bias estimation |
| Normal (known variance) | Normal(mu_0, sigma_0) | Normal(weighted mean, smaller variance) | Sensor calibration |
| Poisson | Gamma(a, b) | Gamma(a + sum of counts, b + n) | Modeling arrival rates |
| Multinomial | Dirichlet(alpha) | Dirichlet(alpha + counts) | Topic modeling, language models |

यह महत्वपूर्ण क्यों हैः बिना संयुग्मित पूर्व के, आप मोन्टे कार्लो नमूना या भिन्नता inference के पीछे के करीब करने की जरूरत है. संयुग्मित पूर्व के साथ, आप सिर्फ दो संख्याओं को अद्यतन.

बीटा वितरण व्यवहार में सबसे आम संयुग्मित पूर्व है। बीटा ((a, b) एक संभावना पैरामीटर के बारे में आपके विश्वास को दर्शाता है। औसत a/(a+b है। जितना बड़ा a+b, उतना अधिक केंद्रित (आत्मविश्वासपूर्ण) वितरण।

बीटा पूर्व के विशेष मामलेः
- बीटा ((1, 1) = वर्दी. आप पैरामीटर के बारे में कोई राय नहीं है.
- बीटा ((10, 10) = 0.5 पर पीक। आप दृढ़ता से मानता है कि पैरामीटर 0.5 के करीब है।
- बीटा ((1, 10) = 0 की ओर झुका हुआ है. आप मानते हैं कि पैरामीटर छोटा है.

अद्यतन नियम बहुत सरल हैः

```
Prior:     Beta(a, b)
Data:      s successes, f failures
Posterior: Beta(a + s, b + f)
```

कोई पूर्णांक नहीं, कोई नमूना नहीं, बस जोड़।

### क्रमशः बेयिसियन अद्यतन

बेयसियन निष्कर्ष स्वाभाविक रूप से अनुक्रमिक है। आज का पीछे का कल का पूर्व बन जाता है। यह वास्तविक प्रणालियों को सभी ऐतिहासिक डेटा को पुनः संसाधित किए बिना क्रमिक रूप से सीखता है।

एक ठोस उदाहरण: यह अनुमान लगाना कि क्या एक सिक्का निष्पक्ष है।

**Day 1: No data yet.**
बेटा से शुरू करें 1, 1) -- एक समान पूर्व. आप कोई राय नहीं है.
- पूर्व औसतः 0.5
- पूर्व की चौड़ाई [0, 1]

**Day 2: Observe 7 heads, 3 tails.**
पछाड़ = बीटा(1 + 7, 1 + 3) = बीटा(8, 4)
- बाद का औसतः 8/12 = 0.667
- सबूत बताते हैं कि सिक्का सिर की ओर झुका हुआ है

**Day 3: Observe 5 more heads, 5 more tails.**
कल के पीछे का उपयोग आज के पूर्व के रूप में करें।
पछाड़ = बीटा ((8 + 5, 4 + 5) = बीटा ((13, 9)
- बाद का औसतः 13/22 = 0.591
- संतुलित नए आंकड़ों ने अनुमान को 0.5 की ओर वापस खींच लिया

```mermaid
graph LR
    A["Prior<br/>Beta(1,1)<br/>mean = 0.50"] -->|"7H, 3T"| B["Posterior 1<br/>Beta(8,4)<br/>mean = 0.67"]
    B -->|"becomes prior"| C["Prior 2<br/>Beta(8,4)"]
    C -->|"5H, 5T"| D["Posterior 2<br/>Beta(13,9)<br/>mean = 0.59"]
```

अवलोकनों का क्रम मायने नहीं रखता है। बीटा(1,1) एक ही समय में सभी 12 सिरों और 8 पूंछों के साथ अपडेट किया गया बीटा(13, 9) - एक ही परिणाम देता है। अनुक्रमिक अद्यतन और बैच अद्यतन गणितीय रूप से समकक्ष हैं। लेकिन अनुक्रमिक अद्यतन आपको कच्चे डेटा को संग्रहीत किए बिना प्रत्येक चरण में निर्णय लेने की अनुमति देता है।

यह उत्पादन एमएल सिस्टम में ऑनलाइन सीखने की नींव है। डाकू के लिए थॉम्पसन नमूनाकरण, वृद्धिशील सिफारिश प्रणाली और स्ट्रीमिंग विसंगति डिटेक्टर सभी इस पैटर्न का उपयोग करते हैं।

### ए/बी परीक्षण से संबंध

ए/बी परीक्षण बेयिसियन निष्कर्ष है।

सेटअपः आप दो बटन रंगों का परीक्षण कर रहे हैं। वेरिएंट ए (नीला) और वेरिएंट बी (हरे) । आप जानना चाहते हैं कि इनमें से किस पर अधिक क्लिक होते हैं।

बेयिसियन ए/बी परीक्षणः

1. **Prior.**दोनों संस्करणों के लिए बीटा 1 से शुरू करें। कोई पूर्व प्राथमिकता नहीं।
2. **Data.**वेरिएंट ए: 1000 विचारों में से 50 क्लिक। वेरिएंट बी: 1000 विचारों में से 65 क्लिक।
3. **Posteriors.**
   - एः बीटा ((1 + 50, 1 + 950) = बीटा ((51, 951) । औसत = 0.051
   - बीः बीटा ((1 + 65, 1 + 935) = बीटा ((66, 936). औसत = 0.066
4. **Decision.**गणना P ((B > A) - संभावना है कि B की वास्तविक रूपांतरण दर A की तुलना में अधिक है।

विश्लेषणात्मक रूप से P (B) > A (A) का गणना करना कठिन है, लेकिन मोन्टे कार्लो इसे तुच्छ बनाता हैः

```
1. Draw 100,000 samples from Beta(51, 951)  -> samples_A
2. Draw 100,000 samples from Beta(66, 936)  -> samples_B
3. P(B > A) = fraction of samples where B > A
```

यदि P(B > A) > 0.95, आप संस्करण B भेजते हैं। यदि यह 0.05 से 0.95 के बीच है, तो आप डेटा एकत्र करना जारी रखते हैं। यदि P(B > A) < 0.05, आप संस्करण A भेजते हैं।

आवृत्ति ए/बी परीक्षणों के मुकाबले लाभः
- आप एक प्रत्यक्ष संभावना कथन प्राप्त करते हैंः "एक 97% संभावना है कि बी बेहतर है"
- कोई p-मूल्य भ्रम नहीं, कोई "नल परिकल्पना को अस्वीकार करने में विफलता" हेजिंग नहीं।
- आप किसी भी समय झूठी सकारात्मक दरों को बढ़ाए बिना परिणामों की जांच कर सकते हैं (कोई "खोजने की समस्या" नहीं)
- आप पूर्व ज्ञान को शामिल कर सकते हैं (उदाहरण के लिए, पिछले परीक्षणों से पता चलता है कि रूपांतरण दरें आमतौर पर 3-8%)

| Aspect | Frequentist A/B | Bayesian A/B |
|--------|----------------|--------------|
| Output | p-value | P(B > A) |
| Interpretation | "How surprising is this data if A=B?" | "How likely is B better than A?" |
| Early stopping | Inflates false positives | Safe at any point (given a well-chosen prior and correctly specified model) |
| Prior knowledge | Not used | Encoded as Beta prior |
| Decision rule | p < 0.05 | P(B > A) > threshold |

## व्यायाम

1. **Multiple tests.**एक रोगी स्वतंत्र परीक्षणों पर दो बार सकारात्मक परीक्षण करता है (दोनों 99% सटीक, बीमारी की प्रवृत्ति 1 में 10,000) । दोनों परीक्षणों के बाद पी(मार) क्या है? पहले परीक्षण से पिछली को दूसरे के लिए पूर्व के रूप में उपयोग करें।

2. **Smoothing impact.**स्पाम वर्गीकरणकर्ता को 0.01, 0.1, 1.0 और 10.0 के स्लाइडिंग मानों के साथ चलाएं। शीर्ष शब्द संभावनाएं कैसे बदलती हैं? स्लाइडिंग=0 और एक शब्द के साथ क्या होता है जो केवल शिन में दिखाई देता है?

3. **Add features.**NaiveBayes वर्ग का विस्तार करें ताकि संदेश लंबाई (लघु / लंबा) को शब्द गणना के साथ एक सुविधा के रूप में भी उपयोग किया जा सके। प्रशिक्षण डेटा से P(short mefspam) और P(shortff) का अनुमान लगाएं और इसे भविष्यवाणी स्कोर में फोल्ड करें।

4. **MAP by hand.**अवलोकन किए गए आंकड़ों को देखते हुए (7 सिक्के में 10 सिर) पूर्ववर्ती बीटा ((2,2) का उपयोग करके पूर्वाग्रह के MAP अनुमान की गणना करें। इसे MLE अनुमान (7/10) के साथ तुलना करें।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Prior | "My initial guess" | P(hypothesis) before observing evidence. In ML: the regularization term. |
| Likelihood | "How well the data fits" | P(evidence\|hypothesis). How probable the observed data is under a specific hypothesis. |
| Posterior | "My updated belief" | P(hypothesis\|evidence). The prior multiplied by the likelihood, then normalized. |
| Evidence | "The normalizing constant" | P(data) across all hypotheses. Ensures the posterior sums to 1. |
| Naive Bayes | "That simple text classifier" | A classifier that assumes features are independent given the class. Works well despite the false assumption. |
| Laplace smoothing | "Add-one smoothing" | Adding a small count to every feature to prevent zero probabilities from unseen data. |
| MLE | "Just use the frequencies" | Choose parameters that maximize P(data\|parameters). No prior. Can overfit with small data. |
| MAP | "MLE with a prior" | Choose parameters that maximize P(data\|parameters) * P(parameters). Equivalent to regularized MLE. |
| Log-probability | "Work in log space" | Using log(P) instead of P to avoid floating-point underflow when multiplying many small numbers. |
| False positive | "A wrong alarm" | The test says positive, but the true state is negative. Drives the base rate fallacy. |

## आगे पढ़ना

- [3Blue1Brown: Bayes' theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)- चिकित्सा परीक्षण के उदाहरण के साथ दृश्य स्पष्टीकरण
- [Stanford CS229: Generative Learning Algorithms](https://cs229.stanford.edu/notes2022fall/cs229-notes2.pdf)- बेयज़ की साफ़ता और भेदभावपूर्ण मॉडल से उसका संबंध
- [Think Bayes](https://greenteapress.com/wp/think-bayes/)- मुक्त पुस्तक, पायथन कोड के साथ बेयसियन सांख्यिकी
- [scikit-learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)- उत्पादन के कार्यान्वयन और प्रत्येक संस्करण का उपयोग कब किया जाना चाहिए
