# पूर्वाग्रह-विविधता व्यापार

> प्रत्येक मॉडल त्रुटि तीन स्रोतों में से एक से आती हैः पूर्वाग्रह, भिन्नता, या शोर। आप केवल पहले दो को नियंत्रित कर सकते हैं।

**Type:** Learn
**Language:**पायथन
**Prerequisites:** Phase 2, Lessons 01-09 (ML basics, regression, classification, evaluation)
**Time:** ~75 minutes

## सीखने के लक्ष्य

- अपेक्षित भविष्यवाणी त्रुटि के पूर्वाग्रह-विभेद विघटन का व्युत्पन्न करें और अपरिवर्तनीय शोर की भूमिका की व्याख्या करें
- यह निदान करें कि क्या एक मॉडल को प्रशिक्षण और परीक्षण त्रुटि पैटर्न का उपयोग करके उच्च पूर्वाग्रह या उच्च भिन्नता से पीड़ित है
- बताएं कि कैसे नियमितकरण तकनीक (L1, L2, ड्रॉपआउट, आरंभिक रोक) भिन्नता के लिए व्यापार पूर्वाग्रह
- जटिलता बढ़ते मॉडल के बीच पूर्वाग्रह-भिन्नता व्यापार को दर्शाने वाले प्रयोगों को लागू करें

## समस्या

आपने एक मॉडल को प्रशिक्षित किया है. इसमें परीक्षण डेटा में कुछ त्रुटि है. यह त्रुटि कहां से आई है?

यदि आपका मॉडल बहुत सरल है (एक घुमावदार डेटासेट पर रैखिक प्रतिगमन), तो यह लगातार सही पैटर्न को याद करेगा। यह पूर्वाग्रह है। यदि आपका मॉडल बहुत जटिल है (15 डेटा बिंदुओं पर डिग्री-20 बहुपद), तो यह प्रशिक्षण डेटा को पूरी तरह से फिट करेगा लेकिन नए डेटा पर बहुत अलग भविष्यवाणियां देगा। यह भिन्नता है।

आप एक निश्चित मॉडल क्षमता के लिए दोनों को एक ही समय में कम नहीं कर सकते। पूर्वाग्रह को नीचे धकेलें और भिन्नता बढ़ जाती है। पूर्वाग्रह को नीचे धकेलें और पूर्वाग्रह बढ़ जाता है। इस व्यापार को समझना मशीन लर्निंग में सबसे उपयोगी नैदानिक कौशल है। यह आपको बताता है कि क्या आप अपना मॉडल अधिक जटिल या कम जटिल बनाना चाहते हैं, क्या आप अधिक डेटा प्राप्त करना चाहते हैं या बेहतर सुविधाओं का निर्माण करना चाहते हैं, क्या आप अधिक या कम नियमित करना चाहते हैं।

## अवधारणा

### पूर्वाग्रहः व्यवस्थित त्रुटि

पूर्वाग्रह मापता है कि आपके मॉडल की औसत भविष्यवाणी वास्तविक मूल्य से कितनी दूर है। यदि आपने एक ही मॉडल को एक ही वितरण से प्राप्त कई अलग-अलग प्रशिक्षण सेटों पर प्रशिक्षित किया और भविष्यवाणियों का औसत बनाया, तो पूर्वाग्रह उस औसत और सच्चाई के बीच अंतर है।

उच्च पूर्वाग्रह का मतलब है कि मॉडल वास्तविक पैटर्न को कैप्चर करने के लिए बहुत कठोर है। एक पैराबोला के अनुरूप एक सीधी रेखा हमेशा वक्र को याद करेगी, चाहे आप इसे कितना डेटा दें। यह अनुचित है।

```
High bias (underfitting):
  Model always predicts roughly the same wrong thing.
  Training error: HIGH
  Test error: HIGH
  Gap between them: SMALL
```

### विविधताः प्रशिक्षण डेटा की संवेदनशीलता

वेरिएंट यह मापता है कि जब आप विभिन्न डेटा उपसमूहों पर प्रशिक्षण देते हैं तो आपकी भविष्यवाणियां कितनी बदलती हैं। यदि प्रशिक्षण सेट में छोटे बदलाव मॉडल में बड़े बदलाव का कारण बनते हैं, तो वेरिएंट अधिक है।

उच्च भिन्नता का अर्थ है कि मॉडल प्रशिक्षण डेटा में शोर फिट करता है, अंतर्निहित संकेत नहीं। एक डिग्री-20 बहुपद प्रत्येक प्रशिक्षण बिंदु के माध्यम से थ्रेड करेगा लेकिन उनके बीच जंगली रूप से झुकता है। यह ओवरफिटिंग है।

```
High variance (overfitting):
  Model fits training data perfectly but fails on new data.
  Training error: LOW
  Test error: HIGH
  Gap between them: LARGE
```

### विघटन

किसी भी बिंदु x के लिए, वर्ग हानि के तहत अपेक्षित भविष्यवाणी त्रुटि ठीक से विघटित होती हैः

```
Expected Error = Bias^2 + Variance + Irreducible Noise

where:
  Bias^2   = (E[f_hat(x)] - f(x))^2
  Variance = E[(f_hat(x) - E[f_hat(x)])^2]
  Noise    = E[(y - f(x))^2]             (sigma^2)
```

- `f(x)`है वास्तविक कार्य
- `f_hat(x)`यह आपके मॉडल की भविष्यवाणी है
- `E[...]`विभिन्न प्रशिक्षण सेटों पर अपेक्षा है
- `y`यह अवलोकन किया गया लेबल (सच्चा फ़ंक्शन प्लस शोर) है

शोर शब्द अपरिहार्य है. कोई मॉडल शोर डेटा पर सिग्मा^2 से बेहतर काम नहीं कर सकता है. आपका काम पूर्वाग्रह^2 और भिन्नता के बीच सही संतुलन ढूंढना है।

### मॉडल जटिलता बनाम त्रुटि

```mermaid
graph LR
    A[Simple Model] -->|increase complexity| B[Sweet Spot]
    B -->|increase complexity| C[Complex Model]

    style A fill:#f9f,stroke:#333
    style B fill:#9f9,stroke:#333
    style C fill:#f99,stroke:#333
```

क्लासिक यू-आकार के वक्रः

| Complexity | Bias | Variance | Total Error |
|-----------|------|----------|-------------|
| Too low | HIGH | LOW | HIGH (underfitting) |
| Just right | MODERATE | MODERATE | LOWEST |
| Too high | LOW | HIGH | HIGH (overfitting) |

### पूर्वाग्रह नियंत्रण के रूप में नियमन

नियमन विसंगति को कम करने के लिए पूर्वग्रह को जानबूझकर बढ़ाता है। यह मॉडल को प्रतिबंधित करता है ताकि यह शोर का पीछा नहीं कर सके।

- **L2 (Ridge):**सभी भारों को शून्य की ओर कम करता है, सभी विशेषताओं को बनाए रखता है, लेकिन उनके प्रभाव को कम करता है।
- **L1 (Lasso):**कुछ वजन को शून्य तक धक्का देता है। सुविधा चयन करता है।
- **Dropout:**प्रशिक्षण के दौरान यादृच्छिक रूप से न्यूरॉन्स को निष्क्रिय करता है।
- **Early stopping:**प्रशिक्षण डेटा के मॉडल को पूरी तरह से फिट करने से पहले प्रशिक्षण बंद कर देता है।

नियमन शक्ति (lambda, ड्रॉपअप दर, युगों की संख्या) सीधे नियंत्रित करती है कि आप पूर्वाग्रह-भिन्नता वक्र पर कहां बैठे हैं। अधिक नियमन का मतलब है अधिक पूर्वाग्रह, कम भिन्नता।

### दोहरी वंशावलीः आधुनिक दृष्टिकोण

क्लासिकल थ्योरी कहती हैः मीठे बिंदु के बाद, अधिक जटिलता हमेशा दर्द करती है। लेकिन 2019 के बाद से अनुसंधान ने कुछ अप्रत्याशित दिखाया है। यदि आप अंतराल सीमा से परे मॉडल क्षमता को बढ़ाते रहते हैं (जहां मॉडल में प्रशिक्षण डेटा को पूरी तरह से फिट करने के लिए पर्याप्त मापदंड होते हैं), तो परीक्षण त्रुटि फिर से कम हो सकती है।

```mermaid
graph LR
    A[Underfit Zone] --> B[Classical Sweet Spot]
    B --> C[Interpolation Threshold]
    C --> D[Double Descent - Error Drops Again]

    style A fill:#fdd,stroke:#333
    style B fill:#dfd,stroke:#333
    style C fill:#fdd,stroke:#333
    style D fill:#dfd,stroke:#333
```

इस "डबल वंश" घटना से पता चलता है कि बड़े पैमाने पर अतिपरिमेय तंत्रिका नेटवर्क (प्रशिक्षण उदाहरणों की तुलना में बहुत अधिक मापदंडों के साथ) अभी भी अच्छी तरह से सामान्य हो जाते हैं। शास्त्रीय पूर्वाग्रह-भिन्नता व्यापार गलत नहीं है, लेकिन यह आधुनिक शासन के लिए अपूर्ण है।

दोहरी गिरावट के बारे में प्रमुख टिप्पणियाँः
- यह रैखिक मॉडल, निर्णय वृक्ष और तंत्रिका नेटवर्क में होता है
- अधिक डेटा वास्तव में अंतराल क्षेत्र में चोट पहुंचा सकता है (सैंपल-wise डबल उतरने)
- अधिक प्रशिक्षण युग भी इसका कारण बन सकते हैं (युग-दृष्टि से दोहरी गिरावट)
- नियमन शिखर को समतल करता है लेकिन उसे समाप्त नहीं करता है

ऐसा क्यों होता है? अंतराल सीमा पर, मॉडल में सभी प्रशिक्षण बिंदुओं के लिए बस पर्याप्त क्षमता है। यह एक बहुत ही विशिष्ट समाधान में मजबूर किया जाता है जो हर बिंदु के माध्यम से धागे है, और डेटा में छोटे व्यवधान फिट में बड़े बदलाव का कारण बनते हैं। यह है जहां भिन्नता चरम पर है। सीमा से परे, मॉडल में कई संभावित समाधान हैं जो डेटा के लिए सही ढंग से फिट होते हैं। सीखने का एल्गोरिथ्म (जैसे, अप्रत्यक्ष नियमितता के साथ ग्रेडिएंट गिरावट) उनमें से सबसे सरल को चुनने का रुझान रखता है। सरल समाधानों की ओर यह अप्रत्यक्ष पूर्वाग्रह है कि अधिक पैरामीटर वाले मॉडल सामान्य क्यों होते हैं।

| Regime | Parameters vs Samples | Behavior |
|--------|----------------------|----------|
| Underparameterized | p << n | Classical tradeoff applies |
| Interpolation threshold | p ~ n | Variance peaks, test error spikes |
| Overparameterized | p >> n | Implicit regularization kicks in, test error drops |

व्यावहारिक उद्देश्यों के लिएः यदि आप तंत्रिका नेटवर्क या बड़े पेड़ समूहों का उपयोग कर रहे हैं, तो इंटरपोलेशन सीमा पर न रुकें। या तो इसके नीचे रहें (स्पष्ट नियमितता के साथ) या उससे बहुत आगे जाएं। सबसे खराब जगह है सीमा पर।

### अपना मॉडल पहचानना

```mermaid
flowchart TD
    A[Compare train error vs test error] --> B{Large gap?}
    B -->|Yes| C[High variance - overfitting]
    B -->|No| D{Both errors high?}
    D -->|Yes| E[High bias - underfitting]
    D -->|No| F[Good fit]

    C --> G[More data / Regularize / Simpler model]
    E --> H[More features / Complex model / Less regularization]
    F --> I[Deploy]
```

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| High train error, high test error | Bias | More features, complex model, less regularization |
| Low train error, high test error | Variance | More data, regularization, simpler model, dropout |
| Low train error, low test error | Good fit | Ship it |
| Train error decreasing, test error increasing | Overfitting in progress | Early stopping |

### व्यावहारिक रणनीति

**When bias is the problem:**
- बहुपद या बातचीत सुविधाएँ जोड़ें
- अधिक लचीला मॉडल का उपयोग करें (रेखीय के बजाय पेड़ समूह)
- नियमितता की ताकत को कम करना
- ट्रेन अधिक समय तक (यदि अभी तक अभिसरण नहीं हुई है)

**When variance is the problem:**
- अधिक प्रशिक्षण डेटा प्राप्त करें
- बैगिंग का प्रयोग करें (अनियमित वन)
- नियमितता में वृद्धि (उच्च लैम्ब्डा, अधिक ड्रॉपआउट)
- फ़ीचर चयन (गर्मीली फ़ीचर हटाएँ)
- इसे जल्दी पहचानने के लिए क्रॉस-वैलिडेशन का उपयोग करें

### विधिओं को एक साथ रखना और भिन्नता को कम करना

असेंबली विधियां भिन्नता से लड़ने का सबसे व्यावहारिक साधन हैं।

**Bagging (Bootstrap Aggregating)**प्रशिक्षण डेटा के विभिन्न बूटस्ट्रैप नमूनों पर कई मॉडल को प्रशिक्षित करता है, फिर उनके अनुमानों का औसत बनाता है। प्रत्येक व्यक्तिगत मॉडल में उच्च भिन्नता होती है, लेकिन औसत में बहुत कम भिन्नता होती है। यादृच्छिक जंगलों में निर्णय के पेड़ों पर लागू बैगिंग होती है।

यह गणितीय रूप से काम क्यों करता हैः यदि आप N स्वतंत्र भविष्यवाणियों का औसत देते हैं, जिनमें से प्रत्येक भिन्नता सिग्मा^2 के साथ, औसत का भिन्नता सिग्मा^2 / N है। मॉडल वास्तव में स्वतंत्र नहीं हैं (वे सभी समान डेटा देखते हैं), इसलिए कमी 1/N से कम है, लेकिन यह अभी भी पर्याप्त है।

**Boosting**मॉडलों को क्रमशः बनाने से पूर्वाग्रह को कम करता है, जहां प्रत्येक नया मॉडल अभी तक समूह की त्रुटियों पर ध्यान केंद्रित करता है। ग्रेडिएंट बूस्टिंग और एडाबॉस्ट मुख्य उदाहरण हैं। यदि आप बहुत सारे मॉडल जोड़ते हैं तो बूस्टिंग ओवरफिट हो सकती है, इसलिए आपको जल्दी रुकने या नियमित करने की आवश्यकता होती है।

| Method | Primary Effect | Bias Change | Variance Change |
|--------|---------------|-------------|-----------------|
| Bagging | Reduces variance | No change | Decreases |
| Boosting | Reduces bias | Decreases | Can increase |
| Stacking | Reduces both | Depends on meta-learner | Depends on base models |
| Dropout | Implicit bagging | Slight increase | Decreases |

**Practical rule:**यदि आपके आधार मॉडल में उच्च भिन्नता है (गहरे पेड़, उच्च डिग्री के बहुपद), बैगिंग का उपयोग करें। यदि आपके आधार मॉडल में उच्च पूर्वाग्रह है (छोटे तने, सरल रैखिक मॉडल), बूस्टिंग का उपयोग करें।

### सीखने की वक्रता

प्रशिक्षण सेट के आकार के आधार पर सीखने के वक्र प्रशिक्षण और सत्यापन त्रुटि का पता लगाते हैं। ये आपके पास सबसे व्यावहारिक नैदानिक उपकरण हैं। एक एकल ट्रेन / परीक्षण तुलना के विपरीत, सीखने के वक्र आपको आपके मॉडल की पटरियों को दिखाते हैं और आपको बताते हैं कि क्या अधिक डेटा मदद करेगा।

```mermaid
flowchart TD
    subgraph HB["High Bias Learning Curve"]
        direction LR
        HB1["Small N: both errors high"]
        HB2["Large N: both errors converge to HIGH error"]
        HB1 --> HB2
    end

    subgraph HV["High Variance Learning Curve"]
        direction LR
        HV1["Small N: train low, test high (big gap)"]
        HV2["Large N: gap shrinks but slowly"]
        HV1 --> HV2
    end

    subgraph GF["Good Fit Learning Curve"]
        direction LR
        GF1["Small N: some gap"]
        GF2["Large N: both converge to LOW error"]
        GF1 --> GF2
    end
```

उन्हें कैसे पढ़ेंः

| Scenario | Training Error | Validation Error | Gap | What It Means | What to Do |
|----------|---------------|-----------------|-----|---------------|------------|
| High bias | High | High | Small | Model cannot capture the pattern | More features, complex model, less regularization |
| High variance | Low | High | Large | Model memorizes training data | More data, regularization, simpler model |
| Good fit | Moderate | Moderate | Small | Model generalizes well | Ship it |
| High variance, improving | Low | Decreasing with more data | Shrinking | Variance problem that data can fix | Collect more data |
| High bias, flat | High | High and flat | Small and flat | More data will NOT help | Change model architecture |

महत्वपूर्ण अंतर्दृष्टिः यदि दोनों वक्रों में समतलता है और अंतर छोटा है लेकिन दोनों त्रुटियां बड़ी हैं, तो अधिक डेटा बेकार है। आपको एक बेहतर मॉडल की आवश्यकता है। यदि अंतर बड़ा है और फिर भी छोटा हो रहा है, तो अधिक डेटा मदद करेगा।

### सीखने की वक्रता कैसे उत्पन्न करें

दो दृष्टिकोण हैंः

**Approach 1: Vary training set size, fixed model.**मॉडल और हाइपरपरपैरामीटर को स्थिर रखें। प्रशिक्षण डेटा के तेजी से बड़े उप-समुच्चों पर अभ्यास करें। प्रत्येक आकार पर प्रशिक्षण त्रुटि और सत्यापन त्रुटि को मापें। यह मानक सीखने वक्र है।

**Approach 2: Vary model complexity, fixed data.**डेटा को स्थिर रखें। एक जटिलता पैरामीटर (बहुपद डिग्री, पेड़ की गहराई, परतों की संख्या) को फ़्रेश करें। प्रत्येक जटिलता पर प्रशिक्षण त्रुटि और सत्यापन त्रुटि को मापें। यह एक सत्यापन वक्र है और सीधे पूर्वाग्रह-भिन्नता व्यापार दिखाता है।

दोनों दृष्टिकोण एक दूसरे के पूरक हैं. पहला आपको बताता है कि क्या अधिक डेटा मदद करेगा। दूसरा आपको बताता है कि क्या एक अलग मॉडल मदद करेगा। अपने अगले कदम के बारे में निर्णय लेने से पहले दोनों को चलाएं।

```mermaid
flowchart TD
    A[Model underperforming] --> B[Generate learning curve]
    B --> C{Gap between train and val?}
    C -->|Large gap, val still decreasing| D[More data will help]
    C -->|Small gap, both high| E[More data will NOT help]
    C -->|Large gap, val flat| F[Regularize or simplify]
    E --> G[Generate validation curve]
    G --> H[Try more complex model]
```

```figure
bias-variance
```

## इसे बनाओ

कोड में `code/bias_variance.py`यहाँ है दृष्टिकोण, कदम से कदम.

### चरण 1: ज्ञात फ़ंक्शन से सिंथेटिक डेटा उत्पन्न करें

हम उपयोग करते हैं`f(x) = sin(1.5x) + 0.5x`सही फ़ंक्शन को जानने से हमें सटीक पूर्वाग्रह और भिन्नता की गणना करने में मदद मिलती है।

```python
def true_function(x):
    return np.sin(1.5 * x) + 0.5 * x

def generate_data(n_samples=30, noise_std=0.5, x_range=(-3, 3), seed=None):
    rng = np.random.RandomState(seed)
    x = rng.uniform(x_range[0], x_range[1], n_samples)
    y = true_function(x) + rng.normal(0, noise_std, n_samples)
    return x, y
```

### चरण 2: बूटस्ट्रैप नमूनाकरण और बहुपद फिटिंग

प्रत्येक बहुपद डिग्री के लिए, हम कई बूटस्ट्रैप प्रशिक्षण सेट खींचते हैं, बहुपद के अनुरूप होते हैं, और एक निश्चित परीक्षण ग्रिड पर भविष्यवाणियों को रिकॉर्ड करते हैं। यह हमें प्रत्येक परीक्षण बिंदु पर भविष्यवाणियों का वितरण देता है।

```python
def fit_polynomial(x_train, y_train, degree, lam=0.0):
    X = np.column_stack([x_train ** d for d in range(degree + 1)])
    if lam > 0:
        penalty = lam * np.eye(X.shape[1])
        penalty[0, 0] = 0
        w = np.linalg.solve(X.T @ X + penalty, X.T @ y_train)
    else:
        w = np.linalg.lstsq(X, y_train, rcond=None)[0]
    return w
```

हम 200 अलग-अलग बूटस्ट्रैप नमूने पर फिट होते हैं। प्रत्येक बूटस्ट्रैप नमूना एक ही अंतर्निहित वितरण से लिया जाता है लेकिन अलग-अलग बिंदुओं को शामिल करता है।

### चरण 3: गणना पूर्वाग्रह^2, विविधता विघटन

प्रत्येक परीक्षण बिंदु पर भविष्यवाणियों के 200 सेट के साथ, हम परिभाषा से सीधे विघटन की गणना कर सकते हैंः

```python
mean_pred = predictions.mean(axis=0)
bias_sq = np.mean((mean_pred - y_true) ** 2)
variance = np.mean(predictions.var(axis=0))
total_error = np.mean(np.mean((predictions - y_true) ** 2, axis=1))
```

- `mean_pred`क्या E[f_hat(x) का अनुमान बूटस्ट्रैप नमूनों से लगाया गया है
- `bias_sq`औसत भविष्यवाणी और सत्य के बीच वर्ग अंतर है
- `variance`यह बूटस्ट्रैप नमूने में भविष्यवाणियों का औसत प्रसार है
- `total_error`लगभग बराबर bias^2 + भिन्नता + शोर

### चरण 4: सीखने की वक्रता

सीखने के वक्र प्रशिक्षण सेट के आकार को साफ करते हैं जबकि मॉडल जटिलता को तय करते हैं। वे बताते हैं कि आपका मॉडल डेटा-सीमित है या क्षमता-सीमित है।

```python
def demo_learning_curves():
    sizes = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300]
    degree = 5

    for n in sizes:
        train_errors = []
        test_errors = []
        for seed in range(50):
            x_train, y_train = generate_data(n_samples=n, seed=seed * 100)
            w = fit_polynomial(x_train, y_train, degree)
            train_pred = predict_polynomial(x_train, w)
            train_mse = np.mean((train_pred - y_train) ** 2)
            test_pred = predict_polynomial(x_test, w)
            test_mse = np.mean((test_pred - y_test) ** 2)
            train_errors.append(train_mse)
            test_errors.append(test_mse)
        # Average over runs gives the learning curve point
```

उच्च-विविधता वाले मॉडल (छोटे डेटा के साथ डिग्री 5) के लिए, आप देखते हैंः
- प्रशिक्षण त्रुटि कम शुरू होता है और अधिक डेटा के रूप में वृद्धि स्मृति कठिन बनाता है
- परीक्षण त्रुटि उच्च शुरू होता है और मॉडल अधिक संकेत प्राप्त करने के रूप में कम हो जाता है
- अधिक डेटा के साथ अंतर कम हो जाता है

उच्च पूर्वाग्रह मॉडल (डिग्री 1) के लिए, दोनों त्रुटियां एक ही उच्च मूल्य के लिए जल्दी से अभिसरण करती हैं और अधिक डेटा मदद नहीं करता है।

### चरण 5: नियमितता की जांच

कोड में भी शामिल है `demo_regularization_sweep()`, जो एक उच्च डिग्री बहुपद (15 डिग्री) तय करता है और 0.001 से 100 तक रिंग नियमितता ताकत को स्वीप करता है। यह एक अलग कोण से पूर्वाग्रह-वैरिएंस व्यापार को दिखाता हैः मॉडल जटिलता के बजाय, हम बाधा ताकत को बदलते हैं।

```python
def demo_regularization_sweep():
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    for alpha in alphas:
        results = bias_variance_decomposition([15], lam=alpha)
        r = results[15]
        print(f"alpha={alpha:.3f}  bias={r['bias_sq']:.4f}  var={r['variance']:.4f}")
```

कम अल्फा पर, डिग्री -15 बहुपद लगभग निर्बाध है। भिन्नता हावी होती है क्योंकि मॉडल प्रत्येक बूटस्ट्रैप नमूने में शोर का पीछा करता है। उच्च अल्फा पर, दंड इतना मजबूत होता है कि मॉडल प्रभावी रूप से एक लगभग-समान फ़ंक्शन बन जाता है। पूर्वाग्रह हावी होता है। इष्टतम अल्फा इन चरम के बीच बैठता है।

यह एक ही यू-कुर्ब है जो विभिन्न बहुपद डिग्री से है, लेकिन एक अलग के बजाय एक निरंतर बटन द्वारा नियंत्रित किया जाता है। व्यवहार में, विनियमितता व्यापार को नियंत्रित करने का पसंदीदा तरीका है क्योंकि यह सुविधा सेट को बदलने के बिना बारीक-बीजा नियंत्रण की अनुमति देता है।

## इसका प्रयोग करें

sklearn प्रदान करता है `learning_curve`और `validation_curve`बूटस्ट्रैप लूप लिखने के बिना इन निदान स्वचालित करने के लिए.

### सत्यापन वक्रः स्वीप मॉडल जटिलता

```python
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

degrees = list(range(1, 16))
train_scores_all = []
val_scores_all = []

for d in degrees:
    pipe = make_pipeline(PolynomialFeatures(d), Ridge(alpha=0.01))
    train_scores, val_scores = validation_curve(
        pipe, X, y, param_name="polynomialfeatures__degree",
        param_range=[d], cv=5, scoring="neg_mean_squared_error"
    )
    train_scores_all.append(-train_scores.mean())
    val_scores_all.append(-val_scores.mean())
```

यह आपको सीधे पूर्वाग्रह-वियरिएंस ट्रेडऑफ वक्र देता है। जहां सत्यापन स्कोर ट्रेन स्कोर के सापेक्ष सबसे खराब है, तब भिन्नता हावी होती है। जहां दोनों खराब हैं, तो पूर्वाग्रह हावी होता है।

### सीखने की वक्रता: स्पाइप ट्रेनिंग सेट आकार

```python
from sklearn.model_selection import learning_curve

pipe = make_pipeline(PolynomialFeatures(5), Ridge(alpha=0.01))
train_sizes, train_scores, val_scores = learning_curve(
    pipe, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring="neg_mean_squared_error"
)
train_mse = -train_scores.mean(axis=1)
val_mse = -val_scores.mean(axis=1)
```

कहानी `train_mse`और `val_mse``train_sizes`आकार आपको अपने मॉडल के बारे में सब कुछ बताता है।

### नियमितता स्वीप के साथ क्रॉस-वैलिडेशन

```python
from sklearn.model_selection import cross_val_score

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
for alpha in alphas:
    pipe = make_pipeline(PolynomialFeatures(10), Ridge(alpha=alpha))
    scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_squared_error")
    print(f"alpha={alpha:>7.3f}  MSE={-scores.mean():.4f} +/- {scores.std():.4f}")
```

यह एक निश्चित मॉडल जटिलता के लिए नियमितता ताकत को साफ़ करता है। आप एक ही पूर्वाग्रह-वियरिएंस ट्रेडऑफ देखेंगेः कम अल्फा का मतलब उच्च भिन्नता है, उच्च अल्फा का मतलब उच्च पूर्वाग्रह है।

### सब कुछ एक साथ रखनाः एक पूर्ण निदान कार्यप्रवाह

व्यवहार में, आप इन निदानों को क्रमशः चलाते हैंः

1. अपने मॉडल को प्रशिक्षित करें, ट्रेन की गणना करें और परीक्षण त्रुटि करें।
2. यदि दोनों उच्च हैं तो आपके पास पूर्वाग्रह समस्या है। चरण 4 पर कूदें।
3. यदि ट्रेन कम है लेकिन परीक्षण उच्च हैः आपके पास एक भिन्नता समस्या है। यह देखने के लिए एक सीखने की वक्र उत्पन्न करें कि क्या अधिक डेटा मदद करेगा। यदि नहीं, तो नियमित करें।
4. एक सत्यापन वक्र उत्पन्न करें जो आपके मुख्य जटिलता पैरामीटर को भुनाने।
5. एक अच्छा बिंदु पर, एक सीखने वक्र उत्पन्न करें. यदि अंतर अभी भी बड़ा है, तो आपको अधिक डेटा या नियमितता की आवश्यकता है.
6.  का प्रयोग करके विभिन्न अल्फा मानों के साथ रिज/लससो का प्रयोग करें`cross_val_score`अल्फा चुनें जहां क्रॉस-वैलिडेटेड त्रुटि सबसे कम है।

यह अधिकांश तालिकागत डेटासेट के लिए 10-15 मिनट की गणना लेता है और अनुमान लगाने के घंटों की बचत करता है।

## इसे भेजें

इस सबक से हमें निम्नलिखित लाभ मिलता हैः `outputs/prompt-model-diagnostics.md`

## व्यायाम

1.  के साथ विघटन को चलाएँ`noise_std=0`(कोई शोर नहीं) क्या होता है कि irreducible त्रुटि शब्द? क्या इष्टतम जटिलता बदलता है?

2. प्रशिक्षण सेट का आकार 30 से बढ़ाकर 300 कर दें। यह भिन्नता घटक को कैसे प्रभावित करता है? क्या इष्टतम बहुपद डिग्री बदलता है?

3. प्रयोग में L2 नियमितकरण (रिज रेग्रिशन) जोड़ें। एक निश्चित उच्च-डिग्री बहुपद (डिग्री 15) के लिए, लैम्ब्डा को 0 से 100 तक झाड़ें।

4. एक बहुपद से वास्तविक फ़ंक्शन को  में संशोधित करें`sin(x)`. कैसे पूर्वाग्रह-वारीएंस विघटन बदलता है? क्या अभी भी एक स्पष्ट अनुकूल डिग्री है?

5. एक सरल बूटस्ट्रैप एग्रीगेटिंग (बैगिंग) रैपर लागू करेंः बूटस्ट्रैप नमूनों और औसत भविष्यवाणियों पर 10 मॉडल प्रशिक्षित करें। दिखाएं कि यह पक्षपात को बहुत बढ़ाए बिना भिन्नता को कम करता है।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Bias | "The model is too simple" | Systematic error from wrong assumptions. The gap between the average model prediction and truth. |
| Variance | "The model is overfitting" | Error from sensitivity to training data. How much predictions change across different training sets. |
| Irreducible error | "Noise in the data" | Error from randomness in the true data-generating process. No model can eliminate it. |
| Underfitting | "Not learning enough" | Model has high bias. It misses the real pattern even on training data. |
| Overfitting | "Memorizing the data" | Model has high variance. It fits noise in training data that does not generalize. |
| Regularization | "Constraining the model" | Adding a penalty to reduce model complexity, trading bias for lower variance. |
| Double descent | "More parameters can help" | Test error decreases again when model capacity far exceeds the interpolation threshold. |
| Model complexity | "How flexible the model is" | The capacity of a model to fit arbitrary patterns. Controlled by architecture, features, or regularization. |

## आगे पढ़ना

- [Hastie, Tibshirani, Friedman: Elements of Statistical Learning, Ch. 7](https://hastie.su.domains/ElemStatLearn/)-- पूर्वाग्रह-वियरिएंस विघटन का अंतिम उपचार
- [Belkin et al., Reconciling modern machine learning practice and the bias-variance trade-off (2019)](https://arxiv.org/abs/1812.11118)-- डबल अवतरण कागज
- [Nakkiran et al., Deep Double Descent (2019)](https://arxiv.org/abs/1912.02292)-- युग-विज्ञानी और नमूना-विज्ञानी दोहरी गिरावट
- [Scott Fortmann-Roe: Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html)-- स्पष्ट दृश्य स्पष्टीकरण
