# एमएल पाइपलाइन

> एक मॉडल एक उत्पाद नहीं है। एक पाइपलाइन है। पाइपलाइन कच्चे डेटा से लेकर तैनात भविष्यवाणी तक सब कुछ है, और हर कदम को पुनः उत्पन्न किया जाना चाहिए।

**Type:** Build
**Language:**पायथन
**Prerequisites:** Phase 2, Lesson 12 (Hyperparameter Tuning)
**Time:** ~120 minutes

## सीखने के लक्ष्य

- स्क्रैच से एक एमएल पाइपलाइन बनाएं जो इम्प्यूटेशन, स्केलिंग, एन्कोडिंग और मॉडल प्रशिक्षण को एक एकल पुनरुत्पादित वस्तु में जोड़ता है
- डेटा लीक के परिदृश्यों की पहचान करें और समझाएं कि कैसे पाइपलाइन केवल प्रशिक्षण डेटा पर ट्रांसफार्मर लगाकर उन्हें रोकती है
- एक कॉलमट्रांसफार्मर का निर्माण करें जो संख्यात्मक और श्रेणीगत विशेषताओं पर विभिन्न पूर्व-संसाधित को लागू करता है
- पाइपलाइन सीरियलाइजेशन को लागू करें और यह प्रदर्शित करें कि एक ही फिट पाइपलाइन प्रशिक्षण और उत्पादन में समान परिणाम देती है

## समस्या

आपके पास एक नोटबुक है जो डेटा लोड करता है, मिडियन के साथ गायब मानों को भरता है, माप सुविधाओं, ट्रेनों एक मॉडल, और सटीकता प्रिंट करता है। यह काम करता है. आप इसे भेजते हैं।

एक महीने बाद, किसी ने मॉडल को फिर से प्रशिक्षित किया और अलग परिणाम प्राप्त हुए। परीक्षण डेटा (डेटा लीक) सहित पूर्ण डेटासेट पर औसत की गणना की गई। स्केलिंग पैरामीटर सहेजे नहीं गए, इसलिए निष्कर्ष अलग-अलग सांख्यिकी का उपयोग करता है। फीचर इंजीनियरिंग कोड को प्रशिक्षण और सेवा के बीच कॉपी-पेस्ट किया गया था, और कॉपी अलग-अलग थीं। एक श्रेणी स्तंभ को उत्पादन में एक नया मूल्य प्राप्त हुआ जो एन्कोडर ने कभी नहीं देखा है।

ये परिकल्पना नहीं हैं. ये सबसे आम कारण हैं कि एमएल सिस्टम उत्पादन में विफल होते हैं। पाइपलाइनें प्रत्येक परिवर्तन चरण को एक एकल, क्रमबद्ध, पुनरुत्पादित वस्तु में पैकेज करके उन सभी को हल करती हैं।

## अवधारणा

### पाइपलाइन क्या है

एक पाइपलाइन एक मॉडल के बाद डेटा परिवर्तनों का एक क्रमबद्ध अनुक्रम है। प्रत्येक चरण पिछले चरण के आउटपुट को इनपुट के रूप में लेता है। प्रशिक्षण डेटा पर एक बार पूरी पाइपलाइन को फिट किया जाता है। निष्कर्ष के समय, एक ही फिट पाइपलाइन नए डेटा को बदलती है और भविष्यवाणियां उत्पन्न करती है।

```mermaid
flowchart LR
    A[Raw Data] --> B[Impute Missing Values]
    B --> C[Scale Numeric Features]
    C --> D[Encode Categoricals]
    D --> E[Train Model]
    E --> F[Prediction]
```

पाइपलाइन गारंटी देता हैः
- परिवर्तन केवल प्रशिक्षण डेटा पर लगाए जाते हैं (कोई लीक नहीं)
- इन्फरेक्शन समय पर समान परिवर्तन लागू होते हैं
- पूरे वस्तु को एक कलाकृतियों के रूप में क्रमबद्ध और तैनात किया जा सकता है
- पार-मान्यीकरण पाइपलाइन को प्रति तह लागू करता है, जिससे सूक्ष्म लीक को रोकना संभव हो जाता है

### डेटा लीकः चुपचाप हत्यारा

डेटा लीक तब होता है जब परीक्षण सेट से जानकारी या भविष्य के डेटा प्रशिक्षण को प्रदूषित करते हैं। पाइपलाइन सबसे आम रूपों को रोकती है।

**Leaky (wrong):**
```python
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test = X_scaled[:800], X_scaled[800:]
y_train, y_test = y[:800], y[800:]
```

स्केलर ने परीक्षण डेटा देखा। औसत और मानक विचलन में परीक्षण नमूने शामिल हैं। यह सटीकता अनुमानों को बढ़ाता है।

**Correct:**
```python
X_train, X_test = X[:800], X[800:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

पाइपलाइन के साथ आपको इसके बारे में सोचने की जरूरत नहीं है। पाइपलाइन इसे स्वचालित रूप से संभालती है।

### स्क्लेर्न पाइपलाइन

sklearn के `Pipeline`चेन ट्रांसफार्मर और एक अनुमानक. यह उजागर करता है `.fit()`,`.predict()`और `.score()`जो सभी चरणों को क्रमशः लागू करते हैं।

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

जब आप कॉल करते हैं`pipe.fit(X_train, y_train)`:
1. स्केलर कॉल `fit_transform`एक्स_ट्रेन पर
2. मॉडल कॉल `fit`स्केल X_train पर

जब आप कॉल करते हैं`pipe.predict(X_test)`:
1. स्केलर कॉल `transform`(फिट_ट्रांसफॉर्म नहीं) पर X_test
2. मॉडल कॉल `predict`स्केल X_test पर

स्केलर कभी भी परीक्षण डेटा को फिटिंग के दौरान नहीं देखता है। यह सब बिंदु है।

### स्तंभट्रांसफार्मरः विभिन्न स्तंभों के लिए विभिन्न पाइपलाइन

वास्तविक डेटासेट में संख्यात्मक और श्रेणीगत स्तंभ होते हैं जिन्हें अलग-अलग पूर्व प्रसंस्करण की आवश्यकता होती है। `ColumnTransformer`यह सब संभालता है।

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, ["age", "income", "score"]),
    ("cat", categorical_pipe, ["city", "gender", "plan"]),
])

full_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", GradientBoostingClassifier()),
])
```

`handle_unknown="ignore"`जब एक नई श्रेणी दिखाई देती है (एक शहर मॉडल कभी नहीं देखा है), यह दुर्घटना के बजाय एक शून्य वेक्टर का उत्पादन करता है।

### प्रयोगों का पता लगाना

एक पाइपलाइन प्रशिक्षण को पुनः उत्पन्न करने योग्य बनाता है, लेकिन आपको यह भी ट्रैक करने की आवश्यकता है कि प्रयोगों में क्या हुआः किस हाइपरपैरामीटर का उपयोग किया गया था, किस डेटासेट संस्करण, क्या मीट्रिक थे, कौन सा कोड चलाया गया था।

**MLflow**सबसे आम ओपन सोर्स समाधान हैः

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)

    pipe.fit(X_train, y_train)
    accuracy = pipe.score(X_test, y_test)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(pipe, "model")
```

प्रत्येक रन को पैरामीटर, मेट्रिक्स, कलाकृतियों और पूर्ण मॉडल के साथ रिकॉर्ड किया जाता है। आप रन की तुलना कर सकते हैं, किसी भी प्रयोग को पुनः पेश कर सकते हैं, और किसी भी मॉडल संस्करण को तैनात कर सकते हैं।

**Weights & Biases (wandb)**एक होस्ट किए गए डैशबोर्ड के साथ एक ही कार्यक्षमता प्रदान करता हैः

```python
import wandb

wandb.init(project="my-pipeline")
wandb.config.update({"max_depth": 5, "n_estimators": 100})

pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)

wandb.log({"accuracy": accuracy})
```

### मॉडल संस्करण

प्रयोगों के बाद, आपको मॉडल संस्करणों का प्रबंधन करने की आवश्यकता है. कौन सा मॉडल उत्पादन में है?

MLflow के मॉडल रजिस्ट्री में निम्नलिखित प्रावधान है:
- **Version tracking:**हर सहेजे गए मॉडल एक संस्करण संख्या प्राप्त करता है
- **Stage transitions:**"स्टेजिंग", "प्रोडक्शन", "आर्काइव्ड"
- **Approval workflow:**मॉडल को स्पष्ट रूप से उत्पादन में बढ़ावा दिया जाना चाहिए
- **Rollback:**तत्काल पूर्ववर्ती संस्करण पर वापस स्विच करें

### डीवीसी के साथ डेटा वर्शनिंग

कोड git के साथ संस्करणित है। डेटा को भी संस्करणित किया जाना चाहिए, लेकिन git बड़ी फ़ाइलों को संभाल नहीं सकता है। DVC (डेटा संस्करण नियंत्रण) इस समस्या को हल करता है।

```
dvc init
dvc add data/training.csv
git add data/training.csv.dvc data/.gitignore
git commit -m "Track training data"
dvc push
```

डीवीसी वास्तविक डेटा को रिमोट स्टोरेज (एस 3, जीसीएस, अज़ूर) में संग्रहीत करता है और एक छोटा सा `.dvc`Git में फ़ाइल हैश रिकॉर्ड करता है. जब आप एक git प्रतिबद्धता चेक,`dvc checkout`सही डेटा जो इस्तेमाल किया गया था बहाल करता है।

इसका मतलब है कि प्रत्येक गिट कॉमिट पिन्स कोड और डेटा दोनों को पूर्ण पुनः प्रयोज्य।

### पुनः प्रयोज्य प्रयोग

पुनरुत्पादित प्रयोग के लिए चार चीज़ें आवश्यक होती हैंः

1. **Fixed random seeds:**नम्पी, रैंडम और फ्रेमवर्क (टर्च, sklearn) के लिए बीज सेट करें
2. **Pinned dependencies:**सटीक संस्करणों के साथ requirements.txt या poetry.lock
3. **Versioned data:**डीवीसी या इसी तरह का
4. **Config files:**एक कॉन्फ़िग में सभी हाइपरपरपैरामीटर, हार्डकोड नहीं

```python
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
```

### नोटबुक से उत्पादन पाइपलाइन तक

```mermaid
flowchart TD
    A[Jupyter Notebook] --> B[Extract functions]
    B --> C[Build Pipeline object]
    C --> D[Add config file for hyperparameters]
    D --> E[Add experiment tracking]
    E --> F[Add data validation]
    F --> G[Add tests]
    G --> H[Package for deployment]

    style A fill:#fdd,stroke:#333
    style H fill:#dfd,stroke:#333
```

सामान्य प्रगति:

1. **Notebook exploration:**त्वरित प्रयोग, विज़ुअलाइज़ेशन, फीचर आइडिया
2. **Extract functions:**प्रीप्रोसेसिंग, फीचर इंजीनियरिंग, मूल्यांकन को मॉड्यूल में स्थानांतरित करें
3. **Build Pipeline:**एक sklearn पाइपलाइन या कस्टम वर्ग में श्रृंखला परिवर्तन
4. **Config management:**सभी हाइपरपरपैरामीटर को यैमएल/जेएसओएन कॉन्फ़िग में स्थानांतरित करें
5. **Experiment tracking:**MLflow या wandb लॉगिंग जोड़ें
6. **Data validation:**प्रशिक्षण से पहले योजनाओं, वितरणों और यादगार मूल्य पैटर्न की जांच करें
7. **Tests:**ट्रांसफार्मर के लिए यूनिट टेस्ट, पूरे पाइपलाइन के लिए इंटीग्रेशन टेस्ट
8. **Deployment:**पाइपलाइन को सीरियल बनाएं, एपीआई (फास्टएपीआई, फ्लास्क) में लपेटें, कंटेनर बनाएं

### पाइपलाइन की आम गलतियाँ

| Mistake | Why it is bad | Fix |
|---------|-------------|-----|
| Fitting on full data before splitting | Data leakage | Use Pipeline with cross_val_score |
| Feature engineering outside pipeline | Different transforms at train vs serve | Put all transforms in the Pipeline |
| Not handling unknown categories | Production crash on new values | OneHotEncoder(handle_unknown="ignore") |
| Hardcoded column names | Breaks when schema changes | Use column name lists from config |
| No data validation | Silently wrong predictions on bad data | Add schema checks before prediction |
| Training/serving skew | Model sees different features in prod | One Pipeline object for both |

```figure
f3-pipeline-flow
```

## इसे बनाओ

कोड में `code/pipeline.py`एक पूर्ण एमएल पाइपलाइन को खरोंच से बनाता हैः

### चरण 1: कस्टम ट्रांसफार्मर

```python
class CustomTransformer:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.means) / self.stds

    def fit_transform(self, X):
        return self.fit(X).transform(X)
```

### चरण 2: खरोंच से पाइपलाइन

```python
class PipelineFromScratch:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        X_current = X.copy()
        for name, step in self.steps[:-1]:
            X_current = step.fit_transform(X_current)
        name, model = self.steps[-1]
        model.fit(X_current, y)
        return self

    def predict(self, X):
        X_current = X.copy()
        for name, step in self.steps[:-1]:
            X_current = step.transform(X_current)
        name, model = self.steps[-1]
        return model.predict(X_current)
```

### चरण 3: पाइपलाइन के साथ क्रॉस-वैलिडेशन

कोड यह दर्शाता है कि पाइपलाइन के साथ क्रॉस-वैलिडेशन डेटा लीक को कैसे रोकता हैः स्केलर प्रत्येक तह के प्रशिक्षण डेटा पर अलग से फिट होता है।

### चरण 4: स्क्लेयरन के साथ पूर्ण उत्पादन पाइपलाइन

 के साथ एक पूर्ण पाइपलाइन`ColumnTransformer`, कई पूर्व प्रसंस्करण पथ, और एक मॉडल, उचित क्रॉस-वैलिडेशन और प्रयोग लॉगिंग के साथ प्रशिक्षित।

## इसे भेजें

इस पाठ से उत्पन्न होता हैः
- `outputs/prompt-ml-pipeline.md`-- एमएल पाइपलाइन बनाने और डिबग करने की क्षमता
- `code/pipeline.py`-- एक पूर्ण पाइपलाइन खरोंच से sklearn के माध्यम से

## व्यायाम

1. एक पाइपलाइन बनाएं जो 3 संख्यात्मक स्तंभों और 2 श्रेणीबद्ध स्तंभों के साथ एक डेटासेट को संभालता है। उपयोग `ColumnTransformer`मध्यवर्ती गणना + संख्यात्मक पर स्केलिंग और सबसे अधिक बार गणना + एक गर्म कोडिंग को श्रेणीबद्ध करने के लिए लागू करें। 5-गुना क्रॉस-वैलिडेशन के साथ अभ्यास करें।

2. जानबूझकर डेटा लीक शुरू करेंः विभाजन से पहले पूरे डेटासेट पर स्केलर को फिट करें। क्रॉस-वैलिडेशन स्कोर (लीक) की तुलना पाइपलाइन क्रॉस-वैलिडेशन स्कोर (क्लीन) से करें। अंतर कितना बड़ा है?

3. अपने पाइपलाइन को क्रमबद्ध करें `joblib.dump`इसे एक अलग स्क्रिप्ट में लोड करें और भविष्यवाणियों को चलाएं।

4. पाइपलाइन में एक कस्टम ट्रांसफार्मर जोड़ें जो दो सबसे महत्वपूर्ण संख्यात्मक स्तंभों के लिए बहुपद विशेषताएं (ग्रेड 2) बनाता है। पाइपलाइन में इसे कहां जाना चाहिए?

5. पाइपलाइन के लिए एमएलफ्लो ट्रैकिंग सेट करें। विभिन्न हाइपरपरपैरामीटर के साथ 5 प्रयोग करें। एमएलफ्लो UI का उपयोग करें (`mlflow ui`) को तुलना करने और सबसे अच्छा मॉडल चुनने के लिए।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Pipeline | "Chain of transforms + model" | An ordered sequence of fitted transformers and a model, applied as one unit to prevent leakage |
| Data leakage | "Test info leaked into training" | Using information from outside the training set to build the model, inflating performance estimates |
| ColumnTransformer | "Different preprocessing per column" | Applies different pipelines to different subsets of columns, combining results |
| Experiment tracking | "Logging your runs" | Recording parameters, metrics, artifacts, and code versions for every training run |
| MLflow | "Track and deploy models" | Open-source platform for experiment tracking, model registry, and deployment |
| DVC | "Git for data" | Version control system for large data files, storing hashes in git and data in remote storage |
| Model registry | "Model version catalog" | A system that tracks model versions with stage labels (staging, production, archived) |
| Training/serving skew | "It worked in the notebook" | Differences between how data is processed during training versus inference, causing silent errors |
| Reproducibility | "Same code, same result" | The ability to get identical results from the same code, data, and configuration |

## आगे पढ़ना

- [scikit-learn Pipeline docs](https://scikit-learn.org/stable/modules/compose.html)-- आधिकारिक पाइपलाइन संदर्भ
- [MLflow documentation](https://mlflow.org/docs/latest/index.html)-- प्रयोगों का ट्रैक और मॉडल रजिस्टर
- [DVC documentation](https://dvc.org/doc)-- डेटा संस्करण
- [Sculley et al., Hidden Technical Debt in Machine Learning Systems (2015)](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)-- एमएल सिस्टम की जटिलता पर अग्रणी पेपर
- [Google ML Best Practices: Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)-- व्यावहारिक उत्पादन एमएल सलाह
