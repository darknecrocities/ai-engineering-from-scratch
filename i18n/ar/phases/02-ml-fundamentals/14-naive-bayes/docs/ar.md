# البراغي (بايز)

> افتراض "الساذج" خاطئ، ويعمل على أي حال، هذا هو الجمال منه.

**Type:** Build
**Language:**بايثون
**Prerequisites:** Phase 2, Lessons 01-07 (classification, Bayes' theorem)
**Time:** ~75 minutes

## أهداف التعلم

- تنفيذ multinomial naive bayes من الصفر مع تسطيح Laplace لتصنيف النص
- شرح لماذا افتراض الاستقلال البطيء خاطئ رياضيًا ولكنه ينتج تصنيفات الطبقات الصحيحة في الممارسة العملية
- مقارنة المتعددة، برنوولي، وغاوسي البديلة Bayes التغيرات واختيار الحق واحد لنوع خاصية معينة
- تقييم البغاء بايز ضد التراجع اللوجستي على البيانات النادرة العالية الأبعاد وتفسير التداول بين التباينات في العمل

## المشكلة

تحتاج إلى تصنيف النص. رسائل البريد الإلكتروني إلى الرسائل غير المرسومة أو غير المرسومة. مراجعات العملاء إلى الإيجابية أو السلبية. تذاكر الدعم إلى الفئات. لديك آلاف الميزات (واحد لكل كلمة) و بيانات التدريب المحدودة.

معظم المصنفين يختنقون هنا. تحتاج الرجعة اللوجستية إلى عينات كافية لتقدير الآلاف من الوزن بشكل موثوق. شجرة القرار تقسم على كلمة واحدة في وقت واحد وتتزايد بشكل وحشي. KNN في 10000 بعد لا معنى لها لأن كل نقطة بعيدة بنفس القدر عن كل نقطة أخرى.

البراذب (بايز) يتولى هذا الأمر يفترض هذا الأمر خطأ رياضيًا (أن كل ميزة مستقلة عن كل ميزة أخرى نظراً للصف) ، وهو ما زال يتفوق على نماذج "أذكى" في تصنيف النص، خاصة مع مجموعات تدريبية صغيرة. إنه يتدرب في مرور واحد عبر البيانات. إنها تتحكم في ملايين الميزات فإنه ينتج تقديرات الاحتمال (على الرغم من أن غالبا ما تكون ضعيفة التعدين بسبب افتراض الاستقلال).

فهم لماذا يؤدي افتراض خاطئ إلى توقعات جيدة يعلمك شيئا أساسيا حول تعلم الآلة: أفضل نموذج ليس الأكثر صحة، هو واحد مع أفضل تحيز-تغيرات التداول للبيانات الخاصة بك.

## المفهوم

### نظرية بايز (تجربة سريعة)

نظرية بايز تغير احتمالات مشروطة:

```
P(class | features) = P(features | class) * P(class) / P(features)
```

نريد`P(class | features)`-- احتمال أن يخص مستند إلى فئة مع العطاء الكلمات في ذلك. يمكننا حساب هذا من:
- `P(features | class)`-- احتمال رؤية هذه الكلمات في وثائق من هذا الفصل
- `P(class)`-- احتمالات المجموعة السابقة (ما هي شيوعية الرسائل غير المرغوب فيها بشكل عام؟)
- `P(features)`-- الأدلة، نفسها لجميع الفصول، حتى نتمكن من تجاهلها عند مقارنة

الفئة ذات أعلى`P(class | features)`فاز

### افتراض استقلالية ساذج

الحوسبة`P(features | class)`وذلك يعني أنّه من الضروريّ أن تُقدّر احتمالية المشتركة لجميع الميزات معاً. مع مجموعة من 10,000 كلمة، ستحتاج إلى تقدير توزيع أكثر من 2^10,000 مزيج ممكن.

افتراض البطيء: كل ميزة مستقلة مشروطاً نظراً للطبقة.

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

بدلا من توزيع واحد غير ممكن المشترك، تقدّر توزيعات بسيطة لكل خصائص. كل واحد فقط يحتاج إلى حساب.

هذا الافتراض خاطئ بشكل واضح. الكلمات "الآلة" و "التعلم" ليست مستقلة في أي وثيقة. ولكن المصنف لا يحتاج إلى تقديرات احتمالية صحيحة. يحتاج إلى تصنيف صحيح - أي فئة لديها أكبر احتمال. افتراض الاستقلال يقدم أخطاء منهجية، ولكن تلك الأخطاء تؤثر على جميع الفئات على نحو مماثل، لذلك يبقى التصنيف صحيحًا.

### لماذا لا يزال يعمل

ثلاثة أسباب:

1. **Ranking over calibration.**التصنيف يحتاج فقط إلى الصف الأعلى للمرتبة صحيحة. حتى لو كان P(سبام) = 0.99999 عندما تكون الاحتمال الحقيقي 0.7, فإن المصنف لا يزال يختار البريد الإلكتروني بشكل صحيح. نحن لا نحتاج إلى الاحتمالات الصحيحة. نحن بحاجة إلى الفائز الصحيح.

2. **High bias, low variance.**افتراض الاستقلال هو سابق قوي. فإنه يقيّد النموذج بشكل كبير، مما يمنع التكيف الزائد. مع بيانات تدريب محدودة، النموذج الخطأ قليلاً ولكن مستقر يفوق النموذج الصحيح نظرياً ولكن غير مستقر بشكل كبير. هذا هو التداول بين التباينات المتغيرة في العمل.

3. **Feature redundancy cancels out.**توفر الميزات المتصلة أدلة زائدة. يقوم المصنف بإعداد هذه الدليل مرتين، لكنه يقوم بإعدادها مرتين بالنسبة للصف الصحيح أيضًا. إذا ظهرت "الآلة" و "التعلم" دائمًا معًا، فإن كلاهما يوفر أدلة للصف "التكنولوجيا". يعد NB هذه الدليلات مرتين، لكنه يعدها مرتين بالنسبة للصف الصحيح.

سبب رابع عملي: البغاء بايز سريع للغاية. التدريب هو مرور واحد عبر ترددات حساب البيانات. التنبؤ هو مضاعفة المصفوفة. يمكنك التدريب على مليون وثيقة في ثوان. هذا السرعة يعني أنه يمكنك التكرار أسرع، ومحاولة مجموعة مزيد من الميزات، وإجراء المزيد من التجارب من النماذج البطيئة.

### الرياضيات خطوة بخطوة

دعونا نتعقب من خلال مثال ملموس. لنفترض أن لدينا فئتين: الرسائل غير المرسومة والرسائل غير المرسومة. قاموسنا يحتوي على ثلاث كلمات: "حرة" "مال" "التقاء".

بيانات التدريب:
- رسائل البريد الإلكتروني المزعج ذكرت "مجان" 80 مرة، "مال" 60 مرة، "التقاء" 10 مرات (150 كلمة في الإجمال)
- رسائل البريد الإلكتروني غير الرسائل البشرية تذكرها "مجان" 5 مرات، "مال" 10 مرات، "التقاء" 100 مرة (115 كلمة في الإجمال)
- 40% من الرسائل الإلكترونية هي الرسائل غير المرسلة، 60% هي غير الرسائل غير المرسلة

مع تسطيح Laplace (ألفا=1):

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

البريد الإلكتروني الجديد يحتوي على: "مجان" (2 مرات) ، "مال" (1 مرة) ، "اجتماع" (0 مرات).

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

يربح الرسائل غير المرغوب فيها بفارق كبير. كلمة "حرة" التي تظهر مرتين هي دليل قوي على الرسائل غير المرغوب فيها. لاحظ أن "التقاء" الذي لا يظهر يساهم صفر في كل من مجموعات السجلات (0 * log(P)) - في المجموعة متعددة النسب، لا يوجد أي تأثير.

### ثلاثة طرق

(بايز) البطيء يأتي في ثلاث طعامات كل طراز`P(feature | class)`بطريقة مختلفة

#### البيانات البديلة متعددة الاسماء

نموذج كل ميزة كعدد. أفضل للبيانات النصية حيث هي ميزات ترددات الكلمات أو قيم TF-IDF.

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

- نعم`alpha`هو تسطيح لابلاس (موضح أدناه). هذا التنوع هو الحصان العمل لتصنيف النص.

#### غوسيان البراغي بييز

النماذج كل ميزة كتوفر طبيعي. أفضل للميزات المستمرة.

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

كل فئة تحصل على متوسطها وتباينها لكل ميزة. هذا يعمل بشكل جيد عندما تتبع الميزات حقا منحنى زجاجة داخل كل فئة.

#### برنولي البراغي بييز

نموذج كل ميزة كمتعدد ثنائي (حاضر أو غائب). أفضل للنص القصير أو متجهات الميزة الثنائية.

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

على عكس Multinomial ، يعاقب Bernoulli صراحة غياب كلمة. إذا ظهرت "حرة" عادة في الرسائل البشرية ولكن غائبة من هذا البريد الإلكتروني ، يعتبر Bernoulli ذلك كدليل ضد الرسائل البشرية.

### متى تستخدم كل نوع

| Variant | Feature Type | Best For | Example |
|---------|-------------|----------|---------|
| Multinomial | Counts or frequencies | Text classification, bag-of-words | Email spam, topic classification |
| Gaussian | Continuous values | Tabular data with normal-ish features | Iris classification, sensor data |
| Bernoulli | Binary (0/1) | Short text, binary feature vectors | SMS spam, presence/absence features |

### "لابلاز"

ماذا يحدث عندما تظهر كلمة في بيانات الاختبار ولكن لم تظهر أبدا في بيانات التدريب للفئة معينة؟

بدون تسطيح:`P(word | class) = 0/N = 0`صفر واحد مضاعف على كل المنتج يجعل`P(class | features) = 0`كلمة واحدة غير مرئية تدمير التنبؤ بأكمله، مهما كانت الأدلة الأخرى تدعمها.

التسهيل يضيف عدد صغير `alpha`(عادة 1) لكل عدد من الميزات:

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

مع ألفا = 1 ، كل كلمة تحصل على احتمال صغير على الأقل. كلمة "تفكيك" التي تظهر في رسالة بريد إلكتروني اختبارية لم تعد تقتل احتمالات البريد الإلكتروني. التسهيل له تفسير بييزي: إنه يعادل وضع ديرشلت موحد قبل توزيع الكلمات.

الفا العليا يعني تسطيع أقوى (مزيد من التوزيعات المتساوية). الفا السفلى يعني أن النموذج يثق بالبيانات أكثر. الفا هو مفاتيح فائقة يمكنك ضبطها.

تأثير الفا:

| Alpha | Effect | When to use |
|-------|--------|-------------|
| 0.001 | Almost no smoothing, trust the data | Very large training set, no unseen features expected |
| 0.1 | Light smoothing | Large training set |
| 1.0 | Standard Laplace smoothing | Default starting point |
| 10.0 | Heavy smoothing, flattens distributions | Very small training set, many unseen features expected |

### حسابات الموقع

مضاعفة مئات من الاحتمالات (كل أقل من 1) تسبب تدفق نقطة عائمة. يصبح المنتج صفر في نقطة عائمة على الرغم من أن القيمة الحقيقية هي عدد إيجابي صغير جدا.

الحل: العمل في مساحة السجلات بدلاً من مضاعفة الاحتمالات، أضف لوغاريتمهم:

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

هذا يحول التنبؤ إلى نسبة نقطة:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

مضاعفة المصفوفة. هذا هو السبب في أن توقعات بييز الباهظة هي سريعة جدا -- انها نفس العملية مثل نموذج خطي من طبقة واحدة.

### البراغي (بايس) مقابل التراجع اللوجستي

كلاهما تصنيفات خطية للنص. الفرق هو ما يُمثّلونه.

| Aspect | Naive Bayes | Logistic Regression |
|--------|------------|-------------------|
| Type | Generative (models P(X\|Y)) | Discriminative (models P(Y\|X)) |
| Training | Count frequencies | Optimize loss function |
| Small data | Better (strong prior helps) | Worse (not enough to estimate weights) |
| Large data | Worse (wrong assumption hurts) | Better (flexible boundary) |
| Features | Assumes independence | Handles correlations |
| Speed | Single pass, very fast | Iterative optimization |
| Calibration | Poor probabilities | Better probabilities |

قاعدة عامة: ابدأ بـ"بايز" البديل، إذا كان لديك بيانات كافية و"بايز" المرتفعات، فانتقل إلى التراجع اللوجستي.

### خط أنابيب التصنيف

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

في الممارسة العملية، نعمل في مساحة السجلات لتجنب تدفقات نقطة عائمة. بدلاً من مضاعفة العديد من الاحتمالات الصغيرة، نضيف محاسباتها:

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## بناءها

الرمز في`code/naive_bayes.py`يطبق كل من MultinomialNB و GaussianNB من الصفر.

### المتعددة النسبNB

التنفيذ من الصفر:

1. **fit(X, y)**: لكل فئة، احتساب تردد كل ميزة. إضافة تسطيح Laplace. حساب احتمالات السجل. تخزين قبلات الفئة (سجل ترددات الفئة).

2. **predict_log_proba(X)**: لكل عينة، حساب سجل P(فئة) + جمع سجل P(صف_i ‬فئة) لجميع الفئات. هذه هي مضاعفة المصفوفة: X @ log_probs.T + log_priors.

3. **predict(X)**: أعد الفئة بأعلى احتمالات التسجيل.

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

المفهوم الرئيسي: بعد التكيف، التنبؤ هو مجرد مضاعفة المصفوفة زائد التحيز. هذا هو السبب في أن الباهى بايز سريع جدا.

### غوسيانNB

بالنسبة للميزات المستمرة، نحن نقدر المتوسط والاختلاف لكل فئة لكل ميزة:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

يستخدم التنبؤات PDF غوسية لكل ميزة ، مضاعفة عبر الميزات (تم إضافةها في مساحة السجل).

### التجربة: تصنيف النص

يولد الرمز بيانات كيس الكلمات الاصطناعية التي تحاكي فئتين (مقالات تكنولوجيا مقابل مقالات رياضية). لكل فئة توزيع تردد كلمات مختلف. تصنفها MultinomialNB باستخدام عدد الكلمات.

تعمل البيانات الاصطناعية على هذا النحو: نخلق 200 "كلمة" (عمودات ميزة). الكلمات 0-39 لها تردد عال في المقالات التقنية والمنخفضة في الرياضة. الكلمات 80-119 لها تردد عال في الرياضة والمنخفضة في التقنية. الكلمات 40-79 هي تردد متوسط في كل منهما. هذا يخلق سيناريو واقعي حيث بعض الكلمات مؤشرات قوية للصف والبعض الآخر ضجيج.

### التجربة: الميزات المستمرة

يولد الرمز بيانات شبيهة بـ Iris (3 فئة ، 4 ميزات ، مجموعات غوسية). تصنف GaussianNB باستخدام المتوسط والفرق لكل فئة. لكل فئة مركز مختلف (متوسط متجه) وانتشار مختلف (فرق) ، مما يحاكي بيانات العالم الحقيقي حيث تختلف القياسات بشكل منهجي بين الفئات.

ويقول أيضاً:
- **Smoothing comparison:**تدريب MultinomialNB مع قيم ألفا مختلفة لإظهار تأثير قوة التسهيل على الدقة.
- **Training size experiment:**كيف تحسن دقة النب مع نمو بيانات التدريب من 20 إلى 1600 عينة. النب يصل إلى دقة لائقة حتى مع عدد قليل جدا من العينات -- هذه هي ميزته الرئيسية.
- **Confusion matrix:**دقة في الفئة، التذكير، والرقم في الفورمولا 1 لإظهار أين يرتكب NB الأخطاء.

### السرعة التنبؤية

تنبؤ بايز البغيض هو مضاعفة المصفوفة. بالنسبة n عينات مع d خصائص و k فئات:
- MultinomialNB: مضاعفة ماتريكية واحدة (n x d) @ (d x k) = O(n * d * k)
- غوسيانNB: n * k تقييمات غوسيان PDF، كل أكثر من d ميزات = O(n * d * k)

كلاهما خطي في كل بعد. مقارن هذا مع KNN (الذي يتطلب حساب المسافة إلى جميع نقاط التدريب) أو SVM مع kernel RBF (الذي يتطلب تقييم النواة ضد جميع المتجهات الدعمية). NB أسرع بأوامر من الكبيرة في وقت التنبؤ.

## استخدمها

مع sklearn، كلتا التغيرات هي خط واحد:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

للتصنيف النصي مع sklearn:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

الرمز في`naive_bayes.py`يُقارن التنفيذات من الصفر مع التنفيذ على نفس البيانات للتحقق من صحة.

### (تف-إيدف) مع (بايز) البديل

تعتبر العد من الكلمات الخام كل كلمة ذات الوزن المتساو في كل حالة. لكن الكلمات الشائعة مثل "ال" و "هو" تظهر بشكل متكرر في كل فئة - لا تحمل أي معلومات. TF-IDF (تردد الموعد - تردد المستند المعاكس) يضع الوزن على الكلمات الشائعة ويعزز الكلمات النادرة والتمييزية.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

قيم TF-IDF غير سلبية ، لذلك تعمل مع MultinomialNB. تعد مزيج TF-IDF + MultinomialNB واحدة من أقوى خطوط أساسية لتصنيف النص. غالبًا ما تغلب على نماذج أكثر تعقيدا على مجموعات بيانات تقل عن 10،000 عينات تدريبية.

### برنوليNB للنص القصير

بالنسبة إلى النص القصير (التغريدات والرسائل الرسومية ، والرداد) ، يمكن أن تتفوق برنوليNB على MultinomialNB. النصوص القصيرة لديها عدد كلمات منخفض ، لذلك تكون المعلومات المتكررة التي يعتمد عليها MultinomialNB ضوضاء. BernoulliNB تهتم فقط بوجود أو غياب ، وهو أكثر موثوقية مع النص القصير.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

- نعم`binary=True`العلم في CountVectorizer يحول جميع العد إلى 0/1. بدونها، BernoulliNB لا يزال يعمل ولكن يرى العد الذي لم يتم تصميمه له.

### تحديد المعدل NB احتمالات

إن احتمالات NB ضعيفة التصفية. عندما تقول NB P(spam) = 0.95, قد تكون الاحتمال الحقيقي 0.7. إذا كنت بحاجة إلى تقديرات احتمالية موثوقة (على سبيل المثال، لتحديد عتبة أو الجمع مع نماذج أخرى) ، استخدم SKULLAARN'S CalibratedClassifierCV:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

هذا يناسب رجعة لوجستية فوق النتائج الخام NB باستخدام التحقق المتقاطع. والاحتمالات الناتجة هي أقرب بكثير إلى ترددات الفئة الحقيقية.

### القوطس المشتركة

1. **Negative feature values.**يتطلب MultinomialNB ميزات غير سلبية. إذا كان لديك قيم سلبية (مثل TF-IDF مع إعدادات معينة أو ميزات موحدة) ، استخدم GaussianNB بدلاً من ذلك ، أو قم بتحويل الميزات لتكون إيجابية.

2. **Zero variance features.**يُقسم GaussianNB بالانحناء. إذا كانت ميزة لها انحناء صفر للفئة (كل القيم متطابقة) ، فإن الحسابات الاحتمالية تنتهي. يضيف الرمز مصطلح مسطح صغير (1e-9) لجميع الانحناءات لمنع ذلك.

3. **Class imbalance.**إذا كان 99% من الرسائل الإلكترونية غير رسائل غير رسائلي، فإن P ((غير رسائلي) = 0.99 السابقة قوية جداً بحيث أنها تغلب على دليل الاحتمال. يمكنك تعيين أسس الدرجة يدوياً أو استخدام المعلم class_prior في sklearn.

4. **Feature scaling.**MultinomialNB لا تحتاج إلى التوسع (يعمل على العد). GaussianNB لا تحتاج إلى التوسع أيضا (إنه يقدر إحصاءات لكل ميزة). وهذا هو ميزة على التراجع اللوجستي و SVM ، والتي هي حساسة لمقاييس الميزات.

## أرسله

هذا الدرس ينتج عن:
- `outputs/skill-naive-bayes-chooser.md`-- مهارة القرار لانتخاب النوع المناسب من النوع
- `code/naive_bayes.py`-- MultinomialNB و GaussianNB من الصفر، مع مقارنة sklearn

### عندما يفشل بايز البطيء

يُفشل ملاحظة النسخ عندما يسبب افتراض الاستقلال تصنيفات غير صحيحة (وليس فقط احتمالات غير صحيحة). يحدث هذا عندما:

1. **Strong feature interactions.**إذا كانت الفئة تعتمد على مزيج من ميزتين وليس أي منهما وحده (أنماط شبيهة بـ XOR) ، فإن NB سوف تفوتها بالكامل. لا توجد دليلًا على كل ميزة وحدها ، ولا يمكن لـ NB دمجها بشكل غير خطي.

2. **Highly correlated features with opposing evidence.**إذا كانت ميزة A تقول "رغوة" و ميزة B تقول "ليس رغوة"، ولكن A و B تتوافق تماما (إنها تتفق دائما في الواقع) ، NB سوف ترى أدلة متناقضة حيث لا يوجد أي.

3. **Very large training sets.**مع وجود بيانات كافية، تتعلم النماذج التمييزية مثل التراجع اللوجستي حدود القرار الحقيقية وتتفوق على NB. افتراض الاستقلال الذي ساعد في بيانات صغيرة الآن يعيق النموذج.

في الممارسة العملية، هذه الطرق الفاشلة نادرة للتصنيف النصي. ميزات النص تعدّد، ضعيفة بشكل فردي، وتعتمد أخطاء افتراض الاستقلال على إلغاءها. بالنسبة للبيانات الجدولية التي لديها بعض الميزات المتصلة بقوة، فكر في التراجع اللوجستي أو النماذج القائمة على الأشجار أولاً.

## التمارين

1. **Smoothing experiment.**تدريب MultinomialNB على بيانات النص مع قيم ألفا من 0.01، 0.1، 1.0, 10.0 و 100.0. دقة المسرحية مقابل ألفا. أين ذروة الأداء؟ لماذا يؤلم الفا عالية جدا؟

2. **Feature independence test.**خذ مجموعة بيانات نصية حقيقية. اختر كلمات مرتبطة بشكل واضح ("الجهاز" و "التعلم") احسب P " كلمة 1 " كلاسة) * P " كلمة 2 " كلاسة) وقارنها مع P " كلمة 1 و كلمة 2 " كلاسة) كم هي خطأ افتراض الاستقلال؟ هل يؤثر على دقة التصنيف؟

3. **Bernoulli implementation.**تمديد الرمز مع فئة بيرنوليNB. تحويل كيس الكلمات إلى ثنائي (حاضر / غائب) ومقارنة الدقة ضد MultinomialNB على بيانات النص. متى يفوز بيرنولي؟

4. **NB vs Logistic Regression.**تدريب كل منهما على بيانات النص. تبدأ مع 100 عينات تدريب وتزداد إلى 10,000. دقة المسرحية مقابل حجم مجموعة تدريب لكل منهما. في أي نقطة تتجاوز رجعة اللوجستية بايز الباهظة؟

5. **Spam filter.**بناء تصنيف كامل للشحنة: إضفاء علامات على النص البريطاني الخام، بناء المفردات، إنشاء ميزات كيس الكلمات، تدريب MultinomialNB، تقييم بدقة وتذكير (ليس فقط دقة -- لماذا?).

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Naive Bayes | "Simple probabilistic classifier" | A classifier that applies Bayes' theorem with the assumption that features are conditionally independent given the class |
| Conditional independence | "Features don't affect each other" | P(A, B \| C) = P(A \| C) * P(B \| C) -- knowing B tells you nothing new about A once you know C |
| Laplace smoothing | "Add-one smoothing" | Adding a small count to every feature to prevent zero probabilities from dominating the prediction |
| Prior | "What you believed before seeing data" | P(class) -- the probability of each class before observing any features |
| Likelihood | "How well the data fits" | P(features \| class) -- the probability of observing these features if the class is known |
| Posterior | "What you believe after seeing data" | P(class \| features) -- the updated probability of the class after observing the features |
| Generative model | "Models how data is generated" | A model that learns P(X \| Y) and P(Y), then uses Bayes' theorem to get P(Y \| X) |
| Discriminative model | "Models the decision boundary" | A model that directly learns P(Y \| X) without modeling how X is generated |
| Log probability | "Avoid underflow" | Working with log P instead of P to prevent the product of many small numbers from becoming zero in floating point |

## المزيد من القراءة

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html)-- جميع الاختلافات الثلاثة مع تفاصيل رياضية
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf)-- المقارنة الكلاسيكية من Multinomial مقابل Bernoulli للنص
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)-- تحسينات في النب للوص
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)-- يثبت أن النب يتقارب أسرع من LR مع بيانات أقل
