# التعامل مع البيانات غير المتوازنة

> عندما تكون 99٪ من بياناتك "طبيعية"، الدقة هي كذبة.

**Type:** Build
**Language:**بايثون
**Prerequisites:** Phase 2, Lessons 01-09 (especially evaluation metrics)
**Time:** ~90 minutes

## أهداف التعلم

- تنفيذ SMOTE من الصفر وتوضيح كيف تختلف الإجراءات الإجمالية عن التكرار العشوائي
- تقييم المصنفين غير المتوازنين باستخدام معدل F1 ، AUPRC ، و Matthews Correlation Coefficient بدلاً من الدقة
- مقارنة معوزة الفئة، وتحديد العد الأدنى، واستراتيجيات إعادة العينة واختيار النهج المناسب لنسبة عدم التوازن المحددة
- بناء خط أنابيب بيانات غير متوازنة كاملة تجمع بين SMOTE ووزن الفئة وتحسين العد

## المشكلة

تقوم ببناء نموذج للكشف عن الاحتيال، يحصل على دقة 99.9٪، تحتفل، ثم تدرك أنه يتوقع "لا احتيال" لكل معاملة واحدة.

هذا ليس خطأ. إنه الشيء المنطقي الذي يجب القيام به عندما يكون 0.1% فقط من المعاملات خدعة. يتعلم النموذج أن تخمين الطبقة الأغلبية دائمًا يقلل من الأخطاء الإجمالية. إنه صحيح تقنيًا وغير مجدي.

يحدث هذا في كل مكان حيثما كان الأمر من المهمة. تشخيص المرض: 1٪ معدل إيجابي. تدخل الشبكة: 0.01٪ هجمات. عيوب التصنيع: 0.5% عيب. تصفية الرسائل غير المرغوب فيها: 20٪ الرسائل غير المرغوب فيها. توقعات التشغيل: 5٪ التشغيل. كلما زادت النتائج من فئة الأقلية، كلما زاد النادرة.

إن الدقة تفشل لأنه يعامل جميع التنبؤات الصحيحة على قدم المساواة. تسمى المعاملة المشروعة بشكل صحيح والقبض على الاحتيال بشكل صحيح يعتبر كلا نقطة دقة واحدة. ولكن القبض على الاحتيال هو السبب الكامل للوجود في النموذج. نحتاج إلى قياسات وتقنيات واستراتيجيات تدريب تجبر النموذج على إيلاء اهتمام لهذه الفئة النادرة ولكن المهمة.

## المفهوم

### لماذا تفشل الدقة

فكر في مجموعة بيانات مع 1000 عينات: 990 سلبية، 10 إيجابية. نموذج الذي يتوقع دائما سلبية:

|  | Predicted Positive | Predicted Negative |
|--|---|---|
| Actually Positive | 0 (TP) | 10 (FN) |
| Actually Negative | 0 (FP) | 990 (TN) |

الدقة = (0 + 990) / 1000 = 99.0%

النموذج يكتشف صفر احتيال، صفر مرض، صفر عيوب، لكن الدقة تقول 99٪، لهذا السبب الدقة خطيرة للمشاكل غير المتوازنة.

### أرقام أفضل

**Precision**= TP / (TP + FP). من كل شيء يتم وضع علامة إيجابية، كم عدد في الواقع؟ الدقة العالية يعني القليل من الإنذارات الكاذبة.

**Recall**= TP / (TP + FN). من كل شيء إيجابي في الواقع، كم من قبضنا عليه؟ التذكير العالي يعني القليل من الإيجابيات المفقودة.

**F1 Score**= 2 * دقة * استدعاء / (دقة + استدعاء). المتوسط الهارموني. يعاقب عدم التوازن الشديد بين الدقة والإستدعاء أكثر من المتوسط الحسابي.

**F-beta Score**= (1 + بيتا^2) * دقة * تذكر / (بيتا^2 * دقة + تذكر). عندما تكون بيتا > 1 ، فإن التذكر أكثر أهمية. عندما تكون بيتا < 1 ، فإن الدقة أكثر أهمية. F2 شائعة في الكشف عن الاحتيال (إن غياب الاحتيال أسوأ من الإنذار الكاذب).

**AUPRC**(منطقة تحت منحنى الاستدعاء الدقيق). مثل AUC-ROC ولكن أكثر معلوماتًا للبيانات غير المتوازنة. مصنف عشوائي لديه AUPRC يساوي معدل الفئة الإيجابية (ليس 0.5 مثل ROC). وهذا يجعل التحسينات أسهل في رؤية.

**Matthews Correlation Coefficient**= (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)). يتراوح من -1 إلى +1. يعطي درجة عالية فقط عندما يعمل النموذج بشكل جيد على كلا الصفين. متوازن حتى عندما تكون الصفين مختلفة جداً.

بالنسبة لنموذج "تنبؤ دائمًا بالسلب" أعلاه: الدقة = 0/0 (غير محددة ، غالبًا ما يتم تعيينها إلى 0) ، والذكرى = 0/10 = 0 ، F1 = 0 ، MCC = 0. هذه المقاييس تحدد بشكل صحيح النموذج بأنه غير قيم.

### خط الأنابيب المزمنة للبيانات

```mermaid
flowchart TD
    A[Imbalanced Dataset] --> B{Imbalance Ratio?}
    B -->|Mild: 80/20| C[Class Weights]
    B -->|Moderate: 95/5| D[SMOTE + Threshold Tuning]
    B -->|Severe: 99/1| E[SMOTE + Class Weights + Threshold]
    C --> F[Train Model]
    D --> F
    E --> F
    F --> G[Evaluate with F1 / AUPRC / MCC]
    G --> H{Good Enough?}
    H -->|No| I[Try Different Strategy]
    H -->|Yes| J[Deploy with Monitoring]
    I --> B
```

### المعلومات: تقنية اختبار الأقليات الاصطناعية

يكرر العينة المُتَعَدّدَة العشوائية العينات القليلة الحالية. هذا يعمل ولكن هناك خطر من المُتَعَدّدَة لأن النموذج يرى النقاط المتطابقة مراراً وتكراراً.

تقوم SMOTE بإنشاء عينات جديدة من الأقليات الاصطناعية التي يمكن اعتبارها ولكن ليست نسخ.

1. لكل عينة من الأقليات x، العثور على أقرب جيرانها k بين عينة من الأقليات الأخرى
2. اختر جيرانه عشوائيا
3. إعداد عينة جديدة على قطاع الخط بين x و ذلك الجار

الصيغة:`new_sample = x + random(0, 1) * (neighbor - x)`

هذا يتداخل بين نقاط الأقلية الحقيقية، وخلق عينات في نفس المنطقة من مساحة الميزات دون مجرد نسخ البيانات القائمة.

```mermaid
flowchart LR
    subgraph Original["Original Minority Points"]
        P1["x1 (1.0, 2.0)"]
        P2["x2 (1.5, 2.5)"]
        P3["x3 (2.0, 1.5)"]
    end
    subgraph SMOTE["SMOTE Generation"]
        direction TB
        S1["Pick x1, neighbor x2"]
        S2["random t = 0.4"]
        S3["new = x1 + 0.4*(x2-x1)"]
        S4["new = (1.2, 2.2)"]
        S1 --> S2 --> S3 --> S4
    end
    Original --> SMOTE
    subgraph Result["Augmented Set"]
        R1["x1 (1.0, 2.0)"]
        R2["x2 (1.5, 2.5)"]
        R3["x3 (2.0, 1.5)"]
        R4["synthetic (1.2, 2.2)"]
    end
    SMOTE --> Result
```

### استراتيجيات أخذ العينات مقارنة

**Random Oversampling**: نموذج الأقلية المكررة لتطابق عدد الأغلبية.
- المزايا: بسيطة، لا فقدان للمعلومات
- السلبيات: النسخ المزدوجة تسبب الإصلاح المفرط، ويزيد من وقت التدريب

**Random Undersampling**: إزالة عينات الأغلبية لتطابق عدد الأقليات.
- المزايا: تدريب سريع، بسيط
- السلبيات: يرمي البيانات الأغلبية المفيدة المحتملة، والفرقة العالية

**SMOTE**: إنشاء عينات الأقلية الاصطناعية عن طريق التقاط.
- المزايا: تولد نقاط بيانات جديدة، وتقلل من الإفراط في التكيف مقارنة مع الإفراط في أخذ العينات العشوائية
- السلبيات: يمكن أن تخلق عينات ضوضاء بالقرب من حدود القرار، لا تأخذ في الاعتبار توزيع الفئة الأغلبية

| Strategy | Data Changed | Risk | When to Use |
|----------|-------------|------|-------------|
| Oversample | Minority duplicated | Overfitting | Small datasets, moderate imbalance |
| Undersample | Majority removed | Information loss | Large datasets, want fast training |
| SMOTE | Synthetic minority added | Boundary noise | Moderate imbalance, enough minority samples for k-NN |

### الوزن الفريقي

بدلاً من تغيير البيانات، قم بتغيير طريقة التعامل مع الأخطاء في النموذج. قم بتخصيص وزن أكبر لتصنيف الفئة الأقلية بشكل خاطئ.

بالنسبة لمشكلة ثنائية مع 950 عينة سلبية و 50 عينة إيجابية:
- الوزن للطبقة السلبية = n_samples / (2 * n_negative) = 1000 / (2 * 950) = 0.526
- الوزن للطبقة الإيجابية = n_samples / (2 * n_positive) = 1000 / (2 * 50) = 10.0

الفئة الإيجابية تحصل على 19x الوزن. تكلّف سوء تصنيف عينة إيجابية واحدة بقدر ما تكلف خطأ تصنيف 19 عينة سلبية. يتم إجبار النموذج على الإنتباه إلى فئة الأقلية.

في التراجع اللوجستي، هذا يغير وظيفة الخسارة:

```
weighted_loss = -sum(w_i * [y_i * log(p_i) + (1-y_i) * log(1-p_i)])
```

حيث w_i يعتمد على فئة العينة i.

ويعادل وزن الفئة رياضياً إلى زيادة العينات في التوقعات، ولكن دون إنشاء نقاط بيانات جديدة. وهذا يجعلها أسرع ويجنب خطر الإفراط في تجميع العينات المكررة.

### تغيير العدوان

معظم المصنفين يخرجون احتمال. عند P ((إيجابي) >= 0.5 ، فإن التنبؤ بالإيجابي. ولكن 0.5 هو تعسفي. عندما تكون الفئات غير متوازنة ، فإن العد الأفضل هو عادة أقل بكثير.

العملية:
1. تدريب النموذج
2. الحصول على الاحتمالات المتوقعة على مجموعة التحقق
3. حدود التصفية من 0.0 إلى 1.0
4. احسب F1 (أو المقياس الذي اخترته) في كل عتبة
5. اختر العد الذي يزيد من مقياسك

```mermaid
flowchart LR
    A[Model] --> B[Predict Probabilities]
    B --> C[Sweep Thresholds 0.0 to 1.0]
    C --> D[Compute F1 at Each]
    D --> E[Pick Best Threshold]
    E --> F[Use in Production]
```

قد يخرج نموذج P ((احتيال) = 0.15 لعملات احتيالية. عند العدالة 0.5 ، يتم تصنيف هذا بأنه ليس احتيالًا. عند العدالة 0.10 ، يتم القبض عليه بشكل صحيح. تحديد الاحتمالات لا يهم أكثر من التصنيف - طالما أن الاحتيال يحصل على احتمالات أعلى من غير الاحتيال ، فهناك عدالة تفصل بينهما.

### التعلم الذي لا يتكلف

التعميم في الوزن الطبقي بدلاً من تكاليف موحدة، تعيين تكاليف خطأ محددة:

| | Predict Positive | Predict Negative |
|--|---|---|
| Actually Positive | 0 (correct) | C_FN = 100 |
| Actually Negative | C_FP = 1 | 0 (correct) |

تكلّف إغفال معاملة احتيالية (FN) 100 مرة أكثر من إنذار مزيف (FP). يُحسن النموذج للتكلفة الإجمالية، وليس عدد الأخطاء الإجمالية.

هذا هو النهج الأكثر مبدأًا عندما يمكنك تقدير التكاليف في العالم الحقيقي. التشخيص المفقود للسرطان له تكلفة مختلفة جداً عن الإنذار الكاذب الذي يؤدي إلى عملية تجزئة زائدة. جعل هذه التكاليف صريحة يضطر للتنازل الصحيح.

### مخطط تدفق القرار

```mermaid
flowchart TD
    A[Start: Imbalanced Dataset] --> B{How imbalanced?}
    B -->|"< 70/30"| C["Mild: try class weights first"]
    B -->|"70/30 to 95/5"| D["Moderate: SMOTE + class weights"]
    B -->|"> 95/5"| E["Severe: combine multiple strategies"]
    C --> F{Enough data?}
    D --> F
    E --> F
    F -->|"< 1000 samples"| G["Oversample or SMOTE, avoid undersampling"]
    F -->|"1000-10000"| H["SMOTE + threshold tuning"]
    F -->|"> 10000"| I["Undersampling OK, or class weights"]
    G --> J[Train + Evaluate with F1/AUPRC]
    H --> J
    I --> J
    J --> K{Recall high enough?}
    K -->|No| L[Lower threshold]
    K -->|Yes| M{Precision acceptable?}
    M -->|No| N[Raise threshold or add features]
    M -->|Yes| O[Ship it]
```

```figure
class-imbalance
```

## بناءها

### الخطوة 1: إنشاء مجموعة بيانات غير متوازنة

```python
import numpy as np


def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    rng = np.random.RandomState(seed)

    X_maj = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    X_min = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]
```

### الخطوة الثانية: إعادة التشغيل من الصفر

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def find_k_neighbors(X, idx, k):
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def smote(X_minority, k=5, n_synthetic=100, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)
    synthetic = []

    for _ in range(n_synthetic):
        idx = rng.randint(0, n_samples)
        neighbors = find_k_neighbors(X_minority, idx, k)
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)

    return np.array(synthetic)
```

### الخطوة الثالثة: الإجراءات العشوائية المفرطة والإجراءات المقللة

```python
def random_oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = list(X)
    y_resampled = list(y)

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_resampled = []
    y_resampled = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        X_resampled.extend(X[chosen])
        y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]
```

### الخطوة الرابعة: تراجع اللوجستية مع أوزان الفئة

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        weighted_error = error * weights

        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)

        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])
```

### الخطوة 5: ضبط العدوان

```python
def find_optimal_threshold(y_true, y_probs, metric="f1"):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```

### الخطوة 6: وظائف التقييم

```python
def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }
```

### الخطوة السابعة: مقارنة جميع الطرق

```python
X, y = make_imbalanced_data(950, 50, seed=42)
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Baseline: no treatment
w_base, b_base = logistic_regression_weighted(
    X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300
)
probs_base = sigmoid(X_test @ w_base + b_base)
preds_base = (probs_base >= 0.5).astype(int)

# Oversampled
X_over, y_over = random_oversample(X_train, y_train)
w_over, b_over = logistic_regression_weighted(
    X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300
)
preds_over = (sigmoid(X_test @ w_over + b_over) >= 0.5).astype(int)

# SMOTE
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
synthetic = smote(X_minority, k=5, n_synthetic=len(y_train) - 2 * int(minority_mask.sum()))
X_smote = np.vstack([X_train, synthetic])
y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
w_sm, b_sm = logistic_regression_weighted(
    X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300
)
preds_smote = (sigmoid(X_test @ w_sm + b_sm) >= 0.5).astype(int)

# Class weights
sample_weights = compute_class_weights(y_train)
w_cw, b_cw = logistic_regression_weighted(
    X_train, y_train, sample_weights, lr=0.1, epochs=300
)
probs_cw = sigmoid(X_test @ w_cw + b_cw)
preds_cw = (probs_cw >= 0.5).astype(int)

# Threshold tuning (tune on held-out validation set, not test set)
probs_val = sigmoid(X_val @ w_cw + b_cw)
best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val, metric="f1")
preds_thresh = (probs_cw >= best_thresh).astype(int)
```

الملف الرمزي يدير كل هذا في نص واحد ويطبع النتائج.

## استخدمها

مع التعلم المزمن والتعلم غير المتوازن، هذه التقنيات هي خط واحد:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model_weighted = LogisticRegression(class_weight="balanced")
model_weighted.fit(X_train, y_train)
print(classification_report(y_test, model_weighted.predict(X_test)))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression()
model_smote.fit(X_resampled, y_resampled)
print(classification_report(y_test, model_smote.predict(X_test)))

pipeline = Pipeline([
    ("smote", SMOTE()),
    ("model", LogisticRegression(class_weight="balanced")),
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```

تظهر التطبيقات من الصفر بالضبط ما تفعله كل تقنية. SMOTE هو مجرد التقاطع k-NN على فئة الأقلية. وزن الفئة مضاعفة الخسارة. ضبط العدالة هو حلقة للخروج فوق الحطام. لا سحر.

## أرسله

هذا الدرس ينتج عن:
- `outputs/skill-imbalanced-data.md`-- قائمة تفحص للقرارات لمعالجة مشاكل التصنيف غير المتوازنة

## التمارين

1. **Borderline-SMOTE**: تعديل تنفيذ SMOTE لتوليد عينات اصطناعية فقط للنقاط الأقلية التي تقترب من حدود القرار (التي تشمل الجيران الأقرب من k عينات فئة الأغلبية). مقارنة النتائج مع SMOTE القياسية على مجموعة بيانات حيث تتداخل الفئات.

2. **Cost matrix optimization**: تنفيذ التعلم الحساس للتكلفة حيث تكون المصفوفة التكلفة مبرمجة. إنشاء وظيفة تأخذ مصفوفة التكلفة وتعطي التنبؤات الأمثل التي تقلل من التكلفة المتوقعة. اختبر مع نسب التكلفة المختلفة (1:10, 1:100, 1:1000) وتخطيط كيف تتغير التكافؤات بين التذكر الدقيق.

3. **Threshold calibration**: تنفيذ مقياسات اللوحة (تناسب رجعة لوجستية على المخرجات الخام للنموذج لإنتاج احتمالات مقياسة). مقارنة منحنى الاستعادة الدقيقة قبل وبعد التصوير. أظهر أن التصوير لا يغير التصنيف (AUC يبقى نفسه) ولكن يجعل الاحتمالات أكثر معنى.

4. **Ensemble with balanced bagging**: تدريب نماذج متعددة، كل منها على عينة توازن من التشغيل (كل الأقلية + مجموعة فرعية عشوائية من الأغلبية). متوسط توقعاتهم. مقارنة هذا النهج مع نموذج واحد مع SMOTE. قياس كل من الأداء والتشابه عبر الجوائز.

5. **Imbalance ratio experiment**: تأخذ مجموعة بيانات متوازنة وزيادة نسبة عدم التوازن تدريجيا (50/50, 70/30, 90/10, 95/5, 99/1). لكل نسبة، تدريب مع ودون SMOTE.

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Class imbalance | "One class has way more samples" | The distribution of classes in the dataset is significantly skewed, causing models to favor the majority class |
| SMOTE | "Synthetic oversampling" | Creates new minority samples by interpolating between existing minority samples and their k-nearest minority neighbors |
| Class weights | "Making errors on rare classes more expensive" | Multiplying the loss function by class-specific weights so the model penalizes minority misclassification more heavily |
| Threshold tuning | "Moving the decision boundary" | Changing the probability cutoff for classification from the default 0.5 to a value that optimizes the desired metric |
| Precision-recall tradeoff | "You cannot have both" | Lowering the threshold catches more positives (higher recall) but also flags more false positives (lower precision), and vice versa |
| AUPRC | "Area under the PR curve" | Summarizes the precision-recall curve into a single number; more informative than AUC-ROC when classes are heavily imbalanced |
| Matthews Correlation Coefficient | "The balanced metric" | A correlation between predicted and actual labels that produces a high score only when the model performs well on both classes |
| Cost-sensitive learning | "Different mistakes cost different amounts" | Incorporating real-world misclassification costs into the training objective so the model optimizes for total cost, not error count |
| Random oversampling | "Duplicate the minority" | Repeating minority class samples to balance class counts; simple but risks overfitting to duplicated points |

## المزيد من القراءة

- [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813)-- ورقة SMOTE الأصلية، ما زالت العمل الأكثر إشارة إلى التعلم غير المتوازن
- [Learning from Imbalanced Data (He & Garcia, 2009)](https://ieeexplore.ieee.org/document/5128907)-- مسح شامل يغطي أخذ العينات، والنهج الحساسة للتكلفة، والخوارزمية
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)-- مكتبة Python مع فوارق SMOTE، استراتيجيات تقليل العينات، وتكامل خط الأنابيب
- [The Precision-Recall Plot Is More Informative than the ROC Plot (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)-- متى ولماذا تفضل منحنى العلاقات العامة على منحنى ROC لمشاكل عدم التوازن
