# العمليات الستوكاستية

> العشوائية مع الهيكل، والرياضيات وراء المشي العشوائي، سلسلة ماركوف، ونماذج التوزيع.

**Type:** Learn
**Language:**بايثون
**Prerequisites:** Phase 1, Lessons 06-07 (probability, Bayes)
**Time:** ~75 minutes

## أهداف التعلم

- محاكاة المشي عشوائي 1D و 2D والتحقق من مقياس الترحيل
- بناء محاكاة سلسلة ماركوف وحساب توزيعها الثابت عن طريق التكوين الخاص
- تنفيذ ديناميكيات ميتروبوليس-هاستينغز MCMC و لانجفين للاستعراض من توزيعات المستهدفة
- ربط عملية التوزيع الأمامي إلى حركة براونية وشرح كيفية توليد البيانات من خلال العملية العكسية

## المشكلة

العديد من أنظمة الذكاء الاصطناعي تتضمن عشوائية تتطور مع مرور الوقت، وليس عشوائية ثابتة -- عشوائية مهيكلة، تسلسلية حيث تعتمد كل خطوة على ما حدث من قبل.

النماذج اللغوية تولد رموز واحدة في كل مرة. كل رموز تعتمد على السياق السابق. النموذج يخرج توزيع الاحتمالات، ومعينة من ذلك، والتحرك على. وهذا هو عملية استوكاستية.

تضيف نماذج التنبع الضوضاء إلى صورة خطوة بخطوة حتى تصبح ثابتة خالصة. ثم يعكسون العملية ، ويقومون بتخفيض خطوة بخطوة حتى تظهر صورة جديدة. العملية الأمامية هي سلسلة ماركوف. العملية العكسية هي سلسلة ماركوف تعلمت تتراجع.

وكلاء التعلم التعزيزية يقومون بعمل في بيئة. كل عمل يؤدي إلى حالة جديدة مع بعض الاحتمالات. العميل يتبع سياسة عشوائية في عالم عشوائي. كل شيء هو عملية قرار ماركوف.

أخذ العينات من MCMC -- العمود الفقري للإستنتاج البايسي -- يبن سلسلة ماركوف التي توزعها الثابت هو الجزء الخلفي الذي تريد أن تأخذ العينات منه.

كل هذه البنائى على أربع أفكار أساسية:
1. المشي عشوائية -- أبسط عملية استوكاستية
2. سلسلة ماركوف -- الاختيار المُهيّن مع ماتريكس انتقالية
3. ديناميكية لانجفين -- انخفاض التدرج مع الضوضاء
4. "ميتروبوليس-هستنغز" -- أخذ عينات من أي توزيع

## المفهوم

### المشي العشوائي

البدء من الموقف 0، في كل خطوة، ضعي عملة عادلة. الرؤوس: تحرك يمينا (+1) والذيل: تحرك يسارا (-1).

بعد n خطوات، موقفك هو مجموع n قيم عشوائية +/-1. الموقع المتوقع هو 0 (المشي غير متحيز). ولكن المسافة المتوقعة من المنشأ ينمو ك مربع ((n).

هذا غير بديهي. المشي عادلا - لا تجرف في أي اتجاه. ولكن مع مرور الوقت، يتجول أبعد وأكثر من حيث بدأ. الانحراف القياسي بعد n خطوات هو مربع ((n).

```
Step 0:  Position = 0
Step 1:  Position = +1 or -1
Step 2:  Position = +2, 0, or -2
...
Step 100: Expected distance from origin ~ 10 (sqrt(100))
Step 10000: Expected distance from origin ~ 100 (sqrt(10000))
```

**In 2D**يتنقل المشي للأعلى أو إلى الأسفل أو إلى اليسار أو إلى اليمين بنفس الاحتمال. نفس المقياس مربع ((n) ينطبق على المسافة من المنشأ. يتتبع المسار نمطًا يشبه الكسور.

**Why sqrt(n)?**كل خطوة هي +1 أو -1 مع احتمال متساو. بعد n خطوات، الموقف S_n = X_1 + X_2 + ... + X_n حيث كل X_i هو +/-1. تغير كل خطوة هو 1, والخطوات مستقلة، لذلك Var(S_n) = n. الانحراف القياسي = sqrt(n. بواسطة نظرية الحد المركزي، S_n / sqrt(n) يتقارب إلى توزيع طبيعي قياسي.

هذا مقياس مربع ((n) يظهر في كل مكان في ML. مقياس SGD الضجيج على أنها 1/sqrt(حجم البطاقة. إدراج مقياس الأبعاد على أنها sqrt(d). الجذر التربيعي هو توقيع إضافات عشوائية مستقلة.

**Connection to Brownian motion.**خذ مشاة عشوائية بحجم الخطوة 1/sqrt(n) و n خطوات لكل وحدة من الوقت. مع وصول n إلى اللانهاية، يتقارب المشي إلى حركة براونية B(t) - عملية متواصلة في الوقت حيث يتم توزيع B(t) عادة مع متوسط 0 والفرق t.

حركة براون هي الأساس الرياضي للتشويق. إنها تمثّل الارتجاج العشوائي للجسيمات في السائل، وتقلبات أسعار الأسهم، و -الأهم من ذلك- عملية الضوضاء في نماذج التشويق.

**Gambler's ruin.**المشي عشوائي يبدأ من وضع k، مع حاجز امتصاص في 0 و N. ما هي احتمالات الوصول إلى N قبل 0؟ للمشي العادل: P(وصول N) = k/N. هذا بسيط ورائح بشكل مفاجئ. يرتبط بنظرية المارتينغال - المشي العادي العادل هو المارتينغال (القدر المستقبلي المتوقع = القيمة الحالية).

### سلاسل ماركوف

سلسلة ماركوف هي نظام ينتقل بين الحالات وفقاً للاحتمالات الثابتة. الملكية الرئيسية: الحالة التالية تعتمد فقط على الحالة الحالية ، وليس على التاريخ.

```
P(X_{t+1} = j | X_t = i, X_{t-1} = ...) = P(X_{t+1} = j | X_t = i)
```

هذا هو خاصية ماركوف. وهذا يعني أنه يمكنك وصف الديناميكيا بأكملها مع المصفوفة الانتقالية P:

```
P[i][j] = probability of going from state i to state j
```

كل صف من P يصل إلى 1 (يجب أن تذهب إلى مكان ما).

**Example -- Weather:**

```
States: Sunny (0), Rainy (1), Cloudy (2)

P = [[0.7, 0.1, 0.2],    (if sunny: 70% sunny, 10% rainy, 20% cloudy)
     [0.3, 0.4, 0.3],    (if rainy: 30% sunny, 40% rainy, 30% cloudy)
     [0.4, 0.2, 0.4]]    (if cloudy: 40% sunny, 20% rainy, 40% cloudy)
```

تبدأ في أي حالة. بعد العديد من الانتقالات، تقسيم الحالات يتقارب إلى التوزيع الثابت pi، حيث pi * P = pi. هذا هو المتجه الخليوي الخاص بـ P مع القيمة الخليوية 1.

بالنسبة لسلسلة الطقس، توزيعها ثابت هو [0.55, 0.18, 0.27] -- على المدى الطويل، هو مشمس 55% من الوقت بغض النظر عن الحالة البداية.

```mermaid
graph LR
    S["Sunny"] -->|0.7| S
    S -->|0.1| R["Rainy"]
    S -->|0.2| C["Cloudy"]
    R -->|0.3| S
    R -->|0.4| R
    R -->|0.3| C
    C -->|0.4| S
    C -->|0.2| R
    C -->|0.4| C
```

**Computing the stationary distribution.**هناك طريقتان:

1. **Power method**: ضرب أي توزيع أولي ب P مراراً وتكراراً. بعد تكرارات كافية، يتقارب.
2. **Eigenvalue method**: العثور على الجهاز المباشر اليسرى لـ P مع القيمة المباشرة 1. هذا هو الجهاز المباشر لـ P^T مع القيمة المباشرة 1.

كل منهج يتطلب أن تلبي سلسلة ظروف التقارب.

**Convergence conditions.**تتقارب سلسلة ماركوف إلى توزيع ثابت فريد إذا كانت:
- **Irreducible**: كل دولة يمكن الوصول إليها من كل دولة أخرى
- **Aperiodic**: لا تتدفق السلسلة مع فترة ثابتة

معظم السلاسل التي تواجهها في ML تلبي كلا الشرطين

**Absorbing states.**حالة استيعاب إذا كنت مرة واحدة تدخلها، لا تترك أبدا (P[i][i] = 1). استيعاب سلسلة ماركوف نموذج العمليات مع الحالات النهائية -- لعبة التي تنتهي، العميل الذي يزعج، تسلسل رمزية التي تضرب رمز نهاية النص.

**Mixing time.**كم خطوة حتى تكون السلسلة "قريبة" من التوزيع الثابت؟ بصورة رسمية، عدد الخطوات حتى تتراجع المسافة الإجمالية للتباين من الثبات إلى أقل من عتبة ما. الاختلاط السريع = عدد قليل من الخطوات المطلوبة. الفجوة الطيفية لـ P (1 ناقص الثاني أكبر قيمة خاصة) تحكم وقت الاختلاط. الفجوة الأكبر = الاختلاط السريع.

### الاتصال مع نماذج اللغة

إن توليد الرمز في نموذج لغة هو عملية ماركوف تقريبًا. بالنظر إلى السياق الحالي، يقوم النموذج بتوزيع على الرمز التالي. تتحكم درجة الحرارة في الحد:

```
P(token_i) = exp(logit_i / temperature) / sum(exp(logit_j / temperature))
```

- درجة الحرارة = 1.0: توزيع قياسي
- درجة الحرارة < 1.0: أكثر حدة (أكثر تحديدية)
- درجة الحرارة > 1.0: مسطحة (أكثر عشوائية)
- درجة الحرارة -> 0: argmax (طموح)

يختصر أخذ العينات من أعلى (k) إلى رموز ذات الاحتمال الأكبر. يختصر أخذ العينات من أعلى (p) إلى أصغر مجموعة من رموز تتجاوز احتمالها التراكمي p. كلتا الحالتين تعدل احتمالات انتقال ماركوف.

### الحركة البروونية

الحد المتواصل في الوقت للمشي عشوائي. الموقف B(t) لديه ثلاث خصائص:
1. B(0) = 0
2. B(t) - B(s) يتم توزيعها عادة مع متوسط 0 و t - s المتباين (ل t > s)
3. الزيادات على فترات غير متداخلة مستقلة

حركة براونية مستمرة ولكن لا يمكن التمييز فيها في أي مكان - إنها تتحرك في كل مقياس. المسار لديه بعد كسر 2 في المستوى.

في محاكاة منفصلة، تقترب من حركة براونية من خلال:

```
B(t + dt) = B(t) + sqrt(dt) * z,    where z ~ N(0, 1)
```

إن مقياس sqrt ((dt) مهم. يأتي من نظرية الحد المركزي المطبقة على المشيات العشوائية.

### ديناميكيات لانجفين

يجد التنزل التدريجي الحد الأدنى من وظيفة. ديناميكا لانجفين يجد توزيع الاحتمالات متناسبة مع exp(-U(x) / T) ، حيث U هو وظيفة الطاقة و T هو درجة الحرارة.

```
x_{t+1} = x_t - dt * gradient(U(x_t)) + sqrt(2 * T * dt) * z_t
```

قوى اثنتان تعمل على الجسيمة:
1. **Gradient force**(-dt * تراجع ((U)): يدفع نحو طاقة منخفضة (مثل تراجع تراجع)
2. **Random force**(sqrt(2*T*dt) * z): يدفع في اتجاهات عشوائية (التنقيب)

عند درجة حرارة T = 0 ، هذا هو انخفاض تراجع خالص. عند درجة حرارة عالية ، فإنه تقريباً مشي عشوائيًا. عند درجة حرارة مناسبة ، تقوم الجسيم باستكشاف المشهد الطياري ويقضي المزيد من الوقت في مناطق منخفضة الطاقة.

**Connection to diffusion models.**عملية التقدم في نموذج التوزيع هي:

```
x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * noise
```

هذه سلسلة ماركوف التي تضمّن البيانات ببطء مع الضجيج بعد خطوات كافية، x_T هو ضجيج غوسيان خالص.

العملية العكسية -- من الضجيج إلى البيانات -- هي سلسلة ماركوف أيضاً، لكن احتمالات انتقالها يتم تعلمها من قبل شبكة عصبية. تتعلم الشبكة التنبؤ بالضجيج الذي تم إضافة كل خطوة، ثم تقوم بإسقاطه.

```mermaid
graph LR
    subgraph "Forward Process (add noise)"
        X0["x_0 (data)"] -->|"+ noise"| X1["x_1"]
        X1 -->|"+ noise"| X2["x_2"]
        X2 -->|"..."| XT["x_T (pure noise)"]
    end
    subgraph "Reverse Process (denoise)"
        XT2["x_T (noise)"] -->|"neural net"| XR2["x_{T-1}"]
        XR2 -->|"neural net"| XR1["x_{T-2}"]
        XR1 -->|"..."| XR0["x_0 (generated data)"]
    end
```

### MCMC: سلسلة ماركوف مونتي كارلو

في بعض الأحيان تحتاج إلى أخذ عينة من التوزيع p ((x) التي يمكنك تقييمها (حتى ثابت) ولكن لا يمكن أن تأخذ عينة من مباشرة.

**Metropolis-Hastings**يُبني سلسلة ماركوف التي يبلغ توزيعها الثابت p(x):

1. ابدأ من موقع ما x
2. اقترح موقف جديد x' من تقسيم الاقتراح Q(x'
3. نسبة قبول الحساب: a(x') * Q(x من الممكن أن يكون) / (p(x) * Q(x من الممكن أن يكون))
4. اقبل x' مع احتمالات min ((1، a) وإلا تبقى في x.
5. أكرر

إذا كان Q متماثلًا على سبيل المثال ، Q(x' (يخ) = Q(x (يخ) = N(x ، sigma^2)) ، فإن النسبة تبسيط إلى a = p(x') / p(x. تحتاج فقط إلى نسبة الاحتمالات - استمرار التطبيع إلغاء.

من المضمون أن السلسلة تتحرك إلى p ((x) في ظروف خفيفة. ولكن التحرك يمكن أن يكون بطيئًا إذا كان الاقتراح صغيرًا جدًا (مشي عشوائي) أو كبيرًا جدًا (رفض مرتفع).

**Why it works.**يضمن نسبة القبول التوازن التفصيلي: احتمال وجود في x والانتقال إلى x' يساوي احتمال وجود في x' والانتقال إلى x. التوازن التفصيلي يعني أن p(x) هو التوزيع الثابت للسلسلة. لذلك بعد خطوات كافية، تأتي العينات من p(x.

**Practical considerations:**
- **Burn-in**: يرمي أول عينات N. تحتاج السلسلة إلى وقت للوصول إلى التوزيع الثابت من نقطة البداية.
- **Thinning**: احتفظ بكل عينة ك- ثمة لتقليل التنسيق الذاتي.
- **Multiple chains**: تشغيل سلسلة متعددة من نقاط بداية مختلفة. إذا تجتمع إلى نفس التوزيع، لديك دليل على التقارب.
- **Acceptance rate**: بالنسبة لمقترحات غوس في d الأبعاد، فإن معدل قبول المثالي هو حوالي 23% (روبرتس وروسنتال، 2001) ؛ ارتفاع كبير يعني أن السلسلة بالكاد تتحرك. انخفاض كبير يعني أنها ترفض كل شيء.

### العمليات الاستوكاستية في الذكاء الاصطناعي

| Process | AI Application |
|---------|---------------|
| Random walk | Exploration in RL, Node2Vec embeddings |
| Markov chain | Text generation, MCMC sampling |
| Brownian motion | Diffusion models (forward process) |
| Langevin dynamics | Score-based generative models, SGLD |
| Markov decision process | Reinforcement learning |
| Metropolis-Hastings | Bayesian inference, posterior sampling |

```figure
random-walk-diffusion
```

## بناءها

### الخطوة الأولى: محاكاة المشي عشوائية

```python
import numpy as np

def random_walk_1d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    steps = rng.choice([-1, 1], size=n_steps)
    positions = np.concatenate([[0], np.cumsum(steps)])
    return positions


def random_walk_2d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    directions = rng.choice(4, size=n_steps)
    dx = np.zeros(n_steps)
    dy = np.zeros(n_steps)
    dx[directions == 0] = 1   # right
    dx[directions == 1] = -1  # left
    dy[directions == 2] = 1   # up
    dy[directions == 3] = -1  # down
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])
    return x, y
```

يحتفظ المشي 1D بمجموعات تجميعية. كل خطوة هي +1 أو -1. بعد n خطوات، يكون الموقف هو المجموع. ينمو التباين خطيا مع n، لذلك ينمو الانحراف القياسي ك مربع ((n).

### الخطوة الثانية: سلسلة ماركوف

```python
class MarkovChain:
    def __init__(self, transition_matrix, state_names=None):
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = len(self.P)
        self.state_names = state_names or [str(i) for i in range(self.n_states)]

    def step(self, current_state, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        probs = self.P[current_state]
        return rng.choice(self.n_states, p=probs)

    def simulate(self, start_state, n_steps, seed=None):
        rng = np.random.RandomState(seed)
        states = [start_state]
        current = start_state
        for _ in range(n_steps):
            current = self.step(current, rng)
            states.append(current)
        return states

    def stationary_distribution(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return np.abs(stationary)
```

التوزيع الثابت هو المتجه الخليوي لـ P مع القيمة الخليوية 1. نجده عن طريق حساب المتجه الخليوي لـ P^T (تحويل المتجه الخليوي إلى متجه الخليوي الأيمن).

### الخطوة الثالثة: ديناميكيات لانجفين

```python
def langevin_dynamics(grad_U, x0, dt, temperature, n_steps, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        noise = rng.randn(*x.shape)
        x = x - dt * grad_U(x) + np.sqrt(2 * temperature * dt) * noise
        trajectory.append(x.copy())
    return np.array(trajectory)
```

يضغط التدرج x نحو طاقة منخفضة. الضجيج يمنعه من التعلق. عند التوازن، توزيع العينات متناسبة مع exp ((-U ((x) / درجة حرارة).

### الخطوة الرابعة: "ميتروبوليس"

```python
def metropolis_hastings(target_log_prob, proposal_std, x0, n_samples, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    samples = [x.copy()]
    accepted = 0
    for _ in range(n_samples - 1):
        x_proposed = x + rng.randn(*x.shape) * proposal_std
        log_ratio = target_log_prob(x_proposed) - target_log_prob(x)
        if np.log(rng.rand()) < log_ratio:
            x = x_proposed
            accepted += 1
        samples.append(x.copy())
    acceptance_rate = accepted / (n_samples - 1)
    return np.array(samples), acceptance_rate
```

يقدم الخوارزمية نقطة جديدة ، ويختبر ما إذا كانت لديها احتمال أكبر (أو تقبل مع احتمال متناسب مع النسبة) ، ويكرر. يجب أن يكون معدل القبول حوالي 23-50٪ للتلاعب الجيد.

## استخدمها

في الممارسة العملية، تستخدم المكتبات القائمة لهذه الخوارزميات ولكن فهم الميكانيكا مهم في التحليل والتحديد.

```python
import numpy as np

rng = np.random.RandomState(42)
walk = np.cumsum(rng.choice([-1, 1], size=10000))
print(f"Final position: {walk[-1]}")
print(f"Expected distance: {np.sqrt(10000):.1f}")
print(f"Actual distance: {abs(walk[-1])}")
```

### النمبي للمصفوفات الانتقالية

```python
import numpy as np

P = np.array([[0.7, 0.1, 0.2],
              [0.3, 0.4, 0.3],
              [0.4, 0.2, 0.4]])

distribution = np.array([1.0, 0.0, 0.0])
for _ in range(100):
    distribution = distribution @ P

print(f"Stationary distribution: {np.round(distribution, 4)}")
```

مضاعفة التوزيع الأولي ب P مراراً وتكراراً. بعد تكرارات كافية، يتقارب إلى التوزيع الثابت بغض النظر عن المكان الذي بدأت فيه. هذه هي طريقة الطاقة للعثور على المتجه المسيطر اليساري.

### الروابط إلى الإطار الحقيقي

- **PyTorch diffusion:**- نعم`DDPMScheduler`في العناق الوجه`diffusers`يطبق سلسلة ماركوف الأمامية والعكسية
- **NumPyro / PyMC:**استخدام MCMC (مُعينة NUTS، التي تحسن على Metropolis-Hastings) للإستنتاج البايسي
- **Gymnasium (RL):**وظيفة خطوة البيئة تعريف عملية قرار ماركوف

### التحقق من التقارب في سلسلة ماركوف

```python
import numpy as np

P = np.array([[0.9, 0.1], [0.3, 0.7]])

eigenvalues = np.linalg.eigvals(P)
spectral_gap = 1 - sorted(np.abs(eigenvalues))[-2]
print(f"Eigenvalues: {eigenvalues}")
print(f"Spectral gap: {spectral_gap:.4f}")
print(f"Approximate mixing time: {1/spectral_gap:.1f} steps")
```

الفجوة الطيفية تخبرك كم السلسلة تنسى حالتها الأولية بسرعة. فجوة 0.2 تعني حوالي 5 خطوات للتخليط. فجوة 0.01 تعني حوالي 100 خطوة. تحقق دائما من هذا قبل تشغيل محاكاة طويلة -- حساب مخلوطة سلسلة نفايات ببطء.

## أرسله

هذا الدرس ينتج عن:
- `outputs/prompt-stochastic-process-advisor.md`-- عرض يساعد على تحديد إطار العملية الاستوكاستية المطبقة على مشكلة معينة

## العلاقات

| Concept | Where it shows up |
|---------|------------------|
| Random walk | Node2Vec graph embeddings, exploration in RL |
| Markov chain | Token generation in LLMs, MCMC sampling |
| Brownian motion | Forward diffusion process in DDPM, SDE-based models |
| Langevin dynamics | Score-based generative models, stochastic gradient Langevin dynamics (SGLD) |
| Stationary distribution | MCMC convergence target, PageRank |
| Metropolis-Hastings | Bayesian posterior sampling, simulated annealing |
| Temperature | LLM sampling, Boltzmann exploration in RL, simulated annealing |
| Mixing time | Convergence speed of MCMC, spectral gap analysis |
| Absorbing state | End-of-sequence token, terminal states in RL |
| Detailed balance | Correctness guarantee for MCMC samplers |

تستحق نماذج الانتشار اهتمامًا خاصًا. تعرّف DDPM (Ho et al., 2020) سلسلة ماركوف الأمامية:

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)
```

حيث beta_t هو جدول ضجيج. بعد خطوات T، x_T حوالي N(0, I). يتم تعريف العملية العكسية بواسطة شبكة عصبية تتوقع الضجيج:

```
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
```

كل خطوة من التوليد هي خطوة في سلسلة ماركوف تعلم. فهم سلسلة ماركوف يعني فهم كيف ولماذا نموذج التوزيع تولد البيانات.

SGLD (حركة Langevin المتحركة من حيث التدريج المتحرك) تجمع بين انخفاض التدريج المحدد مع ضجيج Langevin. بدلاً من حساب التراجع الكامل، تستخدم تقديرًا استوكاستيكيًا وتضيف ضجيجًا مقياسيًا. مع تراجع معدل التعلم، فإن SGLD تنتقل من التحسين إلى أخذ العينات -- تحصل على عينات بايزية خلفية تقريبا مجانا. هذه واحدة من أبسط الطرق للحصول على تقديرات عدم اليقين من شبكة عصبية.

المفهوم الرئيسي عبر كل هذه العلاقات: العمليات الاستوكاسية ليست أدوات نظرية فقط. إنها آليات الحساب داخل أنظمة الذكاء الاصطناعي الحديثة. عندما تقوم بتحديد درجة حرارة ماجستير في العلوم، أنت تقوم بتعديل سلسلة ماركوف. عندما تدرب نموذج التوزيع، تتعلمين كيفية عكس عملية مثل حركة براون. عندما تقوم بإجراء استنتاج بايسي، تقوم ببناء سلسلة تتقارب إلى الخلفية.

## التمارين

1. **Simulate 1000 random walks of 10000 steps.**رسم توزيع المواقع النهائية. التحقق من أنه تقريبا غوسيان مع متوسط 0 والانحراف القياسي sqrt(10000) = 100.

2. **Build a text generator using a Markov chain.**تدريب على مجموعة صغيرة: لكل كلمة، عد الانتقالات إلى الكلمة التالية. بناء ماتريكية الانتقال. توليد جمل جديدة عن طريق أخذ عينات من السلسلة.

3. **Implement simulated annealing**استخدام متروبوليس-هستينغز. تبدأ عند درجة حرارة عالية (تقبل كل شيء تقريبا) وتبرد تدريجيا (تقبل تحسينات فقط). استخدمها للعثور على الحد الأدنى من وظيفة مع العديد من الحد الأدنى المحلية.

4. **Compare Langevin dynamics at different temperatures.**عينة من إمكانات البئر المزدوجة U(x) = (x^2 - 1)^2. عند درجة حرارة منخفضة، تجمع العينات في بئر واحد. عند درجة حرارة عالية، تنتشر على كليهما. حدد درجة الحرارة الحرجة حيث تتلاشى السلسلة بين البئر.

5. **Implement the forward diffusion process.**ابدأ بإشارة 1D (مثل موجة الصدري). أضف الضجيج تدريجياً على مدى 100 خطوة مع جدول ضجيج خطي. أظهر كيف تتدهور الإشارة إلى ضجيج نقي. ثم قم بتنفيذ مؤشر بسيط يعكس العملية (حتى واحد ساذج يخص فقط الضجيج المقدر).

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Random walk | "Coin-flip movement" | A process where position changes by random increments at each step |
| Markov property | "Memoryless" | The future depends only on the present state, not on the history |
| Transition matrix | "The probability table" | P[i][j] = probability of moving from state i to state j |
| Stationary distribution | "The long-run average" | The distribution pi where pi*P = pi -- the chain's equilibrium |
| Brownian motion | "Random jiggling" | The continuous-time limit of a random walk, B(t) ~ N(0, t) |
| Langevin dynamics | "Gradient descent with noise" | Update rule that combines deterministic gradient and random perturbation |
| MCMC | "Walking toward the target" | Constructing a Markov chain whose stationary distribution is the one you want |
| Metropolis-Hastings | "Propose and accept/reject" | MCMC algorithm that uses acceptance ratios to ensure convergence |
| Temperature | "The randomness knob" | Parameter controlling the tradeoff between exploration and exploitation |
| Diffusion process | "Noise in, noise out" | Forward: gradually add noise. Reverse: gradually remove it. Generates data. |

## المزيد من القراءة

- **Ho, Jain, Abbeel (2020)**-- "إرفض نماذج احتمالية التوزيع". ورقة DDPM التي أطلقت ثورة نموذج التوزيع. مشتق واضح من سلسلة ماركوف الأمامية والعكسية.
- **Song & Ermon (2019)**-- "المؤثرات التجاريّة عن طريق تقدير درجات توزيع البيانات". النهج القائم على النتيجة باستخدام ديناميكيات لنجفين للاستعراض.
- **Roberts & Rosenthal (2004)**-- "سلسلة ماركوف العامة في الفضاء والخوارزميات MCMC". النظرية وراء متى و لماذا يعمل MCMC.
- **Norris (1997)**-- "سلسلة ماركوف" -- كتاب دراسي قياسي يغطي التقارب، التوزيعات الثابتة، وأوقات الضربة
- **Welling & Teh (2011)**-- "التعلم البايسي عن طريق ديناميكة لانجفين المتحركة". يجمع بين SGD و ديناميكة لانجفين
