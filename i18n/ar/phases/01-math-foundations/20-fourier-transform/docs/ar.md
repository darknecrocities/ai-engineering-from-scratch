# تحويل فوريه

> كل إشارة هي مجموع موجات الصين، تحويل فوريير يخبرك عن أي موجات

**Type:** Build
**Language:**بايثون
**Prerequisites:** Phase 1, Lessons 01-04, 19 (complex numbers)
**Time:** ~90 minutes

## أهداف التعلم

- تنفيذ DFT من الصفر وتحقق منه ضد O(N سجل N) Cooley-Tukey FFT
- تفسير معايير التردد: استخراج الطول والمرحلة ومتعدد الطاقة من الإشارة
- تطبيق نظرية التخزين لتنفيذ التخزين عبر مضاعفة FFT
- ربط تدمير تردد فوريير إلى ترنسفورمات التشفير الموضعي وطبقات التخزين CNN

## المشكلة

تسجيل الصوت هو سلسلة من قياسات الضغط على مر الزمن. سعر الأسهم هو سلسلة من القيم على مدى أيام. صورة هي شبكة من كثافة البيكسل على الفضاء. كل هذه هي بيانات في مجال الزمن (أو مجال الفضاء). ترى القيم تتغير على بعض المؤشر.

ولكن العديد من الأنماط غير مرئية في مجال الزمن. هل هذه الإشارة الصوتية هي نغمة نقية أو صفات؟ هل سعر الأسهم هذا له دورة أسبوعية؟ هل هذه الصورة لديها نسيج متكرر؟ هذه الأسئلة تتعلق بمحتويات التردد، ويمتلك مجال الزمن ذلك.

تحويل فوريه تحويل البيانات من مجال الوقت إلى مجال الترددات. يأخذ إشارة ويفسدها في موجات السينوس من ترددات مختلفة. كل موجة السينوس لديها amplitude (كم قوية هي) ومرحلة (من حيث تبدأ). تحويل فوريه يخبرك كليهما.

هذا مهم بالنسبة لـ ML لأن التفكير في مجال التردد يظهر في كل مكان. تقوم الشبكات العصبية المتحركة بإجراء التحويل، وهو مضاعفة في مجال التردد. تُستخدم مُشفّرات المُحوّل المُوضّع التفكّر المتردد لتمثيل المُوضّع. نماذج الصوت (تعرف الكلام، توليد الموسيقى) تعمل على الطيفيات -- تمثيلات تردد للصوت. نماذج سلسلة الزمن تبحث عن أنماط دورية. فهم تحويل فوريير يعطيك المفردات للعمل مع كل هذه.

## المفهوم

### تعريف DFT

في ظل عينات N x[0] ، x[1], ..., x[N-1] ، ينتج تحويل فوريه المميز معايير تردد N X[0] ، X[1], ..., X[N-1]:

```
X[k] = sum_{n=0}^{N-1} x[n] * e^(-2*pi*i*k*n/N)

for k = 0, 1, ..., N-1
```

كل X[k] هو عدد معقد. حجمها \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \

المعلومات الرئيسية:`e^(-2*pi*i*k*n/N)`هو فازور متناوب في تردد k. يقوم DFT بحساب التواصل بين الإشارة وكل من ترددات N المتساوية المساحة. إذا كان الإشارة تحتوي على طاقة في تردد k ، فإن التواصل كبير. وإذا لم يكن ، فإنه قريب من الصفر.

### ما تعنيه كل معدل

**X[0]: the DC component.**هذا هو مجموع جميع العينات -- متناسبة مع المتوسط. وهو يمثل استمرار (الصفر التردد) التداخل من الإشارة.

```
X[0] = sum_{n=0}^{N-1} x[n] * e^0 = sum of all samples
```

**X[k] for 1 <= k <= N/2: positive frequencies.**يمثل X[k] دورات التردد k لكل عينات N. تعني k أعلى تردد أعلى (التذبذب أسرع).

**X[N/2]: the Nyquist frequency.**أعلى تردد يمكنك أن تمثله مع عينات N. فوق هذا، تحصل على الاسم الألي -- ترددات عالية تتمثل في أقل.

**X[k] for N/2 < k < N: negative frequencies.**بالنسبة للإشارات ذات القيمة الحقيقية، X[N-k] = conj(X[k] . الترددات السلبية هي صور مرآة للاتجاهات الإيجابية. لهذا السبب تكون المعلومات المفيدة في معايير N/2 + 1 الأولى.

### الـ DFT العكس

يقوم DFT العكسي بإعادة تشكيل الإشارة الأصلية من معايير ترددها:

```
x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * e^(2*pi*i*k*n/N)

for n = 0, 1, ..., N-1
```

الفرق الوحيد عن DFT الأمامي: علامة في المضرب إيجابية (ليس سلبية) ، وهناك عامل طبيعة 1/N.

DFT العكسية هي إعادة بناء مثالية. لا تفقد أي معلومات. يمكنك الذهاب من نطاق الوقت إلى نطاق التردد والعودة دون أي خطأ. DFT هو تغيير الأساس -- يعيد التعبير عن نفس المعلومات في نظام إحداثيات مختلفة.

### الـ (ف.إف.تي): جعل الأمر سريعاً

DFT كما هو محدد أعلاه هو O(N^2): لكل من معايير الخروج N، يمكنك جمع على عينات المدخل N. بالنسبة N = 1 مليون، وهذا هو 10^12 العمليات.

تحسب تحويل فوريه السريع نفس النتيجة في O  N log N. بالنسبة إلى N = 1 مليون، وهذا حوالي 20 مليون عملية بدلاً من تريليون. هذا هو ما يجعل تحليل التردد عمليًا.

يعمل خوارزمية كولي-توكي (أكثر FFT شيوعًا) عن طريق القسم والغزو:

1. تقسيم الإشارة إلى عينات متصفحة مساوية وممتصفحة غير مساوية.
2. احسب DFT لكل نصف بشكل متكرر.
3. قم بتجميع اثنين من DFTs نصف الحجم باستخدام "عاملات الدوما" e^(-2*pi*i*k/N).

```
X[k] = E[k] + e^(-2*pi*i*k/N) * O[k]          for k = 0, ..., N/2 - 1
X[k + N/2] = E[k] - e^(-2*pi*i*k/N) * O[k]    for k = 0, ..., N/2 - 1

where E = DFT of even-indexed samples
      O = DFT of odd-indexed samples
```

التناظر يعني أن كل مستوى من التكرار يعمل O(N) ، وهناك مستويات log2(N. إجمالي: O(N log N).

```mermaid
graph TD
    subgraph "8-point FFT (Cooley-Tukey)"
        X["x[0..7]<br/>8 samples"] -->|"split even/odd"| E["Even: x[0,2,4,6]"]
        X -->|"split even/odd"| O["Odd: x[1,3,5,7]"]
        E -->|"4-pt FFT"| EK["E[0..3]"]
        O -->|"4-pt FFT"| OK["O[0..3]"]
        EK -->|"combine with twiddle factors"| XK["X[0..7]"]
        OK -->|"combine with twiddle factors"| XK
    end
    subgraph "Complexity"
        C1["DFT: O(N^2) = 64 multiplications"]
        C2["FFT: O(N log N) = 24 multiplications"]
    end
```

يتطلب FFT أن يكون طول الإشارة قوة 2. في الممارسة العملية ، يتم إضافة الإشارات إلى الصفر إلى قوة 2.

### تحليل الطيف

- نعم**power spectrum**هو X [k]^2 -- الكبيرة المربعة لكل معدل تردد.

- نعم**phase spectrum**هو الزاوية ((X[k]) -- تعويض مرحلة لكل تردد. بالنسبة لمعظم مهام التحليل، تهتمين بمتصفح الطاقة وتجاهلون المرحلة.

```
Power at frequency k:  P[k] = |X[k]|^2 = X[k].real^2 + X[k].imag^2
Phase at frequency k:  phi[k] = atan2(X[k].imag, X[k].real)
```

### تحديد التردد

يعتمد قرار تردد DFT على عدد العينات N ومعدل أخذ العينات fs.

```
Frequency of bin k:      f_k = k * fs / N
Frequency resolution:    delta_f = fs / N
Maximum frequency:       f_max = fs / 2  (Nyquist)
```

لتحديد ترددات قريبة من بعضها البعض، تحتاج إلى المزيد من العينات. لالتقاط ترددات عالية، تحتاج إلى معدل أعلى من أخذ العينات.

### نظرية التخزين

هذه واحدة من أهم النتائج في معالجة الإشارات و ذات صلة مباشرة مع قنوات سي إن إن

**Convolution in the time domain equals pointwise multiplication in the frequency domain.**

```
x * h = IFFT(FFT(x) . FFT(h))

where * is convolution and . is element-wise multiplication
```

لماذا هذا مهم:

- التخطي المباشر لـ إشارات طول N و M يأخذ عمليات O(N*M).
- التخزين القائم على FFT يأخذ O(N log N): تحويل كل منهما، وتضاعف، وتحويل مرة أخرى.
- بالنسبة للجوهرات الكبيرة، فإن تحويل FFT أسرع بشكل كبير.
- هذا بالضبط ما يحدث في الطبقات الملتوية مع مجالات استقبلية كبيرة.

ملاحظة: يقوم DFT بحساب التخزين الدائري (تتتلف الإشارة حولها). بالنسبة للتخزين الخطي (لا توجد تغطية) ، قم بتحديد كل من الإشارات إلى طول N + M - 1 قبل الحساب.

```mermaid
graph LR
    subgraph "Time Domain"
        TA["Signal x[n]"] -->|"convolve (slow: O(NM))"| TC["Output y[n]"]
        TB["Filter h[n]"] -->|"convolve"| TC
    end
    subgraph "Frequency Domain"
        FA["FFT(x)"] -->|"multiply (fast: O(N))"| FC["FFT(x) * FFT(h)"]
        FB["FFT(h)"] -->|"multiply"| FC
        FC -->|"IFFT"| FD["y[n]"]
    end
    TA -.->|"FFT"| FA
    TB -.->|"FFT"| FB
    FD -.->|"same result"| TC
```

### النوافذ

يفترض DFT أن الإشارة دورية - يعامل عينات N كفترة واحدة من إشارة تكرارية لا نهاية لها. إذا لم تبدأ الإشارة وتنتهي في نفس القيمة، فإنه يخلق عدم استمرارية في الحدود، والتي تظهر كمحتوى عالية التردد وهمية. وهذا يسمى تسرب الطيف.

يقلل التسرب من النافذة عن طريق تقليل الإشارة إلى الصفر في كلا الطرفين قبل حساب DFT.

النوافذ المشتركة:

| Window | Shape | Main lobe width | Side lobe level | Use case |
|--------|-------|----------------|-----------------|----------|
| Rectangular | Flat (no window) | Narrowest | Highest (-13 dB) | When signal is exactly periodic in N samples |
| Hann | Raised cosine | Moderate | Low (-31 dB) | General purpose spectral analysis |
| Hamming | Modified cosine | Moderate | Lower (-42 dB) | Audio processing, speech analysis |
| Blackman | Triple cosine | Wide | Very low (-58 dB) | When side lobe suppression is critical |

```
Hann window:    w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
Hamming window: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
```

تطبيق النافذة عن طريق مضاعفةها حسب العناصر مع الإشارة التي تسبق DFT: `X = DFT(x * w)`. . .

### خصائص DFT

| Property | Time Domain | Frequency Domain |
|----------|-------------|-----------------|
| Linearity | a*x + b*y | a*X + b*Y |
| Time shift | x[n - k] | X[f] * e^(-2*pi*i*f*k/N) |
| Frequency shift | x[n] * e^(2*pi*i*f0*n/N) | X[f - f0] |
| Convolution | x * h | X * H (pointwise) |
| Multiplication | x * h (pointwise) | X * H (circular convolution, scaled by 1/N) |
| Parseval's theorem | sum \|x[n]\|^2 | (1/N) * sum \|X[k]\|^2 |
| Conjugate symmetry (real input) | x[n] real | X[k] = conj(X[N-k]) |

نظرية بارسيفال تقول أن الطاقة الكلية هي نفسها في كلا المجالين

### اتصال بتشفيرات المواقع

المصدر الأصلي يستخدم تشفيرات موقف سينوسيدال:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

كل زوج من الأبعاد (2i، 2i+1) يتذبذب عند تردد مختلف. يتم توزيع الترددات بشكل هندسي من عالية (أبعاد 0.1) إلى منخفضة (أبعاد أخيرة). وهذا يعطي كل موقع نمطًا فريدًا عبر جميع فئات التردد - مما يشبه كيفية تحديد معايير فوريري بشكل فريد إشارة.

الخصائص الرئيسية التي يقدمها:

- **Uniqueness:**لا توجد موقفين لديهم نفس التشفير
- **Bounded values:**الخطيئة والصوص دائما في [-1, 1].
- **Relative position:**يمكن تعبير تشفير الموقف p + k كعمل خطي للتشفير في الموقف p. يمكن أن يتعلم النموذج التركيز على المواقع النسبية.

### اتصال مع قنوات سي إن إن

طبقة التخزين تطبق مرشحًا تعلمًا (كور) على المدخل عن طريق نقله عبر الإشارة أو الصورة.

من خلال نظرية التخزين، هذا يعادل:
1. FFT المدخل
2. FFT النجم
3. مضاعفة في مجال التردد
4. إذا كان النتيجة

تستخدم تنفيذات CNN القياسية التناغم المباشر (أسرع بالنسبة للنواة الصغيرة 3x3). ولكن بالنسبة للنواة الكبيرة أو التناغم العالمي ، فإن النهج القائم على FFT أسرع بكثير. بعض الهندسة المعمارية (مثل FNet) تحل محل الاهتمام بالكامل مع FFT ، وتحقيق دقة تنافسية مع O(N log N) بدلاً من تعقيد O(N^2).

### البرامج المتطرفة و تحويل فوريه القصير

يقدم لك FFT واحد محتوى تردد الإشارة بأكملها ، ولكن لا يخبرك بأي شيء عن متى تحدث هذه الترددات. يمكن أن يكون لدى إشارة (إشارة تزداد ترددها مع مرور الوقت) وخط (جميع الترددات الموجودة في نفس الوقت) نفس الطيف الكبير.

تحل تحويل فوريه القصير من الوقت (STFT) هذا الأمر عن طريق حساب FFTs على نوافذ تتداخل من الإشارة. النتيجة هي طيفية: تمثيل ثنائي الأبعاد مع الوقت على محور واحد والتياتر على الآخر. يظهر كثافة كل نقطة الطاقة في تلك التردد في ذلك الوقت.

```
STFT procedure:
1. Choose a window size (e.g., 1024 samples)
2. Choose a hop size (e.g., 256 samples -- 75% overlap)
3. For each window position:
   a. Extract the windowed segment
   b. Apply a Hann/Hamming window
   c. Compute FFT
   d. Store the magnitude spectrum as one column of the spectrogram
```

تعد التصويرات الطيفية هي التمثيل القياسي للمدخلات للنموذجات الصوتية ML. تعمل نماذج التعرف على الكلام (Whisper، DeepSpeech) على التصويرات الطيفية - التصويرات الطيفية ذات الترددات المخطوطة إلى مقياس الميل، والتي تطابق بشكل أفضل تصور الصوت البشري.

### التعبير

إذا كانت الإشارة تحتوي على ترددات فوق fs/2 (تردد نيكوست) ، فإن العينات عند سرعة fs ستخلق نسخة مستعارة. يبدو إشارة 90 Hz التي يتم أخذها في 100 Hz متطابقة مع إشارة 10 Hz. لا توجد طريقة للتمييز بينها من العينات وحدها.

```
Example:
  True signal: 90 Hz sine wave
  Sampling rate: 100 Hz
  Apparent frequency: 100 - 90 = 10 Hz

  The samples from the 90 Hz signal at 100 Hz sampling rate
  are identical to the samples from a 10 Hz signal.
  No amount of math can recover the original 90 Hz.
```

هذا هو السبب في أن محولات التناظر إلى الرقميات تشمل مرشحات مكافحة التناظر التي تزيل ترددات فوق Nyquist قبل أخذ العينات. في ML، يظهر التناظر عند أخذ العينات إلى أسفل خرائط ميزات دون تصفية المنخفضة المناسبة - بعض الهندسة المعمارية تعالج هذا مع طبقات تجمع مكافحة التناظر.

### عدم زيادة الحلّة من خلال إصلاح الصفر

تصور خاطئ شائع: إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إعادة إ إ إ إ إعادة إ إ إ إعادة إعادة إ إ إ إ إعادة إعادة إ إ إ إ إعادة إعادة إ إ إعادة إعادة إ إ إ إ إ إ إ إ إ إ إ إعادة إعادة إ إ إ إ إ إ

يعتمد قرار التردد الحقيقي فقط على وقت الملاحظة T = N / fs. لحل ترددات منفصلة عن delta_f ، تحتاج إلى T = 1 / delta_f ثانية على الأقل من البيانات. لا يوجد مقدار من التدفق الصفر يغير هذا الحد الأساسي.

```figure
fourier-synthesis
```

## بناءها

### الخطوة الأولى: DFT من الصفر

DFT O ((N^2) يتبع مباشرة من التعريف.

```python
import math

class Complex:
    ...

def dft(x):
    N = len(x)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))
            xn = x[n] if isinstance(x[n], Complex) else Complex(x[n])
            total = total + xn * w
        result.append(total)
    return result
```

### الخطوة الثانية: DFT العكسية

نفس الهيكل، المعبر الإيجابي، تقسيم بـ N.

```python
def idft(X):
    N = len(X)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            w = Complex(math.cos(angle), math.sin(angle))
            total = total + X[k] * w
        result.append(Complex(total.real / N, total.imag / N))
    return result
```

### الخطوة الثالثة: FFT (كولي-توكي)

المعدل السريع يتطلب قوة من 2 طول. تقسيم إلى متكافئة ومتكافئة، الجمع مع عوامل التويدل.

```python
def fft(x):
    N = len(x)
    if N <= 1:
        return [x[0] if isinstance(x[0], Complex) else Complex(x[0])]
    if N % 2 != 0:
        return dft(x)

    even = fft([x[i] for i in range(0, N, 2)])
    odd = fft([x[i] for i in range(1, N, 2)])

    result = [Complex(0)] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = Complex(math.cos(angle), math.sin(angle))
        t = twiddle * odd[k]
        result[k] = even[k] + t
        result[k + N // 2] = even[k] - t
    return result
```

### الخطوة الرابعة: مساعدي تحليل الطيف

```python
def power_spectrum(X):
    return [xk.real ** 2 + xk.imag ** 2 for xk in X]

def convolve_fft(x, h):
    N = len(x) + len(h) - 1
    padded_N = 1
    while padded_N < N:
        padded_N *= 2

    x_padded = x + [0.0] * (padded_N - len(x))
    h_padded = h + [0.0] * (padded_N - len(h))

    X = fft(x_padded)
    H = fft(h_padded)

    Y = [xk * hk for xk, hk in zip(X, H)]

    y = idft(Y)
    return [y[n].real for n in range(N)]
```

## استخدمها

للعمل الحقيقي، استخدم Numpy's FFT الذي يدعم من قبل المكتبات C المثلى للغاية.

```python
import numpy as np

signal = np.sin(2 * np.pi * 5 * np.arange(256) / 256)
spectrum = np.fft.fft(signal)
freqs = np.fft.fftfreq(256, d=1/256)

power = np.abs(spectrum) ** 2

positive_freqs = freqs[:len(freqs)//2]
positive_power = power[:len(power)//2]
```

لتحليل النوافذ والتحليلات الطيفية الأكثر تقدما:

```python
from scipy.signal import windows, stft

window = windows.hann(256)
windowed = signal * window
spectrum = np.fft.fft(windowed)
```

للفصل:

```python
from scipy.signal import fftconvolve

result = fftconvolve(signal, kernel, mode='full')
```

للطيفيات:

```python
from scipy.signal import stft

frequencies, times, Zxx = stft(signal, fs=sample_rate, nperseg=256)
spectrogram = np.abs(Zxx) ** 2
```

المصفوفة المعدنية لها شكل (n_frequencies, n_time_frames). كل عمود هو الطيف الطاقة في نافذة زمنية واحدة. هذا ما تستخدمها نماذج ML الصوتية كمدخول.

## أرسله

أركض`code/fourier.py`لإنتاج`outputs/prompt-spectral-analyzer.md`. . .

## التمارين

1. **Pure tone identification.**قم بإنشاء إشارة بموجة سينوسية واحدة في تردد مجهول (بين 1 و 50 هرتز) ، وخذ عينات عند 128 هرتز لمدة ثانية واحدة. استخدم DFT الخاص بك لتحديد التردد. تحقق من مطابقة الإجابة. الآن أضف ضجيج غوسيان مع انحراف معيار 0.5 واكرر. كيف يؤثر الضجيج على الطيف؟

2. **FFT vs DFT verification.**توليد إشارة عشوائية بطول 64. حساب كل من DFT (O(N^2)) و FFT. التحقق من أن جميع المعاملات تتطابق مع داخل 1e-10. الوقت يعمل على الإشارات بطول 256 ، 512 ، 1024, و 2048. رسم نسبة DFT وقت إلى FFT وقت.

3. **Convolution theorem proof by example.**قم بإنشاء إشارة x = [1, 2, 3, 4, 0, 0, 0, 0] والتحذير h = [1, 1, 1, 0, 0, 0, 0, 0]. احسب تحوُّلها الدائري مباشرةً (حلقة حلقة). ثم احسبها عن طريق FFT (تحول، مضاعفة، تحويل عكسية). تحقق من مطابقة النتائج. الآن قم بتحوُّل خطيّة عن طريق إضافة الصفر بشكل مناسب.

4. **Windowing effects.**قم بإنشاء إشارة هي مجموع موجات الصناعية الثنائية عند 10 هرتز و 12 هرتز (قريبة جداً). قم بعمل عينات عند 128 هرتز لمدة ثانية واحدة. احسب طيف الطاقة دون نافذة، نافذة هان، ونفذة هامينغ. أي نافذة تجعل من الأسهل تمييز النقبتين؟ لماذا؟

5. **Positional encoding analysis.**توليد التشفيرات الموضعية السينوسوايدالية d_model = 128 و max_pos = 512. لكل زوج من المواقع (p1, p2), حساب نسبة نقطة من التشفيرات الخاصة بهم. أظهر أن نسبة نقطة تعتمد فقط على p1 - p2 ، وليس على المواقع المطلقة. ماذا يحدث لمجموعة نقطة مع زيادة المسافة؟

## الشروط الرئيسية

| Term | What it means |
|------|---------------|
| DFT (Discrete Fourier Transform) | Converts N time-domain samples into N frequency-domain coefficients. Each coefficient is the correlation with a complex sinusoid at that frequency |
| FFT (Fast Fourier Transform) | An O(N log N) algorithm to compute the DFT. The Cooley-Tukey algorithm splits even/odd indices recursively |
| Inverse DFT | Reconstructs the time-domain signal from frequency coefficients. Same formula as DFT with flipped exponent sign and 1/N scaling |
| Frequency bin | Each index k in the DFT output represents frequency k*fs/N Hz. The "bin" is the discrete frequency slot |
| DC component | X[0], the zero-frequency coefficient. Proportional to the signal mean |
| Nyquist frequency | fs/2, the maximum frequency representable at sampling rate fs. Frequencies above this alias |
| Power spectrum | \|X[k]\|^2, the squared magnitude of each frequency coefficient. Shows energy distribution across frequencies |
| Phase spectrum | angle(X[k]), the phase offset of each frequency component. Often ignored in analysis |
| Spectral leakage | Spurious frequency content caused by treating a non-periodic signal as periodic. Reduced by windowing |
| Window function | A tapering function (Hann, Hamming, Blackman) applied before DFT to reduce spectral leakage |
| Twiddle factor | The complex exponential e^(-2*pi*i*k/N) used to combine sub-DFTs in the FFT butterfly computation |
| Convolution theorem | Convolution in time domain equals pointwise multiplication in frequency domain. Fundamental to signal processing and CNNs |
| Circular convolution | Convolution where the signal wraps around. This is what the DFT naturally computes |
| Linear convolution | Standard convolution without wraparound. Achieved by zero-padding before DFT |
| Parseval's theorem | Total energy is preserved through the Fourier transform. sum \|x[n]\|^2 = (1/N) sum \|X[k]\|^2 |
| Aliasing | When frequencies above Nyquist appear as lower frequencies due to insufficient sampling rate |

## المزيد من القراءة

- [Cooley & Tukey: An Algorithm for the Machine Calculation of Complex Fourier Series (1965)](https://www.ams.org/journals/mcom/1965-19-090/S0025-5718-1965-0178586-1/)- ورقة FFT الأصلية التي غيرت الحوسبة
- [3Blue1Brown: But what is the Fourier Transform?](https://www.youtube.com/watch?v=spUNpyF58BY)- أفضل تعريف بصري لتحولات فوريير
- [Lee-Thorp et al.: FNet: Mixing Tokens with Fourier Transforms (2021)](https://arxiv.org/abs/2105.03824)- يبدل الاهتمام الذاتي بـ FFT في المحولات
- [Smith: The Scientist and Engineer's Guide to Digital Signal Processing](http://www.dspguide.com/)- كتاب دراسي مجاني على الإنترنت يغطى التحليل المعمق للفوتوغرافية والنوافذ والطيفيات
- [Vaswani et al.: Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)- تشفيرات المواقع السينوسيدالية المستمدة من تدهور تردد فوريير
- [Radford et al.: Whisper (2022)](https://arxiv.org/abs/2212.04356)- التعرف على الكلام باستخدام طيفيات الميل كتمثيل مدخل
