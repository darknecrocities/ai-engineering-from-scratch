# Saçma Bayes

> "Sane" varsayımı yanlış ve yine de işe yarıyor.

**Type:** Build
**Language:**Python
**Prerequisites:** Phase 2, Lessons 01-07 (classification, Bayes' theorem)
**Time:** ~75 minutes

## Öğrenme Hedefleri

- Metin sınıflandırması için Laplace düzeltmesi ile Multinomial Naive Bayes'i sıfırdan uygulayın
- Neden saf bağımsızlık varsayımının matematiksel olarak yanlış olduğunu, ancak pratikte doğru sınıf sıralamasını ürettiğini açıklayın
- Multinomial, Bernoulli ve Gaussian Naive Bayes varianlarını karşılaştırın ve verilen bir özellik tipi için doğru olanı seçin
- Yüksek boyutlu nadir veriler üzerinde mantıksal gerileme karşısında Naive Bayes'i değerlendirmek ve işyerindeki önyargı-varians pazarlamasını açıklamak

## Sorun

Metinleri sınıflandırmanız gerekir. E-postaları spam veya spam olmayan olarak. Müşteri incelemeleri olumlu veya negatif olarak. Destek biletleri kategorilere. Binlerce özelliğe (her kelime için bir tane) ve sınırlı eğitim verisine sahipsiniz.

Çoğu sınıflandırıcı burada boğulur. Logistik geri dönüş binlerce ağırlığı güvenilir bir şekilde tahmin etmek için yeterli örneklere ihtiyaç duyar. Karar ağaçları bir seferde bir kelime üzerine bölünür ve vahşice aşırıya çarpar. 10.000 boyutta KNN anlamsızdır çünkü her nokta diğer noktalardan eşit derecede uzakta.

Naif Bayes bu işi halleder. Matematik olarak yanlış bir varsayım yapar (her bir özelliğin sınıf verilmiş diğer tüm özelliğe bağımsız olması), ve özellikle küçük eğitim setleri ile metin sınıflandırması üzerindeki "akıllı" modellerden daha üstün bir performans sergiliyor. Verileri tek bir atışla geçirir. Milyonlarca özelliklere kadar ölçeklendiriyor. Bu, olasılık tahminlerini üretir (bağımsızlık varsayımından dolayı genellikle kötü kalibre edilse de).

Yanlış bir varsaymanın neden iyi tahminlere yol açtığını anlamak, makine öğrenimi hakkında temel bir şey öğretir: En iyi model en doğru model değil, verileriniz için en iyi önyargı-varians pazarlaması olan model.

## Anlaşım

### Bayes teoremi (Hızlı inceleme)

Bayes teoremi koşullu olasılıkları tersine çevirir:

```
P(class | features) = P(features | class) * P(class) / P(features)
```

İstiyoruz .`P(class | features)`- bir belgenin içindeki kelimeleri göz önüne alarak bir sınıfın birine ait olma olasılığı.
- `P(features | class)`-- bu kelimeleri bu sınıfın belgelerinde görme olasılığı
- `P(class)`-- sınıfın önceki olasılığı (spam genel olarak ne kadar yaygın?)
- `P(features)`- kanıtlar, tüm sınıflar için aynı, böylece karşılaştırırken görmezden gelebiliriz

En yüksek sınıfı .`P(class | features)`- Kazandı.

### Saf Bir Bağımsızlık Farkında

Bilgisayar `P(features | class)`Bu, tüm özelliklerin ortak olasılığını tahmin etmenizi gerektirir. 10.000 kelimelik bir kelime birikimi ile, 2^10.000 olası kombinasyonların dağılımını tahmin etmeniz gerekir.

Saf bir varsayım: her özellik sınıfı göz önüne alındığında koşullarla bağımsızdır.

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

Bir imkansız ortak dağılım yerine, n basit özellik başına dağılım tahmin ediyorsunuz.

Bu varsayım açıkça yanlış. "makine" ve "öğrenme" kelimeleri hiçbir belgede bağımsız değildir. Ama sınıflandırıcı doğru olasılık tahminlerine ihtiyaç duymaz. Doğru sıralamalara ihtiyaç duyar. Hangi sınıfın en yüksek olasılıkları vardır. Bağımsızlık varsayımı sistematik hatalar getiriyor, ancak bu hatalar tüm sınıfları benzer şekilde etkilemektedir, bu nedenle sıralama doğru kalır.

### Neden Hala Çalışıyor?

Üç neden:

1. **Ranking over calibration.**Sınıflandırma sadece en yüksek sıralamalı sınıfın doğru olması gerekir. Gerçek olasılık 0.7 olduğunda P(spam) = 0.99999 olsa bile, sınıflandırıcı hala spam'i doğru seçer. Doğru olasılıklara ihtiyacımız yok. Doğru kazananı ihtiyacımız var.

2. **High bias, low variance.**Bağımsızlık varsayımı güçlü bir önlemdir. Modelli ağır bir şekilde kısıtlar ve bu da aşırı uyum sağlamayı önler. sınırlı eğitim verileri ile, hafif yanlış ama istikrarlı bir model teorik olarak doğru ama çok istikrarlı bir modelden daha üstün bir modeldir. Bu, aksiyonda önyargı-varians pazarlamasıdır.

3. **Feature redundancy cancels out.**İlişkili özellikler fazladan kanıt sağlar. sınıflandırıcı bu kanıtları iki kat sayır, ancak doğru sınıf için de iki kat sayır. "makine" ve "öğrenme" her zaman birlikte görünürse, her ikisi de "teknoloji" sınıfı için kanıt sağlar. NB onları iki kez sayır, ancak doğru sınıf için iki kez sayır.

Dördüncü, pratik bir neden: Naive Bayes son derece hızlıdır. Eğitim veri sayım frekansları üzerinden tek bir geçiştir. Tahmin bir matris çarpımıdır. Bir milyon belge üzerinde saniyeler içinde eğitim alabilirsiniz. Bu hız daha hızlı tekrarlayabileceğiniz, daha fazla özellik seti denemeyebileceğiniz ve daha yavaş modellere kıyasla daha fazla deney yapabileceğiniz anlamına gelir.

### Matematika Adım Adım

Şimdi, bir örnekle bir araya geliyoruz. Diyelim ki iki sınıfımız var: spam ve spam olmayan. Sözcüklerimiz üç kelimeye sahiptir: "bezgin", "para", "bir araya gelme".

Eğitim verileri:
- Spam e-postalarında "be bedava" 80 kez, "para" 60 kez, "bir araya gelme" 10 kez (150 kelimenin toplamı) bahsedildi.
- Spam olmayan e-postalarda "be bedava" 5 kez, "para" 10 kez, "bir araya gelme" 100 kez (115 kelimenin toplamı)
- E-postaların %40'ı spam, %60'ı spam değil.

Laplace düzeltmesi ile (alfa=1):

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

Yeni e-posta: "be bedava" (2 kez), "para" (1 kez), "bir toplantı" (0 kez).

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

Spam büyük bir farkla kazanır. "Özgür" kelimesi iki kez ortaya çıkmak spam için güçlü bir kanıtdır. "Dörüşme" görünmemesinin her iki log toplamına sıfır katkıda bulunduğunu unutmayın (0 * log(P)) - Multinomial NB'de, yok kelimelerin hiçbir etkisi yoktur.

### Üç Çeşit

Naive Bayes üç tadda gelir.`P(feature | class)`Farklı bir şekilde.

#### Çoklu isimler Naif Bayes

Modeller her özelliği bir sayım olarak oluşturur. Özellikleri kelimeler sıklığı veya TF-IDF değerleri olan metin verileri için en iyisidir.

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

- Evet .`alpha`Bu variant metin sınıflandırması için iş atıdır.

#### Gaussian Naive Bayes

Her bir özellik normal bir dağılım olarak modeller.

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

Her sınıfın özelliği ve farklılığı vardır. Bu, özellikler her sınıf içinde gerçekten bir çan eğrisini takip ettiğinde iyi çalışır.

#### Bernoulli Naive Bayes

Modeller her özelliği ikili ( mevcut veya yok) olarak oluşturur. Kısa metin veya ikili özelliği vektörleri için en iyisi.

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

Multinomial'den farklı olarak, Bernoulli bir kelimenin yokluğuna açıkça cezalandırır. Eğer "belir" tipik olarak spam'de görünür ancak bu e-postada yoksa, Bernoulli bunu spam karşı bir kanıt olarak sayır.

### Her Bir Variantı Ne Zaman Kullanmalıyız?

| Variant | Feature Type | Best For | Example |
|---------|-------------|----------|---------|
| Multinomial | Counts or frequencies | Text classification, bag-of-words | Email spam, topic classification |
| Gaussian | Continuous values | Tabular data with normal-ish features | Iris classification, sensor data |
| Bernoulli | Binary (0/1) | Short text, binary feature vectors | SMS spam, presence/absence features |

### Laplace Düzeltme

Bir kelime test verilerinde görünse de belirli bir sınıf için eğitim verilerinde hiç görünmese ne olur?

Düzeltmeden:`P(word | class) = 0/N = 0`. Tüm ürünün çarpıtıyla bir sıfır yapar `P(class | features) = 0`Tek bir görünmeyen sözcük tüm tahminleri yok eder, ne kadar başka kanıt desteklemesine rağmen.

Laplace düzeltmesi küçük bir sayıyı ekler .`alpha`(genellikle 1) her özellik sayısına:

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

Alpha = 1 ile her kelime en az küçük bir olasılık elde eder. Test e-postalarında görünen "discombobulate" kelimesi artık spam olasılığını öldürmez. Düzeltmenin Bayesian bir yorumuna sahiptir: kelime dağılımlarına bir benzer Dirichlet ön koymaya eşittir.

Daha yüksek alfa, daha güçlü bir düzeltme (daha benzer dağılımlar) anlamına gelir. Daha düşük alfa, modelin verilere daha fazla güvendiğini gösterir.

Alfa etkisi:

| Alpha | Effect | When to use |
|-------|--------|-------------|
| 0.001 | Almost no smoothing, trust the data | Very large training set, no unseen features expected |
| 0.1 | Light smoothing | Large training set |
| 1.0 | Standard Laplace smoothing | Default starting point |
| 10.0 | Heavy smoothing, flattens distributions | Very small training set, many unseen features expected |

### Log-Space Hesaplama

Yüzlerce olasılık (her biri 1'den daha az) çarpması, yüzen noktaların aşağı akışına neden olur. Gerçek değer çok küçük bir pozitif sayısıysa da ürün yüzen noktalarda sıfır olur.

Çözüm: log alanında çalış. Muhtemelenlikleri çarpmak yerine, logaritmlerini ekleyin:

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

Bu tahminleri nokta ürünü haline getirir:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

Matrix çarpımı. Bu yüzden Naive Bayes tahminleri bu kadar hızlı -- tek katlı bir doğrusal modelle aynı işlemdir.

### Naif Bayes vs. Logistik Geri Dönüş

Her ikisi de metin için doğrusal sınıflandırıcılar. Fark, modellemelerindedir.

| Aspect | Naive Bayes | Logistic Regression |
|--------|------------|-------------------|
| Type | Generative (models P(X\|Y)) | Discriminative (models P(Y\|X)) |
| Training | Count frequencies | Optimize loss function |
| Small data | Better (strong prior helps) | Worse (not enough to estimate weights) |
| Large data | Worse (wrong assumption hurts) | Better (flexible boundary) |
| Features | Assumes independence | Handles correlations |
| Speed | Single pass, very fast | Iterative optimization |
| Calibration | Poor probabilities | Better probabilities |

Basamak kural: Naive Bayes ile başlayın. Yeterince verileriniz ve NB platolarınız varsa, lojistik geri dönüşe geçin.

### Sınıflandırma boru hattı

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

Bu nedenle, bu işlemler, bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre sonra bir süre daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha daha

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## Yapın

Kodun içinde .`code/naive_bayes.py`Hem MultinomialNB hem de GaussianNB'yi sıfırdan uyguluyor.

### Çoklu isimNB

Baştan başlayan uygulama:

1. **fit(X, y)**: Her sınıf için, her özelliğin sıklığını sayın. Laplace düzeltmesini ekleyin. Log olasılıklarını hesaplayın. Sınıf önceleri (sınıf frekanslarının logu)

2. **predict_log_proba(X)**: Her örnek için, hesap log P(sınıf) + tüm sınıflar için log P(kaynak_i sınıfı) toplamı. Bu bir matris çarpımı: X @ log_probs.T + log_priors.

3. **predict(X)**: En yüksek log olasılığı olan sınıfı geri gönderin.

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

Anahtar anlayış: uyumlandıktan sonra tahmin sadece matris çarpımı artı bir önyargı.

### GaussianNB

Sürekli özellikler için, sınıf başına ortalama ve varyasyonu bir özellik başına tahmin ediyoruz:

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

Tahmin, özellikler boyunca çarpılmış Gaussian PDF'yi kullanır (log alanında eklenir).

### Demo: Metin sınıflandırması

Kod iki sınıfı simüle eden sentetik sözcükler verisini oluşturur (teknoloji makaleleri vs. spor makaleleri). Her sınıfın farklı bir kelime frekans dağılımına sahiptir. MultinomialNB onları kelime sayılarını kullanarak sınıflandırır.

Sentetik veriler şöyle çalışır: 200 "söz" (kaynak sütunları) oluşturuyoruz. 0-39 kelimeleri teknik makalelerde yüksek frekanslı ve sporda düşük. 80-119 kelimeleri sporda yüksek frekanslı ve teknikte düşük. 40-79 kelimeleri her ikisinde orta frekanslı. Bu, bazı kelimelerin güçlü sınıf göstergeleri olduğu ve diğerlerinin gürültü olduğu gerçekçi bir senaryo yaratır.

### Demo: Sürekli Özellikler

Kod Iris benzeri verileri oluşturur (3 sınıf, 4 özellik, Gaussian kümeleri). GaussianNB sınıf başına ortalama ve varyansa kullanılarak sınıflandırır. Her sınıfın farklı bir merkezi (ortalama vektörü) ve farklı bir yayılması (varyansa) vardır. Ölçümlerin kategoriler arasında sistematik olarak farklı olduğu gerçek dünya verilerini taklit eder.

Kod ayrıca şunları gösterir:
- **Smoothing comparison:**Düzgünliğe yumuşaklık etkisini göstermek için farklı alfa değerleri ile MultinomialNB eğitimi.
- **Training size experiment:**NB'nin doğruluğu eğitim verileri 20'den 1600'e kadar arttıkça nasıl gelişiyor. NB çok az örnekle bile uygun doğruluğa ulaşır. Bu onun ana avantajıdır.
- **Confusion matrix:**Sınıf başına hassaslık, hatırlama ve F1 skorları NB'nin nerede hata yaptığını gösterir.

### Tahmin Hızı

Naive Bayes tahminleri bir matris çarpımıdır.
- MultinomialNB: bir matris çarpı (n x d) @ (d x k) = O(n * d * k)
- GaussianNB: n * k Gaussian PDF değerlendirmeleri, her biri d özellikler = O(n * d * k)

Her iki boyutda da doğrusaldır. Bunu KNN ile (bütün eğitim noktalarına uzaklık hesaplama gerektiren) veya RBF çekirdeği ile SVM ile (bütün destek vektörlerine karşı çekirdeği değerlendirme gerektiren) karşılaştırın. NB, tahmin zamanında büyüklük sıralamaları ile daha hızlıdır.

## Kullan

sklearn ile, her iki varians da tek satırlı:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

Sklüarn ile metin sınıflandırması için:

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

Kodun içinde .`naive_bayes.py`Düzgünlüğü kontrol etmek için aynı verilere göre sıfırdan uygulanmaları sklearn ile karşılaştırır.

### TF-IDF, Naive Bayes ile

Çiğ kelimeler sayımı her kelimeyi olay başına eşit ağırlık verir. Ama "i" ve "i" gibi yaygın kelimeler her sınıfta sıkça görünür - hiçbir bilgi taşımıyorlar. TF-IDF (Term Frequency - Inverse Document Frequency) yaygın kelimeleri ağırlık altına alır ve nadir, ayrımcı kelimeleri ağırlık altına alır.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF değerleri negatif değildir, bu nedenle MultinomialNB ile çalışır. TF-IDF + MultinomialNB kombinasyonu metin sınıflandırması için en güçlü temel çizgilerden biridir.

### BernoulliNB kısa metin için

Kısa metinler için (tweetler, SMS, sohbet mesajları), BernoulliNB MultinomialNB'yi üstlenebilir. Kısa metinler düşük kelime sayısına sahiptir, bu nedenle MultinomialNB'nin güvendiği frekans bilgileri gürültülüdür. BernoulliNB sadece varlık veya yoklukla ilgilenir, bu da kısa metinle daha güvenilirdir.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

- Evet .`binary=True`CountVectorizer'deki bayrak tüm sayıları 0/1'ye dönüştürür.

### Kalibrasyon NB Muhtemelenlikler

NB olasılıkları kötü kalibrlenmiştir. NB'de P(spam) = 0.95 olduğu zaman, gerçek olasılık 0.7 olabilir.

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

Bu, NB'nin çiğ puanlarının üstündeki bir lojistik geri dönüşe karşılık gelir.

### Ortak Gotchas

1. **Negative feature values.**MultinomialNB negatif olmayan özellikleri gerektirir. Negatif değerleriniz varsa (bazı ayarlarla veya standart özelliklerle TF-IDF gibi), GaussianNB'yi kullanın veya özellikleri pozitif olarak değiştirin.

2. **Zero variance features.**GaussianNB, bir sınıf için sıfır bir değişikliğe sahipse (tüm değerler aynıdır), olasılık hesaplaması bozulur.

3. **Class imbalance.**Eğer e-postaların %99'u spam değilse, önceki P(not-spam) = 0.99 o kadar güçlüdür ki olasılık kanıtlarını aşıyor.

4. **Feature scaling.**MultinomialNB'nin ölçeklendirmeye ihtiyacı yoktur (sayılar üzerinde çalışır). GaussianNB'nin de ölçeklendirmeye ihtiyacı yoktur (sözümlü özellik istatistiklerini tahmin eder). Bu, özellik ölçeklerine duyarlı olan lojistik gerileme ve SVM'ye göre bir avantaj.

## Gönder

Bu ders şunları ortaya çıkarır:
- `outputs/skill-naive-bayes-chooser.md`-- doğru NB varianti seçmek için karar verme becerisi
- `code/naive_bayes.py`-- MultinomialNB ve GaussianNB sıfırdan, sklearn karşılaştırması ile

### Naif Bayes Başarısız olduğunda

NB bağımsızlık varsayımının yanlış sıralamalara neden olduğu (sadece yanlış olasılıklar değil) durumlarda başarısız olur.

1. **Strong feature interactions.**Eğer sınıf iki özelliğin birleşmesine bağlıysa ancak tek başına (XOR benzeri desenler) değilse, NB onu tamamen kaçırır.

2. **Highly correlated features with opposing evidence.**Eğer A özelliği "spam" ve B özelliği "spam değil" diyor, ancak A ve B mükemmel bir şekilde ilişkili (gerçekte her zaman aynı fikirde) ise, NB hiçbir kanıt olmadığı yerde çelişkili kanıt görür.

3. **Very large training sets.**Yeterince veri ile, lojistik gerileme gibi ayrımcılık modelleri gerçek karar sınırını öğrenir ve NB'yi aşırır. Küçük verilerle yardımcı olan bağımsızlık varsayımı artık modeli geri tutuyor.

Bu tür hata modları metin sınıflandırması için nadirdir. Metin özellikleri sayısızdır, bireysel olarak zayıfdır ve bağımsızlık varsayımının hataları iptal edilme eğilimindedir.

## Egzersizler

1. **Smoothing experiment.**MultinomialNB'yi 0.01, 0.1, 1.0, 10.0 ve 100.0 alfa değerleri olan metin verilerine çalıştır.

2. **Feature independence test.**Gerçek bir metin veri kümesi alın. Açıkça ilişkili olan iki kelimeyi seçin ("makine" ve "öğrenme"). P  word1  class * P  word2  class) hesaplayın ve P  word1 AND word2  class ile karşılaştırın. Bağımsızlık varsayımı ne kadar yanlış?

3. **Bernoulli implementation.**BernoulliNB sınıfıyla kodu genişlet. Sözcük çantalarını ikili ( mevcut/ yok) olarak dönüştürün ve metin verilerindeki MultinomialNB ile doğruluğu karşılaştırın. Bernoulli ne zaman kazanır?

4. **NB vs Logistic Regression.**Her ikisini de metin verileri üzerine eğit. 100 eğitim örneği ile başlayın ve 10.000'e yükseltsin. Her ikisinin de plan doğruluğu vs. eğitim setinin boyutu. Logistik Geri dönüş hangi noktada Naive Bayes'i geçiyor?

5. **Spam filter.**Tam bir spam sınıflandırıcısı oluşturun: çiğ e-posta metnini işaretleyin, kelime birikimi oluşturun, sözcük çanta özellikleri oluşturun, MultinomialNB'yi eğitiniz, doğru bir şekilde değerlendirin ve geri çağırın (sadece doğruluk değil - neden?).

## Anahtar Terimler

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

## Daha Fazla Okumak

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html)- Matematik detaylarla birlikte üç farklılık
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf)-- metin için Multinomial vs Bernoulli'nin klasik karşılaştırması
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)-- metin için NB'de gelişmeler
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)-- daha az veri ile NB'nin LR'den daha hızlı bir şekilde yakınlaştığını kanıtlar
