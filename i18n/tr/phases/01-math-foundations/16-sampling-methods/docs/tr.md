# Örnekleme yöntemleri

> Örnekleme, AI'nin olasılık alanını nasıl keşfettiğini gösterir.

**Type:** Build
**Language:**Python
**Prerequisites:** Phase 1, Lessons 06-07 (Probability, Bayes' Theorem)
**Time:** ~120 minutes

## Öğrenme Hedefleri

- Teker teker rastgele sayılar kullanarak ters CDF, reddetme ve önem örneğini sıfırdan uygulayın
- Dil model simge üretimi için sıcaklık, üst-k ve üst-p (yüzey) örneklemesini oluştur
- Reparametreleme hilesini ve neden VAE'lerde örnekleme yoluyla geri yayılmayı mümkün kılanını açıklayın.
- Metropolis-Hastings MCMC'yi normalleşmemiş bir hedef dağıtımdan örnek almak için çalıştır

## Sorun

Dil modeli, sorguyu işlemeyi bitirir ve 50.000 logitlik bir vektör üretir. Sözlükte bir simge için bir tane. Şimdi bir tane seçmelidir. Nasıl?

Eğer her zaman en yüksek olasılık belirtiğini seçerse, her cevap aynıdır. Deterministik. Sıkıcı. Eğer rastgele olarak eşit seçerse, çıkış kayıp olur. Cevap bu aşırılıkların arasında bir yerde yaşar ve bir yerlerde örnekleme ile kontrol edilir.

Örnekleme metin üretimi ile sınırlı değildir. Güçlendirme öğrenimi, örnekleme rotaları ile politika gradiyentlerini tahmin eder. VAE'ler öğrenilen dağılımlardan örnek alarak ve rastgelelik yoluyla geri yayılarak gizli temsilleri öğrenir. Diffüzyon modelleri gürültü örneği ve tekrar tekrar denoizasyon yoluyla görüntüler üretir. Monte Carlo yöntemleri, kapalı biçimli çözüm olmayan bütünleri tahmin eder. MCMC algoritmaları, saymak imkansız olan yüksek boyutlu arka dağılımları araştırır.

Her üreticik AI sistemi bir örnekleme sistemidir. Örnekleme stratejisi çıkışın kalitesini, çeşitliliğini ve kontrol edilebilirliğini belirler. Bu ders, her büyük örnekleme yöntemini sıfırdan inşa eder, eşit rastgele sayılardan başlayarak modern LLM'leri ve üreticik modellerle güçlendiren tekniklerle sona erer.

## Anlaşım

### Örnek Almanın Önemli Olduğu Nedeni

Örnekleme, AI ve makine öğrenimi genelinde dört temel rolde ortaya çıkar:

**Generation.**Dil modelleri, difüzyon modelleri ve GAN'lar tümüyle örnekleme yoluyla çıkış üretir. Örnekleme algoritması yaratıcılığı, tutarlılığı ve çeşitliliği doğrudan kontrol eder.

**Training.**Stochastic gradient descent örnekleri mini-batch. Deaktivasyon için nöron örnekleri bırakın. Veri artırma örnekleri rastgele dönüşümler. Önemlilik örneklemesi güçlendirme öğreniminde gradient variansını azaltmak için örnekleri yeniden ağırlaştırır (PPO, TRPO).

**Estimation.**ML'deki birçok miktarda kapalı bir çözüm yoktur. Veriler dağıtımında beklenen kayıp, enerji tabanlı bir modelin bölünme fonksiyonu, Bayesian sonuçlarındaki kanıtlar. Monte Carlo tahminleri tüm bunları örnekler üzerinde ortalama olarak yaklaştırır.

**Exploration.**MCMC algoritmaları Bayesian sonuçlarında arka dağılımları araştırır. Evrimsel stratejiler parametreler bozukluklarını örnekler. Thompson örneklemesi, suikastçılarda keşif ve sömürü dengeler.

Temel zorluk: sadece basit dağılımlardan (eşit, normal) doğrudan örnek alabilirsiniz. Diğer her şey için, basit örnekleri hedef dağılımınızdan örneklere dönüştürmek için bir yöntem gerekir.

### Eşsiz Rastgele Örnekleme

Her örnekleme yöntemi burada başlar. Bir benzer rastgele sayı jeneratörü, eşit uzunlukta her alt aralığın eşit olasılıkla olduğu [0, 1) değerlerini üretir.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

N öğelerin ayrı bir setinden benzer bir şekilde örnek almak için U oluşturun ve tekrar katı ((n * U).

Anahtar bir anlayış: tek bir eşsiz rastgele sayı herhangi bir dağılımdan bir örnek üretmek için tam olarak doğru miktarda rastgeleliği içerir.

### Ters CDF Yöntem (Dönüştürülmüş Değişiklik Örnekleme)

Kumulatif dağılım fonksiyonu (CDF) değerleri olasılıklara haritası yapar:

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

Ters CDF olasılıkları değerlere geri harfler. Eğer U ~ Uniform(0, 1), o zaman X = F_inverse(U) hedef dağılım takip eder.

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Exponential distribution example:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

Bu, F_inverse'i kapalı biçimde yazabildiğinizde mükemmel bir şekilde çalışır. Normal dağılım için, kapalı biçimdeki ters CDF yoktur, bu yüzden diğer yöntemleri kullanıyoruz (Box-Muller veya sayısal yaklaşım).

**Discrete version:**Diskre dağılımlar için CDF'yi kumülatîf bir toplam olarak oluşturun, U oluşturun ve kumülatîf toplamın U'yu aştığı ilk indeks bulun.`sample_categorical`6. Ders'te çalışmalar.

### Reddetme Örnekleme

CDF'yi tersine çeviremediğinizde ama hedef PDF'yi sabit bir şekilde değerlendirebildiğinizde, reddetme örneği çalışmaktadır.

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

M'nin bağlanması ne kadar sıkı olursa, kabul oranı o kadar yüksek olur. Düşük boyutlarda (1-3), reddetme örneği iyi çalışır. Yüksek boyutlarda, kabul oranı, önerinin büyük bir kısmı reddedildiği için eksponansal olarak düşer. Bu reddetme örneği için boyutluk lanetidir.

**Example: sampling from a truncated normal.**Kısaltılmış aralık üzerinde bir teklifte bulun. M zarf, bu aralığın normal PDF'lerinin maksimumıdır.

**Example: sampling from a semicircle.**Yönlü düzbuzda teker teker önerin. Nokta yarı döngü içinde düşerse kabul edin. Monte Carlo pi'yi böyle hesaplar: kabul oranı alan oranı pi/4'e eşittir.

### Önemlilik Örnekleme

Bazen hedef dağılım p(x'den örneklere ihtiyacınız yoktur. Bir beklentiyi p(x altında tahmin etmek gerekir ve farklı bir dağılım q(x'den örnekler vardır.

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

Bu, güçlendirme öğreniminde kritik bir önem taşır. PPO (Prosimal Policy Optimization) da eski bir politika pi_old altında yoldurları toplarsınız ancak yeni bir politika pi_new'i optimize etmek istiyorsunuz. Önemliyet ağırlığı pi_new'ler / pi_old'lar.

Önemlilik örnekleme tahminçisinin değişimi q'nın p'ye ne kadar benzer olduğuna bağlıdır. q p'den çok farklıysa, birkaç örnek büyük ağırlıklar alır ve tahminin baskısıdır.

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Monte Carlo Tahmini

Monte Carlo tahminleri rastgele örneklerin ortalamasını kullanarak bütünlükleri yakındırır. Büyük sayılar kanunu, yakınlaşmayı garanti eder.

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

Hata oranı boyutlara bağlıdır. Bu nedenle Monte Carlo yöntemleri, ağ tabanlı entegrasyonun mümkün olmadığı yüksek boyutlarda baskındır.

**Estimating pi:**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**Estimating expectations:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo (MCMC): Metropolis-Hastings

MCMC, sabit dağılımının hedef dağılım p ((x) olan Markov zinciri oluşturur. Yeterli adımlardan sonra, zincirden örnekler (yaklaşık olarak) p ((x) örnekler olur.

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

Simetrik öneriler için (q(x'taitx) = q(x tık x') oranı p(x')/p(x'e basitleştirilir.

**Why it works.**Kabul kuralı detaylı dengeyi sağlar: x'de bulunma ve x'ye taşınma olasılığı x'de bulunma ve x'ye taşınma olasılığına eşit olur. Detaylı denge p ((x) zincirin sabit dağılımını içerir.

**Practical considerations:**
- Yanma: zincir dengeleme ulaşmadan önce erken örnekler atılır
- İnceleme: Otokorelasyonu azaltmak için her k-te örnek tutun
- Teklif ölçeği: çok küçük ve zincir yavaş hareket eder (yüksek kabul, yavaş araştırmalar); çok büyük ve çoğu teklif reddedilmiştir ( düşük kabul, yerinde kalmıştır)
- Yüksek boyutlarda Gaussian önerisi için en iyi kabul oranı yaklaşık 0.234'dir.

### Gibbs Örnekleme

Gibbs örnekleme, çok değişken dağılımlar için MCMC'nin özel bir durumudur. Tüm boyutlarda bir seferde bir hareket önermek yerine, koşullu dağılımından bir değişkenyi bir seferde güncelleştirir.

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs örneği, her koşullu dağılımdan örnek alabilmenizi gerektirir.
- Bayesian ağlar: grafik yapısından şartlar
- Gaussian karışımları: şartlar Gaussian
- İzin modelleri: her spin'in koşulları sadece komşularına bağlıdır

Kabul oranı her zaman 1'dir (her teklif kabul edilir), çünkü tam şartlı örnekleme otomatik olarak ayrıntılı dengeni tatmin eder.

**Limitation.**Değişkenler yüksek bir korelasyonda olduğunda, Gibbs örnekleme yavaş karışır çünkü bir değişkenyi bir seferde güncelleme dağılım boyunca büyük çapraz hareketler yapamaz.

### Temperatür örneği (LLM'lerde kullanılır)

Dil modelleri sözcüklükteki her token için z_1, ..., z_V logitlerini çıkarır. Softmax bunları olasılıklara dönüştürür.

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**Why it works.**Logitleri T < 1 ile bölmek logitler arasındaki farkları artırır. Z_1 = 2 ve z_2 = 1 ise, T = 0.5 ile bölmek z_1/T = 4 ve z_2/T = 2 verir.

**In practice:**
- T = 0.0: açgözlülükle çözme, gerçekçi soru ve cevaplar için en iyisi
- T = 0.3-0.7: biraz yaratıcı, kod üretimi için iyi
- T = 0,7-1,0: dengeli, genel konuşma için iyi
- T = 1.0-1.5: Yaratıcı yazma, beyin fırtınası
- T > 1,5: giderek rastgele, nadiren kullanışlı

Temperatür, hangi tokenlerin mümkün olduğunu değiştirmez. Her token'a verilen olasılık kütlesini değiştirir.

### Top-k Örnekleme

Top-k örnekleme, aday seti en yüksek olasılıklara sahip k jetonlarına sınırlandırır, ardından bu sınırlı setten örnekleri yeniden normalleştirir ve gösterir.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k, sözlük dağılımının uzun kuyruğunda bulunan son derece olası olmayan jetonları (tipos, saçma) seçmekten modelin kaçınıyor. Sorun: k bağlamdan bağımsız olarak sabitlenir. Model güvenlidir (bir jeton 95% olasılıklara sahiptir), k = 40 hala 39 alternatif izin verir.

### Üst-p (Nucleus) Örnekleme

Top-p örneklemesi aday setinin boyutunu dinamik olarak ayarlar.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

Modelle güven olduğu zaman, çekirdek örneklemesi birkaç simgeyi (belki 2-3) tutar.Modelle belirsiz olduğunda, birçok simgeyi (belki 200) tutar. Bu uyarlayıcı davranış, çekirdek örneklemesinin genellikle üst-k'den daha iyi metin üretmesinin nedeni.

**Common combinations:**
- Temperatür 0,7 + üst-p 0,9: İyi genel amaçlı ayarlama
- Sıcaklık 0.0 (açık): Deterministik görevler için en iyi
- Temperatür 1.0 + üst-k 50: Fan et al. (2018) orijinal kağıt ayarları

Top-k ve top-p bir araya gelebilir.

### Reparametreleme hilesi (AVE'lerde kullanılır)

Variasyonel otokodlayıcılar (VAE) girişleri gizli bir alanın bir dağılımına kodlayarak, bu dağılımdan örnek alarak ve örnekten geri kodlama yoluyla öğrenirler.

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

Reparametreleme hilesi rastgeleliği parametrelerden ayırır:

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

Bu, N(mu, sigma^2) ile mu + sigma * N(0,1 arasındaki aynı dağılım nedeniyle çalışır. Anahtar anlayış: rastlantıyı bir parametre dışı kaynağa (epsilon) taşıyarak, sonra örneği parametrelerin farklılaştırılabilir bir dönüşümü olarak ifade edin.

**In the VAE training loop:**
1. Kodlayıcı çıkışları mu ve log(sigma^2) her giriş için
2. Örnek epsilon ~ N(0, 1)
3. Z = mu + sigma * epsilon hesaplayın
4. Girişi yeniden yapılandırmak için z kodunu çöz
5. 4, 3, 2, 1 adımları üzerinden geriye yayılmak (mümkün çünkü adım 3 farklılaştırılabilir)

Reparametrizasyon hilesi olmadan, VAE'ler standart geri yayılma ile eğitilmez.

### Gumbel-Softmax (Farklı Kategorik Örnekleme)

Devamlı dağılımlar için (Gaussian) reparametreleme hilesi işe yarıyor. Ayrı kategorik dağılımlar için farklı bir yaklaşım gerekmektedir. Gumbel-Softmax kategorik örnekleme için farklılaştırılabilir bir yaklaşım sağlar.

**The Gumbel-Max trick (non-differentiable):**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax (differentiable approximation):**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax, ayrı bir örnekin sürekli gevşemesini sağlar. Çıktı sonuç sert bir sıcak yerine bir olasılık vektörü (yumuşak bir sıcak) olur. Gradiyentler yumuşak maksimum üzerinden akıyor. Eğitimde ileri geçiş sırasında, "doğru-önce" tahminçisini kullanabilirsiniz: ileri geçiş için sert argmax kullanın, ancak geri geçiş için yumuşak Gumbel-Softmax gradiyentleri kullanın.

**Applications:**
- VAE'lerde gizli değişkenlerin ayrıntılı olması
- Nöral mimarlık araması (diskret işlemleri seçmek)
- Zor dikkat mekanizmaları
- Ayrı hareketlerle güçlendirme öğrenimi

### Katmanlı Örnekleme

Standart Monte Carlo örneklemesi, örnek alanında rastlantı sonucu boşluklar bırakabilir.

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

Stratifikasyonlu örneklemenin her zaman standart Monte Carlo'ya kıyasla daha düşük veya eşit bir değişimi vardır:

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**Applications:**
- Sayısal entegrasyon (quasi-Monte Carlo)
- Eğitim verileri bölünür (her kattaki sınıf dengesini sağlamak)
- Stratifikasyonla birlikte önemlilik örneği alımı (eki tekniği birleştirmek)
- Neural Radiance Fields (Neural Radiance Fields) kamera ışınları boyunca katlandırılmış örnekleme kullanır

### Diffüzyon Modelleri ile Bağlantı

Diffusion modelleri örnekleme süreci aracılığıyla görüntüler üretir. Ön süreci saf gürültü haline gelene kadar bir görüntüye T adımları üzerinde Gaussian gürültüsü ekler. Ters süreci, adım adım orijinal görüntüyü geri kazanarak denosiyon yapmayı öğrenir.

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

Bu dersdeki yöntemlerle bağlantı:
- Her denoizing adım reparameterizasyon hilesi kullanır (sample gürültüsü, deterministik dönüşüm uygula)
- Gürültü programı {alpha_t} bir tür sıcaklık kaydırma kontrolü sağlar.
- Eğitim, ELBO'yu (gösteriler alt sınır) yakınlaştırmak için Monte Carlo tahminini kullanıyor.
- Diffüzyon modellerinde ata örneklemesi Markov zinciridir (her adım sadece mevcut durumdan bağlıdır)

Tüm görüntü oluşturma süreci tekrarlayıcı örnekleme: gürültüden başlayarak, her adımda öğrenilen denoizing modeline bağlı olarak biraz daha az gürültülü bir versiyon örnekleyin.

```figure
monte-carlo-pi
```

## Yapın

### Adım 1: Teker teker CDF örneği ve ters CDF örneği

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

10.000 eksponensial örnek oluşturun ve ortalamanın 1/lambda olduğunu doğrulayın.

### Adım 2: Reddedilme örneği

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

Kısaltılmış normal bir dağılımdan çekmek için reddedilme örneklemesini kullanın.

### Adım 3: Önemlilik örneği

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

E[X^2]'yi, bir teker teker öneride normal bir dağılım altında tahmin edin. Bilinen yanıtla karşılaştırın (mu^2 + sigma^2).

### Adım 4: Monte Carlo pi tahmin

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Adım 5: Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

İki Gaussian karışımı olan bimodal dağılımdan örnek.

### Adım 6: Gibbs örneği

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Adım 7: Temperatür örneği

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

Bir dizi token logit için sıcaklık çıkış dağılımını nasıl değiştirdiğini göster.

### Adım 8: Üst-k ve üst-p örnekleme

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Adım 9: Reparametreleme hilesi

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

Değişikliklerin yeniden ölçülmüş örnekten geçerken doğrudan örnekleme yoluyla geçer olmadığını göster.

### Adım 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

Düşen sıcaklığın çıkışın bir sıcaktan uzak bir vektöre nasıl yaklaştığını göster.

Tüm görsellemeler ile birlikte tam uygulamalar `code/sampling.py`- Evet .

## Kullan

NumPy ve SciPy ile üretim sürümleri:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

MCMC'nin ölçekli olması için özel kütüphaneler kullanın:
- PyMC: NUTS (adaptif HMC) ile tam Bayesian modelleme
- emcee: MCMC örneği
- NumPyro/JAX: GPU hızlandırılmış MCMC

Şimdi kütüphane aramalarının ne yaptığını biliyorsun.

## Egzersizler

1. Cauchy dağılımında ters CDF örneklemesini uygulayın. CDF F(x) = 0.5 + arctan(x) / pi. 10.000 örnek oluşturun ve histogramı gerçek PDF ile çizin. Ağır kuyruğu (ekstrem değerler merkezden uzak) dikkat edin.

2. Bir Beta ((2, 5) dağıtımından örnekler oluşturmak için Uniform ((0, 1) önerisi kullanın. Kabul edilen örnekleri gerçek Beta PDF ile çizin.

3. Sin ((x) entegralını, Monte Carlo'dan 1000, 10.000 ve 100.000 örnekle 0'dan pi'ye kadar hesaplayın. Her seviyede hata ile karşılaştırın. Hata ölçeğinin O(1/sqrt(N) olduğunu kontrol edin.

4. Metropolis-Hastings'i uygulayın, 2D dağılımından örnek almak için p ((x, y) x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2).

5. Tam bir metin oluşturma demo oluşturun: logitlerle 10 kelimelik bir sözlük verildiğinde, (a) açgözlülük, (b) sıcaklık = 0,7, (c) üst-k = 3, (d) üst-p = 0,9 kullanarak 20 jetonlu bir dizi oluşturun.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Sampling | "Drawing random values" | Generating values according to a probability distribution. The mechanism behind all generative AI |
| Uniform distribution | "All equally likely" | Every value in [a, b] has equal probability density 1/(b-a). The starting point for all sampling methods |
| Inverse CDF | "Probability transform" | F_inverse(U) converts a uniform sample into a sample from any distribution with known CDF. Exact and efficient |
| Rejection sampling | "Propose and accept/reject" | Generate from a simple proposal, accept with probability proportional to target/proposal ratio. Exact but wastes samples |
| Importance sampling | "Reweight samples" | Estimate expectations under p(x) using samples from q(x) by weighting each sample by p(x)/q(x). Core to PPO in RL |
| Monte Carlo | "Average random samples" | Approximate integrals as sample averages. Error O(1/sqrt(N)) regardless of dimension |
| MCMC | "Random walk that converges" | Construct a Markov chain whose stationary distribution is the target. Metropolis-Hastings is the foundational algorithm |
| Metropolis-Hastings | "Accept uphill, sometimes downhill" | Propose moves, accept based on density ratio. Detailed balance ensures convergence to target distribution |
| Gibbs sampling | "One variable at a time" | Update each variable from its conditional distribution holding others fixed. 100% acceptance rate |
| Temperature | "Confidence knob" | Divides logits by T before softmax. T<1 sharpens (more confident), T>1 flattens (more diverse) |
| Top-k sampling | "Keep the k best" | Zero out all but the k highest-probability tokens, renormalize, sample. Fixed candidate set size |
| Nucleus sampling (top-p) | "Keep the probable ones" | Keep the smallest set of tokens whose cumulative probability exceeds p. Adaptive candidate set size |
| Reparameterization trick | "Move randomness outside" | Write z = mu + sigma * epsilon where epsilon ~ N(0,1). Makes sampling differentiable. Essential for VAE training |
| Gumbel-Softmax | "Soft categorical sampling" | Differentiable approximation to categorical sampling using Gumbel noise + softmax with temperature |
| Stratified sampling | "Forced coverage" | Divide sample space into strata, sample from each. Always lower variance than naive Monte Carlo |
| Burn-in | "Warm-up period" | Initial MCMC samples discarded before the chain reaches its stationary distribution |
| Detailed balance | "Reversibility condition" | p(x) * T(x->y) = p(y) * T(y->x). Sufficient condition for p to be the stationary distribution of a Markov chain |
| Diffusion sampling | "Iterative denoising" | Generate data by starting from noise and applying learned denoising steps. Each step is a conditional sampling operation |

## Daha Fazla Okumak

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010)- MCMC temelleri hakkında ayrıntılı dersler
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144)- orijinal Gumbel-Softmax kağıdı
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)- çekirdek (top-p) örnekleme kağıdı
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)- VAE kağıdı, reparametreleme hilesini tanıtan
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)- DDPM örneklemeyi görüntü üretimi ile bağlar
