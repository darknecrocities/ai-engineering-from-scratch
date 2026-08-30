# Doğrudan Tercihleri Optimizasyon Ailesi

> Rafailov et al. (2023) RLHF'nin optimumu tercih verileri açısından kapalı bir biçime sahip olduğunu gösterdi, bu nedenle açık ödül modelini atlayabilir ve politikayı doğrudan optimize edebilirsiniz. Bu anlayış bir aile doğurdu  IPO, KTO, SimPO, ORPO, BPO  her biri DPO'nun başarısızlık modunu düzeltti. 2026 yılında, doğrudan uyum algoritmaları, PPO'dan daha fazla sınır sonrası antrenman yolculuğu gönderir. Ama 2. Dersin aşırı optimize eğri hala geçerlidir: DAA'lar Goodhart'tan kaçmazlar, sadece ısırık olduğu yere hareket ederler.

**Type:** Learn
**Languages:** Python (stdlib, six-variant preference-loss comparator)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking), Phase 10 · 08 (DPO basics)
**Time:** ~75 minutes

## Öğrenme Hedefleri

- DPO kapalı formunu RLHF-KL-optimum'dan çıkarın.
- DPO'daki her bir IPO, KTO, SimPO, ORPO, BPO düzeltmesinin başarısızlık modunu belirtin.
- "İplak ödül boşluğu" ile "Öncelik gücü" arasında fark yapın ve IPO'nun kimlik haritasının neden önemli olduğunu açıklayın.
- Rafailov et al. (NeurIPS 2024) açık bir RM olmamasına rağmen DAAs'ın aşırı optimize olduğunu neden kanıtladığını açıklayın.

## Sorun

RLHF amacı (Desin 1):

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

bilinen bir en iyisi vardır:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

Yani ödül, optimum politika ile referans oranı ile bilinçli olarak tanımlanır:

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

Bunu Bradley-Terry tercih olasılığı ve partisyon fonksiyonuna değiştirin .`Z(x)`Sadece `x`. Geride kalan tek politika parametrelerinde bir kayıp  ödül modeli gerekmiyor.

Korkusu: Kökülme, en iyisini elde edilebilir olduğunu varsayır, tercih verileri dağıtım içindedir ve referans politikası gerçek mod ankeri. Bunlardan hiçbiri tam olarak geçerli değildir.

## Anlaşım

### DPO (Rafailov et al., 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

Yanlış giden ne olabilir:

- - Ödül boşluğu .`beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)`Küçük bir tercih keyfiyle büyük bir boşluk oluşturabilir.
- Kayıp sürücüler seçilen ve reddedilen log-probları karşı yönde hareket ettirir. Seçilen mutlak log-probı reddedilen daha hızlı düşerse aşağıya itirebilir. Bu, Degraded Chosen Response fenomeni.
- Paylaştırılmamış tercihler (kuşkusuz nadir nadir çiftler vs. nadir nadir çiftler) keyfi anlamda bilinçli ödüller üretir.

### Açıklama (Azar et al., 2024)

Kimlik Tercihi Optimizasyonu, log-sigmoid'i tercih olasılığı üzerindeki kimlik haritasıyla değiştirir. Kayıp sınırlı bir hedefte bir kare hatası haline gelir:

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

Marjinin sınırları `1/(2 beta)`-Prifesyon gücü ve içten ödül farkı orantılıdır.

### KTO (Ethayarajh et al., 2024)

Kahneman-Tversky Optimizasyon, çiftlik yapısını tamamen düşürür. Tek bir etiketlenmiş çıkış ve ikili bir "istiği" veya "istiği olmayan" sinyal verildiğinde, bir prospek teorisi kullanımı için haritası yapar:

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

Fayda: çiftsiz verileri kullanabilirsiniz, bu çok daha boldur.

### SimPO (Meng et al., 2024)

Basit Seçenek Optimizasyonu eğitim sinyali ile jenerasyonu uyumlu hale getirir. İpucu politikasını tamamen kaldırın ve uzunlukla log olasılığını normalleştirin:

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

bir kenara sahip`gamma`Uzunluk normallendirme DPO'nun uzunluk-bias başarısızlık modunu kullanma teşvikini ortadan kaldırır (daha uzun `y_w`Yapım yoluyla daha büyük bir log-prob boşluğu verir).

### ORPO (Hong et al., 2024)

Odds-Ratio Preference Optimization, standart SFT negatif kayıt olasılığına bir tercih terimi ekler:

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

Referans politikası yok  SFT terimi düzenleyici. Üssü modelden uyumlu modeline tek bir aşamada tren. Ayrı bir SFT kontrol noktası yoktur.

### BPO (ICLR 2026 başvurusu, OpenReview id=b97EwMUWu7)

Degraded Selected Response sorunu belirler: DPO sıralamayı korur `y_w > y_l`Ama mutlak log-prob `y_w`BPO, seçilen cevapta aşağıya doğru hareketleri cezalandıran tek satırlı bir düzeltme ekler. Llama-3.1-8B-DPO'ya göre matematik akıl yürütme konusunda +10.1% doğruluk rapor edildi.

### Evrensel sonuç: DAA'lar hala aşırı optimize

Rafailov et al. "Direct Alignment Algorithms'te Ödül Modelinin Aşırı Optimizasyonu için Ölçekleme Kanunları" (NeurIPS 2024) DPO, IPO, KL bütçelerindeki birden fazla veri kümesi üzerine politika eğitimi aldı. Altın ödülleri vs. KL eğrilikleri aynı Gao et al. zirve ve çöküş şekline sahiptir.

DAAs Goodhart'tan kaçmaz. "Önemli olarak optimize edilen ödül modeli"nden "Önemli olarak optimize edilen referans politika oranı"na geçiyor.

### Seçimler (2026)

- Büyük çiftli tercih verileriniz varsa: DPO, koruyucu beta ile, SimPO uzunluk kayıtsızlığı belirginse.
- Eğer çiftsiz ikili geri bildiriminiz varsa: KTO.
- Eğer bir basamak modelesinden tek aşamalı bir boru hattı istiyorsanız: ORPO.
- Eğer DPO kayıtlarında seçilmiş kayıt araştırmalarının bozulduğunu görürseniz: BPO.
- Eğer tercih güçleri çok farklı ve DPO doymuşsa: IPO.

Her laboratuvar beş tane de bir pille çalışır ve her görev için kazananı seçer.

```figure
dpo-margin
```

## Kullan

`code/main.py`Oyuncak tercih verisi, gerçek tercih gücü çiftlere göre değişir. Her kayıp, küçük bir softmax politikası ile aynı 500 çift örneğine göre optimize edilir.

## Gönder

Bu ders bize çok yararlı .`outputs/skill-preference-loss-selector.md`. Verita set istatistikleri (bir çiftle eşleşmemiş, değişkenle eşleşmiş tercih gücü, uzunluk dağılımı) ve bir hedef (tek aşama veya SFT-den sonra tercih) göz önüne alındığında, tercih kaybını önerin ve bu durumun önüne geçirilmesi gereken başarısızlık modunu bildirin.

## Egzersizler

1. Çık .`code/main.py`. DPO ve BPO için son seçilen kayıt sorgu düşüşünü bildirin. BPO seçilen mutlak olasılıkları daha yüksek tutmalıdır  bunu doğrulayın.

2. Bu nedenle, bu iki yöntemin en güçlü olanı ve en düşük dereceli olanını belirtmek için, bir çiftin aynı kuvvetli olmasını sağlayan bir preferans verisini değiştirin.

3. Yönülmüş yanıtları seçileninden ortalama 2 kat daha uzun yapın.

4. Rafailov et al. (NeurIPS 2024) DAAs'ın aşırı optimize olduğunu iddia ediyor. Tek nokta bir versiyonu üretmek: plan seçilen-minus reddedilen KL farklılığı ve büyük beta'da DPO'da aşırı optimize gözlemlemek.

5. BPO kağıdı özetini okuyun (OpenReview b97EwMUWu7).`code/main.py`- Evet .

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| DPO | "RLHF without a reward model" | Loss derived from the closed-form RLHF optimum; policy parameters only |
| Implicit reward | "the log-ratio" | `beta * log(pi(y\|x) / pi_ref(y\|x))` — the DPO-implied reward |
| IPO | "bounded DPO" | Replaces log-sigmoid with identity; implicit reward gap capped by `1/(2 beta)` |
| KTO | "unpaired DPO" | Prospect-theory utility over single labels with loss aversion |
| SimPO | "reference-free DPO" | Length-normalized log-likelihood + margin; no reference policy |
| ORPO | "one-stage DPO" | NLL + odds-ratio preference term; trains from base model in one pass |
| BPO | "chosen-preserving DPO" | DPO plus a penalty for decreasing the chosen response's absolute log-prob |
| Degraded Chosen | "chosen goes down" | DPO decreases chosen log-prob so long as rejected falls faster |
| DAA | "direct alignment algorithm" | Any preference-loss method that skips an explicit RM |

## Daha Fazla Okumak

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) Açıklama
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
