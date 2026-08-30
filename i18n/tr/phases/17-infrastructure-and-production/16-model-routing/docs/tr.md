# Model Routing, Masrafları Azaltacak Bir Primitif

> Dinamik bir broker her talebi değerlendirir (tüm görev türü, token uzunluğu, yerleştirme benzerliği, güven) ve basit sorguları ucuz bir modele gönderir, karmaşık sorguları sınır modeline yükselir. Model kaskadı olarak da adlandırılır. Üretim vaka çalışmaları, ABD/İngiltere/AB dağıtımları arasında iso-kalite'de maliyetlerin %20-60 oranında azalması göstermektedir; yüksek hacmi SaaS'de %30 yönlendirme verimliliğinin iyileştirilmesi, yıllık altı rakamlı tasarruf haline gelir. 2026 bağlamında LLM sonucu fiyatları yılda ~ 10 kat düştü  bir GPT-4 sınıfı token gitti $20/M to ~$2022'nin sonundan 2026'ya kadar 0.40/M. Çoğu düşüş, donanım değil, daha iyi servis halinde (Fase 17 · 04-09), Routing, fiyat düşüşünü ürün gerilemesi olmadan marj olarak dönüştürmenin yoludur. Başarısızlık modusu ucuz model sürüşüdür: rota %40'ı zayıf bir modele doğru itiyor, kalitesi akıl yürütme görevlerinde %3-5% düşüyor, kimse bir çeyrek bile fark etmez. Geçit yolları çevrimiçi kalite ölçümleri ile, sadece çevrimiçi değerleme setleri ile değil.

**Type:** Learn
**Languages:** Python (stdlib, toy cascading router simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 19 (AI Gateways)
**Time:** ~60 minutes

## Öğrenme Hedefleri

- Modelin kaskadını açıklayın: güven kontrolü ile ucuz-birincisi, düşük güven üzerinde yükseliş.
- Dört yönlendirme sinyali (tüm görev sınıflandırması, hızlı uzunluk, bilinen sert set ile benzerliği yerleştirmek, ilk geçişten itibaren kendine güven)
- Hedef yönlendirme bölünmesi ve kalite kaybı toleransında beklenen karışık maliyetin hesaplanması.
- Ucuz model sürüklemeyi takip eden metrik (online kalite kapısı) isimlendirin.

## Sorun

Servisiniz GPT'de ayda 80 bin dolar masraf ediyor.Analytikleriniz, soruların %70'inin basit olduğunu gösteriyor: "Paris'te saat kaç?" "Bu cümleyi yeniden ifade et". Haiku sınıfı modeli, maliyetin %3'ünde bunları mükemmel şekilde ele alır. %30'a GPT-5'in mantıklılığı gerek  kodlama, matematik, çok adımlı planlama.

Eğer %70'i ucuz ve %30'u pahalı bir şekilde yönlendirseniz, faturalarınız aynı ürün kalitesiyle %65 oranında düşer. Bu yönlendirme.

## Anlaşım

### Dört yönlendirme sinyali

1. **Task classification**: basit/ karmaşık/ kodegen/ matematik/ sohbet. Kurallara dayalı bir sınıflandırıcı, küçük bir LLM (Haiku sınıfı $ 0.25 / M) veya etiketlenmiş kovalara benzerliği yerleştirmek olabilir.

2. **Prompt length**Bu nedenle, bu durumun bir diğer nedeni de, bu durumun da bir diğer nedeni de olabilir.

3. **Embedding similarity to known-hard set**: sorgu bilinen sert bir kova yakınsa (kosine > 0,88) doğrudan sınırlara tırmanın.

4. **Self-confidence from first-pass**: ucuz gönderin; modelin log-probleri düşük güven gösterirse YA da reddederse YA da koruma dilini çıkarırsa, sınırda yeniden deneyin. Trafikin %10'unda P95 gecikme oranını artırır, diğer %90'da ise %50+ tasarruf sağlar.

### Üç örneği

**Pre-route**(telif önde): ~ 5-10 ms gecikme eklenmiştir; genel olarak en hızlı.

**Cascade**(incil olarak ucuz, düşük güvenle yükselir): ~1.2x ortalama gecikme ( ucuz çalıştırma artı doğrulama), ~2x yükselir. En iyi kalite zemin.

**Ensemble route**(çizgi ve sınır paralel olarak çalıştırılır, ödül model seçilir): en yüksek kalitede, en yüksek maliyette; yalnızca kritik A/B için kullanılır.

### Uygulama

AI geçitleri (Fase 17 · 19) yönlendirmeyi ortaya çıkarır. LiteLLM `router`Portkey'in gardiyanları + yönlendirme. Kong AI Gateway'in eklenti tabanlı yönlendirme. OpenRouter'ın model pazarı bir önerme API'si ortaya çıkarır.

Açık kaynaklı: RouteLLM (LMSYS), Diamond değil (ticari), Prompt Mule.

### 2026 fiyat eğri

| Model class | Late 2022 | 2026 | Change |
|-------------|-----------|------|--------|
| GPT-4-level quality | ~$20/M | ~$0.40/M | 50x cheaper |
| Frontier (GPT-5, Claude 4) | — | ~$3-10/M | new tier |

En büyük gelişme verimliliği hizmet vermektir. 17 · 04-09 Eğlence'deki temel dersler sağlayıcı tarafındaki maliyet düşüşüne dönüştü. Routing tüm kullanıcılarınızın ucuz seviyeye taşınmasını beklemek yerine uygulama katmanında bu kazançları yakalamanıza olanak sağlar.

### - Drift gerçek bir risk.

Routeniz ucuz model için %40 gönderir. Altı ay içinde görev dağılımları değişir (kullanıcılar daha karmaşık hale gelir, daha uzun soru sorarlar). Router fark etmez çünkü sınıflandırıcısı Q1 verilerine eğitim almıştır. Kalite sessiz düşer. Kimse yeterince yüksek sesle şikayet etmez. Rakip benchmarkında kaybettiğini öğreniyorsun.

Online kalite ölçümleri ile geçit yolları:

- Kullanıcılar yol boyunca parmak parmaklarını yukarı / aşağıya kaldırır.
- Yol başına bir örnek (5%) üzerinde otomatik LLM yargıcı.
- Aşama oranı: Kaskadada %30'luk bir yükselme oranı varsa ucuz model aşırı yönlendiriliyor.
- Yol başına reddedilme oranı.

### Hatırlamalısın numaralar

- 2026 yönlendirme tasarrufu iso-kalitede: 20-60% vaka çalışmaları.
- LLM fiyatlarının düşüşü 2022-2026: ~ 10 kat yıllık toplam.
- GPT-4 seviyesinde 2022 vs. 2026: ~$20/M → ~$0.40/M.
- Kaskadalı gecikme etkisi: ~1.2x ortalama, ~2x yükseldi (trafikin ~ 10%).

```figure
model-cascade-router
```

## Kullan

`code/main.py`-Küçük maliyet, kalite kaybı ve artış oranı raporları.

## Gönder

Bu ders bize çok yararlı .`outputs/skill-router-plan.md`- İş yükü ve kalite bütçesi göz önüne alındığında, bir yönlendirme örneği ve sinyalleri seçer.

## Egzersizler

1. Çık .`code/main.py`Kaskadın önceden giden yolları hangi düzeyde geçer?
2. Kullanıcı tabanınız %30 işletme (mükemmel sorular), %70 ücretsiz seviyedir (sadece).
3. Bir yol kalitesi %2 düşer ama %40 tasarruf eder.
4. OpenAI / Anthropic API'lerden logprobs kullanarak güven kontrolü uygulayın.
5. Altı ay içinde, tırmanma oranı %8'den %22'e yükseldi.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Model routing | "cost broker" | Dynamic choice of model per request |
| Model cascade | "cheap-first escalate" | Run cheap, fall through to frontier on low confidence |
| Pre-route | "classify first" | Classifier up front; no re-run |
| Ensemble route | "parallel pick" | Run multiple, reward-model picks best |
| Escalation rate | "uprouted %" | Fraction of cascade requests that escalated |
| RouteLLM | "LMSYS router" | OSS router library |
| Not Diamond | "commercial router" | SaaS model-routing product |
| Drift | "cheap creep" | Distribution shift without router noticing |
| Online quality gate | "live check" | Automated LLM-judge sampling live traffic |

## Daha Fazla Okumak

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) Routing primitipleri ile çoklu model geçit.
