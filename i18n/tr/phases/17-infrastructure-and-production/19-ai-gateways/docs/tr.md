# AI Gateways  LiteLLM, Portkey, Kong AI Gateway, Bifrost

> Uygulamalarınız ve model sağlayıcılarınız arasında bir geçit yer alır. Temel özellikler sunucu yönlendirme, geri dönüş, tekrar deneme, hız sınırlaması, gizli referanslar, gözlemlenebilirlik, korumalar. 2026'da piyasa bölünmesi: **LiteLLM**OpenAI ile uyumlu olan 100'den fazla sağlayıcı ile MIT OSS'dir, ancak yaklaşık 2000 RPS (8 GB bellek, yayınlanan referans değerlerinde kaskadaki hatalar) arasında ayrılır; Python için en iyi, <500 RPS, dev/prototipleme. **Portkey**kontrol düzeni konumlandırılmış (güvenlikler, PII redaksiyonu, jailbreak algılama, denetim izleri), Apache 2.0 açık kaynak Mart 2026'da 20-40 ms gecikme overhead gitti, $49/mo production tier. **Kong AI Gateway** built on Kong Gateway — Kong's own benchmark on same 12 CPUs: 228% faster than Portkey, 859% faster than LiteLLM; $Model/ay fiyatı 100 (Plus seviyesinde maksimum 5); Kong'da zaten varsa işletme için uygun. **Bifrost**(Maxim AI)  Otomatik geri dönüş, OpenAI 429'da Anthropic'e geri dönmek. **Cloudflare / Vercel AI Gateways** yönetilen, sıfır operasyonlar, temel yeniden deneme. Veriler konumu kendi kendine konukseverlik kararını yönlendirir; Portkey ve Kong, OSS + seçmeli yönetilen ortada oturuyor.

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 minutes

## Öğrenme Hedefleri

- Altı temel geçit özelliklerini (routing, fallback, retry, hız limitleri, sırlar, gözlemlenebilirlik, koruma) listeleyin.
- Çatılar ve kullanım durumları ölçeklendirilirken 2026'da dört geçit (LiteLLM, Portkey, Kong AI, Bifrost) haritasını yapın.
- Kong referans değerini alıntılayın (228% vs Portkey, 859% vs LiteLLM) ve neden >500 RPS için önemli olduğunu açıklayın.
- Verilerin konut ve operasyon bütçesi verildiği için kendi kendine barındırılan vs yönetilen seçin.

## Sorun

Ürününüz OpenAI, Anthropic ve kendi kendine barındırılan Llama'yı çağırır. Her sağlayıcı farklı bir SDK, hata modeli, oran sınırı ve ot programına sahiptir. Başarısızlık yapmak istiyorsunuz (OpenAI 429'lar varsa, Anthropic'i deneyin), tek bir kredileme mağazası, tek bir gözlemsellik ve kiracı başına oran sınırı.

Bu uygulama katmanında yeniden icat etmek her hizmeti her sağlayıcıya eşleştirir. Bir geçit katmanı, hizmeti sağlayıcılara yayarak bir API ile (genellikle OpenAI uyumlu) bir süreçte birleştirir.

## Anlaşım

### Altı temel özellik

1. **Provider routing** OpenAI, Anthropic, Gemini, kendi kendine barındırılan, vb.
2. **Fallback**429, 5xx'te veya kalite başarısızlığı, başka bir yerde tekrar deneyin.
3. **Retries** Eksponansiyel geri dönüş, sınırlı girişimler.
4. **Rate limits** Kiracı başına, anahtar başına, model başına.
5. **Secret references** Kullanım zamanı (birde uygulamada) güvenirlik bilgileri kasadan çek.
6. **Observability** OTel + GenAI özellikleri (Fase 17 · 13) + maliyet atributları.
7. **Guardrails** PII düzenleme, jailbreak algılama, izin verilen konular filtreleri.

### LiteLLM  MIT OSS, Python

- 100+ sağlayıcı, OpenAI uyumlu, yönlendirme yapılandırması, geri dönüş, temel gözlemsellik.
- Kong'un referans değerinde 2000 RPS'nin kırılması; 8 GB hafıza ayak izleri, sürekli yük altında kaskadaki başarısızlıklar.
- En iyi uyum: Python uygulaması, <500 RPS, dev/staging geçitleri, deneysel yönlendirme.
- Fiyat: OSS için 0 dolar; bulutsuz bir kat var.

### Portkey  kontrol uçağının konumlandırılması

- Apache 2.0 OSS Mart 2026 tarihli olarak.
- 20-40 ms talep başına gecikme maliyeti.
- 49 $ / ay üretim seviyesine, tutma + SLA ile.
- En uygun: korumalara ihtiyaç duyan düzenlenmiş endüstriler + gözlemsellik birleştirilmiş.

### Kong AI Gateway  ölçek oyunu

- Kong Gateway üzerinde inşa edilmiş (olgun API geçit ürün, lua+OpenResty).
- Kong'un kendi 12 CPU eşdeğerindeki referans değerleri: Portkey'den 228% daha hızlı, LiteLLM'den 859% daha hızlı.
- Fiyat: 100 dolar / model / ay, maksimum 5 Plus seviyesinde.
- En iyi uygunluk: Kong'da zaten; > 1000 RPS; lisans almak için hazır.

### Bifrost (Maxim AI)

- Otomatik bir şekilde yeniden çalıştırma.
- OpenAI 429'da Anthropic'e geri dönmek, kanonik bir tarif.
- Yeni giriş; ticari.

### Cloudflare AI Gateway / Vercel AI Gateway

- Başarılı, sıfır operasyon, temel yeniden deneme ve gözlemsellik.
- En iyi uyum: Cloudflare/Vercel'de Edge hizmet veren JavaScript uygulamaları.
- Kong/Portkey'e kıyasla koruma ve hız sınırları için sınırlı.

### Kendi kendine barındırılan vs yönetilen

Veri ikametciliği zorlayıcı işlevdir. Sağlık ve finans öntanımlı kendi-host (LiteLLM veya Portkey OSS veya Kong). İsteğe bağlı olarak yönetilen tüketici ürünleri (Cloudflare AI Gateway) veya orta seviye (Portkey yönetilen).

### Gecikme bütçesi

- LiteLLM: 5-15 ms genel yük tipik.
- Portkey: 20-40 ms üst.
- Kong: 3-8 ms üst.
- Cloudflare/Vercel: 1-3 ms genel maliyet (geçer avantajı).

Gateway gecikmesi doğrudan TTFT'ye eklenir. TTFT P99 < 100 ms SLA, Kong veya Cloudflare için. P99 < 500 ms için, herhangi bir.

### Sınır sınırı semantik meselesi

Basit token-bucket orta ölçekte çalışır. Çoklu kiracı sürükleyici pencere + patlama izin + kiracı başına bir katlama gerektirir. LiteLLM token-bucket gemileri; Kong gemileri sürükleyici pencere; Portkey gemileri katlandırılmış.

### Geçit + gözlemlenebilirlik + yönlendirme oluştur

17 · 13 aşaması (gözleyicilik) + 16 (modelleme yönlendirme) + 19 (kapı) aynı üretim katmanıdır. Üçünü de kaplayan bir araç seçin veya dikkatlice telleştirin: 2026'daki çoğu dağıtım Helicone (gözleyicilik) veya Portkey (kuzen) ile Kong (skala) bölünmüş roller için birleştirir.

### Hatırlamalısın numaralar

- LiteLLM: 2000 RPS'de kırılır, 8 GB bellek.
- Portkey: 20-40 ms üst düzey; Apache 2.0 Mart 2026'dan beri.
- Kong: Portkey'den 228% daha hızlı, LiteLLM'den 859% daha hızlı.
- Kong fiyatı: $ 100 / model / ay, 5 maksimum Plus seviyesinde.
- Cloudflare/Vercel: 1-3 ms uçta.

```figure
mx-gateway-fallback
```

## Kullan

`code/main.py`429/5xx enjeksiyon altında 3 sunucu arasında geri dönüşle geçit yönlendirmeyi simüle eder. Gecikme, yeniden deneme oranı ve geri dönüş çarpma oranını rapor eder.

## Gönder

Bu ders bize çok yararlı .`outputs/skill-gateway-picker.md`Ölçek, operasyon pozisyonu, uyumluluk, gecikme bütçesi göz önüne alındığında, bir kapı seçer.

## Egzersizler

1. Çık .`code/main.py`. OpenAI→Anthropic→self-hosted'den geri dönüş ayarlayın. % 5 servisçi hata oranında beklenen hit oranı nedir?
2. SLA'nın TTFT P99 < 200 ms'dir. 300 ms'lik bir başlangıç çizgisinde. Hangi geçitler bütçenin içinde kalır?
3. Sağlık hizmetleri için kendi kendine konutlama + kişisel bilgi düzenleme + denetim gerekir. Portkey OSS veya Kong'u seçin.
4. LiteLLM vs Kong'u karşılaştırın: Bir takım hangi RPS tavanına göç etmeli?
5. Çok kiracı SaaS için ücret sınırlama politikası tasarlayın: ücretsiz, deneme, ücretli seviyeler. Token-bucket veya kaydırıcı pencere?

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Gateway | "API broker" | Process sitting between apps and providers |
| LiteLLM | "the MIT one" | Python OSS, 100+ providers, breaks at 2K RPS |
| Portkey | "guardrails gateway" | Control plane + observability, Apache 2.0 |
| Kong AI Gateway | "the scale one" | Built on Kong Gateway, benchmark leader |
| Bifrost | "Maxim's gateway" | Retries + Anthropic fallback recipe |
| Cloudflare AI Gateway | "edge managed" | Edge-deployed managed gateway, zero-ops |
| PII redaction | "data scrub" | Regex + NER mask before sending to model |
| Jailbreak detection | "prompt injection guard" | Classifier on user input |
| Audit trail | "regulated log" | Immutable record of every LLM call |
| Token-bucket | "simple rate limit" | Refill-based rate limiter |
| Sliding-window | "precise rate limit" | Time-windowed rate limiter; better fairness |

## Daha Fazla Okumak

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
