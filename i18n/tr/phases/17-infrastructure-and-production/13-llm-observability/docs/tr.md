# LLM Gözlem Gücü Stak Seçimi

> 2026 gözlemlenebilirlik pazarı iki kategoriye ayrılır. Gelişim platformları (LangSmith, Langfuse, Comet Opik) değerlendirmeler, hızlı yönetim, seans tekrarlamalarıyla birlikte izlemeyi birleştirir. Gateway/instrumentasyon araçları (Helicone, SigNoz, OpenLLMetry, Phoenix) telemetri üzerine odaklanmaktadır. Langfuse, güçlü OSS dengesi (50K etkinlik / ay ücretsiz bulut) ile MIT lisanslı bir çekirdektir. Phoenix, Elastic License 2.0 altında OpenTelemetry- doğuşlu, sürükleme / RAG görselleştirme için mükemmel, sürekli bir üretim arka planı değil. Arize AX, monolit gözlemsellikten 100 kat daha ucuz olduğunu iddia eden sıfır kopyalı Iceberg/Parquet entegrasyonunu kullanıyor. LangSmith, LangChain/LangGraph için liderlik ediyor, 39 $ / user / mo, sadece Enterprise'da kendi kendine barındırıyor. Helicone 15-30 dakikalık ayarlama ile proxy tabanlı, 100K req/mo ücretsiz, ama ajan izleri daha az derinlik. Ortak üretim tarzı: Gateway (Helicone/Portkey) + eval platformu (Phoenix/TruLens) OpenTelemetry ile yapıştırılmış.

**Type:** Learn
**Languages:** Python (stdlib, toy trace-sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Öğrenme Hedefleri

- Gelişme platformlarını (bundled: evals + prompt + sessions) gateway/telemetry araçlarından (sadece izler + metrikler) ayırt edin.
- Altı ana aletin (Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik) lisanslama, fiyatlandırma ve tatlı nokta kullanım durumlarına haritasını yapın.
- Bir geçit aracı ile ayrı bir değerleme platformu birleştirmenizi sağlayan OpenTelemetry-kalem örneğini açıklayın.
- 2026 maliyet farkçısını (Arize AX'in sıfır kopya yaklaşımı vs. monolit alımı) isimlendirin ve kaba 100x katlayıcısını belirtin.

## Sorun

Bir LLM özelliği göndermişsiniz. Çalışıyor. Hemen başarısızlık, araç döngüleri, gecikme gerilemeleri, maliyet zirvesi veya hızlı önbelleğe ulaşma oranına dair hiçbir görünürlüğünüz yok. "LLM gözlemliliği" Google'a girin ve sekiz araç elde edin. Hepsi aynı sorunu üç farklı fiyat noktasında çözeceklerini iddia eder.

LangSmith, "Bu LangGraph çalışması neden başarısız oldu?" diye cevap verir. Phoenix, "RAG borumu akıntı mı?" diye cevap verir. Helicone, "Hangi uygulama jetonları yakıyor?" diye cevap verir. Langfuse, "Bütünü kendiliğimle otirebilir miyim?" diye cevap verir.

Seçim dört etkisi içerir: yığın (LangChain? çiğ SDK? çok satıcı?), lisans toleransı (sadece MIT? Elastik Tamam mı? ticari ceza mı?), bütçe ( ücretsiz seviyede? $100/mo? $1000/ay?), ve kendi kendine ev sahibi (heç zaman mı?

## Anlaşım

### İki kategori

**Development platforms**Bu, bir dizi deney çalıştırmak, hangi bir istek işe yaradı, hangi bir istek geri dönüşü, eski kazananlara karşı yeni bir istek. LangSmith, Langfuse, Comet Opik.

**Gateway/telemetry tools**Bu yöntemler,  prompt, response, tokens, latency, model, cost. Helicone, SigNoz, OpenLLMetry, Phoenix. Minimalist. OpenTelemetry üzerinden ayrı bir değerleme aracı ile birleştirilebilir.

### Langfuse  OSS dengesi

- Core Apache / MIT lisanslı; Docker üzerinden kendi kendine barındırma.
- Bulut ücretsiz seviyesi: ayda 50 bin etkinlik.
- Evals, hızlı yönetim, izler, veri kümeleri.
- Tatlı nokta: LangSmith sınıfı özellikleri istiyorsanız ama kendi kendini barındırmak veya OSS lisansı ile kalmak zorundasınız.

### Phoenix (Arize)  Telemetri-birincisi, OpenTelemetry-native

- Elastik Lisans 2.0; kendi kendine ev sahibi önemsiz.
- RAG ve drift görselleştirme konusunda mükemmel.
- Sürekli üretim arka uç olarak tasarlanmamış  öncelikle gelişme zamanında gözlemlenebilirlik.
- Tatlı nokta: RAG boru hattı geliştirme, drift debugging, üretim için ayrı bir kapı ile çiftleştirme.

### Arize AX  ölçek oyunu

- Ticari, buzberg/parket üzerinden sıfır kopyalı veri gölü entegrasyonu.
- S3'te kendi Parket'inizde izleri sakladığınız için matematik, Arize'nin doğrudan okuduğu bir şey.
- Tatlı nokta: Günde > 10M iz, mevcut veri gölü, DATADOG fiyatlandırması olmadan LLM özel araç çubuğu istiyor.

### LangSmith  LangChain/LangGraph önce

- Ticari, 39 dolar, aylık kullanıcı.
- LangChain ve LangGraph yığınları için sınıfında en iyi.
- Tatlı nokta: LangChain'e bağlı bir ekip, ödeme yapmaya hazır.

### Helicone  Proxy tabanlı minimum uygulanabilir

- 15-30 dakika ayarlayarak değiştir .`OPENAI_API_BASE`Helicone vekili.
- MIT lisanslı; 100 bin dolar ücretsiz, 20 dolar ücretli.
- Bu, failover, caching, rate limitleri de içerir.
- Ajan / çok adımlı izler üzerinde daha az derinlik.
- Tatlı nokta: hızlı başlatma, tek yığınlı uygulama, bir kapı + gözlemliliğe ihtiyaç duyar.

### Opik (Comet)  OSS geliştirme platformu

- Apache 2.0, tamamen OSS.
- Langfuse'nin Komet mirasına benzer bir özellik.
- Şimdiden Comet'te olan ML takımları aynı panelde LLM gözlemini istiyor.

### SigNoz  OpenTelemetry-first full APM

- Apache 2.0. OpenTelemetry üzerinden genel APM ve LLM ile ilgileniyor.
- Sweet spot: hizmetler ve LLM çağrıları arasında tek bir gözlemsellik.

### Yapıştırıcı: OpenTelemetry + GenAI semantik konvansiyonları

OpenTelemetry, 2025'in sonlarında GenAI semantik sözleşmelerini yayınladı (`gen_ai.system`- Evet .`gen_ai.request.model`- Evet .`gen_ai.usage.input_tokens`Otel tüketen araçlar birbirine etkileşime girebilir.

1. Her LLM görüşmesinden GenAI'nin toplantılarını yayınlayın.
2. Gündüzlü olarak kapıya giden yol (Helicone / Portkey).
3. Geri dönüşler için ikili gemi değerlendirme platformu (Phoenix / Langfuse).
4. Arize AX veya DuckDB üzerinden uzun süreli analiz için veri gölünde (Iceberg) arşiv.

### Tuzak: Yanlış katman üzerinde alet kullanmak

Ajan çerçevesinin içinde araç kullanmak (örneğin LangSmith izlerini eklemek) sizi bu çerçeveye bağlar. HTTP/OpenAI-SDK katmanında (OpenLLMetry veya geçitinizle) araç kullanmak taşınabilir.

### Örnek almak  her şeyi saklayamazsın

Günlük > 1M talep, tam iz tutma LLM çağrılarından daha fazla maliyetlidir. Kurallar ile örnek: 100% hata, 100% yüksek maliyet, 5% başarı. Toplantıları her zaman tutun; uzun kuyruğu için ham tutun.

### Hatırlamalısın numaralar

- Langfuse ücretsiz bulut: Ayda 50 bin olay.
- LangSmith: 39 dolar / ay.
- Helikon ücretsiz: ayda 100 bin.
- Arize AX iddiası: ~ 100 kat daha ucuz monolit ölçekte.
- OpenTelemetry GenAI Sözleşmeleri: 2025 nakliye, 2026 geniş çapta kabul edildi.

```figure
i4-otel-glue
```

## Kullan

`code/main.py`%100'lik tüketim, örnekleme, örnekleme + hatalar) depolama maliyetini ve her bir altta ne kaybolduklarını rapor eder.

## Gönder

Bu ders bize çok yararlı .`outputs/skill-observability-stack.md`.Skala, bütçe, lisans duruşu göz önüne alındığında, araçları seçer.

## Egzersizler

1. LangChain'deki ekibiniz OSS'in kendi kendine konutlanmış gözlemselliğini istiyor.
2. 5M izleri / gün ile Datadog ayda 150K $ teklifler, hesaplama Arize AX için eşitlik kırıklığı.
3. OpenTelemetry GenAI özelliğini oluşturun.
4. Phoenix'in üretime yeterli olup olmadığını tartışın.
5. Helicone 20 ms proxy overhead. P99 TTFT 300 ms, kabul edilebilir mi?

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| OpenLLMetry | "OTel for LLMs" | Open-source OpenTelemetry instrumentation for LLMs |
| GenAI conventions | "OTel attributes" | Standard OTel attribute names for LLM calls |
| LangSmith | "LangChain observability" | Commercial platform bundled with LangChain ecosystem |
| Langfuse | "OSS LangSmith" | MIT OSS with similar feature set |
| Phoenix | "Arize dev tool" | OpenTelemetry-native dev/eval platform |
| Arize AX | "scale observability" | Commercial zero-copy Iceberg/Parquet observability |
| Helicone | "proxy observability" | HTTP proxy collecting LLM telemetry + gateway features |
| Opik | "Comet LLM" | Apache 2.0 OSS dev platform from Comet |
| Session replay | "trace rerun" | Replay a full agent session with tool calls |
| Eval | "offline test" | Running candidate model/prompt over labeled dataset |

## Daha Fazla Okumak

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
