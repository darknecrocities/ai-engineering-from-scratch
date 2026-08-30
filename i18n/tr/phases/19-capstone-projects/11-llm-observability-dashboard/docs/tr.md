# Capstone 11  LLM Gözlem ve Eval Dashboard

> Langfuse açık çekirdeğe girdi. Arize Phoenix 2026 GenAI semconv haritalarını yayınladı. Helicone ve Braintrust her iki kullanıcı başına maliyet oranını ikiye katladı. Traceloop'un OpenLLMetry'si gerçekte SDK aletleri haline geldi. Üretim şekli izler için ClickHouse, metadata için Postgres, UI için Next.js ve örnek izler üzerinde çalışan değerlendirme işlerinin küçük bir ordusu (DeepEval, RAGAS, LLM-cömert) dir. Kendi kendine barındırılan bir tane oluşturun, en az dört SDK ailesinden alın ve enjekte edilen bir gerilemeyi beş dakikadan az bir süre içinde yakalamanın kanıtını gösterin.

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:**P11 · P13 · P17 · P18
**Time:** 25 hours

## Sorun

2026'da üretim trafiğini yürüten her AI ekibi modelin yanında gözlemsellik düzeni tutar. Masraflar. Halüsinasyon algısı. Akıntı izleme. - Cezaevi açma sinyali. SLO arac tablosu. PII sızdırma uyarıları. Açık kaynaklı referanslar  Langfuse, Phoenix, OpenLLMetry , OpenTelemetry GenAI semantik konvensiyonlarında enfet şeması olarak birleşmiştir. Şimdi OpenAI, Anthropic, Google, LangChain, LlamaIndex ve vLLM araçlarını bir SDK ile gönderebilir ve uyumlu uzantıları gönderebilirsiniz.

En az dört SDK ailesinden gelen, örneklenmiş izler üzerinde küçük bir değerlendirme çalışmaları, sürüklemeyi tespit eden ve uyarılar üzerinde çalışan kendi kendine barındırılan bir araç tablosunu oluşturacaksınız. Ölçüm çubuğu: kasıtlı olarak enjekte edilen bir geri dönüş (PII üretmeye başlayan bir istek) verildiğinde, araç tablosu onu yakalar ve beş dakikadan az bir süre içinde bir uyarı ateşler.

## Anlam

Ingest OTLP HTTP. SDK GenAI-semconv uzantıları üretir: `gen_ai.system`- Evet .`gen_ai.request.model`- Evet .`gen_ai.usage.input_tokens`- Evet .`gen_ai.response.id`- Evet .`llm.prompts`- Evet .`llm.completions`. Sütun analitikleri için ClickHouse'da yer alır; metadata (kullanıcılar, seanslar, uygulamalar) Postgres'te yer alır.

Eval'ler örneklenmiş izlere göre bir seri olarak çalıştırılır. DeepEval sadakat, toksisite ve cevap bağlamını notlar. RAGAS iz geri alım metriklerini iz geri alım bağlamını taşıdığında notlar. Özel LLM yargıçları alan özel kontrolleri (PII sızması, politika dışı yanıt) yürütür. Eval çalışmalar ana izle bağlantılı değerlendirme aralıkları olarak aynı ClickHouse'a yazır.

Drift algılama saatleri zaman içinde yerleşim- uzay dağılımlarını (sürekli yerleşimlerde PSI veya KL farklılığı) artı değerlendirme puanı eğilimlerini izler. Alarmlar Prometheus Alertmanager ve ardından Slack / PagerDuty ile beslenir. UI Next.js 15 Recharts ile.

## Mimarlık

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## Yüküm

- İçebilirlik: OpenTelemetry SDK + GenAI semantik sözleşmeleri; OTLP HTTP taşımacılığı
- Toplayıcı: İpuç örneği işlemcisi olan OpenTelemetry Toplayıcı (maliyet kontrolü için)
- Kayıt: ClickHouse için süreler, Metadata için Postgres, S3 için çiğ olay arşivleri
- Evals: DeepEval, RAGAS 0.2, Arize Phoenix değerlendirici paketi, özel LLM yargıç
- Aktarım: PSI / KL haftalık toplu hızlı yerleştirmeler (cümle dönüştürücüler)
- Alarm: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + sunucu eylemleri
- Kutunun dışında desteklenen SDK'lar: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

```figure
ce-otel-drift
```

## Yapın

1. **Collector config.**OTLP HTTP alıcısı, %100 hata izini ve %10 başarıyı koruyan bir kuyruğu örneği ve ClickHouse ve S3'e ihracatçı olan OpenTelemetry Collector.

2. **ClickHouse schema.**Tablo`spans`GenAI semconv'i yansıtan sütunlarla: `gen_ai_system`- Evet .`gen_ai_request_model`- Evet .`input_tokens`- Evet .`output_tokens`- Evet .`latency_ms`- Evet .`prompt_hash`- Evet .`trace_id`- Evet .`parent_span_id`, uzun payloadlar için JSON çanta ekle. kullanıcı_id ve app_id ile ikincici indeksler ekle.

3. **SDK coverage test.**Her SDK (OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM) ile OpenLLMetry otomatik enstrümanı ile küçük bir istemci uygulaması yazın. Her birinin ClickHouse'a yerleştirilen kanonik GenAI uzantıları ürettiğini doğrulayın.

4. **Eval jobs.**Bir programlı iş son 15 dakikalık örnek izlerini okuyor ve DeepEval sadakat, toksisite ve cevap bağlamını çalıştırır.

5. **Custom LLM-judge.**Bir PII sızdırma yargıç: bir cevap verildiğinde, PII sızdırma olasılığını puanlamak için bir gardiyan LLM çağırın. Yüksek puanlı cevaplar bir triage kuyrukuna yerleştirilir.

6. **Drift detection.**Haftalık iş, bu haftaki toplu istek yerleştirmeler ile sonraki 4 haftalık başlangıç arasında PSI'yi hesaplar.

7. **Dashboard.**Next.js 15 sayfaları: genel bakış (span/sec, maliyet/kullanıcı, p95 gecikme), izler (arayan + su düşüşü), değerlendirmeler (davranlık eğilimleri, toksisite), sürükleme (zamanla PSI), uyarılar.

8. **Alerting chain.**Prometheus ihracatçısı eval puan toplamlarını ve gecikme yüzdelerini okuyor; Alertmanager uyarılar için Slack ve kritik ihlal için PagerDuty yollarını okuyor.

9. **Regression probe.**Bir hata enjekte edin: değerlendirilmiş chatbot% 1'de sahte SSN sızmaya başlar. MTTR ölçün: deployed bug Slack uyarısı.

## Kullan

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## Gönder

`outputs/skill-llm-observability.md`LLM uygulaması verildiğinde, ara çubuğu izlerini yutar, değerlendirmeleri, sürükleme uyarıları yürütür ve Next.js'te maliyet/kullanıcı dağılımı yüzeylerini açar.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Trace-schema coverage | Number of SDK families producing canonical GenAI spans (target: 6+) |
| 20 | Eval correctness | DeepEval / RAGAS scores vs hand-labeled set |
| 20 | Dashboard UX | MTTR on injected regression (under 5 minutes target) |
| 20 | Cost / scale | Sustained ingest at 1k spans/sec without backlog |
| 15 | Alerting + drift detection | Prometheus/Alertmanager chain exercised end to end |
| **100** | | |

## Egzersizler

1. Haystack çerçevesine özel aletler ekleyin.`gen_ai.*`- Bu özellikler.

2. Aynı izlerde Phoenix değerlendirici için DeepEval'i değiştirin.

3. Sürüm algılayıcısını keskinleştir: uygulama kimliği yerine, genel olarak PSI hesaplayın.

4. "Kullanıcı etkisi" sayfasını ekleyin: kullanıcı başına maliyet ve kullanıcı başına başarısızlık oranı, parlaklık çizgilerle.

5. %100 toksisite ile % 0,5'den fazla olan izlerin ve geri kalanın %10'u stratifiyen bir örnekle birlikte %100'i koruyan bir kuyruğu örnekleme politikası oluşturun.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM attributes" | 2025 OpenTelemetry spec for LLM span attributes (system, model, tokens) |
| Tail sampling | "Post-trace sample" | Collector decides to keep or drop a trace after it completes (can peek errors) |
| PSI | "Population stability index" | Drift metric comparing two distributions; > 0.2 typically signals meaningful drift |
| LLM-judge | "Eval as model" | An LLM scoring another LLM's output on a rubric (faithfulness, toxicity, PII) |
| Tail-sampling policy | "Keep-rule" | Rule that decides which traces to persist vs drop; errored + sample-rate |
| Eval span | "Linked eval trace" | Child span carrying an eval score linked to the original LLM call span |
| Cost per user | "Unit economics" | Dollar cost attributed to a user_id over a window; key product metric |

## Daha Fazla Okumak

- [Langfuse](https://github.com/langfuse/langfuse) Açık çekirdek gözlemleme platformu
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) Güçlü akıntı desteği ile alternatif referans
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) Otomatik aletleşme SDK ailesi
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) içme şeması
- [Helicone](https://www.helicone.ai) alternatif ev sahipliği yapılmış gözlemsellik
- [Braintrust](https://www.braintrust.dev) alternatif eval-first platform
- [ClickHouse documentation](https://clickhouse.com/docs) sütun uzantısı depolama
- [DeepEval](https://github.com/confident-ai/deepeval) değerlendirici kütüphanesi
