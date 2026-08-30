# Hızlı Kaşlama ve Semantik Kaşlama Ekonomisi

> **Pricing snapshot dated 2026-04.**Aşağıdaki sayısal iddialar bu dersin yayınlandığı sırada kaydedilen satıcı oran kartlarını yansıtır; aşağıdaki değeri aktarmadan önce bağlantılı belgelere karşı doğrulayın.

> Önbelleğe kaydetme iki katmanla gerçekleşir. L2 (sunducu düzeyinde) önbelleğe / önbelleğe kaydetme tekrarlanan önbellekler için dikkat KV'yi tekrarlar  Anthropic'in önbelleğe kaydetme belgeleri uzun çağrılarda %90'a kadar maliyet azaltımı ve %85 gecikme azaltımı ile reklam yapmaktadır. Claude 3.5 Sonnet önbelleği okumaları $0.30/M vs $3.00/M taze, 5 dakikalık TTL ve 1 saatlik TTL seçeneği için 2 kez yazma primosu ile (docs.anthropic.com, 2026-04). OpenAI prompt caching, istekler için otomatik olarak geçerlidir ≥1024 token ve fiyatlar cached giriş yaklaşık %90 indirim vs. taze (platform.openai.com, 2026-04); model başına tam cached oran canlı oran kartına bağlıdır. L1 (app düzeyinde) semantik önbelleği, LLM'yi tamamen benzerlik hitlerini yerleştirmekle atlıyor. Satıcı "95% doğruluk" eşleşme doğruluğunu ifade eder, hit oranı değil  rapor edilen üretim hit oranları %10 (açık sohbet) ile %70 (strukturel FAQ) arasında değişir; hiçbir sağlayıcı resmi bir temel çizgi yayınlamaz, bu nedenle bunları garanti yerine topluluk telemetrisi olarak değerlendirin. Üretim tuzağı: paralelleşme önbelleği öldürür (birinci önbelleğe yazmadan önce verilen N paralel istekler harcamaları birkaç kat artırabilir) ve önleme içindeki dinamik içeriğin önbelleğin vurulmasını tamamen engeller. ProjectDiscovery, kaydedilebilir önbellekten dinamik metin taşımakla %7'den %74'lik bir hit oranına (2025-11) geçiş yapıldığını bildirdi.

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## Öğrenme Hedefleri

- L2 önbellek/önbellek önbellek (providerde KV yeniden kullanımı) ile L1 semantik önbellek (tıpkı bu gibi önbelleklerde LLM bypass) arasında ayrım yapın.
- Antropik'in ne olduğunu açıkla `cache_control`açık bir işaretleme ve iki TTL seçeneği (5-min vs. 1 saat) fiyat çarpıcılarıyla.
- Hits oranı, prompt/response mix ve token fiyatları ile beklenen aylık tasarruf hesaplayın.
- Paralelleşme karşıtı örneği, faturaları 5-10 kat yükseltir ve çarpma oranını düşüren dinamik içeriğe karşı örneği.

## Sorun

RAG hizmetinize prompt caching eklersiniz. Hesap sabit kalır. Çıkış oranını ölçersiniz; %7'dir. İstekleriniz statik görünüyor ama değil  Sistem istekleri, dakikaya biçimlendirilmiş mevcut tarihini, bir istek kimliğini ve çeşitlilik için rastgele bir örnek yeniden düzenlemesini içerir. Her istek yeni bir cache girişini yazar, sıfır okuyor.

Bu nedenle, bu işlemler, bir kullanıcı sorusu başına 10 paralel araç çağrısı yapar. On kişi de ilk önbelleğe yazma tamamlanmadan önce sunucuya ulaşır. On kişi yazar, sıfır okur. Hesabınız "önbelleğe kaydetme" ile maliyetinin 5-10 katı.

Önbelleğe alma bir protokol, bir bayrak değil.

## Anlaşım

### L2  sağlayıcı önbellek/önbellek önbellek

Sağlayıcı dikkat KV'sini bir önbellek için saklar ve önbellekle eşleşen bir sonraki talepte tekrar kullanır.

**Anthropic (Claude 3.5 / 3.7 / 4 series)**Açıkça`cache_control`TTL: 5 dakika (yazma maliyeti 1.25x baz) veya 1 saat (yazma maliyeti 2x baz).$0.30/M on Claude 3.5 Sonnet vs $3.00/M taze  10 kat daha ucuz (docs.anthropic.com, 2026-04) Fiyatlar model başına değişir (Opus/Haiku ayrı olarak yayınlanmıştır); canlı fiyatlandırma sayfasını her zaman çapraz olarak kontrol edin.

**OpenAI**Gpt-4o/gpt-5 oran kartlarında önbelleğe girme işlemleri: önbelleğe girme işlemleri: (platform.openai.com, 2026-04) 1024 token için otomatik önbelleğe girme işlemleri. Açık bir bayrak yoktur. Önbelleğe girme işlemleri mevcut gpt-4o/gpt-5 oran kartlarında taze olanlardan yaklaşık 10 kat daha ucuz. Ne belge ne de açıklama notları resmi bir hit-rate temel çizgisini yayınlamaktadır. Topluluk raporları dikkatli bir önbelleğe sahip olarak yaklaşık 3060%'a gruplanır.`usage.cached_tokens`Kendi gücünü ölçmek için.

**Google (Gemini)**: açık bir API üzerinden bağlam önbelleği; 1M-token bağlamı önbelleği daha da fazla ödeme anlamına gelir.

**Self-hosted (vLLM, SGLang)**: 17 · 06 aşaması RadixAttention  kendi hesaplamalarınızda aynı kalıpları kapsar.

### L1  uygulama düzeyinde semantik önbelleğe kaydetme

LLM'yi aramazdan önce, isteklenmeyi hash edin, yerleştirin ve benzer bir önbelleğe alınmış talebi (sözde 0.95+'in üzerinde eşsiz benzerlik) arayın.

Açık kaynaklı: Redis vektör benzerliği, GPTCache, Qdrant. Ticari: Portkey Cache, Helicone Cache.

Satıcı doğruluk iddiaları, geri gönderilen önbelleğe kaydedilen yanıtın semantik olarak ne kadar sıklıkla doğru olduğunu gösterir.

- Açık uçlu sohbet: 10-15%.
- Yapılandırılmış Soru sorusu / destek: 40-70%.
- Kod sorular: 20-30% (küçük çeşitler vurguları öldürür).
- Sesli ajanlar tekrarlama çağrıları: 50-80% (sessi normallaştırma sabit seti).

### Paralelleşme karşıtı örneği

Ajanınız 10 araç çağrısını paralel olarak yapar. 10'unun hepsi aynı 4K-token sistem uyarısına sahiptir. Antropik önbelleği yazıları istek başına; ilk önbelleği yazma, sağlayıcı istekleri gördükten sonra yaklaşık 300 ms tamamlanır. 2-10 istek aynı milisaniye penceresinde gelir ve her bir önbelleği eksik görür. 10 yazma primini ödersiniz, 0 okuma indirimini.

Düzeltme: sıradan ilk  ile parti tek başına 1 talebi yapın, sonra 1'in önbelleği doldurulduğunda 2-10'u ateşleyin. İlk araç çağrısına 300 ms ekler; faturanın 5-10 katını kaydeder.

### Dinamik içeriği anti-önemli

Sistem istasyonunuz şöyle görünüyor:

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

Her istek eşsiz, her istek yazıyor.

Düzeltme: gerçekten statik olan her şeyi cache edilebilir önbelleklere taşı; cache sınırının ardından dinamik içeriği ekle:

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery bu şekilde %7'den %74'e kayıp hızı geçirdi ve anatomiyi yayınladı.

### Gecelik iş yükleri için toplu seri + önbelleği

Satır API'leri (Fase 17 · 15) 24 saatlik dönüşümde %50 indirim sağlar. Önbelleğe girilen giriş, bunun üzerinde ~ 10 kat daha fazla elde eder. Gece içi sınıflandırma, etiketleme ve rapor üretimi iş yükleri, yığılımı yoluyla sinkron-yığılmamış maliyetin ~ 10%'ine düşebilir.

### Hatırlamalısın numaralar

Fiyat noktaları 2026-04'te bağlantılı satıcı belgeleri üzerinden ele alınır ve birkaç ayda bir  tekrar kontrol edilir.

- Antropik önbelleğe alınan okuma: Claude 3.5 Sonnet'te 0.30 $/M, taze girişten yaklaşık 10 kat daha ucuz (docs.anthropic.com).
- Antropik önbelleği yazma primü: 1.25x (5 dakikalık TTL) veya 2x (1 saatlik TTL).
- OpenAI otomatik önbelleği: ≥1024 token için geçerlidir; mevcut oran kartlarında yeni girişlerin yaklaşık% 10'u karşılığında önbelleğe girilen giriş (platform.openai.com).
- Semantik önbelleği isabet oranı (halk tarafından bildirilmiş): ~ 10% açık sohbet; ~ 70% yapılandırılmış Soru sorusu. Satıcı belgelemiş bir temel çizgi değil.
- ProjectDiscovery: %7 → %74 hit oranı, dinamikleri öntanımdan çıkararak (proyect blog, 2025-11).
- Paralelleşme karşıtı örneği: N paralel istekler ilk önbelleği yazmayı kaçırırken tipik 510x fatura enflasyonu raporları.

```figure
semantic-cache-hit
```

## Kullan

`code/main.py`Raporlar oranları, faturaları ve paralellik cezasını gösterir.

## Gönder

Bu ders bize çok yararlı .`outputs/skill-cache-auditor.md`. Hızlı bir şablon ve trafik göz önüne alındığında, önbelleğe alınma kabiliyetini denetler ve yeniden yapılandırmayı önerir.

## Egzersizler

1. Çık .`code/main.py`Paralelleşme bayrağını değiştir.
2. Sistem sorgulamanın bir tarihi var.
3. Arama gelme oranını göz önüne alarak 1 saatlik TTL (2x yazmak) vs 5 dakikalık TTL (1.25x yazmak) için eşitlik hesaplayın.
4. 0.95 eşiğinde semantik önbelleğe %20 ulaşır. 0.85'de %50'e ulaşır ama yanlış önbelleğe alınan yanıtları görürsünüz. Doğru eşiği seçin ve haklı gösterin.
5. Kullanıcı sorusu başına 10 paralel alt sorgu toplarsınız.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| L2 prompt cache | "prefix cache" | Provider stores KV for repeated prefix |
| `cache_control` | "Anthropic cache marker" | Explicit attribute marking cacheable blocks |
| Cache write premium | "write tax" | Extra cost for first miss-to-cache (1.25x or 2x) |
| L1 semantic cache | "embedding cache" | App-level hash-and-embed before calling LLM |
| GPTCache | "LLM caching lib" | Popular OSS L1 cache library |
| Cache hit rate | "hits / total" | Fraction of requests served from cache |
| Parallelization anti-pattern | "the N-write trap" | N parallel requests miss cache N times |
| Dynamic content trap | "the time-in-prompt trap" | Dynamic bytes in prefix kill hit rate |
| RadixAttention | "intra-replica cache" | SGLang's prefix-cache implementation |

## Daha Fazla Okumak

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) resmi `cache_control`semantik ve TTL'ler.
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) otomatik önbelleğe kaydetme davranışı ve uygunluk.
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
