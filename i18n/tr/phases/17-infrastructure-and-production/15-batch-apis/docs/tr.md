# Satış API'leri  Endüstri Standartı olarak %50 indirim

> Her büyük sağlayıcı %50 indirim ve ~24 saatlik dönüşüm ile async parti API'si gönderir. OpenAI, Anthropic, Google ve çoğu sonuçlama platformu (Fireworks parti seviyesinde, Together parti) aynı kalıpı uyguluyor. Hızlı önbelleğe alınan ve gece geçici boru hattı olan yığın seri, sinkron-yüklenmemiş maliyetin %10'una düşer. Kural çok basit: Eğer etkileşimsizse, partiye ait. İçerik üretimi boruları, belge sınıflandırması, veri çıkarımı, rapor üretimi, toplu etiketleme, katalog etiketleme  24 saat gecikmeye toleranslı olan her şey, toplamaya geçene kadar masada kalan para. 2026 üretim tarzı, her yeni LLM çalışma yükünün üç dizine ayrılmasıdır: etkileşimli (memleketle eşzamanlı), yarı etkileşimli (sinkron olmayan sırada geri dönüş), seri (gece, önbelleğe girilen giriş yığılmış). İnteraktifmiş gibi davranan ama dakikalarca gecikme süreci geçiren iş yükleri en çok harcanır.

**Type:** Learn
**Languages:** Python (stdlib, toy batch-vs-sync cost simulator)
**Prerequisites:** Phase 17 · 14 (Prompt & Semantic Caching)
**Time:** ~45 minutes

## Öğrenme Hedefleri

- Üç tedarikçi parti API'si (OpenAI, Anthropic, Google) ve ortak %50 indirim + 24 saatlik dönüş garantilerini isimlendirin.
- Gecelik sınıflandırma iş yükü üzerinde bir seriyi yığma + önbelleğe girilen giriş maliyetini hesaplayın ve senkronizasyon-önceltilmemiş başlangıç seviyesine karşılaştırın.
- Bir iş yükünü etkileşimli / yarı etkileşimli / partiye ayırıp, şeridi haklı çıkarın.
- İki tuzağı isimlendirin: kısmi etkileşim (kullanıcı 24 saatten daha hızlı bekliyor) ve çıkış şeması süresi (batch dosya biçimi sağlayıcıya göre farklıdır).

## Sorun

Ekibiniz her gece rapor üretimi hattı gönderir. 50.000 belge, her birini özetleyin, özetleri gruplandırın, bir yönetim kurulu raporu hazırlayın. Sinkron çalıştırmak, gece başına 2.000 dolar için 4 saat sürer.

Bu paketle %50 indirim elde ediyorsunuz. Sistem istekleri üzerinde de önbelleğe girme işlemini etkinleştirirsiniz (bütün 50k aramalarda paylaşılan).

Bu nedenle, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temelinde, bu programın temel olarak, bu programın temelinde, bu programın temel olarak, bu programın temelinde, bu programın temel olarak, bu programın temel olarak, bu programın temel olarak, bu programın temel olarak, bu programın temel olarak, bu programın temel olarak, bu programın temel olarak, bu programın temel olarak, bu programın temel olarak, bu programın temel olarak, bu programın temel olarak, bu programın temel olarak, bu programın temel olarak, bu programın temel olarak, bu programın temel olarak, bu programın temel olarak, bu programın temel olarak, bu programın, bu programın temel olarak, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu programın, bu na bağlı olarak, bu programın, bu programın, bu programın, bu programın, bu na bağlı olarak, "İstanbul, "İstanbul, "İstanbul, "İstanbul, "İstanbul, "İstanbul, "İstanbul, "İstanbul, "İstanbul, "İstanbul, "İstanbul, a.

## Anlaşım

### Üç parti API'si

**OpenAI Batch API**JSONL dosyası yüklenmesi, talepler listesine sahip. 24 saatlik dönüşü söz verilir (genellikle pratikte ~ 2-8 saat). Girme ve çıkış tokenlerinde %50 indirim. `/v1/batches`Kaynaklı girişler de önbelleğe girme fiyatlandırmasını üstlenir.

**Anthropic Message Batches**JSONL yükleme, 24 saatlik dönüş, %50 indirim.`cache_control` önbelleğe yazılar açıkça, okumalar otomatik olarak seri içinde gerçekleşir.

**Google Vertex AI Batch Prediction**BigQuery veya GCS girişleri. Gemini için benzer %50 indirim. Vertex boru hattlarıyla entegre.

### Semantik: Asinkron, yavaş değil

Satış "24 saat içinde geri döneceğime söz veriyorum"  değil "Bu 24 saat sürecek". Tipik P50 2-6 saat.

### Önbelleği ile dolu

Aynı 4K token sistemi ile 50k belge özetleme:

- Sinkron kaydedilmemiş: 50000 × ($input × 4000 + $Çıktı × 200) tam hızlarda.
- Sinkron önbelleğe alınan: sistem tesisi ilk yazımdan sonra önbelleğe alınır; kalan 49999'ün 10 kat daha ucuz giriş elde edilir.
- Parça önbelleğe alınmış: yukarıdaki tümü artı okuma ve yazma için %50 indirim.

Satır: parti + önbelleğe = %10 eşzamanlı önbelleğe alınmamış faturanın. Gece boyunca çalışan ve paylaşılan bir sistem uyarısı olan herhangi bir iş yükü bunu kullanmalıdır.

### İş yükü sınıflandırması

**Interactive** kullanıcı cevap bekliyor. TTFT önemli. Hızlı önbelleğe sahip eşzamanlı arama.

**Semi-interactive** kullanıcı bir görev gönderir, dakika içinde geri kontrol eder. Batch mevcut değilse senkronize etmek için fallback ile asynk kuyruk. Orta çaplı RAG indeksleme düşünün.

**Batch** kullanıcı sonuçları "sabah" veya "kötü saat"e kadar bekler. İçerik boruları, ölçekte sınıflandırma, çevrimdışı analiz. Her zaman seri, her zaman yığın önbelleği.

Genel hata: her şeyi etkileşimli olarak sınıflandırmak çünkü boru hattı üretimdir. Üretim bir gecikme speçikası değildir  SLA.

### Bölümsel etkileşim tuzağı

Bazı özellikler etkileşimli görünse de 5-10 dakika tolerantlık gösterir. Örnek: "Yenileştir" düğmesi ile gecelik bir müşteri sağlık raporu. Kullanıcı yenilgiye tıklıyor; 10 dakika beklemek iyidir. Takım onu eşzamanlı olarak gönderir. 50 eşzamanlı yenilgiye e-posta yoluyla toplanan ve teslim edilen maliyetin 10 katı maliyetini verir.

Eğer cevap "eğer fark etmezlerse"se, "24 saat bu kullanıcı için ne anlama gelir?" diye sormak için bir soru sor.

### Çıktı-sema tuzağı

Satış dosya biçimleri, sunucuya göre farklıdır:

- JSONL, her satırda bir talep.
- Antropik: JSONL, her satırda bir mesaj; cevap biçimi yerleştirilmiştir.
- Vertex: BigQuery tablo veya TFRecord ile GCS önlüğü.

"Bir seri istemcisi" yazmak, her bir sağlayıcı için adaptör kodunu ifade eder. Çoklu sağlayıcı seriyi reklamlayan geçitler (Portkey, LiteLLM bazı seviyeler) hala ham biçimi incelikle sarar.

### Hatırlamalısın numaralar

- Satış sağlayıcıları arasında parti indirme indirim + çıkış oranı %50 oranında sabit.
- Dönüşüm SLA: 24 saat garanti, 2-6 saat tipik P50.
- Yüklü seri + önbelleğe alınan giriş: Sinkronize edilmemiş önbelleğe alınan maliyetin %10'u.
- İş yükü sıralama kuralı: 24 saat gecikme kabul edilebilirse, her zaman seri.

```figure
batch-lane-triage
```

## Kullan

`code/main.py`50k belge iş yükü için senkronize, senkronize + kas, parti ve parti + kas maliyetlerini hesaplar.

## Gönder

Bu ders bize çok yararlı .`outputs/skill-batch-triager.md`. İş yükü özelliklerini göz önüne alarak, etkileşimli/yarı/batch'a ayırılır ve tasarruf tahmin edilir.

## Egzersizler

1. Çık .`code/main.py`. 3K-token sistem istekleri ve 500-token çıkışı ile 100k-doc boru hattı için, tam yığın (batch + cache) vs. senkronize tabanının tasarrufu hesaplanmalıdır.
2. Tanıdığınız gerçek bir ürünün üç özelliğini seçin.
3. Bir kullanıcı raporlarının 3 saat sürdüğünden şikayet ediyor.
4. Bu durumun üstü, kapalı, kapalı ve kapalı sistemlerin üzerinde nasıl hareket ettiğini belirler.
5. Hesaplama dengesi: Paylaşılan prefiks uzunluğunda, batch + cache, kendi rezerve GPU'da bir gecede çalışmaktan daha ucuz olur.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Batch API | "async discount" | 50% off with 24h turnaround |
| JSONL | "batch format" | One JSON request per line; OpenAI/Anthropic standard |
| Message Batches | "Anthropic batch" | Anthropic's batch API product name |
| Batch prediction | "Vertex batch" | Vertex AI's batch API product |
| Turnaround SLA | "24h promise" | Guarantee, not typical; typical is 2-6h |
| Workload triage | "interactivity decision" | Interactive / semi / batch routing decision |
| Output schema | "response format" | Per-provider JSONL layout; not portable |
| Stacked discount | "batch + cache" | ~10% of uncached sync bill when both apply |

## Daha Fazla Okumak

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) JSONL biçimi ve `/v1/batches`- Semantik.
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) parti biçimi ve `cache_control`etkileşim.
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini)Gemini parti semantikası.
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
