# Araç Kullanımı ve İşlem Aramaları

> Toolformer (Schick et al., 2023) kendi kendine denetimli araç notasyonu başlattı. Berkeley Fonksiyon Çağrı Liderboard V4 (Patil et al., 2025) 2026 barını belirler: 40% ajantik, 30% çok dönüş, 10% canlı, 10% canlı olmayan, 10% halüsinasyon. Tek dönüş çözülür. Hatıra, dinamik karar verme ve uzun ufuklı araç zincirleri değildir.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 minutes

## Öğrenme Hedefleri

- Toolformer'ın kendi kendine denetimli eğitim sinyali açıklayın: araç notlarını ancak icracılık bir sonraki jeton kaybını azaltırsa tutun.
- BFCL V4'in beş değerlendirme kategorisini ve her birinin ölçümlerini belirtin.
- Şema doğrulama, argüman zorlaması ve yürütme sandboxing ile stdlib araç kayıtlarını uygulayın.
- 2026'da açık olan üç sorunu teşhis edin: uzun uzayda araç zinciri, dinamik karar verme ve hafıza.

## Sorun

İlk araç kullanımı soruldu: model doğru bir fonksiyon çağrısını tahmin edebilir mi? Modern araç kullanımı sorar: model zincirli araçlar 40 adım boyunca, bellek ile, kısmi gözlemlenebilirlik ile, araç kusurlarından kurtulmak ile var olmayan aletleri halüsinasyon yapmadan olabilir mi?

Toolformer, temel çizgiyi oluşturdu: modeller araçları ne zaman çağrılacağını kendiliğinden denetleme ile öğrenebilir. BFCL V4 2026 değerlendirme hedefini tanımlar.

## Anlaşım

### Toolformer (Schick et al., NeurIPS 2023)

Fikir: modelin kendi antrenman öncesi korpusunu aday API çağrılarıyla not etmesine izin verin. Her aday için, onu yürütün. Notasyonu yalnızca araç sonucu eklenmesiyle bir sonraki token'daki kayıpları azaltırsa tutun. Filtreli korpus'ta ince ayarlama yapın.

Altınlanmış araçlar: hesap makinesi, kalite sistemleri, arama motorları, tercüman, takvim.

Skala sonucu: araç kullanımı ölçekte ortaya çıkar. Küçük modeller araç notasyonlarından zarar görür; büyük modeller kazanır. Bu nedenle 2026 sınır modelleri güçlü araç kullanımı pişirmişken çoğu 7B modeli güvenilir olmak için açıkça araç kullanımı ince ayarlamalara ihtiyaç duyar.

### Berkeley Fonksiyon Çağrıları Liderboard V4 (Patil et al., ICML 2025)

BFCL, 2026'da gerçek değerlendirilmiş bir değerlendirme.

- **Agentic (40%)** Tam ajan yolları: hafıza, çok dönüşlü, dinamik kararlar.
- **Multi-Turn (30%)** Araç zincirleri ile etkileşimli konuşmalar.
- **Live (10%)** Kullanıcı tarafından gönderilen gerçek istekler (kısıtlama daha zor).
- **Non-Live (10%)** sentetik test vakaları.
- **Hallucination (10%)** Hiçbir araç çağırılmaması gerektiğinde tespit et.

V3 durum tabanlı değerlendirmeyi tanıttı: bir araç dizisinden sonra, araç çağrılarının AST'sine eşleşmek yerine API'nin gerçek durumunu kontrol edin (örneğin "fayl yaratıldı mı?"). V4 web arama, bellek ve format hassasiyet kategorilerini ekledi.

Anahtar 2026 bulgu: tek dönüş fonksiyon çağrısı neredeyse çözülmüştür. Başarısızlıklar hafıza (turnuvlar boyunca bağlam taşıma), dinamik karar verme (önceki sonuçlara dayanarak araç seçme), uzun uzayda zincirler (20+ adımdan sonra sürükleme) ve halüsinasyon algısı (bir araç uyumsuz olduğunda arama reddetmek) odaklanır.

### Araç Şeması

Her sağlayıcı bir şema sahiptir. Detaylarda farklılık gösterirler ama aynı şekli paylaşırlar:

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Antropik kullanımları `input_schema`Bu, OpenAI'nin doğrudan kullanımı.`function.parameters`. Her ikisi de JSON Şema'yı kabul eder. Açıklamalar yük taşıyıcıdır  model doğru aracı seçmek için okuyor. Kötü araç açıklamaları yanlış araç seçilen başarısızlıkların ilk temel nedeni.

### Düzgünleştirme

Kullanıcı çağrısına güvenme.

1. **Type coercion.**Model, şema int diyen "5" bir dizinini geri verebilir. Açıkça belirlenirse zorlayın; değilse reddedin.
2. **Enum validation.**Eğer şema şöyle diyorsa`status in {"open", "closed"}`ve model emisyonları `"in_progress"`, açıklama hatası ile reddet.
3. **Required fields.**Gerekli alan eksik -> hemen hata gözlemini modeline geri döndürmek, bir kaza değil.
4. **Format validation.**Tarihler, e-postalar, URL'ler  regex değil beton parserlerle doğrulanır.

Her onay başarısızlığı, modelin doğru şekliyle tekrar denemesinin için yapılandırılmış bir gözlem göndermelidir.

### Paralel araç çağrıları

Modern sağlayıcılar paralel araç çağrılarını bir asistan dönüşüyle destekler.

1. Model 3 araç çağrısı ile farklı `tool_use_id`S.
2. Çalışma zamanı onları (bağımsızsa paralel olarak) gerçekleştirir.
3. Her sonuç bir `tool_result`blok ile ilişkili`tool_use_id`- Evet .

Mühendislik kuralı: ilişki kimliklerini yük taşıyan olarak değerlendirin. Onları değiştirin ve yanlış araçtan yanlış sonuçlara yönlendirme elde edin.

### Kum kutucu

Araç işlevleri kum kutu sınırıdır. Detay için 09 dersine bakın. Kısa sürüm: her araç okuma/yazma yüzeyini, ağ erişimini, zaman sonunu, hafıza kapısını belirtmelidir. Genel `run_shell(cmd)`Kırmızı bayraklı;`git_status()`Daha güvenli.

```figure
tool-routing
```

## Yapın

`code/main.py`üretim biçimindeki araçlar listesini uyguluyor:

- JSON Schema alt kümesi onaylayıcı (sdlib sadece).
- Araç kaydı, giriş şeması, zaman sonlaması ve yürütücü ile.
- Dolayısıyla, bu bir gerçek.
- İlişki kimlikleri ile paralel araç gönderimi.
- Yapılandırılmış dizileri olarak hata gözlemleri.

Çek şunu:

```
python3 code/main.py
```

İz bir mini ajanı üç alet bir sırada çağıracak ve bir modelin hareket edebileceği bir açıklama hatası ile reddedilen kasıtlı bir yanlış çağrı ile göstermektedir.

## Kullan

Her sağlayıcı kendi araç şeması  Anthropic, OpenAI, Gemini, Bedrock. Çoklu sağlayıcıya ihtiyacınız varsa bir çeviri katmanı kullanın.

## Gönder

`outputs/skill-tool-registry.md`Verilen görev alanı için bir araç katalogunu, şeması ve kayıt yapılır. Açıklama kalitesi kontrollerini içerir (her bir araçın açıklaması ne zaman kullanılacağını modeline söyler mi?).

## Egzersizler

1. Modelin başka bir araç kullanmayı açıkça reddetmesine izin veren "no-op" bir araç ekleyin.
2. İçeride bir ip ve akışta bir ip olarak zorlama uygulamak.
3. Bir araç başına bir zaman kesimi ve bir devrim kesicisi ekleyin (tümel 3 ardıcıl başarısızlıktan sonra 60 yıl boyunca kullanmayı reddedin).
4. BFCL V4 açıklamasını okuyun. Bir kategoriyi seçin (örneğin "çok dönüş") ve ajanınıza 10 örnek istek gönderin.
5. Pydantic/Zod'un oyuncakta kaçırdığı neyi yakaladı?

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Function calling | "Tool use" | Structured-output tool invocation with validated schema |
| Toolformer | "Self-supervised tool annotation" | Schick 2023 — keep tool calls whose results reduce next-token loss |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 benchmark: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination |
| Tool schema | "Function signature for the model" | name, description, JSON Schema of arguments |
| tool_use_id | "Correlation ID" | Ties a tool call to its result; essential for parallel dispatch |
| Hallucination detection | "Know when not to call" | V4 category: refuse to call when no tool fits |
| Argument coercion | "String-to-int repair" | Narrow fixes for predictable schema-mismatch; reject if ambiguous |
| Sandboxing | "Tool execution boundary" | Per-tool read/write surface, network, timeout, memory cap |

## Daha Fazla Okumak

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) Kendiliğinden denetim gören araç notasyonu
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) 2026 değerlendirme referansı
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) Claude Agent SDK'de üretim araç şeması
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) fonksiyon araç türü ve Guardrails
