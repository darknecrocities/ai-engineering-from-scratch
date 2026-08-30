# Araç Aracı  Neden Ajanların Yapılandırılmış İ/İ işleme ihtiyacı var

> Bir dil modeli jetonlar üretir. Bir program eylemler yapar. Bu ikisi arasındaki boşluk araç arayüzüdür: modelin bir eylem istemesine ve barındırmacı tarafından gerçekleştirilmesine izin veren bir sözleşme. 2026'da her yığın  OpenAI, Anthropic ve Gemini'yi çağıran fonksiyon; MCP'nin `tools/call`; A2A'nın görev parçaları  aynı dört adımlı döngünün farklı bir kodlamasıdır. Bu ders döngüye isim verir ve onu çalıştırmak için en az makineyi gösterir.

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11 (LLM completion APIs)
**Time:** ~45 minutes

## Öğrenme Hedefleri

- Sadece metin oluşturabilen bir LLM'nin kendi başına gerçek dünyaya karşı neden eylemler yapamayacağını açıklayın.
- Dört adımlı araç çağrısı döngüsünü çizin (describe → decide → execute → observe) ve her adımın sahibi kim olduğunu söyleyin.
- Bir araç açıklamasını üç bölüm olarak yazın: isim, JSON Schema girişi ve belirleyici bir yürütücü fonksiyonu.
- Saf ve yan etkileri olan araçları ayırt edin ve bölünmenin neden güvenliğe önem verdiğini belirtin.

## Sorun

Bir LLM bir sonraki token üzerinde bir olasılık dağılımını yayar. Bu tüm çıkış yüzeyi. Eğer bir sohbet modeline "Şu anda Bengaluru'da hava ne durumda" diye sorarsanız, "onlar olası bir cümle yazabilir, ancak hava API'ye diyal edemez. cümle tesadüf veya üç gün eskiden doğru olabilir.

Bu boşluğu kapatmak, araç arayüzünün amacıdır. Ev sahibi programı  ajanın çalıştırma zamanı, Claude Desktop, ChatGPT, Cursor veya özel bir senaryo  model için çağrılabilir araçların bir listesini reklam eder. Modelle, bir eylemin gerekli olduğuna karar verdiğinde, bir aracı ve argümanlarını adlandıran yapılandırılmış bir pay yükü yayar. Ev sahibi yükü analiz eder, aracı gerçekleştiriyor ve sonuçları geri gönderir. Modelle daha fazla arama gerekmediğine karar verene kadar döngü devam eder.

Bu sözleşmenin ilk versiyonu OpenAI'nin "fonksiyonlar" parametresi olarak Haziran 2023'te gönderildi.`tool_use`Claude 2.1. blokları. İkizler eklendi.`functionDeclarations`Her sağlayıcı şimdi aynı şekli ortaya çıkarır: JSON-Schema-typed tool list, JSON-payload tool call out. Model Context Protocol (Kasım 2024) sözleşmeyi genelleştirdi, böylece her model için bir araç kayıtları hizmet eder. A2A (April 2026, v1.0) aynı primitif bir ajan-a-agent delegasyonu için katladı.

Dört adımlı döngü bunların altındaki değişmezliktir.

## Anlaşım

### Birinci adım: tanımlayın

Ev sahibi her bir aracı üç alanla açıklar.

- **Name.**Kalıcı, makine okuyabilir bir kimlik.`get_weather`"Vahitler" değil.
- **Description.**Bir paragraflık doğal dil kısacası. "Kullanıcı belirli bir şehrin mevcut koşulları hakkında sorular sorarken kullanın. Tarihsel veriler için kullanmayın".
- **Input schema.**A JSON Schema nesne (çeft 2020-12) aracı argümanlarını açıklayan.

Modelle listeyi alır. Modern sağlayıcılar bu bildirimleri sunucu-specifik bir şablon kullanarak sistem istatistiklerine seriye eder, böylece siz, arayan yalnızca yapılandırılmış formla ilgilenirsiniz.

### İkinci adım: karar ver

Kullanıcının mesajını ve mevcut araçları göz önüne alındığında, model üç davranıştan birini seçer.

1. **Answer directly**- Mesajla.
2. **Call one or more tools.**Yapılandırılmış çağrı nesneleri gönder.`parallel_tool_calls: true`(OpenAI ve Gemini'de öntanımlı olarak, Anthropic'e seçme) model bir dönüşte birden fazla arama gönderebilir.
3. **Refuse.**Sıkı modda yapılandırılmış çıkışlar bir türden `refusal`Bir arama yerine blok.

Bir araç çağrısı paylı yükü üç sabit alanı vardır: bir çağrı `id`, bir araç .`name`, ve bir JSON `arguments`Bu, paralel aramaların sıradan geldiğinde önemli olan belirli çağrı ile sonucunu ilişkilendirebilmesi için var.

### Üçüncü adım: Yürüt

Ev sahibi çağrıyı alır, açıklanan şema ile ilgili argümanları onaylar ve uygulayıcıyı çalıştırır. Geçersiz argümanlar, modelin bir alan halüsinasyonuna uğradığını veya yanlış bir tür kullanmış olduğunu gösterir. Zayıf modellerde çok yaygın bir başarısızlık modudur. Üretim sunucuları geçersiz argümanlar üzerinde üç şeyden birini yapar: hızlı başarısız olur ve modelde hatayı yüze çıkarır, kısıtlı bir analizci ile JSON'u onarır veya istekle dahil edilen doğrulama hatası ile modelini yeniden dener.

İcracı kendiliğinden sıradan bir koddur. Python, TypeScript, bir shell komutu, bir veritabanı sorusu. Genellikle bir dizilmiş bir sonuç üretir, ancak herhangi bir JSON değeri veya yapılandırılmış bir içerik bloğu (MCP'de metin, görüntü veya kaynak referansı) olabilir. Sonuç serize edilebilir olmalıdır.

### Dördüncü adım: gözlem

Ev sahibi, araç sonucuyu sohbete ekler (bir  olarak).`tool`eşleşme ile rol mesajı `id`Modelle, araç çıkışını bağlamda bulur ve son bir cevap verebilir veya daha fazla arama isteyebilir. Bu, model aramaları yayınlamayı durdurana veya ev sahibi tekrarlama sayısında güvenlik sınırına ulaşana kadar devam eder.

### Güven bölünmüştü .

Araçlar güvenlik için iki farklı tadda gelir.

- **Pure.**Sadece okunabilir, belirleyici, yan etkileri yok.`get_weather`- Evet .`search_docs`- Evet .`get_current_time`- İhtiyaçlı bir şekilde aramak güvenli.
- **Consequential.**Değişiklikler yaparak, para harcar, kullanıcı verilerine dokunur.`send_email`- Evet .`delete_file`- Evet .`execute_trade`Kapalı olmalı.

Meta'nın 2026'daki ajan güvenliği için "İki Kuralı" bir dönemde en fazla iki şeyi birleştirebileceğini belirtir: güvenilmeyen giriş, hassas veriler, sonuç eylemleri. Araç arayüzü bu kuralı  çağrıları reddeterek, kullanıcı onayını gerektirerek veya alanları yükselterek uyguladığınız yerdir. Tam güvenlik bölümü için 13 · 15 ve ajan düzeyinde izin politikaları için 14 · 09 aşamasını gör.

### Çubuk yaşadığı yer

| Context | Who describes | Who decides | Who executes |
|---------|---------------|-------------|--------------|
| Single-turn function calling (OpenAI/Anthropic/Gemini) | App developer | LLM | App developer |
| MCP | MCP server | LLM via MCP client | MCP server |
| A2A | Agent Card publisher | Calling agent | Called agent |
| Web browser (function-calling agent) | Browser extension / WebMCP | LLM | Browser runtime |

Her yerde aynı dört adım var.

### Neden sadece modelin JSON yaymasını istemezsin?

"JSON'da cevap vermesini iste" fonksiyon öncesi çağrı örneğiydi. Sınır modelleri üzerinde %5 ila %15 arasında başarısız olur ve daha küçük modellerde daha fazla. Başarısızlık modları eksik olan kollar, arka komlar, halüsinasyonlu alanlar ve yanlış türler içerir.

Doğal fonksiyon çağrısı üç nedenden dolayı daha iyidir. Öncelikle, sağlayıcı modelini tam çağrı şekli üzerine son-son eğitimi verir, böylece geçerli JSON oranı sıkı modda yüzde 98 ila 99'a kadar yükselir. İkincisi, arama payload kendi protokol boşluğunda oturur, serbest metin  içinde değil, bu yüzden bir araç çağrısı asla kullanıcı görebilen cevapta sızmaz. Üçüncü olarak, sağlayıcılar kısıtlı dekodlama ile şema uyumluluğunu zorlar (OpenAI'nin sıkı modu, Anthropic'in `tool_use`, İkizlerin `responseSchema`) Verilen ürünlerin doğrulanması garantilidir.

13 · 02 aşaması, üç sağlayıcı API'nin yan yana yürümesi. 13 · 04 aşaması, yapılandırılmış çıkışlara derinlemesine gider.

### Çeviri kesicileri

Modelin çağrıları yayınlamayı durdurduğu veya ev sahibi maksimum dönüş sayısını vurduğu zaman döngü sona erer. Üretim ev sahibi bunu 5 ile 20 dönüş arasında ayarlar. Bundan sonra, modelin çıkamayacağı bir döngüde neredeyse kesinlikle olursunuz. Claude Code varsayılan olarak 20'e; OpenAI Asistanları 10'a; Cursor'un ajan moduna 25.

Alternatif  sınırsız döngüler  her altı ayda bir "Agent gece içinde API çağrılarına 400 $ harcadı" post-mortem olarak ortaya çıkar.

Fase 14 · 12 hatayı iyileştirmeyi ve kendi kendini iyileştirmeyi derinlemesine kapsar; Fase 17 üretim oranının sınırlarını kapsar.

### Buradan itibaren 13. aşama devam edecek.

- 02 ile 05 dersleri, tedarikçi düzeyinde araç çağrı yüzeyini parlatır.
- 06-14 dersler, bu döngüyü MCP'ye genelleştirir.
- Dersler 15-18 düşman sunucular, düşman kullanıcılar ve doğrulanmamış uzaktan yazılım yüzeylerine karşı döngüyü korur.
- Dersler 19-22 örneği ajan-a-agent işbirliği, gözlemlenebilirlik, yönlendirme ve ambalajlara uzattır.
- Ders 23 her ilkelerle bir ekosistem oluşturur.

Geri kalan her ders bu dört adımlı döngünün bir gelişmesidir.

```figure
tp-tool-loop
```

## Kullan

`code/main.py`LLM olmadan dört adımlı döngü çalıştırır. Sahte bir "kararlı karar veren" işlevi, kullanıcı mesajında bir örneğe eşleşerek modeli simüle eder; uygulayıcı, şema onaylayıcısı ve gözlem-adım harness gerçeklerdir.

Neye bakılır:

- Araç kayıtları her bir araç için üç alanı içerir: isim, açıklama, şema ve bir yürütücü referansı.
- Validatör, sadece stdlib'de yazılmış en az JSON Schema alt kümesi (tipler, gerekli, enum, min/max) dir.
- Çaplıkların tekrar sayısı beş.

## Gönder

Bu ders bize çok yararlı .`outputs/skill-tool-interface-reviewer.md`. Bir proje araç tanımını (ad + açıklama + şema + yürütücü çizgisi) göz önüne alındığında, beceri bunu döngü uygunluğu için denetler: isim makine-sağlam mı, açıklama tam bir kullanım kısa mı, şema JSON Schema 2020-12'yi doğru kullanıyor mu, ve saf karşı sonuçlu sınıflandırma açık mı.

## Egzersizler

1. Dördüncü bir araç ekle `code/main.py`- Evet .`get_stock_price(ticker)`. Açıklamasını "Kullanıcı bir stok fiyatını ticker ile istediğinde kullanın. Tarihsel fiyatlar veya piyasa özetleri için kullanmayın". olarak yazın.

2. Şema onaylayıcıyı kırın ve kimin çağrısını yapın.`arguments`Eğer bir nesne istenen bir alanı eksikse, host'ın onu reddettiğini gerçekleştirmeden önce onaylayın.

3. Haresindeki her aletin saf veya sonuç olarak sınıflandırılmasını sağlayın.`consequential: true`Bu, her üretim host'unun ihtiyaç duyduğu onay kapısının şekli.

4. En sevdiğiniz müşteri için (Claude Desktop, Cursor, ChatGPT veya özel bir yığın) yukarıda bulunan sağlayıcı sütun tablosu ile kağıt üzerinde dört adımlı döngü çizin.

5. OpenAI'nin fonksiyon çağrısı rehberini yukarıdan aşağıya okuyun. İstekte yer alan ancak burada sunulduğu gibi dört adımlı döngüde olmayan bir alanı belirleyin. Ne eklediğini ve neden gerekli değil, uygun olduğunu açıklayın.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Tool | "A thing the model can call" | A triple of name + JSON-Schema-typed input + executor function |
| Function calling | "Native tool use" | Provider-level API support for emitting structured tool calls instead of prose |
| Tool call | "The model's request to act" | A JSON payload with `id`, `name`, `arguments` emitted by the model |
| Tool result | "What the tool returned" | The executor's output, wrapped in a `tool` role message with matching id |
| Parallel tool calls | "Many calls at once" | Multiple call objects in one model turn, independent and orderable by id |
| Strict mode | "Guaranteed JSON" | Constrained decoding that forces the model's output to validate against the declared schema |
| Pure tool | "Read-only tool" | No side effects; safe to re-run |
| Consequential tool | "Action tool" | Mutates external state; requires gate, audit, or user confirmation |
| Four-step loop | "The tool-call cycle" | describe → decide → execute → observe |
| Host | "Agent runtime" | The program that holds the tool registry, calls the model, and runs the executor |

## Daha Fazla Okumak

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) OpenAI tarzındaki araç açıklamaları ve çağrı biçimleri için kanonik referans
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)Claude'un.`tool_use`- Ne ?`tool_result`blok biçimi
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) `functionDeclarations`ve İkizler'de paralel çağrı semantikası
- [Model Context Protocol — Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) araç arayüzünün mevcut devletsiz, tedarikçi-agnostik genelleşmesi
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) her modern araç API konuşan schema dili
