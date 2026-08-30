# Rol Baslı Ajan Takımları  Roller, Görevler, İşlemler

> Dört ilkel: Ajan, Görev, Ekip, İşlem. İki üst düzey şekil: Ekipler (özerk, rol tabanlı işbirliği) ve Akışlar (event yönlendirilmiş, belirleyici). CrewAI 2026 referans uygulanmasıdır ve dokümanları kesintedir: "herhangi bir üretim hazır uygulaması için, bir Akış ile başlayın".

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 minutes

## Öğrenme Hedefleri

- CrewAI'nin dört ilkinin (Agent, Görev, Ekip, İşlem) ve her birinin sahip olduğu isimlerini verin.
- Sequential, Hierarchical ve planlanan Konsens süreci arasında ayrım yapın; iş yüküne göre birini seçin.
- Ekipleri (özerk rol tabanlı) ve Akışları (evet yönlendirici belirleyici) ayırt edin ve doktorların üretim önerilerini açıklayın.
- Sınırlı aletler `@tool`dekorasyoncı ve `BaseTool`Alt sınıf; yapılandırılmış çıkışlar vs. serbest metin hakkında neden.
- Dört CrewAI hafıza tipini ve her birinin ne zaman ödediğini söyleyin.
- Kısa bir özet oluşturan bir stdlib üç ajan ekibi ( araştırmacı, yazar, editör) uygulayın.
- CrewAI'nin üç başarısızlık modunu tespit edin: Hızlı şişkinlik, yöneticilerle LLM vergisi, kırılgan elveriler.

## Sorun

Bir de, bir müşteri bir hata dosyası gönderir ve belirleyici bir tekrarlama gerekir. ya da finans, bir LLM yönlendirilmiş ekibin her koşuş için ne kadar maliyetini sorar. ya da çağrıda hangi ajanın sabah 3'te durduğunu bilmesi gerekir.

Özgür formlu LLM yönlendirici ekipler bunların hiç birine temiz cevap vermez.

CrewAI'nin bölümü iş konusunda dürüst. İşbirliği, rol tabanlı, keşif işi için ekipler.

## Anlaşım

### Dört ilkel

CrewAI'nin yüzeyi küçük, bunu ezberle ve geri kalanı yapılandır.

- **Agent.** `role + goal + backstory + tools + (optional) llm`Bu, ajanın durduğu zaman ton, yargı şeklini şekillendirir.
- **Task.** `description + expected_output + agent + (optional) context + (optional) output_pydantic`- Tekrar kullanılabilir birim.`expected_output`- Sözleşme.`context`çıkışlarının aktarıldığı yukarıdaki görevlerin listesi. `output_pydantic`yapılmış bir şekil oluşturur.
- **Crew.**Kontaner, listesi sahibi.`agents`, listesi `tasks`, `process`, ve seçmeli `memory`+ `verbose`+ `manager_llm`ayarları.
- **Process.**İcra stratejisi: sıralama, hiyerarşik, konsensüs (planlı)

Ajanlar birbirlerini doğrudan görmezler, görev referans ajanları, mürettebat görevleri sıraya koyuyor, süreç bir sonraki görevi kimin seçtiğini belirler.

> **Validated against**CrewAI 0.86 (2026-05). Yeni sürümler süreç türlerini yeniden adlandırılabilir veya birleştirilebilir; [CrewAI Processes docs](https://docs.crewai.com/concepts/processes)Bir şekil üzerine güvenmeden önce.

### İleri sıra vs. Hiyerarşik vs. Konsens

- **Sequential.**Görevler açıklama sırasıyla çalışmaktadır.`context`N+1 görevini yapın. En düşük maliyet. En tahmin edilebilir.
- **Hierarchical.**Bir yöneticisi (ABL çağrısı) uzmanlar arasında yollar. CrewAI yöneticisini ya sizin`manager_llm`Bu işlemler, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir sonraki görevden sonra, bir diğer görevden sonra, bir diğer görevden sonra, bir diğer görevden sonra, bir diğer görevden sonra, bir görevden sonra, bir görevden sonra, bir görevden sonra, bir görevden sonra, bir görevden sonra, bir görevden sonra, bir görevden sonra, bir görevden sonra, bir görevden sonra, bir görevden sonra, bir görevden sonra, bir görevden sonra, bir görevden sonra, bir görevden başlı olarak, bir görevden başlı olarak, bir görevden başlı olarak, bir görevden, bir görevden başlı olarak, bir görevden başlı olarak, bir görevden başlı olarak, yönlendirmeyi, yönlendirmeyi, yönlendirmeyi, yönlendirmeyi, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde, yönde,
- **Consensus.**Şimdilik kamu API'de uygulanmamış, planlanmış. Dokümanlar gelecekte oylama tabanlı bir süreç için adını saklıyor.

Hierarşik, her uzman çağrısının üstüne bir turlık LLM çağrısı (menjör) ekler. Token maliyeti beş adımlı bir çalışmada üç katına çıkabilir.

### Ekipleri Akışlara Karşı

Doktorlar 2026'da bu şekilde öne çıkıyor.

- **Crew.**LLM'den kaynaklanan özerklik. Çerçeve çalıştırma zamanında şekli seçer. Araştırma, beyin fırtınası, ilk taslaklar için iyi. Yol cevabın bir parçası olan her yerde. Tekrar oynamak zor. Test etmek zor. Prototip için ucuz.
- **Flow.**- Evrenle ilgili bir grafik.`@start`Girişi işaretliyor.`@listen(topic)`Bu, bir adımın diğer bir adımın bu konuyu yaydığı zaman ateş eden bir adımdır. Her adım basit Python'dur (bir Ekibi içten çağırabilirsiniz).

Doktörlerin 2026 üretim önerisi: bir akışla başlayın.`Crew.kickoff()`Akış, size denetim izini verir, mürettebat size keşifini verir.

### Araç entegrasyonu

Bir ajanı bir araçla kullanmanın üç yolu var.

1. **`@tool` decorator.**Temiz fonksiyonlar araç haline gelir. İmza şema, dokstring LLM'nin gördüğü açıklama.

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` subclass.**Açık args şeması, async desteği, tekrar deneme ile sınıf tabanlı araç. Araçın bir durumu (klient, bir önbelleği) veya yapılandırılmış args gerektiğinde kullanın.

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Built-in toolkits.**CrewAI ilk taraflı adaptörler gönderir: `SerperDevTool`- Evet .`FileReadTool`- Evet .`DirectoryReadTool`- Evet .`CodeInterpreterTool`- Evet .`RagTool`- Evet .`WebsiteSearchTool`- Bir import ile kablolu.

Yapılandırılmış çıkışlar Pydantic kullanıyor.`output_pydantic=MyModel`CrewAI, LLM tepkisini modelle karşılaştırır ve ya zorlar ya da tekrar dener.`expected_output`String. Özgür metin çıkışları taslaklar için iyidir; yapılandırılmış çıkışlar aşağı akıntılar tüketebilir.

### Hatıra hakları

CrewAI, dört bellek türünü kutudan çıkarır ve bir mürettebatın hepsini aynı anda etkinleştirmesini sağlar.

> **Validated against**CrewAI 0.86 (2026-05). Son yayınlar her şeyi birleştirilmiş bir yolla yönlendirir.`Memory`Aşağıdaki kavramsal model hala geçerli, ancak kamu sınıfı yüzeyi tek bir hale düşebilir.`Memory`Yeni versiyonlarda giriş noktası; kontrol [CrewAI memory docs](https://docs.crewai.com/concepts/memory)mevcut API için.

- **Short-term.**Bir seferinde konuşma tamponu, sonunda silinmiş.
- **Long-term.**Sürümler boyunca devam eder. Vektor DB'de depolanır (Önlü standard olarak Chrome, değiştirilebilir).
- **Entity.**"Müşteri X şirket planında" Şirket tarafından değil benzerlik tarafından belirlenir.
- **Contextual.**Toplama zamanı geri alımı, ajanın ihtiyacı olan an için ilgili hafızaları çekir, önceden yüklenmemiş.

 Ekibine etkinleştir`memory=True`Bu özellikler, bir içe gömülme sağlayıcısı tarafından desteklenir (Ölkel AI'ye özelleştirilmiş, yerel olarak değiştirilebilir).

### Rol tabanlı ekipler uyumlu olduğunda

- Adlı roller ve işbirliği iş akışı olan üç ila altı ajan, taslak, inceleme, planlama, beyin fırtınası.
- LLM'nin bir sonraki adımı hakkında hükmünün değerin bir parçası olduğu yönlendirme (hierarşik).
- Takımın okumaktan daha mutlu olduğu her yerde .`role + goal + backstory`Grafik tanımını okumaktan daha iyi.

### - Hayır.

- Sıkı sıralama ile belirleyici DAGlar. LangGraph kullanın (Deneyim 13). Graf şekli doğru soyutlama; CrewAI'nin rol çerçevesinde sürtünme vardır.
- İkinci alt gecikme bütçeleri. Hiyerarşik geri dönüşler ekler. Hatta Sequential arka plan hikayeleri ve önceki çıkışları içeren istekleri seriye eder.
- Tek ajan döngüleri. Çerçeveyi atlayın; bir ajan döngüsü (Deneyim 1) artı bir araç kayıtları daha kısa.

Ders 17 (Agent Framework Tradeoffs) bunu bir matriste açıklıyor. Kısa versiyonu: CrewAI "özellikle rol tabanlı" köşede oturuyor.

### Bağımlılık şekli

LangChain'den bağımsız. Python 3.10 ile 3.13. Kullanımları `uv`Yıldız sayımı:[crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)AWS Bedrock entegrasyonu belgelenmiştir; tedarikçi referansları, QA iş yüklerinde önemli bir hızlanma ile LangGraph karşılaştırıldığında rapor eder, ancak metodoloji ( veri kümesi, donanım, değerlendirme ölçüsü) yayınlanmamıştır, bu nedenle çerçeve- tedarikçi sayısını yalnızca yönlendirme olarak değerlendirin.

### Bu kalıp yanlış gittiğinde

- **Prompt-bloat from backstories.**Bir ajan ve beş ajanlı ekibinin başına 2000 kelimelik bir arka hikaye ilk araç çağrısı öncesi bağlam bütçesini yakar. arka hikayeyi 200 kelime altında tutun. Ajanlar arasında ifadeleri tekrarlayın; ev tarzını beş kez tekrarlamayın.
- **Manager-LLM token tax.**Hiyerarşik süreç, her uzman çağrısından önce bir yöneticinin LLM çağrısı ekler. Beş görevli bir ekip üzerinde beş yerine altı LLM çağrısı vardır ve yöneticinin çağrısı, tüm görev listesini ekleyip önceki çıkışları taşır.
- **Brittle handoffs.**Görev N's `expected_output`N+1 görevi, `context`Bu yüzden, bu yeni bir programın, bu yeni bir programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni programın, bu yeni yeni programın, bu yeni yeni yeni programın, yeni yeni yeni yeni yeni yeni programın, yeni yeni yeni yeni yeni yeni yeni yeni yeni programın, yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni programı, yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni yeni`output_pydantic`N görevi üzerine N+1 görevi, serbest metin değil, bir yazılmış nesne okur.
- **Crew-as-prod.**Ücretsiz formlı mürettebat, Flow kapak olmadan üretime gönderilmektedir. Çıktı değişkenliği yüksek; tekrar oynamak imkansızdır; çağrıda kötü bir koşuyu iyi bir koşuya karşı ayıramaz.

```figure
ae-crew-vs-flow
```

## Yapın

`code/main.py`Her iki şekilden STDlib sürümlerini uyguluyor ve üç ajanlı bir ekip.

Şekil:

- `Agent`- Evet .`Task`CrewAI'nin yüzeyinle eşleşen veri sınıfları.
- `SequentialCrew.kickoff(inputs)`Görevleri açıklama sırasıyla yürütür, çıkışları `context`- Evet .
- `HierarchicalCrew.kickoff(topic)`Bir müdür ekliyor. Bir ajan her turda bir sonraki uzmanı seçer. "Yaptı"da durur.
- `Flow`- Evet .`@start`ve `@listen(topic)`Dekorasyoncular, küçük bir olay döngüsü ve bir iz.
- `tool(name)`CrewAI'nin dekorasyonunu yansıtan dekorasyoncu.`@tool`şekli.
- `Memory`- Evet .`short_term`- Evet .`long_term`- Evet .`entity`Mağazalar; alaylı benzerlik, numpy kullanır.
- Sahte LLM cevapları, rol ve giriş önlüğü ile anahtarlanmış sert kodlanmış dizilerdir.

Beton demo: araştırmacı, yazar, editör ekibi "Agent mühendisliği 2026" üzerine kısa bir özet üretmektedir. Araştırmacı (gelenek) kaynakları çekmektedir. yazar taslakları. editör sıkıştırılır. Aynı ekip belirleyici şekli göstermek için bir akıştan geçer.

Çek şunu:

```bash
python3 code/main.py
```

İzleme kaplamaları: mürettebatın izlenimsel iplik çıkışları `context`, yöneticiler seçimi ( araştırmacı, yazar, editör, sonra "doldurulmuş") ile hiyerarşik ekip, açık konularla aynı üç adımı yürütme akışı (`researched`- Evet .`drafted`- Evet .`edited`), araç çağrıları yönlendirilir `@tool`, ve uzun süreli hafıza iki tekme üzerinden hayatta kalmak.

Mürettebat izleri sıvıdır, yöneticiler prensip olarak yeniden düzenleyebilir.

## Kullan

- **CrewAI Flow**Akışın bir adım olması bile,`Crew.kickoff()`Akış, denetim sınırını belirler.
- **CrewAI Crew (Sequential)**net bir işbirliği yaparak, özellikle ilk taslaklar ve inceleme döngüleri için.
- **CrewAI Crew (Hierarchical)**Eğer bir yönlendirme çıkışa bağlıysa ve dört veya daha fazla uzmanınız varsa.
- **LangGraph**(Disim 13) açık durum makineleri, kalıcı özetleme, sıkı düzenleme.
- **AutoGen v0.4**(Disim 14) aktör model eşzamanlılığı ve hata izolesi için.
- **OpenAI Agents SDK**(Disim 16) OpenAI-first ürünler için, el kolları ve koruma rayları ile.
- **Claude Agent SDK**(Deneyim 17) subagent ve sesyon mağazası ile Claude-first ürünler için.

## Gönder

`outputs/skill-crew-or-flow.md`Hard, Crew-without-backstory, Flow-without-explicit-topics, Hierarchical ile üç uzmanın altında reddetmektedir.

## Tuzaklar

- **Backstory as flavor.**Çıktıkları şekillendirir. Her ajan için üç variansı test eder.
- **Skipping `expected_output`.**Görev başına bir sözleşme olmadan, aşağıdaki görevler LLM'nin ürettiği her şeyi alır.
- **Memory always-on.**Uzun vadeli her çalışmayı yazar. vektör DB büyür. Arama gürültülü olur. Gerçek devamlı olan görevlere kapsam yazılır.
- **Manager prompt drift.**Yerel yöneticinin istekleri gizli, yönlendirme garip hale gelirse, sözcük moduna atıp oku.
- **Tool side effects in Crews.**Bir mürettebat, bir aracı beklenenden daha çok çağırabilir.

## Egzersizler

1. Sequential ekibini akışa dönüştürün, değişkenlik düştüğü noktaları sayın, okunma oranının düştüğünü not edin.
2. Ekibe bir varlık belleğini ekleyin: bir müşteriyle ilgili gerçekler, tekme atma sırasında kalır.
3. Yöneticinin yazarın çıkışının en az üç paragrafı olana kadar editörüne yönlendirmeyi reddettiği bir Hiyerarşik süreç uygulayın.
4. - Bir tel .`BaseTool`(Sıkıştırılmış) bir web arama için alt sınıf.`@tool`Dekorasyon versiyonu.
5. Ekle`output_pydantic=Brief`editör görevine,`Brief`- Evet .`title`- Evet .`summary`- Evet .`sections`. Yazıcı görevi çıkışı yanlış biçimlendirilmiş JSON bir kez yapın; takip CrewAI'nin tekrar deneme davranışını doğrulayın.
6. CrewAI'nin doküman girişini okuyun.`crewai`STDlib versiyonu hangi garantiyi atladı?
7. AgentOps veya Langfuse'i (Denevi 24) gerçek bir koşuya bağlayın.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "Persona" | Role + goal + backstory + tools |
| Task | "Unit of work" | Description + expected output + assignee + optional structured output |
| Crew | "Agent team" | Container for Agents + Tasks + Process |
| Process | "Execution strategy" | Sequential / Hierarchical / Consensus (planned) |
| Flow | "Deterministic workflow" | Event-driven, code-owned, testable |
| Backstory | "Persona prompt" | Tone and judgment shaper for the Agent |
| `@tool` | "Function tool" | Decorator that turns a function into a tool the Agent can call |
| `BaseTool` | "Class tool" | Class-based tool with args schema, retries, async support |
| Entity memory | "Per-entity facts" | Memory scoped to a customer / account / issue |
| Long-term memory | "Cross-run memory" | Vector-backed memory that survives between kickoffs |
| Contextual memory | "Just-in-time retrieval" | Memory pulled at the moment the Agent needs it |
| Manager LLM | "Router agent" | Extra LLM in Hierarchical process that picks the next task |
| `expected_output` | "Task contract" | String that tells the Agent (and audit) what shape to return |

## Daha Fazla Okumak

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction): kavramlar ve önerilen üretim yolu
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows): olay yönlendirici şekil, `@start`- Evet .`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)- Evet .`@tool`- Evet .`BaseTool`, yerleşik alet takımları
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory)Kısa süreli, uzun süreli, kurum, bağlamsal
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): multi-agent yardım ettiğinde ve yardım etmediğinde
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): devlet makinesi alternatif
