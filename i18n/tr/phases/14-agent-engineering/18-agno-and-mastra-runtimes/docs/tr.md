# Üretim Ajansı Çalışma Zamanları  Hızlı Kuruluş ve Tiplenen Çalışma Akışları

> Bir üretim ajansı çalıştırma süresi, prototipleme çerçevelerinin görmezden gelmesini optimize eder: örnekleme maliyeti, yazılmış iş akışı yüzeyleri ve hizmet hazır bir arka uç. 2026 çiftleşmesi: Agno (Python) mikro saniye ajanı örnekleme ve devletsiz FastAPI arka uçları hedefliyor. Mastra ajanları, araçları, iş akışları, birleşik model yönlendirme ve Vercel AI SDK altyapısında bileşik depolama gönderir.

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## Öğrenme Hedefleri

- Agno'nun performans hedeflerini ve ne zaman önemli olduklarını belirle.
- Mastra'nın üç primitif adını verin  Ajanlar, Aletler, Çalışma Akışları  ve desteklenen sunucu adaptörleri.
- Devleti olmayan bir oturum ölçülü FastAPI arka planının neden Agno üretim yolunun önerildiğini açıklayın.
- Verilmiş bir yığın için Agno vs Mastra seçin (Python-birincisi vs TypeScript-birincisi).

## Sorun

LangGraph, AutoGen, CrewAI çerçeve ağırdır. "sadece ajan döngüsü, hızlı, benim çalıştırma süremde" isteyen takımlar Agno (Python) veya Mastra (TypeScript) için ulaşıyor.

## Anlaşım

### Agno

- Python çalıştırma zamanı, eski Phi-data.
- "Grafikler, zincirler ya da karmaşık desenler yok  saf Python".
- Dokümanlarından performans hedefleri: ~ 2μs ajan istansiasyonu, ~ 3.75 KiB hafıza/ ajan, ~ 23 model sağlayıcı.
- Üretim yolu: stateless sesyon-scoped FastAPI arka plan. Her istek yeni bir ajan başlatır; sesyon durumu bir DB'de yaşar.
- Doğal multimodal (metin, görüntü, ses, video, dosya) ve ajantik RAG.

Hız hedefleri saniyede binlerce kısa ömürlü ajan (chat fan-in, değerlendirme boru hattı) olduğunda önemlidir.

### Mastra

- TypeScript, Vercel AI SDK'ye dayalı.
- Üç ilkel:**Agents**- Evet .**Tools**(Zod tipi),**Workflows**- Evet .
- Birleştirilmiş Modeldeki Router  94 sağlayıcıda 3.300+ model (Mart 2026).
- Kompozite depolama: bellek, iş akışları, farklı arka planlara gözlemlenebilirlik; ClickHouse'nin ölçekte gözlemlenebilirlik için önerilen.
- Apache 2.0 ile `ee/`Kaynak kullanılabilir işletme lisansı altında dizinler.
- Express, Hono, Fastify, Koa için sunucu adaptörleri; birinci sınıf Next.js ve Astro entegrasyonu.
- Debug için Mastra Studio'yu (yalı host:4111) gönderir.
- 22k+ GitHub yıldızları, 300k+ haftalık npm indirimi 1.0 (Ocak 2026).

### Konumlandırma

LangGraph olmaya çalışmıyorlar.

- **Language fit.**Python ilk takımları için Agno, TypeScript için Mastra ilk.
- **Runtime ergonomics.**Agno = neredeyse sıfır uçuş masrafları; Mastra = Vercel ekosistemine entegre.
- **Observability.**Her ikisi de Langfuse/Phoenix/Opik (Denevi 24) ile entegre olur, ancak Mastra Studio birinci taraftır.

### Her birini ne zaman seçmeliyiz?

- **Agno**Python arka planı, çok kısa ömürlü ajanlar, güçlü perf gereksinimleri, FastAPI dükkanı.
- **Mastra** TypeScript arka planı, Next.js / Vercel dağıtım, birleşik çok sağlayıcı model yönlendirme, Zod-typed araçlar.
- **LangGraph**(Disim 13)  Kalıcı durum ve açık grafik mantık çabuklığından daha önemli olduğunda.
- **OpenAI / Claude Agent SDK** sağlayıcı tarafından üretilen şekli istediğinizde (Deneyim 1617).

### Bu kalıp yanlış gittiğinde

- **Perf-for-perf's-sake.**Agno'yu seçmek çünkü iş yükü istek başına bir yavaş ajan çağrısı olduğunda "2μs" iyi geliyor.
- **Ecosystem lock-in.**Mastra'nın Vercel aromalı entegrasyonu Vercel'de artı, başka yerlerde ise eksik.
- **Enterprise license confusion.**Mastra'nın `ee/`Dizinler kaynak kullanılabilir, Apache 2.0 değil.

```figure
wb-runtime-spawn
```

## Yapın

Bu ders öncelikle karşılaştırmalıdır  tek bir kod eserinin her iki çerçeveye de adalet sağlamayacağı görülmez.`code/main.py`Bir yan yana oyuncak için: iki kez (bir kez Agno şeklinde, bir kez Mastra şeklinde) uygulanan minimal "bir ajan çalıştır, çıkış akışı, devam sesyonu" akışı.

Çek şunu:

```
python3 code/main.py
```

Yapısal olarak farklı ama işlevsel olarak eşdeğer iki iz.

## Kullan

- **Agno** Hızlılık ve FastAPI şekli gerektiren Python arka uç.
- **Mastra** TypeScript arka planı birçok sağlayıcı ve iş akışı primitifleri ile.
- Her ikisi de Langfuse ile entegre.

## Gönder

`outputs/skill-runtime-picker.md`Stak, gecikme bütçesi ve operasyonel şekli üzerine kurulmuş Agno, Mastra, LangGraph veya bir sunucu SDK'yi seçer.

## Egzersizler

1. Agno'nun belgelerini okuyun, STDlib ReAct döngüsünü Agno'ya aktarın.
2. Mastra'nın belgeleri okuyun. Aynı döngüyü Mastra'ya aktarın. Araç yazımında ne değişmiştir (Zod vs. hiçbir şey)?
3. Benchmark: Ajanın istantasyon gecikmesini ölç.
4. Bir göç tasarlayın: Eğer Python'da CrewAI çalıştırıyorsanız, Agno'ya taşınırsanız ne kalır?
5. Mastra'nın kitabını okuyun.`ee/`Açık kaynaklı bir çatal için ne tür kısıtlamalar geçerli?

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agno | "Fast Python agents" | Stateless session-scoped agent runtime |
| Mastra | "TypeScript agents on Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "Multi-provider access" | Single client for 3,300+ models across 94 providers |
| Composite storage | "Multiple backends" | Memory/workflows/observability each to a different store |
| Mastra Studio | "Local debugger" | localhost:4111 UI for introspecting agents |
| Source-available | "Not OSS" | License permits source reading but restricts commercial use |

## Daha Fazla Okumak

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) Performans hedefleri, FastAPI entegrasyonu
- [Mastra docs](https://mastra.ai/docs) Primitivler, sunucu adaptörleri, Model yönlendiricisi
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) devlet grafik alternatif
- [Comet Opik](https://www.comet.com/site/products/opik/) Mastra entegrasyonları tarafından alıntılanan gözlemsellik karşılaştırmaları
