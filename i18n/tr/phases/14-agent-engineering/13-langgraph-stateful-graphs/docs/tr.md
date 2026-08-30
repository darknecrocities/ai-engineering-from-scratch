# Devlet Grafi Orkestrasyonu  Sürdürülebilir İcra ve Kontrol Noktalar

> Agent bir durum makinesi; düğümler fonksiyonlardır; kenarlar geçişlerdir; her düğümden sonra durum kontrol noktasıdır. Son başarılı kontrol noktasında herhangi bir başarısızlıktan devam edin. LangGraph, düşük düzeyde durumlu orkestrasyonun bu modeli için 2026 referansıdır.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## Öğrenme Hedefleri

- LangGraph'un temel modelini açıklayın: Tiplenmiş durum, işlev düğümleri, koşullu kenarlar ve düğüm sonrası kontrol noktaları olan devlet makinesi.
- Dokümanların vurguladığı dört yetenekle ne anlatabilirsiniz: dayanıklı çalıştırma, akış, insan-da-da-da-da, kapsamlı hafıza.
- LangGraph'in desteklediği üç orkestrasyon topolojisini açıklayın: denetçi, eşeğen (swarm), hiyerarşik (bir yuva yapan altgraflar).
- Tiplenmiş durum, koşullu kenarları ve kontrol noktası/kurtarma döngüsü ile stdlib durum grafiğini uygulayın.

## Sorun

Ajanlar ve iş akışları bir sorunu paylaşıyor: 40 adımlı bir çalışmanın 38'inci adımdan başarısız olduğu zaman, ikinci sınıf devlet modelleri, yeni çalışmalar varsayılan bir kütüphane etrafında tekrar girişimleri hackleyen operatörleri bırakır.

LangGraph'in tasarım cevabı: durum birinci sınıf bir türden nesnedir, mutasyonlar açıkça belirlenir ve kontrol noktaları her düğümden sonra kalır.`load_state(session_id)`- Arayın.

## Anlaşım

### Grafiği

Bir grafik aşağıdakilerle tanımlanır:

- **State type.**Her düğümün okuduğu ve mutasyon yapdığı bir dikt (veya Pydantic modeli).
- **Nodes.**Saf fonksiyonlar `(state) -> state_update`Güncellemeler geri döndükten sonra devlete birleştirilir.
- **Edges.**Kodular arasındaki koşullu veya doğrudan geçişler.
- **Entry and exit.** `START`ve `END`Sentinel düğümleri sınırı işaretler.

Örnek: `classify`- Evet .`refund`- Evet .`bug`- Evet .`sales`- Evet .`done`düğümler  bir grafik olarak yönlendirme iş akışı.

### Sürekli çalıştırma

Her düğüm döndükten sonra, çalıştırma süresi durumu serilize eder ve bir kontrol noktasına yazar (SQLite, Postgres, Redis, özel).`resume(session_id)`ve tam durumla N+1 adımından devam et.

LangGraph dosyaları, önemli olan üretim kullanıcılarını açıkça vurguluyor: Klarna, Uber, JP Morgan. İddia grafik şekli değil; grafik şekli ve kontrol noktası kurtarmayı ucuzlaştırıyor.

### Akış

Her düğüm kısmi çıkış verebilir. Graf, düğüm-delta olayları için aramacıya akışlar. Böylece UIs graf çalıştırılırken güncellenir.

### - İnsanlık.

İşlemler: kritik bir düğümün önünde durak, insan için yüzey durumu, değişiklikleri kabul edin, devam edin. Kontrol noktası durumu zaten serileştirildiği için bunu kolaylaştırır.

### Hatıra

Kısa süreli (bir çalışmanın içinde  konuşma tarihi durumunda) ve uzun süreli (çekici çalışmalar  kontrol noktası ve ayrı uzun süreli bir depo aracılığıyla devamlı). LangGraph araçlar aracılığıyla dış bellek sistemleriyle (Mem0, özel) entegre edilir.

### Üç topoloji

1. **Supervisor.**Merkez yönlendirici LLM uzman subbagentlere gönderir.`create_supervisor()`İçeride`langgraph-supervisor`(LangChain ekibi 2026'da bunu doğrudan daha fazla bağlam kontrolü için araç çağrıları yoluyla yapmayı önerse de).
2. **Swarm / peer-to-peer.**Ajanlar doğrudan ortak bir araç yüzeyinden teslim edilir.
3. **Hierarchical.**Yönetim alt-generleri yöneten denetçiler, yuva alt-graf olarak uygulanır.

### Bu kalıp yanlış gittiğinde

- **Checkpoints too small.**Sadece kontrol noktaları konuşma dönerken araç durumu bırakır ve hafıza geri alınamayan yazılır.
- **Non-deterministic nodes.**Resume, düğüm girişlerinin aynı durum güncelleştirmesini sağladığını varsayır.
- **Over-use of conditional edges.**Her kenar koşullu bir grafik, mantık yapamayan bir durum makinesi.

```figure
langgraph-state
```

## Yapın

`code/main.py`stdlib durumlu bir grafik uyguluyor:

- `State` bir dikte ile yazılmış bir dikte`messages`- Evet .`step`- Evet .`route`- Evet .`output`- Evet .`human_approval`- Evet .
- `Node` Çağrılama durumunu alıyor ve güncelleştirilmiş bir dikteyi iade ediyor.
- `StateGraph` düğümler + kenarlar + koşullu kenarlar + çalıştır + devam edin.
- `SQLiteCheckpointer`(memory fake)  her düğümden sonra durumunu serilize eder; `load(session_id)`Geri getirir.
- Bir demo grafik: sınıflandır -> şubesi(refund / hata / satış) -> insan kapısı -> gönder.

Çek şunu:

```
python3 code/main.py
```

İz ilk koşunun insan kapısında başarısız olduğunu gösteriyor, ısrarcılık, sonra son çıkışı üretmeyi yeniden başlatıyor.

## Kullan

- **LangGraph** Referans, üretim hazır. Kullanım `create_react_agent`- Evet .`create_supervisor`Ya da kendi grafikini yap.
- **AutoGen v0.4**(Düşünme 14)  Yüksek rekabet senaryoları için oyuncu model alternatif.
- **Claude Agent SDK**(Deneyim 17)  İçinde yerleştirilmiş seans mağazası ile yönetilen harness.
- **Custom** durum şekli veya kontrol noktası arka uç üzerinde tam kontrol gerektirdiğinde.

## Gönder

`outputs/skill-state-graph.md`Herhangi bir hedef çalıştırma süresi içinde, kontrol noktası ve devamı kablo ile LangGraph şeklinde bir durum grafiği oluşturur.

## Egzersizler

1.  Şartlı bir kenar ekle`classify`- ...`end`Eğer bir sınıflandırma güveninin bir eşiğin altında olduğu zaman, insan ayarlarını takip ederken koşuyu devam ettirin.`route`El ile.
2. SQLite gibi sahte bir verilebilir.
3. paralel kenarları uygulayın: iki düğüm aynı anda çalışır, özel bir azaltıcı ile birleşir.
4. Oku `langgraph-supervisor`Oyuncakları buraya getir.`create_supervisor`- İz şekilleri karşılaştır.
5. Akış ekleyin: her düğüm çalıştırılırken kısmi durum verir.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| State graph | "Agent as state machine" | Typed state + nodes + edges + reducers |
| Checkpointer | "Persistence backend" | Serializes state after every node; enables resume |
| Reducer | "State merger" | Function that combines current state with a node's update |
| Conditional edge | "Branch" | Edge chosen by a function of state |
| Subgraph | "Nested graph" | A graph used as a node inside another graph |
| Durable execution | "Resume from failure" | Restart at the last successful node with exact state |
| Supervisor | "Router LLM" | Central dispatcher for specialist subagents |
| Swarm | "P2P agents" | Agents hand off via shared tools; no central router |

## Daha Fazla Okumak

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) referans belgeleri
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) Gözetmenlik örneği API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) Oyuncu-Model Alternatif
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) Sessiyon mağazası ve alt üslenme
