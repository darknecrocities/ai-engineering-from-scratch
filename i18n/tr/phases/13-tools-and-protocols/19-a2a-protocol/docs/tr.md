# A2A  Ajan-Ajan Protokolü

> MCP, ajan-a-alıt. A2A (Agent2Agent) farklı çerçevelerde inşa edilmiş açık olmayan ajanların işbirliği yapmasına izin veren açık bir protokol. Google tarafından Nisan 2025'te yayınlanan, Haziran 2025'te Linux Vakfına bağışlanan, Nisan 2026'da AWS, Cisco, Microsoft, Salesforce, SAP ve ServiceNow dahil 150+ destekleyici ile v1.0'ya ulaştı. IBM'in ACP'ini absorbe etti ve AP2 ödeme uzatmalarını ekledi. Bu ders ajan kartı, görev yaşam döngüsü ve iki taşımacılık bağını içerir.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## Öğrenme Hedefleri

- Ajan-a-ağent (A2A) kullanım durumlarından ajan-a-ağent (MCP) kullanımı ayırt edin.
- Bir ajan kartı yayınlayın .`/.well-known/agent.json`Bilgiler ve son nokta metadataları ile.
- Görev yaşam döngüsünü izleyin (gelip gönderilen → çalışkan → girme gereksinimli → tamamlanmış / başarısız / iptal edilmiş / reddedilmiş).
- Çıktı olarak Parts (metin, dosya, veri) ve Artifacts ile Mesajlar kullanın.

## Sorun

Müşteri hizmetleri ajanı rapor yazmayı uzman bir yazar ajanına devretmelidir.

- Yapısal REST API'si çalışır ama her çiftleme bir kerelik.
- Ortak kod tabanı. İki ajanın aynı çerçeveyi çalıştırmasını gerektirir.
- MCP, iki ajanın birbirleriyle işbirliği yaparak her bir ajanın iç mantığını koruduğu halde, çağrı araçları için değil.

A2A boşluğu dolduruyor. Bir ajanın bir görevi diğerine gönderdiği etkileşimi, bir yaşam döngüsü, mesajlar ve eserlerle modellediyor. Çağrılan ajanın iç durumu netsiz kalır.

A2A, "cadre arası ajanların birbirleriyle konuşmasına izin verin" protokolüdür.

## Anlaşım

### Ajan Kartı

A2A ' ya uygun her ajan bir kart yayınlar .`/.well-known/agent.json`- ...

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

Bulma URL tabanlı: kartı getir, A2A son noktasının URL'sini öğren, becerileri say.

### İmzalanmış Ajan Kartları (AP2)

AP2 uzantısı (Eylül 2025) Agent Kartlarına kriptografik imzalar ekler. Bir yayıncı JWT ile kendi kartını imzalar; tüketiciler doğruluyor.

### Görev yaşam döngüsü

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

Müşteriler başlıyor `tasks/send`- Çağrılan ajan devletler üzerinden geçiş yapar; müşteriler SSE veya anket yoluyla devlet güncellemelerine abone olurlar.

### Mesajlar ve Bölümler

Bir mesaj bir veya daha fazla parçayı taşır:

- `text` Basit bir içerik.
- `file`Base64 blobı mimeType ile.
- `data` JSON payload (sırh edilen ajan için yapılandırılmış giriş) yazıldı.

Örnek:

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Sanat eserleri

Çıktıkları çiğ ip değil, eserler.

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Sanat eserleri parça olarak akıştırabilir.

### İki nakliye bağlaması

1. **JSON-RPC over HTTP.** `/a2a`Son nokta, istekler için POST, akış için seçeneği SSE. Öntanımlı bağlama.
2. **gRPC.**GRPC'nin yerli olduğu işletme ortamları için.

Her iki bağlama da aynı mantıklı mesaj şeklini taşır.

### Açıklık koruma

Ana tasarım prensibi: çağrılan ajanın iç durumu açık değildir. Çağrılan görev durumunu ve eserleri görür. çağrılan ajanın düşünce zinciri, araç çağrıları, alt ajan delegasyonu  hepsi görünmez. Bu, araç çağrılarının şeffaf olduğu MCP'den farklıdır.

A2A, rekabetçilerin içsel bilgileri açığa çıkarmadan işbirliği yapmasını sağlar. A2A, arama yapanın hizmeti nasıl uyguladığını öğrenmeden "bu müşteri hizmetleri ajanını arayın" olabilir.

### Zaman çizgisi

- **2025-04-09.**Google A2A'yı duyurdu.
- **2025-06-23.**Linux Vakfına bağışlanmış.
- **2025-08.**IBM'in ACP'ini emiriyor.
- **2025-09.**AP2 uzatma (Agent Ödeme) gemileri.
- **2026-04.**150+ destekleyici organizasyonla yayınlanan v1.0.

### MCP ile ilişki

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | JWT-signed Agent Cards (AP2) |
| Transport | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

Bir özel aracı çağrıştırmak istediğinizde MCP kullanın. Bir tüm görevi başka bir ajan'a devretmek istediğinizde A2A kullanın. Birçok üretim sistemi her ikisini de kullanır: bir ajan araç katmanı için MCP'yi ve işbirliği katmanı için A2A'yı kullanır.

```figure
a2a-task-lifecycle
```

## Kullan

`code/main.py`A2A'nın en az bir harnesini uyguluyor: bir araştırma ajanı kartını yayınlar, bir yazar ajanı bir `tasks/send`PDF ve metin talimatı dahil olmak üzere parçalar ile, çalışmak → input_required → working → tamamlanmış geçişler ve bir metin eserini iade eder. Tüm stdlib; mesaj şekillerine odaklanmak için bir hafıza taşımacılığı kullanır.

Neye bakılır:

- Ajan Kartı JSON şekli.
- Görev kimliği ve devre geçişleri.
- Karışık parçalarla mesajlar.
- Giriş gerektiren dal görev ortasında.
- Artifak tamamlandığında geri döner.

## Gönder

Bu ders bize çok yararlı .`outputs/skill-a2a-agent-spec.md`. Diğer ajanlar tarafından çağrılabilir olan yeni bir ajan verildiğinde, yetenek Agent Kart JSON, yetenek skemi ve son nokta çizelgesini üretir.

## Egzersizler

1. Çık .`code/main.py`. Çağırılan ajanın açıklama istediği giriş gerektiren durak dahil olmak üzere görev yaşam döngüsünün tamamını takip edin.

2. İmzalanmış bir ajan kartı ekleyin. HMAC ile kartın kanonik JSON'una imza atın. Bir doğrulama yazın ve mutasyonlu bir kartta başarısız olduğunu onaylayın.

3. Görev akışı uygulamak: yazar ajanı SSE üzerinden üç artfakta parçalarını yayar ve çağıran onları biriktirir.

4. Bir MCP sunucusuyla bir A2A ajanı tasarlayın. Her MCP aracı bir A2A yeteneğine göre bir harita yapın.

5. A2A v1.0 duyuruyu okuyun ve Nisan 2026 itibariyle herhangi bir çerçeve tarafından henüz uygulanmayan tek özelliği belirleyin.

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | "Agent-to-Agent protocol" | Open protocol for opaque agent collaboration |
| Agent Card | "`.well-known/agent.json`" | Published metadata describing an agent's skills and endpoint |
| Skill | "A callable unit" | A named operation the agent supports (analog to MCP tool) |
| Task | "Unit of delegation" | A work item with a lifecycle and final artifact |
| Message | "Task input" | Carries Parts (text, file, data) |
| Part | "Typed chunk" | `text` / `file` / `data` element of a message |
| Artifact | "Task output" | Named, typed output returned on completion |
| AP2 | "Agent Payments Protocol" | Signed Agent Cards extension for trust and payments |
| Opacity | "Black-box collaboration" | Called agent's internals are hidden from caller |
| Input-required | "Task pause" | Lifecycle state when the agent needs more info |

## Daha Fazla Okumak

- [a2a-protocol.org](https://a2a-protocol.org/latest/) Kanonik A2A spesifikasyonu
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) Referans uygulamalar ve SDK'lar
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) Haziran 2025 yönetim transfer
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) Yol haritası ve ortakların hareketi
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) v1.0 serbest bırakma notları ve geriye doğru kompak rehberlik
