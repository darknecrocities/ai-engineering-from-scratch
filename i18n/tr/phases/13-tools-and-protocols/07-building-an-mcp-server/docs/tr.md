# MCP Server Oluşturma: Devletsiz Python ve TypeScript

> Modern bir MCP sunucusu el sıkışmasını hatırlamıyor. Her istek üzerinde metadataları doğruluyor, bir işlemci çalıştırıyor ve bir yazılmış sonucu gönderir.

**Type:** Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 13, Lesson 06
**Time:** ~85 minutes

## Öğrenme Hedefleri

- İletişim zorunlu `server/discover`MCP için `2026-07-28`- Evet .
- Her istek üzerine protokol versiyonu ve istemci özelliklerini doğrulayın.
- Deterministik liste sırasıyla araçları, kaynakları ve ipuçlarını açıklayın.
- Geri dön .`resultType`, sunucu kimliği ve doğru sonuçları önbelleğe alıyor.
- Python ve TypeScript'te yeni satır sınırlı stüdyodan aynı devletsiz sözleşmeyi hizmet et.

## Sorun

İlk mesajdan sonra müşteri yeteneklerini depolayan bir sunucu, oluşturulması kolay ve çalıştırılması zor. Aynı süreç sıralı müşteriye hizmet verebilir. Uzak bir talebe farklı bir işçiye düşebilir. Eski bir yetenek açıklaması yetki sınırları üzerinden davranış sızdırmasını sağlayabilir.

MCP `2026-07-28`Bu uygulama, her istekleri kendi kendine tanımlayarak bu sorunun protokol kısmını çözer. Uygulama hala kalıcı notları, işleri veya açık durum ele geçirme işlemlerini tutabilir.

Bu ders, iki kez not sunucusu oluşturur. Python ve TypeScript sürümleri sadece protokol çekirdeği için standart kütüphanelerini kullanır.

## Anlaşım

### Modern gönderme döngüsü

```text
read one JSON-RPC line
parse the envelope
if it is a notification, do not respond
validate params._meta for this request
route by method
wrap success with resultType and serverInfo
write one JSON-RPC response line
forget request-scoped metadata
```

Üç stüdyo kuralları hala geçerlidir:

- Sadece JSON-RPC mesajlarını stdout'a yazın.
- Mesajları yeni bir çizgiyle sınırlandır ve her cevabı sıfırla.
- Stdin EOF'e ulaştığında derhal çıkın.

Bu süreç ömrü, modern bir MCP seansı değil.

### Başvuru onaylanması

Her talebin içinde:

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

İlk iki alan gereklidir.`clientInfo`mevcut kimlik biçimini doğrulayın, ancak onu kimlik doğrulama olarak değerlendirmeyin.

Eğer versiyon desteklenmiyorsa, return code `-32022`- Evet .`requested`ve `supported`. Kayıp istek metadata geçersiz params, kod `-32602`Daha önceki bir çağrıdan kayıp alanları asla doldurma.

### İhtiyaclı keşif

Modern sunucular uygulamalı `server/discover`. Tam bir keşif sonucu desteklenen modern sürümleri, özellikleri, seçmeli talimatları, önbelleği ipuçları ve sonuçta sunucu kimliğini içerir `_meta`- ...

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

Discovery sunucuyu kilitlemez.`tools/list`Bir keşif diye çağırmadan çünkü`tools/list`zaten aynı talep metadataları taşıyor.

### Araçlar

`tools/list`Devamlı sıralama, cevap önbelleğini iyileştirir ve model bağlamını istikrarlı tutar. Sonuç ayrıca `ttlMs`ve `cacheScope`- Evet .

`tools/call`İçerik bloklarını gönderir ve `isError`Protokol zarfı veya yöntem parametreleri geçersiz olduğunda JSON-RPC hatası kullanın. Kullanın `isError: true`geçerli bir araç çağrısı çalıştırıldığında ama araçın kendisi başarısız olduğunda.

Araç açıklamaları, uygulanma değil ipuçları kalır:

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

Ev sahibi bunları onay ve sunum için kullanmalıdır.

### Kaynaklar

`resources/list`sabit URI tanımlayıcılarını gönderir. `resources/read`Bu da, tıklanan içeriği geri gönderir.`2026-07-28`, yani her ikisi de `ttlMs`ve `cacheScope`- Evet .

Kullanım`cacheScope: "private"`Kullanıcıya özgü not verileri için. Paylaşılan bir önbelleğe yetki bağlamlarında özel bir yanıt yeniden kullanılamaz.

Modern değişim teslimatı kullanmıyor `resources/subscribe`Bir müşteri açılıyor .`subscriptions/listen`ve istekler `resourceSubscriptions`10. dersi bu akışı oluşturur.

### İpuçlar

`prompts/list`- Bu, gizliden gizlenebilir ve belirleyici.`prompts/get`gösterilen prompt sonucu tamamlanmıştır, ancak bu, önbelleğe alınan veya önbelleğe alınan sonuçlardan biri değildir.

### Her başarılı sonuç yazılır.

Örnekler her başarı için bir ambalaj kullanır:

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

Listesi, okuma ve keşif işçileri ekle `ttlMs`Ek olarak .`cacheScope`Bu sarkıtı merkezileştirmek bir işlemcinin modern sonuç alanlarını sessizce atmasını engeller.

### Sunucu tarafından başlatılan istekler yok

Modern bir sunucu bir istemci talebi ile ilgili bildirimler gönderebilir veya istemci tarafından açılan bir bildirimlere gönderebilir `subscriptions/listen`Kendi JSON-RPC talebini göndermemelidir.

Bir işlemci örnekleme, çıkartma veya kök girişi gerektiğinde, bir  gönderir.`input_required`Sonuç. Müşteri gömülü giriş isteklerini yerine getirir ve orijinal yöntemi yeni bir istek kimliği ile tekrar dener. 11. ders, çok yönlü bir yolculuğu istek biçimini kapsar.

### Açıkça miras verenlik

İki çağ sunucuları da `2025-11-25`Modern bir şekilde hareket etmeyi seçer.`_meta`alanlar mevcut ve geleneksel davranışlar .`initialize`- Evet .

Bir `2026-07-28`Eskiden gelen el sıkışması yoluyla talep edin.`resultType`Bu dersdeki kod kasıtlı olarak modern, bu yüzden değişkenleri görünür kalır.

```figure
t3-dispatch-loop
```

## Kullan

Python sunucusunun son demo ve testlerini çalıştır:

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

TypeScript portunu TypeScript çalıştırıcı ile çalıştır:

```bash
npx tsx main.ts --demo
```

Demo gönderir `server/discover`Bu sayede, her bir primitif listelenir, araçları çağrıştırır ve desteklenmeyen bir sürüm hatasını gösterir.

## Gönder

Bu ders gemileri `outputs/skill-mcp-server-scaffolder.md`. Bir keşif sözleşmesi, talep başına doğrulama, belirleyici cache listeleri ve seçmeli olarak izole edilmiş bir miras adaptörü ile modern bir sunucu planı üretir.

## Egzersizler

1. Bir talebinden özellikleri kaldırın ve sunucu, önceki talebin açıklamasını tekrar kullanmadığını kanıtlayın.
2. Değiştir `TOOLS`- Evet .`PROMPTS`Tüm liste sonuçlarının sabit kaldığını onaylayın.
3. Yıkıcı bir ekle.`notes_delete`Bu araçlar, yetki kontrolünü gerçekleştiricinin içinde yaptırmak için kullanılır.`destructiveHint`Sadece bir UX ipucu olarak.
4. Ekle`resources/templates/list`- Evet .`ttlMs`- Evet .`cacheScope`, ve belirleyici bir düzenleme.
5.  için ayrı bir miras adaptörü oluşturun`2025-11-25`Modern bir talebin asla girmediğini kanıtlayan testler ekleyin.

## Anahtar Terimler

| Term | Meaning |
|------|---------|
| Stateless server | Handles each request from its own metadata without protocol-session memory |
| `server/discover` | Mandatory modern method that advertises versions and capabilities |
| Complete result | Successful modern result with `resultType: "complete"` |
| Cacheable result | Discovery, list, or resource-read result with `ttlMs` and `cacheScope` |
| Deterministic list | Same logical registry produces the same item order |
| Server identity | Recommended `io.modelcontextprotocol/serverInfo` in result `_meta` |
| Tool error | Valid tool call that returns content with `isError: true` |
| Protocol error | Invalid JSON-RPC or MCP request returned through `error` |

## Daha Fazla Okumak

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
