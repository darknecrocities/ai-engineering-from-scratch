# एक एमसीपी सर्वर बनानाः स्टेटलेस पायथन और टाइपस्क्रिप्ट

> आधुनिक एमसीपी सर्वर एक हाथ मिलाकर याद नहीं करता है। यह प्रत्येक अनुरोध पर मेटाडेटा को मान्य करता है, एक हैंडलर चलाता है, और एक टाइप परिणाम लौटाता है।

**Type:** Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 13, Lesson 06
**Time:** ~85 minutes

## सीखने के लक्ष्य

- अनिवार्य कार्यान्वयन`server/discover`एमसीपी के लिए `2026-07-28`. .
- प्रत्येक अनुरोध पर प्रोटोकॉल संस्करण और क्लाइंट क्षमताओं को मान्य करें।
- निर्धारक सूची क्रमबद्धता के साथ उपकरण, संसाधन और संकेतों का प्रदर्शन करें।
- लौटें`resultType`, सर्वर की पहचान, और सही परिणामों पर कैश संकेत.
- पायथन और टाइपस्क्रिप्ट में नए-लाइन-सीमित स्टूडियो पर एक ही राज्य रहित अनुबंध की सेवा करें।

## समस्या

एक सर्वर जो पहले संदेश के बाद क्लाइंट क्षमताओं को संग्रहीत करता है, निर्माण करना आसान है और संचालित करना मुश्किल है। एक ही प्रक्रिया अनुक्रमिक क्लाइंटों की सेवा कर सकती है। एक रिमोट अनुरोध एक अलग कार्यकर्ता पर उतर सकता है। एक पुराने क्षमता घोषणा प्राधिकरण सीमाओं के पार व्यवहार लीक कर सकती है।

एमसीपी `2026-07-28`आपके आवेदन में अभी भी टिकाऊ नोट्स, नौकरियां, या स्पष्ट राज्य हैंडल रख सकते हैं। यह जो नहीं रख सकता है वह छिपा हुआ प्रोटोकॉल राज्य है जो बाद में अनुरोध को कैसे डिकोड किया जाता है, उसे बदलता है।

इस पाठ में दो बार नोट्स सर्वर बनाया गया है। पायथन और टाइपस्क्रिप्ट संस्करण प्रोटोकॉल कोर के लिए केवल अपनी मानक पुस्तकालयों का उपयोग करते हैं। दोनों एक ही तरीकों का खुलासा करते हैं और एक ही तार अनुबंध को लागू करते हैं।

## अवधारणा

### आधुनिक डिस्पैच लूप

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

तीन स्टूडियो नियम अभी भी मायने रखते हैंः

- केवल JSON-RPC संदेशों को stdout में लिखें।
- संदेशों को एक नई रेखा के साथ परिभाषित करें और प्रत्येक प्रतिक्रिया को फ्लश करें।
- जब stdin EOF तक पहुँचता है शीघ्र ही बाहर निकलें।

प्रक्रिया जीवनकाल एक परिवहन जीवनकाल है यह आधुनिक एमसीपी सत्र नहीं है।

### अनुरोध सत्यापन

प्रत्येक अनुरोध में निम्नलिखित होना चाहिएः

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

पहले दो क्षेत्रों की आवश्यकता होती है।`clientInfo`वर्तमान पहचान के रूप को मान्य करें, लेकिन इसे प्रमाणीकरण के रूप में न देखें।

यदि संस्करण समर्थित नहीं है, तो कोड वापस करें `-32022`के साथ`requested`और `supported`. अनुपस्थित अनुरोध मेटाडेटा अमान्य पैरामीटर, कोड `-32602`. कभी भी पिछले कॉल से गायब क्षेत्रों को भरें.

### अनिवार्य खोज

आधुनिक सर्वरों को लागू करना होगा `server/discover`. एक पूर्ण खोज परिणाम में समर्थित आधुनिक संस्करण, क्षमताओं, वैकल्पिक निर्देश, कैश सुझाव और परिणाम में सर्वर पहचान शामिल है `_meta`:

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

डिस्कवरी सर्वर को अनलॉक नहीं करता है. एक क्लाइंट कॉल कर सकता है`tools/list`बिना खोज के बुलाया क्योंकि `tools/list`पहले से ही एक ही अनुरोध मेटाडेटा है।

### उपकरण

`tools/list`टूल वर्णकों की एक निर्धारात्मक सूची लौटाता है। स्थिर क्रम प्रतिक्रिया कैशिंग में सुधार करता है और मॉडल संदर्भ स्थिर रखता है। परिणाम भी आवश्यकता होती है `ttlMs`और `cacheScope`. .

`tools/call`सामग्री ब्लॉकों को लौटाता है और `isError`. प्रोटोकॉल के लिफाफे या विधि मापदंडों अमान्य हैं जब एक JSON-RPC त्रुटि का उपयोग करें। उपयोग `isError: true`जब एक मान्य उपकरण कॉल चल रहा है लेकिन उपकरण स्वयं विफल रहता है।

उपकरण के विवरण संकेत हैं, बल नहींः

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

होस्ट को पुष्टि और प्रस्तुति के लिए उनका उपयोग करना चाहिए. सर्वर को अभी भी वास्तविक प्राधिकरण लागू करना चाहिए।

### संसाधन

`resources/list`स्थिर URI वर्णक लौटाता है। `resources/read`टाइप सामग्री वापस करता है. दोनों में कैश करने योग्य हैं `2026-07-28`, तो दोनों शामिल हैं `ttlMs`और `cacheScope`. .

उपयोग करें`cacheScope: "private"`उपयोगकर्ता विशिष्ट नोट डेटा के लिए. एक साझा कैश प्राधिकरण संदर्भों में एक निजी प्रतिक्रिया का पुनः उपयोग नहीं कर सकता है.

आधुनिक परिवर्तन वितरण का उपयोग नहीं करता `resources/subscribe`. एक ग्राहक खुलता है .`subscriptions/listen`और अनुरोध `resourceSubscriptions`पाठ 10 उस प्रवाह का निर्माण करता है।

### संकेत

`prompts/list`यह छिपाया जा सकता है और निर्धारक है।`prompts/get`प्रस्तुत किया गया शीघ्र परिणाम पूरा है, लेकिन यह कैश करने योग्य सूची या पढ़ने के परिणामों में से एक नहीं है जो कैश सुझावों की आवश्यकता है।

### हर सफल परिणाम टाइप किया जाता है

उदाहरणों में प्रत्येक सफलता के लिए एक लपेट का उपयोग किया जाता हैः

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

सूची, पढ़ने और खोज संभाल करने वाले जोड़ें `ttlMs`और `cacheScope`इस लपेट को केंद्रीकृत करने से एक हैंडलर को मौन रूप से आधुनिक परिणाम क्षेत्रों को छोड़ने से रोकता है।

### सर्वर द्वारा आरंभित अनुरोध नहीं

आधुनिक सर्वर क्लाइंट अनुरोध से संबंधित सूचनाएं या क्लाइंट द्वारा खोले गए सूचनाएं भेज सकता है `subscriptions/listen`यह अपने स्वयं के JSON-RPC अनुरोध भेजने नहीं चाहिए।

जब एक प्रसंस्करणकर्ता को नमूना लेने, निकालने या जड़ों के इनपुट की आवश्यकता होती है, तो यह एक `input_required`परिणाम. क्लाइंट एम्बेडेड इनपुट अनुरोधों को पूरा करता है और एक नई अनुरोध आईडी के साथ मूल विधि को पुनः प्रयास करता है। पाठ 11 उस मल्टी राउंड-ट्रिप अनुरोध पैटर्न को कवर करता है।

### स्पष्ट विरासत संगतता

एक दोहरे युग सर्वर भी लागू कर सकते हैं `2025-11-25`यह आधुनिक व्यवहार का चयन करता है जब आवश्यक आधुनिक`_meta`क्षेत्र वर्तमान हैं और विरासत व्यवहार जब यह प्राप्त करता है `initialize`. .

एक न डालें `2026-07-28`आधुनिक के माध्यम से अनुरोध विरासत हाथ मिलाना पथ.`resultType`इस पाठ में कोड जानबूझकर आधुनिक है, इसलिए इसके अपरिवर्तित दृश्यमान रहते हैं।

```figure
t3-dispatch-loop
```

## इसका प्रयोग करें

पायथन सर्वर के अंतहीन डेमो और परीक्षण चलाएंः

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

टाइपस्क्रिप्ट रनर के साथ टाइपस्क्रिप्ट पोर्ट चलाएंः

```bash
npx tsx main.ts --demo
```

डेमो भेजता है `server/discover`, प्रत्येक आदिम को सूचीबद्ध करता है, उपकरण को कॉल करता है, और एक असमर्थित संस्करण त्रुटि दिखाता है। हर आधुनिक अनुरोध मेटाडेटा दोहराता है। हर सफलता में सर्वर पहचान शामिल है।

## इसे भेजें

यह सबक जहाजों `outputs/skill-mcp-server-scaffolder.md`. यह एक खोज अनुबंध, प्रति अनुरोध सत्यापन, निर्धारणीय कैश सूची और एक वैकल्पिक अलग विरासत एडाप्टर के साथ एक आधुनिक सर्वर योजना का उत्पादन करता है।

## व्यायाम

1. एक अनुरोध से क्षमताओं को हटा दें और साबित करें कि सर्वर पिछले अनुरोध की घोषणा का पुनः उपयोग नहीं करता है।
2. `TOOLS`,`PROMPTS`सभी सूची परिणाम स्थिर रहने की पुष्टि करें।
3. एक विनाशकारी जोड़ें `notes_delete`उपकरण और निष्पादक के अंदर प्राधिकरण की जांच की आवश्यकता है.`destructiveHint`केवल एक UX संकेत के रूप में.
4. जोड़ें `resources/templates/list`के साथ`ttlMs`,`cacheScope`, और निर्धारक क्रम।
5.  के लिए एक अलग विरासत एडाप्टर बनाएँ`2025-11-25`. आधुनिक अनुरोध कभी भी प्रवेश नहीं करता है कि साबित करने के लिए परीक्षण जोड़ें.

## प्रमुख शर्तें

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

## आगे पढ़ना

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
