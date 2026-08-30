# ए2ए  एजेंट-एजेंट प्रोटोकॉल

> एमसीपी एजेंट-टू-टू-टूल है। ए2ए (एजेंट2एजेंट) एजेंट-टू-एजेंट  एक खुला प्रोटोकॉल है जो विभिन्न ढांचे पर निर्मित अस्पष्ट एजेंटों को सहयोग करने की अनुमति देता है। अप्रैल 2025 में Google द्वारा जारी किया गया, जून 2025 में लिनक्स फाउंडेशन को दान किया गया, अप्रैल 2026 में 150+ समर्थकों सहित AWS, सिस्को, माइक्रोसॉफ्ट, सेल्सफोर्स, SAP और सर्विस नाउ सहित v1.0 तक पहुंच गया। इसने आईबीएम के एसीपी को अवशोषित किया और एपी 2 भुगतान विस्तार जोड़ा। यह सबक एजेंट कार्ड, कार्य जीवन चक्र और दो परिवहन बंधन के बारे में है।

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## सीखने के लक्ष्य

- एजेंट-टू-टू-टूल (MCP) के उपयोग के मामलों में एजेंट-टू-एजेंट (A2A) के बीच अंतर करना।
-  पर एजेंट कार्ड प्रकाशित करें`/.well-known/agent.json`कौशल और अंत बिंदु मेटाडेटा के साथ।
- कार्य जीवन चक्र (सदस्य → कार्य → इनपुट-आवश्यक → पूरा / विफल / रद्द / अस्वीकार) को चलाना।
- आउटपुट के रूप में भागों (पाठ, फ़ाइल, डेटा) और कलाकृतियों के साथ संदेशों का उपयोग करें।

## समस्या

ग्राहक सेवा एजेंट को रिपोर्ट लिखने को एक विशेषज्ञ लेखक एजेंट को सौंपने की आवश्यकता होती है।

- कस्टम REST एपीआई काम करता है, लेकिन हर जोड़ने एक बार है.
- साझा कोडबेस. दो एजेंटों को एक ही ढांचे को चलाने की आवश्यकता होती है.
- MCP: MCP को कॉल करने के लिए उपकरण है, दो एजेंटों के लिए नहीं जो एक साथ काम करते हैं जबकि प्रत्येक एजेंट की अस्पष्ट आंतरिक तर्क को संरक्षित करते हैं।

A2A अंतर को भरता है। यह एक एजेंट के जीवन चक्र, संदेशों और कलाकृतियों के साथ एक कार्य को दूसरे को भेजने के रूप में बातचीत का मॉडल बनाता है। बुलाए गए एजेंट की आंतरिक स्थिति अस्पष्ट रहती है।

A2A "चैरेक के माध्यम से एजेंटों को एक दूसरे से बात करने दें" प्रोटोकॉल है। यह MCP की जगह नहीं लेता है; दोनों पूरक हैं।

## अवधारणा

### एजेंट कार्ड

A2A के अनुरूप प्रत्येक एजेंट एक कार्ड प्रकाशित करता है `/.well-known/agent.json`:

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

खोज URL पर आधारित हैः कार्ड लाओ, A2A एंडपॉइंट का URL जानें, कौशल सूचीबद्ध करें।

### हस्ताक्षरित एजेंट कार्ड (AP2)

एपी 2 एक्सटेंशन (सितंबर 2025) एजेंट कार्ड में क्रिप्टोग्राफिक हस्ताक्षर जोड़ता है। एक प्रकाशक एक JWT के साथ अपना कार्ड साइन करता है; उपभोक्ता सत्यापित करते हैं। नकलीकरण को रोकता है।

### कार्य जीवन चक्र

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

ग्राहक  से शुरू करते हैं`tasks/send`. कॉल एजेंट राज्यों के माध्यम से संक्रमण करता है; ग्राहक एसएसई या सर्वेक्षण के माध्यम से राज्य अपडेट की सदस्यता लेते हैं।

### संदेश और भाग

एक संदेश में एक या अधिक भाग होते हैंः

- `text` सादा सामग्री।
- `file` mimeType के साथ आधार 64 ब्लेब।
- `data` टाइप किया गया JSON उपयोगिता लोड (कॉल एजेंट के लिए संरचित इनपुट) ।

उदाहरण:

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

### कलाकृतियाँ

आउटपुट आर्टिफैक्ट हैं, कच्चे स्ट्रिंग नहीं। एक आर्टिफैक्ट एक नामित, टाइप आउटपुट हैः

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

कलाकृतियों को टुकड़ों के रूप में स्ट्रीम किया जा सकता है।

### दो परिवहन बंधन

1. **JSON-RPC over HTTP.** `/a2a`अंत बिंदु, अनुरोधों के लिए POST, स्ट्रीमिंग के लिए वैकल्पिक SSE। डिफ़ॉल्ट बाध्यकारी।
2. **gRPC.**उद्यम वातावरण के लिए जहां gRPC मूल है।

दोनों बंधनों में एक ही तार्किक संदेश का आकार होता है।

### खुलेपन का संरक्षण

एक प्रमुख डिजाइन सिद्धांतः बुलाए गए एजेंट की आंतरिक स्थिति अस्पष्ट है। कॉल करने वाला कार्य स्थिति और कलाकृतियों को देखता है। बुलाए गए एजेंट की सोच श्रृंखला, उसके उपकरण कॉल, उसके उप-एजेंट प्रतिनिधि  सभी अदृश्य हैं। यह एमसीपी से अलग है, जहां उपकरण कॉल पारदर्शी हैं।

तर्कः A2A प्रतियोगियों को आंतरिक जानकारी के बिना सहयोग करने की अनुमति देता है। A2A "इस ग्राहक सेवा एजेंट को कॉल करें" हो सकता है, बिना कॉल करने वाले को यह जानने के कि एजेंट सेवा को कैसे लागू करता है।

### समय रेखा

- **2025-04-09.**गूगल A2A की घोषणा करता है।
- **2025-06-23.**लिनक्स फाउंडेशन को दान किया गया।
- **2025-08.**आईबीएम के एसीपी को अवशोषित करता है।
- **2025-09.**एपी2 विस्तार (एजेंट भुगतान) जहाज।
- **2026-04.**150+ सहायक संगठनों के साथ जारी v1.0।

### एमसीपी के साथ संबंध

| Dimension | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | Transparent tool calls | Opaque inner reasoning |
| Typical caller | Agent runtime | Another agent |
| State | Tool-call result | Task with lifecycle |
| Authorization | OAuth 2.1 (Phase 13 · 16) | JWT-signed Agent Cards (AP2) |
| Transport | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

जब आप किसी विशिष्ट टूल को कॉल करना चाहते हैं तो MCP का उपयोग करें। जब आप किसी अन्य एजेंट को पूरा कार्य सौंपना चाहते हैं तो A2A का उपयोग करें। कई उत्पादन प्रणालियों दोनों का उपयोग करती हैंः एक एजेंट अपने टूल लेयर के लिए MCP का उपयोग करता है और A2A अपने सहयोग लेयर के लिए।

```figure
a2a-task-lifecycle
```

## इसका प्रयोग करें

`code/main.py`एक शोध एजेंट अपना कार्ड प्रकाशित करता है, एक लेखक एजेंट एक प्राप्त करता है `tasks/send`एक पीडीएफ और एक पाठ निर्देश सहित भागों के साथ, काम → input_required → काम → पूरा कर के माध्यम से संक्रमण, और एक पाठ कलाकृतियों लौटता है। सभी stdlib; संदेश के रूपों पर ध्यान केंद्रित करने के लिए एक इन-मेमोरी परिवहन का उपयोग करता है।

क्या देखना हैः

- एजेंट कार्ड JSON आकार।
- कार्य आईडी असाइनमेंट और राज्य संक्रमण।
- मिश्रित प्रकार के भागों के साथ संदेश।
- इनपुट आवश्यक शाखा मध्य कार्य।
- कलाकृतियों को पूरा होने पर वापस।

## इसे भेजें

यह सबक हमें फल देता है`outputs/skill-a2a-agent-spec.md`. एक नए एजेंट को देखते हुए जिसे अन्य एजेंटों द्वारा बुलाया जाना चाहिए, कौशल एजेंट कार्ड JSON, कौशल योजना और अंत बिंदु ब्लूप्रिंट का उत्पादन करता है।

## व्यायाम

1. दौड़ें`code/main.py`. इनपुट के लिए आवश्यक विराम सहित कार्य के पूरे जीवन चक्र का पता लगाएं, जहां बुलाए गए एजेंट स्पष्टीकरण के लिए कहता है।

2. एक हस्ताक्षरित एजेंट कार्ड जोड़ें, कार्ड के कैनोनिक JSON पर HMAC के साथ हस्ताक्षर करें, एक सत्यापनकर्ता लिखें और पुष्टि करें कि यह एक उत्परिवर्तन कार्ड पर विफल रहता है।

3. कार्य प्रवाह को लागू करेंः लेखक एजेंट SSE पर तीन वृद्धिशील आर्टिफैक्ट टुकड़े जारी करता है और कॉल करने वाला उन्हें जमा करता है।

4. एक ए 2 ए एजेंट डिजाइन करें जो एक एमसीपी सर्वर को लपेटता है। प्रत्येक एमसीपी टूल को एक ए 2 ए कौशल में मैप करें। व्यापारिक अंतरों पर ध्यान दें  क्या अस्पष्टता खो गई है?

5. A2A v1.0 घोषणा को पढ़ें और एक विशेषता की पहचान करें जो अप्रैल 2026 तक किसी भी ढांचे द्वारा लागू नहीं की गई है। (संकेतः यह मल्टी-हॉप टास्क डेलिगेशन से संबंधित है) ।

## प्रमुख शर्तें

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

## आगे पढ़ना

- [a2a-protocol.org](https://a2a-protocol.org/latest/) कैनोनिक ए2ए विनिर्देश
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) संदर्भ कार्यान्वयन और एसडीके
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) जून 2025 शासन हस्तांतरण
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) रोडमैप और भागीदार गति
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) v1.0 रिलीज नोट्स और बैकवर्ड कॉम्पैक्ट गाइड
