# इन्फेरेंस अनुकूलन

> दो चरण LLM inference को परिभाषित करते हैं. प्रीफिल आपके प्रॉम्प्ट को समानांतर में संसाधित करता है - कम्प्यूटेबल-बाउंड. डिकोड एक समय में टोकन उत्पन्न करता है - मेमोरी-बाउंड. प्रत्येक अनुकूलन एक या दोनों को लक्षित करता है.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lessons 01-08 (Transformer architecture, attention)
**Time:** ~120 minutes

## सीखने के लक्ष्य

- ऑटोरेग्रेसिव टोकन जनरेशन के दौरान अधिशेष गणना को समाप्त करने के लिए KV-कैश लागू करें
- एलएलएम निष्कर्षण के प्रीफिल बनाम डिकोड चरणों को समझाएं और प्रत्येक में अलग-अलग बोतल की खाई क्यों है (कंप्यूटर-बाउंड बनाम मेमोरी-बाउंड)
- समवर्ती अनुरोधों के तहत GPU उपयोग को अधिकतम करने के लिए निरंतर बैचिंग और पेजडएटेंशन अवधारणाओं को लागू करें
- निष्कर्ष अनुकूलन तकनीकों (केवी कैश, अनुमानात्मक डिकोडिंग, फ्लैश ध्यान) और उनके आउटपुट/लैटेंसी ट्रेडऑफ की तुलना करें

## समस्या

आप 4xA100 GPU पर Llama 3 70B को तैनात करते हैं. एक एकल उपयोगकर्ता को प्रति सेकंड ~ 50 टोकन मिलते हैं। यह तेजी से लगता है। फिर 100 उपयोगकर्ता एक साथ एंडपॉइंट पर पहुंचते हैं। पारगमन 3 टोकन / सेकंड / उपयोगकर्ता तक गिर जाता है। आपका $ 25,000 / महीने का GPU बिल मानव प्रकार की तुलना में धीमा प्रतिक्रियाएं प्रदान कर रहा है।

मॉडल स्वयं 1 उपयोगकर्ता से 100 उपयोगकर्ता के बीच नहीं बदलता है। एक ही वजन, एक ही वास्तुकला, एक ही गणित. जो बदलाव होता है वह है कि आप काम को कैसे शेड्यूल करते हैं। साफ़-साफ़ अनुमान उपलब्ध GPU कम्प्यूटर का 90%+ बर्बाद करता है। एक उपयोगकर्ता टोकन 47 के लिए इंतजार कर रहा है एक पूरे बैच स्लॉट खुला है जबकि GPU स्मृति बस matmuls के बीच निष्क्रिय बैठता है। इस बीच, एक नए उपयोगकर्ता के 2,000 टोकन का अनुरोध उस निष्क्रिय समय को उपयोगी गणना से भर सकता है।

यह एक स्केलिंग समस्या नहीं है. यह एक शेड्यूलिंग समस्या है. इस पाठ में तकनीक - KV कैशिंग, निरंतर बैचिंग, पेजड ध्यान, अनुमानात्मक डिकोडिंग, उपसर्ग कैशिंग - जो अलग कर रहे हैं एक$25k/month inference bill from a $5k/महीने एक समान यातायात सेवा.

vLLM 4xA100-80GB पर Llama 3 70B की सेवा करने वाले कम समवर्तीता पर ~ 50 टोकन / सेकंड / उपयोगकर्ता प्राप्त करते हैं, और निरंतर बैचिंग और पेजडएटेंशन के माध्यम से 100 समवर्ती अनुरोधों पर 15-25 TPS / उपयोगकर्ता को बनाए रखते हैं। इन अनुकूलन के बिना, एक ही हार्डवेयर उस समवर्तीता पर 5 TPS / उपयोगकर्ता की सेवा करता है। एक ही GPU, एक ही मॉडल, 4x थ्रूपुट।

## अवधारणा

### प्रीफिल बनाम डिकोड

प्रत्येक LLM inference request में दो अलग-अलग चरण होते हैं।

**Prefill**सभी टोकन ज्ञात हैं, इसलिए पूरे अनुक्रम में ध्यान समानांतर में गणना की जा सकती है। यह एक बड़ा मैट्रिक्स गुणा है - जीपीयू कोर व्यस्त रहते हैं। बोतल की खाई गणना हैः आपका हार्डवेयर प्रति सेकंड कितने फ्लोप्स वितरित कर सकता है। एक ए 100 312 टीएफएलओपीएस (बीएफ 16) करता है। 70 बी मॉडल पर 4,096-टोकन प्रॉम्प्ट के लिए प्रीफिल करने में ~ 400 एमएस का समय लगता है।

**Decode**एक समय में एक आउटपुट टोकन उत्पन्न करता है। प्रत्येक नए टोकन में सभी पिछले टोकन शामिल होते हैं, लेकिन प्रत्येक फॉरवर्ड पास पर केवल एक टोकन ही उत्पन्न होता है। वजन मैट्रिक्स के आकार पूर्व भरने के दौरान के रूप में ही हैं, लेकिन आप उन्हें एक मैट्रिक्स के बजाय एक वेक्टर से गुणा कर रहे हैं. GPU कोर माइक्रोसेकंड में समाप्त हो जाते हैं, फिर अगले बैच के वजन की मेमोरी से आने का इंतजार करते हैं। बोतल की गली मेमोरी बैंडविड्थ हैः आप HBM से कंप्यूटिंग इकाइयों में मॉडल वजन को कितनी तेजी से स्ट्रीम कर सकते हैं। एक A100 2 TB / s बैंडविड्थ है। FP16 में 70B मॉडल 140 GB है। एक बार पूरा मॉडल पढ़ने में 70ms लगते हैं -- यह एक ही डिकोडिंग चरण के लिए आपका तल है।

```mermaid
graph LR
    subgraph "Prefill (compute-bound)"
        P1["All prompt tokens"] --> P2["Parallel attention"]
        P2 --> P3["Full matmul utilization"]
    end

    subgraph "Decode (memory-bound)"
        D1["One token at a time"] --> D2["Sequential generation"]
        D2 --> D3["Waiting on memory reads"]
    end

    P3 --> D1
```

**ops:byte ratio**(जिसे अंकगणितीय तीव्रता भी कहा जाता है) इस व्यापार को कैप्चर करता है. यह मापता है कि आप प्रति बाइट स्मृति से लोड किए गए कितने संचालन करते हैं.

```
ops:byte ratio = FLOPs per token / bytes read from memory
```

4,096 टोकन के बैच के साथ प्रीफिल के दौरान, आप लोड किए गए वजन के प्रति ~ 4,096 गुण संचय संचालन करते हैं। अनुपात उच्च है - आप कंप्यूटिंग-बाउंड हैं। बैच आकार 1 के साथ डिकोडिंग के दौरान, आप लोड किए गए वजन के प्रति ~ 1 ऑपरेशन करते हैं। अनुपात कम है - आप मेमोरी-बाउंड हैं।

मूल अंतर्दृष्टिः *डिकोड मेमोरी-बंद है क्योंकि आप एक ही टोकन उत्पन्न करने के लिए पूरे मॉडल को पढ़ते हैं।* नीचे प्रत्येक अनुकूलन या तो आप जो पढ़ते हैं उसे कम करता है, प्रति पठन पर संसाधित टोकन की बैच बढ़ाता है, या पूरी तरह से पढ़ने से बचाता है।

### KV कैश

ध्यान के दौरान, प्रत्येक टोकन का क्वेरी प्रत्येक पिछले टोकन के कुंजी और मूल्य वेक्टरों का ध्यान रखती है। कैशिंग के बिना, टोकन N को उत्पन्न करने के लिए सभी N-1 से पहले टोकन के लिए कुंजी और मूल्य अनुमानों की पुनः गणना की आवश्यकता होती है। टोकन 1 को टोकन 2 उत्पन्न करते समय, फिर फिर टोकन 3 के लिए, फिर फिर टोकन 4 के लिए प्रोजेक्ट किया जाता है। टोकन 1,000 के माध्यम से, आपने टोकन 1 को कुल 999 बार प्रोजेक्ट किया है।

KV कैश सभी पिछले टोकन से कुंजी और मूल्य अनुमानों को संग्रहीत करता है। जब टोकन N उत्पन्न करते हैं, तो आप केवल टोकन N के लिए कुंजी और मूल्य की गणना करते हैं, फिर उन्हें टोकन 1 से N-1 तक कैश किए गए K / V के साथ जोड़ते हैं।

```mermaid
graph TD
    subgraph "Without KV Cache"
        A1["Token 5: recompute K,V for tokens 1-4"]
        A2["Token 6: recompute K,V for tokens 1-5"]
        A3["Token 7: recompute K,V for tokens 1-6"]
    end

    subgraph "With KV Cache"
        B1["Token 5: compute K5,V5, read K1-4,V1-4 from cache"]
        B2["Token 6: compute K6,V6, read K1-5,V1-5 from cache"]
        B3["Token 7: compute K7,V7, read K1-6,V1-6 from cache"]
    end
```

**Memory formula for KV cache:**

```
KV cache size = 2 * num_layers * num_kv_heads * head_dim * seq_len * bytes_per_param
```

Llama 3 70B (80 परतें, 8 KV सिरों के साथ GQA, head_dim=128, BF16) के लिएः

```
per token: 2 * 80 * 8 * 128 * 2 bytes = 327,680 bytes = 320 KB
at 4,096 tokens: 320 KB * 4,096 = 1.28 GB
at 128K tokens: 320 KB * 131,072 = 40 GB
```

Llama 3 70B के लिए एक एकल 128K-सामग्री वार्तालाप 40 GB केवी कैश का उपभोग करता है - आधा A100 की स्मृति। प्रत्येक 4K टोकन पर 100 समवर्ती उपयोगकर्ताओं के साथ, केवल KV कैश को 128 GB की आवश्यकता होती है। यही कारण है कि KV कैश प्रबंधन निष्कर्ष अनुकूलन की केंद्रीय चुनौती है।

### निरंतर बैचिंग

स्टैटिक बैचिंग N अनुरोधों के बैच के आगमन तक इंतजार करता है, उन्हें एक साथ संसाधित करता है, और नए अनुरोधों को स्वीकार करने से पहले *all* समाप्त होने तक इंतजार करता है। यदि एक अनुरोध को 500 टोकन और दूसरे को 10 की आवश्यकता होती है, तो यह समाप्त होने के बाद 490 डिकोडिंग चरणों के लिए निष्क्रिय रहता है।

निरंतर बैचिंग (जिसे पुनरावृत्ति स्तर बैचिंग भी कहा जाता है) किसी भी अनुरोध को पूरा करने के तुरंत बाद बैच में नए अनुरोधों को सम्मिलित करता है। प्रत्येक डिकोडिंग चरण में बैच का पुनर्मूल्यांकन किया जाता है। 10 टोकन के बाद समाप्त होने वाला अनुरोध तुरंत प्रतीक्षा अनुरोध से प्रतिस्थापित होता है।

```mermaid
sequenceDiagram
    participant GPU
    participant R1 as Request 1 (50 tokens)
    participant R2 as Request 2 (10 tokens)
    participant R3 as Request 3 (30 tokens)
    participant R4 as Request 4 (waiting)

    Note over GPU: Static batching
    GPU->>R1: Process batch [R1, R2, R3]
    Note over R2: R2 done at step 10
    Note over R2: Wasting 40 steps...
    Note over R3: R3 done at step 30
    Note over R3: Wasting 20 steps...
    GPU->>R4: Finally start R4 at step 50

    Note over GPU: Continuous batching
    GPU->>R1: Process batch [R1, R2, R3]
    Note over R2: R2 done at step 10
    GPU->>R4: Insert R4 at step 11
    Note over R3: R3 done at step 30
```

आउटपुट लंबाई में सुधार इस बात पर निर्भर करता है कि आउटपुट लंबाई कितनी भिन्न होती है। समान लंबाई के साथ, निरंतर बैचिंग स्थिर बैचिंग से मेल खाती है। चर लंबाई (सामान्य मामला) के साथ, निरंतर बैचिंग 2-5 गुना अधिक आउटपुट प्रदान कर सकती है क्योंकि GPU स्लॉट कभी भी खाली नहीं बैठते हैं।

### पृष्ठ ध्यान

प्रत्येक अनुरोध के लिए KV कैश एक आसन्न स्मृति ब्लॉक है। जब अनुरोध आते हैं और जाते हैं, तो स्मृति टुकड़े - ऑपरेटिंग सिस्टम में रैम टुकड़े टुकड़े की तरह। 4K टोकन अनुरोध को 1.28 GB आसन्न की आवश्यकता होती है। यहां तक कि अगर आपके पास 2 GB मुफ्त कुल है, तो आपके पास 1.28 GB *समीक्षा* नहीं हो सकता है। आप या तो स्मृति बर्बाद करते हैं या अनुरोध को अस्वीकार करते हैं।

PagedAttention (vLLM से) ओएस शैली की वर्चुअल मेमोरी को KV कैश में लागू करता है। प्रति अनुरोध एक आसन्न ब्लॉक आवंटित करने के बजाय, यह निश्चित आकार के "पृष्ठ" (आमतौर पर प्रत्येक 16 टोकन) आवंटित करता है। पृष्ठ भौतिक GPU मेमोरी में कहीं भी हो सकते हैं। एक पृष्ठ तालिका प्रत्येक अनुरोध के तार्किक अनुक्रम स्थानों को भौतिक पृष्ठ स्थानों पर मानचित्रित करती है।

```mermaid
graph TD
    subgraph "Contiguous allocation"
        C1["Request A: 2GB block"]
        C2["[free: 0.5GB]"]
        C3["Request B: 1GB block"]
        C4["[free: 1.5GB -- but fragmented]"]
    end

    subgraph "PagedAttention"
        P1["Page pool: 256 pages of 16 tokens each"]
        P2["Request A: pages 3,7,12,45,88..."]
        P3["Request B: pages 1,4,9,22,67..."]
        P4["No fragmentation, no waste"]
    end
```

PagedAttention भी सक्षम करता है **copy-on-write**यदि 50 अनुरोध एक ही सिस्टम प्रॉम्प्ट को साझा करते हैं, तो उस सिस्टम प्रॉम्प्ट के लिए KV कैश पृष्ठ एक बार संग्रहीत होते हैं और सभी 50 अनुरोधों द्वारा संदर्भित होते हैं। केवल जब एक अनुरोध भिन्न होता है (अलग उपयोगकर्ता संदेश) तो इसे अपने स्वयं के पृष्ठ प्राप्त होते हैं। यह साझा सिस्टम प्रॉम्प्ट वाले अनुप्रयोगों के लिए मेमोरी उपयोग को नाटकीय रूप से कम करता है।

vLLM ने PagedAttention के माध्यम से लगभग शून्य स्मृति अपशिष्ट (~ 4% बनाम ~ 60-80% साफ़ आवंटन में) की रिपोर्ट की।

### अनुमानित डिकोडिंग

डिकोड धीमा है क्योंकि यह अनुक्रमिक है -- आप एक टोकन उत्पन्न करते हैं, इसे वापस खिलाते हैं, अगले को उत्पन्न करते हैं. लेकिन क्या होगा अगर आप अगले 5 टोकन को सस्ते में अनुमान लगा सकते हैं, फिर उन्हें एक साथ सत्यापित कर सकते हैं?

अनुमानित डिकोडिंग एक छोटे, तेजी से उपयोग करता है **draft model**K उम्मीदवार टोकन उत्पन्न करने के लिए।**target model**फिर एक आगे के पास में सभी K उम्मीदवारों को संसाधित करता है (जो एक पूर्व-पूर्ति की तरह दिखता है - समानांतर, कम्प्यूटिंग-बाउंड, कुशल) यदि लक्ष्य मॉडल ड्राफ्ट मॉडल की भविष्यवाणियों से सहमत है, तो आप एक लक्ष्य आगे के पास के समय में सभी K टोकन स्वीकार करते हैं। यदि यह स्थिति j पर असहमत है, तो आप टोकन 1 से j-1 तक स्वीकार करते हैं और बाकी को फेंक देते हैं।

```mermaid
graph LR
    D["Draft model (1B)"] -->|"Generate 5 tokens<br/>~5ms"| C["Candidates: the cat sat on the"]
    C --> T["Target model (70B)"]
    T -->|"Verify all 5 in one pass<br/>~70ms"| V{"Match?"}
    V -->|"4 of 5 match"| A["Accept 4 tokens in 75ms<br/>vs 280ms sequential"]
    V -->|"Mismatch at pos 5"| R["Reject token 5<br/>Resample from target"]
```

गति गति पर निर्भर करता है **acceptance rate**- कितनी बार मसौदा मॉडल की भविष्यवाणियों लक्ष्य से मेल खाती हैं। Llama 3 8B के लिए Llama 3 70B के लिए मसौदा के लिए, 70-85% की स्वीकृति दरें प्राकृतिक भाषा पर विशिष्ट हैं। यह 2-3x डिकोडिंग गति का अनुवाद करता है।

अनुमानित डिकोडिंग के तीन दृष्टिकोणः

| Method | Draft source | Acceptance rate | Overhead |
|--------|-------------|-----------------|----------|
| Draft-target (Leviathan et al.) | Separate small model | 70-85% | Draft model memory |
| EAGLE (Li et al.) | Lightweight head on target | 75-90% | ~1% extra parameters |
| N-gram lookup | Token n-gram table | 40-60% | Negligible |

**EAGLE**लक्ष्य मॉडल के छिपे राज्यों के शीर्ष पर एक छोटे से ऑटोरेग्रेसिव सिर को प्रशिक्षित करता है। यह लक्ष्य मॉडल की दूसरी से अंतिम परत सुविधाओं का उपयोग करके अगले टोकन के एम्बेडिंग की भविष्यवाणी करता है। क्योंकि यह लक्ष्य मॉडल के स्वयं के प्रतिनिधित्वों (अलग मॉडल के नहीं) पर काम करता है, यह न्यूनतम अतिरिक्त स्मृति के साथ उच्च स्वीकृति दर प्राप्त करता है। ईगल-2 एक गतिशील ड्राफ्ट ट्री जोड़ता है जो संदर्भ के आधार पर उम्मीदवारों की संख्या को समायोजित करता है।

**N-gram speculative decoding**वर्तमान संदर्भ या पूर्वनिर्मित कॉर्पस से n-ग्राम निरंतरता की तालिका बनाए रखता है। यदि मसौदा उसी बातचीत में पहले दिखाई देने वाले से मेल खाता है (पुनरावृत्ति पैटर्न, कोड, संरचित आउटपुट), तो यह शून्य तंत्रिका नेटवर्क ओवरहेड के साथ फायर करता है। स्वीकृति दर औसत से कम है लेकिन प्रति अटकल की लागत अनिवार्य रूप से मुफ्त है।

अनुमानात्मक डिकोडिंग *गणितीय रूप से सटीक* है - आउटपुट वितरण लक्ष्य मॉडल के वितरण के समान है। यह एक अनुमान नहीं है। सत्यापन चरण यह सुनिश्चित करता है कि प्रत्येक स्वीकार किए गए टोकन में लक्ष्य मॉडल द्वारा असाइन की गई संभावना है।

### पूर्वसर्ग कैशिंग

कई अनुरोध एक ही पूर्वावलोकन साझा करते हैं। एक चैटबॉट सिस्टम प्रॉम्प्ट। एक RAG संदर्भ ब्लॉक। कुछ शॉट उदाहरण सेट। पूर्वावलोकन कैशिंग के बिना, प्रत्येक अनुरोध इन साझा टोकन के लिए KV कैश को खरोंच से पुनः गणना करता है।

प्रीफिक्स कैशिंग सामान्य प्रीफिक्स के लिए KV कैश को संग्रहीत करता है और अनुरोधों के बीच इसका पुनः उपयोग करता है। जब एक नया अनुरोध एक ज्ञात प्रीफिक्स के साथ आता है, तो सिस्टम कैश किए गए KV प्रविष्टियों को कॉपी (या संदर्भ) करता है और केवल अद्वितीय प्रत्यय के लिए KV की गणना करता है।

2,000 टोकन वाले सिस्टम प्रॉम्प्ट के लिए सभी अनुरोधों के बीच साझा किया जाता है, प्रीफिक्स कैशिंग प्रति अनुरोध ~ 400ms प्रीफिल को समाप्त करता है। 100 अनुरोध / सेकंड पर, यह प्रति सेकंड 40 सेकंड GPU गणना बचाता है - एक GPU के मूल्य से अधिक काम।

SGLang के RadixAttention एक रेडिक्स पेड़ (trie) के साथ पूर्वावलोकन कैशिंग को लागू करता है जो उनके टोकन सामग्री के अनुसार पूर्वावलोकनों को अनुक्रमित करता है। किसी भी अनुरोध को संग्रहीत पूर्वावलोकन से मेल खाता है, उसे मुफ्त में KV कैश मिलता है। पेड़ आंशिक पूर्वावलोकन मैच सक्षम करता है - यदि आप एक कैश किए गए प्रविष्टि के साथ 2,000 पूर्वावलोकन टोकन में से 1,500 साझा करते हैं, तो आप उन 1,500 का पुनः उपयोग करते हैं और केवल 500 को पुनः गणना करते हैं।

### इन्फेरेंस इंजन

तीन इंजन उत्पादन एलएलएम सेवा पर हावी हैंः

| Engine | Key innovation | Best for |
|--------|---------------|----------|
| vLLM | PagedAttention, continuous batching | General-purpose serving, highest compatibility |
| SGLang | RadixAttention (prefix caching), structured generation | Multi-turn chatbots, constrained decoding |
| TensorRT-LLM | NVIDIA kernel fusion, FP8 quantization | Maximum single-GPU throughput on NVIDIA hardware |

**vLLM**यह सबसे व्यापक मॉडल रेंज का समर्थन करता है, किसी भी GPU आपूर्तिकर्ता (NVIDIA, AMD, Intel) पर चलता है, और PagedAttention + निरंतर बैचिंग के माध्यम से मजबूत आउटपुट प्राप्त करता है। OpenAI- संगत एपीआई का मतलब है कि आप इसे किसी भी OpenAI एपीआई कॉल के लिए प्रतिस्थापन के रूप में छोड़ सकते हैं।

**SGLang**vLLM के समान नींव पर बनाता है लेकिन संरचित LLM कार्यक्रमों के लिए प्रीफिक्स कैशिंग के लिए रडिक्सएटेंशन और एक डोमेन-विशिष्ट भाषा जोड़ता है। यदि आपके कार्यभार में मल्टी-टर्न वार्तालाप, उपकरण उपयोग या प्रतिबंधित डिकोडिंग (JSON आउटपुट, रेजेक्स-ग्यूडेड जनरेशन) शामिल है, तो एसजीएलएंग अक्सर प्रीफिक्स पुनः उपयोग के माध्यम से 2-5 गुना वीएलएलएम से बेहतर प्रदर्शन करता है।

**TensorRT-LLM**यह NVIDIA हार्डवेयर पर उच्चतम एकल-जीपीयू आउटपुट प्राप्त करता है लेकिन अधिक सेटअप की आवश्यकता होती है और केवल NVIDIA GPU पर काम करता है।

Llama 3 70B (4xA100-80GB, BF16) के लिए वास्तविक दुनिया के नंबरः

| Metric | vLLM | SGLang | TensorRT-LLM |
|--------|------|--------|---------------|
| Throughput (1 user) | ~50 TPS | ~55 TPS | ~65 TPS |
| Throughput (100 users) | ~2,500 total TPS | ~3,200 total TPS | ~3,000 total TPS |
| Time to first token | ~400ms | ~300ms (prefix hit) | ~350ms |
| Max context | 128K | 128K | 128K |

### ओप्सःबाइट फ्रेमवर्क

आप जो नहीं मापते हैं उसे अनुकूलित नहीं कर सकते। आपरेशनःबाइट अनुपात आपको बताता है कि आप कम्प्यूटेबल-बाउंड या मेमोरी-बाउंड हैं, जो यह निर्धारित करता है कि कौन से अनुकूलन मायने रखते हैं।

```
Compute roof: peak FLOPS of the GPU
Memory roof:  peak bandwidth * ops:byte ratio
```

जब ops:byte कम होता है (डेकोड, छोटे बैच), तो आप मेमोरी बैंडविड्थ छत पर हिट होते हैं। अधिक गणना (उच्च घड़ी, अधिक कोर) जोड़ना मदद नहीं करता है। आपको मेमोरी रीड्स (क्वांटिकेशन, KV कैश संपीड़न) को कम करने या अधिक उपयोगी काम पर रीड्स को निरस्त करने के लिए बैच आकार में वृद्धि करने की आवश्यकता होती है।

जब ops:byte उच्च होता है (पूर्व भरने, बड़े बैच), तो आप कंप्यूटिंग छत पर पहुंच जाते हैं। मेमोरी बैंडविड्थ अनुकूलन मदद नहीं करता है। आपको अधिक FLOPS को निचोड़ने के लिए तेजी से GPU, कर्नेल फ्यूजन या कम परिशुद्धता की आवश्यकता होती है।

| Scenario | ops:byte | Bound | Optimize with |
|----------|----------|-------|---------------|
| Prefill, batch=1 | ~4,096 | Compute | Kernel fusion, FP8 |
| Decode, batch=1 | ~1 | Memory | Quantization, KV compression |
| Decode, batch=32 | ~32 | Memory | Larger batch, continuous batching |
| Decode, batch=256 | ~256 | Transitioning | Both matter |
| Decode, batch=1024 | ~1,024 | Compute | Kernel fusion, tensor parallelism |

A100 पर क्रॉसओवर बिंदु ops:byte = 156 (312 TFLOPS / 2 TB/s) के आसपास है। 156 के नीचे, आप मेमोरी-बाउंड हैं। 156 के ऊपर, आप कंप्यूटिंग-बाउंड हैं। निरंतर बैचिंग प्रति पुनरावृत्ति अधिक टोकन पैक करके इस क्रॉसओवर की ओर डिकोड को धक्का देती है।

```figure
context-window-slide
```

## इसे बनाओ

### चरण 1: स्क्रैच से KV कैश

हम एक बहु-हेड केवी कैश बनाते हैं जो प्रति परत, प्रति सिर, कुंजी और मूल्य अनुमानों को संग्रहीत करता है, और स्मृति विकास पैटर्न का प्रदर्शन करता है।

```python
import numpy as np

class KVCache:
    def __init__(self, num_layers, num_heads, head_dim, max_seq_len, dtype=np.float16):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        self.k_cache = np.zeros(
            (num_layers, num_heads, max_seq_len, head_dim), dtype=dtype
        )
        self.v_cache = np.zeros(
            (num_layers, num_heads, max_seq_len, head_dim), dtype=dtype
        )
        self.seq_len = 0

    def update(self, layer_idx, new_keys, new_values):
        num_new = new_keys.shape[1]
        end = self.seq_len + num_new
        self.k_cache[layer_idx, :, self.seq_len:end, :] = new_keys
        self.v_cache[layer_idx, :, self.seq_len:end, :] = new_values
        return (
            self.k_cache[layer_idx, :, :end, :],
            self.v_cache[layer_idx, :, :end, :]
        )

    def advance(self, num_tokens):
        self.seq_len += num_tokens

    def memory_bytes(self):
        return self.k_cache.nbytes + self.v_cache.nbytes

    def used_bytes(self):
        per_token = 2 * self.num_layers * self.num_heads * self.head_dim * np.dtype(self.dtype).itemsize
        return per_token * self.seq_len
```

### चरण 2: KV कैश के साथ ध्यान

एक सरल बहु-हेड ध्यान जो डिकोड चरणों के लिए KV कैश का उपयोग करता है।

```python
def scaled_dot_product_attention(query, keys, values):
    head_dim = query.shape[-1]
    scores = np.matmul(query, keys.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    seq_len_q = scores.shape[-2]
    seq_len_k = scores.shape[-1]
    if seq_len_q > 1:
        mask = np.triu(np.ones((seq_len_q, seq_len_k), dtype=np.float32), k=seq_len_k - seq_len_q + 1)
        scores = scores + mask * (-1e9)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return np.matmul(attn_weights, values)


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * scale

    def forward(self, x, kv_cache=None, layer_idx=0):
        batch, seq_len, d_model = x.shape
        Q = np.matmul(x, self.W_q).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = np.matmul(x, self.W_k).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = np.matmul(x, self.W_v).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            K_full, V_full = kv_cache.update(layer_idx, K[0], V[0])
            K = K_full[np.newaxis, :, :, :]
            V = V_full[np.newaxis, :, :, :]
            if seq_len == 1:
                kv_cache.advance(1)

        attn_out = scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, -1, d_model)
        return np.matmul(attn_out, self.W_o)
```

### चरण 3: निरंतर बैचिंग सिम्युलेटर

यह स्थैतिक और निरंतर बैचिंग के बीच अनुसूची अंतर का अनुकरण करता है।

```python
import heapq

class Request:
    def __init__(self, request_id, prompt_tokens, output_tokens, arrival_step):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.arrival_step = arrival_step
        self.tokens_generated = 0
        self.start_step = None
        self.end_step = None

    def is_done(self):
        return self.tokens_generated >= self.output_tokens


def simulate_static_batching(requests, batch_size):
    step = 0
    completed = []
    queue = list(requests)
    queue.sort(key=lambda r: r.arrival_step)

    while queue:
        batch = []
        while queue and len(batch) < batch_size:
            r = queue.pop(0)
            r.start_step = max(step, r.arrival_step)
            batch.append(r)

        if batch:
            step = max(step, max(r.start_step for r in batch))
            max_output = max(r.output_tokens for r in batch)
            for r in batch:
                r.tokens_generated = r.output_tokens
                r.end_step = step + max_output
            step += max_output
            completed.extend(batch)

    return completed


def simulate_continuous_batching(requests, batch_size):
    step = 0
    completed = []
    queue = sorted(requests, key=lambda r: r.arrival_step)
    queue_idx = 0
    active = []
    waiting = []

    while queue_idx < len(queue) or active or waiting:
        while queue_idx < len(queue) and queue[queue_idx].arrival_step <= step:
            waiting.append(queue[queue_idx])
            queue_idx += 1

        while waiting and len(active) < batch_size:
            r = waiting.pop(0)
            r.start_step = step
            active.append(r)

        if not active:
            if waiting:
                step += 1
                continue
            elif queue_idx < len(queue):
                step = queue[queue_idx].arrival_step
                continue
            else:
                break

        for r in active:
            r.tokens_generated += 1

        done = [r for r in active if r.is_done()]
        for r in done:
            r.end_step = step + 1
            completed.append(r)
        active = [r for r in active if not r.is_done()]

        step += 1

    return completed


def batching_stats(completed):
    latencies = [r.end_step - r.arrival_step for r in completed]
    total_time = max(r.end_step for r in completed) - min(r.arrival_step for r in completed)
    total_tokens = sum(r.output_tokens for r in completed)
    return {
        "avg_latency": np.mean(latencies),
        "p50_latency": np.median(latencies),
        "p99_latency": np.percentile(latencies, 99),
        "total_time": total_time,
        "throughput": total_tokens / total_time if total_time > 0 else 0,
    }
```

### चरण 4: पूर्वनिर्धारित कैश

एक tri आधारित पूर्वावधान कैश जो साझा पूर्वावधानों के लिए KV प्रविष्टियों को संग्रहीत करता है।

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.kv_data = None
        self.hit_count = 0


class PrefixCache:
    def __init__(self, max_entries=1000):
        self.root = TrieNode()
        self.max_entries = max_entries
        self.total_entries = 0
        self.hits = 0
        self.misses = 0

    def _walk(self, token_ids):
        node = self.root
        depth = 0
        for tid in token_ids:
            if tid not in node.children:
                break
            node = node.children[tid]
            depth += 1
        return node, depth

    def lookup(self, token_ids):
        node, depth = self._walk(token_ids)
        if depth > 0:
            self.hits += 1
            current = self.root
            for tid in token_ids[:depth]:
                current = current.children[tid]
                current.hit_count += 1
            kv_entries = []
            current = self.root
            for tid in token_ids[:depth]:
                current = current.children[tid]
                if current.kv_data is not None:
                    kv_entries.append(current.kv_data)
            return depth, kv_entries
        self.misses += 1
        return 0, []

    def insert(self, token_ids, kv_per_token):
        node = self.root
        for i, tid in enumerate(token_ids):
            if tid not in node.children:
                if self.total_entries >= self.max_entries:
                    return i
                node.children[tid] = TrieNode()
                self.total_entries += 1
            node = node.children[tid]
            if i < len(kv_per_token):
                node.kv_data = kv_per_token[i]
        return len(token_ids)

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

### चरण 5: अनुमानित डिकोडिंग सिम्युलेटर

हम विन्यास योग्य स्वीकृति दरों के साथ मसौदा-लक्ष्य अनुमानात्मक डिकोडिंग का अनुकरण करते हैं।

```python
class DraftModel:
    def __init__(self, vocab_size, acceptance_rate=0.8):
        self.vocab_size = vocab_size
        self.acceptance_rate = acceptance_rate

    def generate(self, context, num_tokens):
        tokens = np.random.randint(0, self.vocab_size, size=num_tokens)
        return tokens

    def get_probs(self, context, token):
        probs = np.random.dirichlet(np.ones(self.vocab_size))
        return probs


class TargetModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def get_probs(self, context, tokens=None):
        if tokens is not None:
            return [np.random.dirichlet(np.ones(self.vocab_size)) for _ in tokens]
        return np.random.dirichlet(np.ones(self.vocab_size))


def speculative_decode(draft_model, target_model, context, num_speculative=5,
                       draft_cost=1.0, target_cost=10.0, verify_cost=12.0):
    total_tokens = 0
    total_cost = 0.0
    accepted_counts = []
    context = list(context)

    max_tokens = 100

    while total_tokens < max_tokens:
        draft_tokens = draft_model.generate(context, num_speculative)
        total_cost += draft_cost * num_speculative

        target_probs = target_model.get_probs(context, draft_tokens)
        total_cost += verify_cost

        accepted = 0
        for i, token in enumerate(draft_tokens):
            draft_p = draft_model.get_probs(context + list(draft_tokens[:i]), token)
            target_p = target_probs[i]

            r = np.random.random()
            acceptance_prob = min(1.0, target_p[token] / (draft_p[token] + 1e-10))

            if r < draft_model.acceptance_rate:
                accepted += 1
                context.append(token)
                total_tokens += 1
            else:
                new_token = np.random.choice(draft_model.vocab_size, p=target_p)
                context.append(new_token)
                total_tokens += 1
                break

        accepted_counts.append(accepted)

        if accepted == num_speculative:
            bonus_probs = target_model.get_probs(context)
            bonus_token = np.random.choice(draft_model.vocab_size, p=bonus_probs)
            context.append(bonus_token)
            total_tokens += 1

    sequential_cost = total_tokens * target_cost
    return {
        "total_tokens": total_tokens,
        "speculative_cost": total_cost,
        "sequential_cost": sequential_cost,
        "speedup": sequential_cost / total_cost if total_cost > 0 else 1.0,
        "avg_accepted": np.mean(accepted_counts),
        "acceptance_rate": np.mean(accepted_counts) / num_speculative,
    }


def compare_speculation_strategies(vocab_size=1000, num_trials=20):
    results = {}

    for name, acceptance_rate, spec_tokens in [
        ("Draft-target (8B->70B)", 0.78, 5),
        ("EAGLE", 0.85, 6),
        ("N-gram", 0.50, 4),
        ("No speculation", 0.0, 0),
    ]:
        if spec_tokens == 0:
            results[name] = {
                "speedup": 1.0,
                "acceptance_rate": 0.0,
                "avg_accepted": 0.0,
            }
            continue

        trial_results = []
        for _ in range(num_trials):
            draft = DraftModel(vocab_size, acceptance_rate=acceptance_rate)
            target = TargetModel(vocab_size)
            context = list(np.random.randint(0, vocab_size, size=10))
            result = speculative_decode(draft, target, context, num_speculative=spec_tokens)
            trial_results.append(result)

        results[name] = {
            "speedup": np.mean([r["speedup"] for r in trial_results]),
            "acceptance_rate": np.mean([r["acceptance_rate"] for r in trial_results]),
            "avg_accepted": np.mean([r["avg_accepted"] for r in trial_results]),
        }

    return results
```

### चरण 6: KV कैश मेमोरी प्रोफाइलर

वास्तविक मॉडल कॉन्फ़िगरेशन के लिए KV कैश मेमोरी आवश्यकताओं की गणना करें।

```python
MODEL_CONFIGS = {
    "Llama-3-8B": {
        "num_layers": 32, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 8, "gqa": True,
    },
    "Llama-3-70B": {
        "num_layers": 80, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 70, "gqa": True,
    },
    "Llama-3-405B": {
        "num_layers": 126, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 405, "gqa": True,
    },
    "Mistral-7B": {
        "num_layers": 32, "num_kv_heads": 8, "head_dim": 128,
        "model_params_b": 7, "gqa": True,
    },
    "GPT-4-est": {
        "num_layers": 120, "num_kv_heads": 96, "head_dim": 128,
        "model_params_b": 1800, "gqa": False,
    },
}


def kv_cache_memory(config, seq_len, dtype_bytes=2):
    per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * dtype_bytes
    total = per_token * seq_len
    return {
        "per_token_bytes": per_token,
        "per_token_kb": per_token / 1024,
        "total_bytes": total,
        "total_mb": total / (1024 ** 2),
        "total_gb": total / (1024 ** 3),
    }


def memory_budget(config, gpu_memory_gb, model_dtype_bytes=2, kv_dtype_bytes=2):
    model_memory_gb = config["model_params_b"] * 1e9 * model_dtype_bytes / (1024 ** 3)
    overhead_gb = gpu_memory_gb * 0.1
    available_for_kv = gpu_memory_gb - model_memory_gb - overhead_gb

    if available_for_kv <= 0:
        return {"error": "Model does not fit in GPU memory", "model_memory_gb": model_memory_gb}

    per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * kv_dtype_bytes
    max_tokens = int(available_for_kv * (1024 ** 3) / per_token)

    return {
        "gpu_memory_gb": gpu_memory_gb,
        "model_memory_gb": round(model_memory_gb, 1),
        "overhead_gb": round(overhead_gb, 1),
        "available_for_kv_gb": round(available_for_kv, 1),
        "max_total_tokens": max_tokens,
        "max_users_at_2k": max_tokens // 2048,
        "max_users_at_4k": max_tokens // 4096,
        "max_users_at_32k": max_tokens // 32768,
    }
```

## इसका प्रयोग करें

vLLM के साथः

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    tensor_parallel_size=4,
    enable_prefix_caching=True,
    max_model_len=8192,
    gpu_memory_utilization=0.9,
)

params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Explain inference optimization in one paragraph."], params)
```

पूर्वनिर्धारित कैशिंग + संरचित आउटपुट के लिए SGLang के साथः

```python
import sglang as sgl

@sgl.function
def classify(s, text):
    s += sgl.system("You are a classifier. Output JSON only.")
    s += sgl.user(f"Classify this text: {text}")
    s += sgl.assistant(sgl.gen("result", regex=r'\{"label": "(positive|negative|neutral)"\}'))

runtime = sgl.Runtime(model_path="meta-llama/Llama-3-70B-Instruct", tp_size=4)
sgl.set_default_backend(runtime)

results = classify.run_batch([
    {"text": "This product is amazing!"},
    {"text": "Terrible experience."},
    {"text": "It was okay I guess."},
])
```

TensorRT-LLM के साथः

```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("./llama-70b-trt-engine/", rank=0)

outputs = runner.generate(
    batch_input_ids=[tokenizer.encode("Explain KV caching.")],
    max_new_tokens=256,
    temperature=0.7,
)
```

## इसे भेजें

इस पाठ से उत्पन्न होता हैः
- `outputs/skill-inference-optimization.md`-- एलएलएम निष्कर्ष सेवा निदान और अनुकूलन के लिए एक कौशल

## व्यायाम

1. 4. 4K संदर्भ में Llama 3 70B के लिए, प्रत्येक के लिए अधिकतम समवर्ती उपयोगकर्ताओं की गणना 4xA100-80GB पर करें। INT4 के लिए KV क्वांटिज़ेशन को उपयोगकर्ता क्षमता का लगभग 4 गुना करना चाहिए।

2. पीजीपी उपयोग (प्रति चरण भरने वाले बैच स्लॉट का अंश) को ट्रैक करने के लिए निरंतर बैचिंग सिम्युलेटर का विस्तार करें। 50 अनुरोधों के साथ स्थैतिक और निरंतर बैचिंग दोनों के लिए समय के साथ प्लॉट उपयोग जिनकी आउटपुट लंबाई एक पारेटो वितरण (आकार = 1.5, पैमाने = 20) का पालन करती है। निरंतर बैचिंग को 80% उपयोग बनाए रखना चाहिए।

3. KV कैश का एक समूहबद्ध-सवाल ध्यान (GQA) संस्करण लागू करें जहां `num_kv_heads < num_query_heads`. Llama 3 70B 64 क्वेरी हेड का उपयोग करता है लेकिन केवल 8 KV हेड। मेमोरी बचत बनाम पूर्ण मल्टी-हेड ध्यान (8 गुना KV कैश आकार में कमी) की गणना करें।

4. एक पूर्वावलोकन कैश बनाएं जो LRU निष्कासन का उपयोग करता है। अधिकतम प्रविष्टियों को 500 पर सेट करें और 1,000 अनुरोध उत्पन्न करें जहां 60% 5 सामान्य पूर्वावधानों में से एक साझा करते हैं। हिट दर को मापें और असीमित कैश की तुलना करें। अच्छे निष्कासन के साथ, हिट दर 55% से ऊपर होनी चाहिए।

5. पेड़ आधारित अटकलों (ईएजीएलई -2) को लागू करने के लिए अनुमानात्मक डिकोडिंग सिम्युलेटर का विस्तार करें। के ड्राफ्ट टोकन की एक एकल श्रृंखला के बजाय, उम्मीदवारों का एक पेड़ उत्पन्न करें (उदाहरण के लिए, 3 स्तरों में से प्रत्येक में 2 शाखाएँ = 8 पत्ती उम्मीदवार) । सत्यापन दौर बनाम रैखिक अटकलों के लिए स्वीकार किए गए कुल टोकन की तुलना करें।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Prefill | "Processing the prompt" | Computing attention over all input tokens in parallel -- compute-bound because the full matrix multiplication keeps GPU cores busy |
| Decode | "Generating tokens" | Producing one token per forward pass, reading the full model weights each time -- memory-bound because compute finishes before the next weights arrive |
| KV cache | "Caching attention states" | Storing the key and value projections for all previous tokens so they are not recomputed at each decode step -- trades memory for compute |
| Continuous batching | "Dynamic batching" | Inserting new requests into the running batch as soon as any request finishes, evaluated at every decode iteration rather than waiting for the whole batch |
| PagedAttention | "Virtual memory for KV cache" | Allocating KV cache in fixed-size pages instead of contiguous blocks, eliminating memory fragmentation and enabling copy-on-write for shared prefixes |
| Speculative decoding | "Draft and verify" | Using a fast draft model to propose multiple tokens, then verifying them all in one target model forward pass -- mathematically exact, 2-3x speedup |
| EAGLE | "Self-speculative decoding" | A speculative decoding variant that trains a lightweight head on the target model's own hidden states, achieving higher acceptance rates than a separate draft model |
| Prefix caching | "Reusing system prompt KV" | Storing computed KV cache entries for common prefixes (system prompts, few-shot examples) and reusing them across requests to skip redundant prefill |
| Ops:byte ratio | "Arithmetic intensity" | The ratio of compute operations to memory bytes read -- determines whether a workload is compute-bound (high ratio) or memory-bound (low ratio) |
| Time to first token | "TTFT" | Latency from receiving a request to producing the first output token -- dominated by prefill time for long prompts |

## आगे पढ़ना

- Kwon et al., "पेज्ड ध्यान के साथ सेवा करने वाले बड़े भाषा मॉडल के लिए कुशल मेमोरी प्रबंधन" (2023) -- vLLM पेपर जिसने पेज्ड KV कैश प्रबंधन की शुरुआत की, अब इन्फरेन्स सेवा के लिए उद्योग मानक
- लेवीयथन और अन्य, "स्पेक्टेटिव डिकोडिंग के माध्यम से ट्रांसफार्मर से फास्ट इन्फेरेंस" (2023) -- आधारभूत पेपर जो साबित करता है कि ड्राफ्ट-सत्यापित अटकलें सटीक लक्ष्य मॉडल वितरण का उत्पादन करती हैं जबकि 2-3x गति प्राप्त करती हैं
- लि और अन्य, "एजीएलः अनुमानित नमूनाकरण को फीचर अनिश्चितता को पुनर्विचार करने की आवश्यकता है" (2024) -- एक अलग ड्राफ्ट मॉडल का उपयोग करने के बजाय लक्ष्य मॉडल की अपनी विशेषताओं पर एक प्रमुख को प्रशिक्षित करके उच्च स्वीकृति दर प्राप्त करता है
- झेंग एट एल., "एसजीएलएंगः स्ट्रक्चरल लैंग्वेज मॉडल प्रोग्राम का कुशल निष्पादन" (2024) -- प्रीफिक्स कैशिंग के लिए रेडिक्सएटेंशन और मल्टी-कॉल एलएलएम कार्यक्रमों के लिए एक प्रोग्रामिंग मॉडल की शुरूआत करता है
- विलियम्स एट अल, "रोफलाइनः मल्टीकोर आर्किटेक्चर के लिए एक अंतर्दृष्टिपूर्ण विज़ुअल परफॉर्मेंस मॉडल" (2009) -- मूल छतलाइन पेपर जिसने कंप्यूटिंग बनाम मेमोरी बोतल गड्ढे के बारे में तर्क के लिए ओएसःबाइट फ्रेमवर्क को औपचारिक बनाया
