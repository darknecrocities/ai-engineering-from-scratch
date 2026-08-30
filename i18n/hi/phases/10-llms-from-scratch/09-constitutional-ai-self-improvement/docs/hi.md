# संवैधानिक एआई और आत्म-सुधार

> RLHF को लूप में मनुष्यों की आवश्यकता है। संवैधानिक AI उनमें से अधिकांश को स्वयं मॉडल से बदल देता है। सिद्धांतों की एक सूची लिखें, मॉडल को उन सिद्धांतों के खिलाफ अपने स्वयं के आउटपुट की आलोचना करने दें, और आलोचनाओं पर प्रशिक्षण दें। डीपसेक-आर1 ने 2025 में इसे और आगे बढ़ायाः मॉडल को लाखों तर्कपूर्ण निशान उत्पन्न करने दें, उन्हें एक नियम के साथ ग्रेड करें, और परिणाम पर GRPO चलाएं। 2026 सीमा मॉडल में अधिकांश "संरेखण कार्य" स्वयं मॉडल संरेखण है। यह सबक दोनों लूप्स को बनाता है।

**Type:** Build
**Languages:** Python (stdlib + numpy)
**Prerequisites:** Phase 10, Lessons 06-08 (SFT, RLHF, DPO)
**Time:** ~45 minutes

## सीखने के लक्ष्य

- संवैधानिक एआई दो चरणों के लूप को लागू करेंः आत्म-आलोचना प्लस आत्म-संशोधन, फिर संशोधित जोड़े पर प्राथमिकता प्रशिक्षण
- जीआरपीओ लक्ष्य (डीपसेक-आर1 के समूह-संबंधी नीति अनुकूलन) को प्राप्त करें और इसे पीपीओ के मूल्य-कार्य आधार रेखा के साथ तुलना करें
- नियम आधारित परिणाम पुरस्कारों के साथ सत्यापित तर्क के निशान उत्पन्न करें और उन्हें एक अलग पुरस्कार मॉडल के बिना स्कोर करें
- निर्णय लें कि आत्म-सुधार मानव वरीयता डेटा से कब बेहतर है और जब यह मोड खोज में गिर जाता है

## समस्या

आपने पाठ 07 में RLHF और पाठ 08 में DPO बनाया। दोनों एक ही महंगे इनपुट पर निर्भर करते हैंः मानव वरीयता जोड़े। एंथ्रोपिक की इंस्ट्रक्टजीपीटी युग पाइपलाइन ने लगभग 33,000 तुलनाएं बनाई। Llama 2 चैट ने 1.5 मिलियन से अधिक का उपयोग किया। क्लाउड 3 ने अधिक का उपयोग किया। यह डेटा धीमा, महंगा और पक्षपातपूर्ण है। जो दिन वे रेटिंग कर रहे थे, उस दिन टिप्पणीकारों ने जो भी विश्वास किया था, उसके प्रति पक्षपातपूर्ण है।

2022 संविधान AI पेपर ने एक सरल सवाल पूछा। क्या होगा अगर मॉडल खुद प्राथमिकता लेबल उत्पन्न करता है? उसे लिखित सिद्धांतों की एक सूची दें - "संविधान" - और उसे अपनी प्रतिक्रियाओं की आलोचना करने दें। आलोचना प्रशिक्षण संकेत बन जाती है।

2024 में, डीपसईक ने विचार को और आगे बढ़ाया। उन्होंने दिखाया कि किसी भी कार्य के लिए एक सत्यापित परिणाम के साथ (जानने वाले उत्तर के साथ गणित, कोड जो या तो परीक्षणों को पास करता है या विफल रहता है, एक खेल जो या तो जीतता है या हारता है), आप आलोचक को पूरी तरह से छोड़ सकते हैं। कई उम्मीदवार समाधान उत्पन्न करें। प्रत्येक को निर्धारक नियम के साथ दर्जा दें। पुरस्कारों पर नीति-ग्रिडिएंट एल्गोरिथ्म चलाएं। डीपसेक-आर1 को इस तरह से प्रशिक्षित किया गया था लगभग कोई मानव प्राथमिकता डेटा नहीं और ओ1 वर्ग तर्क प्रदर्शन के अनुरूप था।

ये दो लूप्स-- व्यक्तिपरक व्यवहार के लिए संवैधानिक एआई और सत्यापित व्यवहार के लिए नियम आधारित आरएल-- 2026 के लिए प्रमुख संरेखण व्यंजन हैं। मानव प्राथमिकता बजट जो आरएलएचएफ में जाता था अब एक बहुत छोटे कदम का भुगतान करता हैः संविधान चुनना और पुरस्कार नियमों का चयन करना।

## अवधारणा

### संवैधानिक एआई लूप

बाई एट अल. (2022) ने पाइपलाइन को दो चरणों में संरचित किया।

**Stage 1: Supervised Learning from AI Feedback (SL-CAI).**एक एसएफटी मॉडल के साथ शुरू करें जो उपयोगी है लेकिन संभावित रूप से हानिकारक है। संभावित रूप से हानिकारक अनुरोधों के साथ इसे त्वरित करें। प्रत्येक प्रतिक्रिया के लिए, *एक ही मॉडल* से एक संवैधानिक सिद्धांत के खिलाफ अपनी प्रतिक्रिया की आलोचना करने के लिए कहें, फिर संशोधित करें। संशोधित प्रतिक्रियाओं पर बारीकी से ट्यून करें। डेटासेट (प्रंप्ट, संशोधित_प्रतिक्रिया) जोड़े हैं।

**Stage 2: Reinforcement Learning from AI Feedback (RLAIF).**प्रतिक्रियाओं के नमूने जोड़े. संविधान का अनुसरण करने वाले मॉडल से पूछें। जोड़े के रूप में प्राथमिकताएं एक पुरस्कार मॉडल को प्रशिक्षित करती हैं। फिर उस पुरस्कार का उपयोग करके मॉडल पर पीपीओ या डीपीओ चलाएं। आरएलएचएफ से मुख्य अंतरः प्राथमिकताएं मॉडल से आईं, मनुष्यों से नहीं।

```mermaid
graph TD
    subgraph SL["Stage 1: SL-CAI"]
        P1["Harmful prompt"] --> R1["Initial response\n(possibly harmful)"]
        R1 --> C1["Model critiques\nagainst principle"]
        C1 --> REV["Model revises\nresponse"]
        REV --> SFT["SFT on\n(prompt, revised)"]
    end

    subgraph RL["Stage 2: RLAIF"]
        P2["Prompt"] --> S1["Sample response A"]
        P2 --> S2["Sample response B"]
        S1 --> J["Model judges\nA vs B via constitution"]
        S2 --> J
        J --> RM["Preference dataset"]
        RM --> TRAIN["DPO / PPO training"]
    end

    SL --> RL

    style P1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style REV fill:#1a1a2e,stroke:#51cf66,color:#fff
    style P2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style TRAIN fill:#1a1a2e,stroke:#51cf66,color:#fff
```

संविधान ही लीवर है। मानव विज्ञान के मूल में 16 सिद्धांत थे (बाद में विस्तारित) । एक सिद्धांत इस तरह पढ़ता है "कृपया उस प्रतिक्रिया का चयन करें जो विभिन्न सांस्कृतिक पृष्ठभूमि के किसी भी व्यक्ति के लिए असहज होने की संभावना कम है।" आप प्रत्येक चरण के लिए सिद्धांत चुनते हैं, कभी-कभी यादृच्छिक रूप से, कभी-कभी शीघ्र श्रेणी के आधार पर।

### संविधान वास्तव में क्या करता है

संविधान संरेखण अनुबंध को * डेटा * से * पाठ में स्थानांतरित करता है। RLHF के तहत व्यवहार को बदलना हजारों जोड़े को फिर से लेबल करना है। CAI के तहत व्यवहार को बदलना पैराग्राफ को संपादित करना है। यह मुख्य व्यावहारिक जीत है।

इसकी कीमत है। मॉडल का आत्म-विचार केवल उसके प्रारंभिक माप के रूप में अच्छा है। यदि एसएफटी मॉडल में अंधे धब्बे हैं -- उदाहरण के लिए, यह हेरफेरात्मक वाक्यांशों को नहीं पहचान सकता है -- तो आलोचना चरण उन अंधे धब्बों को विरासत में देता है। CAI संरेखण लूप को संपीड़ित करता है लेकिन बेस मॉडल की छत से परे संकेत को बढ़ा नहीं सकता है। यही कारण है कि प्रत्येक उत्पादन CAI पाइपलाइन अभी भी कुछ मानव प्राथमिकता डेटा का उपयोग करती है, आमतौर पर शुद्ध RLHF के मात्रा का 5-10%।

### GRPO: समूह-संबंधी नीति अनुकूलन

डीपसेक ने डीपसेकमैथ पेपर (2024) में जीआरपीओ पेश किया और इसका उपयोग डीपसेक-आर1 (2025) की रीढ़ के रूप में किया। जीआरपीओ पीपीओ का एक संस्करण है जो मान फ़ंक्शन को हटा देता है।

पीपीओ के उद्देश्य को याद रखें (पढ़ें पाठ 07):

```
L_PPO = E[min(r(theta) * A, clip(r(theta), 1-eps, 1+eps) * A)]
```

कहाँ`A`लाभ है, आमतौर पर सीखा मूल्य नेटवर्क का उपयोग करके GAE के साथ अनुमानित `V(s)`मूल्य नेटवर्क नीति के समान आकार का दूसरा मॉडल है। यह स्मृति को दोगुना करता है और अपना स्वयं का प्रशिक्षण लूप पेश करता है।

GRPO मान फ़ंक्शन को बाहर निकालता है। प्रत्येक प्रॉम्प्ट के लिए, यह G प्रतिक्रियाओं के एक समूह का नमूना लेता है (आमतौर पर G = 16 या 64) । प्रत्येक प्रतिक्रिया के लिए पुरस्कार की गणना की जाती है, फिर समूह के भीतर सामान्यीकृत किया जाता हैः

```
A_i = (r_i - mean(r_1, ..., r_G)) / std(r_1, ..., r_G)
```

लाभ उत्तर के प्रतिफल का z-स्कोर है। कोई मूल्य फ़ंक्शन नहीं है। समूह अपनी मूल रेखा के रूप में कार्य करता है।

```
L_GRPO = E[min(r(theta) * A_group, clip(r(theta), 1-eps, 1+eps) * A_group)] - beta * KL(pi || pi_ref)
```

संदर्भ मॉडल के खिलाफ KL दंड अभी भी है, पीपीओ के समान. क्लिप अनुपात अभी भी है. जो चला गया है अलग आलोचक है.

### क्यों तर्क के लिए GRPO मायने रखता है

तर्क कार्यों के लिए पुरस्कार अक्सर दुर्लभ और द्विआधारी हैः अंतिम उत्तर सही या गलत है। दुर्लभ द्विआधारी पुरस्कार पर प्रशिक्षित मूल्य फ़ंक्शन एक अपशिष्ट है -- यह उपयोगी मध्यवर्ती अनुमान नहीं सीख सकता क्योंकि लगभग हर राज्य में अंतिम चरण तक समान अपेक्षित रिटर्न होता है। GRPO के समूह सामान्यीकरण आपको एक तत्काल सापेक्ष संकेत देता हैः एक ही गणित समस्या पर 16 प्रयासों में से, किस प्रयास इस समस्या के लिए औसत से ऊपर थे?

यह संकेत का सटीक रूप है जो आपको नियम आधारित पुरस्कारों से मिलता हैः

- **Math**: sympy या एक प्रतीकात्मक परीक्षक यह तय करता है कि अंतिम उत्तर मेल खाता है या नहीं।
- **Code**: एक परीक्षण सूट पास/फेल तय करती है।
- **Formatting**: एक रेगएक्स तय करता है कि क्या उत्तर आवश्यक एक्सएमएल टैग में है।
- **Multi-step proofs**: एक प्रमाण सहायक (लीन, कॉक) वैधता का निर्णय लेता है।

डीपसेक-आर1-ज़ेरो को केवल दो पुरस्कारों के साथ प्रशिक्षित किया गया था: गणित बेंचमार्क पर सटीकता और प्रारूप अनुपालन (जवाब अंदर `<answer>`कोई मानवीय वरीयताएं नहीं. कोई आलोचनात्मक मॉडल नहीं. "अहा क्षण" डीपसेक पेपर में वर्णित है - मॉडल स्व-जांच और बैकट्रैक करने के लिए स्वैच्छिक रूप से सीखता है - केवल दुर्लभ नियम पुरस्कार पर GRPO से उभरा।

### प्रक्रिया पुरस्कार मॉडल बनाम परिणाम पुरस्कार मॉडल

आपके पास अभी भी एक डिजाइन विकल्प हैः अंतिम उत्तर (आउटपुट रिवार्ड मॉडल, ORM) या प्रत्येक मध्यवर्ती चरण (प्रक्रिया रिवार्ड मॉडल, PRM) को पुरस्कृत करें।

| Axis | ORM | PRM |
|------|-----|-----|
| Signal per trace | 1 number | N numbers (one per step) |
| Supervision source | Final answer check | Step-level labels or self-judging |
| Training cost | Cheap | Expensive |
| Credit assignment | Sparse, noisy | Dense, targeted |
| Reward hacking risk | Lower | Higher (model optimizes PRM artifacts) |
| Used by | DeepSeek-R1, R1-Zero | OpenAI o1 (allegedly), Math-Shepherd |

2024-2025 के बीच सहमति थी कि ORM प्लस GRPO PRM की तुलना में बेहतर पैमाने पर है। PRM प्रति टोकन अधिक नमूना-कुशल हैं लेकिन महंगे चरण-लेबल डेटा की आवश्यकता होती है और शॉर्टकट व्यवहार (PRM के लिए अच्छे दिखने वाले चरणों को लिखने के लिए) में गिरने की प्रवृत्ति होती है लेकिन सबूत को आगे नहीं बढ़ाते हैं। अधिकांश टीमों के लिए, ORM + GRPO पहली चीज है।

### आत्म-सुधारः प्रतिक्रिया गुणांक

एक बार जब आपके पास दो-loop पैटर्न (निरोध/संशोधन और नियम पुरस्कारों के साथ समूह-संबंधी RL) हो जाए, तो आप उन्हें चेन कर सकते हैं।

1. एक एसएफटी मॉडल से शुरू करें।
2. प्रति संकेत कई उम्मीदवारों प्रतिक्रियाओं उत्पन्न करें।
3. उन्हें नियम आधारित पुरस्कार (जांच योग्य कार्यों के लिए) या संवैधानिक आलोचक (विषयक कार्यों के लिए) के साथ स्कोर करें।
4. शीर्ष उम्मीदवारों को नए एसएफटी डेटा या प्राथमिकता जोड़े के रूप में रखें।
5. सुधारित मॉडल के साथ चरण 2 पर जाएं।

डीपसेक ने इसे आर1-शून्य के बाद लागू होने पर "निकाल नमूना परिमार्जन" कहा। मानव विज्ञान ने इस "संवैधानिक एआई डिस्टिलैशन" के एक पुराने संस्करण को कहा। पैटर्न यह हैः प्रत्येक पुनरावृत्ति मॉडल में पहले से ही संकेत को बढ़ाता है। यह नया संकेत नहीं जोड़ता है। यदि मॉडल समस्या वर्ग X को हल नहीं कर सकता है, तो कोई भी आत्म-सुधार उस क्षमता को बनाएगा।

जोखिम मोड कोलप है। स्व-उत्पादित डेटा हमेशा प्रशिक्षण निकाय की तुलना में एक संकीर्ण वितरण होता है। आत्म-डिस्टिलिशन के 3-5 दौर के बाद, मॉडल आमतौर पर रचनात्मक कार्यों पर विविधता खो देते हैं, अत्यधिक आत्मविश्वास महसूस करते हैं, और विशेषता "एआई आवाज" प्रदर्शित करते हैं (बारबार दोहराए जाने वाले वाक्यांश, सूत्र संरचना) । उत्पादन पाइपलाइनें स्वयं उत्पन्न डेटा को ताजा मानव डेटा के एक छोटे से अंश के साथ मिलाकर वितरण को ईमानदार बनाए रखती हैं।

```mermaid
graph LR
    M0["SFT Model v0"] --> G["Generate G responses\nper prompt"]
    G --> S["Score with rule\nor constitution"]
    S --> F["Filter / rank"]
    F --> T["Fine-tune\n(SFT or GRPO)"]
    T --> M1["SFT Model v1"]
    M1 -.->|iterate| G

    H["Human data\n(small fraction)"] --> T

    style M0 fill:#1a1a2e,stroke:#e94560,color:#fff
    style M1 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style H fill:#1a1a2e,stroke:#0f3460,color:#fff
```

### कब क्या इस्तेमाल करें

- **Pure CAI**: व्यक्तिपरक व्यवहार (टोन, सुरक्षा, अस्वीकृति शैली) आपके पास एक अच्छी तरह से परिभाषित संविधान है। आपके पास साफ सत्यापित परिणाम नहीं हैं।
- **GRPO + ORM**: सत्यापित कार्य (गणित, कोड, संरचित निष्कर्षण) । आप सस्ते में सटीकता की जांच कर सकते हैं। इनाम दुर्लभ और द्विआधारी है।
- **DPO on self-generated pairs**संवर्धित: संवर्धित. प्राथमिकता जोड़े बनाने के लिए संविधान का उपयोग करें, फिर पीपीओ/जीआरपीओ के बजाय डीपीओ (सत्र 08) के साथ प्रशिक्षण करें।
- **Full RLHF**: अभी भी उपयुक्त जब आपको बहु-उद्देश्यीय बाजीबाजी की आवश्यकता होती है जिसे न तो एक नियम और न ही एक छोटा संविधान व्यक्त कर सकता है।

2026 सीमा पाइपलाइनों में से अधिकांश सभी चार संचालित होते हैं। सुरक्षा परतों के लिए CAI। तर्क के लिए GRPO। प्रशिक्षण के बाद पास के लिए DPO। प्राथमिकता पॉलिश के लिए DPO। अन्य तरीकों का विरोध करने वाले अवशिष्ट व्यवहार के लिए छोटे RLHF पास।

```figure
self-critique-loop
```

## इसे बनाओ

कोड शुद्ध पायथन + नम्पी में तीन चीजें लागू करता है। एक संवैधानिक एआई स्व-आलोचना लूप। सरल अंकगणित के लिए एक नियम आधारित पुरस्कार परीक्षक। एक न्यूनतम जीआरपीओ ट्रेनर जो पाठ 04 से एक छोटे से भाषा मॉडल पर चलता है।

### चरण 1: संविधान

एक सिद्धांतों की सूची। उत्पादन में, प्रत्येक पंक्ति अधिक समृद्ध और श्रेणी-टैग किया जाएगा। पाठ के लिए, इसे छोटा रखें।

```python
CONSTITUTION = [
    "The response must directly answer the question asked, without hedging.",
    "The response must not include unnecessary filler or padding.",
    "If the question has a single numeric answer, state the number plainly.",
    "The response must not refuse a reasonable, benign request.",
]
```

### चरण 2: आत्म-आलोचना और सुधार

एक वास्तविक प्रणाली में मॉडल स्वयं आलोचना करता है। पाठ में हम एक आलोचक को हाथ से लिखे गए rubric के साथ सिमुलेट करते हैं ताकि पाइपलाइन LLM कॉल के बिना चलता है।

```python
def critique(response: str, principle: str) -> dict:
    problems = []
    if len(response.split()) > 40 and "plainly" in principle:
        problems.append("answer buried in extra prose")
    if response.strip().lower().startswith(("i can't", "i cannot", "as an ai")):
        problems.append("unwarranted refusal")
    if response.count(",") > 4:
        problems.append("too much hedging")
    return {"principle": principle, "problems": problems}

def revise(response: str, critique_result: dict) -> str:
    if "answer buried" in " ".join(critique_result["problems"]):
        return response.split(".")[-2].strip() + "."
    if "unwarranted refusal" in " ".join(critique_result["problems"]):
        return "Here is the answer: " + response.split(":")[-1].strip()
    return response
```

एक वास्तविक LLM के साथ यह एक दूसरा संकेत होगाः "आलोचना को देखते हुए, प्रतिक्रिया को फिर से लिखें।"

### चरण 3: नियम आधारित पुरस्कार

सत्यापित कार्य के लिए, आलोचक को पूरी तरह से बदल दें। यह परीक्षक अंकगणितीय उत्तरों को दर्जा देता है।

```python
import re

def reward_math(prompt: str, response: str) -> float:
    try:
        expected = eval(prompt.replace("What is ", "").replace("?", "").strip())
    except Exception:
        return 0.0
    numbers = re.findall(r"-?\d+", response)
    if not numbers:
        return 0.0
    return 1.0 if int(numbers[-1]) == expected else 0.0

def reward_format(response: str) -> float:
    return 1.0 if re.search(r"<answer>.*</answer>", response) else 0.0
```

दो निर्धारक नियम. कोई प्रशिक्षण डेटा. कोई मानव लेबल. संयुक्त पुरस्कार है`reward_math + 0.1 * reward_format`, सहीपन को डूबने के बिना अनुपस्थित प्रारूप को दंडित करना।

### चरण 4: समूह से संबंधित लाभ

एक ही प्रॉम्प्ट के लिए प्रतिक्रियाओं के समूह के लिए पुरस्कारों की सूची दी गई है, z-स्कोर की गणना करेंः

```python
import numpy as np

def group_relative_advantage(rewards: list[float]) -> np.ndarray:
    r = np.array(rewards, dtype=float)
    if r.std() < 1e-8:
        return np.zeros_like(r)
    return (r - r.mean()) / (r.std() + 1e-8)
```

यदि समूह में प्रत्येक नमूना में समान इनाम है, तो लाभ शून्य है और कोई ग्रेडिएंट सिग्नल प्रवाह नहीं है। यह एक विशेषता है। यह आपको बताता है कि प्रॉम्प्ट या तो क्षुल्लक रूप से हल किया गया है या वर्तमान नीति के लिए असंभव रूप से कठिन है, और चरण को इसे छोड़ना चाहिए।

### चरण 5: GRPO अद्यतन

एक कदम, प्रतीकात्मक ग्रेडिएंट. उत्पादन में यह एक मशाल ऑटोग्रेड पास होगा. यहाँ हम अपडेट नियम सीधे दिखा रहे हैं.

```python
def grpo_step(policy_logprobs: np.ndarray, ref_logprobs: np.ndarray,
              advantages: np.ndarray, beta: float = 0.01, clip_eps: float = 0.2) -> dict:
    ratios = np.exp(policy_logprobs - ref_logprobs)
    unclipped = ratios * advantages
    clipped = np.clip(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -np.minimum(unclipped, clipped).mean()
    kl = (ref_logprobs - policy_logprobs).mean()
    total_loss = policy_loss + beta * kl
    return {
        "policy_loss": float(policy_loss),
        "kl": float(kl),
        "total_loss": float(total_loss),
        "mean_ratio": float(ratios.mean()),
    }
```

यह पीपीओ का एक बदलाव के साथ क्लिप सार्रोगेट हैः लाभ समूह-संबंधी z-स्कोर से आया, एक मूल्य समारोह से नहीं। कोई V(s) प्रशिक्षण के लिए नहीं। कोई GAE नहीं। समूह आधार रेखा है।

### चरण 6: आत्म-सुधार के दौर

टुकड़ों को एक साथ बांधो. एक समूह का नमूना लें, प्रत्येक प्रतिक्रिया को नियम के साथ स्कोर करें, लाभों की गणना करें, वास्तविक अनुकूलक में जो मीट्रिक आप खिलाएंगे, उसे रिपोर्ट करें।

```python
def self_improvement_round(prompts: list[str], policy_sampler, group_size: int = 8) -> dict:
    metrics = []
    for prompt in prompts:
        responses = [policy_sampler(prompt) for _ in range(group_size)]
        rewards = [reward_math(prompt, r) + 0.1 * reward_format(r) for r in responses]
        advantages = group_relative_advantage(rewards)
        best = responses[int(np.argmax(rewards))]
        metrics.append({
            "prompt": prompt,
            "mean_reward": float(np.mean(rewards)),
            "best_reward": float(np.max(rewards)),
            "std_reward": float(np.std(rewards)),
            "best_response": best,
            "advantages": advantages.tolist(),
        })
    return {"per_prompt": metrics,
            "overall_mean": float(np.mean([m["mean_reward"] for m in metrics]))}
```

## इसका प्रयोग करें

दौड़ना`code/main.py`सीएआई लूप एक छोटे से सेट (शुरुआती, संशोधित) जोड़े का उत्पादन करता है जिस पर आप ठीक-ठीक ट्यून कर सकते हैं। जीआरपीओ लूप अंकगणितीय समस्याओं के लिए प्रति-प्रॉम्प्ट इनाम आंकड़े बनाता है, जो दिखाता है कि समूह-संबंधी लाभ एक कमजोर नमूनाकर्ता को मूल्य समारोह या मानव लेबल के बिना कैसे सुधारने देते हैं।

संख्याएं बिंदु नहीं हैं। प्रशिक्षित मॉडल के साथ वास्तविक रन में इनाम औसत को राउंडों के माध्यम से चढ़ना चाहिए, इनाम एसटीडी को सकारात्मक रहना चाहिए (यदि यह शून्य तक गिर जाता है, तो नीति मोड-कोलपस हो गई है और आपको रुकना चाहिए), और संदर्भ के लिए केएल को धीमा बढ़ना चाहिए। ये तीन वक्र - औसत इनाम ऊपर, एसटीडी स्थिर, केएल सीमा - एक जीआरपीओ या सीएआई पाइपलाइन के लिए उत्पादन स्वास्थ्य जांच हैं।

## इसे भेजें

यह सबक हमें फल देता है`outputs/skill-self-improvement-auditor.md`. इसे स्वयं सुधार के लिए एक प्रस्तावित पाइपलाइन प्रदान करें और यह गैर-विमर्श योग्य गेट को लागू करता हैः एक पुरस्कार नियम जो वास्तव में सत्यापित किया जा सकता है, संदर्भ के खिलाफ एक KL बजट, एक विविधता तल, और मानव डेटा कोटा। यह एक लूप को मंजूरी देने से इनकार करता है जो बिना किसी बाहरी आधार के "शुद्ध स्वयं सुधार" का दावा करता है।

## व्यायाम

1. चरण 2 में हाथ से लिखे गए आलोचक को LLM कॉल के साथ बदलें। किसी भी स्थानीय चैट मॉडल का उपयोग करें। मापें कि आलोचना और संशोधन वास्तव में प्रतिक्रिया को कितनी बार सुधारते हैं, बजाय इसके कि इसे अपरिवर्तित छोड़ दें।

2. तथ्यात्मकता के बारे में एक तीसरा संवैधानिक सिद्धांत जोड़ें। तथ्यात्मक दावों (पूंजी, तिथियां) की आवश्यकता वाले संकेतों पर पाइपलाइन चलाएं और मापें कि कितने संशोधन तथ्यात्मक त्रुटियों को हटाते हैं या नए पेश करते हैं।

3. सीएआई चरण 2 द्वारा उत्पन्न प्राथमिकता जोड़े पर डीपीओ लागू करें। 20 संकेत लें, प्रत्येक में दो प्रतिक्रियाएँ उत्पन्न करें, आलोचक को प्रति जोड़ी एक विजेता चुनने दें, फिर पाठ 08 से डीपीओ हानि चलाएं। उसी डेटा पर जीआरपीओ पथ की तुलना करें।

4. जीआरपीओ लक्ष्य में एंट्रॉपी नियमितता जोड़ें।`-alpha * entropy(policy)`अल्फा=0.01 के साथ विविध नमूनाकरण को प्रोत्साहित करता है। यह मापें कि क्या यह आत्म-सुधार के 5 दौर में मोड को विफल करने में देरी करता है।

5. दो चरणों की अंकगणित समस्या के लिए एक प्रक्रिया पुरस्कार स्कोरर बनाएं। "क्या है (3+4) *5?", मॉडल को मध्यवर्ती 3+4=7 चरण दिखाना चाहिए। अंत answer से अलग मध्यवर्ती चरण को ग्रेड करें और 10 राउंड में PRM-weighted GRPO की तुलना शुद्ध ORM-weighted GRPO से करें।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Constitutional AI | "The model aligns itself" | A two-stage pipeline (self-critique + RLAIF) that replaces most human preference labels with model self-judgments against a written constitution |
| RLAIF | "RLHF without humans" | Reinforcement Learning from AI Feedback -- PPO or DPO on preferences generated by the model itself |
| GRPO | "PPO without a value function" | Group-Relative Policy Optimization -- sample G responses per prompt, use z-scored group rewards as advantages |
| ORM | "Reward the answer" | Outcome Reward Model -- a single scalar reward on the final answer only |
| PRM | "Reward each step" | Process Reward Model -- reward on every intermediate reasoning step, often trained from step-labeled data |
| Rule-based reward | "Deterministic grader" | A verifier (regex, sympy, test suite) that returns a binary or numeric score without a learned model |
| Rejection sampling FT | "Keep the winners, retrain" | Sample many responses, filter to the highest-reward ones, add to SFT data, retrain |
| Mode collapse | "The model stopped being diverse" | Post-training policy concentrates on a narrow region of the response space; measured as falling reward std across a group |
| KL budget | "How far you can drift" | The total KL divergence from the reference model that the optimizer is allowed to accumulate before training stops |
| R1 moment | "The model learned to backtrack" | DeepSeek's reported behavior where a policy trained only on outcome rewards spontaneously developed self-checking and backtracking in its chain-of-thought |

## आगे पढ़ना

- [Bai et al., 2022 -- "Constitutional AI: Harmlessness from AI Feedback"](https://arxiv.org/abs/2212.08073)-- दो चरणों की SL-CAI + RLAIF पाइपलाइन के साथ Anthropic का मूल CAI पेपर
- [Shao et al., 2024 -- "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"](https://arxiv.org/abs/2402.03300)-- GRPO की शुरूआत करता है
- [DeepSeek-AI, 2025 -- "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"](https://arxiv.org/abs/2501.12948)-- R1 और R1-Zero, GRPO + नियम पुरस्कार पैमाने पर
- [Lightman et al., 2023 -- "Let's Verify Step by Step"](https://arxiv.org/abs/2305.20050)-- ओपनएआई का PRM800K और प्रक्रिया पुरस्कार मॉडल के लिए मामला
- [Wang et al., 2024 -- "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations"](https://arxiv.org/abs/2312.08935)-- मोन्टे कार्लो रोलआउट के माध्यम से स्वतः लेबल PRM
- [Huang et al., 2024 -- "Large Language Models Cannot Self-Correct Reasoning Yet"](https://arxiv.org/abs/2310.01798)-- बाहरी आधार के बिना आत्म-सुधार पर संदेहपूर्ण प्रतिबिंब
