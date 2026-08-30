# निर्देश ट्यूनिंग (SFT)

> एक बेस मॉडल अगले टोकन की भविष्यवाणी करता है. यह है. यह निर्देशों का पालन नहीं करता है, प्रश्नों का जवाब नहीं देता है, या हानिकारक अनुरोधों को अस्वीकार नहीं करता है. एसएफटी एक टोकन भविष्यवाणी और एक उपयोगी सहायक के बीच पुल है. हर मॉडल आप कभी बात की है - क्लाउड, जीपीटी, लामा चैट - इस कदम से गुजर गया है.

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10, Lesson 04 (Pre-Training a Mini GPT)
**Time:** ~90 minutes

## सीखने के लक्ष्य

- एक मूल भाषा मॉडल को निर्देशों का पालन करने वाले सहायक में परिवर्तित करने के लिए पर्यवेक्षित सूक्ष्म-ट्यूनिंग (SFT) को लागू करना
- सिस्टम, उपयोगकर्ता और सहायक भूमिकाओं के साथ चैट टेम्पलेट्स का उपयोग करके प्रशिक्षण डेटा प्रारूपित करें, और गैर-सहायक टोकन पर मास्क हानि
- एसएफटी की आवश्यकता क्यों है, यह समझाएंः आधार मॉडल प्रश्नों के उत्तर देने के बजाय पाठ को जारी रखते हैं
- एक लंबे समय तक चल रहे निर्देश सेट पर आधार मॉडल बनाम बारीक-की-करने वाले मॉडल प्रतिक्रियाओं की तुलना करके एसएफटी गुणवत्ता का मूल्यांकन करें

## समस्या

आप पाठ 04 में एक मॉडल प्रशिक्षित किया है. यह एक अनुक्रम दिया अगले टोकन की भविष्यवाणी कर सकते हैं. इसे "ट्रांसफॉर्मर वास्तुकला" खिला और यह "प्राकृतिक भाषा प्रसंस्करण में क्रांति लाया है" के साथ जारी रख सकता है. यह एक अगले टोकन भविष्यवाणी के लिए प्रभावशाली है.

अब यह कोशिश करोः इसे "फ्रांस की राजधानी क्या है? " एक आधार मॉडल "पेरिस" का जवाब नहीं देता। यह पैटर्न जारी रखता है। यह "जर्मनी की राजधानी क्या है? स्पेन की राजधानी क्या है?" क्योंकि उसने उन दस्तावेजों से सीखा जिसमें प्रश्नों की सूची है। या यह "एक सवाल है कि कई लोगों को पूछता है" पैदा कर सकता है क्योंकि यह एक संभावित अगले टोकन के निरंतरता है. मॉडल में *उत्तर देने* की कोई अवधारणा नहीं है। यह केवल जानता है * जारी रखने के लिए।

यह GPT-3 (आधार मॉडल, जून 2020 में जारी) और ChatGPT (निर्देश-ट्यून, नवंबर 2022 में जारी) के बीच अंतर है। एक ही वास्तुकला। एक ही पूर्व-प्रशिक्षण। अंतर 20,000 से 100,000 सावधानीपूर्वक तैयार किए गए (निर्देश, प्रतिक्रिया) जोड़े है जो मॉडल को बातचीत पैटर्न का पालन करने के लिए सिखाया।

स्टैनफोर्ड अल्पाका ने साबित किया कि आपको लाखों उदाहरणों की आवश्यकता नहीं है। मार्च 2023 में, उन्होंने GPT-3.5 द्वारा उत्पन्न केवल 52,000 निर्देश-प्रतिसाद जोड़े पर Llama 7B को ठीक से समायोजित किया। कुल लागतः $600. The result was a chatbot that could follow instructions, answer questions, and hold conversations. Not as good as ChatGPT, but shockingly close for $600 और कुछ घंटे का प्रशिक्षण।

मेटा के Llama 2 चैट ने अपने प्रारंभिक SFT चरण के लिए केवल ~27,000 उच्च गुणवत्ता वाले उदाहरणों का उपयोग किया। मुख्य अंतर्दृष्टिः गुणवत्ता मात्रा से अधिक मायने रखती है। कुशल टिप्पणीकारों द्वारा लिखे गए 27,000 उदाहरण इंटरनेट से स्क्रैप किए गए 1 मिलियन शोर उदाहरणों को हराते हैं।

## अवधारणा

### एसएफटी वास्तव में क्या करता है

पर्यवेक्षित ठीक-ठीक प्रशिक्षण पहले से ही प्रशिक्षण लूप जारी रखता है - आगे पास, गणना हानि, पीछे पास, अद्यतन वजन - लेकिन एक अलग प्रकार के डेटा पर। कच्चे पाठ के बजाय, आप संरचित बातचीत पर प्रशिक्षण देते हैंः

```json
{
  "system": "You are a helpful assistant.",
  "user": "What is the capital of France?",
  "assistant": "The capital of France is Paris."
}
```

मॉडल पहले से ही जानता है कि पेरिस फ्रांस की राजधानी है। यह विकिपीडिया, पाठ्यपुस्तकों और वेब पृष्ठों पर पूर्व-शिक्षण के दौरान सीखा गया था। एसएफटी मॉडल को नए तथ्य नहीं सिखाता है। यह मॉडल को एक नया *व्यवहार* सिखाता हैः जब आप एक प्रश्न देखते हैं, तो एक उत्तर उत्पन्न करें। जब आप एक निर्देश देखते हैं, तो एक पूरा करें। जब आप एक हानिकारक अनुरोध देखते हैं, तो एक अस्वीकार उत्पन्न करें।

इसे इस तरह से सोचिए. पूर्व प्रशिक्षण मॉडल ज्ञान देता है. एसएफटी मॉडल शिष्टाचार देता है.

### डेटा प्रारूप

तीन प्रारूप उद्योग पर हावी हैं. प्रत्येक एक ही जानकारी को कोड करता है - किसने क्या कहा - अलग-अलग सीमांकन के साथ।

**Alpaca Format**(स्टैनफोर्ड, मार्च 2023):

```json
{
  "instruction": "Summarize the following article in 3 sentences.",
  "input": "The European Central Bank raised interest rates...",
  "output": "The ECB increased rates by 25 basis points..."
}
```

सरल और व्यापक रूप से प्रयोग किया जाता है।`input`स्टैनफोर्ड ने इस प्रारूप में 52,000 उदाहरण जारी किए, $600 के लिए GPT-3.5 द्वारा उत्पन्न। इसने ओपन सोर्स निर्देश ट्यूनिंग आंदोलन की शुरुआत की।

**ShareGPT Format**(संयुक्त राष्ट्र, 2023):

```json
{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "What causes tides?"},
    {"from": "gpt", "value": "Tides are caused by the gravitational pull of the Moon..."},
    {"from": "human", "value": "How often do they occur?"},
    {"from": "gpt", "value": "Most coastal areas experience two high tides and two low tides per day..."}
  ]
}
```

"फॉर" फ़ील्ड में "मानव" और "gpt" का उपयोग कन्वेंशन द्वारा किया जाता है, भले ही वास्तविक मॉडल का कोई फर्क न पड़े। Vicuna को 70,000 ShareGPT वार्तालापों पर प्रशिक्षित किया गया था जो उपयोगकर्ता-साझा ChatGPT प्रतिलेखन से स्क्रैप किए गए थे।

**ChatML Format**(ओपनएआई, जो कई ओपन सोर्स मॉडल द्वारा प्रयोग किया जाता है):

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

विशेष टोकन का उपयोग करता है (`<|im_start|>`,`<|im_end|>`) भूमिकाओं को परिभाषित करने के लिए। इन टोकन को टोकन के शब्दावली में जोड़ दिया जाता है। क्यूवेन, यी और कई अन्य मॉडल चैटएमएल का उपयोग करते हैं।

तीनों प्रारूप एक ही काम करते हैंः वे मॉडल को कहते हैं "यह निर्देश है, यह प्रतिक्रिया है, यह पैटर्न सीखें।"

### यह क्यों काम करता है

मॉडल पहले से ही पूर्व प्रशिक्षण से भाषा को जानता है। उसने प्रश्नों के अरबों उदाहरण देखे हैं, जिसके बाद उत्तर, निर्देशों के बाद पूर्णता और लोगों के बीच बातचीत के बाद। पैटर्न पहले से ही वजन में एन्कोड किए गए हैं।

एसएफटी इस लट्त क्षमता को केंद्रित करता है। मॉडल को संदर्भ से यह पता लगाने की आवश्यकता होने के बजाय कि उसे किसी प्रश्न का उत्तर देना चाहिए या दस्तावेज़ को जारी रखना चाहिए, एसएफटी स्पष्ट रूप से बातचीत के पैटर्न पर प्रशिक्षण देता है। कुछ हज़ार उदाहरणों के बाद, मॉडल सीखता हैः जब आप सहायक भूमिका मार्कर देखते हैं, तो एक उपयोगी प्रतिक्रिया उत्पन्न करते हैं।

यही कारण है कि 27,000 उदाहरण पर्याप्त हैं. आप मॉडल अंग्रेजी नहीं सिखा रहे हैं. आप उसे दुनिया के बारे में तथ्य नहीं सिखा रहे हैं. आप उसे एक सरल व्यवहार सिखा रहे हैंः निर्देशों का जवाब दें. ज्ञान पहले से ही वहाँ था.

### छिपी हुई हानि

यह एसएफटी में सबसे महत्वपूर्ण तकनीकी विवरण है, और अधिकांश ट्यूटोरियल इसे छोड़ देते हैं।

प्री-ट्रेनिंग के दौरान, आप प्रत्येक टोकन पर हानि की गणना करते हैं। मॉडल अनुक्रम में प्रत्येक अगले टोकन की भविष्यवाणी करना सीखता है। एसएफटी के दौरान, आप केवल * प्रतिक्रिया* टोकन पर हानि की गणना करते हैं। निर्देश टोकन संदर्भ के लिए वहां हैं, लेकिन मॉडल को गलत तरीके से "पूर्वानुमान" करने के लिए दंडित नहीं किया जाता है।

क्यों? क्योंकि आप नहीं चाहते कि मॉडल * निर्देश उत्पन्न करना सीखें। आप चाहते हैं कि यह * निर्देशों का जवाब देना सीखें। यदि आप निर्देश टोकन पर हानि का गणना करते हैं, तो आप मॉडल को भविष्यवाणी करने के लिए प्रशिक्षित कर रहे हैं "फ्रांस की राजधानी क्या है? " जैसे कि यह वह है जो सवाल पूछता है। यह ग्रेडिएंट सिग्नल को बर्बाद करता है और मॉडल को इसकी भूमिका के बारे में भ्रमित कर सकता है।

अभ्यास में, आप एक हानि मुखौटा बनाते हैंः 1 प्रतिक्रिया टोकन के लिए, 0 निर्देश टोकन के लिए. औसत से पहले इस मुखौटा से प्रति टोकन हानि गुणा करें.

```
Tokens:    [SYS] You are helpful [USER] What is the capital? [ASST] Paris is the capital [EOS]
Loss mask:   0    0    0     0      0     0   0  0     0       1     1    1   1     1      1
```

केवल टोकन के बाद `[ASST]`मॉडल आगे के पास के दौरान पूरी बातचीत को देखता है (उपलब्ध प्रतिक्रिया के लिए निर्देश की आवश्यकता होती है) लेकिन केवल प्रतिक्रिया की कितनी अच्छी तरह से भविष्यवाणी की गई है, इसके आधार पर अपने वजन को अपडेट करता है।

### प्रशिक्षण हाइपरपरपैरामीटर

एसएफटी में पूर्व-प्रशिक्षण से काफी अलग हाइपरपैरामीटर होते हैं. आप स्क्रैच से प्रशिक्षण नहीं दे रहे हैं. आप एक मॉडल को समायोजित कर रहे हैं जो पहले से ही काम करता है.

| Parameter | Pre-Training (Llama 2 7B) | SFT (Llama 2 Chat) |
|-----------|---------------------------|---------------------|
| Learning rate | 3e-4 (peak) | 2e-5 |
| Epochs | 1 (single pass over data) | 2 |
| Batch size | 4M tokens | 64 examples |
| Warmup steps | 2,000 | 0-100 |
| Weight decay | 0.1 | 0.0-0.1 |
| Data size | 2T tokens | 27,000 examples |

एसएफटी के लिए सीखने की दर 15 गुना कम है। यह महत्वपूर्ण है। बारीक-बारी से ट्यूनिंग के दौरान उच्च सीखने की दर पूर्व-शिक्षित ज्ञान को नष्ट कर देती है। मॉडल "भूल" जाता है कि उसने क्या सीखा है और छोटे बारीक-बारी से ट्यूनिंग डेटासेट पर ओवरटाइप करता है। यह विनाशकारी भूल है।

दो युग का मतलब है कि मॉडल प्रत्येक प्रशिक्षण उदाहरण को दो बार देखता है. एक छोटे से डेटासेट पर 3 से अधिक युग याद करने के लिए नेतृत्व करते हैं - मॉडल सामान्यीकरण के बजाय प्रशिक्षण उदाहरणों को शाब्दिक पुनः पेश करना शुरू करता है.

### भूलना विनाशकारी

ठीक से ट्यूनिंग सामान्य क्षमताओं को नष्ट कर सकता है। निर्देशों के बाद डेटा पर बहुत लंबे समय तक प्रशिक्षित करें और मॉडल कोड लिखने, गणित करने या रचनात्मक पाठ बनाने की अपनी क्षमता खो देता है। यह अपने प्रशिक्षण डेटा के विशिष्ट प्रारूप में बहुत अच्छा हो जाता है और बाकी सब कुछ में भयानक हो जाता है।

तीन उपाय:

1. **Low learning rate.**1ई-5 से 5ई-5 तक छोटे अपडेट से पूर्व-प्रशिक्षित सुविधाओं का कम विनाश होता है।

2. **Short training.**मॉडल ओवरफिट होने से पहले रुकें।

3. **Mix in pre-training data.**Llama 2 Chat ने एसएफटी डेटासेट में कच्चे पूर्व-प्रशिक्षण डेटा का एक छोटा प्रतिशत (2-5%) मिलाया। यह निर्देशों के बाद नए व्यवहार को सीखने के दौरान इसकी सामान्य क्षमताओं के मॉडल को "याद दिलाता है"।

### वास्तविक संख्याएँ

10,000 उच्च गुणवत्ता वाले निर्देश जोड़े पर एक 7B मॉडल को ठीक से समायोजित करने में लगभग एक घंटे का समय लगता है एक ही NVIDIA A100 80GB GPU पर। यहाँ गणित हैः

- 10,000 उदाहरणों x 512 टोकन औसत = 5.12M टोकन
- 2 युग = 10.24M टोकन कुल
- A100 7B मॉडल के लिए बारीक-बारी से ट्यूनिंग के लिए आउटपुटः ~3,000 टोकन/सेकंड
- 10.24M / 3,000 = ~3,400 सेकंड = ~57 मिनट

हमारे मिनी जीपीटी (4 परतें, 128 डिम्स) के लिए प्रशिक्षण लगभग तत्काल है। मुद्दा मैकेनिक्स को समझना है, पैमाने को नहीं।

```mermaid
graph TD
    subgraph SFT["Supervised Fine-Tuning Pipeline"]
        direction TB
        D["Instruction Dataset\n(10K-100K examples)"] --> F["Format into\n(instruction, response) pairs"]
        F --> T["Tokenize with\nchat template"]
        T --> M["Create loss mask\n(1 for response, 0 for instruction)"]
        M --> FW["Forward pass\n(full sequence)"]
        FW --> L["Compute masked loss\n(response tokens only)"]
        L --> BW["Backward pass"]
        BW --> U["Update weights\n(lr=2e-5, 1-3 epochs)"]
    end

    subgraph Base["Base Model\n(pre-trained)"]
        B1["Knows language"]
        B2["Knows facts"]
        B3["No conversation pattern"]
    end

    subgraph Chat["Chat Model\n(after SFT)"]
        C1["Knows language"]
        C2["Knows facts"]
        C3["Follows instructions"]
    end

    Base --> SFT --> Chat

    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style L fill:#1a1a2e,stroke:#e94560,color:#fff
    style B3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style C3 fill:#1a1a2e,stroke:#51cf66,color:#fff
```

```figure
loss-masking
```

## इसे बनाओ

### चरण 1: निर्देश डेटासेट

एक सिंथेटिक निर्देश डेटा सेट बनाएं. उत्पादन में, स्केल एआई और एंथ्रोपिक जैसी कंपनियां इनको लिखने के लिए मानव एनोटेटरों को नियोजित करती हैं. हम उन्हें प्रारूप प्रदर्शित करने के लिए प्रोग्रामेटिक रूप से बनाएंगे।

```python
import numpy as np

INSTRUCTION_DATA = [
    {
        "instruction": "What is the capital of France?",
        "response": "The capital of France is Paris."
    },
    {
        "instruction": "Explain gravity in one sentence.",
        "response": "Gravity is the force that attracts objects with mass toward each other."
    },
    {
        "instruction": "Write a haiku about the ocean.",
        "response": "Waves crash on the shore, salt and foam beneath the sun, endless blue expanse."
    },
    {
        "instruction": "What is 15 multiplied by 7?",
        "response": "15 multiplied by 7 is 105."
    },
    {
        "instruction": "Name three programming languages.",
        "response": "Three programming languages are Python, Rust, and TypeScript."
    },
    {
        "instruction": "Summarize photosynthesis.",
        "response": "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen."
    },
    {
        "instruction": "What year did World War II end?",
        "response": "World War II ended in 1945."
    },
    {
        "instruction": "Define machine learning.",
        "response": "Machine learning is a field where algorithms learn patterns from data to make predictions."
    },
]
```

आठ उदाहरण छोटे हैं स्टैनफोर्ड अल्पाका ने 52,000 का इस्तेमाल किया है, लेकिन यांत्रिकी एक जैसी है चाहे आपके पास 8 हो या 52,000: टोकन, मास्क, केवल उत्तरों पर गणना हानि।

### चरण 2: चैट टेम्पलेट के साथ टोकन बनाएं

निर्देश-उत्तर जोड़े को विशेष रोल मार्करों के साथ टोकन अनुक्रमों में परिवर्तित करें। मार्कर मॉडल को बताते हैं कि निर्देश कहां समाप्त होता है और प्रतिक्रिया कहां शुरू होती है।

```python
SPECIAL_TOKENS = {
    "INST_START": 253,
    "INST_END": 254,
    "RESP_START": 255,
}


def tokenize_instruction_pair(instruction, response, vocab_size=256):
    inst_tokens = list(instruction.encode("utf-8"))
    resp_tokens = list(response.encode("utf-8"))

    inst_tokens = [min(t, vocab_size - 4) for t in inst_tokens]
    resp_tokens = [min(t, vocab_size - 4) for t in resp_tokens]

    tokens = (
        [SPECIAL_TOKENS["INST_START"]]
        + inst_tokens
        + [SPECIAL_TOKENS["INST_END"]]
        + [SPECIAL_TOKENS["RESP_START"]]
        + resp_tokens
    )

    return tokens


def create_loss_mask(tokens):
    mask = np.zeros(len(tokens), dtype=np.float32)
    in_response = False

    for i, token in enumerate(tokens):
        if token == SPECIAL_TOKENS["RESP_START"]:
            in_response = True
            continue
        if in_response:
            mask[i] = 1.0

    return mask
```

हानि मुखौटा सभी निर्देश टोकन के लिए शून्य और सभी प्रतिक्रिया टोकन के लिए है।`RESP_START`टोकन स्वयं 0 का मुखौटा प्राप्त करता है क्योंकि यह एक सीमा है, प्रतिक्रिया सामग्री का हिस्सा नहीं है.

### चरण 3: छिपा हुआ क्रॉस-एंट्रोपी हानि

मानक क्रॉस-एंट्रोपी, लेकिन हानि मुखौटा से गुणा. केवल प्रतिक्रिया टोकन gradient में योगदान.

```python
def masked_cross_entropy_loss(logits, targets, loss_mask):
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    mask_flat = loss_mask.reshape(-1)

    max_logits = logits_flat.max(axis=-1, keepdims=True)
    log_softmax = logits_flat - max_logits - np.log(
        np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
    )

    per_token_loss = -log_softmax[np.arange(len(targets_flat)), targets_flat]

    masked_loss = per_token_loss * mask_flat
    num_response_tokens = mask_flat.sum()
    if num_response_tokens == 0:
        return 0.0
    loss = masked_loss.sum() / num_response_tokens

    return loss
```

नामक है `num_response_tokens`नहीं`seq_len`यदि आप अनुक्रम की कुल लंबाई से विभाजित करते हैं, तो लंबे निर्देश ग्रेडिएंट सिग्नल को पतला करते हैं। प्रतिक्रिया टोकन गिनती द्वारा विभाजित करने से निर्देश लंबाई के बावजूद प्रति प्रतिक्रिया टोकन के लिए समान वजन सुनिश्चित होता है।

### चरण 4: एसएफटी प्रशिक्षण लूप

Lesson 04 से MiniGPT का पुनः उपयोग करें। प्रशिक्षण लूप प्री-ट्रेनिंग के समान दिखता है, लेकिन निर्देश स्वरूपण और छिपे हुए नुकसान के साथ।

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "04-pre-training-mini-gpt", "code"))
from main import MiniGPT, LayerNorm, FeedForward, MultiHeadAttention, TransformerBlock, Embedding


def sft_train(model, dataset, num_epochs=2, lr=2e-5, seq_len=64):
    formatted_data = []
    for example in dataset:
        tokens = tokenize_instruction_pair(example["instruction"], example["response"])
        mask = create_loss_mask(tokens)
        formatted_data.append((tokens, mask))

    print(f"SFT Training: {len(formatted_data)} examples, {num_epochs} epochs, lr={lr}")
    print(f"Total tokens: {sum(len(t) for t, _ in formatted_data):,}")
    print()

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        indices = np.random.permutation(len(formatted_data))

        for idx in indices:
            tokens, mask = formatted_data[idx]

            if len(tokens) < 3:
                continue
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
                mask = mask[:seq_len]

            input_ids = np.array(tokens[:-1]).reshape(1, -1)
            target_ids = np.array(tokens[1:]).reshape(1, -1)
            loss_mask = np.array(mask[1:]).reshape(1, -1)

            logits = model.forward(input_ids)
            loss = masked_cross_entropy_loss(logits, target_ids, loss_mask)

            batch_size, s_len, v_size = logits.shape
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)
            dlogits = probs.copy()
            dlogits[np.arange(batch_size)[:, None], np.arange(s_len), target_ids] -= 1.0

            mask_expanded = loss_mask[:, :, np.newaxis]
            num_resp = loss_mask.sum()
            if num_resp > 0:
                dlogits = dlogits * mask_expanded / num_resp

            for block in model.blocks:
                block.ffn.W1 -= lr * np.random.randn(*block.ffn.W1.shape) * 0.01
                block.ffn.W2 -= lr * np.random.randn(*block.ffn.W2.shape) * 0.01
                block.ffn.b1 -= lr * np.random.randn(*block.ffn.b1.shape) * 0.01
                block.ffn.b2 -= lr * np.random.randn(*block.ffn.b2.shape) * 0.01

            epoch_loss += loss
            num_batches += 1
            losses.append(loss)

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

    return model, losses
```

सीखने की दर 2e-5 है, जो Llama 2 चैट से मेल खाती है. इसे प्री-ट्रेनिंग में इस्तेमाल किए जाने वाले 3e-4 की तुलना करें - 15 गुना छोटा। ग्रेडिएंट छिपा हुआ हैः निर्देश टोकन शून्य ग्रेडिएंट का उत्पादन करते हैं। केवल प्रतिक्रिया टोकन वजन को धक्का देते हैं।

### चरण 5: बेस बनाम एसएफटी मॉडल की तुलना करें

SFT का पूरा बिंदु व्यवहार परिवर्तन है. आइए यह मापें कि मॉडल कच्चे पाठ के निरंतरताओं के मुकाबले निर्देश-फॉर्मेट इनपुट पर कैसे प्रतिक्रिया करता है।

```python
def generate_response(model, prompt_tokens, max_new_tokens=50, temperature=0.8):
    tokens = list(prompt_tokens)
    seq_len = model.embedding.pos_embed.shape[0]

    for _ in range(max_new_tokens):
        context = np.array(tokens[-seq_len:]).reshape(1, -1)
        logits = model.forward(context)
        next_logits = logits[0, -1, :]

        next_logits = next_logits / max(temperature, 1e-8)
        probs = np.exp(next_logits - next_logits.max())
        probs = probs / probs.sum()
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / probs.sum()

        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(int(next_token))

    return tokens


def evaluate_instruction_following(model, instructions):
    print("Evaluating instruction following:")
    print("-" * 50)

    for instruction in instructions:
        tokens = (
            [SPECIAL_TOKENS["INST_START"]]
            + [min(t, 252) for t in list(instruction.encode("utf-8"))]
            + [SPECIAL_TOKENS["INST_END"]]
            + [SPECIAL_TOKENS["RESP_START"]]
        )

        output = generate_response(model, tokens, max_new_tokens=30, temperature=0.6)
        response_start = len(tokens)
        response_tokens = output[response_start:]
        response_bytes = bytes([t for t in response_tokens if t < 128])
        response_text = response_bytes.decode("utf-8", errors="replace")

        print(f"  Q: {instruction}")
        print(f"  A: {response_text[:80]}")
        print()
```

8 उदाहरणों के साथ एक छोटे से मॉडल पर, प्रतिक्रियाएं सार्थक नहीं होंगी। यह उम्मीद की जाती है। महत्वपूर्ण बात * संरचना* हैः मॉडल आगे बढ़ने के बजाय प्रतिक्रिया मार्कर के बाद आउटपुट उत्पन्न करना सीखता है।

### चरण 6: भूलने के भयानक उपाय

SFT से पहले और बाद में मॉडल की अगली टोकन भविष्यवाणी क्षमता की तुलना करें। यदि SFT सामान्य क्षमताओं को नुकसान पहुंचाता है, तो कच्चे पाठ पर नुकसान बढ़ जाएगा।

```python
def measure_forgetting(model, test_text, seq_len=64):
    tokens = np.array(list(test_text.encode("utf-8")[:512]))

    total_loss = 0.0
    num_windows = 0

    for start in range(0, len(tokens) - seq_len - 1, seq_len):
        input_ids = tokens[start:start + seq_len].reshape(1, -1)
        target_ids = tokens[start + 1:start + seq_len + 1].reshape(1, -1)

        logits = model.forward(input_ids)

        batch, s_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_ids.reshape(-1)

        max_logits = logits_flat.max(axis=-1, keepdims=True)
        log_softmax = logits_flat - max_logits - np.log(
            np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
        )

        loss = -log_softmax[np.arange(len(targets_flat)), targets_flat].mean()
        total_loss += loss
        num_windows += 1

    return total_loss / max(num_windows, 1)
```

वास्तविक ठीक-ठीक करने में, आप प्रशिक्षण के दौरान इस मीट्रिक को ट्रैक करेंगे। यदि कच्चे पाठ में नुकसान 10-15% से अधिक बढ़ जाता है, तो आपका एसएफटी बहुत आक्रामक है। सीखने की दर को कम करें या युगों की संख्या को कम करें।

## इसका प्रयोग करें

### पूर्ण एसएफटी पाइपलाइन डेमो

```python
if __name__ == "__main__":
    np.random.seed(42)

    test_text = """The transformer architecture processes sequences through self-attention.
Each layer applies multi-head attention followed by a feedforward network.
Residual connections and layer normalization stabilize deep networks.
The model learns to predict the next token given all previous tokens."""

    print("=" * 70)
    print("INSTRUCTION TUNING (SFT) DEMO")
    print("=" * 70)
    print()

    model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    print(f"Model: {model.count_parameters():,} parameters")
    print(f"Config: 4 layers, 4 heads, 128 dims (mini GPT from Lesson 04)")
    print()

    print("PRE-SFT: Measuring base model loss on raw text")
    base_loss = measure_forgetting(model, test_text)
    print(f"  Base model loss: {base_loss:.4f}")
    print()

    print("=" * 70)
    print("SFT TRAINING")
    print("=" * 70)

    model, losses = sft_train(
        model, INSTRUCTION_DATA, num_epochs=3, lr=2e-5, seq_len=128
    )

    print()
    print("POST-SFT: Measuring fine-tuned model loss on raw text")
    sft_loss = measure_forgetting(model, test_text)
    print(f"  SFT model loss: {sft_loss:.4f}")
    print(f"  Change: {((sft_loss - base_loss) / base_loss * 100):+.1f}%")
    if abs(sft_loss - base_loss) / base_loss < 0.15:
        print("  Minimal forgetting (< 15% change)")
    else:
        print("  Significant forgetting detected")
    print()

    print("=" * 70)
    print("INSTRUCTION FOLLOWING EVALUATION")
    print("=" * 70)
    print()

    test_instructions = [
        "What is the capital of France?",
        "Name a programming language.",
        "Define gravity.",
    ]
    evaluate_instruction_following(model, test_instructions)

    print("=" * 70)
    print("DATA FORMAT EXAMPLES")
    print("=" * 70)
    print()

    for i, example in enumerate(INSTRUCTION_DATA[:3]):
        tokens = tokenize_instruction_pair(example["instruction"], example["response"])
        mask = create_loss_mask(tokens)
        resp_count = int(mask.sum())
        total_count = len(tokens)
        print(f"  Example {i + 1}: {total_count} tokens, {resp_count} response tokens ({resp_count/total_count:.0%} of sequence)")
        print(f"    Instruction: {example['instruction']}")
        print(f"    Response: {example['response']}")
        print()

    print("=" * 70)
    print("TRAINING LOSS CURVE")
    print("=" * 70)
    print()

    if losses:
        window = max(1, len(losses) // 5)
        for i in range(0, len(losses), window):
            chunk = losses[i:i + window]
            avg = sum(chunk) / len(chunk)
            print(f"  Steps {i:3d}-{i + len(chunk) - 1:3d}: avg loss = {avg:.4f}")
```

## इसे भेजें

यह सबक हमें फल देता है`outputs/prompt-sft-data-curator.md`-- एक प्रॉम्प्ट जो आपको एसएफटी के लिए निर्देश डेटा सेट डिजाइन और क्यूरेट करने में मदद करता है। एक लक्ष्य क्षमता (कोड जनरेशन, गणित, बातचीत) को देखते हुए, यह प्रारूप विनिर्देशों, गुणवत्ता मानदंडों और विविधता आवश्यकताओं के साथ डेटा संग्रह योजना का उत्पादन करता है।

## व्यायाम

1. सिस्टम शीघ्र समर्थन जोड़ें. संशोधित करें `tokenize_instruction_pair`एक प्रणाली संदेश को स्वीकार करने और निर्देश से पहले इसे तैयार करने के लिए। विभिन्न प्रणाली संकेतों के साथ 5 उदाहरण बनाएं ("आप एक कवि हैं", "आप एक गणित शिक्षक हैं") और सत्यापित करें कि मॉडल प्रशिक्षण के दौरान विभिन्न प्रणाली संकेतों को देखता है।

2. डेटा मिक्सिंग लागू करें। एक फ़ंक्शन बनाएं जो एक एसएफटी डेटासेट और एक कच्चे पाठ कॉर्पस लेता है, फिर प्रशिक्षण बैच उत्पन्न करता है जहां 5% उदाहरण कच्चे पाठ (कोई मास्क नहीं) और 95% निर्देश जोड़े (मास्क किए गए) हैं। 3 युग चलाएं और शुद्ध एसएफटी प्रशिक्षण के साथ भूलने वाले मीट्रिक की तुलना करें।

3. एक डेटा गुणवत्ता स्कोरर बनाएं। प्रत्येक निर्देश-उत्तर जोड़ी के लिए, गणना करेंः (ए) टोकन में प्रतिक्रिया लंबाई, (बी) निर्देश-प्रतिसाद अनुपात, (सी) शब्दावली विविधता (अद्वितीय टोकन / कुल टोकन) । प्रतिक्रिया लंबाई < 10 टोकन या विविधता < 0.3 के साथ उदाहरणों को फ़िल्टर करें। दिखाएं कि फ़िल्टरिंग अंतिम हानि को कैसे प्रभावित करता है।

4. मल्टी-टर्न वार्तालाप प्रशिक्षण लागू करें। 3-टर्न वार्तालापों को संभालने के लिए टोकनकरण का विस्तार करें (उपयोगकर्ता-सहायक-उपयोगकर्ता-सहायक-उपयोगकर्ता-सहायक) । हानि मुखौटा को तीनों सहायक मोड़ों को कवर करना चाहिए। एक उदाहरण के लिए टोकन-मुखौटा संरेखण प्रिंट करके मुखौटा सही है।

5. सीखने की दरों की तुलना करें। एक ही मॉडल को तीन बार lr=1e-4, lr=2e-5, और lr=1e-6 के साथ प्रशिक्षित करें। हानि वक्रों को रेखांकित करें। 1e-4 रन में तेजी से प्रारंभिक गिरावट दिखानी चाहिए लेकिन उच्च अंतिम नुकसान (ओवरफिटिंग) दिखाना चाहिए। 1e-6 रन को मुश्किल से चलना चाहिए। 2e-5 रन को सबसे अच्छा स्थान होना चाहिए।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| SFT | "Fine-tuning on conversations" | Supervised Fine-Tuning: continuing training on (instruction, response) pairs with loss computed only on response tokens |
| Instruction tuning | "Teaching the model to follow instructions" | Training on explicit instruction-response pairs so the base model learns the conversation pattern, not new knowledge |
| Loss masking | "Ignoring the prompt" | Setting loss to zero for instruction tokens so gradients only flow from response token predictions |
| ChatML | "Chat Markup Language" | A token format using `<\|im_start\|>` and `<\|im_end\|>` delimiters to mark speaker roles in conversation data |
| Alpaca format | "Stanford's format" | A JSON format with instruction/input/output fields, used for 52K GPT-3.5-generated examples that cost $600 |
| Catastrophic forgetting | "The model gets dumber" | Fine-tuning destroys pre-trained capabilities because gradient updates overwrite general knowledge with task-specific patterns |
| Weight tying | "Shared embeddings" | Using the same matrix for input token embeddings and output prediction head, saving parameters and improving coherence |
| Chat template | "How you format the prompt" | The specific token sequence (role markers, delimiters) that structures a conversation for the model |

## आगे पढ़ना

- [Ouyang et al., 2022 -- "Training language models to follow instructions with human feedback" (InstructGPT)](https://arxiv.org/abs/2203.02155)-- पेपर जो OpenAI में निर्देश ट्यूनिंग + RLHF की शुरुआत करता है
- [Taori et al., 2023 -- "Stanford Alpaca: An Instruction-following LLaMA Model"](https://github.com/tatsu-lab/stanford_alpaca)-- $600 के लिए 52K निर्देश उदाहरण, SFT छोटे डेटा सेट पर काम करता है साबित करने के लिए
- [Touvron et al., 2023 -- "Llama 2: Open Foundation and Fine-Tuned Chat Models"](https://arxiv.org/abs/2307.09288)-- मेटा की SFT + RLHF पाइपलाइन 27K उच्च गुणवत्ता वाले उदाहरणों के साथ
- [Chiang et al., 2023 -- "Vicuna: An Open-Source Chatbot Impressing GPT-4"](https://lmsys.org/blog/2023-03-30-vicuna/)-- 70K ShareGPT वार्तालापों पर प्रशिक्षण
- [Zhou et al., 2023 -- "LIMA: Less Is More for Alignment"](https://arxiv.org/abs/2305.11206)-- यह साबित करना कि 1000 सावधानीपूर्वक चुनिंदा उदाहरण बहुत बड़े डेटा सेट पर एसएफटी से मेल खा सकते हैं
