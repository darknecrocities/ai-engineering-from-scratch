# लोरा और क्यूलोरा के साथ ठीक से ट्यूनिंग

> 7B मॉडल के लिए पूर्ण ठीक-ठाक करने के लिए 56GB VRAM की आवश्यकता होती है. आपके पास यह नहीं है. अधिकांश कंपनियों में भी नहीं है. लोरा आपको एक ही मॉडल को 6GB में ठीक-ठाक करने की अनुमति देता है, पैरामीटर के 1% से कम का प्रशिक्षण देकर। यह समझौता नहीं है - यह अधिकांश कार्यों पर पूर्ण ठीक-ठाक गुणवत्ता से मेल खाता है। पूरी ओपन-सोर्स ठीक-ठाक पारिस्थितिकी तंत्र इस एक चाल पर चलता है।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 06 (Instruction Tuning / SFT)
**Time:** ~75 minutes
**Related:**चरण 10 में एसएफटी/डीपीओ लूप को खरोंच से कवर किया गया है। यह पाठ उन्हें 2026 पीईएफटी टूलकिट (पीईएफटी, टीआरएल, अनलॉथ, एक्सोलोटल, एलएएमए-फैक्टरी) में प्लग करता है।

## सीखने के लक्ष्य

- एक पूर्व प्रशिक्षित मॉडल की ध्यान परतों में निम्न श्रेणी के एडाप्टर मैट्रिक्स (ए और बी) इंजेक्ट करके LoRA लागू करें
- LoRA बनाम पूर्ण सूक्ष्म-ट्यूनिंग की पैरामीटर बचत की गणना करेंः d_model आयामों के साथ रैंक r d2 के बजाय 2*r*d पैरामीटर को ट्रेन करता है
- उपभोक्ता जीपीयू मेमोरी में फिट होने के लिए QLoRA (4-बिट क्वांटिज़्ड बेस + LoRA एडाप्टर) का उपयोग करके एक मॉडल को ठीक से समायोजित करें
- लोर्र वजन को तैनाती के लिए मूल मॉडल में वापस मिलाएं और एडाप्टर के साथ और बिना अनुमानित गति की तुलना करें

## समस्या

आप एक आधार मॉडल है. Llama 3 8B. आप चाहते हैं कि यह आपकी कंपनी की आवाज में ग्राहक सहायता टिकट का जवाब देने के लिए है. SFT जवाब है. लेकिन SFT एक लागत समस्या है.

पूर्ण ठीक-ठाक मॉडल में प्रत्येक पैरामीटर को अपडेट करता है। Llama 3 8B में 8 बिलियन पैरामीटर हैं। fp16 में, प्रत्येक पैरामीटर में 2 बाइट्स लगते हैं। वजन को लोड करने के लिए यह 16GB है। प्रशिक्षण के दौरान, आपको एडम के लिए ग्रेडिएंट (16GB), ऑप्टिमाइज़र राज्यों (32GB गति + भिन्नता के लिए) और सक्रियण की भी आवश्यकता होती है। कुलः एक एकल 8B मॉडल के लिए लगभग 56GB VRAM।

एक A100 80GB शायद ही इस फिट बैठता है. दो A100s लागत .$3-4/hour on cloud providers. Training for 3 epochs on 50,000 examples takes 6-10 hours. That's $30-40 प्रति प्रयोग. हाइपरपरपैरामीटर सही पाने के लिए 10 प्रयोग चलाओ और आप कुछ भी तैनात करने से पहले $400 खर्च किया है.

Llama 3 70B के लिए यह मापें और संख्या absurd हो जाते हैं. 140GB वजन के लिए अकेले. आपको एक क्लस्टर की जरूरत है. $ 100 + प्रति प्रयोग.

एक और गहरी समस्या भी है. पूर्ण ठीक-ठाक मॉडल के हर वजन को संशोधित करता है. यदि आप ग्राहक सहायता डेटा पर ठीक-ठाक करते हैं, तो आप मॉडल की सामान्य क्षमताओं को कम कर सकते हैं। इसे आपदा भूलना कहा जाता है. मॉडल आपके कार्य में बेहतर हो जाता है और बाकी सब कुछ में बदतर हो जाता है।

आपको एक ऐसी विधि की आवश्यकता है जो कम मापदंडों को प्रशिक्षित करे, कम मेमोरी का उपयोग करे, और मॉडल के मौजूदा ज्ञान को नष्ट न करे।

## अवधारणा

### लोरा: निम्न श्रेणी के अनुकूलन

एडवर्ड हु और माइक्रोसॉफ्ट के सहयोगियों ने जून 2021 में लोरा प्रकाशित किया। पेपर की अंतर्दृष्टिः ठीक-ट्यूनिंग के दौरान वजन अपडेट में कम आंतरिक रैंक होती है। आपको 4096x4096 वजन मैट्रिक्स में सभी 16.7 मिलियन मापदंडों को अपडेट करने की आवश्यकता नहीं है। अद्यतन में उपयोगी जानकारी 16 या 32 रैंक के मैट्रिक्स द्वारा कैप्चर की जा सकती है।

यहाँ गणित है. एक मानक रैखिक परत गणना करता हैः

```
y = Wx
```

जहां W एक d_out x d_in मैट्रिक्स है. 4096x4096 ध्यान प्रक्षेपण के लिए, यह 16,777,216 पैरामीटर है.

लोरा W को जमे हुए और एक निम्न श्रेणी का विघटन जोड़ता हैः

```
y = Wx + BAx
```

जहां B (d_out x r) और A (r x d_in) है। रैंक r d से बहुत छोटा है - आमतौर पर 8, 16 या 32.

4096x4096 परत पर r=16 के लिएः
- मूल मापदंडः 4096 x 4096 = 16,777,216
- लोरा पैरामीटरः (4096 x 16) + (16 x 4096) = 65,536 + 65,536 = 131,072
- कटौती: 131,072 / 16,777,216 = 0.78%

आप 0.78% मापदंडों का प्रशिक्षण कर रहे हैं और 95-100% गुणवत्ता प्राप्त कर रहे हैं।

```mermaid
graph LR
    X["Input x"] --> W["Frozen W (d x d)"]
    X --> A["A (r x d)"]
    A --> B["B (d x r)"]
    W --> Plus["+ (merge)"]
    B --> Plus
    Plus --> Y["Output y"]

    style W fill:#1a1a2e,stroke:#e94560,color:#fff
    style A fill:#0f3460,stroke:#16213e,color:#fff
    style B fill:#0f3460,stroke:#16213e,color:#fff
```

A को एक यादृच्छिक गौशियन के साथ आरंभ किया जाता है. B को शून्य से आरंभ किया जाता है. इसका मतलब है कि लोरा योगदान शून्य से शुरू होता है - मॉडल अपने मूल व्यवहार से प्रशिक्षण शुरू करता है और धीरे-धीरे अनुकूलन सीखता है।

### स्केलिंग फैक्टरः अल्फा

लोरा एक स्केलिंग कारक अल्फा पेश करता है जो नियंत्रित करता है कि निम्न श्रेणी के अपडेट आउटपुट को कितना प्रभावित करता हैः

```
y = Wx + (alpha / r) * BAx
```

जब अल्फा = आर, स्केलिंग 1x है। जब अल्फा = 2r (सामान्य डिफ़ॉल्ट), स्केलिंग 2x है। यह हाइपरपैरामीटर मूल सीखने की दर से स्वतंत्र रूप से LoRA पथ की सीखने की दर को नियंत्रित करता है।

व्यावहारिक मार्गदर्शनः
- अल्फा = 2 * रैंक एक आम सामुदायिक सम्मेलन है (मूल पेपर में प्रयोग किया गया अल्फा = रैंक अधिकांश प्रयोगों में)
- अल्फा = रैंक 1x स्केलिंग देता है, संरक्षित लेकिन स्थिर
- उच्च अल्फा का अर्थ है प्रति चरण बड़े अपडेट, जो अभिसरण को गति दे सकता है या अस्थिरता पैदा कर सकता है

### लोरा को कहाँ लागू किया जाए

एक ट्रांसफार्मर में कई रैखिक परतें होती हैं. आपको उन सभी में लोरा जोड़ने की आवश्यकता नहीं है। मूल पेपर ने विभिन्न संयोजनों का परीक्षण कियाः

| Target Layers | Trainable Params (7B) | Quality |
|--------------|----------------------|---------|
| q_proj only | 4.7M | Good |
| q_proj + v_proj | 9.4M | Better |
| q_proj + k_proj + v_proj + o_proj | 18.9M | Best for attention |
| All linear (attention + MLP) | 37.7M | Marginal gain, 2x params |

अधिकांश कार्यों के लिए मीठा बिंदुः q_proj + v_proj। यह स्वयं ध्यान में क्वेरी और मूल्य अनुमानों को लक्षित करता है, जो नियंत्रित करते हैं कि मॉडल क्या देखता है और यह क्या जानकारी निकालता है। एमएलपी परतों को जोड़ना कोड जेनरेशन जैसे जटिल कार्यों में मदद करता है लेकिन सरल कार्यों पर घटते रिटर्न के लिए पैरामीटर की गिनती को दोगुना करता है।

### रैंक चयन

रैंक r अनुकूलन की अभिव्यक्तिशीलता को नियंत्रित करता हैः

| Rank | Trainable Params (per layer) | Best For |
|------|---------------------------|----------|
| 4 | 32,768 | Simple classification, sentiment |
| 8 | 65,536 | Single-domain Q&A, summarization |
| 16 | 131,072 | Multi-domain tasks, instruction following |
| 32 | 262,144 | Complex reasoning, code generation |
| 64 | 524,288 | Diminishing returns for most tasks |
| 128 | 1,048,576 | Rarely justified |

ह्यू और अन्य ने दिखाया कि r=4 पहले से ही सरल कार्यों के लिए अनुकूलन के अधिकांश को कैप्चर करता है। r=8 और r=16 व्यवहार में सबसे आम विकल्प हैं। r=64 से परे जाना शायद ही कभी गुणवत्ता में सुधार करता है और लोरा के मेमोरी लाभ को खोना शुरू कर देता है।

### QLoRA: 4-बिट क्वांटिज़ेशन + लोरा

वाशिंगटन विश्वविद्यालय के टिम डेटमर और उनके सहयोगियों ने मई 2023 में QLoRA प्रकाशित किया। विचारः स्थिर आधार मॉडल को 4-बिट सटीकता तक क्वांटिफाई करें, फिर ऊपर fp16 में LoRA एडाप्टर संलग्न करें।

यह स्मृति समीकरण को नाटकीय रूप से बदलता हैः

| Method | Weight Memory (7B) | Training Memory (7B) | GPU Required |
|--------|-------------------|---------------------|-------------|
| Full fine-tune (fp16) | 14GB | ~56GB | 1x A100 80GB |
| LoRA (fp16 base) | 14GB | ~18GB | 1x A100 40GB |
| QLoRA (4-bit base) | 3.5GB | ~6GB | 1x RTX 3090 24GB |

QLoRA तीन तकनीकी योगदान देता हैः

**NF4 (Normal Float 4-bit)**: न्यूरल नेटवर्क के वजन के लिए विशेष रूप से डिज़ाइन किया गया एक नया डेटा प्रकार। न्यूरल नेटवर्क के वजन लगभग सामान्य वितरण का पालन करते हैं। एनएफ 4 अपने 16 क्वांटिज़ेशन स्तरों को मानक सामान्य वितरण के क्वांटिल पर रखता है। यह सामान्य रूप से वितरित डेटा के लिए सूचना-सैद्धांतिक रूप से अनुकूल है। यह समान 4-बिट क्वांटिज़ेशन (INT4) या मानक Float4 से कम जानकारी खोता है।

**Double quantization**: क्वांटिज़ेशन स्थिरांक स्वयं मेमोरी लेते हैं. 64 वजन के प्रत्येक ब्लॉक को fp32 स्केल कारक (4 बाइट्स) की आवश्यकता होती है। 7B मॉडल के लिए, यह अतिरिक्त 0.4GB है। डबल क्वांटिज़ेशन इन स्थिरांक को fp8 में क्वांटिज करता है, ओवरहेड को 0.1GB तक कम करता है। छोटा है लेकिन यह जोड़ता है।

**Paged optimizers**प्रशिक्षण के दौरान, ऑप्टिमाइज़र स्टेटस (आदम की गति और भिन्नता) लंबे अनुक्रमों पर GPU मेमोरी से अधिक हो सकती है। पेज ऑप्टिमाइज़र स्वचालित रूप से NVIDIA की एकीकृत मेमोरी का उपयोग करके GPU मेमोरी समाप्त होने पर CPU RAM में ऑप्टिमाइज़र स्टेटस को पेज करते हैं, और जब आवश्यक हो तो उन्हें वापस पेज करते हैं। यह कुछ आउटपुट की कीमत पर OOM क्रैश को रोकता है।

### गुणवत्ता का प्रश्न

क्या पैरामीटर को कम करना या आधार को मात्राबद्ध करना गुणवत्ता को नुकसान पहुंचाता है?

| Method | MMLU (5-shot) | MT-Bench | HumanEval |
|--------|--------------|----------|-----------|
| Full fine-tune (Llama 2 7B) | 48.3 | 6.72 | 14.6 |
| LoRA r=16 | 47.9 | 6.68 | 14.0 |
| QLoRA r=16 (NF4) | 47.5 | 6.61 | 13.4 |
| QLoRA r=64 (NF4) | 48.1 | 6.70 | 14.2 |

आर = 16 पर लोरा अधिकांश बेंचमार्क पर पूर्ण ठीक-ठाक के 1% के भीतर है। आर = 16 पर क्यूलोरा एक प्रतिशत का एक और अंश खो देता है। आर = 64 पर क्यूलोरा अनिवार्य रूप से 90% कम मेमोरी का उपयोग करते हुए पूर्ण ठीक-ठाक से मेल खाता है।

### वास्तविक दुनिया की लागत

50,000 नमूनों (3 कालखंडों) पर ल्लामा 3 8B को ठीक से समायोजित किया गयाः

| Method | GPU | Time | Cost |
|--------|-----|------|------|
| Full fine-tune | 2x A100 80GB | 8 hours | ~$32 |
| LoRA r=16 | 1x A100 40GB | 4 hours | ~$8 |
| QLoRA r=16 | 1x RTX 4090 24GB | 6 hours | ~$5 |
| QLoRA r=16 (Unsloth) | 1x RTX 4090 24GB | 2.5 hours | ~$2 |
| QLoRA r=16 | 1x T4 16GB | 12 hours | ~$4 |

एक एकल उपभोक्ता जीपीयू पर क्यूलोरा की लागत एक दोपहर के भोजन से कम है। यही कारण है कि 2023 में ओपन-वेट फाइन-ट्यूनिंग समुदाय विस्फोट हुआ और 2026 में डिफ़ॉल्ट रूप से क्यूलोरा को नीचे के प्रत्येक प्रशिक्षण ढांचे में भेज दिया गया।

### 2026 पीईएफटी स्टैक

| Framework | What it is | Pick when |
|-----------|-----------|-----------|
| **Hugging Face PEFT** | The canonical LoRA/QLoRA/DoRA/IA3 library | You want raw control and your training loop is already on `transformers.Trainer` |
| **TRL** | HF's reinforcement-from-feedback trainers (SFT, DPO, GRPO, PPO, ORPO) | You need DPO/GRPO after SFT; built on top of PEFT |
| **Unsloth** | Triton-kernel rewrite of the forward/backward pass | You want 2-5x speedup + half the VRAM with no accuracy loss; Llama/Mistral/Qwen family |
| **Axolotl** | YAML-config wrapper over PEFT + TRL + DeepSpeed + Unsloth | You want reproducible, version-controlled training runs |
| **LLaMA-Factory** | GUI/CLI/API over PEFT + TRL | You want zero-code fine-tuning; 100+ model families supported |
| **torchtune** | Native PyTorch recipes, no `transformers` dep | You want minimal deps and your org already standardizes on PyTorch |

अंगूठे का नियमः अनुसंधान उपयोग या एक बार प्रयोग → पीईएफटी। पुनरावर्ती उत्पादन पाइपलाइन → अनलॉथ नाखून सक्षम के साथ Axolotl। थ्रोवे प्रोटोटाइपिंग → LLaMA-फैक्टरी।

### एडाप्टर को मिलाएं

प्रशिक्षण के बाद, आपके पास दो चीजें हैंः जमे हुए बेस मॉडल और एक छोटा लोरा एडाप्टर (आमतौर पर 10-100 एमबी) । आप या तोः

1. **Keep them separate**: बेस मॉडल लोड करें, शीर्ष पर एडाप्टर लोड करें। विभिन्न कार्यों के लिए स्विच एडाप्टर। इस तरह आप एक बेस मॉडल से कई बारीक-ट्यून वैरिएंट की सेवा करते हैं।

2. **Merge them permanently**: गणना W' = W + (alpha/r) * BA और एक नए पूर्ण मॉडल के रूप में परिणाम सहेजें. विलय मॉडल मूल के समान आकार है. कोई निष्कर्ष ओवरहेड नहीं है. कोई एडाप्टर प्रबंधित करने के लिए नहीं है.

कई कार्यों (ग्राहक सहायता एडाप्टर, कोड एडाप्टर, अनुवाद एडाप्टर) को पूरा करने के लिए, उन्हें अलग रखें। एक ही विशेष मॉडल को तैनात करने के लिए, एकजुट करें।

कई एडाप्टरों को जोड़ने के लिए उन्नत विलय तकनीकेंः

- **TIES-Merging**(यादाव और सहयोगियों 2023): छोटे परिमाण के मापदंडों को ट्रिम करता है, संकेत संघर्षों को हल करता है, फिर विलय करता है। एडाप्टरों के बीच हस्तक्षेप को कम करता है।
- **DARE**(यू एट अल. 2023): विलय से पहले एडाप्टर पैरामीटर को यादृच्छिक रूप से गिरा देता है और शेष को फिर से स्केल करता है। क्षमताओं को जोड़ने में आश्चर्यजनक रूप से प्रभावी।
- **Task arithmetic**एक "कोड" एडाप्टर और एक "गणित" एडाप्टर को जोड़ने से अक्सर दोनों में एक अच्छा मॉडल उत्पन्न होता है।

### जब ठीक-ठाक नहीं करना चाहिए

ठीक-ठीक ट्यूनिंग तीसरा विकल्प है, पहला नहीं।

**First: prompt engineering.**एक बेहतर सिस्टम प्रॉम्प्ट लिखें. कुछ शॉट उदाहरण जोड़ें. विचार श्रृंखला का उपयोग करें. यह कुछ भी नहीं लागत है और मिनट लेता है. यदि प्रॉम्प्टिंग आपको 80% तक पहुंचता है, तो आपको शायद ठीक से ट्यून करने की आवश्यकता नहीं है।

**Second: RAG.**यदि मॉडल को आपके विशिष्ट डेटा (दस्तावेज़, ज्ञान आधार, उत्पाद सूची) के बारे में जानने की आवश्यकता है, तो इसे वजन में बेकिंग से अधिक सस्ता और अधिक बनाए रखा जा सकता है। पाठ 06 देखें।

**Third: fine-tuning.**जब आपको मॉडल की आवश्यकता हो तो इसका उपयोग एक विशिष्ट शैली, प्रारूप या तर्क पैटर्न को अपनाने के लिए करें जो प्रॉम्प्टिंग के माध्यम से प्राप्त नहीं किया जा सकता है। जब आपको एक सुसंगत संरचित आउटपुट की आवश्यकता हो। जब आपको एक बड़े मॉडल को एक छोटे से अलग करने की आवश्यकता हो। जब विलंबता मायने रखती है और आप कुछ शॉट प्रॉम्प्टिंग से अतिरिक्त टोकन का भुगतान नहीं कर सकते हैं।

```mermaid
graph TD
    Start["Need better model behavior?"] --> PE["Try prompt engineering"]
    PE -->|"Works"| Done["Ship it"]
    PE -->|"Not enough"| RAG["Need external knowledge?"]
    RAG -->|"Yes"| RAGBuild["Build RAG pipeline"]
    RAG -->|"No, need style/format change"| FT["Fine-tune with LoRA/QLoRA"]
    RAGBuild -->|"Works"| Done
    RAGBuild -->|"Also need style change"| FT
    FT --> Done

    style Start fill:#1a1a2e,stroke:#e94560,color:#fff
    style Done fill:#0f3460,stroke:#16213e,color:#fff
```

```figure
lora-params
```

## इसे बनाओ

हम शुद्ध PyTorch में LoRA को खरोंच से लागू करते हैं कोई पुस्तकालय नहीं कोई जादू नहीं आप LoRA परत बना सकते हैं, इसे एक मॉडल में इंजेक्ट कर सकते हैं, इसे प्रशिक्षित कर सकते हैं, और वजन को वापस मिला सकते हैं।

### चरण 1: लोरा परत

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.randn(in_features, rank) * (1 / math.sqrt(rank)))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling
```

A को स्केल किए गए यादृच्छिक मानों के साथ आरंभ किया जाता है। B को शून्य पर आरंभ किया जाता है। उत्पाद BA शून्य से शुरू होता है, इसलिए मॉडल अपने मूल व्यवहार से शुरू होता है।

### चरण 2: लोरा-वाप्पे हुए रैखिक परत

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

मूल रैखिक परत को जमे रखा गया है। केवल लोरा पैरामीटर (ए और बी) प्रशिक्षित किए जा सकते हैं।

### चरण 3: एक मॉडल में लोरा इंजेक्ट करें

```python
def inject_lora(model, target_modules, rank=8, alpha=16):
    for param in model.parameters():
        param.requires_grad = False

    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]
                lora_linear = LinearWithLoRA(module, rank, alpha)
                setattr(parent, child_name, lora_linear)
                lora_layers[name] = lora_linear
    return lora_layers
```

सबसे पहले, मॉडल में प्रत्येक पैरामीटर को फ्रीज करें। फिर मॉडल ट्री पर जाएं, अपने लक्ष्य नामों से मेल खाने वाली रैखिक परतें खोजें, और उन्हें लोरा-पैक किए गए संस्करणों से बदलें। लोरा ए और बी मैट्रिक्स पूरे मॉडल में एकमात्र प्रशिक्षित पैरामीटर हैं।

### चरण 4: पैरामीटर गिनें

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }
```

### चरण 5: वजन को वापस मिलाएं

```python
def merge_lora_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            with torch.no_grad():
                merged = (
                    module.lora.A @ module.lora.B
                ) * module.lora.scaling
                module.linear.weight.data += merged.T
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, module.linear)
```

एकीकरण के बाद, लोरा परतें गायब हो गई हैं मॉडल मूल के समान आकार है वजन में बेक अनुकूलन के साथ। कोई निष्कर्ष ओवरहेड नहीं।

### चरण 6: अनुकरण QLoRA मात्रा

```python
def quantize_to_nf4(tensor, block_size=64):
    blocks = tensor.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
    scales = torch.clamp(scales, min=1e-8)
    quantized = torch.round(blocks / scales).clamp(-8, 7).to(torch.int8)
    return quantized, scales

def dequantize_from_nf4(quantized, scales, original_shape):
    dequantized = quantized.float() * scales
    return dequantized.reshape(original_shape)
```

यह 64 के ब्लॉक के भीतर 16 अलग-अलग स्तरों में वजन को मैप करके 4-बिट क्वांटिज़ेशन का अनुकरण करता है। उत्पादन QLoRA GPU पर वास्तविक NF4 के लिए बिट्स एंड बाइट्स लाइब्रेरी का उपयोग करता है।

### चरण 7: प्रशिक्षण लूप

```python
def train_lora(model, data, epochs=5, lr=1e-3, batch_size=4):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(len(data["inputs"]))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x = data["inputs"][batch_idx]
            y = data["targets"][batch_idx]

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

    return losses
```

### चरण 8: पूर्ण डेमो

```python
def demo():
    torch.manual_seed(42)
    d_model = 256
    n_classes = 10

    model = nn.Sequential(
        nn.Linear(d_model, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    )

    n_samples = 500
    x = torch.randn(n_samples, d_model)
    y = torch.randint(0, n_classes, (n_samples,))
    y_onehot = torch.zeros(n_samples, n_classes).scatter_(1, y.unsqueeze(1), 1.0)

    data = {"inputs": x, "targets": y_onehot}

    params_before = count_parameters(model)

    lora_layers = inject_lora(
        model, target_modules=["0", "2"], rank=8, alpha=16
    )

    params_after = count_parameters(model)

    losses = train_lora(model, data, epochs=20, lr=1e-3)

    merge_lora_weights(model)
    params_merged = count_parameters(model)

    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_merged": params_merged,
        "losses": losses,
    }
```

डेमो एक छोटा मॉडल बनाता है, लोरा को दो परतों में इंजेक्ट करता है, इसे प्रशिक्षित करता है, और वजन को वापस मिलाता है। पैरामीटर गिनती पूर्ण प्रशिक्षित से ~ 1% को घट जाती है जो लोरा प्रशिक्षण के दौरान प्रशिक्षित होती है, फिर विलय के बाद मूल वास्तुकला में लौटती है।

## इसका प्रयोग करें

गले लगाने वाले चेहरे पारिस्थितिकी तंत्र के साथ, एक वास्तविक मॉडल पर लोरा लगभग 20 पंक्तियों को लेता हैः

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

QLoRA के लिए, बिट्सएंडबाइट्स क्वांटिज़ेशन जोड़ेंः

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

model = get_peft_model(model, lora_config)
```

यही है. एक ही प्रशिक्षण लूप. एक ही डेटा पाइपलाइन. बेस मॉडल अब 4 बिट में रहता है, LoRA एडाप्टर FP16 में ट्रेन, और पूरी बात 6GB में फिट बैठता है.

गले लगाने वाले चेहरे के प्रशिक्षक के साथ प्रशिक्षण के लिएः

```python
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./lora-adapter")
```

सहेजे गए एडाप्टर 10-100MB है. मूल मॉडल को छुआ नहीं जाता है. आप पूरे मॉडल को पुनः वितरित किए बिना Hugging Face Hub पर एडाप्टर साझा कर सकते हैं।

## इसे भेजें

इस पाठ से उत्पन्न होता हैः
- `outputs/prompt-lora-advisor.md`-- एक संकेत जो आपको अपने विशिष्ट कार्य के लिए लोरा रैंक, लक्ष्य मॉड्यूल और हाइपरपैरामीटर का निर्णय लेने में मदद करता है
- `outputs/skill-fine-tuning-guide.md`-- एक कौशल है कि एजेंटों को निर्णय के पेड़ सिखाता है जब और कैसे ठीक करने के लिए

## व्यायाम

1. **Rank ablation study.**रैंक 2, 4, 8, 16, 32 और 64 के साथ डेमो चलाएं। अंतिम हानि बनाम रैंक प्लॉट करें। घटते रिटर्न का बिंदु ढूंढें जहां रैंक को दोगुना करने से नुकसान आधी नहीं होता है। 256-अवसर सुविधाओं पर एक सरल वर्गीकरण कार्य के लिए, यह लगभग आर = 8-16 होना चाहिए।

2. **Target module comparison.**केवल परत "0", केवल परत "2", केवल परत "4", और सभी तीनों को लक्षित करने के लिए इंजेक्ट_लोरा को संशोधित करें। प्रत्येक संस्करण को 20 युगों के लिए प्रशिक्षित करें। अभिसरण गति और अंतिम हानि की तुलना करें। यह सभी रैखिक परतों के खिलाफ q_proj बनाम v_proj को लक्षित करने के वास्तविक निर्णय को दर्शाता है।

3. **Quantization error analysis.**प्रशिक्षित मॉडल के वजन मैट्रिक्स को क्वांटिज़_टू_एनएफ4 / डिक्वांटिज़_फॉर_एनएफ4 से पहले और बाद में लें। औसत वर्ग त्रुटि, अधिकतम पूर्ण त्रुटि और मूल और पुनर्निर्माण किए गए वजन के बीच संबंध की गणना करें। 32, 64, 128 और 256 के ब्लॉक_साइज मानों के साथ प्रयोग करें।

4. **Multi-adapter serving.**दो लोरा एडाप्टर को डेटा के विभिन्न उपसमूहों (इंडेक्स बनाम विषम सूचकांक) पर प्रशिक्षित करें। दोनों एडाप्टरों को सहेजें। एक बार बेस मॉडल को लोड करें, फिर एडाप्टरों को स्वैप करें और सत्यापित करें कि प्रत्येक एक ही इनपुट पर अलग-अलग आउटपुट उत्पन्न करता है। यह है कि उत्पादन प्रणाली एक आधार से कई बारीक-बारी से ट्यून किए गए मॉडल की सेवा कैसे करती है।

5. **Merge vs. unmerged inference.**उसी 100 इनपुट पर merge_lora_weights से पहले और बाद में LoRA मॉडल के आउटपुट की तुलना करें। जांचें कि आउटपुट समान हैं (फ्लोटिंग-पॉइंट टॉलरेंस 1e-5 के भीतर) । फिर दोनों के लिए बेंचमार्क निष्कर्ष की गति - विलय थोड़ा तेज़ होना चाहिए क्योंकि यह दो के बजाय एक मैट्रिक्स गुणा है।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| LoRA | "Efficient fine-tuning" | Low-Rank Adaptation: freeze base weights, train two small matrices A and B whose product approximates the full weight update |
| QLoRA | "Fine-tune on a laptop" | Quantized LoRA: load the base model in 4-bit NF4, train LoRA adapters in fp16 on top, enabling 7B fine-tuning in 6GB VRAM |
| Rank (r) | "How much the model can learn" | The inner dimension of the A and B matrices; controls expressiveness vs. parameter count |
| Alpha | "LoRA learning rate" | Scaling factor applied to the LoRA output; alpha/r scales the adaptation's contribution to the final output |
| NF4 | "4-bit quantization" | Normal Float 4: a 4-bit data type with quantization levels at normal distribution quantiles, optimal for neural network weights |
| Adapter | "The small trained part" | The LoRA A and B matrices saved as a separate file (10-100MB), loadable on top of any copy of the base model |
| Target modules | "Which layers to LoRA" | The specific linear layers (q_proj, v_proj, etc.) where LoRA adapters are injected |
| Merging | "Bake it in" | Computing W + (alpha/r) * BA and replacing the original weight, eliminating the adapter overhead at inference |
| Paged optimizers | "Don't OOM during training" | Offloading optimizer states (Adam momentum, variance) to CPU when GPU memory is exhausted |
| Catastrophic forgetting | "Fine-tuning broke everything else" | When updating all weights causes the model to lose previously learned capabilities |

## आगे पढ़ना

- हु और अन्य, "लोराः लार्ज लैंग्वेज मॉडल का निम्न-रैंक अनुकूलन" (2021) -- निम्न-रैंक विघटन विधि की शुरुआत करने वाला मूल पेपर, जीपीटी-3 175 बी पर परीक्षण किया गया, जिसमें 4 की तरह निम्न रैंक है
- Dettmers et al., "QLoRA: क्वांटिज़ेड लैंग्वेज मॉडल का कुशल फाइनट्यूनिंग" (2023) -- NF4, डबल क्वांटिज़ेशन और पेजड ऑप्टिमाइज़र पेश करता है, जो एक एकल 48GB GPU पर 65B फाइनट्यूनिंग को सक्षम बनाता है
- PEFT पुस्तकालय प्रलेखन (huggingface.co/docs/peft) - Hugging Face पारिस्थितिकी तंत्र में LoRA, QLoRA और अन्य पैरामीटर-कुशल तरीकों के लिए मानक पुस्तकालय
- यादव और अन्य, "टीआईईएस-मर्जिंगः मर्ज मॉडलों में हस्तक्षेप को हल करना" (2023) -- गुणवत्ता में गिरावट के बिना कई लोरा एडाप्टरों को जोड़ने की तकनीक
- [Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (NeurIPS 2023)](https://arxiv.org/abs/2305.18290)-- डीपीओ व्युत्पन्न; प्राथमिकता-ट्यूनिंग चरण जो एसएफटी के बाद आता है, कोई इनाम मॉडल की आवश्यकता नहीं है।
- [TRL documentation](https://huggingface.co/docs/trl/)-- आधिकारिक संदर्भ के लिए `SFTTrainer`,`DPOTrainer`,`KTOTrainer`, और PEFT/bitsandbytes/Unsloth के साथ एकीकरण सतह।
- [Unsloth documentation](https://docs.unsloth.ai/)-- विलय कर्नेल जो बारीक-ट्यूनिंग आउटपुट को दोगुना करते हैं और मेमोरी को आधा करते हैं; TRL के तहत प्रदर्शन परत।
- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/)-- YAML-कॉन्फिगर बहु-GPU SFT/DPO/QLoRA ट्रेनर; हस्तलिखित स्क्रिप्ट के लिए कॉन्फिग-ए-कोड विकल्प।
