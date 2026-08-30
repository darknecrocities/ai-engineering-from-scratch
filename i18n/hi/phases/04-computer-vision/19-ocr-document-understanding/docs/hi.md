# ओसीआर और दस्तावेज समझ

> ओसीआर एक तीन चरणों वाला पाइपलाइन है  पाठ बक्से का पता लगाएं, वर्णों को पहचानें, फिर उन्हें बिछाएं। प्रत्येक आधुनिक ओसीआर प्रणाली इन चरणों को फिर से क्रमबद्ध करती है या उन्हें मिला देती है।

**Type:** Learn + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06 (Detection), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~45 minutes

## सीखने के लक्ष्य

- क्लासिक ओसीआर पाइपलाइन (डिटेक्ट -> पहचान -> लेआउट) और आधुनिक एंड-टू-एंड विकल्पों (डोनट, क्यूवेन-वीएल-ओसीआर) का पता लगाएं
- सीटीसी (कनेक्शनिस्ट टाइमओरियल क्लासिफिकेशन) को क्रमिक-से-क्रमिक ओसीआर प्रशिक्षण के लिए लागू करना
- बिना प्रशिक्षण के उत्पादन दस्तावेज पार्सिंग के लिए PaddleOCR या EasyOCR का उपयोग करें
- ओसीआर, लेआउट पार्सिंग और दस्तावेज़ समझ में अंतर करें  और प्रत्येक कार्य के लिए सही उपकरण चुनें

## समस्या

पाठ से भरे चित्र हर जगह हैंः रसीदें, चालान, आईडी, स्कैन की गई किताबें, फॉर्म, व्हाइटबोर्ड, साइन, स्क्रीनशॉट। उनमें से संरचित डेटा निकालना  न केवल वर्ण, बल्कि "यह कुल राशि है"  उच्चतम मूल्य वाली लागू दृष्टि समस्याओं में से एक है।

क्षेत्र तीन कौशल परतों में विभाजित हैः

1. **OCR proper**: पिक्सेल को पाठ में बदल दें।
2. **Layout parsing**: समूह ओसीआर आउटपुट क्षेत्रों (शीर्षक, शरीर, तालिका, हेडर) में।
3. **Document understanding**: लेआउट से संरचित फ़ील्ड ("इंवॉइस_टोटल = $42.50") निकालें।

प्रत्येक परत में शास्त्रीय और आधुनिक दृष्टिकोण होते हैं, और "मुझे एक छवि से पाठ चाहिए" और "मुझे इस रसीद से कुल राशि चाहिए" के बीच का अंतर अधिकांश टीमों की कल्पना से बड़ा है।

## अवधारणा

### शास्त्रीय पाइपलाइन

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **Text detection**प्रति पंक्ति या प्रति शब्द चतुर्भुज उत्पन्न करता है।
- **Recognition**प्रत्येक क्षेत्र को एक निश्चित ऊंचाई पर काटता है, एक चरित्र अनुक्रम उत्पन्न करने के लिए एक सीएनएन + BiLSTM + सीटीसी चलाता है।
- **Layout**पढ़ने के क्रम को पुनर्निर्माण करता है (लैटिन के लिए शीर्ष से नीचे, बाएं से दाएं; अरबी, जापानी के लिए अलग) ।

### एक पैराग्राफ में सीटीसी

OCR मान्यता एक निश्चित लंबाई के फीचर मानचित्र से एक चर-लंबाई अनुक्रम का उत्पादन करती है। CTC (Graves et al., 2006) आपको वर्ण-स्तर संरेखण के बिना इसे प्रशिक्षित करने देता है। मॉडल प्रत्येक समय चरण पर एक वितरण (शब्द + खाली) आउटपुट करता है; CTC हानि सभी संरेखणों पर हाशिए पर छोड़ता है जो पुनरावृत्ति और रिक्त स्थानों को हटाने के बाद लक्षित पाठ में कम हो जाते हैं।

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

सीटीसी इस कारण है कि सीआरएनएन ने 2015 में काम किया और अभी भी 2026 में अधिकांश उत्पादन ओसीआर मॉडल को प्रशिक्षित करता है।

### आधुनिक अंत-से-अंत मॉडल

- **Donut**(किम और अन्य, 2022)  एक ViT एन्कोडर + एक पाठ डिकोडर; एक छवि पढ़ता है और सीधे JSON उत्सर्जित करता है। कोई पाठ डिटेक्टर, कोई लेआउट मॉड्यूल नहीं।
- **TrOCR** लाइन स्तर के ओसीआर के लिए वीटी + ट्रांसफार्मर डिकोडर।
- **Qwen-VL-OCR / InternVL** पूर्ण दृष्टि-भाषा मॉडल ओसीआर कार्यों के लिए ठीक से समायोजित; जटिल दस्तावेजों पर 2026 में सर्वोत्तम सटीकता।
- **PaddleOCR** परिपक्व उत्पादन पैकेज में क्लासिक डीबी + सीआरएनएन पाइपलाइन; अभी भी ओपन सोर्स वर्कहॉर्स।

अंत-से-अंत मॉडल को अधिक डेटा और गणना की आवश्यकता होती है लेकिन बहु-चरण पाइपलाइनों के त्रुटि संचय को छोड़ दें।

### लेआउट पार्सिंग

संरचित दस्तावेजों के लिए, एक लेआउट डिटेक्टर (लेआउटLMv3, डॉकलेनेट) चलाएं जो प्रत्येक क्षेत्र को लेबल करता हैः शीर्षक, पैराग्राफ, आकृति, तालिका, फुटनोट। पढ़ने का क्रम फिर "लेआउट क्रम में क्षेत्रों के माध्यम से दोहराया जाता है, संगत" हो जाता है।

प्रपत्रों के लिए प्रयोग करें **Key-Value extraction**मॉडल (दृश्य-समृद्ध दस्तावेजों के लिए डोनेट, सादे स्कैन के लिए लेआउटLMv3) । वे छवि + पता लगाया पाठ + स्थान लेते हैं और संरचित कुंजी-मूल्य जोड़े की भविष्यवाणी करते हैं।

### मूल्यांकन मेट्रिक्स

- **Character Error Rate (CER)** लेवेंसस्टीन दूरी / संदर्भ लंबाई। निचला बेहतर है। उत्पादन लक्ष्यः < 2% स्वच्छ स्कैन पर।
- **Word Error Rate (WER)** शब्द स्तर पर भी यही।
- **F1 on structured fields** प्रमुख मूल्य के कार्यों के लिए; यह मापें कि क्या `{invoice_total: 42.50}`सही दिखाई देता है।
- **Edit distance on JSON** अंत से अंत तक दस्तावेज़ विश्लेषण के लिए; डोनट पेपर ने रूख संपादन दूरी को सामान्य बनाया।

```figure
cv3-ctc-collapse
```

## इसे बनाओ

### चरण 1: सीटीसी हानि + लोभी डिकोडर

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss`जब उपलब्ध हो तो कुशल CuDNN कार्यान्वयन का उपयोग करता है। लालची डिकोडर एक बीम खोज से सरल है और आमतौर पर 1% सीईआर के भीतर है।

### चरण 2: छोटा सीआरएनएन पहचानकर्ता

लाइन ओसीआर के लिए न्यूनतम सीएनएन + बीएलएसटीएम।

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

निश्चित ऊंचाई इनपुट (सीएनएन अधिकतम पूल ऊंचाई को 1 तक) चौड़ाई सीटीसी के लिए समय आयाम है।

### चरण 3: सिंथेटिक ओसीआर

अंत-से-अंत धुएं परीक्षण के लिए काले-से-सफेद अंक स्ट्रिंग उत्पन्न करें।

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

एक वास्तविक ओसीआर डेटासेट फ़ॉन्ट, शोर, घूर्णन, धुंधलापन और रंग जोड़ता है। ऊपर पाइपलाइन समान है।

### चरण 4: प्रशिक्षण स्केच

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

इस क्षुल्लक सिंथेटिक डेटा पर 200 चरणों पर नुकसान ~ 3 से ~ 0.2 तक गिरना चाहिए।

## इसका प्रयोग करें

तीन उत्पादन मार्गः

- **PaddleOCR** परिपक्व, तेज, बहुभाषी। एक पंक्ति का उपयोगः `paddleocr.PaddleOCR(lang="en").ocr(image_path)`. .
- **EasyOCR** पायथन-नेटिव, बहुभाषी, पायटॉर्च रीढ़ की हड्डी।
- **Tesseract** शास्त्रीय; अभी भी पुराने स्कैन किए गए दस्तावेजों के लिए उपयोगी है जब मॉडल संघर्ष करते हैं।

अंत-से-अंत दस्तावेज़ विश्लेषण के लिए, डोनट या वीएलएम का उपयोग करेंः

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

रिसीव, चालान और दोहराए जाने योग्य संरचना वाले फॉर्म के लिए, डोनट को ठीक से ट्यून करें। मनमाने दस्तावेजों या तर्क के साथ ओसीआर के लिए, क्यूवेन-वीएल-ओसीआर जैसे वीएलएम वर्तमान डिफ़ॉल्ट है।

## इसे भेजें

इस पाठ से उत्पन्न होता हैः

- `outputs/prompt-ocr-stack-picker.md` एक संकेत जो टेसरैक्ट / पैडलओसीआर / डोनट / वीएलएम-ओसीआर को दस्तावेज़ प्रकार, भाषा और संरचना के अनुसार चुनता है।
- `outputs/skill-ctc-decoder.md` एक कौशल जो लालची और बीम-सर्च सीटीसी डिकोडर को खरोंच से लिखता है, जिसमें लंबाई सामान्यीकरण भी शामिल है।

## व्यायाम

1. **(Easy)**500 चरणों के लिए 5-अंकीय यादृच्छिक संख्यात्मक स्ट्रिंग पर TinyCRNN को प्रशिक्षित करें। एक लंबे समय तक आयोजित सेट पर सीईआर रिपोर्ट करें।
2. **(Medium)**लालची डिकोडिंग को बीम सर्च (beam_width=5) से बदलें। CER डेल्टा रिपोर्ट करें। किस इनपुट पर बीम सर्च जीतता है?
3. **(Hard)**20 रसीदों के सेट पर PaddleOCR का उपयोग करें, लाइन आइटम निकालें, और {item_name, price} जोड़े के लिए हाथ से लेबल किए गए ग्राउंड सत्य के खिलाफ F1 गणना करें।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| OCR | "Text from pixels" | Turning image regions into character sequences |
| CTC | "Alignment-free loss" | Loss that trains a sequence model without per-timestep labels; marginalises over alignments |
| CRNN | "Classic OCR model" | Conv feature extractor + BiLSTM + CTC; the 2015 baseline still used in production |
| Donut | "End-to-end OCR" | ViT encoder + text decoder; emits JSON directly from image |
| Layout parsing | "Find regions" | Detect and label Title/Table/Figure/Paragraph regions in a document |
| Reading order | "Text sequence" | Ordering of recognised regions into a sentence; trivial for Latin, non-trivial for mixed layouts |
| CER / WER | "Error rates" | Levenshtein distance / reference length at character or word granularity |
| VLM-OCR | "LLM that reads" | A vision-language model trained or prompted for OCR tasks; current SOTA on complex documents |

## आगे पढ़ना

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) मूल सीएनएन+आरएनएन+सीटीसी वास्तुकला
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) मूल सीटीसी पेपर; एल्गोरिथम विचारों से घने पैक
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664) ओसीआर मुक्त दस्तावेज़ समझ ट्रांसफार्मर
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) ओपन सोर्स प्रोडक्शन ओसीआर स्टैक
