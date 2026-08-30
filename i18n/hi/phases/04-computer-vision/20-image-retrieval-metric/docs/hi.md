# छवि पुनर्प्राप्ति और मीट्रिक सीखने

> एक रिकवरी प्रणाली में अंतरिक्ष को एम्बेड करने में दूरी के आधार पर उम्मीदवारों को रैंक किया जाता है। मीट्रिक सीखने उस अंतरिक्ष को आकार देने का अनुशासन है ताकि दूरी का मतलब आप क्या चाहते हैं।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 14 (ViT), Phase 4 Lesson 18 (CLIP)
**Time:** ~45 minutes

## सीखने के लक्ष्य

- त्रिपुला, विपरीत और प्रॉक्सी आधारित मीट्रिक सीखने के नुकसान की व्याख्या करें और किसी दिए गए डेटासेट के लिए सही एक चुनें
- L2 मानकीकरण और कॉसिन समानता को सही ढंग से लागू करें और "एक ही आइटम" और "एक ही वर्ग" के बीच अंतर का लेखांकन करें
- एक FAISS सूचकांक बनाएं, इसे पाठ और छवि द्वारा क्वेरी करें, और एक लंबे समय तक किए गए क्वेरी सेट के लिए recall@K रिपोर्ट करें
- DINOv2, CLIP, और SigLIP का उपयोग शेल्फ से बाहर एम्बेडिंग रीढ़ की हड्डी के रूप में करें और जानें कि प्रत्येक जीतता है

## समस्या

उत्पादन दृष्टि में रिकवरी हर जगह हैः डुप्लिकेट डिटेक्शन, रिवर्स छवि खोज, दृश्य खोज ("समान उत्पादों को ढूंढें"), अनुहार की पुनः पहचान, निगरानी के लिए व्यक्ति की पुनः पहचान, ई-कॉमर्स के लिए इंस्टेंट-लेवल मिलान। उत्पाद प्रश्न हमेशा एक ही हैः "इस क्वेरी छवि को देखते हुए, मेरी सूची को रैंक करें। "

दो डिजाइन निर्णय पूरे सिस्टम को आकार देते हैं। एम्बेडिंग  कौन सा मॉडल वेक्टर उत्पन्न करता है। सूचकांक  पैमाने पर निकटतम पड़ोसियों को कैसे ढूंढें। दोनों 2026 में वस्तु हैं (एम्बेडिंग के लिए DINOv2, इंडेक्स के लिए FAISS), जो बार को बढ़ाता हैः कठिन हिस्सा आपके आवेदन के लिए *क्या समान है* को परिभाषित करना है, फिर एम्बेडिंग अंतरिक्ष को आकार देना ताकि दूरी मेल खाए।

यह आकार देना मीट्रिक सीखने है। यह एक छोटा लेकिन उच्च लीवरिंग अनुशासन है।

## अवधारणा

### एक नज़र में प्राप्त करना

```mermaid
flowchart LR
    Q["Query image<br/>or text"] --> ENC["Encoder"]
    ENC --> EMB["Query embedding"]
    EMB --> IDX["FAISS index"]
    CAT["Catalogue images"] --> ENC2["Encoder (same)"] --> IDX_BUILD["Build index"]
    IDX_BUILD --> IDX
    IDX --> RANK["Top-k nearest<br/>by cosine / L2"]
    RANK --> OUT["Ranked results"]

    style ENC fill:#dbeafe,stroke:#2563eb
    style IDX fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

### चार हार परिवार

| Loss | Requires | Pros | Cons |
|------|----------|------|------|
| **Contrastive** | (anchor, positive) + negatives | Simple, works with any pair label | Slow to converge without many negatives |
| **Triplet** | (anchor, positive, negative) | Intuitive; direct margin control | Hard-triplet mining is expensive |
| **NT-Xent / InfoNCE** | Pairs + batch-mined negatives | Scales to large batches | Needs big batch or momentum queue |
| **Proxy-based (ProxyNCA)** | Class labels only | Fast, stable, no mining | Can overfit to proxies on small datasets |

अधिकांश उत्पादन उपयोग मामलों के लिए, एक पूर्व-प्रशिक्षित रीढ़ की हड्डी के साथ शुरू करें और केवल एक मीट्रिक-लर्निंग बारीक-ट्यूनिंग जोड़ें यदि शेल्फ-ऑफ-द-शेल्फ एम्बेडमेंट आपके परीक्षण सेट पर खराब प्रदर्शन करते हैं।

### औपचारिक रूप से त्रिभुज हानि

```
L = max(0, ||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + margin)
```

लंगर खींचें `a`सकारात्मक के निकट `p`, नकारात्मक से दूर धक्का `n`, के साथ `margin`तीन-छवि संरचना किसी भी समानता क्रम के लिए सामान्यीकरण।

खनन विषयः आसान त्रिगुट (`n`पहले से ही दूर `a`) शून्य हानि का योगदान देते हैं; केवल हार्ड ट्रिपल नेटवर्क को पढ़ाते हैं।`n` से अधिक`p`लेकिन मार्जिन के भीतर) 2016 फेसनेट नुस्खा है और अभी भी हावी है।

### कॉसिन समानता बनाम L2

दो मीट्रिक, दो सम्मेलनः

- **Cosine**: वेक्टरों के बीच कोण। L2 मानकीकृत एम्बेडमेंट की आवश्यकता होती है।
- **L2**: यूक्लिडियन दूरी. कच्चे या सामान्यीकृत एम्बेडमेंट पर काम करता है, लेकिन आमतौर पर L2-नियमित + वर्ग L2 के साथ जोड़ा जाता है।

अधिकांश आधुनिक नेट के लिए दोनों समान हैंः `||a - b||^2 = 2 - 2 cos(a, b)`कब`||a|| = ||b|| = 1`. उस सम्मेलन को चुनें जो आपके एम्बेडिंग प्रशिक्षण से मेल खाता हो; उन्हें चुपचाप मिलाकर "सभी से करीब" का अर्थ बदल जाता है।

### याद@के

मानक निकासी मीट्रिकः

```
recall@K = fraction of queries where at least one correct match is in the top K results
```

रिपोर्ट recall@1, @5, @10 साथ में। एक recall@10 0.95 से ऊपर है और recall@1 0.5 से नीचे है इसका मतलब है कि एम्बेडिंग स्पेस में सही संरचना है लेकिन रैंकिंग शोर है  लंबे समय तक ठीक-ट्यून या एक फिर से रैंकिंग चरण का प्रयास करें।

डुप्लिकेट डिटेक्शन के लिए, सटीकता @ K अधिक मायने रखता है क्योंकि प्रत्येक झूठी सकारात्मक उपयोगकर्ता द्वारा दिखाई देने वाली गलती है। दृश्य खोज के लिए, recall@ K उत्पाद संकेत है।

### FAISS एक पैराग्राफ में

फेसबुक एआई समानता खोज. निकटतम पड़ोसी खोज के लिए वास्तविक पुस्तकालय. तीन सूचकांक विकल्पः

- `IndexFlatIP`/`IndexFlatL2` क्रूर बल, सटीक, कोई प्रशिक्षण. ~ 1M वेक्टर तक का उपयोग करें.
- `IndexIVFFlat` K कोशिकाओं में विभाजन, केवल निकटतम कुछ कोशिकाओं की खोज करें.
- `IndexHNSW` ग्राफ आधारित, कई क्वेरी के लिए सबसे तेज, बड़े सूचकांक आकार।

100k वेक्टर के लिए आप शायद चाहते हैं `IndexFlatIP`10M के लिए आप चाहते हैं`IndexIVFFlat`. उत्पाद मात्रा के साथ 100M+ के लिए (`IndexIVFPQ`) ।

### इंस्टेंस लेवल बनाम कैटेगरी लेवल रिट्रीव

एक ही नाम के साथ दो बहुत अलग समस्याएंः

- **Category-level** "मेरी सूची में बिल्लियों को ढूंढें।" वर्ग-सशर्त समानता; ऑफ-द-शेल्फ CLIP / DINOv2 एम्बेडमेंट अच्छी तरह से काम करते हैं।
- **Instance-level** "मेरी सूची में यह उत्पाद ढूंढें।" एक ही वर्ग के समान वस्तुओं के बीच बारीक भेदभाव की आवश्यकता है; शेल्फ में एम्बेडेड कम प्रदर्शन करते हैं; मीट्रिक सीखने के मामलों के साथ बारीक समायोजन।

मॉडल चुनने से पहले हमेशा पूछें कि आप किस एक को हल कर रहे हैं।

```figure
metric-embedding
```

## इसे बनाओ

### चरण 1: त्रिप्लट हानि

```python
import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=0.2):
    d_ap = F.pairwise_distance(anchor, positive, p=2)
    d_an = F.pairwise_distance(anchor, negative, p=2)
    return F.relu(d_ap - d_an + margin).mean()
```

एक पंक्ति. L2 मानकीकृत या कच्चे एम्बेडेड पर काम करता है.

### चरण 2: अर्ध-कठिन खनन

एम्बेडमेंट्स और लेबल के बैच को देखते हुए, प्रत्येक एंकर के लिए सबसे कठिन अर्ध-कठिन नकारात्मक खोजें।

```python
def semi_hard_negatives(emb, labels, margin=0.2):
    dist = torch.cdist(emb, emb)
    same_class = labels[:, None] == labels[None, :]
    diff_class = ~same_class
    N = emb.size(0)

    positives = dist.clone()
    positives[~same_class] = float("-inf")
    positives.fill_diagonal_(float("-inf"))
    pos_idx = positives.argmax(dim=1)

    semi_hard = dist.clone()
    semi_hard[same_class] = float("inf")
    d_ap = dist[torch.arange(N), pos_idx].unsqueeze(1)
    semi_hard[dist <= d_ap] = float("inf")
    neg_idx = semi_hard.argmin(dim=1)

    fallback_mask = semi_hard[torch.arange(N), neg_idx] == float("inf")
    if fallback_mask.any():
        hardest = dist.clone()
        hardest[same_class] = float("inf")
        neg_idx = torch.where(fallback_mask, hardest.argmin(dim=1), neg_idx)
    return pos_idx, neg_idx
```

प्रत्येक एंकर को वर्ग में सबसे कठिन सकारात्मक और अर्ध-कठोर नकारात्मक मिलता है जो सकारात्मक से आगे है लेकिन मार्जिन के भीतर है।

### चरण 3: याद करें

```python
def recall_at_k(query_emb, gallery_emb, query_labels, gallery_labels, k=1):
    sim = query_emb @ gallery_emb.T
    _, top_k = sim.topk(k, dim=-1)
    matches = (gallery_labels[top_k] == query_labels[:, None]).any(dim=-1)
    return matches.float().mean().item()
```

L2 मानकीकृत एम्बेडेड पर आंतरिक उत्पाद द्वारा शीर्ष-के कॉसिन द्वारा शीर्ष-के के बराबर है। कम से कम एक सही पड़ोसी के साथ क्वेरीओं के औसत अनुपात की रिपोर्ट करें।

### चरण 4: इसे एक साथ रखना

```python
import torch
import torch.nn as nn
from torch.optim import Adam

class Encoder(nn.Module):
    def __init__(self, in_dim=128, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

torch.manual_seed(0)
num_classes = 6
protos = F.normalize(torch.randn(num_classes, 128), dim=-1)

def sample_batch(bs=32):
    labels = torch.randint(0, num_classes, (bs,))
    x = protos[labels] + 0.15 * torch.randn(bs, 128)
    return x, labels

enc = Encoder()
opt = Adam(enc.parameters(), lr=3e-3)

for step in range(200):
    x, y = sample_batch(32)
    emb = enc(x)
    pos_idx, neg_idx = semi_hard_negatives(emb, y)
    loss = triplet_loss(emb, emb[pos_idx], emb[neg_idx])
    opt.zero_grad(); loss.backward(); opt.step()
```

कुछ सौ चरणों के बाद एम्बेडिंग क्लस्टर प्रति वर्ग एक क्लस्टर बनाते हैं।

## इसका प्रयोग करें

2026 में उत्पादन स्टैकः

- **DINOv2 + FAISS**सामान्य प्रयोजन के दृश्य निकासी। शेल्फ से काम करता है।
- **CLIP + FAISS** जब प्रश्न पाठ हैं।
- **Fine-tuned DINOv2 + FAISS** उदाहरण स्तर पर पुनर्प्राप्ती, फेस री-आईडी, फैशन, ई-कॉमर्स।
- **Milvus / Weaviate / Qdrant** FAISS या HNSW के आसपास प्रबंधित वेक्टर DB रैपर।

SOTA उदाहरण निकालने के लिए, नुस्खा हैः DINOv2 रीढ़ की हड्डी, एम्बेडिंग सिर जोड़ें, एक तिप्पी के साथ ठीक-ट्यून या उदाहरण-लेबल जोड़े पर InfoNCE हानि, FAISS में सूचकांक।

## इसे भेजें

इस पाठ से उत्पन्न होता हैः

- `outputs/prompt-retrieval-loss-picker.md` एक संकेत जो किसी दिए गए पुनर्प्राप्ति समस्या के लिए त्रिपुला / InfoNCE / ProxyNCA चुनता है।
- `outputs/skill-recall-at-k-runner.md` एक कौशल जो ट्रेन/वॉल/गैलरी विभाजन और उचित डेटा अनुबंध के साथ recall@K के लिए एक स्वच्छ मूल्यांकन हर्नस लिखता है।

## व्यायाम

1. **(Easy)**ऊपर दिए गए खेलौना उदाहरण को चलाएं। प्रशिक्षण से पहले और बाद में पीसीए के साथ एम्बेडमेंट को रेखांकित करें ताकि छह क्लस्टरों का गठन हो सके।
2. **(Medium)**एक प्रोक्सीएनसीए हानि कार्यान्वयन जोड़ेंः प्रति वर्ग एक सीखा "प्रॉक्सी", कॉसिन समानता पर मानक क्रॉस-एंट्रोपी। खिलौना डेटा पर अभिसरण गति बनाम तिपाई हानि की तुलना करें।
3. **(Hard)**1,000 ImageNet सत्यापन छवियों को ले लो, HuggingFace के माध्यम से DINOv2 के साथ एम्बेड करें, एक FAISS फ्लैट सूचकांक बनाएं, और उसी छवियों के खिलाफ रिकॉल@{1, 5, 10} रिपोर्ट करें जैसे क्वेरी (1.0) और ImageNet लेबल के साथ एक लंबे समय तक विभाजित के खिलाफ ग्राउंड सत्य।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Metric learning | "Shape the space" | Training an encoder so distances in its output space reflect a target similarity |
| Triplet loss | "Pull and push" | L = max(0, d(a, p) - d(a, n) + margin); the canonical metric-learning loss |
| Semi-hard mining | "Useful negatives" | Negatives further from the anchor than the positive but within margin; empirically the most informative |
| Proxy-based loss | "Class prototypes" | One learned proxy per class; cross-entropy over similarity-to-proxies; no pair mining |
| Recall@K | "Top-K hit rate" | Fraction of queries with at least one correct result in the top K |
| Instance retrieval | "Find this exact thing" | Fine-grained matching; off-the-shelf features usually underperform |
| FAISS | "The NN library" | Facebook's nearest-neighbour library; supports exact and approximate indexes |
| HNSW | "Graph index" | Hierarchical navigable small world; fast approximate NN with small memory overhead |

## आगे पढ़ना

- [FaceNet: A Unified Embedding for Face Recognition (Schroff et al., 2015)](https://arxiv.org/abs/1503.03832) त्रिपुट हानि / अर्ध-कठिन खनन कागज
- [In Defense of the Triplet Loss for Person Re-Identification (Hermans et al., 2017)](https://arxiv.org/abs/1703.07737) त्रिपुराण सूक्ष्म समायोजन के लिए व्यावहारिक मार्गदर्शिका
- [FAISS documentation](https://github.com/facebookresearch/faiss/wiki) प्रत्येक सूचकांक, प्रत्येक व्यापार-बदला
- [SMoT: Metric Learning Taxonomy (Kim et al., 2021)](https://arxiv.org/abs/2010.06927) आधुनिक घाटे और उनके संबंध का सर्वेक्षण
