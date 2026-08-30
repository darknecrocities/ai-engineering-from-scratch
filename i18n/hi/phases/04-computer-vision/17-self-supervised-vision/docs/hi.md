# स्व-निरीक्षण दृष्टि  सिमकलर, डिनो, एमएई

> लेबल पर्यवेक्षित दृष्टि का बोतल गला है। आत्म-नियंत्रित पूर्व प्रशिक्षण उन्हें हटा देता हैः 100 मिलियन लेबल रहित छवियों से दृश्य विशेषताएं सीखें, 10 हजार लेबल वाले पर बारीकी से ट्यून करें।

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 04 (Image Classification), Phase 4 Lesson 14 (ViT)
**Time:** ~75 minutes

## सीखने के लक्ष्य

- तीन प्रमुख स्व-निरीक्षण परिवारों का पता लगाएं  विपरीत (SimCLR), शिक्षक-छात्र (DINO), मुखौटा पुनर्निर्माण (MAE)  और बताएं कि प्रत्येक क्या अनुकूलित करता है
- एक InfoNCE हानि को खरोंच से लागू करें और समझाएं कि 512 बैच का काम क्यों होता है लेकिन 32 बैच का विफलता क्यों होती है
- समझाएं कि क्यों MAE का 75% मास्किंग अनुपात मनमाने ढंग से नहीं है और यह पाठ के लिए BERT के 15% से कैसे भिन्न है
- रैखिक जांच और शून्य शॉट निकासी के लिए DINOv2 या MAE ImageNet चेकपोस्ट का उपयोग करें

## समस्या

पर्यवेक्षित इमेजनेट में 1.3 मिलियन लेबल वाली छवियां हैं, जिनकी टिप्पणी करने के लिए अनुमानित $ 10 मिलियन की लागत है। चिकित्सा और औद्योगिक डेटा सेट छोटे और लेबल करने के लिए अधिक महंगे हैं। प्रत्येक दृष्टि टीम पूछती हैः क्या हम सस्ते अनलेबल डेटा पर प्री-ट्रेन कर सकते हैं  यूट्यूब फ्रेम, वेब क्रॉल, वेबकैम फुटेज, उपग्रह स्वीप  और फिर एक छोटे लेबल सेट पर ठीक-ठीक ट्यून कर सकते हैं?

आत्म-निरीक्षण सीखने का जवाब है। LAION या JFT पर प्रशिक्षित एक आधुनिक आत्म-निरीक्षण ViT जब ठीक से ट्यून किया जाता है तो निगरानी ImageNet सटीकता तक पहुंचता है या उससे बढ़ जाता है। यह पर्यवेक्षित प्री-प्रशिक्षण की तुलना में डाउनस्ट्रीम कार्यों (डिटेक्शन, खंडन, गहराई) को बेहतर स्थानांतरित करता है। DINOv2 (मेटा, 2023) और MAE (मेटा, 2022) स्थानांतरित दृष्टि सुविधाओं के लिए वर्तमान उत्पादन डिफ़ॉल्ट हैं।

अवधारणात्मक बदलाव यह है कि बहाना कार्य  वह चीज जो मॉडल को करने के लिए प्रशिक्षित किया गया है  डाउनस्ट्रीम कार्य नहीं होना चाहिए। इससे मॉडल को उपयोगी विशेषताएं सीखने को मजबूर किया जाता है। ग्रे स्केल की छवियों के रंग की भविष्यवाणी करें, छवियों को घुमाएं और मॉडल से घूर्णन को वर्गीकृत करने के लिए कहें, पैचों को मास्क करें और उन्हें पुनर्निर्माण करें  सभी काम किया है। इस पैमाने के तीन दृष्टिकोण विपरीत सीखने, शिक्षक-छात्रा डिस्टिलिशन और मास्क पुनर्निर्माण हैं।

## अवधारणा

### तीन परिवार

```mermaid
flowchart LR
    A["Contrastive<br/>SimCLR, MoCo, CLIP"] --> AT["positive pairs<br/>(same image, 2 augs)<br/>pulled together,<br/>negatives pushed apart"]
    B["Teacher-student<br/>DINO, BYOL, iBOT"] --> BT["student predicts<br/>teacher's output;<br/>teacher is EMA of student"]
    C["Masked reconstruction<br/>MAE, BEiT, SimMIM"] --> CT["mask 75% of patches;<br/>reconstruct pixel or<br/>token targets"]

    style A fill:#dbeafe,stroke:#2563eb
    style B fill:#fef3c7,stroke:#d97706
    style C fill:#dcfce7,stroke:#16a34a
```

### विपरीत सीखने (SimCLR)

एक छवि लें, दो यादृच्छिक वृद्धि करें, दो दृश्य प्राप्त करें। दोनों को एक ही एन्कोडर और एक प्रोजेक्शन हेड के माध्यम से फ़ीड करें। एक नुकसान को कम करें जो कहता है "ये दो एम्बेडमेंट करीब होना चाहिए" और "यह एम्बेडमेंट बैच में हर अन्य छवि के एम्बेडमेंट से दूर होना चाहिए।"

```
Loss for positive pair (z_i, z_j) among 2N views per batch:

   L_ij = -log( exp(sim(z_i, z_j) / tau) / sum_k in batch \ {i} exp(sim(z_i, z_k) / tau) )

sim = cosine similarity
tau = temperature (0.1 standard)
```

यह InfoNCE हानि है. इसमें प्रति सकारात्मक कई नकारात्मक की आवश्यकता होती है, इसलिए बैच आकार मायने रखता है।

### शिक्षक-छात्रा (DINO)

एक ही वास्तुकला वाले दो नेटवर्कः छात्र और शिक्षक। शिक्षक छात्र के वजन का एक घातीय चलती औसत (ईएमए) है। दोनों छवि के बढ़े हुए दृश्य देखते हैं। छात्र का उत्पादन शिक्षक के  स्पष्ट नकारात्मकताओं से मेल खाने के लिए प्रशिक्षित किया जाता है।

```
loss = CE( student_output(view_1),  teacher_output(view_2) )
     + CE( student_output(view_2),  teacher_output(view_1) )

teacher_weights = m * teacher_weights + (1 - m) * student_weights   (m ≈ 0.996)
```

"एक स्थिर की भविष्यवाणी" करने के लिए यह क्यों नहीं गिरता हैः शिक्षक का आउटपुट केंद्रित (प्रति आयाम औसत घटाएं) और तेज (छोटे तापमान से विभाजित करें) है। केंद्रित एक आयाम को हावी होने से रोकता है; तेज आउटपुट को समान रूप से गिरने से रोकता है।

DINO 142M क्यूरेट की गई छवियों पर DINOv2 का स्केल बढ़ाता है। परिणामस्वरूप शून्य-शॉट दृश्य निकासी और घने भविष्यवाणी के लिए वर्तमान SOTA है।

### मास्क पुनर्निर्माण (MAE)

एक ViT इनपुट के 75% पैचों को मास्क करें। केवल दृश्यमान 25% को एन्कोडर के माध्यम से पारित करें। एक छोटा डिकोडर मास्क किए गए स्थानों पर एन्कोडर के आउटपुट प्लस मास्क टोकन प्राप्त करता है, और मास्क किए गए पैचों के पिक्सेल को पुनर्निर्माण करने के लिए प्रशिक्षित होता है।

```
Encoder:  visible 25% of patches -> features
Decoder:  features + mask tokens at masked positions -> reconstructed pixels
Loss:     MSE between reconstructed and original pixels on masked patches only
```

MAE को काम करने के लिए प्रमुख डिजाइन विकल्पः

- **75% mask ratio** उच्च. एन्कोडर को अर्थिक विशेषताएं सीखने के लिए मजबूर करता है; 25% का पुनर्निर्माण लगभग तुच्छ होगा (पड़ोसी पिक्सल इतने सहसंबंधित हैं कि सीएनएन इसे नाखून दे सकता है) ।
- **Asymmetric encoder/decoder** बड़े ViT एन्कोडर केवल दृश्यमान पैच देखता है; एक छोटा डिकोडर (8 परत, 512-dim) पुनर्निर्माण संभालता है। 3x तेजी से पूर्व प्रशिक्षण naive BEiT की तुलना में।
- **Pixel-space reconstruction target** BEiT के टोकन लक्ष्य से सरल और ViT पर बेहतर काम करता है।

पूर्व प्रशिक्षण के बाद, डिकोडर को फेंक दें।

### 75% क्यों नहीं 15%

BERT 15% टोकन को कवर करता है। MAE 75% को कवर करता है। अंतर सूचना घनत्व है।

- प्राकृतिक भाषा में प्रति टोकन उच्च एंट्रॉपी है. 15% टोकन की भविष्यवाणी करना अभी भी कठिन है क्योंकि प्रत्येक मुखौटा स्थिति में कई व्यवहार्य पूर्णताएं हैं।
- छवि पैच में कम एंट्रॉपी होती है  एक अनमास्क पड़ोस अक्सर मास्क पैच के पिक्सल को लगभग सटीक रूप से निर्धारित करता है। भविष्यवाणी करने के लिए अर्थिक समझ की आवश्यकता होती है, आपको आक्रामक रूप से मास्क करना होगा।

75% पर्याप्त उच्च है कि सरल स्थानिक निष्कर्षण कार्य को हल नहीं कर सकता है; एन्कोडर को छवि सामग्री का प्रतिनिधित्व करना चाहिए।

### रैखिक जांच मूल्यांकन

स्व-निरीक्षण पूर्व प्रशिक्षण के बाद, मानक मूल्यांकन एक है **linear probe**: एन्कोडर को फ्रीज करें, ImageNet लेबल पर शीर्ष पर एक एकल रैखिक वर्गीकरण को प्रशिक्षित करें। शीर्ष-1 सटीकता रिपोर्ट करें।

- सिमकलर रेज़नेट-50: ~71% (2020)
- DINO ViT-S/16: ~77% (2021)
- एमएई ViT-L/16: ~76% (2022)
- DINOv2 ViT-g/14: ~86% (2023)

रैखिक जांच सुविधा गुणवत्ता का शुद्ध उपाय है; बारीक-बारी से समायोजित करने से आमतौर पर 2-5 अंक जोड़ते हैं लेकिन सिर को फिर से प्रशिक्षित करने के प्रभाव में भी मिश्रण होता है।

```figure
data-augmentation
```

## इसे बनाओ

### चरण 1: दो दृश्य वृद्धि पाइपलाइन

```python
import torch
import torchvision.transforms as T

two_view_train = lambda: T.Compose([
    T.RandomResizedCrop(96, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
])


class TwoViewDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
        self.aug = two_view_train()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, _ = self.base[i]
        v1 = self.aug(img)
        v2 = self.aug(img)
        return v1, v2
```

प्रत्येक __getitem__एक ही छवि के दो बढ़े हुए दृश्य लौटाता है; लेबल की आवश्यकता नहीं है।

### चरण 2: InfoNCE हानि

```python
import torch.nn.functional as F

def info_nce(z1, z2, tau=0.1):
    """
    z1, z2: (N, D) L2-normalised embeddings of paired views
    """
    N, D = z1.shape
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = z @ z.T / tau              # (2N, 2N)

    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)
```

फोन करने से पहले L2 एम्बेड को सामान्य करें। `tau=0.1`यह SimCLR डिफ़ॉल्ट है; कम नुकसान को तेज बनाता है और अधिक नकारात्मक की आवश्यकता होती है।

### चरण 3: सूचना आयोग के स्वास्थ्य जांच

```python
z1 = F.normalize(torch.randn(16, 32), dim=-1)
z2 = z1.clone()
loss_same = info_nce(z1, z2, tau=0.1).item()
z2_random = F.normalize(torch.randn(16, 32), dim=-1)
loss_random = info_nce(z1, z2_random, tau=0.1).item()
print(f"InfoNCE with identical pairs:  {loss_same:.3f}")
print(f"InfoNCE with random pairs:     {loss_random:.3f}")
```

समान जोड़े कम हानि (बड़े बैच और ठंडे तापमान के लिए 0 के करीब) देना चाहिए। यादृच्छिक जोड़े 16 जोड़े बैच के साथ log(2N-1) = ~log(31) = ~3.4 देना चाहिए।

### चरण 4: MAE शैली का मास्क

```python
def random_mask_indices(num_patches, mask_ratio=0.75, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_keep = int(num_patches * (1 - mask_ratio))
    perm = torch.randperm(num_patches, generator=g)
    visible = perm[:n_keep]
    masked = perm[n_keep:]
    return visible.sort().values, masked.sort().values


num_patches = 196
visible, masked = random_mask_indices(num_patches, mask_ratio=0.75)
print(f"visible: {len(visible)} / {num_patches}")
print(f"masked:  {len(masked)} / {num_patches}")
```

एक दिए गए बीज के लिए सरल, तेज़ और निर्धारक। वास्तविक MAE कार्यान्वयन इसे बैच करते हैं और प्रति नमूना मास्क रखते हैं।

## इसका प्रयोग करें

DINOv2 2026 में उत्पादन मानक हैः

```python
import torch
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")
model.eval()

# Per-image embeddings for zero-shot retrieval
with torch.no_grad():
    inputs = processor(images=[pil_image], return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0]  # CLS token
```

768-डिम एम्बेडिंग आधुनिक छवि पुनर्प्राप्ति, घने पत्राचार और शून्य-शॉट ट्रांसफर पाइपलाइन की रीढ़ है। डाउनस्ट्रीम कार्य पर ठीक-ठीक ट्यूनिंग के लिए शायद ही कभी एक रैखिक सिर से अधिक की आवश्यकता होती है।

छवि-पाठ एम्बेडमेंट के लिए, SigLIP या OpenCLIP समकक्ष है; MAE शैली के सूक्ष्म-ट्यूनिंग के लिए, `timm`प्रत्येक MAE चेकपोस्ट पर रेपो जहाज।

## इसे भेजें

इस पाठ से उत्पन्न होता हैः

- `outputs/prompt-ssl-pretraining-picker.md` एक प्रॉम्प्ट जो डेटासेट आकार, गणना और डाउनस्ट्रीम कार्य को देखते हुए SimCLR / MAE / DINOv2 का चयन करता है।
- `outputs/skill-linear-probe-runner.md` किसी भी ठंडे एन्कोडर + लेबल वाले डेटासेट के लिए रैखिक-सॉन्ड मूल्यांकन लिखने का कौशल।

## व्यायाम

1. **(Easy)**जांचें कि यदि आप अच्छी तरह से संरेखित एम्बेडमेंट के लिए तापमान कम करते हैं तो InfoNCE हानि घटती है और यदि आप यादृच्छिक एम्बेडमेंट के लिए तापमान कम करते हैं तो बढ़ जाती है। एक प्लॉट बनाएं `tau in [0.05, 0.1, 0.2, 0.5]`हानि के विरुद्ध।
2. **(Medium)**एक DINO शैली केंद्र बफर लागू करें। दिखाएं कि केंद्र के बिना, छात्र कुछ युगों के भीतर एक निरंतर वेक्टर में गिर जाता है।
3. **(Hard)**सीआईएफएआर -100 पर टीनीयूएनट का उपयोग करके टीनीयूएनट को पाठ 10 से रीढ़ की हड्डी के रूप में प्रशिक्षित करें। 10, 50 और 200 युगों पर रैखिक-सॉन्ड सटीकता की रिपोर्ट करें। दिखाएं कि एक एमएई-प्रशिक्षित रैखिक जांच उसी 1,000-छवि उपसमूह पर खरोंच से पर्यवेक्षित रैखिक जांच से बेहतर है।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Self-supervised | "Label-free" | A pretext task that produces useful representations from unlabelled data |
| Pretext task | "The fake task" | The objective used during SSL (reconstruct patches, match views); discarded after pretraining |
| Linear probe | "Frozen encoder + linear head" | Standard SSL evaluation: train only a linear classifier on top of frozen features |
| InfoNCE | "Contrastive loss" | softmax over cosine similarities; positive pair is the target class, all others are negatives |
| EMA teacher | "Moving-average teacher" | Teacher whose weights are an exponential moving average of the student's; used by BYOL, MoCo, DINO |
| Mask ratio | "% of patches hidden" | Fraction of patches masked during MAE; 75% for vision, 15% for text |
| Representation collapse | "Constant output" | SSL failure where the encoder outputs a constant vector for all inputs; prevented by centring, sharpening, or negatives |
| DINOv2 | "Production SSL backbone" | Meta's 2023 self-supervised ViT; strongest general-purpose image features in 2026 |

## आगे पढ़ना

- [SimCLR (Chen et al., 2020)](https://arxiv.org/abs/2002.05709) विपरीत सीखने का संदर्भ
- [DINO (Caron et al., 2021)](https://arxiv.org/abs/2104.14294) गति, केन्द्रितता, तेज करने वाले शिक्षक-छात्रा
- [MAE (He et al., 2022)](https://arxiv.org/abs/2111.06377) विटी के लिए पूर्व प्रशिक्षण के लिए मास्क ऑटोकोडर
- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193) उत्पादन सुविधाओं के लिए स्व-निरीक्षण वाले वीटी को स्केल करना
