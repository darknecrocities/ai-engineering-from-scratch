# छवि पीढ़ी  GAN

> एक GAN एक निश्चित खेल में दो तंत्रिका नेटवर्क है एक ड्रॉ, एक आलोचना। वे एक साथ बेहतर हो जाते हैं जब तक कि चित्र आलोचक को मूर्ख नहीं बनाते।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 3 Lesson 06 (Optimizers), Phase 3 Lesson 07 (Regularization)
**Time:** ~75 minutes

## सीखने के लक्ष्य

- जनरेटर और भेदभावकर्ता के बीच न्यूनतम खेल की व्याख्या करें और संतुलन p_model = p_data के अनुरूप क्यों है
- PyTorch में एक DCGAN लागू करें और इसे 60 से कम लाइनों में सुसंगत 32x32 सिंथेटिक छवियों उत्पन्न करने के लिए प्राप्त करें
- तीन मानक ट्रिक्स के साथ GAN प्रशिक्षण को स्थिर करेंः गैर-स saturating हानि, स्पेक्ट्रल मानदंड, TTUR (दो-टाइम-स्केल अपडेट नियम)
- प्रशिक्षण वक्रों को पढ़ें जो मोड कोलप, ओस्किलेशन और भेदभाव-जीत-पूर्ण से स्वस्थ अभिसरण को अलग करते हैं

## समस्या

वर्गीकरण एक नेटवर्क को छवियों को लेबल पर मैप करने के लिए सिखाता है। पीढ़ी समस्या को उलट देती हैः नए छवियों का नमूना लें जो ऐसा लगते हैं जैसे वे एक ही वितरण से आए हैं। कोई "सही" आउटपुट नहीं है जिसके खिलाफ आप भिन्न हो सकते हैं; केवल एक वितरण है जिसे आप नक्कल करना चाहते हैं।

मानक हानि फ़ंक्शन (MSE, क्रॉस-एंट्रोपी) "क्या यह नमूना वास्तविक वितरण से आया है" को माप नहीं सकते। प्रति पिक्सेल त्रुटि को कम करने से धुंधली औसत उत्पन्न होती है, यथार्थवादी नमूने नहीं। सफलता नुकसान को सीखना थाः एक दूसरे नेटवर्क को प्रशिक्षित करना जिसका काम वास्तविक से नकली को अलग करना है, और जनरेटर को धक्का देने के लिए अपने निर्णय का उपयोग करना।

GANs (Goodfellow et al., 2014) ने उस ढांचे को परिभाषित किया। 2018 तक StyleGAN 1024x1024 चेहरे का उत्पादन कर रहा था जो तस्वीरों से अलग नहीं हो सकते थे। तब से प्रसार मॉडल ने गुणवत्ता और नियंत्रण में सिंहासन संभाला है, लेकिन प्रत्येक चाल जो प्रसार को व्यावहारिक बनाती है  सामान्यीकरण विकल्प, लटेंट स्पेस, सुविधा हानि  पहली बार GAN पर समझा गया था।

## अवधारणा

### दोनों नेटवर्क

```mermaid
flowchart LR
    Z["z ~ N(0, I)<br/>noise"] --> G["Generator<br/>transposed convs"]
    G --> FAKE["Fake image"]
    REAL["Real image"] --> D["Discriminator<br/>conv classifier"]
    FAKE --> D
    D --> OUT["P(real)"]

    style G fill:#dbeafe,stroke:#2563eb
    style D fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

**generator**G शोर का वेक्टर लेता है `z`और एक छवि आउटपुट करता है।**discriminator**D एक छवि लेता है और एक एकल स्केलर आउटपुट करता हैः छवि वास्तविक होने की संभावना।

### खेल

जी चाहता है कि डी गलत हो। डी सही होना चाहता है। औपचारिक रूप सेः

```
min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

दाएं से बाएं पढ़ेंः D वास्तविक पर सटीकता अधिकतम कर रहा है (`log D(real)`) और नकली (`log (1 - D(fake))`) छवियों. G नकली पर D की सटीकता को कम कर रहा है  वह चाहता है `D(G(z))`उच्च होने के लिए.

गुडफेलो ने साबित किया कि इस न्यूनतम में वैश्विक संतुलन है जहां `p_G = p_data`, डी हर जगह 0.5 आउटपुट है, और उत्पन्न और वास्तविक वितरण के बीच जेन्सेन-शैनन विचलन शून्य है. कठिन भाग वहाँ प्राप्त करने के लिए है.

### न संतोषजनक हानि

उपरोक्त रूप संख्यात्मक रूप से अस्थिर है। प्रशिक्षण के शुरुआती समय में,`D(G(z))`हर नकली के लिए लगभग शून्य है, तो `log(1 - D(G(z)))`G के संबंध में गायब हो रहे gradients है. फिक्सः फ्लिप G का नुकसान.

```
L_D = -E_x[log D(x)] - E_z[log(1 - D(G(z)))]
L_G = -E_z[log D(G(z))]                          # non-saturating
```

अब कब`D(G(z))`जी का नुकसान बड़ा है और इसका ग्रेडिएंट सूचनात्मक है. हर आधुनिक जीएएन इस प्रकार के साथ ट्रेनों.

### डीसीजीएएन वास्तुकला नियम

Radford, Metz, Chintala (2015) ने वर्षों के असफल प्रयोगों को पांच नियमों में विभाजित किया जो GAN प्रशिक्षण को स्थिर बनाते हैंः

1. एकजुटता को चरणबद्ध कन्व (दोनों जाल) से प्रतिस्थापित करें।
2. G के आउटपुट और D के इनपुट को छोड़कर जनरेटर और भेदभावकर्ता दोनों में बैच मानदंड का उपयोग करें।
3. गहरे वास्तुकला पर पूरी तरह से जुड़े परतों को हटा दें।
4. G आउटपुट को छोड़कर सभी परतों पर ReLU का उपयोग करता है ([-1, 1] में आउटपुट के लिए टैन) ।
5. D सभी परतों पर LeakyReLU (नकारात्मक_झल = 0.2) का उपयोग करता है।

हर आधुनिक कंवा-आधारित GAN (स्टाइलगेन, बिगगेन, गिगागेन) अभी भी इन नियमों से शुरू होता है और एक-एक करके टुकड़े बदल देता है।

### विफलता मोड और उनके हस्ताक्षर

```mermaid
flowchart LR
    M1["Mode collapse<br/>G produces a narrow<br/>set of outputs"] --> S1["D loss low,<br/>G loss oscillating,<br/>sample variety drops"]
    M2["Vanishing gradients<br/>D wins completely"] --> S2["D accuracy ~100%,<br/>G loss huge and static"]
    M3["Oscillation<br/>G and D keep trading<br/>wins forever"] --> S3["Both losses swing<br/>wildly with no downward trend"]

    style M1 fill:#fecaca,stroke:#dc2626
    style M2 fill:#fecaca,stroke:#dc2626
    style M3 fill:#fecaca,stroke:#dc2626
```

- **Mode collapse**: G एक छवि पाता है जो D को मूर्ख बनाता है और केवल वही उत्पन्न करता है। फिक्सः मिनी बैच भेदभाव, स्पेक्ट्रल मानदंड, या लेबल-कंडीशनिंग जोड़ें।
- **Discriminator wins**D बहुत मजबूत हो जाता है, G के ग्रेडिएंट गायब हो जाते हैं। ठीकः छोटे D, कम D सीखने की दर, या वास्तविक लेबल पर लेबल चिकनाई लागू करें।
- **Oscillation**TTUR (D G से 2-4 गुना तेजी से सीखता है), या Wasserstein हानि पर स्विच करें।

### मूल्यांकन

जीएएन के पास कोई जमीन की सच्चाई नहीं है, तो आप कैसे जानते हैं कि वे काम कर रहे हैं?

- **Sample inspection** बस प्रत्येक युग के अंत में 64 नमूनों को देखें।
- **FID (Fréchet Inception Distance)** वास्तविक और उत्पन्न सेटों के वितरण के बीच की दूरी।
- **Inception Score** वृद्ध, अधिक नाजुक; एफआईडी पसंद करते हैं।
- **Precision/Recall for generative models** गुणवत्ता (सटीकता) और कवरेज (रिटर्न) को अलग से मापता है।

छोटे संश्लेषण डेटा रन के लिए नमूना निरीक्षण पर्याप्त है।

```figure
cv-gan-image
```

## इसे बनाओ

### चरण 1: जनरेटर

एक छोटा डीसीजीएएन जनरेटर जो 64 आयामी शोर लेता है और 32x32 छवि का उत्पादन करता है।

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=64, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feat * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 4, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat * 2, feat, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))
```

चार ट्रांसपोस्टेड कन्वे, प्रत्येक के साथ `kernel_size=4, stride=2, padding=1`तो वे साफ स्थानिक आकार दोगुना. आउटपुट सक्रियण में [-1, 1] via tanh.

### चरण 2: भेदभाव

जनरेटर का दर्पण. लीकरीलु, कदम convs, एक स्केलर लॉजिट के साथ समाप्त होता है.

```python
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feat, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat, feat * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 2, feat * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat * 4, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x).view(-1)
```

अंतिम कन्वर्ट एक `4x4` तक विशेषता मानचित्र`1x1`. आउटपुट प्रति छवि एक स्केलर है; हानि गणना के दौरान सिग्मोइड का उपयोग करें।

### चरण 3: प्रशिक्षण चरण

वैकल्पिकः एक बार डी अद्यतन, फिर जी एक बार, प्रत्येक बैच.

```python
import torch.nn.functional as F

def train_step(G, D, real, z, opt_g, opt_d, device):
    real = real.to(device)
    bs = real.size(0)

    # D step
    opt_d.zero_grad()
    d_real = D(real)
    d_fake = D(G(z).detach())
    loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
              + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
    loss_d.backward()
    opt_d.step()

    # G step
    opt_g.zero_grad()
    d_fake = D(G(z))
    loss_g = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
    loss_g.backward()
    opt_g.step()

    return loss_d.item(), loss_g.item()
```

`G(z).detach()`D चरण महत्वपूर्ण हैः हम नहीं चाहते कि ग्रेडिएंट G में प्रवाह के दौरान इसके अद्यतन. यह भूलना है कि क्लासिक शुरुआती बग है.

### चरण 4: सिंथेटिक आकारों पर पूर्ण प्रशिक्षण लूप

```python
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def synthetic_images(num=2000, size=32, seed=0):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((num, 3, size, size), dtype=np.float32) - 1.0
    for i in range(num):
        r = rng.uniform(6, 12)
        cx, cy = rng.uniform(r, size - r, size=2)
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        color = rng.uniform(-0.5, 1.0, size=3)
        for c in range(3):
            imgs[i, c][mask] = color[c]
    return torch.from_numpy(imgs)

device = "cuda" if torch.cuda.is_available() else "cpu"
data = synthetic_images()
loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=True)

G = Generator(z_dim=64, img_channels=3, feat=32).to(device)
D = Discriminator(img_channels=3, feat=32).to(device)
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(10):
    for (batch,) in loader:
        z = torch.randn(batch.size(0), 64, device=device)
        ld, lg = train_step(G, D, batch, z, opt_g, opt_d, device)
    print(f"epoch {epoch}  D {ld:.3f}  G {lg:.3f}")
```

`Adam(lr=2e-4, betas=(0.5, 0.999))`डीसीजीएएन डिफ़ॉल्ट है  कम बीटा 1 गति अवधि को विरोधी खेल को बहुत अधिक स्थिर करने से रोकता है।

### चरण 5: नमूनाकरण

```python
@torch.no_grad()
def sample(G, n=16, z_dim=64, device="cpu"):
    G.eval()
    z = torch.randn(n, z_dim, device=device)
    imgs = G(z)
    imgs = (imgs + 1) / 2
    return imgs.clamp(0, 1)
```

हमेशा नमूना लेने से पहले मूल्यांकन मोड पर स्विच करें। DCGAN के लिए यह मायने रखता है क्योंकि बैच मानदंड चलाने के आंकड़े बैच के आंकड़ों के बजाय उपयोग किए जाते हैं।

### चरण 6: स्पेक्ट्रल सामान्यीकरण

नेटवर्क की गारंटी देने वाले भेदभावकर्ता में बीएन के लिए एक ड्रॉप-इन प्रतिस्थापन 1-लिप्सचिट है। अधिकांश "डी बहुत मुश्किल से जीतता है" विफलताओं को ठीक करता है।

```python
from torch.nn.utils import spectral_norm

def build_sn_discriminator(img_channels=3, feat=64):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(img_channels, feat, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat, feat * 2, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 2, feat * 4, 4, 2, 1)),
        nn.LeakyReLU(0.2, inplace=True),
        spectral_norm(nn.Conv2d(feat * 4, 1, 4, 1, 0)),
    )
```

स्वैप `Discriminator`के लिए`build_sn_discriminator()`स्पेक्ट्रल मानदंड सबसे आसान एकल मजबूती उन्नयन है आप लागू कर सकते हैं।

## इसका प्रयोग करें

गंभीर पीढ़ी के लिए, पूर्व-प्रशिक्षित वजन का उपयोग करें या प्रसार पर स्विच करें। दो मानक पुस्तकालयः

- `torch_fidelity`अनुकूलन मूल्यांकन कोड लिखने के बिना अपने जनरेटर पर FID / आईएस गणना करता है।
- `pytorch-gan-zoo`(उपरास) और `StudioGAN`जहाज द्वारा DCGAN, WGAN-GP, SN-GAN, StyleGAN और BigGAN के परीक्षण किए गए कार्यान्वयन।

2026 में, GAN अभी भी वास्तविक समय छवि उत्पादन (लैटेंसी <10 ms), शैली हस्तांतरण, सटीक नियंत्रण के साथ छवि-से-छवि अनुवाद (Pix2Pix, CycleGAN) के लिए सबसे अच्छा विकल्प है। फैलाव फोटोरियलिज्म और पाठ कंडीशनिंग पर जीतता है।

## इसे भेजें

इस पाठ से उत्पन्न होता हैः

- `outputs/prompt-gan-training-triage.md` एक संकेत जो प्रशिक्षण वक्र विवरण पढ़ता है और विफलता मोड (मोड कोलप, डी-विन्स, ओस्किलेशन) प्लस एकल अनुशंसित फिक्स चुनता है।
- `outputs/skill-dcgan-scaffold.md` एक कौशल जो एक DCGAN से एक मंच लिखा है `z_dim`, लक्ष्य `image_size`और `num_channels`, प्रशिक्षण लूप और नमूना बचत सहित।

## व्यायाम

1. **(Easy)**सिंथेटिक सर्कल डेटासेट पर उपरोक्त DCGAN को प्रशिक्षित करें और प्रत्येक युग के अंत में 16 नमूनों का एक ग्रिड सहेजें। किस युग द्वारा उत्पन्न सर्कल स्पष्ट रूप से गोल हो जाते हैं?
2. **(Medium)**भेदभावकर्ता के बैच मानदंड को स्पेक्ट्रल मानदंड से बदलें. दोनों संस्करणों को एक साथ प्रशिक्षित करें. कौन सा तेजी से अभिसरण करता है? तीन बीजों में कौन सा कम भिन्नता है?
3. **(Hard)**एक सशर्त DCGAN लागू करेंः वर्ग लेबल को G और D दोनों में फ़ीड करें (G में शोर को एक-गर्म करें, D में एक वर्ग एम्बेडिंग चैनल को संकुचित करें) पाठ 7 से सिंथेटिक "चक्र बनाम वर्ग" डेटासेट पर अभ्यास करें और दिखाएं कि विशिष्ट लेबल के साथ नमूना ले कर वर्ग कंडीशनिंग काम करती है।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Generator (G) | "The draws-stuff net" | Maps noise to images; trained to fool the discriminator |
| Discriminator (D) | "The critic" | Binary classifier; trained to distinguish real from generated images |
| Minimax | "The game" | min over G, max over D of an adversarial loss; equilibrium is p_G = p_data |
| Non-saturating loss | "The numerically sane version" | G's loss is -log(D(G(z))) instead of log(1 - D(G(z))) to avoid vanishing gradients early in training |
| Mode collapse | "Generator makes one thing" | G produces only a small subset of the data distribution; fix with SN, minibatch discrimination, or larger batch |
| TTUR | "Two learning rates" | D learns faster than G, typically by a factor of 2-4; stabilises training |
| Spectral norm | "1-Lipschitz layer" | A weight-normalisation that bounds each layer's Lipschitz constant; stops D from becoming arbitrarily steep |
| FID | "Fréchet Inception Distance" | Distance between Inception-v3 feature distributions of real and generated sets; the standard evaluation metric |

## आगे पढ़ना

- [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661) कागज जो सब कुछ शुरू किया
- [DCGAN (Radford, Metz, Chintala, 2015)](https://arxiv.org/abs/1511.06434) जीएएन को प्रशिक्षित करने वाले वास्तुकला नियम
- [Spectral Normalization for GANs (Miyato et al., 2018)](https://arxiv.org/abs/1802.05957) सबसे उपयोगी स्थिरता चाल
- [StyleGAN3 (Karras et al., 2021)](https://arxiv.org/abs/2106.12423) सोटा गान; यह पिछले दशक के हर ट्रिक की सबसे बड़ी हिट एल्बम की तरह पढ़ता है
