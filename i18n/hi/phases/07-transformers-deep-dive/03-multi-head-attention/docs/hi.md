# बहु-उपदेश्य ध्यान

> एक ध्यान सिर एक समय में एक संबंध सीखता है आठ सिर आठ सीखते हैं सिर मुक्त हैं उनमें से अधिक ले लो

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 minutes

## समस्या

एक एकल स्व-ध्यान सिर एक ध्यान मैट्रिक्स की गणना करता है। यह मैट्रिक्स एक प्रकार का संबंध कैप्चर करता है  आमतौर पर वह जो प्रशिक्षण संकेत जो भी है पर नुकसान को कम करता है। यदि आपके डेटा में विषय-क्रियापद समझौते, सह-संदर्भ, लंबी दूरी की प्रवचन और वाक्य रचना के टुकड़े हैं जो सभी एक साथ उलझे हुए हैं, तो एक एकल सिर उन्हें एक एकल नरम-अधिक वितरण में स्मूइड करता है और आधा संकेत खो देता है।

2017 के वास्वनी पेपर से फिक्सः समानांतर में कई ध्यान फ़ंक्शन चलाएं, प्रत्येक अपने स्वयं के क्यू, के, वी प्रोजेक्शन के साथ, और आउटपुट को एक साथ जोड़ें। प्रत्येक सिर आयाम की एक छोटी उप-स्थान में संचालित होता है `d_model / n_heads`कुल मापदंड समान रहते हैं. अभिव्यक्ति शक्ति बढ़ जाती है.

मल्टी-हेड ध्यान 2026 जहाजों में प्रत्येक ट्रांसफार्मर के लिए डिफ़ॉल्ट है। एकमात्र तर्क *कितने* हेड के बारे में है और क्या कुंजी और मान अनुमान साझा करते हैं (समूह-प्रश्न ध्यान, मल्टी-प्रश्न ध्यान, मल्टी-हेड लातेंट ध्यान) ।

## अवधारणा

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Split.**ले लो`X`आकार का `(N, d_model)`. प्रत्येक आकार के Q, K, V तक परियोजना `(N, d_model)`. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .`(N, n_heads, d_head)`कहाँ`d_head = d_model / n_heads`. ट्रांसपॉजिट करने के लिए`(n_heads, N, d_head)`. .

**Attend in parallel.**प्रत्येक सिर के अंदर स्केल बिंदु उत्पाद ध्यान चलाएं. प्रत्येक सिर उत्पादन करता है`(N, d_head)`. सिर एम्बेडिंग के विभिन्न उप-स्थानों पर काम करते हैं और ध्यान गणना के दौरान कभी बात नहीं करते हैं।

**Concatenate and project.**स्टैक सिर वापस करने के लिए`(N, d_model)`और एक सीखा आउटपुट मैट्रिक्स से गुणा `W_o`आकार का `(d_model, d_model)`. .`W_o`जहाँ सिर मिलते हैं.

**Why it works.**प्रत्येक सिर प्रतिनिधित्व बजट के लिए दूसरों के साथ प्रतिस्पर्धा किए बिना विशेषज्ञता प्राप्त कर सकता है। 20192024 के सर्वेक्षण अध्ययनों में प्रमुख भूमिकाओं को अलग-अलग दिखाया गया हैः स्थितिगत सिर, पिछले टोकन का पालन करने वाला सिर, कॉपी हेड, नामित इकाई के सिर, प्रेरण सिर (जो संदर्भ में सीखने के आधार पर हैं) ।

**The 2026 lineage of variations:**

| Variant | Q heads | K/V heads | Used by |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA आधुनिक डिफ़ॉल्ट है क्योंकि यह KV-कैश मेमोरी को घटाने के कारक से कम करता है `N/G`MLA एक लटते स्थान में K/V को संपीड़ित करके और आगे बढ़ता है, फिर गणना समय पर वापस प्रक्षेपित करके FLOPs की लागत होती है, बहुत अधिक मेमोरी बचाता है।

```figure
multihead-split
```

## इसे बनाओ

### चरण 1: हमारे पास पहले से ही एकल-मुख ध्यान से सिर विभाजित करें

ले लो `SelfAttention`पाठ 02 से और इसे एक विभाजित/कंकट जोड़ी के साथ लपेटें।`code/main.py`एक नंबरी कार्यान्वयन के लिए; तर्क हैः

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

एक रीफॉर्म और एक ट्रांसपोज. कोई लूप. यह ठीक है कि PyTorch के तहत क्या करता है.`nn.MultiheadAttention`. .

### चरण 2: प्रति व्यक्ति स्केल-डॉट-उत्पाद ध्यान चलाएं

प्रत्येक सिर को Q, K, V का अपना स्लाइस मिलता है। ध्यान एक बैचदार मत्मुल बन जाता हैः

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

असली हार्डवेयर पर `Qh @ Kh.transpose(...)`एक है`bmm`. GPU एक ही बैच आकार के matmul की देखता है .`(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`सिर जोड़ना मुफ़्त है.

### चरण 3: समूह-प्रश्न ध्यान संस्करण

केवल कुंजी और मूल्य अनुमानों को बदलते हैं।`n_heads`समूहों; K और V प्राप्त `n_kv_heads < n_heads`समूहों में और मेल खाने के लिए दोहराया जाता हैः

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

निष्कर्ष यह स्मृति बचाता है क्योंकि केवल`n_kv_heads`प्रतियां KV कैश में रहते हैं, नहीं `n_heads`. Llama 3 70B 8 KV सिर के साथ 64 क्वेरी हेड का उपयोग करता है  एक 8× कैश संकुचित.

### चरण 4: जांचें कि प्रत्येक सिर ने क्या सीखा

चार सिरों के साथ एक छोटे वाक्य पर MHA चलाएं। प्रत्येक सिर के लिए, प्रिंट करें `(N, N)`ध्यान मैट्रिक्स. आप देखेंगे अलग सिर अलग संरचना चुनते हैं यादृच्छिक आरंभिकरण के साथ भी  जो कि आंशिक संकेत है, आंशिक रूप से घूर्णन समता में उप-स्थानों.

## इसका प्रयोग करें

PyTorch में, एक पंक्ति संस्करणः

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ के अनुसार GQA:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**How many heads?**2026 में उत्पादन मॉडल से अंगूठे के नियमः

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head`लगभग हमेशा 64 या 128 पर लैंड करता है। यह इकाई है कि एक सिर कितना "देख सकता है।" 32 से नीचे गिरता है और सिर स्केलिंग कारक से लड़ना शुरू करते हैं।`sqrt(d_head)`256 से ऊपर जाने पर आप "बहुत छोटे विशेषज्ञ" लाभ खो देते हैं।

## इसे भेजें

देखो`outputs/skill-mha-configurator.md`. कौशल पैरामीटर बजट, अनुक्रम लंबाई और तैनाती लक्ष्य के अनुसार नए ट्रांसफार्मर के लिए सिर की संख्या, kv-head की संख्या और प्रोजेक्शन रणनीति की सिफारिश करता है।

## व्यायाम

1. **Easy.**MHA से ले लो `code/main.py`और परिवर्तन`n_heads`1 से 16 तक `d_model=64`एक सिंथेटिक कॉपी कार्य पर एक छोटे से एक परत मॉडल के नुकसान की योजना. अधिक सिर मदद, पठार, या चोट?
2. **Medium.**MQA (सभी क्वेरी हेड्स में साझा एक KV हेड) को लागू करें। मापें कि पैरामीटर की गिनती कितनी बूंदों बनाम पूर्ण MHA के खिलाफ गिरती है। गणना करें कि N = 2048 के लिए निष्कर्ष पर KV-कैश आकार कितना छोटा होता है।
3. **Hard.**मल्टी-हेड लटेंट ध्यान का एक छोटा संस्करण लागू करेंः एक रैंक-`r`लटेंट, KV कैश में लटेंट स्टोर, ध्यान समय पर decompress.`r`क्या कैश मेमोरी पूर्ण MHA के 1/8 से नीचे जाती है जबकि गुणवत्ता सत्यापन पीपीएल के 1 बिट के भीतर रहती है?

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Head | "A single attention circuit" | One Q/K/V projection of dimension `d_head = d_model / n_heads` with its own attention matrix. |
| d_head | "Head dimension" | Per-head hidden width; almost always 64 or 128 in production. |
| Split / combine | "Reshape tricks" | `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose around attention. |
| W_o | "Output projection" | `(d_model, d_model)` matrix applied after concatenating heads; where heads mix. |
| MQA | "One KV head" | Multi-Query Attention: single shared K/V projection. Smallest KV cache, some quality loss. |
| GQA | "The default since Llama 2" | Grouped-Query Attention with `n_kv_heads < n_heads`; repeats to match Q. |
| MLA | "DeepSeek's trick" | Multi-head Latent Attention: K,V compressed to low-rank latent, decompressed at attend time. |
| Induction head | "The circuit behind in-context learning" | A pair of heads that detect previous occurrences and copy what followed them. |

## आगे पढ़ना

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) मूल बहु-मुख्य विनिर्देश।
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) एमसीए पेपर।
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) प्रशिक्षण के बाद एमएचए को जीएए में कैसे परिवर्तित किया जाए।
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) MLA और यह कैश मेमोरी पर MHA/GQA से क्यों बेहतर है।
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) यंत्रसामग्री देखो कि वास्तव में सिर क्या करते हैं।
