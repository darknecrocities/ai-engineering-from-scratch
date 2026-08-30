# स्थिति एन्कोडिंग  सिनोसाइडल, रोपी, एलीबी

> ध्यान परमिट-अनवियरेंट है। "मक्खी मैट पर बैठी" और "मैट पर बैठी बिल्ली" स्थिति सिग्नल के बिना एक ही आउटपुट का उत्पादन करते हैं। तीन एल्गोरिदम इसे  प्रत्येक "स्थिति" के अर्थ पर एक अलग दांव के साथ तय करते हैं।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 minutes

## समस्या

स्केल बिंदु उत्पाद ध्यान आदेश अंधा है। ध्यान मैट्रिक्स`softmax(Q K^T / √d) V`जोड़ी के समानताओं से गणना की जाती है।`X`ध्यान के अंदर कुछ भी स्थिति की परवाह नहीं करता है।

यह शब्द के बैग मॉडल में कोई बग नहीं है। भाषा, कोड, ऑडियो, वीडियो के लिए  कुछ भी जहां आदेश अर्थ है  यह घातक है।

समाधान किसी तरह एम्बेडमेंट में स्थिति इंजेक्ट करने के लिए है। तीन उत्तर युगोंः

1. **Absolute sinusoidal**(वस्वनी 2017) । जोड़ें `sin/cos`सरल, सीखने योग्य मुक्त, प्रशिक्षित लंबाई से परे खराब रूप से बाहर निकाले।
2. **RoPE — Rotary Position Embeddings**(Su 2021). Q और K वेक्टरों को स्थिति के समानुपातिक कोण से घुमाएं। बिंदु उत्पाद में सीधे * सापेक्ष * स्थिति को एन्कोड करता है। 2026 में प्रमुख।
3. **ALiBi — Attention with Linear Biases**(प्रेस 2022) पूरी तरह से एम्बेडिंग को छोड़ दें; दूरी के आधार पर ध्यान स्कोर पर प्रति सिर रैखिक दंड जोड़ें। उत्कृष्ट लंबाई निष्कर्षण।

2026 तक, अनिवार्य रूप से हर सीमा मुक्त मॉडल RoPE का उपयोग करता हैः Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi। कुछ लंबे संदर्भ वाले मॉडल ALiBi या इसके आधुनिक संस्करणों का उपयोग करते हैं। पूर्ण सिनोसाइडल ऐतिहासिक है।

## अवधारणा

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### पूर्ण सिनोसोइडल

एक निश्चित मैट्रिक्स पूर्व-गणना `PE`आकार का `(max_len, d_model)`:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

तो `X' = X + PE[:N]`ध्यान से पहले. प्रत्येक आयाम एक अलग आवृत्ति पर एक sinus है. मॉडल चरण पैटर्न से स्थिति पढ़ने के लिए सीखता है.`max_len`: मॉडल को कुछ भी नहीं बताया कि स्थिति 2048 पर क्या होता है जब उसने केवल स्थिति 02047 देखी।

### रोपी

Q और K वेक्टरों (इनबेड नहीं) घुमाएं। एक जोड़ी आयामों के लिए `(2i, 2i+1)`:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

उसी घूर्णन को स्थिति के साथ कुंजी पर लागू करें `pos_k`. डॉट उत्पाद `q'_m · k'_n``(m - n)`अकेले।**the attention score depends only on the relative distance**, भले ही घूर्णन पूर्ण स्थिति से कुंजीबद्ध किया गया था.

RoPE का विस्तार करनाः `base`Llama 3 इस तरह 8K से 128K संदर्भ तक विस्तारित किया गया।

### अलैबी

ध्यान सीधे स्कोर करता है:

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

कहाँ`m_h`सिर के लिए विशिष्ट ढलान है (जैसे `1 / 2^(8·h/H)`) निकट टोकन को बढ़ावा दिया जाता है; दूर टोकन को दंडित किया जाता है। कोई प्रशिक्षण समय लागत नहीं। पेपर में लंबाई एक्सट्रापोलेशन सिनोसाइडल से अधिक है और इसकी मूल प्रशिक्षित लंबाई पर RoPE से मेल खाता है।

### 2026 में क्या चुनना है

| Variant | Extrapolation | Training cost | Used by |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | poor | free | original transformer, early BERT |
| Learned absolute | none | tiny | GPT-2, GPT-3 |
| RoPE | good with scaling | free | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | excellent | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | excellent | free | BLOOM, MPT, Baichuan |

रोपी जीता क्योंकि यह वास्तुकला को बदलने के बिना ध्यान आकर्षित करता है, सापेक्ष स्थिति को कोड करता है, और इसकी `base`हाइपरपरमैटर लंबे संदर्भ के लिए एक साफ बटन देता है।

```figure
rope-explorer
```

## इसे बनाओ

### चरण 1: सिनोसॉइडल एन्कोडिंग

देखो`code/main.py`चार पंक्ति गणनाः

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

पहली ध्यान परत से पहले इसे एम्बेडिंग मैट्रिक्स में जोड़ें।

### चरण 2: Q, K पर RoPE लागू किया गया

RoPE Q और K पर स्थान पर संचालित होता है। प्रत्येक जोड़ी के लिए dims:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

महत्वपूर्ण: स्थिति पर Q पर एक ही फ़ंक्शन लागू करें `m`और K स्थिति में `n`उनके डॉट उत्पाद एक उठाता है`cos((m-n)·θ_i)`ध्यान मुफ्त में सापेक्ष स्थिति सीखता है।

### चरण 3: ALiBi ढलान और पूर्वाग्रह

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

जोड़ें `bias[h]``(seq_len, seq_len)`ध्यान स्कोर मैट्रिक्स सिर `h`, फिर सॉफ्टमैक्स.

### चरण 4: RoPE के सापेक्ष दूरी गुण की जांच करें

दो यादृच्छिक वेक्टर चुनें `a, b`. घूमकर`(pos_a, pos_b)`. . फिर से .`(pos_a + k, pos_b + k)`. दोनों बिंदु उत्पादों को फ्लोटिंग-पॉइंट त्रुटि के भीतर मेल खाना चाहिए. यह गुण RoPE का पूरा बिंदु है  यह पूर्ण ऑफसेट के लिए अपरिवर्तनीय है, केवल सापेक्ष अंतर मायने रखता है।

## इसका प्रयोग करें

PyTorch 2.5+ जहाजों RoPE उपयोगिताओं में `torch.nn.functional`. अधिकांश उत्पादन कोड उपयोग करता है `flash_attn`या `xformers`जहां RoPE ध्यान कर्नेल के अंदर लगाया जाता है।

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Long-context tricks in 2026:**

- **NTK-aware interpolation.**पुनरावृत्ति`base``base * (scale_factor)^(d/(d-2))`जब 4K से 16K+ तक विस्तारित किया जाता है।
- **YaRN.**अधिक स्मार्ट इंटरपोलेशन जो लंबे संदर्भों पर ध्यान एंट्रोपी को संरक्षित करता है। Llama 3.1 128K इसका उपयोग करता है।
- **LongRoPE.**माइक्रोसॉफ्ट की 2024 विधि जो विकासवादी खोज का उपयोग प्रति आयाम पैमाने के कारकों को चुनने के लिए करती है।
- **Position interpolation + fine-tuning.**बस विस्तार कारक से पदों को छोटा करें और 15B टोकन के लिए ठीक-ठीक समायोजित करें. आश्चर्यजनक रूप से प्रभावी.

## इसे भेजें

देखो`outputs/skill-positional-encoding-picker.md`. कौशल लक्ष्य संदर्भ लंबाई, अतिरिक्त आवश्यकताओं और प्रशिक्षण बजट को देखते हुए एक नए मॉडल के लिए एक कोडिंग रणनीति चुनता है।

## व्यायाम

1. **Easy.**सिनोसोइडल को रेखांकित करें `PE` के लिए एक हीटमैप के रूप में मैट्रिक्स`max_len=512, d=128`. "आकार सूचकांक बढ़ने के साथ पट्टी चौड़ी हो जाती है" पैटर्न की पुष्टि करें.
2. **Medium.**एनटीके-जागरूक रोपी स्केलिंग लागू करें. लंबाई 256 के अनुक्रमों पर एक छोटी सी एलएम को प्रशिक्षित करें, फिर लंबाई 1024 पर स्केलिंग के साथ और बिना परीक्षण करें। भ्रम को मापें।
3. **Hard.**एक ही ध्यान मॉड्यूल में ALiBi और RoPE को लागू करें। लंबाई 512 के अनुक्रमों के साथ एक कॉपी कार्य पर 4-परत ट्रांसफार्मर को प्रशिक्षित करें। परीक्षण के समय 2048 तक एक्सट्रापोलेट करें। गिरावट की तुलना करें।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Positional encoding | "Tells attention about order" | Any signal added to embeddings or attention that encodes position. |
| Sinusoidal | "The original one" | `sin/cos` at geometric frequencies added to embeddings; doesn't extrapolate. |
| RoPE | "Rotary embeddings" | Rotate Q, K by position-dependent angle; dot product encodes relative distance. |
| ALiBi | "Linear bias trick" | Add `-m·\|i-j\|` to attention scores; no embedding needed, great extrapolation. |
| base | "RoPE's knob" | The frequency scaler in RoPE; increase to extend context at inference. |
| NTK-aware | "A RoPE scaling trick" | Rescale `base` so high-frequency dims aren't squeezed when context expands. |
| YaRN | "The fancy one" | Per-dimension interpolation+extrapolation that preserves attention entropy. |
| Extrapolation | "Works beyond trained length" | Can the position scheme serve correct output past `max_len` seen in training? |

## आगे पढ़ना

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) मूल सिनोसॉइडल।
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) RoPE कागज।
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) ALiBi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) अत्याधुनिक रोपी स्केलिंग।
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) मेटा के लामा 2 लंबे संदर्भ का पेपर।
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) माइक्रोसॉफ्ट द्वारा इस्तेमाल की गई विधि Phi-3-Long और उपयोग यह अनुभाग में उद्धृत किया गया है।
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) प्रत्येक RoPE स्केलिंग योजना के उत्पादन स्तर पर कार्यान्वयन (पूर्वनिर्धारित, रैखिक, गतिशील, YaRN, LongRoPE, Llama-3) ।
