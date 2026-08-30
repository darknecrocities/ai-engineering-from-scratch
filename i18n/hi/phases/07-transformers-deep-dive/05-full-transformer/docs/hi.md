# पूर्ण ट्रांसफार्मर  एन्कोडर + डिकोडर

> ध्यान तारा है. बाकी सब कुछ  अवशिष्ट, सामान्यीकरण, फ़ीड-फॉरवर्ड, क्रॉस-अटेंशन  है जो आपको इसे गहराई से ढेर करने देता है।

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## समस्या

एक एकल ध्यान परत एक विशेषता निकालने वाला है, न कि एक मॉडल। प्रति परत एक मटमूल भाषा के लिए पर्याप्त क्षमता नहीं है। आपको सही नल के बिना गहराई  और गहराई ब्रेक की आवश्यकता है।

2017 के वास्वनी पेपर में छह डिजाइन निर्णय पैक किए गए थे जिन्होंने एक ध्यान परत को एक स्टैक करने योग्य ब्लॉक में बदल दिया था।  एन्कोडर-केवल (BERT), डिकोडर-केवल (GPT), एन्कोडर-डिकोडर (T5)  से प्रत्येक ट्रांसफार्मर एक ही कंकाल का उत्तराधिकारी है। 2026 में ब्लॉक को परिष्कृत किया गया है (RMSNorm, SwiGLU, प्री-नॉर्म, RoPE) लेकिन कंकाल समान है।

यह पाठ कंकाल है। अगले पाठों में इसे  06 कोडर के लिए, 07 कोडर के लिए, 08 कोडर-डेकोडर के लिए विशेषज्ञता दी गई है।

## अवधारणा

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### छह टुकड़े

1. **Embedding + positional signal.**टोकन → वेक्टर। RoPE (आधुनिक) या sinusidal (क्लासिक) के माध्यम से इंजेक्ट की गई स्थिति।
2. **Self-attention.**प्रत्येक स्थिति एक दूसरे की देखभाल करती है.
3. **Feed-forward network (FFN).**स्थिति के अनुसार दो-परत MLP: `W_2 · activation(W_1 · x)`. विस्तार अनुपात 4 × डिफ़ॉल्ट रूप से।
4. **Residual connection.** `x + sublayer(x)`इसके बिना, ग्रेडिएंट लगभग 6 परतों के बाद गायब हो जाते हैं।
5. **Layer normalization.** `LayerNorm`या `RMSNorm`(आधुनिक) शेष प्रवाह स्थिर करता है।
6. **Cross-attention (decoder only).**क्वेरीएं डिकोडर से आती हैं, कुंजी और कोडर आउटपुट से मान।

एक ब्लॉक के माध्यम से एक वेक्टर प्रवाह को देखेंः ध्यान स्थिति में मिश्रित होता है, शेष इसे आगे ले जाता है, FFN इसे बदल देता है, और सामान्य प्रवाह को स्थिर रखता है।

```figure
transformer-block
```

### एन्कोडर ब्लॉक (BERT, T5 एन्कोडर द्वारा उपयोग किया जाता है)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

एन्कोडर दो दिशा है, कोई मास्किंग नहीं है, सभी स्थिति सभी स्थिति को देखते हैं।

### डिकोडर ब्लॉक (जीपीटी, टी5 डिकोडर द्वारा उपयोग किया जाता है)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

डेकोडर में प्रति ब्लॉक तीन उप-परत होते हैं। मध्य में  क्रॉस-अटेंशन  एकमात्र स्थान है जहां जानकारी को एन्कोडर से डेकोडर तक प्रवाह होता है। शुद्ध डेकोडर-केवल वास्तुकला (जीपीटी) में, क्रॉस-अटेंशन को छोड़ दिया जाता है और आपके पास बस मास्क स्व-अटेंशन + एफएफएन है।

### पूर्व-नियमित बनाम बाद के मानक

मूल कागज: `x + sublayer(LN(x))`vs `LN(x + sublayer(x))`. 2019 के आसपास पोस्ट-नॉर्म ने अपना पक्ष खो दिया  सावधानीपूर्वक वार्मिंग के बिना गहरे प्रशिक्षण करना कठिन है।`LN`*पूर्व* उपपरत) 2026 डिफ़ॉल्ट हैः Llama, Qwen, GPT-3+, Mistral सभी इसका उपयोग करते हैं।

### 2026 में आधुनिक ब्लॉक

Vaswani 2017 लेयरनॉर्म + ReLU शिप किया गया। आधुनिक स्टैक दोनों को बदल दिया। उत्पादन ब्लॉक वास्तव में कैसा दिखते हैंः

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices, total params match) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

RMSNorm लेयरनॉर्म (एक कम घटाने) के औसत-केंद्रण को कम करता है, जो गणना को बचाता है और अनुभवजन्य रूप से कम से कम स्थिर है।`Swish(W1 x) ⊙ W3 x`) लगातार एलएएमए, पॉलम और क्यूवेन पेपर में रेलू/गेलू एफएफएन से लगभग 0.5 अंक अधिक प्रदर्शन करता है।

### पैरामीटर गिनती

एक ब्लॉक के लिए `d_model = d`और एफएफएन विस्तार `r`:

- एमएचए: `4 · d²`(Q, K, V, O प्रक्षेपण)
- FFN (SwiGLU): `3 · d · (r · d)`≈ ≈`3rd²`
- मानदंडः नाटकीय

`d = 4096, r = 2.6, layers = 32`(लगभग Llama 3 8B), कुलः `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`(अधिक एम्बेड और सिर) मैच प्रकाशित गिनती।

## इसे बनाओ

### चरण 1: निर्माण की सामग्री

छोटे का उपयोग करके `Matrix`कक्षा 03 से (स्वतंत्रता के लिए इस फ़ाइल में कॉपी किया गया):

- `layer_norm(x, eps=1e-5)` औसत घटाएँ, std से विभाजित करें।
- `rms_norm(x, eps=1e-6)` RMS द्वारा विभाजित करें। कोई औसत घटाव नहीं।
- `gelu(x)`और `silu(x) * W3 x`(स्विगलू) ।
- `ffn_swiglu(x, W1, W2, W3)`. .
- `encoder_block(x, params)`और `decoder_block(x, enc_out, params)`. .

देखो`code/main.py`पूरी तारों के लिए.

### चरण 2: एक 2-परत एन्कोडर और एक 2-परत डेकोडर तार

उन्हें ढेर करें. प्रत्येक डिकोडर क्रॉस-अटेंशन में एन्कोडर आउटपुट पारित करें. आउटपुट प्रोजेक्शन से पहले एक अंतिम LN जोड़ें.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### चरण 3: एक खिलौना उदाहरण पर आगे चलें

एक 6-टोकन स्रोत और 5-टोकन लक्ष्य के माध्यम से फ़ीड करें. आउटपुट आकार सत्यापित करें है `(5, vocab)`कोई प्रशिक्षण नहीं है यह सबक वास्तुकला के बारे में है, नुकसान के बारे में नहीं है।

### चरण 4: RMSNorm + SwiGLU में स्वैप करें

LayerNorm और ReLU-FFN को RMSNorm और SwiGLU के साथ बदलें। आकारों की पुष्टि अभी भी मेल खाती है। यह एक फ़ंक्शन प्रतिस्थापन के साथ 2026 आधुनिकीकरण है।

## इसका प्रयोग करें

PyTorch/TF संदर्भ कार्यान्वयनः `nn.TransformerEncoderLayer`,`nn.TransformerDecoderLayer`लेकिन 2026 उत्पादन कोड के अधिकांश अपने स्वयं के ब्लॉक रोल क्योंकिः

- फ्लैश ध्यान ध्यान के अंदर बुलाया जाता है, ध्यान के माध्यम से नहीं `nn.MultiheadAttention`. .
- जीसीए/एमएलए स्टडीलिब संदर्भ में नहीं हैं।
- RoPE, RMSNorm, SwiGLU PyTorch डिफ़ॉल्ट नहीं हैं।

HF `transformers`इसमें स्पष्ट संदर्भ ब्लॉक हैं जिन्हें आपको पढ़ना चाहिएः `modeling_llama.py`यह लगभग 500 पंक्तियों है और एक बार के माध्यम से चलने लायक है।

**Encoder vs decoder vs encoder-decoder — when to pick:**

| Need | Pick | Example |
|------|------|---------|
| Classification, embeddings, QA over text | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

केवल डिकोडर भाषा को जीता क्योंकि यह सबसे साफ पैमाने पर है और समझ और पीढ़ी दोनों को संभालता है। जब इनपुट में स्पष्ट "स्रोत अनुक्रम" पहचान (अनुवाद, भाषण पहचान, संरचित कार्य) होती है तो भी एन्कोडर-डिकोडर सबसे अच्छा होता है।

## इसे भेजें

देखो`outputs/skill-transformer-block-reviewer.md`. कौशल 2026 डिफ़ॉल्ट के खिलाफ एक नए ट्रांसफार्मर ब्लॉक कार्यान्वयन की समीक्षा करता है और गायब टुकड़ों (पूर्व-मानक, RoPE, RMSNorm, GQA, FFN विस्तार अनुपात) को चिह्नित करता है।

## व्यायाम

1. **Easy.**अपने encoder_block में पैरामीटर गिनें `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. ब्लॉक को लागू करके और उपयोग करके सत्यापित करें `sum(p.numel() for p in block.parameters())`. .
2. **Medium.**पोस्ट-नॉर्म से प्री-नॉर्म पर स्विच करें। दोनों को प्रारंभ करें और यादृच्छिक इनपुट पर 12 स्टैक किए गए परतों के बाद सक्रियण मानक को मापें। पोस्ट-नॉर्म के सक्रियण को विस्फोट करना चाहिए; प्री-नॉर्म को सीमित रहना चाहिए।
3. **Hard.**एक खिलौना कॉपी कार्य पर एक 4-परत एन्कोडर-डेकोडर लागू करें (कॉपी `x`100 कदम ट्रेन करें। हानि रिपोर्ट करें। RMSNorm + SwiGLU + RoPE में स्वैप करें  क्या हानि घटती है?

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Block | "One transformer layer" | Stack of norm + attention + norm + FFN, wrapped in residual connections. |
| Residual | "Skip connection" | `x + f(x)` output; enables gradient flow through deep stacks. |
| Pre-norm | "Normalize before, not after" | Modern: `x + sublayer(LN(x))`. Trains deeper without warmup gymnastics. |
| RMSNorm | "LayerNorm without the mean" | Divide by RMS; one less op, same empirical stability. |
| SwiGLU | "The FFN everyone switched to" | `Swish(W1 x) ⊙ W3 x → W2`. Beats ReLU/GELU on LM ppl. |
| Cross-attention | "How the decoder sees the encoder" | MHA with Q from decoder, K/V from encoder outputs. |
| FFN expansion | "How wide the middle MLP is" | Ratio of hidden-size to d_model, usually 4 (LayerNorm) or 2.6 (SwiGLU). |
| Bias-free | "Drop the +b terms" | Modern stacks omit biases in linear layers; slight ppl improvement, smaller model. |

## आगे पढ़ना

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) मूल ब्लॉक स्पेसिफिकेशन।
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) क्यों पूर्व-नियमित पूर्व-नियमित से गहराई से आगे निकलता है।
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) आरएमएसनॉर्म।
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) SwiGLU पेपर।
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) कैनोनिक 2026 केवल डिकोडर ब्लॉक।
