# क्रमिक चेकपॉइंटिंग और सक्रियण पुनः गणना

> Backprop हर मध्यवर्ती सक्रियण को रखता है। 70B पैरामीटर और 128K संदर्भ पर जो प्रति रैंक 3 TB सक्रियण है। चेकपॉइंटिंग मेमोरी के लिए FLOPs का व्यापार करता हैः सहेजने के बजाय पुनः गणना करें। सवाल यह है कि कौन से खंड छोड़ना है, और उत्तर "वे सभी" नहीं है।

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## समस्या

एक ट्रांसफार्मर को प्रशिक्षित करना प्रत्येक परत के लिए, प्रत्येक ऑपरेशन के इनपुट को संग्रहीत करता है जो पीछे की ओर भिन्न होता हैः ध्यान इनपुट, क्यू / के / वी प्रोजेक्शन, सॉफ्टमैक्स आउटपुट, एफएफएन इनपुट, मानक आउटपुट और अवशिष्ट धारा। छिपे हुए आकार के साथ परत के लिए `d`, अनुक्रम की लंबाई `L`, बैच `B`, यह आदेश पर है `12 * B * L * d`प्रति परत तैरता है।

के लिए`d=8192, L=8192, B=1`, जो कि BF16 में 800 MB / परत है। एक 64-स्तर मॉडल सक्रियण के 51 GB है  और यह है कि आप माइक्रोबैच आकार से गुणा करने से पहले, आप ध्यान-softmax मध्यवर्ती जोड़ने से पहले (`L^2`प्रति सिर), और इससे पहले कि आप टेन्सर-समान आंशिक प्रतियां कारक.

दो पक्षीय बिलः BF16 वजन प्लस ऑप्टिमाइज़र स्टेट 80GB में फिट हो सकता है, लेकिन सक्रियण आपको आगे बढ़ाता है। ग्रेडिएंट चेकपॉइंटिंग (उर्फ सक्रियण पुनः गणना) मानक फिक्स है। अधिकांश सक्रियण को छोड़ दें; उन्हें वापस पाने के लिए आगे के दौरान फिर से करें। लागतः अतिरिक्त FLOPs। लाभः मेमोरी में कमी चेकपॉइंट सेगमेंट्स के अनुपात से कुल परतों तक।

कुख्यात तरीके से किया गया, चेकपॉइंटिंग की लागत प्रति चरण लगभग 33% अधिक है। अच्छी तरह से किया गया  कोर्थिकान्टी और अन्य के "स्मार्ट चयन" के अनुसार चुनिंदा चेकपॉइंटिंग  आप 5% से कम FLOP ओवरहेड के लिए 5x मेमोरी बचाते हैं। और FP8 matmuls, FSDP offload, और विशेषज्ञ समानांतर MoE के साथ यह वास्तव में मायने रखता हैः आप या तो मेमोरी या बर्बाद किए गए कंप्यूटिंग का खर्च नहीं उठा सकते।

## अवधारणा

### पिछड़े लोगों की क्या ज़रूरत है

`output = layer(input)`. . पीछे की इच्छाएँ .`grad_input`और `grad_params`. उन्हें गणना करने के लिए यह आवश्यक हैः

- `input`(गणना करने के लिए `grad_params = input.T @ grad_output`रैखिक परतों के लिए)
- कुछ सक्रियण व्युत्पन्न मध्यवर्ती (ReLU/GELU/softmax का व्युत्पन्न सक्रियण मूल्य पर निर्भर करता है)

आगे पास स्वचालित रूप से ऑटोग्रेड ग्राफ में इन संग्रहीत करता है।`tensor.retain_grad()`और प्रत्येक ऑपरेशन को जो अपने इनपुट की आवश्यकता है एक संदर्भ बनाए रखता है।

### पूरी तरह से चेकपोइंटिंग

नेटवर्क को विभाजित करें `N`अग्रिम के दौरान, प्रत्येक खंड के लिए केवल * इनपुट * को संग्रहीत करें। जब पिछड़े को मध्यवर्ती की आवश्यकता होती है, तो उन्हें भौतिक बनाने के लिए खंड के अग्रिम पास को फिर से चलाएं, फिर अंतर करें।

उदाहरण: 32-परत ट्रांसफार्मर प्रत्येक 1 परत के 32 खंडों में विभाजित है।

- मेमोरीः 32 परत-इनपुट (छोटा) बनाम 32 * (प्रति परत सक्रियण मात्रा) (बड़ी) ।
- अतिरिक्त गणनाः प्रति खंड 1 अतिरिक्त आगे, यानी, ~33% आगे FLOP कुल (क्योंकि पीछे 2x आगे है, पूर्ण चरण 1 + 1 + 2 = 4 इकाइयों के बजाय 1 + 2 = 3 हो जाता है) ।

यह मूल Chen et al. 2016 नुस्खा हैः प्रत्येक एक चेक पॉइंट `sqrt(L)`L = 64 के लिए, यह 8 चेकपॉइंट है.

### चुनिंदा चेकपॉइंटिंग (कोर्तिकान्ती 2022)

सभी सक्रियणों की लागत समान नहीं है। ध्यान softmax आउटपुट है`B*L*L*heads`और क्रम लंबाई के साथ *क्वाड्रैटिकली* बढ़ता है। FFN छिपे सक्रियण है `B*L*4d`लंबे अनुक्रमों के लिए softmax हावी है।

चयनित चेकपोइंटिंग सस्ते-से-स्टोर सक्रियण (रेखीय अनुमान, अवशेष) को बनाए रखता है और केवल महंगे (ध्यान) को पुनः गणना करता है। आप पुनः गणना करने के लिए न्यूनतम FLOPs का भुगतान करते हैं लेकिन O(L^2) मेमोरी को सहेजते हैं।

मेगाट्रॉन-कोर इसे "निर्णय" सक्रियण पुनः गणना के रूप में लागू करता है। अधिकांश 2024+ सीमा प्रशिक्षण रन में उपयोग किया जाता है।

### अपलोड

पुनर्गणना के विकल्पः आगे और पीछे के बीच सीपीयू रैम को सक्रिय करने के लिए जहाज। पीसीआईई बैंडविड्थ की आवश्यकता होती है; लाभदायक जब निष्क्रिय बैंडविड्थ पुनर्जन्म की लागत से अधिक होती है। मिश्रित रणनीतियां आम हैंः कुछ परतों को चेकपॉइंट करें, दूसरों को उतार दें।

FSDP2 एक प्रथम श्रेणी विकल्प के रूप में अपलोड जहाजों. अपलोड चमकता है जब GPU स्मृति पर बोतल गला है लेकिन CPU-GPU हस्तांतरण हैडरूम है।

### लागत मॉडल को पुनः गणना करें

प्रत्येक चरण के साथ भोले चेकपोइंटिंग के साथ फ्लोप`k` से बाहर लेयर`L`:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

चयनात्मक चेकपोइंटिंग के साथ आप केवल ध्यान कर्नेल को पुनः गणना करते हैं, पूरी परत नहींः

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### स्मृति बचत मॉडल

प्रति परत सक्रियण मात्राः `A`. . के लिए .`L`परतें, कुल सक्रियण स्मृतिः `L * A`. .

पूर्ण चेकपॉइंट (सेगमेंट आकार 1): केवल स्टोर `L * input_volume`(~`L * 1/10 A`एक मानक ट्रांसफार्मर के लिए) बचत ~`9 * L * A * 1/10`. .

हर बार चेकपॉइंट`k`परतेंः भंडारण `L/k * A`और `k-1`सक्रिय खंड के भीतर परतों का मूल्य।

`k = sqrt(L)`, स्मृति और पुनः गणना लागत दोनों पैमाने के साथ `sqrt(L)` समान लागत परतों के लिए इष्टतम व्यापार समझौता।

### जब चेकपॉइंट नहीं जाना है

- पाइपलाइन के अंदर के परतों को पहले से ही उड़ान में है. वे वैसे भी खत्म करना होगा.
- यदि वे स्टेज की गणना पर हावी हैं तो पहली और अंतिम परतें (ट्रांसफार्मर में दुर्लभ) ।
- ध्यान कर्नेल पहले से ही FlashAttention  फ्लैश पहले से ही softmax तेजी से पुनः गणना करता है, इसलिए अतिरिक्त परत स्तर की जांच ऊपर थोड़ा जोड़ता है।

### कार्यान्वयन के पैटर्न

1. **Function wrapper:**एक खंड को `torch.utils.checkpoint.checkpoint(fn, input)`. केवल पाइटॉर्च स्टोर`input`, पीछे की ओर सब कुछ फिर से गणना करता है।

2. **Decorator-based:**लेबल परतों को चेकपोइंट के रूप में लेबल किया जा सकता है; प्रशिक्षक कॉन्फ़िगरेशन समय पर तय करता है कि कौन से खंड लपेटे जाएंगे।

3. **Manual explicit recompute:**अपने आप को पीछे की तरफ पास लिखें, एक आदत कहते हैं `recompute_forward`जो भंडारित इनपुट के साथ फॉरवर्ड को दोहराता है।

तीनों समान कार्यकारी परिणाम देते हैं।

### टीपी / पीपी / एफपी8 के साथ बातचीत

- **Tensor parallel:**चेकपॉइंट इनपुट को पुनः गणना पर एकत्र या फिर से वितरित किया जाना चाहिए; संचार लागत को संभालना।
- **Pipeline parallel:**एक विशिष्ट पैटर्न प्रत्येक पाइपलाइन चरण के आगे की जांच करना है ताकि रिवर्स ऑर्डर माइक्रोबैच सक्रियण स्मृति का पुनः उपयोग कर सकते हैं।
- **FP8 recompute:**पुनर्गणना के दौरान अद्यतन amax इतिहास मूल अग्रिम या FP8 पैमाने के परिचालन के अनुरूप होना चाहिए। अधिकांश फ्रेमवर्क पैमाने की तस्वीर लेते हैं।

```figure
activation-recompute
```

## इसे बनाओ

### चरण 1: टुकड़ों के साथ खिलौना मॉडल

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### चरण 2: सभी सक्रियण की आवश्यकता

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### चरण 3: चेकपॉइंट-हर-क मेमोरी

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### चरण 4: लागत मॉडल

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### चरण 5: स्मृति अनुमानक

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### चरण 6: अनुभाग का इष्टतम आकार

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### चरण 7: चयनात्मक चेकपॉइंट निर्णय

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## इसका प्रयोग करें

- **torch.utils.checkpoint**`from torch.utils.checkpoint import checkpoint` PyTorch में कैनोनिकल रैपर। एक फ़ंक्शन को रैप करता है; केवल इनपुट को संग्रहीत करता है, पीछे की ओर पुनः गणना करता है।
- **Megatron-Core activation recomputation**: समर्थन `selective`,`full`और `block`2024+ सीमा प्रशिक्षण में मानक।
- **FSDP2 offload**`module.to_empty(device="cpu")`के साथ`offload_policy`FSDP2 में पुनः गणना के बजाय सीपीयू में सक्रियण को स्कार्ड करता है।
- **DeepSpeed ZeRO-Offload**: ऑप्टिमाइज़र राज्यों और सक्रियण के लिए सीपीयू का अपलोड, चेकपोइंटिंग को पूरक।

## इसे भेजें

यह सबक हमें फल देता है`outputs/prompt-activation-recompute-policy.md` एक प्रॉम्प्ट जो आपके मॉडल कॉन्फ़िग (लेयर्स, छिपे हुए, सीक्यू, बैच) और उपलब्ध जीपीयू मेमोरी को लेता है और प्रति परत रीकंप्यूटिंग नीति (नहीं / चयन / पूर्ण / ऑफलोड) जारी करता है।

## व्यायाम

1. सही होने की पुष्टि करें।`model_forward`+ `model_backward`(पूर्ण सक्रियण) vs `model_forward_checkpointed`+ `model_backward_checkpointed`पैरामीटर ग्रेडिएंट मशीन सटीकता के समान होना चाहिए।

2. पोंछ खंड आकार `k`1 से `L`. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

3. चयनात्मक चेकपोइंटिंग लागू करेंः ध्यान मॉड्यूल इनपुट को स्टोर करें लेकिन इसके मध्यवर्ती नहीं। 32 परत मॉडल के लिए FLOP ओवरहेड बनाम पूर्ण-परत चेकपोइंटिंग को seq=8192 पर मापें।

4. अपलोड जोड़ें. अनुकरण "सीपीयू बफर" (अलग सूची) में खंड इनपुट सहेजें। "पीसीआई बैंडविड्थ" को बाइट्स/समय के रूप में मापें और अपलोड और पुनः गणना के बीच ब्रेक-ईक्वेंस बिंदु ढूंढें।

5. एक वास्तविक PyTorch ट्रांसफार्मर के साथ और बिना बेंचमार्क `torch.utils.checkpoint`. स्मृति का मापन (via `torch.cuda.max_memory_allocated`) और कदम समय।

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Gradient checkpointing | "Save memory by redoing forward" | Store segment inputs only; recompute intermediates during backward to get gradient-support tensors |
| Activation recomputation | "Same as checkpointing" | The HPC-flavored name for the same technique |
| Segment size (k) | "How many layers per checkpoint" | Number of layers whose intermediates are dropped and rematerialized together |
| Selective checkpointing | "Korthikanti's trick" | Recompute only expensive-to-store activations (attention softmax); keep cheap ones |
| Full checkpointing | "The naive version" | Recompute every layer's intermediates in every segment |
| Block checkpointing | "Coarse-grained" | Checkpoint whole transformer blocks; largest granularity |
| FLOP overhead | "The compute tax" | Extra FLOPs per step = (recompute FLOPs) / (fwd + bwd FLOPs); 33% naive, 5% selective |
| Activation offload | "Ship to CPU" | Move activations to CPU RAM across forward->backward; alternative to recompute |
| sqrt-L rule | "The classical optimum" | For uniform-cost layers, optimal checkpoint spacing is sqrt(L) layers |
| Attention-softmax volume | "The O(L^2) problem" | L^2 * heads * batch floats; dominates activation memory at long contexts |

## आगे पढ़ना

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174)-- मूल कागज जो ग्रेडिएंट चेकपोइंटिंग को औपचारिक रूप देता है
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198)-- चुनिंदा सक्रियण पुनः गणना और औपचारिक लागत विश्लेषण
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645)-- रिवर्स मोड रीमैटेरियलाइज़ेशन के माध्यम से निरंतर स्मृति के वैकल्पिक दृष्टिकोण
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840)-- सक्रियण उतार-चढ़ाव
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html)-- मानक एपीआई
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html)-- चयनात्मक, पूर्ण और ब्लॉक मोड
