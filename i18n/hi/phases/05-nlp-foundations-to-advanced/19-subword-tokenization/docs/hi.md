# उपशब्द टोकनाइज़ेशन  BPE, WordPiece, Unigram, SentencePiece

> शब्द टोकन बनाने वाले अदृश्य शब्दों पर थूक जाते हैं। वर्ण टोकन करने वाले अनुक्रम की लंबाई को बढ़ा देते हैं। उपशब्द टोकन करने वाले अंतर को विभाजित करते हैं। हर आधुनिक LLM एक पर जहाज करता है।

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## समस्या

आपके शब्दावली में 50,000 शब्द हैं। एक उपयोगकर्ता टाइप करता है "अनटोकनेज"। आपका टोकनेज़र वापस आता है।`[UNK]`. मॉडल में अब शब्द के बारे में कोई संकेत नहीं है. और इससे भी बदतर: आपके कॉर्पस में 90वें प्रतिशत दस्तावेज़ में 40 दुर्लभ शब्द हैं, जिसका अर्थ है कि प्रति दस्तावेज़ 40 बिट्स की जानकारी गिर गई है.

उपशब्द टोकनकरण इस समस्या का समाधान करता है. सामान्य शब्द एकल टोकन बने रहते हैं. दुर्लभ शब्द सार्थक टुकड़ों में विघटित होते हैंः`untokenizable`→ `un`,`token`,`izable`प्रशिक्षण डेटा सब कुछ शामिल है क्योंकि किसी भी स्ट्रिंग अंततः बाइट्स का एक अनुक्रम है।

2026 में हर फ्रंटियर एलएलएम तीन एल्गोरिदम (बीपीई, यूनिग्राम, वर्डपीस) में से एक पर जहाज करता है, तीन पुस्तकालयों (टिकटोकन, सेंटेन्सपीस, एचएफ टोकनइज़र) में से एक में लपेटा जाता है। आप एक चुनने के बिना भाषा मॉडल नहीं भेज सकते।

## अवधारणा

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).**वर्ण स्तर की शब्दावली से शुरू करें. प्रत्येक आसन्न जोड़ी की गणना करें. सबसे अधिक बार होने वाली जोड़ी को नए टोकन में मिलाएं. जब तक आप लक्ष्य शब्दावली आकार को नहीं प्राप्त करते तब तक दोहराएं। प्रभुत्व वाला एल्गोरिथ्मः GPT-2/3/4, Llama, Gemma, Qwen2, Mistral।

**Byte-level BPE.**वही एल्गोरिथ्म लेकिन यूनिकोड वर्णों के बजाय कच्चे बाइट्स (256 बेस टोकन) पर।`[UNK]`टोकन  किसी भी बाइट अनुक्रम कोड। GPT-2 50,257 टोकन (256 बाइट + 50,000 विलय + 1 विशेष) का उपयोग करता है।

**Unigram.**एक विशाल शब्दावली के साथ शुरू करें। प्रत्येक टोकन को एक एकल-सूची की संभावना असाइन करें। प्रतिवारात्मक रूप से टोकन काटें जिनके हटाने से कॉर्पस लॉग-संभावना कम से कम बढ़ जाती है। निष्कर्ष पर संभावनाः नमूना टोकन (उपशब्द नियमितकरण के माध्यम से डेटा बढ़ाने के लिए उपयोगी) । T5, mBART, ALBERT, XLNet, Gemma द्वारा उपयोग किया जाता है।

**WordPiece.**मिश्रण जोड़े जो कच्चे आवृत्ति की बजाय प्रशिक्षण corpus की संभावना को अधिकतम करते हैं।

**SentencePiece vs tiktoken.**SentencePiece वह पुस्तकालय है जो *ट्रेन* शब्दावली (बीपीई या यूनोग्राम) को सीधे कच्चे यूनिकोड पाठ पर, श्वेत क्षेत्र को कोडिंग के रूप में `▁`. tiktoken पूर्व निर्मित शब्दावली के खिलाफ OpenAI का तेज *encoder* है; यह प्रशिक्षण नहीं देता है।

अंगूठे का नियमः

- **Training a new vocabulary:**SentencePiece (बहुभाषी, कोई पूर्व-टोकनाइज़ेशन नहीं) या HF Tokenizers।
- **Fast inference against GPT vocab:**tiktoken (cl100k_base, o200k_base) ।
- **Both:**एचएफ टोकनेजर्स  एक पुस्तकालय, प्रशिक्षण + सेवा।

```figure
bpe-merge
```

## इसे बनाओ

### चरण 1: खरोंच से बीपीई

देखो`code/main.py`. . लूपः

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

एल्गोरिथ्म कोड तीन तथ्य।`</w>`शब्द अंत के निशान इसलिए "कम" (सफल) और "कम" (पूर्व) अलग रहते हैं। आवृत्ति भार उच्च आवृत्ति जोड़े जल्दी जीतता है। विलय सूची क्रमबद्ध है  निष्कर्ष प्रशिक्षण क्रम में विलय लागू होता है।

### चरण 2: सीखे गए विलय के साथ कोड करें

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

उत्पादन कार्यान्वयन (टिंकटोकन, एचएफ टोकन) प्राथमिकता कतारों के साथ मर्ज-रैंक खोज का उपयोग करते हैं और लगभग रैखिक समय में चलते हैं।

### चरण 3: वाक्यप्रक्रिया

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

नोटः कोई पूर्व-टोकनाइज़ेशन की आवश्यकता नहीं, स्थान को कोडित किया गया है `▁`,`character_coverage`नियंत्रण करता है कि आक्रामक रूप से दुर्लभ वर्णों को संरक्षित किया जाता है बनाम मानचित्रित किया जाता है `<unk>`. .

### चरण 4: OpenAI संगत शब्दावली के लिए टिक टोकन

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

केवल एन्कोडिंग। तेजी से (रस्ट बैकेंड) बाइट-कंटेंट, लागत अनुमान, संदर्भ-विंडो बजटिंग के लिए जीपीटी-4/5 टोकनकरण के साथ सटीक मेल खाता है।

## 2026 में भी फंसे हुए जाल

- **Tokenizer drift.**वक्कल ए पर प्रशिक्षण, वक्कल बी के खिलाफ तैनाती। टोकन आईडी अलग हैं; मॉडल आउटपुट कचरा है। चेक `tokenizer.json`आईसी में हैश।
- **Whitespace ambiguity.**BPE "hello" बनाम "hello" अलग टोकन उत्पन्न करते हैं। हमेशा निर्दिष्ट करें `add_special_tokens`और `add_prefix_space`स्पष्ट रूप से।
- **Multilingual undertraining.**अंग्रेजी-भारी कॉर्पोरेस शब्दावली का उत्पादन करते हैं जो गैर-लैटिन लिपि को 5-10 गुना अधिक टोकन में विभाजित करते हैं। उसी प्रॉम्प्ट की लागत 5-10 गुना अधिक है जापानी / अरबी पर GPT-3.5। o200k_base ने आंशिक रूप से इसे ठीक किया।
- **Emoji splits.**एक एकल इमोजी 5 टोकन ले सकता है। संदर्भ बजट करते समय चेकपॉइंट इमोजी संभाल।

## इसका प्रयोग करें

2026 स्टैकः

| Situation | Pick |
|-----------|------|
| Training a monolingual model from scratch | HF Tokenizers (BPE) |
| Training a multilingual model | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Serving an OpenAI-compatible API | tiktoken (`o200k_base` for GPT-4+) |
| Domain-specific vocab (code, math, protein) | Train custom BPE on domain corpus, merge with base vocab |
| Edge inference, small model | Unigram (smaller vocabularies work better) |

शब्दकोश आकार एक स्केलिंग निर्णय है, एक स्थिर नहीं। कड़ा हेरिस्टिकः <1B पैरामीटर के लिए 32k, 1-10B के लिए 50-100k, बहुभाषी / सीमा के लिए 200k+।

## इसे भेजें

`outputs/skill-bpe-vs-wordpiece.md`:

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## व्यायाम

1. **Easy.**500-मिलन BPE को चालू करें`code/main.py`तीन लंबे समय तक चलने वाले शब्दों को एन्कोड करें. कितने ने 1 टोकन बनाम > 1 टोकन उत्पन्न किया?
2. **Medium.**100 अंग्रेजी विकिपीडिया वाक्यों पर टोकन गिनती की तुलना करें `cl100k_base`,`o200k_base`, और एक SentencePiece BPE आप वोकैब के साथ प्रशिक्षित = 32k. प्रत्येक के संपीड़न अनुपात की रिपोर्ट.
3. **Hard.**एक ही कॉर्पस को बीपीई, यूनिग्राम और वर्डपीस के साथ प्रशिक्षित करें। एक छोटे से संवेदना वर्गीकरणकर्ता पर प्रत्येक का उपयोग करते समय डाउनस्ट्रीम सटीकता मापें। क्या विकल्प ने सुई को 1 बिंदु से अधिक F1 से आगे बढ़ाया है?

## प्रमुख शर्तें

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | Greedy merge of most-frequent character pairs until target vocab size hit. |
| Byte-level BPE | No unknown tokens ever | BPE over raw 256 bytes; GPT-2 / Llama use this. |
| Unigram | Probabilistic tokenizer | Prunes from a large candidate set using log-likelihood; used by T5, Gemma. |
| SentencePiece | The whitespace one | Library that trains BPE/Unigram on raw text; space encoded as `▁`. |
| tiktoken | The fast one | OpenAI's Rust-backed BPE encoder for pre-built vocabs. No training. |
| Merge list | The magic numbers | Ordered list of `(a, b) → ab` merges; inference applies in order. |
| Character coverage | How rare is too rare? | Fraction of characters in training corpus the tokenizer must cover; ~0.9995 typical. |

## आगे पढ़ना

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) बीपीई पेपर।
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) यूनिग्राम पेपर।
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) पुस्तकालय।
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) संक्षिप्त संदर्भ।
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) रसोई पुस्तक + कोडिंग सूची।
