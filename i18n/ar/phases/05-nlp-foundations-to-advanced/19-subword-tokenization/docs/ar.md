# رمزية الكلمات الفرعية  BPE، WordPiece، Unigram، SentencePiece

> الـ "توكينيزر" الكلمات يختنقون بالكلمات غير المرئية، الـ "توكينيزر" الشخصيات ينفجر طول التسلسل، الـ "توكينيزر" الكلمات الفرعية، كل درجة ماجستير في العلوم الحديثة تُركب على واحدة.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## المشكلة

قاموسك يحتوي على 50 ألف كلمة، يكتب المستخدم "غير قابل للتوجه"`[UNK]`النموذج الآن لا يحتوي على إشارة حول الكلمة. والأسوأ: الوثيقة التي تبلغ 90 في المائة في جسمك تحتوي على 40 كلمة نادرة، مما يعني 40 جزء من المعلومات التي تم إسقاطها لكل وثيقة.

تُحلّ رمزية الكلمات الفرعية هذا. الكلمات الشائعة تبقى رموزًا واحدةً. الكلمات النادرة تتفكّك إلى قطع ذات معنى: `untokenizable``un`،`token`،`izable`بيانات التدريب تغطي كل شيء لأن أي سلسلة هي في نهاية المطاف تسلسل من البايتز.

كل ماجستير في مجال الحدود في عام 2026 يتم شحنها على واحد من ثلاثة خوارزميات (BPE، Unigram، WordPiece) ، مغلفة في واحدة من ثلاث مكتبات (tiktoken، SentencePiece، HF Tokenizers). لا يمكنك شحن نموذج لغة دون اختيار واحد.

## المفهوم

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).**ابدأ بمفردات على مستوى الأحرف، احصي كل زوج مجاور، دمج الزوج الأكثر تكرارًا في رمز جديد، كرر حتى تصل إلى حجم المفردات المستهدفة. الخوارزمية السائدة: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**Byte-level BPE.**نفس الخوارزمية ولكن أكثر من البايتات الخام (256 رمزاً أساسية) بدلاً من أحرف يونيكود. الضمانات صفر `[UNK]`الوهم  أي رموز تسلسل البايت. GPT-2 تستخدم 50,257 رمزا (256 بايت + 50,000 دمج + 1 خاص).

**Unigram.**تبدأ مع مجموعة كبيرة من المفردات. خصص لكل رمز احتمالية واحد. قم بتعديل رموز تزيد من احتمالات سجل الجسم على الأقل. من المحتمل في الاستنتاج: يمكن أن تكون نموذجية (فائدة لزيادة البيانات عن طريق تنظيم الكلمات الفرعية). تستخدم من قبل T5 ، mBART ، ALBERT ، XLNet ، Gemma.

**WordPiece.**مزيج أزواج التي تعزز احتمالات جسم التدريب بدلا من التردد الخام.

**SentencePiece vs tiktoken.**SentencePiece هي المكتبة التي * تدرب* المفردات (BPE أو Unigram) مباشرة على نص يونيكود الخام، وتشفير الفضاء الأبيض على شكل `▁`. tiktoken هو *مُشفّر * سريع OpenAI ضد المفردات المُبنية مسبقاً؛ فإنه لا يتدرب.

قاعدة عامة:

- **Training a new vocabulary:**SentencePiece (متعددة اللغات، لا توكنيزية مسبقة) أو HF Tokenizers.
- **Fast inference against GPT vocab:**تيكتون (cl100k_base، o200k_base).
- **Both:**HF Tokenizers  مكتبة واحدة، تدريب + خدمة.

```figure
bpe-merge
```

## بناءها

### الخطوة الأولى: BPE من الصفر

انظر`code/main.py`- الحلقة:

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

ثلاثة حقائق يرمزها الخوارزمية`</w>`علامات نهاية الكلمة بحيث "منخفض" (التضامن) و "أدنى" (التضامن) يبقى متميزة. تسبب الوزن المتردد في الفوز في أزواج التردد العالي مبكرا. يتم ترتيب قائمة الاندماج  تنطبق الاستنتاج الاندماج في ترتيب التدريب.

### الخطوة الثانية: ترميز مع المدمجات المتعلمة

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

البغاء O(n· نَعمَرَجَ (إِنْسَانَ) تنفيذات الإنتاج، Token HF Tokenizers يستخدمون التركيب الرتبة البحث مع طوابير الأولوية والعمل في وقت خطي تقريبا.

### الخطوة الثالثة: جملةقطعة في الممارسة العملية

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

ملاحظة: لا توكيين مسبق مطلوب، مساحة مرموزة`▁`،`character_coverage`يسيطر على كيفية الحفاظ على الشخصيات النادرة بشكل عنيف مقابل وضعها على الخريطة`<unk>`. . .

### الخطوة 4: تيكتون لمفردات متوافقة مع OpenAI

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

تشفير فقط. سريع (الخلفية الخرق). مطابقة تمامًا مع رمزية GPT-4/5 لعد البايتات، تقدير التكلفة، ميزانية النافذة السياقية.

## الفخاخ التي لا تزال تشغل في عام 2026

- **Tokenizer drift.**تدريب على الكلمة A، تنشر ضد الكلمة B. تسميات الوهم تختلف، النموذج يخرج القمامة. تحقق`tokenizer.json`الـ (هاشي) في (إي سي)
- **Whitespace ambiguity.**BPE "مرحبا" مقابل "مرحبا" تنتج رموز مختلفة.`add_special_tokens`و`add_prefix_space`صراحة
- **Multilingual undertraining.**إن الكوربوس اللاتينية الثقيلة تنتج المفردات التي تقسم النصوص غير اللاتينية إلى 5 إلى 10 أضعاف أكثر من الرموز. نفس الفوركس يكلف 5 إلى 10 أضعاف في اليابانية / العربية على GPT-3.5. o200k_base تم إصلاح هذا جزئيًا.
- **Emoji splits.**إموجي واحد يمكن أن تأخذ 5 رموز.

## استخدمها

"مجموعة 2026"

| Situation | Pick |
|-----------|------|
| Training a monolingual model from scratch | HF Tokenizers (BPE) |
| Training a multilingual model | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Serving an OpenAI-compatible API | tiktoken (`o200k_base` for GPT-4+) |
| Domain-specific vocab (code, math, protein) | Train custom BPE on domain corpus, merge with base vocab |
| Edge inference, small model | Unigram (smaller vocabularies work better) |

حجم الكلمات هو قرار مقياس وليس ثابت. هيرستيكا قاسية: 32k ل < 1B العوامل، 50-100k ل 1-10B، 200k + ل اللغة متعددة الحدود.

## أرسله

إبقوا`outputs/skill-bpe-vs-wordpiece.md`:

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

## التمارين

1. **Easy.**قم بتشغيل 500 مركبة إضافية`code/main.py`كود ثلاث كلمات متبقية. كم عدد من أنتج بالضبط 1 رمز مقابل > 1 رمز؟
2. **Medium.**مقارنة عدد الرمز على 100 جمل في ويكيبيديا الإنجليزية بين `cl100k_base`،`o200k_base`و (بيتس) ، و (بيتس) ، التي تدربينها مع (فوكاب) = 32 ألف
3. **Hard.**قم بتدريب نفس الجسم باستخدام BPE و Unigram و WordPiece. قم بتقييم دقة التدفق عند استخدام كل منهما على مصنف صغير للمشاعر. هل يقلق الخيار الإبرة بأكثر من 1 نقطة F1؟

## الشروط الرئيسية

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | Greedy merge of most-frequent character pairs until target vocab size hit. |
| Byte-level BPE | No unknown tokens ever | BPE over raw 256 bytes; GPT-2 / Llama use this. |
| Unigram | Probabilistic tokenizer | Prunes from a large candidate set using log-likelihood; used by T5, Gemma. |
| SentencePiece | The whitespace one | Library that trains BPE/Unigram on raw text; space encoded as `▁`. |
| tiktoken | The fast one | OpenAI's Rust-backed BPE encoder for pre-built vocabs. No training. |
| Merge list | The magic numbers | Ordered list of `(a, b) → ab` merges; inference applies in order. |
| Character coverage | How rare is too rare? | Fraction of characters in training corpus the tokenizer must cover; ~0.9995 typical. |

## المزيد من القراءة

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)ورقة BPE
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959)ورقة يونيغرام
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226)المكتبة
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) إشارة موجزة
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) كتاب الطهي + قائمة التشفير
