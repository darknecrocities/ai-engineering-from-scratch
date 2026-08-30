# Alt kelime işaretleme  BPE, WordPiece, Unigram, SentencePiece

> Sözcüğü işaretleyenler görünmeyen kelimelerle boğulur. Karakter işaretleyicileri dizilerin uzunluğunu patlatır. Alt kelime işaretleyicileri farkı bölür. Her modern LLM bir teker gemiler.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## Sorun

Sözcükte 50.000 kelime var. Kullanıcı "içerçilemez" yazıyor. Tokenizer geri dönüyor.`[UNK]`Daha da kötüsü, korpusunuzdaki %90'lu belge 40 nadir kelimeyi içerir. Bu da her belgeye 40 bit kaybedilen bilgi anlamına gelir.

Alt kelime işaretleme bunu çözür. Ortak kelimeler tek işaret olarak kalır. Nadir kelimeler anlamlı parçalara parçalanır:`untokenizable`→ `un`- Evet .`token`- Evet .`izable`Eğitim verileri her şeyi kapsar çünkü her bir dizilenin sonunda bir bayt dizisi vardır.

2026'da her sınır LLM üç algoritmadan (BPE, Unigram, WordPiece) birine gönderilmektedir.

## Anlaşım

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).**Karakter seviyesindeki kelime birikimi ile başlayın. Her yanlışı çift sayın. En sık gelen çiftleri yeni bir jetonla birleştirin. Hedef kelime birikimi boyutuna ulaşana kadar tekrarlayın.

**Byte-level BPE.**Aynı algoritma ama Unicode karakterleri yerine çiğ baytlar (256 temel jeton) üzerinde.`[UNK]`GPT-2 50.257 jeton kullanır (256 byte + 50.000 birleşim + 1 özel).

**Unigram.**Büyük bir kelime hazinesinden başlayın. Her bir token'e bir unigram olasılığı verin. En az korpus log olasılığını artıran tokenleri tekrar tekrar kesin. Tahminde muhtemel: örnek tokenizasyonları (azı kelime düzenlenmesi yoluyla veri artırması için yararlı) kullanabilir. T5, mBART, ALBERT, XLNet, Gemma tarafından kullanılır.

**WordPiece.**Birleştirme çiftleri, çürük frekans yerine eğitim korpusunun olasılığını arttırır.

**SentencePiece vs tiktoken.**SentencePiece, kelime kitaplarını (BPE veya Unigram) doğrudan çiğ Unicode metnini kullanarak beyaz alanı `▁`. tiktoken, OpenAI'nin önceden oluşturulmuş kelime hazinelerine karşı hızlı *encoder*idir; eğitim almıyor.

- Başparmak kuralı:

- **Training a new vocabulary:**SentencePiece (çok dilli, önceden tokenizasyon yapılmamış) veya HF Tokenizers.
- **Fast inference against GPT vocab:**tiktoken (cl100k_base, o200k_base).
- **Both:**HF Tokenizers  bir kütüphanede eğitim + hizmet.

```figure
bpe-merge
```

## Yapın

### Adım 1: BPE sıfırdan

Bakın .`code/main.py`- Çubuk:

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

Algoritm'in kodladığı üç gerçek.`</w>`Bu nedenle, "low" (suffix) ve "lower" (prefix) farklı kalır.

### Adım 2: Öğrenilen birleşmelerle kodlayın

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

Üretim uygulamalar, HF Tokenizers, öncelikli kuyruklarla birleşik sıra arama kullanır ve neredeyse doğrusal zamanda çalıştırılır.

### Adım 3: CezaSırası

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

Not: önceden tokenizasyon gerekmiyor, alanı  olarak kodlanmıştır.`▁`- Evet .`character_coverage`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             `<unk>`- Evet .

### Adım 4: OpenAI uyumlu kelime için tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Sadece kodlama. Hızlı (Rust backend). Bağlantı penceresi bütçeleme, byte sayımı, maliyet tahminleri için GPT-4/5 tokenizasyon ile tam uyumludur.

## 2026'da hala yolculuk eden tuzaklar

- **Tokenizer drift.**A kelimeciliği eğitimi, B kelimeciliği ile karşılaştırmak.`tokenizer.json`İK'de hash.
- **Whitespace ambiguity.**BPE "hello" vs "hello" farklı simgeler üretir.`add_special_tokens`ve `add_prefix_space`Açıkça.
- **Multilingual undertraining.**İngilizce ağır corpora, Latin olmayan yazılarını 5-10 kat daha fazla tokene bölünen kelime depoları üretir. Aynı tescil GPT-3.5'te Japonca / Arapça'da 5-10 kat daha fazla maliyet kazanır. o200k_base kısmen bunu düzeltti.
- **Emoji splits.**Tek bir emoji 5 tokeni alabilir.

## Kullan

2026'da:

| Situation | Pick |
|-----------|------|
| Training a monolingual model from scratch | HF Tokenizers (BPE) |
| Training a multilingual model | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Serving an OpenAI-compatible API | tiktoken (`o200k_base` for GPT-4+) |
| Domain-specific vocab (code, math, protein) | Train custom BPE on domain corpus, merge with base vocab |
| Edge inference, small model | Unigram (smaller vocabularies work better) |

Sözcük boyutu, sabit değil, ölçekleme bir karardır. Kaba heuristik: < 1B parametreleri için 32k, 1-10B için 50-100k, çok dilli / sınır için 200k+.

## Gönder

- Kaydet .`outputs/skill-bpe-vs-wordpiece.md`- ...

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

## Egzersizler

1. **Easy.**500 katmanlı bir BPE'yi çalıştır .`code/main.py`Bu küçük bir korpus. 3 uzun süreli kelimeyi kodlayın.
2. **Medium.**İngilizce Wikipedia cümlelerinin simge sayısını `cl100k_base`- Evet .`o200k_base`...ve sözcükle eğitilen SentencePiece BPE'yi 32k ile.
3. **Hard.**Aynı corpus'u BPE, Unigram ve WordPiece ile eğit. Her birini küçük bir duygu sınıflandırıcısı üzerinde kullanırken akıntılı doğruluğu ölçün. Seçim iğneyi 1 noktadan fazla F1'e hareket ettirir mi?

## Anahtar Terimler

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | Greedy merge of most-frequent character pairs until target vocab size hit. |
| Byte-level BPE | No unknown tokens ever | BPE over raw 256 bytes; GPT-2 / Llama use this. |
| Unigram | Probabilistic tokenizer | Prunes from a large candidate set using log-likelihood; used by T5, Gemma. |
| SentencePiece | The whitespace one | Library that trains BPE/Unigram on raw text; space encoded as `▁`. |
| tiktoken | The fast one | OpenAI's Rust-backed BPE encoder for pre-built vocabs. No training. |
| Merge list | The magic numbers | Ordered list of `(a, b) → ab` merges; inference applies in order. |
| Character coverage | How rare is too rare? | Fraction of characters in training corpus the tokenizer must cover; ~0.9995 typical. |

## Daha Fazla Okumak

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)BPE kağıdı.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) Unigram gazetesi.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226)Kütüphane.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) Kısa bir referans.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) yemek kitabı + kodlama listesi.
