# 字母标记 BPE,WordPiece,单字体,句子Piece

> 字符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符符

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## 问题

你的词汇有5万个词.一个用户输入"不可代码化".你的代码化器返回.`[UNK]`现在模型没有任何信号,更糟糕的是,您的文件中90个百分比的文件包含40个罕见的词,这意味着每文件中丢掉的信息是40个.

常见词语保持单个标记.罕见词语分解成有意义的部分:`untokenizable`其他`un`现在`token`现在`izable`训练数据涵盖了一切,因为任何字符串最终都是字节的序列.

2026年每一个跨境LLM都使用三个算法 (BPE,UniGram,WordPiece) 运行,包裹在三个图书馆 (tiktoken,SentencePiece,HF Tokenizers) 中.

## 概念

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).**开始一个字符级词汇. 计算每一个相邻的对. 融合最频繁的对 into a new token. 重复直到你达到目标词汇尺寸. 主导算法:GPT-2/3/4,Llama,Gemma,Qwen2,Mistral.

**Byte-level BPE.**虽然它是无码的,但它是无码的.`[UNK]`代币 任何字节序列编码.GPT-2使用了50,257个代币 (256字节+50,000合并+1个特殊).

**Unigram.**开始一个巨大的词汇库. 赋予每个代币一个单数概率. 偶尔切割代币,其删除至少增加了体积日志概率. 推断可能:可以样本代币化 (通过子词规范化来增强数据). T5, mBART, ALBERT, XLNet, Gemma 使用.

**WordPiece.**结合对,最大化了训练体的可能性,而不是原始频率.

**SentencePiece vs tiktoken.**文本Piece是直接在原始的Unicode文本上训练语文库 (BPE或Unigram),编码白色空间为`▁`提克是OpenAI的快速*编码器*对待预先构建的词汇库;它不训练.

基本规则:

- **Training a new vocabulary:**语句Piece (多语言,没有预先代币化) 或HF代币化器.
- **Fast inference against GPT vocab:**投资者:
- **Both:**一本图书馆,培训+服务.

```figure
bpe-merge
```

## 建立它

### 步骤1:从零开始的BPE

看到`code/main.py`循环:

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

算法编码的三个事实.`</w>`标记词尾,所以"低" (后) 和"低" (前) 保持分别.频率权重使高频对得早. 合并列表是顺序的推理应用合并在训练顺序.

### 步骤2:使用学习的合并编码

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

无的O  时代 (N 时代) 制作实现,HF 托克尼斯人使用与优先排列的并列排列搜索,运行在近线性时间.

### 步骤3:实践中的句子

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

通知:不需要预先进行代币化,空间编码为 `▁`现在`character_coverage`控制如何保护和映射到`<unk>`现在,我们要去.

### 步骤4:为OpenAI兼容的词汇的Tik Token

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码,快速 (结后台).与GPT-4/5代码化完全匹配,用于字节计数,成本估计,文本窗口预算.

## 陷在2026年仍存在

- **Tokenizer drift.**标记识别不同,模型输出垃圾.查看`tokenizer.json`鱼在CI.
- **Whitespace ambiguity.**您的位置: 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页 首页`add_special_tokens`其他`add_prefix_space`显然.
- **Multilingual undertraining.**对于GPT-3.5的阿拉伯语,这个提示成本是5-10倍.
- **Emoji splits.**检查点处理当预算背景.

## 用它

现在,我们要做什么?

| Situation | Pick |
|-----------|------|
| Training a monolingual model from scratch | HF Tokenizers (BPE) |
| Training a multilingual model | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Serving an OpenAI-compatible API | tiktoken (`o200k_base` for GPT-4+) |
| Domain-specific vocab (code, math, protein) | Train custom BPE on domain corpus, merge with base vocab |
| Edge inference, small model | Unigram (smaller vocabularies work better) |

词汇大小是扩展决定,而不是常数.粗略的论: <1B参数的32k, 1-10B的50-100k,多语言/边界的200k+.

## 运送它

保存如`outputs/skill-bpe-vs-wordpiece.md`其他:

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

## 运动

1. **Easy.**起500个合式BPE`code/main.py`如何将一个代币与一个代币相比产生?
2. **Medium.**比较100个英语维基百科句子中的代币数量`cl100k_base`现在`o200k_base`报道每一个压缩比率.
3. **Hard.**练习相同的体积,使用BPE,Unigram和WordPiece.在使用小情感分类器时,测量下游精度.选择是否将针移动超过1点F1?

## 关键词

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | Greedy merge of most-frequent character pairs until target vocab size hit. |
| Byte-level BPE | No unknown tokens ever | BPE over raw 256 bytes; GPT-2 / Llama use this. |
| Unigram | Probabilistic tokenizer | Prunes from a large candidate set using log-likelihood; used by T5, Gemma. |
| SentencePiece | The whitespace one | Library that trains BPE/Unigram on raw text; space encoded as `▁`. |
| tiktoken | The fast one | OpenAI's Rust-backed BPE encoder for pre-built vocabs. No training. |
| Merge list | The magic numbers | Ordered list of `(a, b) → ab` merges; inference applies in order. |
| Character coverage | How rare is too rare? | Fraction of characters in training corpus the tokenizer must cover; ~0.9995 typical. |

## 进一步阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)BPE纸
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) 单机报纸
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226)图书馆
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary)简要的参考.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken)厨房书 +编码列表.
