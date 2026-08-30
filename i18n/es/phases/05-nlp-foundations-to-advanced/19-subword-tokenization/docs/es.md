# Tokenización de palabras subsiguientes  BPE, WordPiece, Unigram, SentencePiece

> Los tokenizadores de palabras se ahogan con palabras invisibles, los tokenizadores de caracteres aumentan la longitud de la secuencia, los tokenizadores de palabras subdividen la diferencia, cada LLM moderno se envía a uno.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## El problema

Tu vocabulario tiene 50.000 palabras. Un usuario escribe "no se puede tokenizar". Tu tokenizer vuelve.`[UNK]`El modelo ahora no tiene señal sobre la palabra. Lo peor: el documento del 90o percentil en su corpus tiene 40 palabras raras, lo que significa 40 bits de información perdida por documento.

La tokenización de palabras subconsumidas resuelve esto. Las palabras comunes permanecen como tokens únicos. Las palabras raras se descomponen en piezas significativas:`untokenizable`¿ Qué es esto ?`un`¿ Qué ?`token`¿ Qué ?`izable`Los datos de entrenamiento cubren todo porque cualquier cadena es en última instancia una secuencia de bytes.

Cada LLM fronterizo en 2026 se envía en uno de los tres algoritmos (BPE, Unigram, WordPiece), envuelto en una de las tres bibliotecas (tiktoken, SentencePiece, HF Tokenizers).

## El concepto

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).**Comience con un vocabulario de nivel de caracteres. Cuente cada par adyacente. Combine el par más frecuente en un nuevo token. Repita hasta que alcances el tamaño del vocabulario objetivo. Algoritmo dominante: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**Byte-level BPE.**El mismo algoritmo pero con más bytes crudos (256 tokens base) en lugar de caracteres Unicode.`[UNK]`Los tokens  codifican cualquier secuencia de byte. GPT-2 utiliza 50.257 tokens (256 bytes + 50.000 fusiones + 1 especial).

**Unigram.**Comience con un vocabulario enorme. Asigná a cada token una probabilidad de unigrama. Iterativamente poda los tokens cuya eliminación aumenta menos la probabilidad de registro del corpus. Probablemente en la inferencia: puede muestrar tokenizaciones (utiles para el aumento de datos a través de la regularización de subpalabra).

**WordPiece.**Combinando parejas que maximizan la probabilidad del cuerpo de entrenamiento en lugar de la frecuencia bruta.

**SentencePiece vs tiktoken.**SentencePiece es la biblioteca que *entraña* vocabularios (BPE o Unigram) directamente en texto Unicode crudo, codificando el espacio blanco como `▁`. tiktoken es el codificador rápido de OpenAI contra vocabularios preconstruidos; no se entrena.

Regla de oro:

- **Training a new vocabulary:**SentencePiece (multilingüe, sin pre-tokenization) o HF Tokenizers.
- **Fast inference against GPT vocab:**Tiktoken (cl100k_base, o200k_base).
- **Both:**HF Tokenizers  una biblioteca, formación + servicio.

```figure
bpe-merge
```

## Construye el mismo

### Paso 1: BPE desde cero

¿ Qué ?`code/main.py`El bucle:

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

Tres hechos que el algoritmo codifica.`</w>`Los puntos de referencia de la lista de combinación se ordenan  la inferencia se aplica a las combinaciones en orden de entrenamiento.

### Paso 2: codificar con las fusiones aprendidas

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

Las implementaciones de producción de tokens, HF Tokenizers, utilizan la búsqueda de rangos de fusión con colas de prioridad y se ejecutan en tiempo casi lineal.

### Paso 3: SentenciaPiece en la práctica

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

Nota: no se requiere pre-tokenization, espacio codificado como `▁`¿ Qué ?`character_coverage`control de cómo se conservan los caracteres agresivamente raros vs.`<unk>`¿ Qué ?

### Paso 4: tiktoken para los vocabularios compatibles con OpenAI

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Sólo codificación. Rápido (backend de Rust). Exactamente coincide con la tokenización GPT-4/5 para el conteo de byte, estimación de costos, presupuesto de ventana de contexto.

## Las trampas que todavía se envían en 2026

- **Tokenizer drift.**Entrenamiento en la vocab A, desplegamiento contra la vocab B. Los tokens de identificación difieren; el modelo de salida basura.`tokenizer.json`el hash en CI.
- **Whitespace ambiguity.**BPE "hola" vs "hola" producen diferentes tokens. Siempre especifique `add_special_tokens`y `add_prefix_space`- Es decir, explícitamente.
- **Multilingual undertraining.**Los corpora de inglés pesados producen vocabularios que dividen las escrituras no latinas en 5-10 veces más tokens.
- **Emoji splits.**Un emoji puede tomar 5 tokens. Control de emoji de punto de manejo cuando presupuesta contexto.

## Usalo

La pila de 2026:

| Situation | Pick |
|-----------|------|
| Training a monolingual model from scratch | HF Tokenizers (BPE) |
| Training a multilingual model | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Serving an OpenAI-compatible API | tiktoken (`o200k_base` for GPT-4+) |
| Domain-specific vocab (code, math, protein) | Train custom BPE on domain corpus, merge with base vocab |
| Edge inference, small model | Unigram (smaller vocabularies work better) |

El tamaño del vocabulario es una decisión de escala, no una constante. Heurística aproximada: 32k para <1B parámetros, 50-100k para 1-10B, 200k+ para multilingüe / frontera.

## Envío

Salvo como`outputs/skill-bpe-vs-wordpiece.md`¿Qué es esto ?

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

## Los ejercicios

1. **Easy.**Entrenar un BPE de 500 fusiones en `code/main.py`¿Cuántas palabras se produjeron exactamente 1 token vs > 1 token?
2. **Medium.**Comparar el recuento de tokens en 100 frases de Wikipedia en inglés entre `cl100k_base`¿ Qué ?`o200k_base`, y un BPE SentencePiece que entrenas con vocab = 32k.
3. **Hard.**Entrenar el mismo corpus con BPE, Unigram y WordPiece. Medir la precisión en el río abajo cuando se utiliza cada uno en un clasificador de sentimiento pequeño. ¿La elección mueve la aguja por más de 1 punto F1?

## Términos clave

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | Greedy merge of most-frequent character pairs until target vocab size hit. |
| Byte-level BPE | No unknown tokens ever | BPE over raw 256 bytes; GPT-2 / Llama use this. |
| Unigram | Probabilistic tokenizer | Prunes from a large candidate set using log-likelihood; used by T5, Gemma. |
| SentencePiece | The whitespace one | Library that trains BPE/Unigram on raw text; space encoded as `▁`. |
| tiktoken | The fast one | OpenAI's Rust-backed BPE encoder for pre-built vocabs. No training. |
| Merge list | The magic numbers | Ordered list of `(a, b) → ab` merges; inference applies in order. |
| Character coverage | How rare is too rare? | Fraction of characters in training corpus the tokenizer must cover; ~0.9995 typical. |

## Leer más

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) el papel BPE.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) el papel de Unigram.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226)La biblioteca.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) Referencia concisa.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) libro de cocina + lista de codificación.
