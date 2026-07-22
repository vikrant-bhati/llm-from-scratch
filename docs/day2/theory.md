# Day 2 — Text Preprocessing, Tokenization, and Evaluation Basics

---

## 1. Why this topic matters

On Day 1 we said a language model predicts the next *word*. That was a lie of convenience. A neural network cannot see words, letters, or text of any kind — it sees numbers. So before any model runs, something has to chop the text into pieces and hand each piece an integer ID. That something is the **tokenizer**, and the pieces are **tokens**.

This step looks like plumbing. It is not. The tokenizer is the one design choice that every later layer inherits and none can undo:

1. **It sets the vocabulary** — the fixed list of tokens the model is even *allowed* to know. Everything outside it must be faked or dropped.
2. **It sets the sequence length** — how many tokens a sentence becomes, which controls memory, speed, and cost. Attention cost grows with the *square* of this number (Day 12), so the tokenizer silently sets your compute bill.
3. **It decides who is a stranger.** On Day 1 we met the OOV ("out-of-vocabulary") problem and papered over it with a `⟨unk⟩` token — every unknown word collapsed to one useless symbol. Whether a model can even *represent* a new name, a typo, or a rare word is decided here, before training begins.

There's a second reason this day exists, hidden in its title: **evaluation basics.** Day 1 gave us perplexity and left a trap wide open — *perplexity depends on what a "token" is.* Change the tokenizer and the same model on the same text reports a completely different perplexity. Before we can compare any two models in this series, we need to understand that trap and how to escape it. Tokenization and evaluation are the same lesson: **the unit you measure in decides the number you get.**

The plot of the day, in one line: naive tokenizers force a brutal trade-off between a giant vocabulary and losing all rare words — and **subword tokenization** dissolves the trade-off, which is why every modern LLM uses it.

## 2. Before subwords: the classical preprocessing pipeline

Long before neural nets, "text preprocessing" was a fixed assembly line, documented in Manning, Raghavan & Schütze's *Introduction to Information Retrieval* (the standard reference for the pre-neural era). Search engines and classifiers ran text through these stages:

1. **Tokenization** — split the character stream into words.
2. **Case folding** — `The` → `the`, so capitalization doesn't split a word in two.
3. **Stop-word removal** — delete ultra-common words (`the`, `of`, `is`) as "noise."
4. **Stemming / lemmatization** — crush `organize`, `organizes`, `organizing`, `organization` down to one root, so they count as the same term. The **Porter stemmer** (1980) did this with a fixed list of suffix-stripping rules; lemmatization did it with a dictionary.

Every stage had the same goal: **throw away variation the system couldn't handle, so different surface forms would collapse to the same unit.** It was the rule-era mindset from Day 1, applied to words: hand-written rules deciding what counts as "the same."

And like the rule era, most of it is **gone** from modern LLMs. GPT does not fold case (it needs to know `US` the country from `us` the pronoun), does not remove stop words (they carry grammar), and does not stem (`running` and `ran` are genuinely different to a model that predicts the next token). The reason is the Day 1 reason: **hand-written normalization throws away information a data-driven model would rather keep and learn from.** The one survivor is tokenization itself — but even that got rebuilt from rules into something learned. That rebuild is the story of this day.

Why did the old pipeline strip so aggressively? Because of the wall we hit next.

## 3. The tokenization dilemma: two bad extremes

Start with the most obvious tokenizer imaginable.

### 3.1 Whitespace and rules

**Whitespace tokenization:** split on spaces. `"the cat sat"` → `["the", "cat", "sat"]`. Done. It even works — until it immediately doesn't:

- `"cat."` and `"cat"` become **different tokens** — punctuation glues onto words.
- `"don't"` → one token? two? Where does the split go — `don` + `'t`, or `do` + `n't`?
- `"New York"` is one concept but two tokens; `"e-mail"` is one word but splits.
- **Chinese and Japanese have no spaces at all.** Whitespace tokenization returns the entire sentence as a single token. It simply doesn't apply.

The classical fix was **rule-based / regex tokenization** — the **Penn Treebank** tokenizer is the canonical one: a pile of regular expressions that peel punctuation off, split contractions into `do` + `n't`, keep `New York` rules, and so on. This is better, and it's still useful. But it is exactly the rule era again: **hand-built, language-specific, and endless.** A new language needs a new rulebook. Emoji, URLs, code, hashtags — each needs another rule. It never converges.

But suppose we solve splitting perfectly. We still face the real wall, and it has nothing to do with punctuation.

### 3.2 The real wall: words vs characters

Once you can split text into words, you must **freeze a vocabulary** — the finite list of tokens the model will know. Now you're forced to choose the *size* of your unit, and both ends of the choice are painful.

**Word-level tokens.** Let each word be one token.

- 👍 Short sequences — one sentence ≈ a dozen tokens. Each token is meaning-rich.
- 👎 **The vocabulary explodes.** English has hundreds of thousands of word forms; with names, typos, and morphology it's effectively unbounded. You must cap it (say, top 50,000 words).
- 👎 **The OOV catastrophe.** Every word past the cap — every rare word, every new name, every misspelling — becomes `⟨unk⟩`. The model literally cannot tell `Shakespeare` from `antidisestablishmentarianism` from a typo; all three are the same blank symbol. And worst of all, **it can never generate them either.** This is the Day 1 wall, and it is fatal: language is *made of* rare words (Zipf's law again — most words are rare). A word-level model is permanently blind to the long tail.

**Character-level tokens.** Go to the opposite extreme: every character is a token.

- 👍 **Zero OOV, ever.** The vocabulary is ~100 characters. Any word — including one invented tomorrow — is spellable. `Shakespeare` is just `S,h,a,k,e,s,p,e,a,r,e`. The stranger problem *vanishes*.
- 👎 **Sequences get long** — 5× to 10× more tokens than words. Since attention cost grows with the square of sequence length, this is brutally expensive.
- 👎 **Each token is nearly meaningless.** The letter `e` carries almost no information on its own. The model must spend its early layers just reassembling characters into words before it can even start on meaning. We throw the whole burden of "what is a word" onto the network.

Here is the dilemma in one table:

| | Vocabulary size | Sequence length | OOV problem | Meaning per token |
| --- | --- | --- | --- | --- |
| **Word-level** | huge (and still caps out) | short 👍 | catastrophic 👎 | high 👍 |
| **Character-level** | tiny 👍 | very long 👎 | none 👍 | almost none 👎 |

Look at the columns: **the two approaches are good and bad in exactly opposite places.** That is the tell-tale sign of a real trade-off — and the invitation to find something in the middle. What if the unit weren't a whole word *or* a single character, but a **frequently reused chunk** — a *subword*?

## 4. Subword tokenization: the resolution

The idea in one sentence:

> **Keep common words whole, and break rare words into smaller reusable pieces.**

So `the`, `cat`, and `sat` stay single tokens (they're common — give them the word-level advantage). But `tokenization` might split into `token` + `ization`, and a novel word like `unhuggable` into `un` + `hug` + `gable`. The pieces are drawn from a fixed vocabulary of subword units, and here's the magic:

> **If the vocabulary includes every single character as a fallback, then any string whatsoever can be spelled out of pieces. OOV is mathematically impossible.**

Read that against the table above. Subwords take the **short sequences and high meaning** of word-level (common words stay whole) *and* the **zero-OOV guarantee** of character-level (rare words decompose, ultimately down to characters). The trade-off didn't get balanced — it got *dissolved*. You choose one knob, the **vocabulary size** (say 30k–50k), and it interpolates smoothly: common stuff is word-like, rare stuff is character-like, and nothing is ever unknown.

This is why **every modern LLM — GPT, BERT, Llama, all of them — uses subword tokenization.** The only real questions left are: *which* pieces should be in the vocabulary, and *how do we learn them from data?* Three answers dominate, and they are the three papers of the day.

## 5. Byte-Pair Encoding (BPE) — Sennrich et al. (2016)

BPE is the most widely used answer and the easiest to understand, because it's pure counting — the Day 1 spirit exactly. Its origin is a twist: it was a **data compression** algorithm (Philip Gage, 1994), and Sennrich, Haddow & Birch repurposed it for tokenization in *Neural Machine Translation of Rare Words with Subword Units* (2016). Their motivating problem was precisely our wall: translation models kept failing on rare and unseen words. Subwords fixed it.

### 5.1 The algorithm: merge the most frequent pair, over and over

The recipe is almost insultingly simple:

1. Start with the text split into **characters** — the smallest possible pieces. This is your starting vocabulary (every character) plus nothing else.
2. **Count every adjacent pair** of symbols in the corpus.
3. **Merge the single most frequent pair** into one new symbol. Add it to the vocabulary.
4. **Repeat** steps 2–3 until the vocabulary reaches the size you wanted.

That's it. Frequent pairs of characters fuse into subwords; frequent pairs of subwords fuse into words. Common sequences bubble up to become single tokens; rare ones stay in pieces. The vocabulary size is just *how many times you run the loop*.

### 5.2 Doing it by hand — the `hug` corpus

Exactly like Day 1's cat corpus, we'll use one tiny example we can check entirely by hand — and we'll reuse it as the unit test for our code. Our whole "training corpus" is five words with these frequencies:

```text
hug  : 10      pug : 5      pun : 12      bun : 4      hugs : 5
```

(BPE works on a table of word counts, not the raw stream — identical words are counted once with a multiplier, which is just bookkeeping.) Split every word into characters. The starting vocabulary is the 7 distinct characters: `b, g, h, n, p, s, u`.

```text
h u g       : 10
p u g       : 5
p u n       : 12
b u n       : 4
h u g s     : 5
```

**Merge 1.** Count every adjacent pair, weighting by word frequency:

| Pair | Where it appears | Count |
| --- | --- | --- |
| (h, u) | hug×10, hugs×5 | 15 |
| **(u, g)** | **hug×10, pug×5, hugs×5** | **20** |
| (p, u) | pug×5, pun×12 | 17 |
| (u, n) | pun×12, bun×4 | 16 |
| (b, u) | bun×4 | 4 |
| (g, s) | hugs×5 | 5 |

Winner: **(u, g) = 20.** Merge it into `ug`. Add `ug` to the vocabulary. The corpus is now:

```text
h ug        : 10
p ug        : 5
p u n       : 12
b u n       : 4
h ug s      : 5
```

**Merge 2.** Recount pairs on the *updated* corpus:

| Pair | Count |
| --- | --- |
| (h, ug) | 15 |
| (p, ug) | 5 |
| (p, u) | 12 |
| **(u, n)** | **16** |
| (b, u) | 4 |
| (ug, s) | 5 |

Winner: **(u, n) = 16.** Merge into `un`:

```text
h ug        : 10
p ug        : 5
p un        : 12
b un        : 4
h ug s      : 5
```

**Merge 3.** Recount:

| Pair | Count |
| --- | --- |
| **(h, ug)** | **15** |
| (p, ug) | 5 |
| (p, un) | 12 |
| (b, un) | 4 |
| (ug, s) | 5 |

Winner: **(h, ug) = 15.** Merge into `hug`.

After three merges the vocabulary is the 7 characters plus `{ug, un, hug}` = 10 tokens, and the model has **learned an ordered list of merge rules**:

```text
1. u + g  → ug
2. u + n  → un
3. h + ug → hug
```

That ordered list *is* the trained tokenizer. Training was counting and merging — no gradients, one concept. (Sound familiar? Day 1's "training = counting.")

### 5.3 Using the trained tokenizer — and why OOV is gone

To tokenize a **new** word, split it to characters and apply the learned merges **in the order they were learned**:

- **`bug`** (never seen in training!): `b u g` → apply rule 1 `(u,g)→ug` → `b ug`. Rules 2 and 3 don't apply. Result: **`[b, ug]`** — two tokens. We never saw `bug`, yet it tokenized cleanly into known pieces. **No `⟨unk⟩`.** This is the payoff we were promised on Day 1: the rare/unseen word didn't collapse to a blank — it decomposed into reusable parts the model already understands.

- **`hugs`**: `h u g s` → `(u,g)→ug` → `h ug s` → `(h,ug)→hug` → `hug s`. Result: **`[hug, s]`**. The common word is (almost) one piece; the plural falls out as a reusable `s`.

The model can even generate `bug` — something a word-level model that never saw it could never do. The stranger problem is genuinely dissolved.

### 5.4 The last leak: byte-level BPE

One crack remains. Tokenize **`mug`**: `m u g` → ... but `m` never appeared in our training corpus, so it's not in the vocabulary. Character-level BPE still has an OOV problem — for *characters* it has never seen (a rare Unicode symbol, an emoji, a Chinese character in an English-trained tokenizer).

GPT-2's fix (Radford et al., 2019) is beautiful and we'll meet it properly later: **run BPE over raw bytes, not characters.** There are only 256 possible bytes, and *every* possible piece of text — every language, emoji, or control character — is a sequence of bytes. Put all 256 bytes in the base vocabulary and there is **nothing left that can be OOV, at any level.** Byte-level BPE is the true final form: a tokenizer that can encode literally any string on Earth, and the one GPT uses. We flag it here and implement the character version now; the jump to bytes is one line.

## 6. WordPiece — Schuster & Nakajima (2012)

WordPiece came *first* (2012, for Google's Japanese/Korean voice search) and became famous as the tokenizer inside **BERT** (Day 19). Algorithmically it's BPE's twin with **one changed line** — and that one line is a genuinely illuminating idea, so it's worth seeing.

BPE merges the **most frequent** pair. WordPiece asks a smarter question: *not "which pair is most common?" but "which pair, when merged, most improves a language model of the corpus?"* In practice this reduces to merging the pair with the best **score**:

$$\text{score}(a, b) = \frac{\text{freq}(a\,b)}{\text{freq}(a) \times \text{freq}(b)}$$

Read the formula: the numerator rewards pairs that co-occur a lot; the denominator **penalizes pairs whose pieces are already common on their own.** So WordPiece doesn't want the most frequent pair — it wants the pair that is *surprisingly* frequent, given how common its parts are. (This ratio is the exponential of pointwise mutual information — a "how much more together than apart?" score. Statisticians will recognize it.)

**Watch it disagree with BPE on our very first merge.** Recompute merge 1 on the `hug` corpus, but with WordPiece's score. We need each symbol's total frequency: `u`=36, `g`=20, `h`=15, `p`=17, `n`=16, `b`=4, `s`=5.

| Pair | freq(pair) | freq(a)×freq(b) | score |
| --- | --- | --- | --- |
| (h, u) | 15 | 15×36 | 0.028 |
| (u, g) | 20 | 36×20 | 0.028 |
| (p, u) | 17 | 17×36 | 0.028 |
| (u, n) | 16 | 36×16 | 0.028 |
| (b, u) | 4 | 4×36 | 0.028 |
| **(g, s)** | **5** | **20×5** | **0.050** |

BPE merged `(u,g)` first (frequency 20, the biggest). **WordPiece merges `(g,s)` first** — even though it's one of the *rarest* pairs! Why? Because `s` appears only ever glued to `g`: the two are rare apart but perfectly bound together, so their score is high. Every pair involving `u` scores the same low value (0.028) precisely *because* `u` is so common that pairing with it is unremarkable — it co-occurs with everything, so co-occurring with any one partner means little.

That's the whole personality difference in one number: **BPE follows raw popularity; WordPiece follows association strength.** BPE will happily merge a common letter onto everything; WordPiece holds out for pairs that "belong together." Two other WordPiece surface details: it marks non-initial pieces with `##` (so `hugs` → `hug`, `##s`), and at inference it greedily matches the **longest** vocabulary token at each position rather than replaying merges. But the soul of it is that one ratio.

## 7. Unigram LM and SentencePiece — Kudo & Richardson (2018)

BPE and WordPiece are **bottom-up**: start from characters, greedily glue pieces together. Kudo's **Unigram Language Model** tokenizer (2018) goes the other direction — **top-down** — and it brings back an idea straight from Day 1.

### 7.1 The Unigram idea: start big, prune down

1. Start with a **huge** candidate vocabulary — way more subwords than you want (all substrings that appear often enough).
2. Treat tokenization **probabilistically**: give every subword a probability, so any way of cutting a word into known pieces has a total probability. A word can be segmented many ways, and the model scores each.
3. **Iteratively prune:** for each candidate token, ask "how much would the corpus's total probability drop if I deleted this token?" Remove the tokens the corpus needs *least*. Repeat until you hit the target vocabulary size.

At inference, tokenizing a word means finding its **most probable segmentation** — the split whose pieces multiply to the highest probability. Finding that best split efficiently is done with the **Viterbi algorithm** — the *exact same* dynamic-programming shortcut we met on Day 1 inside Hidden Markov Models. Day 1's tool for "find the single most likely hidden story" is Day 2's tool for "find the single most likely way to cut this word." The ideas keep coming back.

There's a bonus the bottom-up methods can't offer: because Unigram assigns a *probability* to every possible segmentation, it can **sample different segmentations during training** (Kudo's "subword regularization") — the same word gets cut different ways on different epochs, which acts like data augmentation and makes the model more robust. BPE gives you exactly one segmentation; Unigram gives you a distribution.

### 7.2 SentencePiece: the tool that made it practical

**SentencePiece** (Kudo & Richardson, 2018, the software) is often confused with the Unigram algorithm, but it's a separate, orthogonal contribution — an *implementation framework* (it can run either Unigram or BPE inside it). Its key move solves a problem we've been quietly dodging:

Every method so far assumed you already **pre-split the text into words** (usually on whitespace). But that pre-splitting is itself the brittle, language-specific rule-pile from Section 3 — and it's *lossy*: after tokenizing you can't perfectly reconstruct the original text (was there one space or two? a tab?). SentencePiece's fixes:

- **Treat the input as a raw stream of Unicode characters — no pre-tokenization at all.** This is what lets it work on Chinese, Japanese, and Thai, which have no spaces to split on. The rulebook is gone.
- **Encode whitespace explicitly** as a visible meta-symbol `▁` (U+2581). A space is just another character to be tokenized. Now tokenization is perfectly **reversible** — glue the tokens back, swap `▁`→space, and you have the *exact* original bytes. This "lossless, language-agnostic, no pre-tokenization" property is why SentencePiece became the default for models like T5, Llama, and Mistral.

### 7.3 The three, side by side

| | Direction | Merge/keep rule | Marks | Famous in |
| --- | --- | --- | --- | --- |
| **BPE** | bottom-up | most **frequent** adjacent pair | (space or `▁`) | GPT-2/3/4, Llama, RoBERTa |
| **WordPiece** | bottom-up | highest **score** = freq(ab)/(freq(a)·freq(b)) | `##` prefix | BERT, DistilBERT |
| **Unigram** | top-down | prune tokens that cost the **least likelihood** | (via SentencePiece `▁`) | T5, ALBERT, XLNet, mBART |

They differ in mechanism but agree on the thing that matters: **a fixed vocabulary of subword pieces, with characters/bytes as the floor, so nothing is ever out-of-vocabulary.** That shared property — not the specific algorithm — is what made the modern LLM possible.

## 8. Evaluation basics: measuring a tokenizer, and the perplexity trap

The day's title promises "evaluation basics," and there are two distinct questions here: how do you judge a *tokenizer*, and how does tokenization poison the way we judge a *model*?

### 8.1 Judging a tokenizer

There's no single score, but a few standard numbers:

- **Vocabulary size** — the knob you set (e.g., 32k, 50k). Bigger vocab → shorter sequences but more parameters in the model's embedding table (Day 4) and more rare tokens seen too little to learn well.
- **Fertility** — average **tokens per word**. 1.0 means every word is one token (word-level); higher means more splitting. A good subword tokenizer on English scores around 1.1–1.5; on a language it wasn't trained for, fertility spikes (this is why non-English text can cost 2–3× more tokens — and dollars — in commercial APIs).
- **Compression / sequence length** — average **tokens per sentence** or per 1000 characters. Fewer tokens for the same text = cheaper and faster, but *only if* meaning is preserved. This is the number you're really trading off against vocabulary size.
- **OOV rate** — for subword tokenizers with a byte/character floor, this is **zero by construction.** For word-level baselines it's brutal, and measuring it is the cleanest way to *see* the wall from Section 3.

The tokenizer craft is picking a vocabulary size that keeps sequences short **and** keeps fertility low **and** doesn't waste the vocabulary on junk — a balance, not an optimum.

### 8.2 The perplexity trap (this is the important one)

Day 1 handed us perplexity and called it *the* scorecard for language models. Here is the landmine it left buried:

> **Perplexity is "average surprise *per token*." So its value depends entirely on what a token is. Change the tokenizer and you change the number — for the same model, on the same text.**

Concretely: a **character-level** model and a **word-level** model reading the identical sentence will report wildly different perplexities, because one is surprised ~once per letter and the other ~once per word — utterly different units. A word-level perplexity of 200 and a character-level perplexity of 3 might describe *equally good* models. **You cannot compare perplexities across tokenizers.** Every leaderboard number "PPL = 18.4" is meaningless without "...on *this* tokenizer." This is one of the most common ways people fool themselves when comparing models.

The escape is exactly the Day 1 escape — go back to a unit that doesn't depend on your choices. Recall the chain: perplexity is $2^{H}$ where $H$ is average surprise per token (cross-entropy in bits). If instead we spread that *total* surprise over a fixed physical unit — **characters** or **bytes**, which no tokenizer can change — we get a tokenizer-independent score:

$$\text{Bits-per-character (BPC)} = \frac{\text{total cross-entropy of the text (bits)}}{\text{number of characters in the text}}$$

Because the number of characters (or bytes) in a passage is fixed no matter how you chop it into tokens, **BPC is comparable across any two models with any two tokenizers.** It's the honest scorecard. And it connects straight back to Day 1's opening: Shannon's "English is about 1 bit per letter" was a **bits-per-character** claim — the tokenizer-free measure was there at the very beginning of the story. Modern LLMs push real text down toward ~0.6–0.9 bits per character, right at the bottom of Shannon's 1951 staircase.

The one-sentence rule to carry forever: **compare models in bits-per-character, never in raw perplexity across different tokenizers.**

## 9. What tokenization can't do → the road ahead

Subwords dissolve the OOV wall, but they hand the model a bag of pieces with **no notion of meaning**. To the tokenizer, `cat` and `dog` are just two unrelated integer IDs, as alien to each other as `cat` and `433`. It knows how to *split* text; it knows nothing about what the pieces *mean* or how they relate.

That missing piece — turning token IDs into vectors where similar tokens sit close together, so what the model learns about `cat` transfers to `dog` — is **embeddings**, and it's exactly where Day 1 also pointed. It's Days 4–6. Tokenization is the input layer's *first* half (text → integer IDs); embeddings are the second half (integer IDs → meaningful vectors). Together they're the on-ramp to every model in the rest of this roadmap.

A few loose threads this day leaves for later, all real and all revisited:

- **Byte-level BPE** (GPT-2) — the truly-nothing-is-OOV final form (Days 18, 23).
- **Tokenization still bites.** It's why LLMs struggle to count letters in a word ("how many *r*'s in strawberry?" — the model sees tokens, not letters) and why arithmetic is weird (numbers tokenize inconsistently). The input layer's choices leak all the way to the model's dumbest failures.
- **Tokenizer-free / byte-level models** are an active research direction — pushing the pieces all the way down to bytes and letting the network do everything. The character-level extreme, revived with modern compute.

But notice what did **not** change from Day 1. The game is still: assign probabilities to sequences, train by minimizing average surprise (cross-entropy), grade by perplexity — *now measured carefully, in bits-per-character.* We changed what a "symbol" in the sequence is. We did not change the game. Every model in this series will still play it.

## 10. Glossary (only what today uses)

| Term | Plain meaning |
| --- | --- |
| **Token** | One unit of text after splitting — a word, a subword piece, a character, or a byte. |
| **Tokenizer** | The program that turns raw text into a sequence of token IDs (and back). |
| **Vocabulary** | The fixed, finite list of tokens a model is allowed to use. |
| **OOV** (out-of-vocabulary) | Text the tokenizer has no token for. With a character/byte floor, this can't happen. |
| **Subword** | A token that is a piece of a word (e.g., `token`, `##ization`). The middle ground between words and characters. |
| **BPE** | Byte-Pair Encoding: build the vocabulary by repeatedly merging the most *frequent* adjacent pair. |
| **WordPiece** | Like BPE, but merges the pair with the highest freq(ab)/(freq(a)·freq(b)) *score* — association, not raw frequency. |
| **Unigram LM** | Top-down tokenizer: start huge, prune tokens that cost the least corpus likelihood; segment via Viterbi. |
| **SentencePiece** | A tokenizer tool that runs on raw text (no pre-splitting), encoding whitespace as `▁` so it's lossless and language-agnostic. |
| **Fertility** | Average tokens per word — a measure of how much a tokenizer splits. |
| **Bits-per-character (BPC)** | Cross-entropy per character — the tokenizer-*independent* way to compare models. Shannon's original unit. |

## 11. References

1. Manning, C. D., Raghavan, P. & Schütze, H. (2008). *Introduction to Information Retrieval*, Ch. 2 "The term vocabulary and postings lists." — the classical tokenization / normalization / stemming pipeline; free online at Stanford.
2. Sennrich, R., Haddow, B. & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL. — brought BPE (Gage, 1994) to NLP as the answer to the rare-word / OOV wall.
3. Schuster, M. & Nakajima, K. (2012). *Japanese and Korean Voice Search.* ICASSP. — the WordPiece algorithm; later the tokenizer inside BERT.
4. Kudo, T. & Richardson, J. (2018). *SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing.* EMNLP (system demo). — the language-agnostic, lossless tokenizer tool. See also Kudo (2018), *Subword Regularization*, for the Unigram LM.
5. Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2). — byte-level BPE, the nothing-is-OOV final form. (Revisited later in the roadmap.)

---

**Next:** we build it. A BPE tokenizer from scratch — the merge loop, checked against the `hug` corpus by hand — then unleashed on Tiny Shakespeare and WikiText-2, where we'll *watch* the word-level OOV wall appear, watch subwords dissolve it, and plot the vocabulary-size ↔ sequence-length trade-off with real numbers.
