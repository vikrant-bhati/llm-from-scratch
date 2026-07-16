# Day 1 — Evolution of NLP: From Rules to Statistical Language Models

---

## 1. Why this topic matters

Today's LLMs converse, reason, write code, and act as autonomous agents. But if we strip away the scale then every one of them is still answering the question this field started with:

> **Given the words so far, what word comes next — and with what probability?**

That question was posed mathematically in 1948, decades before neural networks were practical. Understanding the original statistical formulation matters for three reasons:

1. **The objective never changed.** Current LLMs like GPT are trained to minimize **cross-entropy** (explained below) of next-token prediction — the exact quantity Shannon defined. An n-gram model and GPT-4 optimize the same loss; they differ in how they represent context.
2. **The evaluation never changed.** **Perplexity** (defined below) is still reported for every foundation model release.
3. **The failure modes of simple models explain the design of complex ones.** Every limitation of n-grams (sparsity, no generalization, fixed context) directly motivates a later invention (embeddings, neural LMs, attention). Day 1 is the "why" for the rest of this roadmap.

## 2. Historical context: the rationalism vs. empiricism war

For roughly 1960–1990, mainstream NLP was **rationalist** which means that language is governed by rules, so encode grammar and linguistic theory by hand (parsers, expert systems, Chomsky's influence). But this produced a brittle systems:

- **Coverage:** real text is endlessly messy — headlines, typos, idioms, novel words. Hand-written rules never covered enough of it.
- **Ambiguity:** rules could enumerate the possible readings of a sentence but had no principled way to say which reading is _likely_. ("I saw the man with the telescope". It doesn't say who has the telescope i.e. either I used a telescope to see the man, or you saw a man who had a telescope with him).
- **Cost:** every domain and language needed expert linguists writing new rules, and rule sets interacted unpredictably as they grew.

But then this shifted. The **empiricist** alternative — that we can estimate how language behaves from _counts over real usage_ — had existed since Shannon, but needed two things that arrived in the 1980s: large machine-readable corpora and cheap computation. Speech recognition at IBM (Jelinek's group) proved the approach worked, and by the early 1990s the field had flipped. Frederick Jelinek's (possibly apocryphal, definitely accurate) quip captures the whole war in one line:

> _"Every time I fire a linguist, the performance of the speech recognizer goes up."_

The lesson was not "linguistics is useless" — it was that **probability estimated from data beats hand-written certainty** whenever language gets ambiguous, which is always.

## 3. Core intuition: language is a predictable sequence

Before any math: language is redundant and predictable. If you see the letter `t` in English text, `h` is a very likely next letter. If you read "The cat sat on the \_\_\_", your brain has already ranked the candidates (`mat`, `floor`, `sofa`...) and ruled out others (`the`, `photosynthesis`).

A **language model (LM)** is just this ability, made explicit:

> **A language model is a function that assigns a probability to a sequence of words** — equivalently, one that predicts a probability distribution over the next word given the words so far.

That one sentence is the entire field. Everything else — n-grams, HMMs, RNNs, Transformers — is a different answer to _"how do we represent 'the words so far'?"_

## 4. Shannon (1948): information, entropy, and approximating English

Claude Shannon's _A Mathematical Theory of Communication_ (Bell System Technical Journal, 1948) was about transmitting messages over noisy channels, but it created the mathematical vocabulary of language modeling.

### 4.1 Self-information: rare events carry more information

The **self-information** of an event $x$ with probability $p(x)$:

$$I(x) = \log_2 \frac{1}{p(x)} = -\log_2 p(x) \quad \text{(bits)}$$

Intuition: a certain event ($p=1$) tells you nothing ($I=0$); a rare event is a big surprise and carries many bits. "The sun rose today" is low-information; "it snowed in Death Valley" is high-information.

### 4.2 Entropy: the average surprise of a source

**Entropy** is the _expected_ self-information over all events the source can produce:

$$H = \sum_x p(x) \log_2 \frac{1}{p(x)} = -\sum_x p(x)\log_2 p(x)$$

Entropy measures how unpredictable a source is, in bits per symbol. A fair coin: $H = 1$ bit. A two-headed coin: $H = 0$. English text: Shannon estimated roughly ~1 bit per letter — far below the $\log_2 27 \approx 4.75$ bits of random letters, which is a _measurement_ of how redundant and predictable language is.

One caution: entropy is a property of **language itself** — the irreducible unpredictability no model can beat. To score a *model*, we need its sibling.

### 4.3 Cross-entropy: the average surprise of your model

**Cross-entropy** uses the same recipe as entropy but crosses two distributions: events happen according to the true distribution $p$, yet surprise is charged using **your model's** probabilities $q$:

$$H(p, q) = \sum_x p(x) \log_2 \frac{1}{q(x)} = -\sum_x p(x)\log_2 q(x)$$

Same shape as entropy — only the probability inside the log changed hands. In practice we never know the true $p$ of English, so real text stands in for it: average the model's surprise over a held-out test text of $N$ words,

$$H \approx -\frac{1}{N}\sum_{i=1}^{N} \log_2 P_{\text{model}}(w_i \mid \text{context}_i) \quad \text{(bits per word)}$$

A tiny example. True weather: rain 50%, sun 50% — entropy $= 1$ bit. A model that also believes 50/50 has cross-entropy $= 1$ bit. A model that believes rain 25%, sun 75% has cross-entropy $= 0.5\log_2\frac{1}{0.25} + 0.5\log_2\frac{1}{0.75} \approx 1.21$ bits — it pays extra surprise on the rainy days it kept betting against. In general:

$$\text{cross-entropy} = \underbrace{\text{entropy}}_{\text{how unpredictable reality is}} + \underbrace{\text{extra}}_{\text{how wrong the model is}}$$

so cross-entropy $\geq$ entropy, with equality only for a perfect model. Training any language model — bigram counter or GPT — is nothing but shrinking that second term; the first is a fixed property of language. One-line version: **entropy is how surprising the world is; cross-entropy is how surprising the world is *to your model*.**

Keep this quantity in mind: **perplexity (§7) is just cross-entropy exponentiated.** If cross-entropy is fuzzy, perplexity will be too.

### 4.4 Language as a Markov source: the n-th order approximations

Shannon's key modeling move: treat language as a **stochastic source emitting symbols one at a time, each with a probability that depends on preceding symbols**. He then built a ladder of approximations to English:

| Order              | Next symbol depends on       | Sample behavior                               |
| ------------------ | ---------------------------- | --------------------------------------------- |
| 0th                | nothing (uniform)            | `XFOML RXKHRJFFJUJ` — random junk             |
| 1st                | letter frequencies only      | `OCRO HLI RGWR` — English-ish letter mix      |
| 2nd                | previous 1 letter (bigram)   | `ON IE ANTSOUTINYS` — pronounceable fragments |
| 3rd                | previous 2 letters (trigram) | `IN NO IST LAT WHEY` — almost word-like       |
| word-level 1st/2nd | previous words               | locally grammatical phrases                   |

Each step up the ladder conditions on **more history** and produces text that looks more like English. This is the founding demo of language modeling — and note that a modern LLM sampling tokens is exactly this experiment with a context of thousands of tokens instead of two letters.

## 5. Church & Mercer (1993): the empiricist manifesto

_Computational Linguistics Using Large Corpora_ is not a formula paper — it's a **position piece** (the introduction to a special issue of _Computational Linguistics_) marking the field's official turn back to empiricism. Its argument:

- Large corpora (millions → billions of words) are now available in machine-readable form.
- Probabilities of linguistic events can therefore be **estimated from real data** rather than stipulated by theory.
- Data-driven methods were already outperforming rule-based ones in speech and beginning to in translation and tagging.

Its contribution is the _argument and the survey of evidence_, not an equation. Cite it as the moment statistical NLP became the mainstream, not as the source of the n-gram formula (that machinery is older — Shannon, and Markov before him).

## 6. Jelinek (1997): statistics meets a real problem — speech recognition

_Statistical Methods for Speech Recognition_ documents the IBM approach that made all of this practical. Speech recognition is the perfect showcase for why language models must exist:

> Acoustically, **"recognize speech"** and **"wreck a nice beach"** are nearly identical. No amount of audio analysis can fully separate them. Only knowledge of _which word sequence is more plausible English_ can.

### 6.1 The noisy-channel / Bayes decomposition

Given audio $A$, find the word sequence $W$ maximizing $P(W \mid A)$. By Bayes' rule:

$$\hat{W} = \arg\max_W P(W \mid A) = \arg\max_W \underbrace{P(A \mid W)}_{\text{acoustic model}} \cdot \underbrace{P(W)}_{\text{language model}}$$

($P(A)$ is constant across candidates, so it drops out.) Two independent components:

- **Acoustic model** $P(A \mid W)$: _does the audio sound like these words?_
- **Language model** $P(W)$: _are these words likely English?_ — this is our n-gram model.

This decomposition — a task-specific likelihood times a reusable prior over language — became the template for statistical machine translation and beyond. The LM is the reusable part, which is why it became the center of the field.

### 6.2 Hidden Markov Models (brief)

The acoustic side used **HMMs**: hidden states (phonemes/words) with a **transition matrix** (probabilities between hidden states) and an **emission matrix** (probability each state produces the observed acoustic features). Efficient algorithms (Viterbi, forward-backward) find the best hidden sequence. HMMs get a full treatment later in the roadmap; here they matter as the vehicle that proved statistical methods on a real, hard, commercially important task.

## 7. The math of n-gram language models

Now the machinery we'll implement.

### 7.1 Chain rule: the exact decomposition

The probability of a sentence $w_1, w_2, \ldots, w_n$ decomposes _exactly_ — no approximation yet:

$$P(w_1 w_2 \cdots w_n) = P(w_1)\,P(w_2 \mid w_1)\,P(w_3 \mid w_1 w_2)\cdots P(w_n \mid w_1 \cdots w_{n-1}) = \prod_{i=1}^{n} P(w_i \mid w_1^{\,i-1})$$

Problem: the final factors condition on arbitrarily long histories. Almost every long history occurs zero or one times in any corpus — the counts don't exist.

### 7.2 The Markov assumption: what we keep, what we throw away

The **n-gram assumption**: the next word depends only on the previous $n{-}1$ words.

$$\text{bigram } (n{=}2): \quad P(w_i \mid w_1^{\,i-1}) \approx P(w_i \mid w_{i-1})$$
$$\text{trigram } (n{=}3): \quad P(w_i \mid w_1^{\,i-1}) \approx P(w_i \mid w_{i-2}\, w_{i-1})$$

**What this throws away:** everything beyond the window — long-range grammar agreement, topic, discourse, who "she" refers to. In _"The keys to the cabinet **are** on the table"_, a bigram model chooses between _are/is_ seeing only "cabinet" — and prefers the wrong one. This single flaw is the thread that runs through the whole roadmap: RNNs, LSTMs, and attention are all attempts to widen this window without the count table exploding.

### 7.3 Estimation: maximum likelihood = counting

Estimate the conditional probabilities from corpus counts:

$$P_{\text{MLE}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}\, w_i)}{C(w_{i-1})}$$

That's the whole training algorithm: count bigrams, divide by unigram counts. (Sentences get padded with boundary markers `<s>` and `</s>` so first/last words have proper contexts.)

### 7.4 The zero-count problem — why smoothing exists

Take a perfectly ordinary test sentence containing a word pair that never appeared in training. MLE assigns that bigram probability **0**, so the whole sentence gets probability 0 — and perplexity, which involves $\log P$, becomes **infinite**. One unseen pair destroys the entire evaluation. And unseen pairs are not rare: language is heavy-tailed (Zipf), so _most_ possible bigrams never occur in any finite corpus, including a huge number of perfectly good ones.

**Smoothing** redistributes a little probability mass from seen events to unseen ones:

- **Laplace (add-one):** pretend every bigram occurred once more than it did:
  $$P_{\text{Lap}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1} w_i) + 1}{C(w_{i-1}) + V}$$
  where $V$ is vocabulary size. Simple, and what we'll implement — but crude: with $V$ in the tens of thousands it steals far too much mass from seen events. (Add-k with $k < 1$ softens this.)
- **Interpolation:** mix trigram, bigram, and unigram estimates with learned weights.
- **Backoff:** use the trigram if seen, else fall back to the bigram, else the unigram.
- **Kneser-Ney:** the classical state of the art — subtracts a fixed discount from every count and bases fallback on how many _distinct contexts_ a word appears in (why "Francisco" is common yet a bad generic prediction). We'll meet it properly when we need it.

Deep insight worth stating in the article: **smoothing is generalization.** A language model's real job is assigning reasonable probability to sequences it has _never seen_. N-grams generalize by crude count adjustment; neural models will do it by learning that "cat" and "dog" are similar. Same problem, better machinery.

### 7.5 Entropy → cross-entropy → perplexity

How good is a language model? Measure how surprised it is by held-out text it has never seen. The **cross-entropy** of the model on a test sequence of $N$ words:

$$H = -\frac{1}{N}\sum_{i=1}^{N} \log_2 P_{\text{model}}(w_i \mid \text{context}_i) \quad \text{(bits per word)}$$

**Perplexity** exponentiates it back to a human-readable scale:

$$\text{PPL} = 2^{H} = P_{\text{model}}(w_1 \cdots w_N)^{-1/N}$$

**Plain-words meaning:** perplexity is the model's **effective branching factor** — "on average, the model is as confused as if it were choosing uniformly among PPL equally likely next words."

- PPL 100 → as uncertain as a fair 100-sided die at every word.
- PPL 50 → the model has narrowed each choice to ~50 effective options. **Lower is better.**
- Sanity checks: a uniform model over a $V$-word vocabulary has PPL exactly $V$; a perfect oracle has PPL 1.

Reference points: trigram models on Penn Treebank sat around PPL ~140 in the 1990s; modern LLMs score in the single digits to low tens (numbers are only comparable on the same corpus + tokenization). This is the metric on which sixty years of progress is plotted — and we'll compute it ourselves next.

## 8. Limitations of n-gram models → the road ahead

1. **Sparsity / no generalization.** "The cat sat" in training tells the model _nothing_ about "the dog sat" — words are atomic symbols with no notion of similarity. → **Embeddings** (Days 4–6) make similar words share statistical strength.
2. **Fixed, tiny context.** The Markov window can't capture long-range dependencies, and the count table grows exponentially ($V^n$) with window size. → **RNNs/LSTMs** (Days 7–8) compress unbounded history into a state vector; **attention/Transformers** (Days 10–18) let every word look at every other word directly.
3. **No semantics.** N-grams model _co-occurrence_, not meaning — fluent locally, incoherent globally. → the entire neural era.

But note what _doesn't_ change after Day 1: the definition of the LM, the chain-rule factorization, the cross-entropy objective, and perplexity. We are upgrading the estimator of $P(w_i \mid \text{context})$, never the problem statement.

## 9. Glossary (only what Day 1 uses)

| Term                  | Definition                                                                                                  |
| --------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Corpus**            | A body of real text used for counting/training (plural: corpora). E.g., the Brown Corpus (~1M words, 1961). |
| **Token**             | A unit of text after splitting — here, a word; in modern LLMs, a subword.                                   |
| **N-gram**            | A contiguous sequence of n tokens. "the cat" is a bigram.                                                   |
| **Language model**    | A function assigning probability to a word sequence / predicting the next-word distribution.                |
| **Entropy**           | Average surprise of a source, in bits: $-\sum p \log_2 p$.                                                  |
| **Cross-entropy**     | Average surprise of _our model_ on real data; the training loss of every LM since.                          |
| **Perplexity**        | $2^{\text{cross-entropy}}$; effective number of equally likely next-word choices. Lower is better.          |
| **Smoothing**         | Redistributing probability mass to unseen events so they don't get probability 0.                           |
| **Markov assumption** | Next word depends only on the previous $n-1$ words.                                                         |

_(Embeddings, neural networks, transformers, parsing, etc. are deliberately excluded — each gets its own day in this roadmap.)_

## 10. References

1. Shannon, C. E. (1948). _A Mathematical Theory of Communication._ Bell System Technical Journal, 27. — information, entropy, n-th order approximations of English. (See also Shannon 1951, _Prediction and Entropy of Printed English_.)
2. Church, K. & Mercer, R. (1993). _Introduction to the Special Issue on Computational Linguistics Using Large Corpora._ Computational Linguistics, 19(1). — the empiricist turn.
3. Jelinek, F. (1997). _Statistical Methods for Speech Recognition._ MIT Press. — noisy channel, HMMs, LMs in production.
4. Jurafsky, D. & Martin, J. H. _Speech and Language Processing_ (3rd ed. draft), Ch. 3 "N-gram Language Models" — the canonical textbook treatment; free online.

---

**Next:** build the bigram/trigram model with Laplace smoothing, train on a real corpus, and watch every section above turn into a line of code — the Markov assumption becomes a count dictionary, the zero-count problem becomes a `KeyError`, and perplexity becomes three lines of NumPy.
