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

**Worked numbers** (all from $I = \log_2 \frac{1}{p}$):

| Event | $p$ | Self-information |
| ----- | --- | ---------------- |
| something certain | $1$ | $0$ bits |
| fair coin lands heads | $1/2$ | $1$ bit |
| a die rolls 5 | $1/6$ | $\approx 2.6$ bits |
| a specific card drawn from a deck | $1/52$ | $\approx 5.7$ bits |
| winning a 1-in-a-million lottery | $10^{-6}$ | $\approx 19.9$ bits |

Now in language: after the letter `q`, English produces `u` with probability $\approx 0.97$ — seeing it costs $\log_2\frac{1}{0.97} \approx 0.04$ bits, almost no news at all. Seeing a rare word like *aardvark* ($p \approx 1/100{,}000$) costs $\approx 16.6$ bits. **Predictable symbols are cheap, surprising ones are expensive** — and language is mostly cheap symbols, which is why it's compressible and predictable.

### 4.2 Entropy: the average surprise of a source

**Entropy** is the _expected_ self-information over all events the source can produce:

$$H = \sum_x p(x) \log_2 \frac{1}{p(x)} = -\sum_x p(x)\log_2 p(x)$$

Entropy measures how unpredictable a source is, in bits per symbol. A fair coin: $H = 1$ bit. A two-headed coin: $H = 0$. English text: Shannon estimated roughly ~1 bit per letter — far below the $\log_2 27 \approx 4.75$ bits of random letters, which is a _measurement_ of how redundant and predictable language is.

**Worked example — same outcomes, different predictability.** A city's weather has four outcomes: sun $\tfrac{1}{2}$, rain $\tfrac{1}{4}$, snow $\tfrac{1}{8}$, hail $\tfrac{1}{8}$. Each term of the sum is (probability) × (that outcome's self-information):

$$H = \tfrac{1}{2}(1) + \tfrac{1}{4}(2) + \tfrac{1}{8}(3) + \tfrac{1}{8}(3) = 1.75 \text{ bits}$$

Sun is a cheap 1-bit event that happens half the time; hail is an expensive 3-bit surprise but rare — entropy is the average bill. Compare: if all four outcomes were equally likely, $H = \log_2 4 = 2$ bits, the maximum possible. Skewing the distribution toward predictable outcomes dropped the entropy from 2 to 1.75 **with the exact same set of outcomes**. Same effect on a biased coin: fair coin $H = 1$ bit, but a 90/10 coin gives $H = 0.9\log_2\frac{1}{0.9} + 0.1\log_2\frac{1}{0.1} \approx 0.47$ bits — still two outcomes, half the uncertainty.

**What entropy really means: the guessing game.** The formula becomes concrete once you know what a "bit" physically is: **one bit = the answer to one perfect yes/no question.** So entropy is simply **the average number of yes/no questions you need to guess the source's next symbol.** That's the whole meaning. Three steps ground Shannon's "~1 bit per letter of English" claim:

**Step 1 — where 4.75 comes from.** Imagine letters produced by a lottery machine: 26 letters plus space, all equally likely, no memory of what came before. To identify the next symbol, your best strategy is to halve the candidates with every question: "Is it in A–M?" → "Is it in A–F?" → ... With 27 possibilities that takes $\log_2 27 \approx 4.75$ questions. So **4.75 bits/letter is the uncertainty of pure gibberish** — the worst case, where nothing helps you guess.

**Step 2 — but English is not a lottery machine.** Play the same game on real text. You're reading:

> The cat sat on the ma\_

How many questions do you need? Essentially zero — "Is it *t*?" → yes, done. Or:

> q\_

One question: "Is it *u*?" → yes, ~97% of the time. Meanwhile a genuinely open position — the first letter of a brand-new sentence — might cost 3–4 questions. Entropy is the **average over all positions**, and for English that average lands around **1 question per letter**, not 4.75.

**Step 3 — so "eliminates ~80% of the uncertainty" means:** before the next letter arrives there are notionally 27 candidates — 4.75 questions' worth of doubt. But spelling rules, grammar, and word habits have already killed most candidates *for free*: after `q` almost nothing survives; after "the ma" in that sentence, everything but *t*, *n*, *p* is dead. Context did the questioning for you. Of the 4.75 questions gibberish would require, context pre-answers ~3.75, leaving you ~1 to actually ask — and $3.75/4.75 \approx 80\%$.

**Why this matters beyond the history lesson:**

- **Redundancy is why compression works.** English needs ~1 bit/letter but ASCII spends 8 — that gap is exactly why ZIP shrinks text about 5×. It's also why "u cn stll rd ths": the deleted letters carried almost no information, so context reconstructs them.
- **Redundancy is why language models are possible at all.** A language model is a machine for exploiting this predictability. If English really were 4.75 bits/letter — a lottery machine — next-word prediction would be impossible and this entire roadmap wouldn't exist. Entropy ≈ 1 bit/letter is the *quantitative proof* that "predict the next symbol" is a learnable task.

**How ~1 bit/letter was actually measured: the staircase of estimates.** Nobody knows the true probability distribution of English, so its entropy can't be computed directly. Shannon's move: compute the entropy of the next letter conditioned on **increasing amounts of context**. Each extra scrap of context can only reduce uncertainty, so the estimates form a descending staircase whose limit is the true entropy:

| Estimate | Conditions on | Bits/letter |
| -------- | ------------- | ----------- |
| $F_0$ | nothing (uniform over 27 symbols) | 4.75 |
| $F_1$ | letter frequencies | 4.14 |
| $F_2$ | previous 1 letter | 3.56 |
| $F_3$ | previous 2 letters | 3.3 |
| word frequencies | whole words | ~2.6 |
| human reader (1951) | ~100 letters of context | **0.6 – 1.3** |

The staircase is clearly still falling at $F_3$, but count tables couldn't go deeper (context of 10 letters needs $27^{10}$ entries — and this was computed by hand from books). So Shannon swapped the count table for the best available model of English: **a human brain**. In his 1951 follow-up (*Prediction and Entropy of Printed English*), subjects guessed the next letter of a hidden text — with the preceding ~100 letters visible — until correct, and the guess counts (mostly 1, 1, 1, 2, 1, ... with occasional hard spots at new words) convert into the entropy bounds in the last row. The entropy of English was literally measured by humans playing hangman.

Two things to notice, because they run through this whole roadmap. First, each staircase row is a *model* of English, and its number is that model's average surprise — an **upper bound** on the true entropy that tightens as the model improves (this is exactly cross-entropy, defined next in §4.3). Second, the staircase is still being descended *today*: text compressed with a modern LLM reaches ~0.6–0.7 bits per character, hugging the bottom of Shannon's 1951 human range. Uniform → unigram → bigram → trigram → human → GPT is one continuous plot on an axis Shannon drew — and the bigram model we build in §7 sits on a known step of it.

One caution: entropy is a property of **language itself** — the irreducible unpredictability no model can beat. To score a *model*, we need its sibling.

### 4.3 Cross-entropy: the guessing game with the wrong beliefs

In §4.2's guessing game there was a hidden assumption: you knew the true probabilities, so you always asked the *smartest possible questions* — "sun?" first, because sun is genuinely the most common. Entropy is the average number of questions **when your beliefs are perfect**.

But a model's beliefs are never perfect. **Cross-entropy is the average number of questions you need when reality deals the outcomes, but your question strategy is built from *your model's* beliefs.** Wrong beliefs → you ask about the wrong things first → you waste questions. That's the entire concept.

**Same weather, wrong model.** Reality (from §4.2): sun $\tfrac{1}{2}$, rain $\tfrac{1}{4}$, snow $\tfrac{1}{8}$, hail $\tfrac{1}{8}$ — entropy 1.75 questions/day. Now suppose your model believes the *exact reverse*: hail $\tfrac{1}{2}$, snow $\tfrac{1}{4}$, rain $\tfrac{1}{8}$, sun $\tfrac{1}{8}$. Trusting it, you ask "hail?" first, then "snow?", then "rain?":

| Day's actual weather | How often (truth) | Your questions | Cost |
| -------------------- | ----------------- | -------------- | ---- |
| hail | $\tfrac{1}{8}$ | "hail?" ✓ | 1 |
| snow | $\tfrac{1}{8}$ | "hail?" ✗ "snow?" ✓ | 2 |
| rain | $\tfrac{1}{4}$ | "hail?" ✗ "snow?" ✗ "rain?" ✓ | 3 |
| sun | $\tfrac{1}{2}$ | "hail?" ✗ "snow?" ✗ "rain?" ✗ → sun | 3 |

Your cheap 1-question shortcut is reserved for hail — which almost never happens — while sun, *half of all days*, costs you 3 questions every time. Average:

$$H(p, q) = \tfrac{1}{2}(3) + \tfrac{1}{4}(3) + \tfrac{1}{8}(2) + \tfrac{1}{8}(1) = 2.625 \text{ questions/day}$$

Reality is only 1.75 questions unpredictable; you're paying 2.625. The extra **0.875 questions/day is pure cost of believing the wrong thing.** Note what each side contributes: **reality decides how often each row happens; your model decides how expensive each row is.**

That sentence *is* the formula. Reality's probability $p(x)$ weights each outcome; your model's belief $q(x)$ sets its cost, $\log_2 \frac{1}{q(x)}$ (believe strongly → cheap when right; doubt → expensive):

$$H(p, q) = \sum_x p(x) \log_2 \frac{1}{q(x)}$$

Compare entropy, $\sum_x p(x)\log_2\frac{1}{p(x)}$ — identical, except there the questioner's beliefs *are* the truth. The "cross" is that the probability inside the log changed hands: reality picks, your model pays.

For language models one adjustment: we never know the true $p$ of English, so real text stands in for it — let reality deal actual words from a held-out test set and average your model's cost over $N$ of them:

$$H \approx -\frac{1}{N}\sum_{i=1}^{N} \log_2 P_{\text{model}}(w_i \mid \text{context}_i) \quad \text{(bits per word)}$$

In general:

$$\text{cross-entropy} = \underbrace{\text{entropy}}_{\text{how unpredictable reality is}} + \underbrace{\text{extra}}_{\text{how wrong your beliefs are}}$$

so cross-entropy $\geq$ entropy, with equality only for a perfect model. Training any language model — bigram counter or GPT — is nothing but shrinking that second term; the first is a fixed property of language. One-line version: **entropy is how many questions the world costs; cross-entropy is how many questions the world costs *you, with your beliefs* — and the gap is exactly how wrong your beliefs are.**

**The same computation on language** — this is exactly what we'll code in §7. A trained bigram model reads the test sentence `<s> the cat sat </s>`. At each step reality reveals the true next word, and we look up the probability the model *had assigned* to it:

| Context | True next word | $P_{\text{model}}$ | Surprise $-\log_2 P$ |
| ------- | -------------- | ------------------ | -------------------- |
| `<s>` | the | 0.25 | 2 bits |
| the | cat | 0.125 | 3 bits |
| cat | sat | 0.5 | 1 bit |
| sat | `</s>` | 0.25 | 2 bits |

Cross-entropy = the average of the surprise column = $\frac{2+3+1+2}{4} = 2$ bits/word. Step 3 is the model's best moment (after "cat" it strongly expected "sat" — barely surprised); step 2 its worst. A better model is simply one that puts higher probability on the words that actually occur, paying fewer bits per word. And spot the trap: had the model assigned "sat" probability **0**, that row would cost $\infty$ bits and no other row could compensate — hold that thought for smoothing (§7.4).

Keep this quantity in mind: **perplexity (§7) is just cross-entropy exponentiated.** If cross-entropy is fuzzy, perplexity will be too.

### 4.4 Language as a Markov source: the n-th order approximations

So far the models have only *judged* text — guessing letters, paying bits. Shannon's last demo flips them around: **if a model can assign probabilities to what comes next, it can also *write* — just roll dice according to those probabilities.** This section is where language models stop being scoring functions and start generating.

**First, the name.** A **Markov source** is a random text generator with deliberately amputated memory: it emits one symbol at a time, and its choice depends only on the **last $n-1$ symbols** — everything earlier is forgotten. (After A. A. Markov, who in 1913 hand-counted vowel/consonant patterns in Pushkin's *Eugene Onegin* — the first statistical language analysis ever.) An "$n$-th order approximation of English" is a Markov source whose probabilities were **estimated by counting real English text**: tally which symbol follows which context, convert to frequencies, done. Notice that's the first appearance of the modern pipeline — *count a corpus, then sample from the counts* — i.e., **training** and **generation**.

**The experiment.** Shannon built these sources at increasing order and let each one write. Read the samples column top to bottom — this is the founding demo of language modeling:

| Order              | Next symbol depends on       | Generated sample                              |
| ------------------ | ---------------------------- | --------------------------------------------- |
| 0th                | nothing (all 27 equally likely) | `XFOML RXKHRJFFJUJ` — random junk             |
| 1st                | letter frequencies only      | `OCRO HLI RGWR` — right letter *mix*, no structure |
| 2nd                | previous 1 letter            | `ON IE ANTSOUTINYS` — pronounceable fragments |
| 3rd                | previous 2 letters           | `IN NO IST LAT WHEY` — almost word-like       |
| word-level 1st/2nd | previous word(s)             | locally grammatical phrases that drift off-topic |

Each row remembers one symbol more than the last, and the output gets visibly more English-like. Two things to notice:

- **This is §4.2's staircase, run in reverse.** Same models, two modes: point them at existing text and they *score* it (the staircase of entropy estimates); let them roll dice and they *generate*. Better scorer ⇔ better writer — one competence, two directions. That equivalence still holds: an LLM's perplexity and the quality of its generations rise and fall together.
- **Fluency is local; meaning is not.** The word-level samples read fine within any 3-word window yet drift globally — because the source *has no memory beyond its window*. Every failure of n-gram models (§8) is this row of the table, and every architecture after it (RNNs, attention) is an attempt to extend the window.

**Do it yourself at word level.** Take a three-sentence corpus: *"the cat sat"*, *"the cat ran"*, *"the dog sat"*. Training = counting what follows each word:

- after `the`: cat $\tfrac{2}{3}$, dog $\tfrac{1}{3}$
- after `cat`: sat $\tfrac{1}{2}$, ran $\tfrac{1}{2}$
- after `dog`: sat $1$

Generation = start at `the`, roll a die weighted by the counts, emit the word, move to it, repeat. You might get *"the dog sat"* — or *"the cat ran"*; both are fluent, and the die decides. This tiny table **is** a 1st-order word approximation, built with nothing but counting — precisely the machinery we formalize in §7 and implement in code. And an LLM sampling its reply to you is this same loop with a vastly bigger (implicit) table and thousands of words of context instead of one: Shannon's dice, still rolling.

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

### 6.2 Hidden Markov Models

**The problem HMMs solve.** A Markov source (§4.4) is fine when you can *see* the sequence you're modeling. But in speech recognition you never observe the words — you observe sound waves. The sequence you care about (words) is **hidden**; the sequence you have (acoustic measurements) is a noisy, ambiguous *consequence* of it. A **Hidden Markov Model** is exactly this two-layer story:

1. **A hidden layer** walks from state to state like an ordinary Markov chain — you never see it.
2. **A visible layer**: at each step, the current hidden state *emits* an observation with some probability — this is all you ever see.

The machine's job: given only the observations, reason backwards to the hidden path that best explains them.

**A hand-sized example.** You work in a windowless office and can't see the weather (hidden states: `sun`, `rain`). All you observe is whether each arriving coworker carries an umbrella (observations: `umbrella`, `none`). Two probability tables define everything:

*Transition matrix* — how the hidden world moves (weather has momentum):

| from \ to | sun | rain |
| --------- | --- | ---- |
| **sun** | 0.8 | 0.2 |
| **rain** | 0.4 | 0.6 |

*Emission matrix* — how the hidden state leaks into what you can see:

| state \ emits | umbrella | none |
| ------------- | -------- | ---- |
| **sun** | 0.2 | 0.8 |
| **rain** | 0.9 | 0.1 |

(Plus a start distribution: day 1 is sun or rain with probability 0.5 each.)

**Inference, by brute force.** Two coworkers arrive on two mornings, both with umbrellas: observations = (`umbrella`, `umbrella`). What was the weather? Score every possible hidden path as (start × emission × transition × emission):

| Hidden path | Probability | |
| ----------- | ----------- | - |
| sun → sun | $0.5 \cdot 0.2 \cdot 0.8 \cdot 0.2$ | $= 0.016$ |
| sun → rain | $0.5 \cdot 0.2 \cdot 0.2 \cdot 0.9$ | $= 0.018$ |
| rain → sun | $0.5 \cdot 0.9 \cdot 0.4 \cdot 0.2$ | $= 0.036$ |
| rain → rain | $0.5 \cdot 0.9 \cdot 0.6 \cdot 0.9$ | $= \mathbf{0.243}$ |

Verdict: rain both days, and it isn't close. Notice *why* it wins: rain explains the umbrellas well (high emissions, 0.9) **and** rain-then-rain is a plausible weather story (transition 0.6). A hidden path must satisfy both masters — sound like the evidence *and* be a likely story on its own. Keep that sentence; it's about to become speech recognition.

**Why brute force dies, and the three classical algorithms.** With 2 states and 2 steps there were $2^2 = 4$ paths. Real speech: thousands of states, hundreds of time steps — $S^T$ paths, more than atoms in the universe. The reason HMMs conquered the field is that all three natural questions have **dynamic-programming** answers that run in $O(T \cdot S^2)$ — the trick being that paths sharing a state share their future, so you compute each state's best/total score once per time step instead of once per path (these are the three problems of Rabiner's famous 1989 tutorial):

| Question | Algorithm | Used for |
| -------- | --------- | -------- |
| How likely are these observations at all? $P(O)$ | **Forward** (sum over paths) | scoring/comparing models |
| What's the single best hidden path? | **Viterbi** (max over paths) | *decoding* — recognition itself |
| How do I learn the two matrices without labeled data? | **Baum–Welch** (EM; uses forward–backward) | training |

Viterbi is just the brute-force table above, kept pruned: at each time step, for each state, remember only the best path that reaches it — every extension of a losing path loses, so nothing of value is discarded.

**Now the payoff: this is §6.1's equation as a machine.** Map the pieces onto speech:

- **Hidden states** = the word/phoneme sequence $W$ being spoken
- **Transitions** = which words plausibly follow which — *this is exactly the n-gram language model $P(W)$* from this article
- **Emissions** = what sounds each word produces — the acoustic model $P(A \mid W)$
- **Recognition** = Viterbi decoding: find the hidden word path that both *sounds like the audio* (emissions) and *reads like English* (transitions)

"Recognize speech" beats "wreck a nice beach" not because it sounds different — the emissions nearly tie — but because the transition side (the language model) says the first is a far likelier walk through English. The two-masters structure of the umbrella example *is* the noisy-channel equation, running in real time.

**Where this thread goes.** HMMs ruled speech recognition and sequence labeling (POS tagging, named entities) for three decades — a hidden cause, a Markov walk, and noisy evidence turn out to describe half of NLP. And file away the deeper idea: *a hidden state that summarizes the past and carries it forward one step at a time*. Strip away the probability tables, make that state a learned vector, and you have the RNN — Day 7. The architecture of the future is hiding inside 1980s speech recognition.

## 7. The math of n-gram language models

Everything so far becomes one buildable machine here. The story of this section, in one paragraph: we want the probability of a whole sentence (7.1) — but computed exactly, it needs statistics nobody can collect, so we take the Markov shortcut (7.2) — which makes training literally just counting (7.3) — but counting has a fatal flaw on unseen word pairs (7.4) — and once that's patched, we grade the machine with perplexity (7.5).

One running example throughout, the mini-corpus from §4.4: *"the cat sat"*, *"the cat ran"*, *"the dog sat"*. We will train on it by hand, break the model with a real sentence, fix it, and score it.

### 7.1 Chain rule: the exact decomposition

A language model must assign a probability to a whole sentence. The chain rule says: **peel off one word at a time, and multiply the probability of each word given everything before it.** For "the cat sat":

$$P(\text{the cat sat}) = P(\text{the}) \cdot P(\text{cat} \mid \text{the}) \cdot P(\text{sat} \mid \text{the cat})$$

Read it as a story: how likely is a sentence to start with "the"? — times — given it started with "the", how likely is "cat" next? — times — given "the cat", how likely is "sat"? In general:

$$P(w_1 w_2 \cdots w_n) = \prod_{i=1}^{n} P(w_i \mid w_1 \cdots w_{i-1})$$

Important: **this is not an approximation.** It's pure algebra — true for any distribution, nothing assumed yet. We've just rewritten "probability of a sentence" as "probability of each next word given its full history" — which is pleasing, because *predict the next word given history* is exactly what we've called a language model since §3.

**So where's the problem?** In the last factor's condition. To estimate $P(\text{sat} \mid \text{the cat})$ from data, you need to find "the cat" many times in your corpus and check what followed. Fine for a 2-word history. But the chain rule demands this for the *full* history — and by word 15 the condition is a 14-word phrase. Try this: google any exact 15-word sentence from today's newspaper, in quotes. Zero results. Nearly every long word sequence in existence has **never been written before** — so the counts we'd need are 0 out of 0. The exact formula is correct and unusable.

### 7.2 The Markov assumption: the shortcut that makes it possible

The fix is the same amputated memory from §4.4, now stated as an approximation: **pretend the next word depends only on the last $n-1$ words, and forget everything earlier.**

$$\text{bigram } (n{=}2): \quad P(\text{sat} \mid \text{the cat}) \;\approx\; P(\text{sat} \mid \text{cat})$$
$$\text{trigram } (n{=}3): \quad P(w_i \mid \text{everything before}) \;\approx\; P(w_i \mid w_{i-2}\, w_{i-1})$$

**What we gain:** the conditions are now short — and short phrases, unlike 14-word phrases, occur over and over in any decent corpus. "the cat" appears twice even in our 9-word toy corpus. The statistics we need suddenly exist.

**What we throw away:** everything beyond the window — long-range grammar, topic, who "she" refers to. Concrete casualty: in *"The keys to the cabinet **are** on the table"*, the verb must agree with "keys", nine characters and three words back. A bigram model chooses between *are/is* seeing only "cabinet" — and confidently prefers the wrong one. It's the "locally fluent, globally lost" behavior we saw in Shannon's word-level samples (§4.4), now with a mechanism attached.

Hold on to this trade: **the entire rest of the roadmap — RNNs, LSTMs, attention — is a series of attempts to widen this window without the statistics falling apart.**

### 7.3 Training: maximum likelihood = counting

How do we get a number for $P(\text{cat} \mid \text{the})$? Ask the corpus the obvious question: **"of all the times I saw 'the', what fraction were followed by 'cat'?"**

$$P_{\text{MLE}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}\, w_i)}{C(w_{i-1})} \qquad \text{(} C(\cdot) = \text{count in corpus)}$$

Two practical details first. We pad every sentence with boundary markers — `<s> the cat sat </s>` — so the first word has a context to be predicted from, and the model can learn where sentences *end*. Now train on our corpus by hand — count every adjacent pair:

| Bigram | Count | | Context | Count |
| ------ | ----- | - | ------- | ----- |
| `<s>` the | 3 | | `<s>` | 3 |
| the cat | 2 | | the | 3 |
| the dog | 1 | | cat | 2 |
| cat sat | 1 | | dog | 1 |
| cat ran | 1 | | sat | 2 |
| dog sat | 1 | | ran | 1 |
| sat `</s>` | 2 | | | |
| ran `</s>` | 1 | | | |

Divide, and the model is trained: $P(\text{cat} \mid \text{the}) = \tfrac{2}{3}$, $P(\text{dog} \mid \text{the}) = \tfrac{1}{3}$, $P(\text{sat} \mid \text{cat}) = \tfrac{1}{2}$, $P(\text{the} \mid \texttt{<s>}) = 1$. That's it. **"Training a language model" in 1990 meant: count pairs, divide.** No gradients, no epochs — one pass over the corpus.

Why the fancy name "maximum likelihood"? Because this dividing-counts recipe is provably the table of probabilities that makes your training corpus *as probable as possible* — no other assignment of numbers scores the training data higher. The corpus is the evidence; MLE is the model that fits the evidence most literally. And "most literally" is exactly what goes wrong next.

### 7.4 The zero-count problem — why smoothing exists

Score a brand-new test sentence with our trained model: **"the dog ran."** Reasonable English — dogs run. Peel and multiply:

$$P = \underbrace{P(\text{the} \mid \texttt{<s>})}_{= 1} \cdot \underbrace{P(\text{dog} \mid \text{the})}_{= 1/3} \cdot \underbrace{P(\text{ran} \mid \text{dog})}_{= \mathbf{0/1 = 0}} \cdot \; \cdots \; = 0$$

"dog ran" never occurred in training, so the model declares the whole sentence **impossible** — probability exactly 0. Not unlikely: *impossible*. And it poisons everything it touches: the sentence's log-probability is $\log 0 = -\infty$, so cross-entropy and perplexity over the *entire test set* become infinite. One missing pair, total evaluation wipeout — this is the trap we flagged in the §4.3 scoring table, sprung.

You might hope unseen pairs are rare. The opposite: word frequencies are heavy-tailed (Zipf's law — a few words are very common, most words are very rare), so **the majority of all possible word pairs never occur in any finite corpus**, including countless perfectly grammatical ones. Every real test set will hit them. The model's true crime is *arrogance*: 0 means "I have seen everything, and this cannot happen," when the honest claim is "I haven't seen this yet."

**Smoothing** is enforced humility: shave a little probability off the pairs you did see, and spread it over the ones you didn't. The classical menu, simplest first:

- **Laplace (add-one)** — what we'll implement: pretend every possible pair occurred once more than it did. With vocabulary size $V$ (ours: the, cat, dog, sat, ran, `</s>` → $V = 6$):

  $$P_{\text{Lap}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1} w_i) + 1}{C(w_{i-1}) + V}$$

  Now $P(\text{ran} \mid \text{dog}) = \frac{0+1}{1+6} = \tfrac{1}{7}$ — small, but not zero. "The dog ran" is unlikely, not impossible. Fixed. But watch the cost: $P(\text{sat} \mid \text{dog})$ fell from $1$ to $\frac{1+1}{1+6} = \tfrac{2}{7}$. To fund the unseen, add-one taxed a *seen* event from certainty down to 29% — and with a realistic vocabulary ($V$ in the tens of thousands) the tax gets far worse. (Add-$k$ with $k < 1$ is the gentler version.)
- **Interpolation:** don't trust one estimate — blend them. Mix trigram, bigram, and unigram probabilities with weights: even if "dog ran" was never seen, "ran" on its own was.
- **Backoff:** use the longest context you've actually seen: trigram if available, else drop to bigram, else to unigram. In plain terms: *if you've never seen the pair, judge the word on its own reputation.*
- **Kneser-Ney** — the classical state of the art: subtract a fixed discount from every count, and when backing off, rank words not by raw frequency but by **how many different contexts** they follow. The classic illustration: "Francisco" is a frequent word, but nearly every occurrence follows "San" — so in a *fresh* context it's a terrible guess, and Kneser-Ney knows it. We'll meet it properly later in the roadmap.

The insight worth a pull-quote in your article: **smoothing is generalization.** A language model's real job is assigning sane probabilities to text it has *never seen*. N-grams generalize by crudely adjusting counts; neural models will do it by learning that "cat" and "dog" are similar, so evidence about one transfers to the other. Same problem — better machinery arrives on Day 4.

### 7.5 Perplexity: the scorecard

The model trains (7.3), survives unseen pairs (7.4) — now grade it. The grade we already own: **cross-entropy (§4.3)** — the model's average surprise, in bits per word, on held-out text it never trained on:

$$H = -\frac{1}{N}\sum_{i=1}^{N} \log_2 P_{\text{model}}(w_i \mid \text{context}_i) \quad \text{(bits per word)}$$

(Held-out is non-negotiable: scoring the model on its own training text is letting the student grade their own exam — our toy model above would look perfect on "the cat sat" and fall over on "the dog ran".)

**Perplexity** is that same number, made human-readable. Bits are logarithms; exponentiate to turn them back into a *count of options*:

$$\text{PPL} = 2^{H} = P_{\text{model}}(w_1 \cdots w_N)^{-1/N}$$

**Plain-words meaning:** perplexity is the model's **effective branching factor** — "at each word, the model is as confused as if it were choosing uniformly among PPL equally likely options." In the guessing-game language of §4.2: $H$ is questions per word, and $\text{PPL} = 2^H$ is the size of the candidate pool those questions have to whittle down.

Worked, from numbers we already have: the §4.3 scoring table gave a cross-entropy of 2 bits/word on `<s> the cat sat </s>` — so $\text{PPL} = 2^2 = 4$: that model navigates the sentence as if picking among 4 options per word. Sanity checks: a model that knows nothing (uniform over $V$ words) has $\text{PPL} = V$ exactly; a perfect oracle has $\text{PPL} = 1$; **lower is better**.

Reference points for calibration: 1990s trigram models on Penn Treebank sat around PPL ~140; modern LLMs reach the single digits to low tens (only comparable on the same corpus and tokenization). Sixty years of progress — Shannon's staircase (§4.2), the n-gram era, the entire neural age — plots as one descending curve of this single number. Next, we compute our own point on it.

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
