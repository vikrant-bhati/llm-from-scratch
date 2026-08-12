# Day 3 — Classical Text Classification: Naive Bayes, TF-IDF, Logistic Regression

---

## 1. Why this topic matters

Days 1 and 2 built a machine that **predicts the next word**. Today we build a different machine: one that **reads a document and puts it in a box.** Is this review positive or negative? Is this email spam? Is this article about sports or politics?

At first glance this looks like a detour from the roadmap. It is not, for four reasons — and each one is a thread we pull all the way to GPT.

1. **The loss function is the same one.** Logistic regression is trained by minimizing **cross-entropy** — the exact quantity Shannon defined in 1948 and we met on Day 1. We're not learning a new objective today. We're watching the old one get used a second way.
2. **We are literally building the last layer of every model in this series.** Strip a Transformer of its Transformer-ness and what remains at the output is a softmax over the vocabulary — a multi-class logistic regression on top of learned features. Today we build that classifier by hand on features we design ourselves. Everything from Day 7 onward keeps the classifier and replaces the hand-designed features with learned ones.
3. **This is the roadmap's first supervised learning.** Language modeling is *self-supervised*: the label is the next word, and text labels itself for free. Classification needs a human to say "this one is positive." That difference in where labels come from is the single biggest reason LLMs could be trained on the entire internet while classifiers were stuck at a few thousand hand-labeled examples. We'll come back to this.
4. **These are still the right tools.** A TF-IDF + logistic regression classifier trains in under a second, runs on a laptop, and is genuinely hard to beat on small labeled datasets. Reaching for a 7-billion-parameter model to sort 800 support tickets is a mistake this day is designed to prevent.

The plot of the day, in one line: **text becomes numbers by counting words, the counts get weighted by how surprising each word is, and then two classifiers — one that counts, one that learns — argue over the result.**

## 2. The task: classification vs. language modeling

A quick contrast, because the shift is easy to miss.

| | Language model (Days 1–2) | Classifier (Day 3) |
| --- | --- | --- |
| Input | the words so far | a whole document |
| Output | a probability for each next word | a probability for each class |
| Number of outputs | vocabulary size (~50,000) | usually 2–20 |
| Where labels come from | the text itself (free) | humans (expensive) |
| Trained by minimizing | cross-entropy | cross-entropy |
| Graded by | perplexity | precision, recall, F1 |

Read the last two rows carefully. **The training objective is identical; only the scorecard changed.** Perplexity made sense when you were guessing among 50,000 words. When there are two classes, "how confused is the model" is less useful than "when it says positive, is it right, and does it catch the positives that exist?" That's Section 8.

Formally, a classifier estimates $P(c \mid d)$ — the probability of class $c$ given document $d$ — and predicts the class with the highest one:

$$\hat{c} = \arg\max_{c \in C} P(c \mid d)$$

Everything today is a different way of getting at that one quantity.

## 3. The review corpus — our toy example

Just like the cat corpus (Day 1) and the hug corpus (Day 2), we need something small enough to compute entirely by hand. Meet the **review corpus**: six one-line movie reviews, three positive and three negative.

```text
POSITIVE                    NEGATIVE
D1: great film              D4: bad film
D2: great great story       D5: bad bad story
D3: good film               D6: dull film
```

The vocabulary is six words: `bad, dull, film, good, great, story`. So $V = 6$ and the number of documents $N = 6$.

It's deliberately symmetric — `great` mirrors `bad`, `good` mirrors `dull` — so that when the two classifiers disagree, we know the disagreement comes from the *method*, not from lopsided data. Every number in the rest of this document comes from these six lines, and you can check all of them on paper.

## 4. Step one: turning text into numbers

A classifier does arithmetic. Text is not arithmetic. Something has to bridge the two, and that bridge is called a **document representation** — a recipe for turning a document into a vector of numbers.

### 4.1 Bag of words

The oldest and still the most important idea:

> **Throw away word order. Just record which words appeared, and how many times.**

Make one column per vocabulary word, one row per document, and fill in the counts. Our corpus becomes a $6 \times 6$ matrix — the **document-term matrix**:

| | bad | dull | film | good | great | story |
| --- | --- | --- | --- | --- | --- | --- |
| **D1** great film | 0 | 0 | 1 | 0 | 1 | 0 |
| **D2** great great story | 0 | 0 | 0 | 0 | 2 | 1 |
| **D3** good film | 0 | 0 | 1 | 1 | 0 | 0 |
| **D4** bad film | 1 | 0 | 1 | 0 | 0 | 0 |
| **D5** bad bad story | 2 | 0 | 0 | 0 | 0 | 1 |
| **D6** dull film | 0 | 1 | 1 | 0 | 0 | 0 |

The name is the metaphor: tip the document into a bag, shake it, and all you have left is a pile of words with no arrangement. `"the movie was good, not bad"` and `"the movie was bad, not good"` produce **exactly the same vector.** That is a catastrophic-sounding flaw, and we will confirm it experimentally later — the two sentences really do get identical predictions, always, no matter how well the model is trained.

So why does anyone use this? Because for deciding *what a document is about*, the pile of words is usually enough. A document containing `refund`, `shipping`, and `damaged` is a complaint regardless of the order those words arrive in. Bag of words throws away syntax and keeps topic — and for classification, topic is most of the signal. We are trading away exactly the thing (word order) that Days 7 onward will spend enormous effort winning back.

Two immediate practical facts about this matrix:

- **It is enormous.** One column per distinct word in the corpus. On the real IMDb dataset we'll use, that's **74,849 columns**.
- **It is almost entirely zeros.** A movie review contains maybe 130 distinct words out of those 74,849. On IMDb, **99.8% of the matrix is zero.** This is called a **sparse** matrix, and it's stored as a list of the non-zero entries only — otherwise 25,000 × 74,849 numbers would not fit comfortably in memory. Sparsity isn't an inconvenience here; Section 7 shows it's the reason linear classifiers work so well on text.

### 4.2 Raw counts are a bad weighting

Use those counts directly and you hit a problem immediately. In any real corpus the highest counts belong to `the`, `of`, `is`, `and` — words that appear everywhere and therefore tell you **nothing** about which class a document belongs to. Meanwhile `superb`, which appeared twice, is enormously informative.

Raw counts get this exactly backwards: **the most frequent words are the least useful ones.**

The classical fix was a stop-word list — delete `the`, `of`, `is` by hand. That's the Day 2 rule-era mindset again, and it has the same problem: it's a hand-written list, it's language-specific, and it's crude (it deletes `not`, which matters enormously for sentiment). We want something that *learns* how much each word is worth from the data itself. Two ingredients do it.

### 4.3 Term frequency (TF): how much this document talks about it

The first ingredient is just the count of word $t$ in document $d$, written $\text{tf}(t,d)$. Hans Peter Luhn proposed using it in the late 1950s: words a document repeats are words that document is about.

One refinement matters. Is a word appearing 10 times really ten times as important as a word appearing once? Usually not — the second mention adds far less than the first. So it's standard to compress the count with a logarithm:

$$\text{tf}(t,d) = 1 + \log(\text{count}(t,d)) \quad \text{if count} > 0, \text{ else } 0$$

This is called **sublinear** or **log** term frequency, and it's one of the things Salton & Buckley tested (Section 4.6). We'll use raw counts in the hand-worked example to keep the arithmetic clean, and try both in the notebook.

### 4.4 Inverse document frequency (IDF) — which is Shannon's surprise again

The second ingredient is the one that matters, and it comes from **Karen Spärck Jones (1972)** — a paper often mis-credited, so let's be precise: Spärck Jones invented IDF; Salton & Buckley later showed how best to combine it with TF.

Her question: *how specific is this term?* Her answer: **a word that appears in nearly every document can't distinguish between them; a word that appears in only a few is highly specific.** So weight each word by the inverse of how many documents contain it.

Let $\text{df}(t)$ be the **document frequency** — the number of documents containing $t$ at least once. Then:

$$\text{idf}(t) = \log \frac{N}{\text{df}(t)}$$

Now look hard at that formula, because you have already seen it. $\text{df}(t)/N$ is the probability that a randomly picked document contains word $t$. So:

$$\text{idf}(t) = \log \frac{1}{P(\text{a document contains } t)} = \textbf{the surprise of seeing } t \textbf{ in a document}$$

That is **exactly Shannon's surprise from Day 1** — $\log(1/p)$, measured in bits if you use $\log_2$ — with the event "this word shows up in a document" in place of "this letter comes next." Spärck Jones arrived at it from library science and information retrieval, not information theory, but it is the same quantity. **IDF is Day 1's first equation, wearing different clothes.**

That reframing makes IDF stop being a formula to memorize and start being obvious: *weight each word by how much information its presence carries.* A word in every document carries zero bits; a word in one document out of a million carries about 20.

Let's compute it on the review corpus. $N = 6$, and using $\log_2$ so the answers are in bits:

| Term | Appears in | df | $\text{idf} = \log_2(6/\text{df})$ | Read as |
| --- | --- | --- | --- | --- |
| film | D1, D3, D4, D6 | 4 | **0.58 bits** | everywhere — nearly worthless |
| bad | D4, D5 | 2 | **1.58 bits** | somewhat specific |
| great | D1, D2 | 2 | **1.58 bits** | somewhat specific |
| story | D2, D5 | 2 | **1.58 bits** | somewhat specific |
| good | D3 | 1 | **2.58 bits** | rare — highly specific |
| dull | D6 | 1 | **2.58 bits** | rare — highly specific |

`film` is our corpus's version of `the`: it shows up in four of six reviews, in both classes, and knowing a review says "film" tells you essentially nothing. IDF has automatically discovered that — no stop-word list required. **The stop-word list was a hand-written rule; IDF is the learned version of it.** Day 1's lesson, third time.

Two footnotes that matter in practice:

- **The log base doesn't matter.** Switching from $\log_2$ to $\ln$ multiplies every IDF by the same constant (1/ln 2), which scales every document vector by that constant — and after the length normalization of Section 4.6, the vectors come out *bit-for-bit identical*. (I verified this on the review corpus; both bases give the same normalized matrix.) Use whichever you like; scikit-learn uses $\ln$.
- **Real implementations smooth it.** scikit-learn actually computes $\ln\frac{1+N}{1+\text{df}} + 1$. The `+1`s avoid dividing by zero for a word never seen in training, and the trailing `+1` keeps a word that appears in *every* document from getting weight exactly zero (which would delete it entirely rather than merely deprioritize it). Same idea, defensive arithmetic — and worth knowing, because it's why hand-computed IDF and sklearn's IDF never quite match.

### 4.5 TF-IDF: multiply them

Put the two together:

$$\boxed{\ \text{tf-idf}(t, d) = \text{tf}(t,d) \times \text{idf}(t)\ }$$

In one sentence: **a word matters in a document if that document uses it a lot (TF) and the rest of the corpus doesn't (IDF).** Both conditions must hold. Frequent-and-everywhere (`the`) scores low because IDF kills it. Rare-and-absent-here scores low because TF is zero. Only frequent-here-and-rare-elsewhere survives — which is precisely the definition of a word that characterizes this document.

Our corpus, multiplied out:

| | bad | dull | film | good | great | story |
| --- | --- | --- | --- | --- | --- | --- |
| **D1** | 0 | 0 | 0.58 | 0 | 1.58 | 0 |
| **D2** | 0 | 0 | 0 | 0 | **3.17** | 1.58 |
| **D3** | 0 | 0 | 0.58 | **2.58** | 0 | 0 |
| **D4** | 1.58 | 0 | 0.58 | 0 | 0 | 0 |
| **D5** | **3.17** | 0 | 0 | 0 | 0 | 1.58 |
| **D6** | 0 | **2.58** | 0.58 | 0 | 0 | 0 |

D2's `great` scores 3.17 — said twice (TF = 2), and reasonably specific (IDF = 1.58). Meanwhile `film` never rises above 0.58 anywhere. The matrix now encodes *what each review is about* rather than *which words it contains*.

### 4.6 Length normalization, and Salton & Buckley (1988)

One flaw remains. A 2,000-word review has bigger numbers everywhere than a 20-word review, simply because it's longer. Compare them and length drowns out content.

The fix: divide each row by its own length, so every document becomes a vector of length 1 — **L2 (cosine) normalization**:

$$\vec{d}_{\text{norm}} = \frac{\vec{d}}{\|\vec{d}\|_2}, \qquad \|\vec{d}\|_2 = \sqrt{\textstyle\sum_t \text{tf-idf}(t,d)^2}$$

For D1: $\|\vec{d}\| = \sqrt{0.58^2 + 1.58^2} = \sqrt{0.34 + 2.51} = 1.69$, giving `film` = 0.35 and `great` = 0.94. All six rows normalized:

| | bad | dull | film | good | great | story |
| --- | --- | --- | --- | --- | --- | --- |
| **D1** | 0 | 0 | 0.346 | 0 | 0.938 | 0 |
| **D2** | 0 | 0 | 0 | 0 | 0.894 | 0.447 |
| **D3** | 0 | 0 | 0.221 | 0.975 | 0 | 0 |
| **D4** | 0.938 | 0 | 0.346 | 0 | 0 | 0 |
| **D5** | 0.894 | 0 | 0 | 0 | 0 | 0.447 |
| **D6** | 0 | 0.975 | 0.221 | 0 | 0 | 0 |

Now every document sits on the unit sphere and only its *direction* — its mix of topics — carries meaning. The dot product between two normalized rows is the cosine of the angle between them, which is why this is the famous **cosine similarity** of information retrieval.

This is where **Salton & Buckley (1988)**, *Term-Weighting Approaches in Automatic Text Retrieval*, comes in. By the mid-1980s there were dozens of proposed TF variants, IDF variants, and normalization schemes, and no agreement about which to use. Gerard Salton (who built the SMART retrieval system, the ancestor of every search engine) and Chris Buckley did the unglamorous, decisive thing: they **systematically combined and benchmarked them** across multiple document collections, and introduced the SMART notation (a code like `ltc` meaning log-TF, IDF, cosine normalization) so people could finally say precisely which scheme they meant.

Their conclusion is what everyone still does: **use TF weighted by IDF, with cosine length normalization.** The paper's real contribution isn't a new formula — it's that it *settled the argument with evidence*. TF-IDF became the default because Salton & Buckley measured it, and it has stayed the default for nearly forty years.

That's our representation. Now: two ways to classify it.

## 5. Naive Bayes — Maron (1961)

### 5.1 The first text classifier ever built

In 1961, **M. E. Maron** published *Automatic Indexing: An Experimental Inquiry* in the *Journal of the ACM*. The setup: 405 abstracts from the journal *Computers and Automation*, and 32 subject categories. The goal: have a computer file each abstract into the right category, with no human reading it.

His method — remarkable for its date — was to pick a set of "clue words," measure how often each clue word occurred in each category, and then apply **Bayes' rule** to compute the probability of each category given the words present. This is, straightforwardly, the first automatic text classifier, and the technique we now call **naive Bayes**. It ran on a computer with less memory than a modern doorbell.

The year before, **Maron & Kuhns (1960)** had introduced the broader framework — *probabilistic indexing*, the idea that a retrieval system should rank documents by the *probability* they're relevant rather than pretending relevance is a yes/no property. That "rank by probability" instinct became the foundation of information retrieval, and it's the reason your search results come in an order.

### 5.2 Bayes' rule, applied to a document

We want $P(c \mid d)$: probability of class $c$ given document $d$. Bayes' rule flips it into things we can actually count:

$$P(c \mid d) = \frac{P(d \mid c) \; P(c)}{P(d)}$$

Since we only want to know which class *wins*, and $P(d)$ is the same for every class, we can drop the denominator:

$$\hat{c} = \arg\max_c \; \underbrace{P(c)}_{\text{prior}} \times \underbrace{P(d \mid c)}_{\text{likelihood}}$$

In plain words, a class wins if it passes two tests at once:

- **Prior** $P(c)$: how common is this class in general? (If 99% of email is spam, start out suspicious.)
- **Likelihood** $P(d \mid c)$: if a document *were* from this class, how likely is it to look like *this*?

If that two-test structure feels familiar, it should — it is exactly the acoustic-model × language-model split from Day 1's speech recognizer. *Does the evidence fit?* times *is this hypothesis plausible to begin with?* Same machine, different problem.

### 5.3 The naive assumption

We're stuck on $P(d \mid c)$: the probability of an entire specific document. We could never count that — a particular 200-word review has appeared exactly once in the history of the universe. This is the **Day 1 wall all over again**: the exact quantity is uncomputable, so we need an approximation that makes counting possible.

Day 1's approximation was the Markov assumption (only the last word matters). Today's is even more aggressive:

> **Assume every word is independent of every other word, given the class.**

Then the probability of the document is just the product of the probabilities of its words:

$$P(d \mid c) = \prod_{i=1}^{n} P(w_i \mid c)$$

and the whole classifier becomes:

$$\hat{c} = \arg\max_c \; P(c) \prod_{i=1}^{n} P(w_i \mid c)$$

This is **wildly false.** Given "positive review," the word `york` is *not* independent of `new`. `special` is not independent of `effects`. The assumption claims a document is generated by drawing words from a bag one at a time with replacement, which is not remotely how anyone writes.

And yet naive Bayes works well. Why? **Because we only need the argmax to be right, not the probabilities.** Naive Bayes' estimate of $P(c\mid d)$ is usually wildly overconfident — we'll measure log-odds near ±17 on IMDb, i.e. odds of about 20 million to one, on documents it gets wrong. But overconfidence in the *right direction* still picks the right class. The ranking survives even when the numbers are nonsense. That's worth remembering as a general lesson: **a model can be badly calibrated and still be a good decision-maker.**

### 5.4 Training = counting (again)

To estimate $P(w \mid c)$, pool every document of class $c$ into one big pile of words and ask: what fraction of that pile is word $w$?

$$P(w \mid c) = \frac{\text{count}(w, c)}{\sum_{w' \in V} \text{count}(w', c)}$$

**Training a naive Bayes classifier is counting words and dividing.** One pass over the data, no gradients, no iterations. This is the third time in three days that "training" has meant "count and divide" — Day 1's n-gram table, Day 2's BPE merge counts, and now this.

And with counting comes the same bug as Day 1. Suppose `dull` never appears in any positive review. Then $P(\text{dull} \mid \text{pos}) = 0$, and because we're *multiplying*, one zero annihilates the entire product — a 500-word glowing review containing one `dull` gets probability exactly zero of being positive. Day 1's zero-probability catastrophe, verbatim.

Same fix, too: **Laplace (add-one) smoothing.** Pretend every word occurred once more than it did:

$$P(w \mid c) = \frac{\text{count}(w, c) + 1}{\left(\sum_{w'} \text{count}(w', c)\right) + V}$$

The $+V$ in the denominator is bookkeeping, exactly as on Day 1: we handed out $V$ phantom counts (one per vocabulary word), so the denominator must grow by $V$ for the probabilities to still sum to 1.

### 5.5 Doing it by hand on the review corpus

Pool the words by class:

| | bad | dull | film | good | great | story | **total** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **positive** (D1–D3) | 0 | 0 | 2 | 1 | 3 | 1 | **7** |
| **negative** (D4–D6) | 3 | 1 | 2 | 0 | 0 | 1 | **7** |

Priors: three documents each, so $P(\text{pos}) = P(\text{neg}) = 1/2$.

Now apply add-one smoothing with $V = 6$ and class total 7, so every denominator is $7 + 6 = 13$:

| | bad | dull | film | good | great | story |
| --- | --- | --- | --- | --- | --- | --- |
| $P(w \mid \text{pos})$ | 1/13 | 1/13 | 3/13 | 2/13 | **4/13** | 2/13 |
| $P(w \mid \text{neg})$ | **4/13** | 2/13 | 3/13 | 1/13 | 1/13 | 2/13 |

(Sanity check: each row sums to 13/13 = 1. ✓)

**That is the whole trained model.** Now classify a new review the model has never seen: **"great story"**.

$$P(\text{pos}) \cdot P(\text{great}\mid\text{pos}) \cdot P(\text{story}\mid\text{pos}) = \tfrac{1}{2} \cdot \tfrac{4}{13} \cdot \tfrac{2}{13} = \tfrac{4}{169} \approx 0.0237$$

$$P(\text{neg}) \cdot P(\text{great}\mid\text{neg}) \cdot P(\text{story}\mid\text{neg}) = \tfrac{1}{2} \cdot \tfrac{1}{13} \cdot \tfrac{2}{13} = \tfrac{1}{169} \approx 0.0059$$

**Positive wins, at odds of exactly 4 to 1.** And notice where the 4 came from: `story` contributed 2/13 to both sides and cancelled out completely; the entire decision was made by `great`, whose ratio is 4/13 ÷ 1/13 = 4. Neutral words cancel; discriminative words decide. Naive Bayes is, underneath, a weighted vote — a fact Section 6.5 makes precise.

**In practice, use logs.** Multiplying hundreds of small probabilities underflows to zero in floating point. Since $\log$ turns products into sums and preserves ordering:

$$\hat{c} = \arg\max_c \; \left[ \log P(c) + \sum_{i} \log P(w_i \mid c) \right]$$

Every real implementation does this. It also reveals the model's true shape: **a sum of per-word scores**, i.e. a linear classifier.

### 5.6 Two event models — McCallum & Nigam (1998)

By 1998, "naive Bayes for text" was being used everywhere and meant two genuinely different models that people kept confusing. **Andrew McCallum and Kamal Nigam** wrote *A Comparison of Event Models for Naive Bayes Text Classification* to separate them and settle which to use.

**Multivariate Bernoulli.** A document is a vector of yes/no answers, one per vocabulary word: *is this word present?* Crucially, it explicitly models **absence** — a positive review that lacks `awful` gets credit for lacking it, via a $(1 - P(\text{awful}\mid\text{pos}))$ factor. Word counts are ignored; saying `bad` five times is the same as saying it once.

**Multinomial.** A document is a sequence of draws from a per-class word distribution. **Counts matter** (five `bad`s contribute five times), and absent words contribute nothing at all. This is the model we worked through above.

| | Bernoulli | Multinomial |
| --- | --- | --- |
| Document is… | a set of present/absent flags | a bag of word draws |
| Repeated words | ignored | counted |
| Absent words | actively penalize | ignored |
| Best when | very small vocabularies, short docs | most real text |

Their finding: **multinomial wins at realistic vocabulary sizes**, often substantially, while Bernoulli can edge ahead when the vocabulary is very small. The intuition is that Bernoulli's explicit absence-modeling means every one of tens of thousands of unused words votes on every document — and that flood of absence-evidence swamps the handful of words actually present.

Our IMDb run reproduces this only partway, which is itself instructive: Bernoulli scored **F1 0.815** against multinomial-on-counts at **0.801**. Bernoulli *won*, mildly. The reason is that IMDb reviews are long, so a word repeated eight times is usually one reviewer's verbal tic rather than eight independent pieces of evidence — and binarizing throws that inflation away. This is the double-counting problem of Section 5.7 showing up as an experimental result, and it's why "binary multinomial NB" (count each word once per document) is a well-known strong baseline.

The general lesson is bigger than either model: **naive Bayes is not one algorithm.** It's a template, and you must state which event model you mean.

### 5.7 Where naive Bayes breaks

Three failure modes, all traceable to the independence assumption.

**1. Double counting correlated features.** If `superb` and `outstanding` tend to appear together, naive Bayes treats them as two independent confirmations and multiplies both in — when really they're close to one piece of evidence stated twice. This inflates confidence without improving decisions. We can demonstrate it surgically: **duplicate every feature column** (so each word appears twice, perfectly correlated). Nothing has been added — the second copy carries zero new information. On IMDb, naive Bayes' mean absolute log-odds went from **16.8 to 33.5, exactly doubling**, while logistic regression's decision margin barely moved and 99.1% of its predictions were unchanged. Naive Bayes cannot notice that two identical columns are redundant. Logistic regression simply splits the weight between them.

**2. Rare words hijack the model.** Look at the twenty words with the highest positive-to-negative log-ratio in our trained IMDb naive Bayes: `edie`, `antwone`, `gunga`, `goldsworthy`, `gypo`, `visconti`, `flavia`… These are proper nouns, appearing in **9 to 40 documents out of 25,000**. They score so highly for one reason: they appear in a couple of well-liked films and *never* in the other class, so smoothing hands them an extreme ratio. The model's most confident evidence is a list of character names. Compare logistic regression's top words on the same data — `great`, `excellent`, `best`, `perfect`, `wonderful` — appearing in 600 to 24,000 documents each. Regularization (Section 6.4) is what makes the difference: it shrinks weights that rest on thin evidence, and naive Bayes has no equivalent.

**3. Negation and word order.** `not good` is invisible to a bag of words. This is a representation failure rather than a naive Bayes failure — logistic regression on the same features is equally blind — but naive Bayes has no way to even partially compensate. The standard patch is bigrams: add `not_good` as its own feature. We'll measure exactly how much that recovers.

## 6. Logistic regression — learning the weights instead of counting them

Naive Bayes computes its weights from counts, in one pass, and never checks whether those weights actually classify well. Logistic regression asks the opposite question: **what weights would minimize the number of mistakes on the training data?** — and then goes and finds them.

### 6.1 Score, then squash

Give every word $j$ a **weight** $w_j$ — positive for words that argue "positive review," negative for words that argue the opposite. Score a document by adding up the weights of the words it contains, scaled by their feature values:

$$z = \sum_{j=1}^{V} w_j x_j + b = \vec{w} \cdot \vec{x} + b$$

where $\vec{x}$ is the document's TF-IDF vector and $b$ is a bias term (the model's default leaning before it reads anything).

That score $z$ runs from $-\infty$ to $+\infty$, but we need a probability in $[0, 1]$. The **sigmoid** (logistic) function does the conversion:

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad P(\text{positive} \mid d) = \sigma(\vec{w} \cdot \vec{x} + b)$$

It's an S-curve with three properties worth internalizing: $\sigma(0) = 0.5$ (no evidence → complete uncertainty), it's symmetric ($\sigma(-z) = 1 - \sigma(z)$), and it saturates — beyond about $|z| = 5$ it's pinned at 0 or 1, so extra evidence stops mattering. That saturation is the model refusing to be *infinitely* confident, and it's precisely what naive Bayes lacks.

| $z$ | $-4$ | $-2$ | $-1$ | 0 | 1 | 2 | 4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\sigma(z)$ | 0.018 | 0.119 | 0.269 | **0.500** | 0.731 | 0.881 | 0.982 |

For more than two classes, the sigmoid generalizes to the **softmax**, which exponentiates every class score and normalizes them to sum to 1:

$$P(c \mid d) = \frac{e^{z_c}}{\sum_{c'} e^{z_{c'}}}$$

**Stop and look at that formula, because it is the last layer of GPT.** A Transformer produces a vector for the current position, multiplies it by a matrix to get one score per vocabulary word, and softmaxes to get $P(\text{next token})$. That is multi-class logistic regression with 50,000 classes, sitting on top of learned features instead of TF-IDF. Today we are building the output layer of every model in the remaining 54 days.

### 6.2 The loss is Day 1's cross-entropy

We need a definition of "good weights." The answer is the one Day 1 already gave us: **the model should be as unsurprised as possible by the true labels.** For a document with true label $y \in \{0, 1\}$ and predicted probability $\hat{y}$:

$$\text{loss} = -\big[\, y \log \hat{y} + (1-y)\log(1-\hat{y}) \,\big]$$

The bracket looks fussy but the two halves are a switch. If the true label is 1, the second term vanishes and the loss is $-\log \hat{y}$ — the **surprise** of the true answer. If the true label is 0, the first vanishes and the loss is $-\log(1 - \hat{y})$ — again the surprise of the true answer. In both cases:

> **loss = how surprised the model was by the correct label.**

That is Day 1's cross-entropy, unchanged, with two outcomes instead of a vocabulary. Averaged over the training set it's called **binary cross-entropy** or **log loss**, and the whole training problem is: find $\vec{w}, b$ minimizing it.

Notice the shape of the penalty: predicting 0.99 when the truth is 0 costs $-\log(0.01) = 4.6$; predicting 0.5 costs 0.69. **Confident and wrong is punished far harder than uncertain.** Day 1's warning about zero probabilities is the same fact — a prediction of exactly 0 on a true label costs infinity.

### 6.3 Gradient descent — one step by hand

There's no closed-form formula for the best weights (unlike naive Bayes, where counting *is* the answer). So we search: start somewhere, work out which direction reduces the loss, take a small step, repeat.

The derivative of the average loss with respect to weight $w_j$ works out to something remarkably clean:

$$\frac{\partial L}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m} (\hat{y}_i - y_i)\, x_{ij}$$

Read it in English: **(prediction − truth) × (feature value), averaged.** The term $(\hat{y}_i - y_i)$ is the error on document $i$. If the model overpredicted, error is positive; multiply by the words present and push those weights *down*. If it underpredicted, push them *up*. Words that weren't in the document have $x_{ij} = 0$ and are untouched.

Then the update, with **learning rate** $\eta$:

$$w_j \leftarrow w_j - \eta \frac{\partial L}{\partial w_j}$$

**Let's do exactly one step on the review corpus, using raw counts and starting from all-zero weights.** With $\vec{w} = 0$ and $b = 0$, every document scores $z = 0$, so $\hat{y} = \sigma(0) = 0.5$ for all six — the model is maximally uncertain, as it should be knowing nothing. The errors are $-0.5$ for the three positive documents and $+0.5$ for the three negative ones.

So the gradient for word $j$ becomes $\frac{0.5}{6}\big[(\text{count in neg}) - (\text{count in pos})\big]$, and after the update (taking $\eta = 1$) each weight is $\frac{1}{12}\big[(\text{count in pos}) - (\text{count in neg})\big]$:

| Word | count in pos | count in neg | difference | $w_j$ after one step |
| --- | --- | --- | --- | --- |
| great | 3 | 0 | +3 | **+0.250** |
| good | 1 | 0 | +1 | **+0.083** |
| film | 2 | 2 | 0 | **0.000** |
| story | 1 | 1 | 0 | **0.000** |
| dull | 0 | 1 | −1 | **−0.083** |
| bad | 0 | 3 | −3 | **−0.250** |

(The bias also stays at 0, because the classes are balanced.) The average loss dropped from $\ln 2 = 0.693$ to $0.567$.

Look at what a **single step** has already achieved. `great` and `bad` — the strongest, most class-exclusive words — got the largest weights, symmetric in sign. `good` and `dull` got smaller weights proportional to their weaker evidence. And `film` and `story` got **exactly zero**, because they appear equally in both classes and carry no discriminative signal at all.

That last point is the heart of the difference between the two classifiers. Naive Bayes assigned `film` a probability of 3/13 **in both classes** — it spent real probability mass modeling a word that decides nothing, because its job is to describe how documents are *generated*. Logistic regression assigned `film` a weight of zero, because its job is only to *separate* the classes, and a word that doesn't separate gets no attention. **Naive Bayes models the data; logistic regression models the boundary.**

### 6.4 Regularization — the essential knob

With 74,849 features and 25,000 documents, there are far more knobs than data points, and the model can memorize. A word appearing in exactly one positive review can be given an enormous weight to classify that one document perfectly — which is worthless on new data. This is **overfitting**.

**Regularization** adds a penalty for large weights:

$$L_{\text{total}} = \underbrace{\text{cross-entropy}}_{\text{fit the data}} + \underbrace{\lambda \sum_j w_j^2}_{\text{keep weights small}}$$

The squared-weights version is **L2**; using $\sum|w_j|$ instead is **L1**, which drives many weights to exactly zero and thus performs feature selection. scikit-learn parameterizes this as `C = 1/λ`, so **small C = strong regularization.** It is confusingly inverted, and it is the single most important hyperparameter you will tune.

Our IMDb sweep shows why it matters — and shows overfitting happening in real time:

| C | train accuracy | test accuracy | test F1 |
| --- | --- | --- | --- |
| 0.01 | 0.802 | 0.791 | 0.797 |
| 0.1 | 0.869 | 0.850 | 0.852 |
| **1** | 0.932 | **0.883** | **0.883** |
| 10 | 0.983 | 0.881 | 0.880 |
| 100 | **1.000** | 0.868 | 0.866 |

At C = 100 the model classifies its training set **perfectly** — 100.0% — and does *worse* on new data than the far humbler C = 1. That gap between the train and test columns is the entire concept of overfitting, visible in one table. Note also that C = 0.01 underfits: too much regularization and the model isn't allowed to learn enough. The best value is in the middle, and you find it by measuring, never by intuition.

This is also the answer to naive Bayes' rare-word hijacking from Section 5.7. `edie` appears in 40 of 25,000 documents; regularization asks "is this weight earning its size?" and shrinks it. Naive Bayes has no such mechanism — smoothing prevents zeros, but nothing prevents extremes.

### 6.5 Generative vs. discriminative — Ng & Jordan (2001)

The two classifiers we just built are the canonical example of a deep distinction in machine learning.

**Naive Bayes is generative.** It models $P(d \mid c)$ — how documents in each class are *produced*. It learns enough to hallucinate a fake positive review by drawing words from the positive distribution.

**Logistic regression is discriminative.** It models $P(c \mid d)$ directly. It has no idea how documents are produced and cannot generate one; it only knows where the boundary between classes lies.

The remarkable connection is that these two are a **generative–discriminative pair**: they have the *same functional form* — both compute a weighted sum of word features and pass it through a sigmoid/softmax — and differ only in how the weights are found. Naive Bayes derives them from counts under an independence assumption; logistic regression optimizes them directly against the loss. That's why Section 5.5's naive Bayes decision reduced to a weighted vote: it *is* a linear classifier with weights $\log P(w|c_1) - \log P(w|c_0)$.

**Andrew Ng and Michael Jordan (2001)** analyzed exactly this pair and found something counterintuitive and useful. Their result, in two lines:

- Logistic regression has **lower asymptotic error** — with enough data, the discriminative model wins, because it isn't hobbled by a false independence assumption.
- Naive Bayes **converges much faster** — roughly $O(\log n)$ training examples to approach its best performance, versus $O(n)$ for logistic regression. With little data, the generative model's assumptions act like a helpful prior, and it wins.

So there should be a **crossover point**: naive Bayes ahead on small training sets, logistic regression ahead on large ones. We measured it on IMDb, refitting the vectorizer on each subset so no information leaks from the full corpus:

| Training documents | NB accuracy | LR accuracy | Winner |
| --- | --- | --- | --- |
| 20 | **0.646** | 0.594 | NB |
| 50 | **0.673** | 0.636 | NB |
| 200 | **0.728** | 0.713 | NB |
| 500 | **0.767** | 0.749 | NB |
| 1,000 | 0.788 | **0.790** | ~tie |
| 5,000 | 0.806 | **0.860** | LR |
| 25,000 | 0.814 | **0.883** | LR |

**The crossover is right around 1,000 documents**, and the theory is confirmed on real data. Naive Bayes gets going faster and then plateaus hard — it gains only 2.6 accuracy points between 1,000 and 25,000 documents, because its ceiling is set by the independence assumption, not by data. Logistic regression is worse early and keeps climbing.

The practical rule this gives you: **if you have a few hundred labeled examples, start with naive Bayes. If you have tens of thousands, use logistic regression.** And notice how completely modern that shape is — "the model with stronger built-in assumptions wins in the low-data regime; the model that assumes less wins once data is abundant" is the same argument that plays out between fine-tuning and pretraining later in this roadmap.

## 7. Joachims (1998) — why text is *made* for linear classifiers

**Thorsten Joachims'** *Text Categorization with Support Vector Machines* (ECML 1998) is the day's fourth paper, and its lasting value is less "use SVMs" than **an explanation of why linear classifiers work so unreasonably well on text.** He listed the properties of text data:

1. **Very high dimensionality.** Tens of thousands of features. Most learning algorithms of the era needed aggressive feature selection to cope; SVMs, whose guarantees depend on the margin rather than the number of dimensions, don't.
2. **Few irrelevant features.** People had assumed most words were noise to be filtered out. Joachims tested it — ranking features by informativeness and training only on the *worst* ones — and even those performed well above chance. **Almost every word carries some signal.** So aggressive feature selection throws away real information.
3. **Sparsity.** Each document touches a tiny fraction of the vocabulary (we measured 0.18% on IMDb).
4. **Most text problems are linearly separable.** In a space with 74,849 dimensions and only 25,000 documents, you can nearly always find a hyperplane that separates the classes — geometrically, there's simply too much room *not* to.

Property 4 is the punchline, and it explains everything else in this document: **you don't need a nonlinear model for text in a bag-of-words representation, because the representation is already high-dimensional enough that a plane suffices.** All that's left is choosing *which* separating plane, which is exactly what the different classifiers argue about. A support vector machine picks the one with the widest **margin** — the largest gap to the nearest training points — which is a principled way of choosing among the many planes that all fit the training data equally well.

Joachims found SVMs beat naive Bayes, Rocchio, C4.5 decision trees, and k-nearest-neighbours on Reuters-21578 and Ohsumed, with the added practical virtue of working well without hyperparameter fiddling. Our IMDb run agrees, and shows how narrow the gap between good linear models is:

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Naive Bayes (counts) | 0.814 | 0.861 | 0.748 | 0.801 |
| Naive Bayes (binary/Bernoulli) | 0.826 | 0.869 | 0.767 | 0.815 |
| Naive Bayes (TF-IDF) | 0.830 | 0.874 | 0.770 | 0.819 |
| Logistic regression (counts) | 0.867 | 0.873 | 0.859 | 0.866 |
| **Logistic regression (TF-IDF)** | **0.883** | 0.884 | 0.881 | **0.883** |
| Linear SVM (TF-IDF) | 0.884 | 0.884 | 0.884 | 0.884 |
| Logistic regression (TF-IDF + bigrams) | 0.890 | 0.886 | 0.895 | **0.890** |

Three things to read off this table. **TF-IDF is worth about 1.6 accuracy points** over raw counts for logistic regression — a real but modest gain, which is a useful corrective to how much attention TF-IDF gets. **Logistic regression and the linear SVM are statistically indistinguishable** (0.883 vs 0.884); once you've chosen a good representation, the specific linear classifier barely matters. And **bigrams add another 0.7 points**, buying back a little of the word order we threw away.

The natural endpoint of this line of work is **Wang & Manning (2012)**, *Baselines and Bigrams*, which showed that a carefully-built naive Bayes / SVM hybrid with bigram features (NBSVM) beat many of the more elaborate methods published against it. Their broader point is one worth keeping: **strong simple baselines are frequently under-reported and frequently sufficient.**

## 8. Evaluation: precision, recall, and F1

We have models. Now — how do we grade them? Perplexity doesn't apply; we need the classification scorecard.

### 8.1 Why accuracy lies

The obvious metric is **accuracy**: what fraction did we get right? It's fine when classes are balanced, as IMDb's are (exactly 50/50).

It becomes actively dangerous when they're not. Suppose 1% of transactions are fraudulent. A model that says "not fraud" for every single transaction scores **99% accuracy** and is completely worthless. Accuracy is dominated by the majority class, so on imbalanced data it measures the imbalance rather than the model.

We need metrics that focus on the class we actually care about.

### 8.2 The confusion matrix

Everything derives from four counts. Pick one class as "positive" (the thing you're trying to detect), then:

| | **predicted positive** | **predicted negative** |
| --- | --- | --- |
| **actually positive** | **TP** (true positive) — caught it | **FN** (false negative) — missed it |
| **actually negative** | **FP** (false positive) — false alarm | **TN** (true negative) — correctly ignored |

FP and FN are the two ways to be wrong, and **they are not interchangeable.** In cancer screening a false negative can be fatal while a false positive means an unnecessary follow-up test. In spam filtering a false positive (a real email deleted) is far worse than a false negative (spam in your inbox). Any single number that averages these two together is hiding the decision that actually matters.

### 8.3 Precision and recall

Two questions, two metrics:

$$\textbf{Precision} = \frac{TP}{TP + FP} \quad \text{— of everything I flagged, how much was right?}$$

$$\textbf{Recall} = \frac{TP}{TP + FN} \quad \text{— of everything I should have caught, how much did I catch?}$$

The one-line mnemonic: **precision is about being trusted; recall is about being thorough.** A precise model rarely cries wolf. A high-recall model rarely misses.

**They trade off, always.** Both are controlled by the same dial — the threshold at which you convert a probability into a decision. Lower the threshold and you flag more documents: recall rises, precision falls. Raise it and you flag fewer: precision rises, recall falls. The extremes are instructive: flag *everything* and recall is a perfect 1.0 while precision collapses to the base rate; flag only your single most confident document and precision is 1.0 while recall is nearly 0.

Here's that dial on our actual IMDb model, moved across its range:

| Threshold | Precision | Recall |
| --- | --- | --- |
| 0.0002 | 0.500 | **1.000** |
| 0.22 | 0.700 | 0.986 |
| **0.54** | 0.900 | 0.856 |
| 0.67 | 0.950 | 0.720 |
| 0.90 | **0.990** | 0.262 |

To make this classifier 99% precise you must accept that it finds only 26% of the positive reviews. **Nothing about the model changed between those rows** — same weights, same training. Only the threshold moved. This is why quoting a single accuracy number for a deployed classifier is close to meaningless: the operating point is a business decision, not a modeling one.

**Worked example.** Take a test set of 10 documents, 5 truly positive, where the model flags 6 as positive and gets 4 of those right:

$$TP = 4, \quad FP = 2, \quad FN = 1, \quad TN = 3$$

$$\text{Precision} = \frac{4}{4+2} = \frac{2}{3} = 0.667 \qquad \text{Recall} = \frac{4}{4+1} = \frac{4}{5} = 0.800$$

The model is thorough (catches 80% of positives) but only moderately trustworthy (a third of its flags are wrong). Accuracy here is 7/10 = 0.70, which tells you neither of those things.

### 8.4 F1 — and why the harmonic mean

Often you do want one number, for ranking models or early stopping. **F1** is the standard choice:

$$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

On our example: $F_1 = \frac{2(0.667)(0.800)}{0.667 + 0.800} = \frac{8}{11} = 0.727$.

The obvious question: why the **harmonic** mean rather than the ordinary average? Because **the harmonic mean refuses to be fooled by imbalance.** It is always pulled toward the smaller of the two values, so you cannot buy a good F1 by maxing out one metric and abandoning the other.

Watch it work. Take the useless classifier that flags *everything* as positive on a dataset with a 1% positive rate. Its recall is a perfect 1.00 and its precision is 0.01:

- **Arithmetic mean:** $(0.01 + 1.00)/2 = 0.505$ — looks like a respectable coin-flip model!
- **Harmonic mean (F1):** $2(0.01)(1.00)/(1.01) = 0.020$ — correctly damning.

The arithmetic mean lets a perfect score on one axis paper over a catastrophe on the other. The harmonic mean does not. **F1 is high only when both precision and recall are high.**

(F1 weights precision and recall equally, which is a choice, not a law. The general form $F_\beta$ lets you weight recall $\beta$ times as much as precision — $F_2$ when misses are costly, $F_{0.5}$ when false alarms are.)

### 8.5 Multiple classes, and the honest split

With more than two classes you compute precision/recall per class and then average, and **how you average changes what you're measuring**:

- **Macro-average:** compute F1 for each class, then average them. Every class counts equally, so a tiny class matters as much as a huge one. Use it when rare classes matter.
- **Micro-average:** pool all TP/FP/FN across classes, then compute once. Every *document* counts equally, so large classes dominate. (For single-label problems, micro-F1 equals accuracy.)
- **Weighted-average:** average the per-class F1s weighted by class size.

Reporting "F1 = 0.83" without saying which averaging you used is ambiguous, and the numbers can differ a lot on imbalanced data.

Finally, three rules about *what you measure on* — each of which people violate constantly:

1. **Never evaluate on training data.** Day 1's rule, unchanged. Our C = 100 model scored 100% on its training set and 86.8% on new data.
2. **Fit the vectorizer on the training set only.** This is the subtle one, and it's the most common bug in beginner text-classification code. Calling `fit_transform` on your whole dataset before splitting lets the vocabulary and the IDF values be computed using the test documents — the model gets information from data it's supposed to have never seen. On IMDb, the vocabulary grows from 74,849 words to 101,895 when the test set is included: **26.5% of the combined vocabulary is words the model has no business knowing about.** The result is a score that's too optimistic and doesn't reproduce in production.
3. **Use three splits when you tune.** Train to fit the weights, **validation** to choose hyperparameters like C, and test touched exactly once at the very end. If you pick C by looking at test scores, your test set has quietly become a training set and your reported number is inflated.

## 9. What classical text classification can't do → the road ahead

Our best model reaches 89% on IMDb, from a laptop, in seconds. What's stopping it from going higher?

1. **No word order.** `"good, not bad"` and `"bad, not good"` produce **byte-for-byte identical vectors** — I verified this on the trained model, and it means the two sentences are guaranteed the same prediction forever. Bigrams patch the worst cases (they lifted our probe sentences apart, 0.087 vs 0.024) but the patch doesn't scale: capturing "not" at a distance of five words needs 5-grams, and the feature count explodes. → **RNNs** (Days 7–8) read sequentially; **attention** (Days 10+) lets any word look at any other.

2. **No notion of meaning.** `excellent` and `superb` are two unrelated columns. Everything the model learns about one transfers *nothing* to the other, so it must see every synonym separately, many times. This is the exact complaint Day 1 made about n-grams treating `cat` and `dog` as unrelated symbols. → **Embeddings** (Days 4–6) place similar words near each other so learning shares across them. This is the single biggest limitation here, and it's next.

3. **The features are hand-designed.** We *chose* bag-of-words, we *chose* TF-IDF, we *chose* whether to include bigrams. Every one of those was a human decision encoding a human guess about what matters. The deep learning era's central move is to **learn the representation** instead of designing it — and the classifier bolted on top stays almost exactly what we built today.

4. **Labels are expensive.** We needed 25,000 hand-labeled reviews to reach 88%. That's the supervised bottleneck, and it's precisely why the field went back to language modeling: predicting the next word needs *no labels at all*, so it can consume the whole internet, and the representations it learns along the way transfer to classification with a handful of labeled examples. **The reason this roadmap is mostly about language models rather than classifiers is contained in that sentence.**

But look at what did **not** change today. We minimized cross-entropy — Day 1's quantity. We trained by counting and fixed zeros with Laplace smoothing — Day 1's algorithm. We weighted words by their surprise — Day 1's first equation. And we ended with a softmax over classes, which is the output layer of every model still to come. **We changed what the model looks at and what we ask it to predict. We did not change the game.**

## 10. Glossary (only what today uses)

| Term | Plain meaning |
| --- | --- |
| **Classification** | Assigning a document to one of a fixed set of classes. |
| **Supervised learning** | Learning from examples that a human has labeled (contrast: language modeling, where the text labels itself). |
| **Bag of words** | A representation that records which words appear and how often, discarding all order. |
| **Document-term matrix** | Rows = documents, columns = vocabulary words, cells = counts or weights. |
| **Sparse matrix** | A matrix that is almost all zeros, stored as just the non-zero entries. |
| **TF** | Term frequency — how often a word occurs in *this* document. |
| **IDF** | Inverse document frequency, $\log(N/\text{df})$ — the Shannon surprise of the word appearing in a document at all. |
| **TF-IDF** | TF × IDF: high for words this document uses a lot and other documents don't. |
| **L2 / cosine normalization** | Scaling each document vector to length 1 so document length stops mattering. |
| **Naive Bayes** | A classifier applying Bayes' rule while pretending words are independent given the class. |
| **Laplace smoothing** | Add 1 to every count so no word gets probability 0. |
| **Multinomial vs. Bernoulli** | The two naive Bayes event models: word *counts* vs. word *presence/absence*. |
| **Logistic regression** | A classifier that learns a weight per feature and squashes the weighted sum through a sigmoid. |
| **Sigmoid / softmax** | Functions converting raw scores into probabilities (2 classes / many classes). |
| **Cross-entropy (log loss)** | The training objective: the model's average surprise at the true labels. Day 1's quantity. |
| **Gradient descent** | Repeatedly nudging the weights in the direction that reduces the loss. |
| **Regularization (L1/L2)** | A penalty on large weights that prevents memorizing the training set. `C = 1/λ` in scikit-learn. |
| **Generative vs. discriminative** | Modeling how documents are produced, $P(d\mid c)$, vs. modeling the boundary directly, $P(c\mid d)$. |
| **Precision** | Of the items flagged, the fraction that were right. |
| **Recall** | Of the items that should have been flagged, the fraction that were. |
| **F1** | The harmonic mean of precision and recall — high only if both are. |
| **Macro / micro average** | Averaging per-class metrics equally vs. pooling all documents. |
| **Data leakage** | Letting information from the test set reach the model — e.g. fitting the vectorizer before splitting. |

## 11. References

1. Maron, M. E. & Kuhns, J. L. (1960). *On Relevance, Probabilistic Indexing and Information Retrieval.* Journal of the ACM 7(3). — ranking by probability of relevance; the foundation of probabilistic IR.
2. Maron, M. E. (1961). *Automatic Indexing: An Experimental Inquiry.* Journal of the ACM 8(3), 404–417. — 405 abstracts, 32 categories, clue words and Bayes' rule: the first automatic text classifier.
3. Spärck Jones, K. (1972). *A Statistical Interpretation of Term Specificity and its Application in Retrieval.* Journal of Documentation 28(1). — the origin of IDF. Credit it here, not to Salton & Buckley.
4. Salton, G. & Buckley, C. (1988). *Term-Weighting Approaches in Automatic Text Retrieval.* Information Processing & Management 24(5), 513–523. — the systematic bake-off that made TF-IDF with cosine normalization the default, plus the SMART notation.
5. Joachims, T. (1998). *Text Categorization with Support Vector Machines: Learning with Many Relevant Features.* ECML. — why text is high-dimensional, sparse, nearly all-relevant, and usually linearly separable.
6. McCallum, A. & Nigam, K. (1998). *A Comparison of Event Models for Naive Bayes Text Classification.* AAAI/ICML Workshop on Learning for Text Categorization. — multinomial vs. multivariate Bernoulli, and when each wins.
7. Nigam, K., Lafferty, J. & McCallum, A. (1999). *Using Maximum Entropy for Text Classification.* IJCAI Workshop. — MaxEnt, which is multi-class logistic regression, applied to text.
8. Ng, A. Y. & Jordan, M. I. (2001). *On Discriminative vs. Generative Classifiers: A Comparison of Logistic Regression and Naive Bayes.* NIPS. — the convergence-rate result behind the crossover we measured.
9. Wang, S. & Manning, C. D. (2012). *Baselines and Bigrams: Simple, Good Sentiment and Topic Classification.* ACL. — NBSVM, and a standing argument for taking simple baselines seriously.
10. Maas, A. L. et al. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL. — the IMDb dataset we use, and (fittingly) an early argument for learned word representations over bag-of-words.
11. Manning, C. D., Raghavan, P. & Schütze, H. (2008). *Introduction to Information Retrieval*, Chapters 6 and 13. — the standard textbook treatment of TF-IDF and naive Bayes; free online.

---

**Next:** we build it. Bag-of-words, TF-IDF, multinomial naive Bayes, and logistic regression with hand-written gradient descent — every function checked against the six-line review corpus above, then turned loose on 50,000 IMDb reviews. We'll watch regularization stop an overfit, watch naive Bayes' confidence double when we feed it the same information twice, and watch two opposite sentences get the same prediction because a bag of words cannot tell them apart.
