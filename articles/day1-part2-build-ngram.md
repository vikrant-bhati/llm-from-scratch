# I Trained a Language Model on a Million Words — and It Called 446 Real Sentences "Impossible"

*Part 2 of "LLM from Scratch." In [Part 1](link), Claude Shannon taught us that a language model is a guessing machine, graded by its average surprise. Today we build one — and break it, twice.*

---

Here's the experiment this whole article is about. I trained the simplest possible language model on a million words of English text — the classic Brown Corpus. Then I asked it to score 500 ordinary sentences it hadn't seen before. Sentences from the same corpus, same style, same vocabulary.

It declared **446 of them impossible.** Not unlikely — *impossible*. Probability exactly zero.

Understanding why that happens, and how a fix rescues it, is the best crash course in language modeling I know. Along the way we'll meet the number that has graded every language model from 1977 to GPT: **perplexity**. Everything below is runnable — the full notebook is in the [repo](link).

*If you skipped Part 1, the one paragraph you need: a language model gives a probability to text — "given the words so far, how likely is each next word?" We grade it by playing a guessing game with real text: at each word, check how likely the model thought the actual next word was, convert to surprise (rare = surprising = expensive), and average. That average surprise is called cross-entropy. Lower = better model.*

## Step 1: the probability of a sentence

How likely is the sentence "the cat sat"? Peel it one word at a time, multiplying as you go:

```
P(the cat sat) = P(the) × P(cat | the) × P(sat | the cat)
```

Read it like a story: how likely does a sentence start with "the"? — times — given "the", how likely is "cat" next? — times — given "the cat", how likely is "sat"?

This is called the **chain rule**, and it's not an approximation — it's just algebra, always exactly true. It turns "the probability of a sentence" into "the probability of each next word, given everything before it." Which is exactly what we said a language model does.

But there's a killer hiding in it. To estimate P(sat | the cat) from data, you find every "the cat" in your corpus and check what followed. Fine — short phrase, appears constantly. But the chain rule needs this for the *whole* history: by word 15 of a sentence, you need counts for a specific 14-word phrase. Try googling any exact 15-word sentence from today's news, in quotes. Zero results. **Almost every long word sequence has never been written before.** The formula is exactly right and completely unusable.

## Step 2: the shortcut

The fix that created a field: **pretend the next word depends only on the last word or two, and forget everything earlier.** This is the Markov assumption, and a model using it is an **n-gram model**:

```
bigram model  (memory = 1 word):  P(sat | the cat)  ≈  P(sat | cat)
trigram model (memory = 2 words): condition on the previous two words
```

What we gain: short phrases repeat, so the counts we need actually exist.

What we lose: everything outside the window. My favorite casualty: *"The keys to the cabinet **are** on the table."* The verb is "are" because of "keys" — three words back. A bigram model deciding between *are* and *is* sees only "cabinet", and confidently picks wrong. Remember Shannon's generated text in Part 1, fluent up close but wandering off topic? Same cause: no memory outside the window. Fixing this one flaw is the story of the next several decades — RNNs, LSTMs, attention. Today, we live with it.

## Step 3: training = counting

Where does a number like P(cat | the) come from? Ask the corpus the obvious question: **out of all the times "the" appeared, how often was "cat" the next word?** Count and divide. That's the entire training algorithm.

Small enough to do by hand. Here's a complete "training corpus" of three sentences — we'll verify all our code against it:

```
the cat sat
the cat ran
the dog sat
```

One practical trick first: wrap each sentence in start/end markers, `⟨s⟩ the cat sat ⟨/s⟩`, so the first word has context and the model learns how sentences end. Now count every adjacent pair:

```
Pair          Count        Word    Count
-----------   -----        -----   -----
⟨s⟩ the        3            ⟨s⟩      3
the cat        2            the      3
the dog        1            cat      2
cat sat        1            dog      1
cat ran        1            sat      2
dog sat        1            ran      1
sat ⟨/s⟩       2
ran ⟨/s⟩       1
```

Divide: P(cat | the) = 2/3. P(dog | the) = 1/3. P(sat | cat) = 1/2. Done — **in 1990, "training a language model" meant: count the pairs, divide.** No gradients, no GPUs, one pass over the data. In code, the whole thing is:

```python
def count_ngrams(sents, n):
    ngram_counts, context_counts = Counter(), Counter()
    for sent in sents:
        tokens = pad(sent, n)                      # add ⟨s⟩ / ⟨/s⟩
        for i in range(n - 1, len(tokens)):
            context = tuple(tokens[i - n + 1:i])
            ngram_counts[context + (tokens[i],)] += 1
            context_counts[context] += 1
    return ngram_counts, context_counts

def mle_prob(word, context, ngram_counts, context_counts):
    c = context_counts[context]
    return ngram_counts[context + (word,)] / c if c else 0.0
```

## Step 4: the crash

Now score a brand-new sentence with our freshly trained toy model: **"the dog ran."** Perfectly good English — dogs run. Multiply through:

```
P = P(the | ⟨s⟩) × P(dog | the) × P(ran | dog) × ...
  =     1        ×     1/3      ×      0       × ...   =  0
```

The pair "dog ran" never appeared in our three training sentences. So the model doesn't call the sentence unlikely — it calls it **impossible**. And zero poisons everything it touches: a zero-probability word has *infinite* surprise, so one missing pair blows up the score of an entire test set.

This is exactly what happened in my Brown Corpus experiment. A million words of training text, and still: 446 of 500 unseen test sentences contained at least one word pair the corpus had never coughed up. Because that's how language is — a handful of words appear constantly, but *most* words are rare (this lopsidedness is called Zipf's law), so most possible word *pairs* never appear in any dataset of any size. Every real test set steps on one.

The model's real crime is arrogance. Probability zero means "I have seen everything; this cannot happen." The honest statement is "I haven't seen this *yet*."

## Step 5: smoothing — teaching the model humility

The fix is called **smoothing**: shave a little probability off the pairs you did see, spread it over the pairs you didn't. The simplest version — **add-one smoothing**, which dates back to Laplace in the 1800s — just pretends every possible pair occurred once more than it did. With a vocabulary of V possible next words:

```
                  count(context + word) + 1
P(word|context) = --------------------------
                  count(context) + V
```

On our toy corpus (V = 6): P(ran | dog) goes from 0 to (0+1)/(1+6) = **1/7**. "The dog ran" is now merely unlikely, not impossible. Fixed!

But look at the price. P(sat | dog) — which was 1, because "dog" was *always* followed by "sat" — falls to (1+1)/(1+6) = **2/7**. From 100% down to 29%. Funding the unseen taxed the seen, heavily. Keep that word — *tax* — in mind; it's about to explain a very weird experimental result.

(Smarter smoothing exists — blending estimates from bigger and smaller contexts, and the classic Kneser-Ney method, which asks not "how common is this word?" but "how many *different* contexts does it follow?" We'll meet them later in the series. Add-one is enough for today's lesson.)

## Step 6: perplexity — the scorecard

Before the results, meet the grade. In 1977, Jelinek's IBM speech team needed one number to compare their competing language models. The number they invented, **perplexity**, is still the standard scorecard for every model release today.

It's our guessing game, converted to a human scale. Measure the model's average surprise per word on **text it has never seen** (never grade a student on the exam they wrote), then convert bits back into a count of options:

```
perplexity = 2 ^ (average surprise per word)
```

The meaning is wonderfully concrete:

> **Perplexity K means: at every word, the model was as unsure as if it were choosing blindly among K equally likely words.**

- Perplexity 4 → like picking from 4 options per word. Confident.
- Perplexity 100 → like rolling a 100-sided die at every word. Confused.
- A model that knows nothing scores exactly the vocabulary size. A perfect oracle scores 1. **Lower is better.**

The name is literal: it measures how *perplexed* the model is.

## The results: two surprises

I trained unigram, bigram, and trigram models on 90% of the Brown Corpus (~26,000-word vocabulary) and computed perplexity on the held-out 10%, with heavy smoothing (add-1) and gentle smoothing (add-0.01):

```
model        add-1      add-0.01
--------   --------    ---------
unigram      714.2        711.0
bigram     1,697.0        470.4
trigram    9,991.8      3,107.2
```

If you expected "more context = better model", this table has two slaps for you.

**Surprise 1: with add-one, the bigram loses to the unigram.** More context made things *worse*? No — the smoothing tax did. With V ≈ 26,000, add-one hands every context 26,000 phantom counts. Most real bigram contexts were seen only a handful of times, so the phantoms drown the evidence. Shrink the tax to add-0.01 and the bigram beats the unigram comfortably: 470 vs 711. Same model, same data — the *hyperparameter* decided the winner. (This was my first-ever hyperparameter search, and it happened in 1990s technology.)

**Surprise 2: the trigram loses even with gentle smoothing.** That's not a smoothing problem — it's a data problem. A million words is simply not enough to see most 3-word combinations even once. More context only helps if you have the data to fill it. Sound familiar? *That's the scaling story of modern AI, in miniature, in a 1961 corpus.*

And the models can write, too — sampling from the counts, exactly like Shannon's dice from Part 1. Real output from my bigram model:

```
rachel , however , cellulose propionate .
dr. louis university and the sand after watching another addition , sleeping pill , music
```

And the trigram:

```
i left behind them large numbers of people will like this gauge and type of
```

Locally fluent, globally lost — every three-word window is fine English, and no sentence is going anywhere. You are looking at the Markov assumption. The model literally cannot remember what it said four words ago.

## What we actually learned

Four lessons, each one load-bearing for the rest of this series:

1. **The zero-count problem is not theoretical.** Nine in ten real sentences were "impossible" to the raw counting model. Smoothing isn't polish — without it, the model is unusable.
2. **Smoothing is generalization, in primitive form.** A language model's real job is assigning sane probabilities to text it has *never seen*. Adjusting counts is the crudest way to do that. The elegant way — teaching the model that "cat" and "dog" are similar, so evidence about one transfers to the other — requires word embeddings. That's where this series goes next.
3. **More context demands more data.** The trigram's defeat is the count-table approach hitting a wall no hyperparameter can fix. Breaking that wall takes models that *learn* instead of count.
4. **The game never changes.** Chain rule, cross-entropy, perplexity — everything we used today is exactly what GPT is trained and graded on. Every model since 1948 upgrades *the guesser*, never *the game*.

## Try it yourself

The full notebook — including the crash demo, all the counting code, and the perplexity table — is at [repo link]. It's built around the three-sentence cat corpus, so every number your code produces is one you can check by hand first. Total runtime on the million-word corpus: about 15 seconds on a laptop.

*Next in the series: tokenization — what exactly is a "word" to a model, why that question is harder than it looks, and how subword pieces dissolve problems we hacked around today.*

---

**Papers behind this article:**

1. Jelinek, F. et al. (1977). *Perplexity — a measure of the difficulty of speech recognition tasks* — the scorecard, named.
2. Jelinek, F. (1997). *Statistical Methods for Speech Recognition* — n-gram models in production.
3. Jurafsky, D. & Martin, J. H. *Speech and Language Processing*, Ch. 3 — the canonical textbook treatment of everything here, free online.
4. Shannon, C. E. (1948). *A Mathematical Theory of Communication* — where Part 1 of this series began.
