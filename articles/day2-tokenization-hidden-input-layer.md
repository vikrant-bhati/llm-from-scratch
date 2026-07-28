# Why LLMs Stumble on Strawberries: Tokenization Is the Hidden Input Layer Behind AI's Weirdest Mistakes

*Part 3 of "LLM from Scratch." In Part 1, Shannon showed us that language modeling is a guessing game. In Part 2, we built an n-gram model and watched zero counts break it. Today we answer a question we quietly avoided: what exactly is a "word" to a language model?*

Ever wondered why LLMs sometimes struggle with questions like "how many r's are in strawberry?"

Or why arithmetic can feel weird, even when the model seems good at explaining calculus?

One reason is surprisingly low-level: the model does not see text the way you do. It does not look at a word and naturally see letters. It does not look at a number and naturally see digits. Before the model gets anything, the text is chopped into pieces called **tokens**, and each token is replaced by an integer ID.

So the model may not see:

```text
strawberry
```

as:

```text
s t r a w b e r r y
```

It may see a few larger chunks instead. And numbers may be split in inconsistent ways: one number as a whole token, another as two pieces, another digit by digit. The model can often learn around this, but the input layer's choices leak all the way up into the model's dumbest failures.

That hidden step is today's topic: **tokenization**.

It sounds like boring plumbing. It is not. Tokenization decides what the model can see, how long the input becomes, how expensive attention will be, what counts as unknown, and even how we evaluate models.

By the end of this article, you will understand why modern LLMs do not use words or characters directly. They use something in the middle: **subwords**.

## The lie we told on Day 1

On Day 1, I said a language model predicts the next word.

That was useful, but not quite true.

A neural network cannot read words. It only accepts numbers. So before training or inference, every piece of text must go through this pipeline:

```text
raw text -> tokens -> token IDs -> vectors -> model
```

For example:

```text
the cat sat
```

might become:

```text
["the", "cat", "sat"]
```

and then:

```text
[42, 918, 3041]
```

The model never receives "cat" as a word. It receives `918`.

Later, an embedding table will turn that ID into a vector. But the first step is tokenization: deciding how text should be split.

That decision matters because the model inherits it forever. If the tokenizer breaks text badly, the model starts with a bad view of the world.

## The old preprocessing pipeline

Before modern neural language models, text preprocessing usually looked like an assembly line:

1. Split text into words.
2. Lowercase everything.
3. Remove stop words like "the", "of", and "is".
4. Stem or lemmatize words, so "running", "runs", and "ran" collapse toward one root.

This made sense for older systems like search engines and bag-of-words classifiers. Those systems could not handle too much surface variation, so preprocessing threw variation away.

But modern LLMs do not want us to throw that information away.

Capitalization matters:

```text
US
us
```

Those are not the same thing.

Stop words matter:

```text
the dog chased the cat
the cat chased the dog
```

Same content words, very different meaning.

Word forms matter too:

```text
run
ran
running
```

These are related, but they are not interchangeable. A model trained to predict the next token wants all of that information. So most of the old cleanup pipeline disappears.

The one part that survives is tokenization.

But even tokenization had to be rebuilt.

## The obvious tokenizer fails immediately

The simplest tokenizer is whitespace:

```text
"the cat sat" -> ["the", "cat", "sat"]
```

It feels reasonable until you try real text.

```text
cat
cat.
cat,
```

Are those the same word or three different tokens?

What about contractions?

```text
don't
```

Should it be one token? Two tokens? `do` and `n't`? `don` and `'t`?

What about:

```text
New York
e-mail
https://example.com
#MachineLearning
```

And then the biggest problem: many languages do not put spaces between words in the same way English does. Chinese and Japanese cannot be handled by "split on spaces."

So people built rule-based tokenizers. A famous example is the Penn Treebank tokenizer, which uses carefully designed rules for punctuation, contractions, and edge cases.

That helps, but it brings back the old problem: rules do not scale cleanly. Every language, domain, emoji, URL pattern, and code format wants another rule.

Even if we solved all of that, a deeper problem remains.

## The real choice: words or characters?

Suppose we magically had perfect word splitting.

Now we need a vocabulary: the fixed list of tokens the model knows.

Should the vocabulary contain whole words?

That gives us short sequences:

```text
the cat sat
```

becomes three tokens.

Whole words also carry meaning. `cat` is a useful unit. `sat` is a useful unit. This seems great.

But word-level tokenization has a fatal flaw: **the vocabulary explodes**.

English has hundreds of thousands of word forms. Add names, typos, slang, product names, code identifiers, and other languages, and the list is effectively endless.

So you must cap the vocabulary:

```text
keep the top 50,000 words
everything else -> <unk>
```

That creates the **out-of-vocabulary problem**, or OOV.

If the model has never seen a word like:

```text
Shakespeare
antidisestablishmentarianism
vikrantGPT2026
```

then all of them may collapse into the same useless symbol:

```text
<unk>
```

The model cannot tell them apart. Worse, it cannot generate them either.

This is brutal because language has a long tail. Most words are rare. New names, new products, new slang, and new typos appear constantly. A word-level model is permanently blind to the long tail.

So maybe we should go smaller.

What if every character is a token?

```text
strawberry -> s t r a w b e r r y
```

Now OOV disappears. Any word can be spelled from characters.

But the cost is huge. Sequences become much longer. A sentence that was 20 word tokens might become 100 or 150 character tokens.

That matters because Transformer attention gets more expensive as sequence length grows. Longer token sequences mean more memory, more time, and more money.

Characters also carry little meaning by themselves. The model has to spend early layers learning how letters combine into useful chunks before it can reason about the sentence.

So we have two bad extremes:

```text
word tokens       short sequences, meaningful tokens, terrible OOV
character tokens  no OOV, tiny vocabulary, very long sequences
```

This is the problem subword tokenization solves.

## The subword idea

The idea is simple:

> Keep common words whole. Break rare words into reusable pieces.

So common words might stay as one token:

```text
the
cat
because
```

But rarer words can split:

```text
tokenization -> token + ization
unhuggable   -> un + hug + gable
```

This gives us the best parts of both extremes.

Common words stay short and meaningful. Rare words do not collapse into `<unk>` because they can be decomposed into smaller pieces.

And if the tokenizer always includes characters or bytes as a fallback, then any text can be represented.

No unknown words. No giant word vocabulary. No character-level explosion for ordinary text.

This is why modern LLMs use subword tokenization.

The question becomes: how do we choose the pieces?

The most famous answer is **BPE**, or Byte-Pair Encoding.

## BPE: learn pieces by counting pairs

BPE is beautiful because it feels almost too simple.

Start with characters. Then repeat:

1. Count every adjacent pair.
2. Merge the most frequent pair.
3. Add the merged pair to the vocabulary.
4. Do it again until the vocabulary is the size you want.

That is it.

Let's use a tiny training corpus we can check by hand:

```text
hug  : 10
pug  : 5
pun  : 12
bun  : 4
hugs : 5
```

Start with characters:

```text
h u g       : 10
p u g       : 5
p u n       : 12
b u n       : 4
h u g s     : 5
```

Now count adjacent pairs.

The pair `u g` appears in:

```text
hug  x 10
pug  x 5
hugs x 5
```

Total:

```text
20
```

That is the most frequent pair, so BPE merges it:

```text
u + g -> ug
```

The corpus becomes:

```text
h ug        : 10
p ug        : 5
p u n       : 12
b u n       : 4
h ug s      : 5
```

Now count again. This time `u n` wins with count 16:

```text
u + n -> un
```

Now:

```text
h ug        : 10
p ug        : 5
p un        : 12
b un        : 4
h ug s      : 5
```

Count again. Now `h ug` wins:

```text
h + ug -> hug
```

After three merges, the learned rules are:

```text
1. u + g  -> ug
2. u + n  -> un
3. h + ug -> hug
```

That ordered list is the trained tokenizer.

No neural network. No gradients. No GPU.

Just counting and merging.

## Using the trained tokenizer

Now try a word the tokenizer never saw during training:

```text
bug
```

Start with characters:

```text
b u g
```

Apply the learned merges in order.

Rule 1 says:

```text
u + g -> ug
```

So:

```text
b u g -> b ug
```

No other rule applies.

Final tokenization:

```text
[b, ug]
```

The word `bug` was never in the training corpus, but it did not become `<unk>`. It was represented using known pieces.

That is the payoff.

Now try:

```text
hugs
```

Start:

```text
h u g s
```

Apply rule 1:

```text
h ug s
```

Apply rule 3:

```text
hug s
```

Final:

```text
[hug, s]
```

The common part `hug` became one token. The plural `s` stayed separate.

This is exactly what we wanted: common patterns become compact, rare combinations remain spellable.

## The final leak: unknown characters

Character-level BPE solves unknown words only if every character in the new text appeared during tokenizer training.

But what if the tokenizer was trained on English and then sees:

```text
你
🙂
```

If those characters are not in the base vocabulary, we still have an OOV problem.

GPT-style tokenizers solve this with **byte-level BPE**.

Instead of starting from characters, start from raw bytes. There are only 256 possible byte values, and every digital string can be represented as bytes.

That means the tokenizer can encode any text at all: English, Chinese, emoji, code, corrupted text, anything.

This is the true "nothing is unknown" version.

## BPE is not the only subword method

BPE is the easiest to understand, but two other tokenization families matter.

**WordPiece** is famous because BERT used it. It is similar to BPE, but it does not simply merge the most frequent pair. Instead, it asks which pair is surprisingly common together compared with how common its parts are separately.

The score is:

```text
score(a, b) = freq(ab) / (freq(a) * freq(b))
```

BPE follows raw popularity. WordPiece follows association strength.

That means BPE may merge a very common pair because it appears everywhere, while WordPiece prefers pairs whose pieces really seem to belong together.

**Unigram LM** goes the other direction.

BPE starts small and merges upward:

```text
characters -> subwords -> words
```

Unigram starts with a large candidate vocabulary and prunes downward:

```text
many possible pieces -> remove least useful pieces -> final vocabulary
```

It also treats tokenization probabilistically. A word can have many possible segmentations, each with a probability. The tokenizer chooses the most likely one, often using a dynamic programming algorithm called Viterbi.

That detail matters because Unigram can sample different segmentations during training. The same word can be cut slightly differently on different passes, which acts like data augmentation.

Then there is **SentencePiece**.

SentencePiece is not exactly one algorithm. It is a tokenizer framework that can use BPE or Unigram. Its important trick is that it treats text as raw text without requiring whitespace pre-splitting.

It also represents spaces explicitly, often with the symbol:

```text
▁
```

That visible space marker is `U+2581`, which looks like a low block. The idea is simple: whitespace is not thrown away. It becomes part of the token stream, so tokenization can be reversed cleanly.

That makes SentencePiece especially useful for multilingual models.

## Why the strawberry example happens

Now return to the hook.

Why might an LLM stumble when asked:

```text
How many r's are in strawberry?
```

The human sees letters. The model sees token IDs.

If the tokenizer splits `strawberry` into larger chunks, the model is not directly handed:

```text
s t r a w b e r r y
```

It has to infer the internal spelling from the token pieces and from patterns learned during training.

Important caveat: tokenization is not the only reason models fail at letter counting. The training objective matters too. A next-token predictor is trained to continue text, not to run a precise character-counting algorithm. But tokenization makes the task less natural because the basic input units are not individual letters.

The same thing can happen with arithmetic.

Humans see:

```text
12345
```

as digits with place value:

```text
1 2 3 4 5
```

A tokenizer might split numbers into chunks inconsistently:

```text
12345 -> 123 + 45
10000 -> 100 + 00
987654 -> 98 + 765 + 4
```

Those examples are illustrative, not a promise about one specific tokenizer. The point is that the model is not guaranteed a clean digit-by-digit representation.

So arithmetic becomes a strange learned text pattern unless the model has enough training, enough examples, or tools that let it calculate directly.

Tokenization is not the whole story behind these failures, but it is the first crack in the input.

## Tokenization also changes the bill

Tokenization affects quality, but it also affects cost.

Commercial LLM APIs usually charge by tokens. Transformer attention also becomes more expensive as token sequences get longer.

So if two tokenizers encode the same paragraph differently:

```text
Tokenizer A: 200 tokens
Tokenizer B: 350 tokens
```

Tokenizer B makes the same text longer for the model.

That means more compute, more memory, and often more money.

This is especially visible across languages. A tokenizer trained mostly on English may split non-English text into many more pieces. The meaning may be preserved, but the sequence gets longer.

One useful measure is **fertility**:

```text
fertility = average tokens per word
```

Low fertility means the tokenizer is compact. High fertility means it is splitting heavily.

A good tokenizer keeps text compact without losing the ability to represent rare strings.

## The perplexity trap

Day 1 gave us perplexity:

```text
perplexity = 2 ^ average surprise
```

It means: how many equally likely choices was the model effectively choosing among at each step?

But now we have to be careful.

Perplexity is average surprise **per token**.

So if you change the tokenizer, you change what "per token" means.

A word-level model might score surprise once per word. A character-level model scores surprise once per character. A subword model is somewhere in the middle.

So this comparison is broken:

```text
Model A: perplexity 20 with tokenizer A
Model B: perplexity 40 with tokenizer B
```

You cannot conclude Model A is better unless the tokenization is comparable.

This is one of the easiest ways to fool yourself when reading model results.

The escape is to measure surprise using a fixed unit that no tokenizer can change: characters or bytes.

That gives us **bits per character**, or BPC:

```text
BPC = total surprise in bits / number of characters
```

Characters in the test text do not change when the tokenizer changes. So BPC lets us compare models more honestly across different tokenizers.

This connects directly back to Shannon. When Shannon estimated English at around 1 bit per letter, he was already using a tokenizer-independent unit.

The lesson:

> Raw perplexity only makes sense when the tokenizer is part of the comparison.

## What tokenization can and cannot do

Subword tokenization fixed the OOV catastrophe.

That is a huge achievement.

A word-level model sees rare words as strangers. A subword model can break them into known pieces. A byte-level tokenizer can represent any string at all.

But tokenization does not create meaning.

To the tokenizer:

```text
cat
dog
chair
7132
```

are just token IDs.

The tokenizer knows how to split. It does not know that cats and dogs are both animals. It does not know that "running" and "ran" are related. It does not know that `123` is a number with place value.

That meaning comes later, from embeddings and training.

Tokenization is only the first half of the input layer:

```text
text -> token IDs
```

Embeddings are the second half:

```text
token IDs -> vectors with learned meaning
```

That is where the next part of the roadmap goes.

## What we learned

The most important idea today is simple:

> A language model does not read text. It reads token IDs.

That one fact explains a lot.

It explains why word-level models fail on rare words. It explains why character-level models are expensive. It explains why subwords became the standard compromise. It explains why byte-level BPE can represent anything. It explains why raw perplexity is dangerous across tokenizers. And it gives us one reason LLMs can act strangely on letter counting and arithmetic.

The game from Day 1 has not changed. We still train models to predict the next unit by minimizing surprise.

But today we learned that the "unit" is not obvious.

Choosing that unit is one of the first design decisions in every LLM.

And once you make it, the entire model has to live with it.

---

**Papers behind this article**

1. Manning, Raghavan, and Schutze, *Introduction to Information Retrieval*, Chapter 2 - the classical preprocessing pipeline.
2. Sennrich, Haddow, and Birch (2016), *Neural Machine Translation of Rare Words with Subword Units* - BPE for rare words in neural machine translation.
3. Schuster and Nakajima (2012), *Japanese and Korean Voice Search* - WordPiece.
4. Kudo and Richardson (2018), *SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing* - language-independent tokenization.
5. Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* - GPT-2 and byte-level BPE.

*Next: we implement BPE from scratch, test it on the tiny `hug` corpus, then compare token counts and OOV behavior on real text.*
