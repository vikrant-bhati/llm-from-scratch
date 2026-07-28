# Why LLMs Stumble on Strawberries: The Hidden Input Layer Behind AI's Weirdest Mistakes

*Part 3 of "LLM from Scratch." In Part 1, Shannon showed us that language modeling is a guessing game. In Part 2, we built an n-gram model and watched zero counts break it. Today we answer the question we quietly avoided: what exactly is a "word" to a language model?*

Ever wondered why an LLM can explain backpropagation, write a decent SQL query, and then somehow mess up a question like:

```text
How many r's are in strawberry?
```

It feels ridiculous. You and I look at `strawberry` and see letters. We can slow down, point at the word, and count: `s t r a w b e r r y`.

But an LLM does not naturally see text that way.

Before the model gets the sentence, another piece of software has already chopped it into chunks. Those chunks are called **tokens**. Then each token is replaced with a number. The model does not receive "strawberry" as a word, or even necessarily as ten clean letters. It receives a short list of token IDs.

That is the hidden step most people skip when they first learn language models:

```text
text -> tokens -> token IDs -> vectors -> model
```

This sounds like implementation detail. It is not.

Tokenization decides what the model can see directly, how long the input becomes, how much computation the model needs, what happens to rare words, and even how honest our evaluation numbers are. It is one of those boring-looking choices that quietly leaks into everything.

And yes, it is one reason letter-counting and arithmetic can look strangely brittle. Not the only reason. A next-token predictor is not trained like a calculator or a character-counting program. But the tokenizer is where the weirdness starts: the model's basic units are not always the units humans expect.

So today is about that input layer. How do we turn text into numbers without losing the long tail of language?

## The Small Lie From Day 1

In the first two articles, I kept saying that a language model predicts the next word.

That was the right simplification for Day 1. But now we have to clean it up.

A neural network cannot read words. It can only operate on numbers. So if we write:

```text
the cat sat
```

the model needs something more like:

```text
[42, 918, 3041]
```

Those IDs point into a vocabulary: a fixed list of pieces the model knows how to handle. Later, an embedding table will turn each ID into a vector. But before embeddings, before attention, before anything impressive happens, we need the tokenizer to answer a deceptively simple question:

Where should the cuts go?

That question turns out to be much harder than it sounds.

## The Old Way: Clean the Text Until It Behaves

Classical NLP systems used to treat preprocessing like a cleanup job.

Lowercase the text. Remove common words like "the" and "is." Stem words so `running`, `runs`, and maybe even `ran` get pushed toward the same root. Strip away variation so the system has fewer cases to worry about.

For older search engines and bag-of-words classifiers, this made sense. Those systems were brittle, so we helped them by simplifying the input.

Modern LLMs are different. They do not want us to throw away information so aggressively.

Capitalization can change meaning:

```text
US
us
```

Stop words carry grammar:

```text
the dog chased the cat
the cat chased the dog
```

And word forms are not just noise:

```text
run
ran
running
```

They are related, but they are not identical. A model trained to predict real text wants those differences available. So most of the old preprocessing pipeline disappears.

The part that remains is tokenization. But even that had to be redesigned.

## Splitting on Spaces Feels Natural Until It Doesn't

The first tokenizer anyone invents is probably this:

```text
split on spaces
```

For a sentence like `the cat sat`, it works perfectly. Three words, three tokens.

Then real text shows up.

Is `cat.` the same token as `cat`? What about `cat,`? Where do we cut `don't`? Is it one token, `do` plus `n't`, or something else? What about URLs, hashtags, code, emoji, product names, and phrases like `New York`?

And English is the friendly case. Chinese and Japanese do not use spaces between words the way English does, so whitespace tokenization can fail before it even gets started.

Rule-based tokenizers tried to patch this with carefully written regular expressions. The Penn Treebank tokenizer is the classic example. It knows a lot about punctuation and contractions, and for many tasks it works well.

But the old problem comes back: every language and domain wants another rulebook. The more real the text becomes, the more edge cases appear.

Even if we somehow solved all the splitting rules, we would still hit the deeper wall.

Should the tokens be words, or should they be characters?

## If Tokens Are Words, Rare Words Break Everything

Word tokens are tempting because they are compact.

```text
the cat sat
```

is only three tokens. Each token is meaningful. `cat` is a nice unit. `sat` is a nice unit. A sentence stays short, which is good because shorter sequences are cheaper for Transformers to process.

But now we need a vocabulary of words.

That vocabulary gets ugly fast. English has hundreds of thousands of word forms. Then add names, typos, slang, domain terms, code identifiers, product names, and other languages. The list never really ends.

So a word-level model has to cap the vocabulary:

```text
keep the top 50,000 words
everything else -> <unk>
```

That creates the out-of-vocabulary problem, usually shortened to **OOV**.

The model may see all of these:

```text
Shakespeare
antidisestablishmentarianism
vikrantGPT2026
```

as the same token:

```text
<unk>
```

That is worse than "I do not know this word." It is "I cannot even tell which unknown word this was."

And language is full of rare words. Names are rare. New slang is rare. Misspellings are rare. Technical terms are rare. The long tail is not an edge case; it is a normal part of text.

Word-level tokenization gives us short sequences, but it makes the model blind to anything outside its fixed dictionary.

So maybe we should go all the way down.

## If Tokens Are Characters, Nothing Is Unknown, But Everything Gets Long

Character tokenization has the opposite personality.

```text
strawberry -> s t r a w b e r r y
```

Now there is basically no unknown-word problem. If you can spell it, the model can represent it.

That is a huge win. `Shakespeare`, a typo, a new username, and a made-up word are all just character sequences.

But now every sentence becomes much longer. A 20-word sentence might become 100 or 150 character tokens. For a Transformer, longer sequences are expensive because attention has to compare tokens with other tokens. More tokens means more memory, more time, and more money.

There is another cost too: a single character is not very meaningful. The letter `e` does not tell you much by itself. The model has to spend capacity learning how characters combine into words before it can even start modeling the sentence.

So the trade-off is painfully clean:

```text
word tokens       short and meaningful, but rare words collapse
character tokens  no unknown words, but long and low-level
```

When a trade-off is that clean, you start looking for a middle.

That middle is subword tokenization.

## Subwords Are the Compromise That Actually Works

The subword idea is the whole article in one sentence:

> Keep common words whole, and break rare words into reusable pieces.

Common words can stay compact:

```text
the
cat
because
```

Rarer words can split into pieces:

```text
tokenization -> token + ization
unhuggable   -> un + hug + gable
```

Now we get most of the word-level advantage for common text, because frequent words do not have to be spelled one character at a time. But we also get the character-level escape hatch for rare text, because unusual words can be decomposed.

This is why subwords became the standard choice for modern LLMs.

The tokenizer has one main knob: vocabulary size. A bigger vocabulary makes common text shorter, but creates more rare tokens that may not be learned well. A smaller vocabulary is more robust, but splits more aggressively and makes sequences longer.

There is no perfect setting. There is only a practical balance.

Now the question becomes: how does the tokenizer learn these pieces?

The easiest method to understand is BPE.

## BPE Learns by Merging What Appears Together

BPE stands for Byte-Pair Encoding. The name sounds more intimidating than the algorithm.

Start with characters. Count which adjacent pair appears most often. Merge that pair into a new token. Repeat.

That is basically it.

Let's use the tiny corpus from the Day 2 notes:

```text
hug  : 10
pug  : 5
pun  : 12
bun  : 4
hugs : 5
```

At the start, every word is split into characters:

```text
h u g       : 10
p u g       : 5
p u n       : 12
b u n       : 4
h u g s     : 5
```

BPE counts neighboring pairs. The pair `u g` appears in `hug`, `pug`, and `hugs`. Weighted by the word counts, it appears 20 times:

```text
hug  x 10
pug  x 5
hugs x 5
---------
20
```

So BPE creates a new token:

```text
u + g -> ug
```

Now the corpus is rewritten using that new piece:

```text
h ug        : 10
p ug        : 5
p u n       : 12
b u n       : 4
h ug s      : 5
```

Count again. This time `u n` wins, so we merge it:

```text
u + n -> un
```

Now we have:

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

After three merges, the tokenizer has learned this ordered list:

```text
1. u + g  -> ug
2. u + n  -> un
3. h + ug -> hug
```

That list is the trained tokenizer.

No gradient descent. No backpropagation. No GPU. Just counting pairs and merging the most common one.

The nice part is what happens next.

## A Word Can Be New Without Being Unknown

Suppose the tokenizer now sees a word it never saw during training:

```text
bug
```

A word-level tokenizer would panic unless `bug` was in the vocabulary. It might become `<unk>`.

BPE starts from characters:

```text
b u g
```

Then it applies the learned merges in order. The first rule says `u + g -> ug`, so:

```text
b u g -> b ug
```

Final result:

```text
[b, ug]
```

The word `bug` was new, but the pieces were not.

That is the whole trick. The model does not need a token for every possible word. It needs useful pieces that can recombine.

For a familiar word like `hugs`, the tokenizer gets even more compact:

```text
h u g s -> h ug s -> hug s
```

So `hugs` becomes:

```text
[hug, s]
```

Common chunks become single tokens. Rare combinations stay spellable.

This is the compromise we wanted.

## The Last Leak: What If Even the Character Is New?

There is still one small hole in the character version of BPE.

What if the tokenizer was trained mostly on English, then sees a Chinese character or an emoji it never had in its starting vocabulary?

That character could still be unknown.

GPT-style tokenizers solve this by going one level lower: bytes.

Every digital string is made of bytes, and there are only 256 possible byte values. If those 256 bytes are in the base vocabulary, then any text can be represented: English, Chinese, emoji, code, weird formatting, anything.

That is byte-level BPE. It is the "nothing is truly OOV" version.

The sequence may get longer for unfamiliar text, but it will not become impossible.

## BPE, WordPiece, Unigram, SentencePiece

BPE is the easiest subword method to explain, but it is not the only one.

**WordPiece**, used by BERT, is close to BPE but changes the merge rule. BPE asks, "Which pair is most frequent?" WordPiece asks, "Which pair is surprisingly attached, given how common the two pieces are separately?"

Its score is:

```text
score(a, b) = freq(ab) / (freq(a) * freq(b))
```

That denominator matters. It penalizes pieces that are common with everyone. BPE follows popularity; WordPiece follows association strength.

**Unigram LM** takes a different route. Instead of starting small and merging upward, it starts with a large set of possible pieces and prunes away the ones the corpus needs least. It treats tokenization probabilistically, so a word can have multiple possible segmentations with different probabilities.

That gives it a useful trick: during training, it can sample different segmentations of the same word. The model becomes less dependent on one exact cut.

**SentencePiece** is slightly different again. It is not just one algorithm; it is a tokenizer framework. Its big practical idea is to avoid pre-splitting text on spaces. It treats the input as raw text and represents whitespace explicitly with a visible marker:

```text
▁
```

That makes tokenization reversible and much more language-agnostic. This is why SentencePiece shows up in many multilingual and modern open-source models.

The details differ, but the shared idea is the same: use a fixed vocabulary of reusable pieces, with characters or bytes as the floor.

## Back to the Strawberry

Now the silly failure makes more sense.

When you ask:

```text
How many r's are in strawberry?
```

you are asking for a character-level operation. But the model may not have been handed character-level input. It may have received `strawberry` as one token, or as a few subword chunks, depending on the tokenizer.

So the model has to recover the spelling from its learned representation and then perform a counting task it was not directly optimized to do. Sometimes it gets it right. Sometimes it confidently answers like it is pattern-completing rather than counting.

Arithmetic has a similar flavor.

Humans see a number like:

```text
12345
```

as digits with place value. A tokenizer may split numbers in uneven chunks:

```text
12345  -> 123 + 45
10000  -> 100 + 00
987654 -> 98 + 765 + 4
```

Those splits are illustrative, not a claim about every tokenizer. The point is that the model is not guaranteed a clean digit-by-digit view of the number. Unless it has learned the right algorithms internally, or is allowed to call a calculator, arithmetic remains partly a learned text pattern.

Again, tokenization is not the entire explanation. Training data, architecture, prompting, reasoning strategies, and tools all matter. But tokenization is the first place where the human view of the problem and the model's view can diverge.

## Tokenization Also Changes the Cost

There is a practical side to this too: tokens are the unit commercial APIs usually charge for, and tokens are the unit Transformers process.

If one tokenizer turns a paragraph into 200 tokens and another turns it into 350, the second version is more expensive to run. The model has to process a longer sequence.

This shows up especially across languages. A tokenizer trained mostly on English may split non-English text into many more pieces. The text still works, but it costs more tokens.

One simple way to measure this is **fertility**:

```text
fertility = average tokens per word
```

Low fertility means compact tokenization. High fertility means the tokenizer is chopping heavily.

A good tokenizer is not just "the one with the biggest vocabulary." It is the one that keeps sequences reasonably short while still handling rare and messy text gracefully.

## The Perplexity Trap

This also changes how we evaluate language models.

In Day 1, perplexity was our scorecard:

```text
perplexity = 2 ^ average surprise
```

The phrase to watch is **average surprise per token**.

If tokenizers differ, then "per token" differs too. A character-level model gets one prediction per character. A word-level model gets one prediction per word. A subword model lives somewhere between them.

So a raw perplexity number is not universal. You cannot safely say:

```text
Model A has perplexity 20.
Model B has perplexity 40.
Therefore A is better.
```

Not unless you also know how the text was tokenized.

The cleaner comparison is to spread total surprise over a fixed unit, like characters or bytes:

```text
BPC = total surprise in bits / number of characters
```

This is called **bits per character**. The number of characters in a test passage does not change when the tokenizer changes, so BPC lets us compare across tokenization choices more honestly.

That connects all the way back to Shannon. His famous estimate that English carries around 1 bit per letter was already using a tokenizer-independent unit.

The rule I want to keep from this section is simple:

> Perplexity is meaningful only when the tokenizer is part of the story.

## What Tokenization Does Not Solve

Subword tokenization solves the brutal unknown-word problem. That is a big deal. A word-level model sees rare words as strangers; a subword model can break them into known pieces; a byte-level tokenizer can represent anything.

But tokenization does not create meaning.

To the tokenizer, these are just IDs:

```text
cat
dog
chair
7132
```

It does not know that cats and dogs are both animals. It does not know that `running` and `ran` are related. It does not know that `123` has place value.

Meaning comes later, when token IDs become vectors and the model learns from data.

So tokenization is only the first half of the input layer:

```text
text -> token IDs
```

Embeddings are the next half:

```text
token IDs -> vectors
```

That is where the roadmap goes next.

## The Takeaway

The sentence I wish I had learned earlier is:

> An LLM does not read text. It reads token IDs.

Once that clicks, a lot of strange behavior becomes less mysterious.

Word-level tokenizers struggle with rare words. Character-level tokenizers make everything long. Subword tokenizers sit in the middle. Byte-level BPE removes the last unknown-character problem. Raw perplexity can mislead you if two models use different tokenizers. And some "easy" tasks for humans, like counting letters in `strawberry`, are not as directly represented for the model as they are for us.

The game from Day 1 has not changed. We are still predicting the next unit and minimizing surprise.

Today we learned that the unit itself is a design choice.

And once that choice is made, the whole model has to live with it.

---

**Papers behind this article**

1. Manning, Raghavan, and Schutze, *Introduction to Information Retrieval*, Chapter 2 - the classical preprocessing pipeline.
2. Sennrich, Haddow, and Birch (2016), *Neural Machine Translation of Rare Words with Subword Units* - BPE for rare words in neural machine translation.
3. Schuster and Nakajima (2012), *Japanese and Korean Voice Search* - WordPiece.
4. Kudo and Richardson (2018), *SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing* - language-independent tokenization.
5. Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* - GPT-2 and byte-level BPE.

*Next: we implement BPE from scratch, test it on the tiny `hug` corpus, then compare token counts and OOV behavior on real text.*
