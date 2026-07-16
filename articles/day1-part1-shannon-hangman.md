# Claude Shannon Measured the English Language by Playing Hangman — and GPT Is Still Playing the Same Game

*Part 1 of "LLM from Scratch" — a series where we learn how language models work by building them, starting from 1948.*

---

ChatGPT can write code, pass exams, and argue about philosophy. But strip away the scale, and it is still doing exactly one thing:

> **Given the words so far, guess the next word — and say how confident you are.**

That's it. That's the whole job. And the first person to treat that job as mathematics wasn't an AI researcher. He was a telephone engineer, working in 1948, who ended up measuring the English language by making people play hangman.

This article is the origin story of language models. By the end you'll understand three ideas — surprise, entropy, and cross-entropy — that are still, today, the exact numbers GPT is trained to optimize. Nothing here requires more math than logarithms, and I'll build every idea from a guessing game you already know how to play.

## Before statistics: thirty years of writing rules by hand

From roughly 1960 to 1990, the mainstream plan for making computers understand language was: **language follows rules, so write the rules into the computer.** Grammar rules, dictionaries, expert systems. Hire linguists, encode what they know.

It failed, for three stubborn reasons.

**Real text is messy.** Typos, slang, headlines, brand-new words. No rulebook ever covered enough of it.

**Rules can't handle ambiguity.** Take the sentence *"I saw the man with the telescope."* Two perfectly grammatical readings: I used a telescope to see him, or he was holding the telescope. A rulebook can list both readings — but it has no way to say which one is *likely*. That's not a fact about grammar. It's a fact about how people actually talk, and you can only get it by counting.

**Rules don't scale.** Every new language, every new domain: more experts, more rules, more unpredictable interactions between rules.

The alternative idea — **don't write rules; count how real people actually use language and turn the counts into probabilities** — had been sitting in the literature since the 1940s. It just needed two things that only arrived in the 1980s: lots of digital text, and cheap computers. When IBM's speech recognition group showed that counting beat rule-writing on a real product, the field flipped sides almost overnight. The group's leader, Frederick Jelinek, summarized the war in one famous jab:

> *"Every time I fire a linguist, the performance of the speech recognizer goes up."*

The lesson wasn't "grammar is useless." The lesson was: **when language is ambiguous — which is always — probabilities learned from data beat rules written by hand.**

So what exactly is this counting approach? For that we go back to 1948.

## What is a language model?

First, the object we're building toward. Language is predictable. If you read "The cat sat on the ___", your brain has already filled the blank. If you see the letter `q`, you know `u` is coming.

A **language model** is just this guessing ability, turned into a machine:

> **A language model is a function that gives a probability to a piece of text.** Equivalently: given the words so far, it tells you how likely each possible next word is.

That one sentence is the entire field. Everything invented since — n-grams, RNNs, Transformers, GPT — is just a better and better way of using "the words so far."

## Shannon's first idea: rare news is big news

Claude Shannon's 1948 paper *A Mathematical Theory of Communication* was about sending messages over telephone lines. To do it, he needed to answer a strange question: **how much information does a message carry?**

His answer: it depends on how *unlikely* the message was. "The sun rose this morning" tells you nothing — you already knew. "It snowed in Death Valley" tells you a lot. So Shannon measured information as **surprise**:

```
surprise of an event = log2(1 / probability)     measured in "bits"
```

The smaller the probability, the bigger the surprise:

```
something certain          p = 1            0 bits
a coin lands heads         p = 1/2          1 bit
a die rolls 5              p = 1/6          ~2.6 bits
you draw the ace of spades p = 1/52         ~5.7 bits
you win the lottery        p = 1/1,000,000  ~20 bits
```

This works on letters and words too. After a `q`, the letter `u` appears ~97% of the time — seeing it is worth almost nothing (0.04 bits). Seeing a rare word like *aardvark* is a big surprise (~17 bits). **Predictable text is cheap; surprising text is expensive.** Hold that thought.

## Entropy: the average surprise (or: the guessing game)

One event has a surprise. A *source* of events — the weather, a stream of English text — has an **average** surprise. Shannon called that average **entropy**, and it answers: *on average, how unpredictable is this thing?*

Formulas are fine, but here is what entropy physically *means*. A "bit" is one perfect yes/no question. And entropy is simply:

> **The average number of yes/no questions you need to guess the next symbol.**

Let's use this to understand Shannon's most famous claim: **English text carries only about 1 bit per letter.**

**Step 1 — the worst case.** Imagine letters spat out by a lottery machine: 26 letters plus space, all equally likely, no memory. To guess the next one, your best strategy is to halve the candidates with each question ("Is it in A–M?"). With 27 candidates, that takes log2(27) ≈ **4.75 questions per letter**. That's the cost of pure gibberish, where nothing helps you.

**Step 2 — English is not a lottery machine.** Play the same game on real text. You're reading:

> The cat sat on the ma_

How many questions do you need? Basically zero — "Is it *t*?" Done. Or you see `q_` — one question. Only genuinely open positions, like the first letter of a new sentence, cost you 3 or 4 questions. Average over all positions, and English comes out around **1 question per letter**. Not 4.75.

**Step 3 — where did the other 3.75 questions go?** Spelling, grammar, and habit answered them *before you asked*. After `q`, nearly everything is ruled out. After "the ma", everything but *t*, *n*, *p* is dead. The structure of English silently answers about 80% of the questions for you. That's what it means to say English is **about 80% redundant**.

Why should you care? Two reasons:

- **This is why compression works.** English needs ~1 bit per letter; your computer stores 8. That gap is exactly what ZIP files squeeze out. It's also why "u cn stll rd ths" — the deleted letters carried almost no information.
- **This is why language models are possible at all.** A language model is a machine for exploiting predictability. If English really were a 4.75-bit lottery machine, "predict the next word" would be hopeless, and neither this series nor GPT would exist. *1 bit per letter is the mathematical proof that the task is winnable.*

## How he measured it: the staircase, and the hangman experiment

There's a catch: to compute entropy exactly, you need the true probabilities of English — and nobody knows them. Shannon's workaround was beautiful. Compute the entropy with **more and more context**, and watch where the numbers head. Each extra piece of context can only reduce your uncertainty, so the estimates descend like a staircase — and the true entropy is wherever the staircase bottoms out:

```
What you're told before guessing          Questions per letter
----------------------------------------  --------------------
nothing (all 27 symbols equal)             4.75
how common each letter is                  4.14
the previous letter                        3.56
the previous 2 letters                     3.3
how common each *word* is                  ~2.6
the previous 100 letters (a human, 1951)   0.6 – 1.3
```

The first rows he computed from letter-count tables — by hand, from books. But the staircase was clearly still falling, and tables couldn't go deeper: tracking just 10 letters of context needs 27^10 table entries.

So in 1951 Shannon replaced the tables with the best model of English in existence: **a person.** He showed people text cut off mid-sentence and had them guess the next letter until they got it right, counting the guesses. Mostly the tally reads 1, 1, 1, 2, 1... — instant hits, with occasional hard spots at the start of new words. From those guess counts he computed the last row of the staircase.

**The entropy of English was literally measured by humans playing hangman.**

And here's the part that gives me chills: that staircase is *still being descended today*. Compress text using a modern LLM's predictions and you get ~0.6–0.7 bits per letter — right at the bottom of the range Shannon measured with human guessers in 1951. Count tables, humans, GPT: one long walk down the same staircase, on an axis Shannon drew 75 years ago.

## Cross-entropy: the guessing game with wrong beliefs

One more idea and the foundation is complete. Notice a hidden assumption in the guessing game: you knew the true probabilities, so you always asked the smartest question first. Entropy is the cost of the game **when your beliefs are perfect**.

A model's beliefs are never perfect. So:

> **Cross-entropy is the average number of questions you need when reality picks the outcomes, but *your model's beliefs* pick the questions.**

Wrong beliefs → you ask about the wrong things first → you waste questions.

Let's play it out with a toy example. A city's weather: sun half the time, rain a quarter, snow an eighth, hail an eighth. Playing optimally ("sun?" first), the game costs **1.75 questions per day** — that's the entropy.

Now suppose your model believes the *exact opposite*: hail half, snow quarter, rain eighth, sun eighth. Trusting it, you ask "hail?" first, then "snow?", then "rain?":

```
Actual weather   How often   Your questions                Cost
--------------   ---------   ---------------------------   ----
hail             1/8         "hail?" yes                    1
snow             1/8         "hail?" no, "snow?" yes        2
rain             1/4         no, no, "rain?" yes            3
sun              1/2         no, no, no → must be sun       3
```

See the disaster? Your cheap 1-question shortcut is reserved for hail — which almost never happens — while sun, *half of all days*, costs 3 questions every time. Average: **2.625 questions per day**, versus the 1.75 the weather actually requires. The extra 0.875 is the pure price of believing the wrong thing.

And notice who contributes what: **reality decides how often each row happens; your model decides how expensive each row is.** That's the whole formula. In general:

```
cross-entropy  =  entropy                +  extra
                  (how unpredictable      (how wrong
                   the world is)           your model is)
```

Here is why this matters more than anything else in this article: **training a language model — a 1950s count table or GPT-4 — means exactly one thing: shrinking that "extra" term.** The entropy part is fixed; it belongs to the language. The wrongness part is the model's. When you hear that GPT was trained to "minimize cross-entropy," you now know precisely what that sentence means: it played the guessing game on billions of sentences, and every wasted question was a nudge to its weights.

## Shannon's dice: the first text generator

Shannon's last demo flipped everything around. If a model can tell you how likely each next letter is, it can also *write* — just roll dice according to those probabilities. Scoring and generating are the same machine, pointed in opposite directions.

He built a series of generators, each remembering one more symbol than the last, and let each one write. Here is what they produced (from the 1948 paper):

```
Memory                      Sample output
-------------------------   ------------------------------------------
none (random symbols)       XFOML RXKHRJFFJUJ          pure junk
letter frequencies          OCRO HLI RGWR              right mix, no structure
1 previous letter           ON IE ANTSOUTINYS          pronounceable chunks
2 previous letters          IN NO IST LAT WHEY         almost words
previous word               grammatical phrases that wander off topic
```

Every extra symbol of memory makes the output visibly more English. This table from 1948 is the founding demo of language modeling — and when ChatGPT streams its reply to you, it is running the exact same loop: look at context, get a probability for every possible next token, roll the dice, repeat. Bigger table, longer memory, same dice.

You can build the last row yourself in ten lines. Take a tiny "training corpus":

```
the cat sat
the cat ran
the dog sat
```

Count what follows each word: after "the" → cat (2/3) or dog (1/3); after "cat" → sat or ran (half each); after "dog" → sat (always). Now start at "the", roll a weighted die, write the word, move to it, repeat. You'll get "the dog sat", or "the cat ran" — fluent little sentences from nothing but counting. Congratulations: that table of counts *is* a language model.

But notice the flaw in Shannon's last row: "grammatical phrases that **wander off topic**." The generator reads fine three words at a time and drifts globally — because it literally cannot remember anything outside its tiny window. File that away. It's about to become the central problem of the next forty years.

## The field makes it official

Two milestones turned Shannon's ideas into a discipline.

**1993 — the manifesto.** Kenneth Church and Robert Mercer's *Computational Linguistics Using Large Corpora* — no famous formula, just the field's formal declaration that the counting approach had won: huge digital text collections now exist, so we can *measure* how language behaves instead of theorizing about it, and the measurements are beating the rulebooks.

**1997 — the proof in production.** Jelinek's book *Statistical Methods for Speech Recognition* documented the IBM system that made it undeniable. Speech recognition, it turns out, is the perfect demonstration of why language models must exist. Spoken aloud, **"recognize speech"** and **"wreck a nice beach"** sound nearly identical. No amount of audio analysis can tell them apart. The only thing that can is knowing *which word sequence is more plausible English* — a language model. IBM's system multiplied two scores: "does the audio sound like these words?" times "do these words read like real English?" — and the second score is exactly the counting machine this article has been building up to. (The machinery wrapping it, the Hidden Markov Model, deserves its own article.)

## What's next

Here's where we've landed. A language model assigns probabilities to text. Its quality is measured by the guessing game — cross-entropy, the average surprise, the exact quantity GPT still minimizes. And even a table of counted word pairs can score and generate language.

But I've skipped the practical question: **where do the probabilities actually come from, and what happens when you try this on real data?**

In Part 2, we build it: a real language model, trained on a million words of English, from scratch. I'll warn you now how it goes: the first version will declare 446 out of 500 perfectly ordinary sentences *impossible*, and fixing that will teach us more than getting it right the first time would have. We'll also meet the single number used to grade every language model from 1977 to GPT — perplexity.

*[Link to Part 2]*

---

**Papers behind this article:**

1. Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal — surprise, entropy, and the dice generators.
2. Shannon, C. E. (1951). *Prediction and Entropy of Printed English* — the hangman experiment.
3. Church, K. & Mercer, R. (1993). *Introduction to the Special Issue on Computational Linguistics Using Large Corpora* — the empiricist manifesto.
4. Jelinek, F. (1997). *Statistical Methods for Speech Recognition* — the ideas in production.

*All code for this series lives at [repo link]. Next: Part 2 — building and breaking a real n-gram model.*
