# Day 1 — Evolution of NLP: From Rules to Statistical Language Models

---

## 1. Why this topic matters

Today's LLMs like ChatGPT can chat, reason, and write code. But under all of that, every one of them is still answering one simple question:

> **Given the words so far, what word comes next — and how likely is it?**

That question was first written down as math in 1948, long before neural networks were practical. It is worth learning the original version, for three reasons:

1. **The goal never changed.** GPT is trained the same way the old models were: it predicts the next word, and it gets punished based on how surprised it was by the true next word. Shannon invented that measure of surprise in 1948. We will meet it in this article under the name **cross-entropy**.
2. **The test never changed.** Every new model today still reports a score called **perplexity** — a number that says how confused the model is when reading real text. It was invented in this era too. We will build up to it step by step.
3. **The failures of the old models explain the new ones.** Every weakness of the simple models in this article led directly to a later invention (embeddings, RNNs, attention). Day 1 is the "why" behind the whole roadmap.

## 2. Before statistics: the rule era, and why it failed

From roughly 1960 to 1990, the main idea in NLP was: **language follows rules, so let's write the rules into the computer by hand.** Grammar rules, dictionaries, expert systems. This is called the **rationalist** approach.

It failed for three practical reasons:

- **Coverage.** Real text is messy — typos, slang, headlines, new words. No rulebook could ever cover it all.
- **Ambiguity.** Rules can list all the possible meanings of a sentence, but they cannot say which meaning is *likely*. Take *"I saw the man with the telescope."* Grammar allows two readings: I used a telescope to see the man, or the man was holding a telescope. Both are correct English. A rulebook has no way to pick one. To pick, you need to know which reading is more *common* in real life — and that is a fact about data, not grammar.
- **Cost.** Every new language and every new domain needed experts writing new rules, and big rulebooks broke in surprising ways as they grew.

The opposite idea is called the **empiricist** approach: don't write rules — **count how real people actually use language, and turn the counts into probabilities.** This idea existed since the 1940s, but it needed two things that only arrived in the 1980s: lots of text in digital form, and cheap computers. When IBM's speech recognition group (led by Frederick Jelinek) showed that counting beats rule-writing, the whole field switched sides. Jelinek's famous joke sums up the war:

> *"Every time I fire a linguist, the performance of the speech recognizer goes up."*

The lesson was not "grammar is useless." The lesson was: **when language is ambiguous — which is always — probabilities learned from data beat rules written by hand.**

## 3. What is a language model?

Language is predictable. If you read "The cat sat on the ___", your brain has already guessed the answer. If you see the letter `q`, you already know `u` is coming. A **language model** is just this guessing ability, turned into a machine:

> **A language model is a function that gives a probability to a piece of text.** Equivalently: given the words so far, it says how likely each possible next word is.

That one sentence is the entire field. Everything that came after — n-grams, RNNs, Transformers, GPT — is just a better and better way to use "the words so far."

### Two toy examples we will reuse everywhere

To keep things easy to follow, every number in this article comes from one of two tiny made-up examples. Meet them once, here:

**The weather city.** A city where each day's weather follows fixed probabilities:

| Outcome | sun | rain | snow | hail |
| ------- | --- | ---- | ---- | ---- |
| Probability | 1/2 | 1/4 | 1/8 | 1/8 |

**The cat corpus.** Our entire "training data" is three sentences:

```text
the cat sat
the cat ran
the dog sat
```

It is ridiculously small on purpose: you can check every number in this article in your head. When we later use a real dataset, only the size changes — the method stays identical.

One small convention: we mark where sentences start and end with special symbols, written `⟨s⟩` (start) and `⟨/s⟩` (end). So "the cat sat" becomes `⟨s⟩ the cat sat ⟨/s⟩`. The reason will become obvious when we start counting.

## 4. Shannon (1948): measuring surprise, and the first language models

Claude Shannon's paper *A Mathematical Theory of Communication* was about sending messages over telephone lines. But to do that, he had to answer a deeper question — *how much information does a message carry?* — and his answer became the foundation of language modeling.

### 4.1 Surprise: rare news is big news

Shannon's starting point: **how much a message tells you depends on how unlikely it was.** "The sun rose this morning" tells you nothing — you already knew. "It snowed in Death Valley" tells you a lot — you did not see it coming. So he measured the information in an event by its surprise:

$$\text{surprise of event } x = \log_2 \frac{1}{p(x)} \quad \text{(measured in bits)}$$

Read it simply: **the smaller the probability $p$, the bigger the surprise.** Some examples:

| Event | Probability | Surprise |
| ----- | ----------- | -------- |
| something certain | 1 | 0 bits |
| a coin lands heads | 1/2 | 1 bit |
| a die rolls 5 | 1/6 | about 2.6 bits |
| you draw the ace of spades | 1/52 | about 5.7 bits |
| you win a 1-in-a-million lottery | 1/1,000,000 | about 20 bits |

The same idea works on letters and words. After a `q`, the letter `u` appears about 97% of the time — so seeing it is worth almost nothing (0.04 bits). Seeing a rare word like *aardvark* is a big surprise (about 17 bits). **Predictable text is cheap. Surprising text is expensive.** Most of language is cheap — and that fact is why language models can exist at all.

### 4.2 Entropy: the average surprise

One event has a surprise. A *source* of events — the weather, a stream of English letters — has an **average** surprise. That average is called **entropy**:

$$H = \sum_x p(x) \cdot \log_2 \frac{1}{p(x)} = \text{(probability of each outcome)} \times \text{(its surprise), summed up}$$

Entropy answers: **"on average, how unpredictable is this source?"**

Let's compute one by hand, using our weather city (sun 1/2, rain 1/4, snow 1/8, hail 1/8):

$$H = \tfrac{1}{2}(1) + \tfrac{1}{4}(2) + \tfrac{1}{8}(3) + \tfrac{1}{8}(3) = 1.75 \text{ bits per day}$$

Each term is easy to read: sun has probability 1/2 and surprise 1 bit; hail has probability 1/8 and surprise 3 bits. Sunny days are cheap and common; hail is expensive but rare. Entropy is the average bill.

Now compare: if all four weathers were equally likely, entropy would be 2 bits — the maximum. Our lopsided city scores only 1.75. **Same four outcomes, less unpredictability — because one outcome dominates.** The more lopsided the probabilities, the lower the entropy. A fair coin has entropy 1 bit; a trick coin that lands heads 90% of the time has entropy 0.47 bits; a two-headed coin has entropy 0 — no surprise at all.

### What entropy *really* means: the guessing game

Here is the most concrete way to understand entropy. A "bit" is not an abstract unit — **one bit = one yes/no question.** And entropy is simply:

> **The average number of yes/no questions you need to guess the next symbol.**

Let's use this to understand a famous claim of Shannon's: *English text carries only about 1 bit per letter.*

**Step 1 — the worst case.** Imagine letters coming from a lottery machine: 26 letters plus space, all equally likely, no memory. To guess the next symbol, your best strategy is to cut the candidates in half with each question: "Is it in A–M?" → "Is it in A–F?" → ... With 27 candidates, you need $\log_2 27 \approx 4.75$ questions. So **random gibberish costs about 4.75 questions per letter.** Nothing helps you; you must ask everything.

**Step 2 — English is not a lottery machine.** Play the same game on real text. You are reading:

> The cat sat on the ma\_

How many questions do you need? Zero, basically — "Is it *t*?" Yes. Done. Or:

> q\_

One question: "Is it *u*?" Yes, almost always. Only genuinely open spots — like the first letter of a brand-new sentence — cost you 3 or 4 questions. Average over all positions, and English comes out around **1 question per letter**, not 4.75.

**Step 3 — what happened to the other 3.75 questions?** Spelling, grammar, and habit answered them *for you, before you asked*. After a `q`, nearly every letter is already ruled out. After "the ma" in that sentence, everything except *t*, *n*, *p* is dead. Out of 4.75 questions' worth of doubt, the structure of English silently answers about 3.75, leaving you roughly 1. That is what it means to say English is about **80% redundant**.

**Why should you care?**

- **This is why compression works.** English only needs ~1 bit per letter, but your computer stores 8 bits per letter. That wasted space is exactly what ZIP files squeeze out — text compresses about 5×. It is also why "u cn stll rd ths": the deleted letters carried almost no information, so your brain fills them back in.
- **This is why language models are possible.** A language model is a machine that exploits predictability. If English really were random gibberish, "predict the next word" would be a hopeless task and none of this field would exist. "1 bit per letter" is the mathematical proof that the task is winnable.

### How Shannon actually measured "1 bit per letter"

There's a problem with the claim above: to compute entropy exactly you need the true probabilities of English — and nobody knows them. Shannon's solution was clever: **compute the entropy with more and more context, and watch where the numbers head.** Each extra piece of context can only make you less uncertain, so the estimates keep falling — like walking down a staircase. The true entropy is wherever the staircase bottoms out:

| What you're told before guessing | Questions per letter |
| -------------------------------- | -------------------- |
| nothing — all 27 symbols equal | 4.75 |
| how common each letter is overall | 4.14 |
| the previous letter | 3.56 |
| the previous 2 letters | 3.3 |
| how common each *word* is | ~2.6 |
| the previous 100 letters (a human guesser, 1951) | **0.6 – 1.3** |

The first rows he computed from letter-count tables — by hand, from books. But the staircase was clearly still falling, and the tables couldn't go deeper (tracking 10 letters of context would need $27^{10}$ table entries). So in a 1951 follow-up paper, Shannon replaced the tables with the best model of English available: **a person.** He showed people a passage cut off mid-sentence and had them guess the next letter until they got it right, counting the guesses. Mostly the counts read 1, 1, 1, 2, 1... — instant hits — with occasional hard spots at the start of new words. From these guess counts he computed the last row of the table. **The entropy of English was literally measured by humans playing hangman.**

Two things about this staircase are worth remembering. First, every row is really a *model* of English — and its number is that model's average surprise, which is always somewhat *above* the true entropy. Better model → smaller number. Second, the staircase is still being walked down *today*: compressing text with a modern LLM gets to about 0.6–0.7 bits per letter, right at the bottom of the range Shannon measured with humans in 1951. From count tables to GPT, it's all one long descent down the same staircase.

One warning before the next idea. Entropy is a property of **the language itself** — the unpredictability that no model, however good, can remove. It doesn't measure any model. To measure a *model*, we need a twin concept.

### 4.3 Cross-entropy: the guessing game with wrong beliefs

In the guessing game above, we quietly assumed you knew the true probabilities — you always asked the smartest question first ("sun?" — because sun really is the most common). So entropy is the cost of the game **when your beliefs are perfect.**

But a model's beliefs are never perfect. That's where cross-entropy comes in:

> **Cross-entropy is the average number of questions you need when reality picks the outcomes, but *your model's beliefs* pick the questions.** Wrong beliefs → you ask about the wrong things first → you waste questions.

Let's play it out. Reality is our weather city: sun 1/2, rain 1/4, snow 1/8, hail 1/8 — entropy 1.75 questions per day. But suppose your model believes the *exact opposite*: hail 1/2, snow 1/4, rain 1/8, sun 1/8. Trusting the model, you ask "hail?" first, then "snow?", then "rain?":

| Today's actual weather | How often this happens | Your questions | Cost |
| ---------------------- | ---------------------- | -------------- | ---- |
| hail | 1/8 | "hail?" ✓ | 1 |
| snow | 1/8 | "hail?" ✗ "snow?" ✓ | 2 |
| rain | 1/4 | "hail?" ✗ "snow?" ✗ "rain?" ✓ | 3 |
| sun | 1/2 | "hail?" ✗ "snow?" ✗ "rain?" ✗ → must be sun | 3 |

See the disaster: your cheap 1-question shortcut is saved for hail, which almost never happens — while sun, *half of all days*, costs you 3 questions every single time. Average it out:

$$\tfrac{1}{2}(3) + \tfrac{1}{4}(3) + \tfrac{1}{8}(2) + \tfrac{1}{8}(1) = 2.625 \text{ questions per day}$$

The weather is only 1.75 questions unpredictable — but *you* are paying 2.625. The extra 0.875 questions per day is the pure price of believing the wrong thing. And notice who contributes what: **reality decides how often each row happens; your model decides how expensive each row is.**

That last sentence *is* the formula. Reality's probability $p(x)$ weights each outcome; your model's belief $q(x)$ sets its price:

$$H(p, q) = \sum_x p(x) \cdot \log_2 \frac{1}{q(x)}$$

It's the entropy formula with one change: the probability inside the log is the *model's*, not reality's. Hence the name "cross" — two different probability lists crossed together: reality picks, your model pays.

For language there is one practical adjustment. We don't know the true probabilities of English — but we have the next best thing: real text. So we let real sentences play the role of reality, and average the model's surprise over $N$ words of test text:

$$H = -\frac{1}{N}\sum_{i=1}^{N} \log_2 P_{\text{model}}(w_i \mid \text{previous words}) \quad \text{(bits per word)}$$

In plain words: **walk through the text word by word, ask the model "how likely did you think THIS word was?", take the surprise, and average.**

The relationship between the two concepts, in one line:

$$\text{cross-entropy} = \underbrace{\text{entropy}}_{\text{how unpredictable the world is}} + \underbrace{\text{extra}}_{\text{how wrong your model is}}$$

Cross-entropy can never be below entropy, and equals it only for a perfect model. **Training a language model — a 1950s count table or GPT — means one thing: shrinking that "extra" part.** The entropy part is fixed; it belongs to the language, not the model.

**Let's do it on an actual sentence.** A model reads the test sentence `⟨s⟩ the cat sat ⟨/s⟩`, one word at a time. At each step, we check what probability the model gave to the word that actually came (these model probabilities are made-up round numbers so the math is easy to follow):

| Words so far | Actual next word | Model's probability | Surprise |
| ------------ | ---------------- | ------------------- | -------- |
| `⟨s⟩` | the | 0.25 | 2 bits |
| the | cat | 0.125 | 3 bits |
| cat | sat | 0.5 | 1 bit |
| sat | `⟨/s⟩` | 0.25 | 2 bits |

Cross-entropy = the average of the surprise column = (2+3+1+2)/4 = **2 bits per word**. The third row is the model's best moment: after "cat" it strongly expected "sat", so it was barely surprised. The second row is its worst. A better model is simply one that puts higher probability on the words that actually show up — lower average surprise.

One trap to notice, because it comes back later: what if the model had given "sat" probability **zero**? Then that row's surprise would be *infinite*, and no amount of good rows could rescue the average. Remember this.

### 4.4 The first text generators: Shannon's dice

So far, models have only *judged* text — guessing letters, measuring surprise. Shannon's last demo flipped the direction: **if a model can say how likely each next symbol is, it can also *write* — just roll dice according to those probabilities.** Judge and writer are the same machine, pointed different ways.

The models he used are called **Markov sources** — a fancy name for a simple thing: a text generator with a very short memory. It spits out one symbol at a time, and its choice depends only on the last few symbols; everything earlier is forgotten. (The name honors A. A. Markov, who in 1913 counted vowel/consonant patterns in a Pushkin poem — the first statistical study of language ever.) Building one is nothing but counting: go through real English text, tally which symbol follows which, turn the tallies into probabilities. Then let it roll.

Shannon built a series of these, each with one more symbol of memory, and let each one write. Read the samples from top to bottom:

| Memory | Sample output |
| ------ | ------------- |
| none — random symbols | `XFOML RXKHRJFFJUJ` — pure junk |
| letter frequencies only | `OCRO HLI RGWR` — the right *mix* of letters, no structure |
| 1 previous letter | `ON IE ANTSOUTINYS` — pronounceable chunks |
| 2 previous letters | `IN NO IST LAT WHEY` — almost words |
| previous word | grammatical-looking phrases that wander off topic |

Every extra symbol of memory makes the output visibly more English. This little table from 1948 is the founding demo of language modeling — and notice the last row's flaw: the phrases read fine three words at a time but drift globally, **because the generator literally cannot remember anything outside its tiny window.** Hold that thought; it becomes the central weakness of everything in this article.

**Try it yourself with the cat corpus.** Our training data: "the cat sat", "the cat ran", "the dog sat". Count what follows each word:

- after `the`: cat 2 times out of 3, dog 1 out of 3
- after `cat`: sat half the time, ran half the time
- after `dog`: sat always

Now generate: start at `the`, roll a die weighted by those counts, write down the winner, move to it, repeat. You might get "the dog sat", or "the cat ran" — both fluent, the dice decide. This little table of counts **is** a language model, built with nothing but counting. And when a modern LLM writes its reply to you, it is running this exact loop — with a memory of thousands of words instead of one, and a vastly bigger (learned, not counted) probability table. Shannon's dice, still rolling.

## 5. Church & Mercer (1993): the field officially switches sides

*Computational Linguistics Using Large Corpora* contains no famous formula — it's a **position paper**, the introduction to a special journal issue, announcing that the counting approach had won. Its argument, in three lines:

- Huge amounts of digital text now exist.
- So we can *measure* how language behaves instead of theorizing about how it should behave.
- And the measured approach is already beating the rulebooks — in speech, and increasingly in translation and tagging.

Cite this paper as **the moment statistical NLP became mainstream** — not as the source of any formula. The math itself is older: Shannon, and Markov before him.

## 6. Jelinek (1997): statistics meets a real product — speech recognition

Jelinek's book *Statistical Methods for Speech Recognition* documents the IBM system that proved all these ideas on a real, hard, commercial problem. Speech recognition is the perfect showcase for why language models must exist, because of one stubborn fact:

> Spoken aloud, **"recognize speech"** and **"wreck a nice beach"** sound nearly identical. No amount of audio analysis can fully tell them apart. The only thing that can is knowing *which word sequence is more plausible English.*

### 6.1 Splitting the problem in two

The task: given audio $A$, find the word sequence $W$ that most likely produced it. Using Bayes' rule (a standard probability identity), that goal splits cleanly into two parts:

$$\hat{W} = \arg\max_W \; \underbrace{P(A \mid W)}_{\text{acoustic model}} \times \underbrace{P(W)}_{\text{language model}}$$

In plain words, the best guess is the word sequence that wins on **two tests at once**:

- **Acoustic model:** *does the audio sound like these words?*
- **Language model:** *do these words sound like real English?* — this is exactly the counting model from this article.

"Wreck a nice beach" passes the first test but flunks the second — English text almost never says it. That's how the right answer wins. This two-part recipe became the template for machine translation and much more, and the language model is the reusable half — which is why it became the center of the whole field.

### 6.2 Hidden Markov Models: guessing what you cannot see

**The problem.** Shannon's generators work when you can *see* the sequence you're modeling. But in speech recognition, you never see the words — you only hear sound. The thing you want (words) is **hidden**; the thing you have (audio) is just noisy evidence about it. A **Hidden Markov Model (HMM)** is the classic machine for this situation. It has two layers:

1. **A hidden layer** that moves from state to state step by step — you never see it.
2. **A visible layer**: at each step, the hidden state *leaks* one piece of evidence you can see.

The machine's job: from the visible evidence alone, work out the most likely hidden story behind it.

**A small example.** You work in a windowless office. You cannot see the weather (hidden: `sun` or `rain`). All you see is whether each coworker walks in with an umbrella (visible: `umbrella` or `none`). Two small tables describe the whole world:

*How the hidden weather moves* (weather has momentum — sunny spells and rainy spells):

| from \ to | sun | rain |
| --------- | --- | ---- |
| **sun** | 0.8 | 0.2 |
| **rain** | 0.4 | 0.6 |

*How the hidden weather leaks evidence:*

| weather \ you see | umbrella | none |
| ----------------- | -------- | ---- |
| **sun** | 0.2 | 0.8 |
| **rain** | 0.9 | 0.1 |

(And on the first day, sun or rain start at 50/50.)

**Now let's solve one by hand.** Two mornings in a row, coworkers arrive carrying umbrellas. What was the weather? Check every possible two-day weather story, multiplying (start chance × evidence × weather change × evidence):

| Weather story | Multiply it out | Probability |
| ------------- | --------------- | ----------- |
| sun → sun | 0.5 × 0.2 × 0.8 × 0.2 | 0.016 |
| sun → rain | 0.5 × 0.2 × 0.2 × 0.9 | 0.018 |
| rain → sun | 0.5 × 0.9 × 0.4 × 0.2 | 0.036 |
| rain → rain | 0.5 × 0.9 × 0.6 × 0.9 | **0.243** |

Rain both days, and it's not even close. Look at *why* it wins: rain explains the umbrellas well, **and** rain-after-rain is a believable weather story. A winning story must pass two tests at once — *match the evidence* and *be plausible on its own*. That should sound familiar: it's the acoustic model and the language model again, working inside one machine.

**The scaling problem, and the three classic algorithms.** With 2 states and 2 days there were only 4 stories to check. Real speech has thousands of states and hundreds of time steps — the number of stories explodes beyond anything computable. HMMs conquered the field because all three natural questions have fast shortcuts, based on one observation: *two stories that pass through the same state share their future, so you can score each state once per step instead of once per story.*

| Question | Algorithm |
| -------- | --------- |
| How likely is this evidence overall? | **Forward** |
| What's the single most likely hidden story? | **Viterbi** — this *is* recognition |
| How do I learn the two tables from raw data? | **Baum–Welch** |

Viterbi is just our brute-force table with pruning: at each step, for each state, keep only the best story that reaches it, and throw the rest away — extending a losing story can never beat extending the winning one.

**Mapping it to speech:** hidden states = the words being spoken; the "weather momentum" table = which words follow which — *that's the language model*; the "evidence leak" table = what sounds each word makes — the acoustic model. Recognition = Viterbi finding the word story that both sounds right and reads like English. "Recognize speech" beats "wreck a nice beach" on the momentum table, not the sound table.

**One idea to file away.** The heart of the HMM is *a hidden state that summarizes the past and gets carried forward one step at a time*. Replace the hand-built probability tables with a learned vector, and you have invented the RNN — which we meet on Day 7. The future was hiding inside 1980s speech recognition.

### 6.3 Perplexity: how do you grade a language model?

Jelinek's team had a practical problem: they kept building competing language models, and they needed **one number** to say which was better — without running an entire expensive speech recognizer every time. The number they invented (in 1977) is called **perplexity**, and it is still the standard scorecard for every language model today, GPT included.

**What are we actually trying to measure?** A language model's whole job is to give high probability to text that people actually write. So the test is natural: take real text the model has **never seen before**, walk through it word by word, and check — *how likely did the model think each actual next word was?* A good model keeps saying "yes, I expected that" (high probabilities). A bad model keeps being surprised (low probabilities).

**The definition, in plain words:**

> **Perplexity measures how confused the model is while reading real text. A perplexity of K means: at each word, the model was as unsure as if it were choosing blindly among K equally likely words.**

- Perplexity 4 → reading felt like picking from 4 options per word. Pretty confident.
- Perplexity 100 → like rolling a 100-sided die at every word. Very confused.
- Perplexity 1 → a perfect oracle that always knows the next word. **Lower is better.**

The name is literal: it measures how *perplexed* the model is. Confidence (when it's justified by being right) scores well; confusion scores badly. In the next section we'll build a model from scratch and end by computing its perplexity — including the exact formula, which turns out to be nothing more than the guessing-game score (average surprise per word) converted back into a count of options.

Everything above now becomes one buildable machine. The plan, in one paragraph: we want the probability of a whole sentence (7.1) — computing it exactly needs statistics nobody can collect, so we take a shortcut (7.2) — the shortcut makes training literally just counting (7.3) — counting has one fatal flaw (7.4) — we patch it, then grade the model with perplexity (7.5).

Our running data, as always, is the cat corpus: *"the cat sat"*, *"the cat ran"*, *"the dog sat"*. We will train on it by hand, break the model, fix it, and score it.

### 7.1 Step one: the probability of a sentence

How likely is the sentence "the cat sat"? Peel it one word at a time, multiplying as you go:

$$P(\text{the cat sat}) = P(\text{the}) \times P(\text{cat} \mid \text{the}) \times P(\text{sat} \mid \text{the cat})$$

Read it like a story: how likely does a sentence start with "the"? — times — given "the", how likely is "cat" next? — times — given "the cat", how likely is "sat"? This works for any sentence of any length:

$$P(w_1 w_2 \cdots w_n) = \prod_{i=1}^{n} P(w_i \mid \text{all the words before } w_i)$$

This step is called the **chain rule**, and it is important to see that it is *not* an approximation — it's just algebra, always exactly true. It rewrites "the probability of a sentence" as "the probability of each next word given everything before it." Which is pleasing, because *predict the next word from what came before* is exactly what we said a language model is.

**But there's a killer hiding in it.** Look at the last factor: probability of "sat" given "the cat". To estimate that from data, you find every "the cat" in your corpus and check what followed. Fine — it's a short phrase, it appears a lot. But the chain rule needs this for the *whole* history: by word 15 of a sentence, you'd need to find a specific 14-word phrase in your data. Try googling any exact 15-word sentence from today's news, in quotes: zero results. **Almost every long sequence of words has never been written before.** The counts we need don't exist. The formula is exactly right and completely unusable.

### 7.2 Step two: the shortcut (the Markov assumption)

The fix: **pretend the next word only depends on the last one or two words, and forget everything earlier.** This is the same short-memory trick as Shannon's generators, now used for scoring:

- **Bigram model** (memory = 1 word): $P(\text{sat} \mid \text{the cat}) \approx P(\text{sat} \mid \text{cat})$
- **Trigram model** (memory = 2 words): condition on the previous two words instead.

**What we gain:** short phrases repeat! "the cat" appears twice even in our 9-word corpus. The counts we need suddenly exist. The model becomes buildable.

**What we lose:** everything outside the memory window. A concrete casualty: in *"The keys to the cabinet **are** on the table"*, the verb must be "are" because of "keys" — three words back. A bigram model, deciding between *are* and *is*, sees only "cabinet" — and confidently picks the wrong verb. This is exactly why Shannon's generated phrases wandered off topic: no memory outside the window. And fixing *this one flaw* — more memory without impossible statistics — is the story of the entire rest of the roadmap: RNNs, LSTMs, attention.

### 7.3 Step three: training = counting

Where does a number like $P(\text{cat} \mid \text{the})$ come from? Ask the corpus the obvious question: **"out of all the times 'the' appeared, how often was 'cat' the next word?"**

$$P(\text{cat} \mid \text{the}) = \frac{\text{count}(\text{the cat})}{\text{count}(\text{the})}$$

Before counting, one practical step: wrap every sentence in the start/end markers, like `⟨s⟩ the cat sat ⟨/s⟩`. Now the first word has something before it to be predicted from, and the model can learn how sentences *end*. This is why the markers exist.

Let's fully train on the cat corpus — count every adjacent word pair:

| Word pair | Count | | Single word | Count |
| --------- | ----- | - | ----------- | ----- |
| `⟨s⟩` the | 3 | | `⟨s⟩` | 3 |
| the cat | 2 | | the | 3 |
| the dog | 1 | | cat | 2 |
| cat sat | 1 | | dog | 1 |
| cat ran | 1 | | sat | 2 |
| dog sat | 1 | | ran | 1 |
| sat `⟨/s⟩` | 2 | | | |
| ran `⟨/s⟩` | 1 | | | |

Divide, and the model is trained: $P(\text{cat}\mid\text{the}) = 2/3$, $P(\text{dog}\mid\text{the}) = 1/3$, $P(\text{sat}\mid\text{cat}) = 1/2$, $P(\text{the}\mid ⟨s⟩) = 1$. That's the whole thing. **In 1990, "training a language model" meant: count the pairs, divide.** No gradients, no GPUs, one pass over the data.

(Why is this called "maximum likelihood estimation"? Because you can prove that these divided counts are the probabilities that make your training data as likely as possible — no other numbers fit the data more literally. Keep the phrase "most literally" in mind; it's about to become the problem.)

### 7.4 Step four: the fatal flaw, and smoothing

Let's score a brand-new sentence with our freshly trained model: **"the dog ran."** Perfectly good English — dogs run. Multiply through:

$$P = \underbrace{P(\text{the} \mid ⟨s⟩)}_{1} \times \underbrace{P(\text{dog} \mid \text{the})}_{1/3} \times \underbrace{P(\text{ran} \mid \text{dog})}_{\mathbf{0}} \times \cdots = 0$$

The pair "dog ran" never appeared in our three training sentences. So the model doesn't say the sentence is *unlikely* — it says it is **impossible**. Probability zero. And zero poisons everything: this is the trap from the cross-entropy table earlier — a zero-probability word has *infinite* surprise, so the model's score on the whole test set becomes infinite. One missing word pair wipes out the entire evaluation.

Could we just hope unseen pairs are rare? No — the opposite is true. In real language, a handful of words ("the", "of", "is") appear constantly, while *most* words are rare (this lopsidedness is called **Zipf's law**). Multiply two mostly-rare word lists together and the conclusion is: **most possible word pairs never appear in any dataset of any size** — including huge numbers of perfectly normal pairs. Every real test text will contain some. Guaranteed.

The model's real crime is arrogance. Probability 0 means "I have seen everything, and this cannot happen." The honest statement would be "I just haven't seen it yet." **Smoothing** forces the model to be honest: take a little probability away from the pairs you *did* see, and spread it over the pairs you didn't. The classic recipes, simplest first:

- **Add-one smoothing (Laplace)** — what we'll implement. Pretend every possible pair appeared once more than it did. With a vocabulary of $V$ possible next words (ours is 6: the, cat, dog, sat, ran, `⟨/s⟩`):

  $$P(\text{ran} \mid \text{dog}) = \frac{\text{count}(\text{dog ran}) + 1}{\text{count}(\text{dog}) + V} = \frac{0+1}{1+6} = \frac{1}{7}$$

  Now "the dog ran" is *unlikely* instead of *impossible*. Fixed! But see the price: $P(\text{sat}\mid\text{dog})$, which used to be 1 (we saw "dog sat" every time "dog" appeared), fell to $\frac{1+1}{1+6} = 2/7$ — from 100% down to 29%. Funding the unseen taxed the seen, heavily. With a real vocabulary of tens of thousands of words, add-one taxes far too much — it's the crude first tool, not the good one.
- **Interpolation:** don't rely on one estimate — blend big and small contexts. Even if "dog ran" was never seen, "ran" by itself was; mix the two estimates with weights.
- **Backoff:** use the longest context you've actually seen; if the pair is unseen, fall back and *judge the word on its own reputation*.
- **Kneser-Ney** — the classic gold standard. Its cleverest idea: when falling back, don't ask "how common is this word?" but "**how many different contexts** has this word followed?" Example: "Francisco" is a common word, but it appears almost only after "San" — so in a fresh, new context it's a terrible guess. Plain frequency can't see that; Kneser-Ney can. (We'll implement it later in the roadmap.)

One sentence worth underlining: **smoothing is generalization** — the model's first-ever attempt to handle text it has never seen. N-grams generalize crudely, by adjusting counts. Neural models will do it beautifully, by learning that "cat" and "dog" are similar, so what you learn about one transfers to the other. Same problem; better machinery comes on Day 4.

### 7.5 Step five: perplexity, the scorecard

The model is trained and patched. Time for the grade we promised when we met Jelinek's team: **perplexity — how confused is the model when reading text it has never seen?** We know what the number *means* (choosing among K equally likely words; lower is better). Now let's see how it's actually computed. First, walk the model through test text and measure its average surprise per word — for each word, ask "how likely did you think that was?", convert to surprise, average. (This is the cross-entropy from earlier — the guessing game score, in bits per word):

$$H = -\frac{1}{N}\sum_{i=1}^{N} \log_2 P_{\text{model}}(w_i \mid \text{previous words})$$

One rule is non-negotiable: the test text must be text the model **never saw during training**. Scoring the model on its own training data is letting a student grade their own exam — our toy model looks perfect on "the cat sat" and falls on its face on "the dog ran."

Then, perplexity is that average surprise converted back into a plain count:

$$\text{Perplexity} = 2^{H}$$

Why convert? Because "the model averages 2 bits of surprise per word" is hard to feel, while its conversion is easy to feel:

> **A perplexity of K means: at every word, the model is as unsure as if it were choosing blindly among K equally likely words.**

- Our example table earlier scored 2 bits per word on "the cat sat" → perplexity $2^2 = 4$. Reading that sentence, the model felt like it was picking from 4 options at each step.
- Perplexity 100 = as lost as rolling a 100-sided die for every word.
- A model that knows nothing (all $V$ words equally likely, always) has perplexity exactly $V$ — for a 50,000-word vocabulary, perplexity 50,000.
- A perfect oracle that always knows the next word has perplexity 1.

So when someone says "model A has perplexity 100 and model B has 50," you can now translate: model B has genuinely *halved* the field of candidates it hesitates between. For calibration: good trigram models in the 1990s scored around 140 on a standard benchmark; today's LLMs score in the single digits to low tens on comparable text. From Shannon's staircase to the n-gram era to GPT — sixty-plus years of progress is one number going down. Next, we'll compute our own.

## 8. What n-grams can't do → the road ahead

1. **No generalization.** Seeing "the cat sat" in training teaches the model *nothing* about "the dog sat" — to an n-gram, "cat" and "dog" are unrelated symbols. → **Embeddings** (Days 4–6) fix this by letting similar words share what the model learns.
2. **Tiny, fixed memory.** The window can't see "keys" three words back, and making the window bigger explodes the count table exponentially. → **RNNs/LSTMs** (Days 7–8) compress unlimited history into a running summary; **attention/Transformers** (Days 10–18) let every word look directly at every other word.
3. **No meaning.** N-grams know which words *appear together*, not what they mean — fluent up close, lost at a distance. → the entire neural era.

But notice what does **not** change after today: the definition of a language model, the peel-one-word-at-a-time chain rule, average surprise (cross-entropy) as the training goal, and perplexity as the score. Every later model upgrades *the guesser* — never *the game*.

## 9. Glossary (only what today uses)

| Term | Plain meaning |
| ---- | ------------- |
| **Corpus** | A body of real text used for counting/training (plural: corpora). |
| **Token** | One unit of text after splitting — here, a word. |
| **N-gram** | A run of n words in a row. "the cat" is a bigram (n=2). |
| **Language model** | A machine that says how likely each next word is, given the words so far. |
| **Entropy** | How unpredictable a source is: the average number of yes/no questions per symbol. |
| **Cross-entropy** | How surprised *your model* is by real text, on average. Always ≥ entropy; the gap is your model's wrongness. |
| **Perplexity** | How confused the model is: "choosing among K equally likely words." $2^{\text{cross-entropy}}$. Lower = better. |
| **Smoothing** | Moving a little probability from seen word pairs to unseen ones, so nothing gets probability 0. |
| **Markov assumption** | The pretense that only the last n−1 words matter. |

*(Embeddings, neural networks, transformers, etc. are deliberately left out — each gets its own day.)*

## 10. References

1. Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal. — surprise, entropy, and the letter-guessing generators. (Also Shannon 1951, *Prediction and Entropy of Printed English* — the hangman experiment.)
2. Church, K. & Mercer, R. (1993). *Introduction to the Special Issue on Computational Linguistics Using Large Corpora.* Computational Linguistics 19(1). — the field switches sides.
3. Jelinek, F. (1997). *Statistical Methods for Speech Recognition.* MIT Press. — the two-part recipe and HMMs, in production.
4. Jurafsky, D. & Martin, J. H. *Speech and Language Processing* (3rd ed. draft), Chapter 3 "N-gram Language Models" — the standard textbook chapter; free online.

---

**Next:** we build it. A bigram/trigram model with add-one smoothing, trained on a real corpus — where the counting table becomes a Python dictionary, the zero-count problem becomes a real crash, and perplexity becomes three lines of NumPy.
