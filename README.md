# LLM from Scratch

> Learn, build, and understand Large Language Models from first principles.

This repository documents my journey to deeply understand modern LLMs by studying foundational research papers, implementing core algorithms from scratch, reproducing important experiments, and building real-world AI systems.

## What you'll find

- 📚 Research paper summaries and reading notes
- 🧠 Intuitive explanations of ML and NLP concepts
- ✍️ Long-form explanations published on [Medium](https://medium.com/@vikrant_bhati)
- 💻 PyTorch implementations from scratch
- 🔬 Reproducible experiments
- 🤖 Real-world LLM applications using Hugging Face, LangChain, LangGraph, LlamaIndex, MCP, and vector databases
- 📊 Visualizations and architecture diagrams
- 🚀 End-to-end projects, from language modeling to agentic AI

## Articles and notebooks

I'm writing this series as I learn: the articles focus on intuition and the notebooks let you work through the ideas yourself.

### Day 1 — From information theory to language models

#### 1. [Claude Shannon Measured the English Language by Playing Hangman](https://medium.com/@vikrant_bhati/claude-shannon-measured-the-english-language-by-playing-hangman-1ce201cfe869)

Shannon measured English by asking people to guess one letter at a time. I use that experiment to explain surprise, entropy, and cross-entropy—the same ideas that still sit at the heart of language-model training.

#### 2. [N-gram Model: From Rules to Statistical Language Models](https://medium.com/@vikrant_bhati/n-gram-model-from-rules-to-statistical-language-models-72ea0d2090f9)

I trained a small language model on the Brown Corpus and watched ordinary unseen sentences break it. This article follows the failure from zero probabilities to smoothing, perplexity, and a model that can finally generate text.

**Notebook:** [Build the Day 1 n-gram model](notebooks/day1/ngram.ipynb)

### Day 2 — Tokenization

#### 3. [Tokenization: Why LLMs Can't Count R's in Strawberry](https://medium.com/@vikrant_bhati/tokenization-why-llms-cant-count-r-s-in-strawberry-059400c33ceb)

Before an LLM reads a sentence, a tokenizer has already chopped it into pieces. I look at why that hidden step causes strange mistakes, then build a small BPE tokenizer to see how modern models handle rare and unfamiliar words.

**Notebook:** [Practice tokenization and build BPE](notebooks/day2/tokenization_practice.ipynb)

### Day 3 — Classical text classification

#### 4. A Computer Filed 405 Abstracts in 1961 — and Accidentally Invented Your Spam Filter

In 1961 a computer sorted 405 abstracts into 32 categories by doing nothing but counting words. I follow that trick to TF-IDF and naive Bayes, and show that the clever half of TF-IDF is really Claude Shannon's measure of surprise in disguise.

*(Publishing soon — link to follow.)*

#### 5. My Sentiment Classifier Is 88% Accurate and Completely Unable to Read

I built bag-of-words, TF-IDF, naive Bayes and logistic regression from scratch and ran them on 50,000 IMDb reviews. The best model reaches 88%, one model hits 100% accuracy on its training data (the worst news in the experiment), and none of them can tell "good, not bad" from "bad, not good".

*(Publishing soon — link to follow.)*

**Notebook:** [Build TF-IDF, naive Bayes and logistic regression](notebooks/day3/text_classification.ipynb)

You can find the rest of my writing on [Medium](https://medium.com/@vikrant_bhati).

## Roadmap

- Classical NLP
- Word Embeddings
- RNNs, LSTMs, and GRUs
- Attention Mechanisms
- Transformers
- BERT, GPT, T5, Llama, and modern foundation models
- Pretraining and Scaling Laws
- Fine-tuning and Alignment (RLHF, DPO, SFT)
- Retrieval-Augmented Generation (RAG)
- Long-context models
- Agentic AI and Coding Agents
- Model Context Protocol (MCP)
- Evaluation and Benchmarks
- Deployment and Optimization
