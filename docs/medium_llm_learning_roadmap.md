# Medium + GitHub Roadmap: Learning and Teaching Modern NLP, Transformers, LLMs, RAG, Agents, and MCP
Start date: **Saturday, June 27, 2026**  
Publishing cadence: **3 articles per week** — first article today, then Monday / Wednesday / Friday.  
Goal: learn deeply, teach other students clearly, and build a public portfolio with papers, code, diagrams, datasets, and reproducible projects.
> Recommended repository structure

```text
llm-learning-roadmap/
├── README.md
├── articles/
├── notebooks/
├── projects/
├── figures/
├── papers.md
├── datasets.md
└── experiments.md
```
## Standard Article Template
Each article should follow this reusable structure:

```markdown
# Article Title

## 1. Why this topic matters
Explain the problem and why previous methods were not enough.

## 2. Historical context
Show where this topic sits in the NLP/LLM timeline.

## 3. Core intuition
Explain the idea without equations first.

## 4. Math and algorithms
Derive the important equations step by step.

## 5. Architecture or system design
Use diagrams, pseudocode, and component-level explanation.

## 6. Paper walkthrough
Cover 3–4 important papers and what each contributed.

## 7. Reproducible work
Build or reproduce something small but real.

## 8. Dataset and evaluation
Describe the dataset, metric, expected output, and failure cases.

## 9. Implementation notes
Include code snippets, libraries, model choices, and debugging tips.

## 10. Limitations and next step
End with what this method cannot solve and how the next article continues the story.
```

## Modern Implementation Stack to Use Across the Series
Use these repeatedly so the series becomes practical, not only theoretical.

- **Core ML:** Python, NumPy, PyTorch, scikit-learn, Hugging Face Transformers, Datasets, Tokenizers, Evaluate.
- **Training/Fine-tuning:** PEFT, TRL, bitsandbytes, Accelerate, Weights & Biases or MLflow.
- **RAG:** FAISS, Chroma, Qdrant or Milvus; sentence-transformers; BM25/Pyserini; cross-encoder rerankers.
- **LLM Apps:** LangChain, LangGraph, LlamaIndex, Pydantic, FastAPI, Streamlit.
- **Evaluation/Observability:** RAGAS, TruLens, LangSmith, custom pytest-style evals.
- **Agents:** LangGraph, AutoGen, CrewAI-style patterns, local tool execution, browser/search tools, shell/code tools.
- **MCP:** Model Context Protocol clients and servers; resources, tools, prompts, permissions, audit logs, and security boundaries.
- **Deployment:** Docker, FastAPI, GitHub Actions, simple cloud deployment, model serving with vLLM or llama.cpp.

## Roadmap Schedule
### Day 1 — Saturday, June 27, 2026 — Evolution of NLP: from rules to statistical language models
**Phase:** Foundations of NLP

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Shannon (1948) A Mathematical Theory of Communication
- Church & Mercer (1993) Computational Linguistics Using Large Corpora
- Jelinek (1997) Statistical Methods for Speech Recognition
- Jurafsky & Martin Speech and Language Processing

**Reproducible work:**
- Build a tiny n-gram language model from scratch; compute perplexity on a toy corpus.

**Related data, tools, and implementation notes:**
- Brown Corpus or WikiText-2; Python, NLTK, NumPy

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 2 — Monday, June 29, 2026 — Text preprocessing, tokenization, and evaluation basics
**Phase:** Foundations of NLP

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Manning et al. IR textbook chapters on tokenization
- Sennrich et al. (2016) BPE for NMT
- Kudo & Richardson (2018) SentencePiece
- Schuster & Nakajima (2012) WordPiece

**Reproducible work:**
- Implement whitespace, regex, BPE-like tokenization; compare token counts and OOV behavior.

**Related data, tools, and implementation notes:**
- Tiny Shakespeare, WikiText-2; Hugging Face tokenizers

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 3 — Wednesday, July 1, 2026 — Classical text classification: Naive Bayes, TF-IDF, logistic regression
**Phase:** Foundations of NLP

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Maron (1961) Automatic Indexing
- Salton & Buckley (1988) Term-weighting approaches
- Joachims (1998) Text categorization with SVMs
- McCallum & Nigam (1998) Naive Bayes for text classification

**Reproducible work:**
- Train TF-IDF + logistic regression sentiment classifier; evaluate precision/recall/F1.

**Related data, tools, and implementation notes:**
- IMDb or AG News; scikit-learn

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 4 — Friday, July 3, 2026 — Neural language models and distributed representations
**Phase:** Word Representations

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Bengio et al. (2003) Neural Probabilistic Language Model
- Collobert & Weston (2008) Unified Architecture for NLP
- Mikolov et al. (2013) Efficient Estimation of Word Representations
- Levy & Goldberg (2014) Neural Word Embedding as Implicit Matrix Factorization

**Reproducible work:**
- Train a small neural LM and inspect nearest neighbors in embedding space.

**Related data, tools, and implementation notes:**
- WikiText-2; PyTorch

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 5 — Monday, July 6, 2026 — Word2Vec: CBOW, Skip-gram, negative sampling
**Phase:** Word Representations

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Mikolov et al. (2013) Efficient Estimation of Word Representations
- Mikolov et al. (2013) Distributed Representations of Words and Phrases
- Goldberg & Levy (2014) word2vec Explained
- Levy et al. (2015) Improving Distributional Similarity

**Reproducible work:**
- Implement Skip-gram with negative sampling and visualize embeddings with PCA/t-SNE.

**Related data, tools, and implementation notes:**
- Text8 or WikiText-2; PyTorch, sklearn

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 6 — Wednesday, July 8, 2026 — GloVe, FastText, and subword representations
**Phase:** Word Representations

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Pennington et al. (2014) GloVe
- Bojanowski et al. (2017) FastText
- Joulin et al. (2016) Bag of Tricks for Efficient Text Classification
- Grave et al. (2018) Learning Word Vectors for 157 Languages

**Reproducible work:**
- Compare Word2Vec/GloVe/FastText on analogy and OOV examples.

**Related data, tools, and implementation notes:**
- GloVe vectors, FastText vectors; gensim

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 7 — Friday, July 10, 2026 — RNNs and backpropagation through time
**Phase:** Sequence Models

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Elman (1990) Finding Structure in Time
- Rumelhart et al. (1986) Learning Representations by Back-propagating Errors
- Pascanu et al. (2013) Difficulty of Training RNNs
- Mikolov et al. (2010) Recurrent Neural Network Based Language Model

**Reproducible work:**
- Build a character-level RNN from scratch; show hidden-state evolution.

**Related data, tools, and implementation notes:**
- Tiny Shakespeare; PyTorch

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 8 — Monday, July 13, 2026 — LSTM and GRU: solving long-term dependency problems
**Phase:** Sequence Models

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Hochreiter & Schmidhuber (1997) LSTM
- Cho et al. (2014) Learning Phrase Representations using RNN Encoder-Decoder
- Chung et al. (2014) Empirical Evaluation of Gated RNNs
- Greff et al. (2017) LSTM: A Search Space Odyssey

**Reproducible work:**
- Train RNN vs LSTM vs GRU on character prediction; compare loss and samples.

**Related data, tools, and implementation notes:**
- Tiny Shakespeare or Penn Treebank; PyTorch

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 9 — Wednesday, July 15, 2026 — Seq2Seq encoder-decoder models
**Phase:** Sequence Models

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Sutskever et al. (2014) Sequence to Sequence Learning
- Cho et al. (2014) RNN Encoder-Decoder
- Bengio et al. (2015) Scheduled Sampling
- Graves (2012) Sequence Transduction with RNNs

**Reproducible work:**
- Build a toy English-to-French translation model; explain teacher forcing and inference.

**Related data, tools, and implementation notes:**
- ManyThings bilingual pairs; PyTorch

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 10 — Friday, July 17, 2026 — Attention before Transformers
**Phase:** Sequence Models

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Bahdanau et al. (2015) Neural Machine Translation by Jointly Learning to Align and Translate
- Luong et al. (2015) Effective Approaches to Attention-based NMT
- Vinyals et al. (2015) Pointer Networks
- Sukhbaatar et al. (2015) End-to-End Memory Networks

**Reproducible work:**
- Add attention to your Seq2Seq model and plot attention heatmaps.

**Related data, tools, and implementation notes:**
- ManyThings bilingual pairs; PyTorch, matplotlib

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 11 — Monday, July 20, 2026 — Why RNNs failed to scale and why attention won
**Phase:** Transformer Core

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Vaswani et al. (2017) Attention Is All You Need
- Britz et al. (2017) Massive Exploration of NMT Architectures
- Kalchbrenner et al. (2016) Neural Machine Translation in Linear Time
- Gehring et al. (2017) ConvS2S

**Reproducible work:**
- Benchmark simple RNN/LSTM/attention blocks on sequence length and parallelism.

**Related data, tools, and implementation notes:**
- Synthetic copy task; PyTorch

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 12 — Wednesday, July 22, 2026 — Scaled dot-product attention from first principles
**Phase:** Transformer Core

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Vaswani et al. (2017) Attention Is All You Need
- Bahdanau et al. (2015) Attention
- Luong et al. (2015) Attention
- Katharopoulos et al. (2020) Transformers are RNNs

**Reproducible work:**
- Implement Q/K/V attention manually; test masking and softmax behavior.

**Related data, tools, and implementation notes:**
- Synthetic tensors; PyTorch

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 13 — Friday, July 24, 2026 — Multi-head attention and representation subspaces
**Phase:** Transformer Core

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Vaswani et al. (2017) Attention Is All You Need
- Michel et al. (2019) Are Sixteen Heads Really Better than One?
- Voita et al. (2019) Analyzing Multi-Head Self-Attention
- Clark et al. (2019) What Does BERT Look At?

**Reproducible work:**
- Visualize attention heads from a small transformer; ablate heads and measure impact.

**Related data, tools, and implementation notes:**
- DistilBERT/BERT attention maps; transformers

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 14 — Monday, July 27, 2026 — Positional encodings: sinusoidal, learned, relative, RoPE, ALiBi
**Phase:** Transformer Core

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Vaswani et al. (2017) sinusoidal encodings
- Shaw et al. (2018) Relative Position Representations
- Su et al. (2021) RoFormer/RoPE
- Press et al. (2022) ALiBi

**Reproducible work:**
- Implement sinusoidal, learned, RoPE, and ALiBi; compare extrapolation on longer sequences.

**Related data, tools, and implementation notes:**
- Synthetic sequence tasks; PyTorch

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 15 — Wednesday, July 29, 2026 — Feed-forward networks, activations, and MLP blocks
**Phase:** Transformer Core

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Vaswani et al. (2017) Transformer FFN
- Hendrycks & Gimpel (2016) GELU
- Shazeer (2020) GLU Variants
- Fedus et al. (2021) Switch Transformers

**Reproducible work:**
- Swap ReLU/GELU/SwiGLU in a tiny transformer and compare training curves.

**Related data, tools, and implementation notes:**
- Tiny Shakespeare; PyTorch

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 16 — Friday, July 31, 2026 — Residual connections, LayerNorm, PreNorm vs PostNorm
**Phase:** Transformer Core

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- He et al. (2016) Deep Residual Learning
- Ba et al. (2016) Layer Normalization
- Xiong et al. (2020) On LayerNorm in Transformers
- Wang et al. (2022) DeepNet

**Reproducible work:**
- Train PostNorm vs PreNorm tiny transformer; inspect stability and gradients.

**Related data, tools, and implementation notes:**
- Tiny Shakespeare; PyTorch

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 17 — Monday, August 3, 2026 — Build a Transformer encoder from scratch
**Phase:** Transformer Core

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Vaswani et al. (2017) Attention Is All You Need
- Devlin et al. (2019) BERT
- Liu et al. (2019) RoBERTa
- Lan et al. (2020) ALBERT

**Reproducible work:**
- Implement encoder block and train classifier on AG News.

**Related data, tools, and implementation notes:**
- AG News; PyTorch

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 18 — Wednesday, August 5, 2026 — Build a decoder-only GPT from scratch
**Phase:** Transformer Core

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Radford et al. (2018) GPT
- Radford et al. (2019) GPT-2
- Brown et al. (2020) GPT-3
- Karpathy nanoGPT reference

**Reproducible work:**
- Train a mini GPT on Tiny Shakespeare; generate samples and explain loss.

**Related data, tools, and implementation notes:**
- Tiny Shakespeare; nanoGPT/PyTorch

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 19 — Friday, August 7, 2026 — BERT and masked language modeling
**Phase:** Transformer Families

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Devlin et al. (2019) BERT
- Liu et al. (2019) RoBERTa
- Lan et al. (2020) ALBERT
- Sanh et al. (2019) DistilBERT

**Reproducible work:**
- Fine-tune BERT on sentiment or NER; show masked-token predictions.

**Related data, tools, and implementation notes:**
- IMDb/SST-2/CoNLL; Hugging Face

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 20 — Monday, August 10, 2026 — GPT-style causal language models
**Phase:** Transformer Families

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Radford et al. (2018) GPT
- Radford et al. (2019) GPT-2
- Brown et al. (2020) GPT-3
- Wei et al. (2022) Emergent Abilities

**Reproducible work:**
- Fine-tune a small GPT-2 model; compare prompting vs fine-tuning.

**Related data, tools, and implementation notes:**
- TinyStories or custom dataset; transformers

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 21 — Wednesday, August 12, 2026 — Encoder-decoder models: T5, BART, UL2
**Phase:** Transformer Families

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Raffel et al. (2020) T5
- Lewis et al. (2020) BART
- Tay et al. (2022) UL2
- Wei et al. (2022) FLAN

**Reproducible work:**
- Fine-tune T5-small for summarization or question generation.

**Related data, tools, and implementation notes:**
- CNN/DailyMail small subset, SQuAD; transformers

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 22 — Friday, August 14, 2026 — Efficient and compressed Transformers
**Phase:** Transformer Families

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Sanh et al. (2019) DistilBERT
- Lan et al. (2020) ALBERT
- Jiao et al. (2020) TinyBERT
- Wang et al. (2020) MiniLM

**Reproducible work:**
- Distill a classifier from BERT to DistilBERT/Tiny model; compare latency and accuracy.

**Related data, tools, and implementation notes:**
- SST-2/AG News; transformers

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 23 — Monday, August 17, 2026 — Tokenization in modern LLMs
**Phase:** Pretraining

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Sennrich et al. (2016) BPE
- Kudo & Richardson (2018) SentencePiece
- Schuster & Nakajima (2012) WordPiece
- Bostrom & Durrett (2020) Byte Pair Encoding is Suboptimal?

**Reproducible work:**
- Train your own tokenizer and use it in a tiny LM.

**Related data, tools, and implementation notes:**
- WikiText-2; tokenizers

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 24 — Wednesday, August 19, 2026 — Pretraining objectives: MLM, CLM, denoising, replaced-token detection
**Phase:** Pretraining

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Devlin et al. (2019) BERT MLM
- Clark et al. (2020) ELECTRA
- Lewis et al. (2020) BART
- Tay et al. (2022) UL2

**Reproducible work:**
- Train tiny models with MLM vs CLM objective and compare downstream behavior.

**Related data, tools, and implementation notes:**
- WikiText-2; transformers

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 25 — Friday, August 21, 2026 — Scaling laws and compute-optimal training
**Phase:** Pretraining

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Kaplan et al. (2020) Scaling Laws
- Hoffmann et al. (2022) Chinchilla
- Henighan et al. (2020) Scaling Laws for Autoregressive Generative Modeling
- Wei et al. (2022) Emergent Abilities

**Reproducible work:**
- Run mini scaling experiment over model size/data size; plot loss curves.

**Related data, tools, and implementation notes:**
- TinyStories/WikiText; PyTorch

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 26 — Monday, August 24, 2026 — Data quality, deduplication, filtering, and contamination
**Phase:** Pretraining

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Dodge et al. (2021) Documenting Large Webtext Corpora
- Raffel et al. (2020) C4/T5
- Lee et al. (2022) Deduplicating Training Data
- Penedo et al. (2023) RefinedWeb

**Reproducible work:**
- Build a small data-cleaning pipeline: language ID, dedup, toxicity filter, eval leakage check.

**Related data, tools, and implementation notes:**
- Common Crawl sample, C4 sample; datasketch, fastText

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 27 — Wednesday, August 26, 2026 — Parameter-efficient fine-tuning: adapters, LoRA, QLoRA
**Phase:** Pretraining

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Houlsby et al. (2019) Adapters
- Hu et al. (2021) LoRA
- Dettmers et al. (2023) QLoRA
- Liu et al. (2022) Few-shot Parameter-Efficient Fine-Tuning

**Reproducible work:**
- Fine-tune a small open model with LoRA; compare full fine-tuning vs PEFT.

**Related data, tools, and implementation notes:**
- Alpaca-style small dataset; PEFT, bitsandbytes

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 28 — Friday, August 28, 2026 — Instruction tuning and supervised fine-tuning
**Phase:** Post-training

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Wei et al. (2022) FLAN
- Sanh et al. (2022) T0
- Ouyang et al. (2022) InstructGPT
- Wang et al. (2023) Self-Instruct

**Reproducible work:**
- Create a small instruction dataset and SFT a small model with LoRA.

**Related data, tools, and implementation notes:**
- Dolly/Alpaca/OpenHermes sample; TRL, PEFT

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 29 — Monday, August 31, 2026 — RLHF: reward models, PPO, preference learning
**Phase:** Post-training

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Ouyang et al. (2022) InstructGPT
- Christiano et al. (2017) Deep RL from Human Preferences
- Stiennon et al. (2020) Learning to Summarize with Human Feedback
- Schulman et al. (2017) PPO

**Reproducible work:**
- Train a toy reward model and run PPO-style preference optimization on summaries.

**Related data, tools, and implementation notes:**
- Anthropic HH or TL;DR preferences; TRL

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 30 — Wednesday, September 2, 2026 — DPO, IPO, ORPO, and preference optimization without RL
**Phase:** Post-training

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Rafailov et al. (2023) DPO
- Azar et al. (2023) IPO
- Hong et al. (2024) ORPO
- Ethayarajh et al. (2024) KTO

**Reproducible work:**
- Run DPO on a small preference dataset; compare before/after outputs.

**Related data, tools, and implementation notes:**
- UltraFeedback sample; TRL

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 31 — Friday, September 4, 2026 — Safety, constitutional AI, red-teaming, and evaluation
**Phase:** Post-training

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Bai et al. (2022) Constitutional AI
- Ganguli et al. (2022) Red Teaming Language Models
- Perez et al. (2022) Discovering Language Model Behaviors
- Liang et al. (2023) HELM

**Reproducible work:**
- Build an eval harness for refusal, helpfulness, and hallucination on a toy model/API.

**Related data, tools, and implementation notes:**
- HELM-style prompts, AdvBench safe subset; inspect manually

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 32 — Monday, September 7, 2026 — Prompting, in-context learning, and chain-of-thought
**Phase:** Reasoning

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Brown et al. (2020) GPT-3
- Wei et al. (2022) Chain-of-Thought Prompting
- Kojima et al. (2022) Zero-shot CoT
- Min et al. (2022) Rethinking Demonstrations

**Reproducible work:**
- Compare zero-shot, few-shot, and CoT prompts on GSM8K sample.

**Related data, tools, and implementation notes:**
- GSM8K sample; OpenAI/local model

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 33 — Wednesday, September 9, 2026 — Self-consistency, Tree of Thoughts, and search over reasoning
**Phase:** Reasoning

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Wang et al. (2022) Self-Consistency
- Yao et al. (2023) Tree of Thoughts
- Besta et al. (2024) Graph of Thoughts
- Lightman et al. (2023) Process Supervision

**Reproducible work:**
- Implement self-consistency voting; optionally add tree search for puzzle tasks.

**Related data, tools, and implementation notes:**
- GSM8K, Game of 24; Python

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 34 — Friday, September 11, 2026 — ReAct, Reflexion, and tool-augmented reasoning
**Phase:** Reasoning

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Yao et al. (2023) ReAct
- Shinn et al. (2023) Reflexion
- Schick et al. (2023) Toolformer
- Madaan et al. (2023) Self-Refine

**Reproducible work:**
- Build a calculator/search tool agent; log thoughts/actions/observations.

**Related data, tools, and implementation notes:**
- LangGraph or plain Python; simple tools

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 35 — Monday, September 14, 2026 — Test-time compute and inference-time scaling
**Phase:** Reasoning

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Cobbe et al. (2021) Training Verifiers
- Wang et al. (2022) Self-Consistency
- Snell et al. (2024) Scaling LLM Test-Time Compute
- Zelikman et al. (2024) Quiet-STaR

**Reproducible work:**
- Plot accuracy vs number of samples/verifier reranks on reasoning problems.

**Related data, tools, and implementation notes:**
- GSM8K sample; local/API model

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 36 — Wednesday, September 16, 2026 — Catastrophic forgetting and continual learning
**Phase:** Continual Learning

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- McCloskey & Cohen (1989) Catastrophic Interference
- Kirkpatrick et al. (2017) EWC
- Zenke et al. (2017) Synaptic Intelligence
- Lopez-Paz & Ranzato (2017) Gradient Episodic Memory

**Reproducible work:**
- Train sequential classifiers on two tasks; measure forgetting and apply replay/EWC.

**Related data, tools, and implementation notes:**
- MNIST splits or text tasks; PyTorch

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 37 — Friday, September 18, 2026 — Model editing: ROME, MEMIT, SERAC
**Phase:** Continual Learning

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Mitchell et al. (2022) Model Editing at Scale/SERAC
- Meng et al. (2022) ROME
- Meng et al. (2023) MEMIT
- Yao et al. (2023) Editing Large Language Models

**Reproducible work:**
- Use a model-editing library to edit one factual association and test side effects.

**Related data, tools, and implementation notes:**
- CounterFact; EasyEdit

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 38 — Monday, September 21, 2026 — Embeddings, semantic search, and vector databases
**Phase:** Retrieval

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Karpukhin et al. (2020) DPR
- Reimers & Gurevych (2019) Sentence-BERT
- Khattab & Zaharia (2020) ColBERT
- Johnson et al. (2019) FAISS

**Reproducible work:**
- Build semantic search over your notes/blog drafts with FAISS or Chroma.

**Related data, tools, and implementation notes:**
- Your notes, Wikipedia snippets; sentence-transformers, FAISS/Chroma

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 39 — Wednesday, September 23, 2026 — RAG fundamentals: chunking, retrieval, generation, citations
**Phase:** Retrieval

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Lewis et al. (2020) RAG
- Guu et al. (2020) REALM
- Izacard & Grave (2021) FiD
- Borgeaud et al. (2022) RETRO

**Reproducible work:**
- Build a PDF/chat RAG app with citations; test chunk size and top-k.

**Related data, tools, and implementation notes:**
- PDFs/arXiv docs; LangChain or LlamaIndex, Chroma

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 40 — Friday, September 25, 2026 — Advanced RAG: reranking, hybrid search, query rewriting
**Phase:** Retrieval

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Nogueira & Cho (2019) Passage Re-ranking with BERT
- Khattab & Zaharia (2020) ColBERT
- Gao et al. (2023) HyDE
- Ma et al. (2023) Query Rewriting for RAG

**Reproducible work:**
- Add BM25 + dense hybrid retrieval + cross-encoder reranker; measure recall@k.

**Related data, tools, and implementation notes:**
- BEIR small dataset; Pyserini, sentence-transformers

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 41 — Monday, September 28, 2026 — RAG evaluation and production observability
**Phase:** Retrieval

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Liu et al. (2023) Lost in the Middle
- Es et al. (2023) RAGAS
- Liang et al. (2023) HELM
- Mallen et al. (2023) When Not to Trust LM Knowledge

**Reproducible work:**
- Create an evaluation set; compute faithfulness, context precision, answer correctness.

**Related data, tools, and implementation notes:**
- RAGAS, LangSmith, TruLens; your RAG app

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 42 — Wednesday, September 30, 2026 — Long-context transformers and memory-efficient attention
**Phase:** Long Context

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Beltagy et al. (2020) Longformer
- Zaheer et al. (2020) BigBird
- Dao et al. (2022) FlashAttention
- Dai et al. (2019) Transformer-XL

**Reproducible work:**
- Benchmark attention memory/runtime for standard vs long-context tricks.

**Related data, tools, and implementation notes:**
- Long documents; PyTorch/xFormers/FlashAttention notes

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 43 — Friday, October 2, 2026 — LangChain basics: chains, tools, agents, LCEL, LangGraph
**Phase:** Modern LLM Apps

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- ReAct (Yao et al. 2023)
- Toolformer (Schick et al. 2023)
- LangChain docs
- LangGraph docs

**Reproducible work:**
- Build a research assistant: search/retrieve/summarize with a graph-based workflow.

**Related data, tools, and implementation notes:**
- LangChain, LangGraph, Tavily/search stub, Chroma

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 44 — Monday, October 5, 2026 — LlamaIndex for document workflows
**Phase:** Modern LLM Apps

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- RAG (Lewis et al. 2020)
- DPR (Karpukhin et al. 2020)
- LlamaIndex docs
- ColBERT (Khattab & Zaharia 2020)

**Reproducible work:**
- Build document ingestion with parsing, indexing, metadata filters, and query engine.

**Related data, tools, and implementation notes:**
- LlamaIndex, local PDFs, vector DB

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 45 — Wednesday, October 7, 2026 — Tool calling, function calling, and structured outputs
**Phase:** Modern LLM Apps

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Toolformer (Schick et al. 2023)
- ReAct (Yao et al. 2023)
- OpenAI tool/function calling docs
- JSON schema / structured output docs

**Reproducible work:**
- Build a weather/calculator/database assistant using JSON-schema tools and structured final answers.

**Related data, tools, and implementation notes:**
- OpenAI API or local tool-call capable model; Pydantic

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 46 — Friday, October 9, 2026 — MCP in detail: clients, servers, resources, tools, prompts, security
**Phase:** Modern LLM Apps

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Anthropic MCP announcement
- MCP official docs
- ReAct (Yao et al. 2023)
- Toolformer (Schick et al. 2023)

**Reproducible work:**
- Build a local MCP server exposing files/search/calculator; connect it to a client; add permissions and audit logs.

**Related data, tools, and implementation notes:**
- MCP Python/TypeScript SDK, Claude Desktop or compatible client

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 47 — Monday, October 12, 2026 — Agent architecture: planner, executor, memory, tools, evaluator
**Phase:** Agents

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Yao et al. (2023) ReAct
- Shinn et al. (2023) Reflexion
- Wang et al. (2024) Voyager
- Park et al. (2023) Generative Agents

**Reproducible work:**
- Implement a minimal agent loop with planning, tool use, memory, and eval traces.

**Related data, tools, and implementation notes:**
- LangGraph or plain Python; SQLite memory

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 48 — Wednesday, October 14, 2026 — Multi-agent systems and orchestration
**Phase:** Agents

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- AutoGen (Wu et al. 2023)
- CAMEL (Li et al. 2023)
- Generative Agents (Park et al. 2023)
- MetaGPT (Hong et al. 2023)

**Reproducible work:**
- Build a two-agent debate/reviewer system for blog outlines; compare to single-agent output.

**Related data, tools, and implementation notes:**
- AutoGen/CrewAI/LangGraph

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 49 — Friday, October 16, 2026 — Code LLMs: Codex, Code Llama, StarCoder, DeepSeek-Coder
**Phase:** Coding Models

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Chen et al. (2021) Codex
- Li et al. (2023) StarCoder
- Rozière et al. (2023) Code Llama
- Guo et al. (2024) DeepSeek-Coder

**Reproducible work:**
- Evaluate small code model on HumanEval-style tasks; measure pass@k.

**Related data, tools, and implementation notes:**
- HumanEval/MBPP; evalplus

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 50 — Monday, October 19, 2026 — Agentic coding: SWE-agent, Devin-style workflows, Claude Code-style systems
**Phase:** Coding Models

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Jimenez et al. (2024) SWE-bench
- Yang et al. (2024) SWE-agent
- ReAct (Yao et al. 2023)
- Toolformer (Schick et al. 2023)

**Reproducible work:**
- Build a repo-fixing agent that reads issue, edits files, runs tests, and submits patch.

**Related data, tools, and implementation notes:**
- SWE-bench lite, local Git repo, pytest, LangGraph

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 51 — Wednesday, October 21, 2026 — Code review, test generation, and CI-integrated agents
**Phase:** Coding Models

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Tufano et al. (2019) Learning to Fix Bugs
- Chen et al. (2021) Codex
- SWE-bench
- CodeQL/security papers

**Reproducible work:**
- Build a PR-review bot that summarizes diff, finds risks, and suggests tests.

**Related data, tools, and implementation notes:**
- GitHub Actions, local repo, OpenAI/local model

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 52 — Friday, October 23, 2026 — Quantization, pruning, and inference optimization
**Phase:** Efficiency

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Dettmers et al. (2022) LLM.int8()
- Frantar et al. (2022) GPTQ
- Lin et al. (2024) AWQ
- Xiao et al. (2023) SmoothQuant

**Reproducible work:**
- Quantize a small model and compare latency, memory, and output quality.

**Related data, tools, and implementation notes:**
- llama.cpp, bitsandbytes, AutoGPTQ

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 53 — Monday, October 26, 2026 — Speculative decoding, KV cache, batching, and serving
**Phase:** Efficiency

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Leviathan et al. (2023) Speculative Decoding
- Dao et al. (2022) FlashAttention
- Kwon et al. (2023) vLLM/PagedAttention
- Pope et al. (2023) Efficiently Scaling Transformer Inference

**Reproducible work:**
- Serve a local model with vLLM or llama.cpp; measure throughput and latency.

**Related data, tools, and implementation notes:**
- vLLM/llama.cpp, small open model

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 54 — Wednesday, October 28, 2026 — Mixture of Experts and sparse models
**Phase:** Efficiency

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Shazeer et al. (2017) Outrageously Large Neural Networks
- Fedus et al. (2021) Switch Transformers
- Lepikhin et al. (2020) GShard
- Jiang et al. (2024) Mixtral

**Reproducible work:**
- Implement toy MoE routing layer; visualize expert load balancing.

**Related data, tools, and implementation notes:**
- PyTorch synthetic data

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 55 — Friday, October 30, 2026 — State space models: S4, Mamba, RWKV
**Phase:** Beyond Transformers

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Gu et al. (2021) S4
- Gu & Dao (2023) Mamba
- Peng et al. (2023) RWKV
- Fu et al. (2023) Hungry Hungry Hippos

**Reproducible work:**
- Train a tiny Mamba/RWKV-style model or run inference and compare with Transformer baseline.

**Related data, tools, and implementation notes:**
- Tiny Shakespeare; mamba-minimal or RWKV libs

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 56 — Monday, November 2, 2026 — Vision Transformers and image-language models
**Phase:** Multimodal

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- Dosovitskiy et al. (2021) ViT
- Radford et al. (2021) CLIP
- Alayrac et al. (2022) Flamingo
- Liu et al. (2023) LLaVA

**Reproducible work:**
- Build CLIP-based image search; optionally fine-tune a vision classifier.

**Related data, tools, and implementation notes:**
- CIFAR-10, COCO sample; transformers/open_clip

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

### Day 57 — Wednesday, November 4, 2026 — End-to-end LLM app: RAG + tools + MCP + eval + deployment
**Phase:** Capstone

**Learning objective:** Understand the concept deeply enough to teach it with intuition, math, code, limitations, and historical context.

**Core papers / references:**
- RAG (Lewis et al. 2020)
- ReAct (Yao et al. 2023)
- MCP docs
- RAGAS / evaluation papers

**Reproducible work:**
- Build a production-style research assistant: ingest docs, retrieve, call tools, expose MCP server, evaluate, log traces, deploy.

**Related data, tools, and implementation notes:**
- LangChain/LangGraph, LlamaIndex, Chroma/FAISS, LangSmith/RAGAS, FastAPI

**Suggested article angle:**
- Explain what problem this topic solved, build the smallest useful reproduction, then connect it to the next topic in the roadmap.

## Capstone Rule
Every 10 articles, publish one recap article: what you learned, what code you built, what failed, what changed in your understanding, and what readers should revise before moving forward.
## Practical Reproduction Levels
Use three levels so the work stays achievable:

1. **From scratch:** NumPy/PyTorch implementation for small concepts like n-grams, attention, tokenizers, RNNs, tiny GPT.
2. **Library-based:** Hugging Face/LangChain/LlamaIndex implementations for realistic workflows.
3. **System project:** RAG apps, agents, MCP servers, evaluation harnesses, deployment, and observability.

## MCP Clarification
In this roadmap, **MCP means Model Context Protocol**, not just a vague 'skills' idea. Cover it as a real integration layer:

- **MCP client:** the AI application or IDE that connects to servers.
- **MCP server:** a service exposing tools, resources, and prompts.
- **Resources:** data sources such as files, databases, APIs, documentation, tickets, or internal knowledge.
- **Tools:** actions the model can request, such as search, SQL query, file read/write, calendar lookup, or code execution.
- **Prompts:** reusable prompt templates or workflows exposed by the server.
- **Security:** permissioning, command whitelisting, sandboxing, prompt injection defense, audit logs, and least-privilege design.

