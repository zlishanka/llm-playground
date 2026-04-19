# LLM Playground

A collection of notes, examples, and experiments covering the full lifecycle of Large Language Models — from building one from scratch to fine-tuning, evaluating, and training at scale.

---

## Topics

### 1. [Build LLM from Scratch](llm.build.from.scratch.md)

An end-to-end walkthrough of what it takes to build an LLM from the ground up, modeled after real-world systems like GPT-3, Llama 2, and Falcon.

#### Data Curation
High-quality training data is the foundation of any capable model. This covers where to source data (web crawls, Wikipedia, code, books), public datasets like Common Crawl and The Pile, private domain-specific datasets, and synthetic data generation via LLMs (e.g., Alpaca). Preparation steps include quality filtering, de-duplication, privacy redaction, and tokenization using algorithms like Byte Pair Encoding (BPE).

#### Model Architecture
Covers the Transformer architecture and its three variants:
- **Encoder-only** — for tasks like text classification and embeddings (e.g., BERT)
- **Decoder-only** — for text generation (e.g., GPT)
- **Encoder-Decoder** — for sequence-to-sequence tasks like translation (e.g., T5)

Also includes key design decisions: residual connections, layer normalization, activation functions (ReLU, GeLU), positional embeddings, and model sizing heuristics (e.g., ~20 tokens per parameter is a common rule of thumb).

#### Training at Scale
Efficient large-scale training techniques including:
- **Mixed precision training** — combining FP32 and FP16 to reduce memory and speed up computation
- **3D parallelism** — combining pipeline, model, and data parallelism across GPU clusters
- **ZeRO Optimizer** — reduces memory redundancy by partitioning optimizer state, gradients, and parameters
- **Training stability** — checkpointing, weight decay, gradient clipping, and cosine learning rate schedules

#### 5-Stage Model Development Pipeline
A practical framework for building production AI models:
1. **Prepare the data** — petabyte-scale collection, filtering, and de-duplication
2. **Train the model** — tokenize and train on the data pile
3. **Validate the model** — benchmark evaluation and model card creation
4. **Tune the model** — fine-tune on task-specific data (accessible to application engineers, not just data scientists)
5. **Deploy the model** — cloud service or edge deployment, with ongoing iteration

---

### 2. [Fine-Tuning LLMs](llm.fine.tuning.md)

Fine-tuning adapts a pre-trained model to a specific task or domain by continuing training on a smaller, targeted dataset. A smaller fine-tuned model can outperform a much larger base model — InstructGPT (1.3B) outperforms raw GPT-3 (175B) on instruction-following tasks.

#### Fine-Tuning Approaches
- **Self-supervised** — curate a domain-specific corpus and continue pretraining (e.g., training on legal documents to specialize in legal language)
- **Supervised** — train on labeled input/output pairs using prompt templates (e.g., question/answer datasets)
- **Reinforcement Learning from Human Feedback (RLHF)** — train a reward model, then optimize with PPO

#### Parameter Training Strategies
| Strategy | Description | Cost |
|---|---|---|
| **Full fine-tuning** | Update all parameters | Very high |
| **Transfer learning** | Freeze most layers, fine-tune only the head | Moderate |
| **PEFT (LoRA)** | Freeze all weights, add small trainable adapters | Low |

#### LoRA (Low-Rank Adaptation)
LoRA injects trainable low-rank matrices into frozen weight layers. Instead of updating the full weight matrix `W₀` (d×k), it learns two smaller matrices `B` (d×r) and `A` (r×k) where rank `r << d,k`. The effective update becomes `W₀ + B·A`, with as little as 0.1% of the original parameter count needing to be trained.

---

### 3. [Fine-Tuning a Small LLM — Practical Examples](llm.fine.tuning.examples.md)

Hands-on guide to fine-tuning compact models (like TinyLlama and Mistral-7B) on custom datasets using a single GPU. The key insight: a well-fine-tuned small model on a specific task often outperforms a large general-purpose model while being far cheaper to run and deploy.

#### Why PEFT Matters
Full fine-tuning of billion-parameter models is infeasible on consumer hardware. PEFT methods like LoRA fine-tune only a tiny fraction of parameters, solving two problems simultaneously:
- **Memory** — tiny checkpoints (a few MBs) vs. multi-GB full model copies
- **Catastrophic forgetting** — frozen base weights preserve general knowledge

#### PEFT + LoRA with bitsandbytes
Walk-through of fine-tuning BLOOM-7B using the HuggingFace PEFT library with 8-bit quantization. Only ~0.11% of parameters are trained (7.8M out of 7B), dramatically reducing GPU memory requirements.

#### QLoRA — Fine-tuning on a Single GPU
QLoRA enables fine-tuning of large models (e.g., Mistral-7B) on a single consumer GPU through four key techniques:
- **4-bit NormalFloat** — smarter quantization that fits model weights into 4 bits using the parameter distribution
- **Double Quantization** — quantizes the quantization constants themselves (FP32 → Int8), saving additional memory
- **Paged Optimizers** — offloads optimizer states to CPU RAM when GPU memory is exhausted
- **LoRA adapters** — only the small adapter matrices are trained, achieving 100–1000× memory savings

Includes a full code example fine-tuning Mistral-7B-Instruct for a domain-specific YouTube comment reply use case.

---

### 4. [LLM Evaluation](llm.evaluation.md)

A structured framework for measuring LLM performance, covering automated metrics, benchmark datasets, and human-centered evaluation methods.

#### Core Performance Metrics
- **Perplexity** — measures how confidently the model predicts the next token; lower is better
- **Accuracy** — fraction of correct predictions; used for classification tasks
- **F1 Score** — harmonic mean of precision and recall; handles class imbalance better than accuracy alone

#### Generation Quality Metrics
- **BLEU** — evaluates machine translation quality by comparing n-gram overlap with reference translations
- **ROUGE** — evaluates summarization quality via recall-oriented n-gram overlap with reference summaries
- **Precision@K / Recall@K** — measures ranking quality for retrieval and recommendation systems

#### Human-Centric and Production Metrics
- **Human Evaluation** — annotators score outputs against predefined quality criteria; gold standard for nuanced tasks
- **Cross-Validation** — k-fold splitting for robust performance estimates and overfitting detection
- **A/B Testing** — compare model variants in live production to measure real-world impact
- **Domain-Specific Metrics** — custom metrics tailored to specialized fields (medical, legal, financial)

#### Standard Benchmark Datasets
| Benchmark | Type | What It Tests |
|---|---|---|
| **ARC** (AI2 Reasoning Challenge) | Multiple-choice | Elementary science reasoning (deductive, inductive, abductive) |
| **HellaSwag** | Multiple-choice | Commonsense reasoning beyond pattern matching |
| **MMLU** | Multiple-choice | Broad knowledge across 57 academic subjects |
| **TruthfulQA** | Open-ended | Factual accuracy and resistance to generating false information |

---

### 5. [Distributed Training of LLMs](llm.distrubuted.training.md)

Training large models requires distributing computation across multiple GPUs or machines. This covers the core strategies, challenges, and frameworks used in production LLM training.

#### Parallelism Strategies

**Data Parallelism**
The dataset is split across workers. Each worker runs a full forward and backward pass on its shard, then local gradients are communicated to a central parameter server for aggregation. The averaged global gradient is used to update model parameters, which are then broadcast back to all workers. Best when the model fits in a single device's memory.

**Model Parallelism**
Different layers or parameter matrices are placed on different devices. Each device handles a portion of the forward and backward passes, with inter-device communication for synchronization. Essential when the model is too large to fit on a single GPU.

#### Key Challenges
- **Synchronization** — ensuring all workers operate on consistent model parameters across iterations
- **Communication overhead** — inter-device data transfer can become a bottleneck at scale; mitigated with asynchronous updates or gradient compression
- **Fault tolerance** — checkpointing and resilient communication protocols to recover from hardware failures mid-training

#### Hyperparameters at Scale
- **Batch size** — typically starts at 16K tokens and may grow dynamically (GPT-3 scaled from 32K to 3.2M tokens)
- **Learning rate** — linear warm-up followed by cosine decay, settling at ~10% of peak value

#### Distributed Training Frameworks
| Framework | Developed By | Key Feature |
|---|---|---|
| **TensorFlow Distributed** | Google | `MirroredStrategy` for multi-GPU, `TPUStrategy` for TPU pods |
| **PyTorch Distributed (DDP)** | Meta (Facebook) | `DistributedDataParallel` for data parallelism, `RPC` for model parallelism |
| **Horovod** | Uber | Framework-agnostic (TF + PyTorch), uses ring-allreduce for efficient gradient aggregation |

### 6. Attention Residuals

#### Attention Residuals. 
- Replace fixed residuals with learned softmax attention
- Block AttnRes cuts `memory` and `communication costs`

#### Scaling Infrastructure. They build system tools for massive scale
- Cross-stage caching speeds up training
- Two-phase batching speeds up inference

#### Comprehensive Evaluation. They test a 48B-parameter model
- The architecture fixes PreNorm dilution
- It consistently beats baseline models

---

## References

| Paper / Resource | Link |
|---|---|
| "A Survey of Large Language Models" | [arXiv:2303.18223](https://arxiv.org/abs/2303.18223) |
| "Language Models are Few-Shot Learners" (GPT-3) | [arXiv:2005.14165](https://arxiv.org/abs/2005.14165) |
| "Spread Your Wings: Falcon 180B is here" | [HuggingFace Blog](https://huggingface.co/blog/falcon-180b) |
| "Cleaned Alpaca Dataset" | [GitHub](https://github.com/gururise/AlpacaDataCleaned) |
| "The Pile: An 800GB Dataset of Diverse Text for Language Modeling" | [arXiv:2101.00027](https://arxiv.org/abs/2101.00027) |
| "Training Compute-Optimal Large Language Models" (Chinchilla) | [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) |
| "Train With Mixed Precision" | [NVIDIA Docs](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) |
| "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" | [PDF](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) |
| "Attention Residuals" | [arXiv:2603.15031](https://arxiv.org/abs/2603.15031) |
| "LLM Wiki Karpathy" | [karpathy/llm-wiki.md](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) |

