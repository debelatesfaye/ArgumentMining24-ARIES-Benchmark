# ArgumentMining24-ARIES-Benchmark

**ArgumentMining24-ARIES-Benchmark** provides a comprehensive framework for **benchmarking argument mining tasks** using transformer models such as **RoBERTa**, **GPT-2**, and **T5**.  

This repository is built around **ARIES (Argument Relation Identification Evaluation Standard)**, a benchmark designed to standardise evaluation in argument mining research. ARIES addresses key challenges in the field, including heterogeneous annotations, varied argumentation domains, and differing theoretical approaches, providing a unified framework for comparison.

---

## Key Features

- Supports multiple **argument mining tasks**:
  - **Sequence Classification** – Classify relations between argument components.
  - **Token Classification** – Tag tokens in arguments with additional sequence classification.
  - **Sequence Alignment** – Align sequences using cross-attention mechanisms.

- Compatible with the **three main Transformer architectures**:
  - **Encoder-only** (e.g., RoBERTa)
  - **Decoder-only** (e.g., GPT-2)
  - **Encoder-decoder** (e.g., T5)

- Standardized over **eight argument mining datasets**, covering diverse domains with consistent annotation structures.

---

## Motivation

Measuring advances in argument mining is challenging due to:
- Diverse theoretical frameworks of argument
- Inconsistent annotation schemes
- Cross-domain variability

ARIES provides a **standardized evaluation benchmark**, enabling reproducible and comparable results across different models, tasks, and datasets. Experiments on ARIES highlight the difficulty of cross-dataset generalization, emphasizing the need for transferable argument mining models.

---

If you use this benchmark for research, please cite the ARIES benchmark:

