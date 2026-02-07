---
title: Transformer
draft: false
tags:
  - NLP
  - LLM
  - Transformers
  - Study-Notes
date: 2024-03-17 00:45
---

# Attention Is All You Need-Transformer

---

## Model

![[Attachments/TransformerModel.png]]

- A Transformer is an encoder-decoder model architecture that uses the attention mechanism.
- Massive advantage over RNN based encoder-decoder architecture since it allows to:
  - Benefiting from the GPU/TPU parallelization.
  - Larger amount of data processed in the same amount of time.
  - process all tokens at once!
- The Encoder / Decoder components are stacks of Encoders and Decoders, respectively.
  - The paper that introduced Transformers stacked 6 of each on top of each other respectively.
- The models were built using attention mechanism at the core.
  - because of the architecture, the attention mechanism helps improve the performance of machine translation applications.
