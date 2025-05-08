---
layout: post
title: HuggingFace 🤗 for DNA LLM inference
date: 2025-05-02 11:59:00-0000
description: Using 🤗 transformers
tags: python machine_learning huggingface
giscus_comments: true
related_posts: false
toc:
  sidebar: left
images:
  compare: true
  slider: true
mermaid:
  enabled: true
---

HuggingFace (🤗) has become a staple of LLM work. In this post I will explore this library with a different use case: handling DNA LLMs.

TODO What is inference

<!-- https://colab.research.google.com/github/hclimente/hclimente.github.io/blob/main/assets/python/2025-05-02-hf-transformers/main.ipynb -->

# The Nucleotide Transformer

I will focus on the [Nucleotide Transformer](https://www.nature.com/articles/s41592-024-02523-z), developed by [InstaDeep](https://www.instadeep.com/). InstaDeep shows a commendable commitment to open science, regularly publishing articles, models, documentation and code. A big part of this article is based off their work, which I have re-used to some degree. In this article I use [a small, 50 million parameter Nucleotide Transformer](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species), available from [their 🤗 organization](https://huggingface.co/InstaDeepAI).

The Nucleotide Transformer was produces as TODO

{% include figure.liquid path="assets/img/posts/2025-05-02-hf-transformers/nucleotide_transformer.jpg" class="img-fluid rounded z-depth-1" %}

<div class="caption">
  Training of the Nucleotide Transformer using masked language modelling. Adapted from Figure 1 in the <a href="https://www.nature.com/articles/s41592-024-02523-z">Nucleotide Transformer article</a>.
</div>

Inputs and outputs

# The 🤗 `transformers` library

[HuggingFace's 🤗 `transformers`](https://huggingface.co/docs/transformers/index) is a powerful library for working with transformer models, providing a wide range of pre-trained models and utilities for fine-tuning and deploying them. Problem that is solved: different models have different APIs. The `transformers` library provides a unified API for all transformer models, making it easy to switch between them without having to learn a new API each time. For instance, fetching the Nucleotide Transformer is as easy as running:

```python
model = AutoModelForMaskedLM.from_pretrained(
  "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
  trust_remote_code=True
)
```

The model cannot be applied directly to a DNA sequence. Usually, the workflow is:

Attention masks

```mermaid
---
config:
  layout: elk
  look: handDrawn
---
flowchart LR
    %% Style definitions
    classDef process fill:#a8dadc,stroke:#2f4f4f,stroke-width:2px,rx:8,ry:8,color:#000
    classDef data fill:#f9c74f,stroke:#2f4f4f,stroke-width:2px,rx:8,ry:8,color:#000

    %% Process nodes
    P2[Tokenizer]:::process
    P3[Model Inference]:::process
    P5[Postprocessing]:::process

    %% Data nodes
    D1[DNA Sequence]:::data
    D21[Tokens]:::data
    D22[Attention Mask]:::data
    D3[Embeddings]:::data
    D5[Masked Embeddings]:::data

    %% Connections
    D1 --> P2
    P2 --> D21
    P2 --> D22
    D21 --> P3
    D22 --> P3
    P3 --> D3
    D22 --> P5
    D3  --> P5
    P5 --> D5
```

[HuggingFace's 🤗 `pipelines`](https://huggingface.co/docs/transformers/pipeline_tutorial) exist to encapsulate these inference steps while cutting the boilerplate code.

# Predicting the species

I will be using this model to embed protein-coding DNA sequences from six species: three animals (human, mouse and fruit fly); one plant (arabidopsis); one bacteria (_E. coli_); and one yeast (_S. cerevisae_).

To this end, I downloaded the DNA sequences of all protein coding genes for the selected species. For each species I randomly subsampled 2,000 sequences of 60 nucleotides each. Half of them were the train set, used for model building; the other half constituted the test set, used exclusively for performance evaluation. You can access the code [here](https://github.com/hclimente/hclimente.github.io/blob/main/assets/python/2025-05-02-hf-transformers/prepare_data.sh).

This is the UMAP of the embedded sequences:

{% include figure.liquid loading="eager" path="assets/python/2025-05-02-hf-transformers/img/umap_embeddings.webp" class="img-fluid rounded z-depth-1" %}

Some disclaimers need to be made. First, I took a minuscule sample of all protein coding sequences, which is somewhat biased towards the beginning of the protein. Second, I am using the smallest Nucleotide Transformer, and its likely that larger models can represent these sequences more richly.

Even with these constraints, sequences from the same species tend to inhabit similar regions of the underlying manifold. If you are unconvinced, just squint your eyes. For those of you still unconvinced, I trained a muticlass logistic regression tasked with predicting the species using only the embeddings. This classifier achieved an accuracy of $$0.47$$, pretty good compared to the accuracy $$\frac 1 6 = 0.16$$ of the random choice. Furthermore, some of the errors are clearly between the most similar species: human and mouse.

{% include figure.liquid loading="eager" path="assets/python/2025-05-02-hf-transformers/img/confusion_matrix_test.webp" class="img-fluid rounded z-depth-1" %}

# Fine-tuning the model

# Conclusions

[The role of HuggingFace's 🤗 in a ML stack]
