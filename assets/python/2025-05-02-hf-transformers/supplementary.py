# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Playing with the tokenizer
#
# There are 4,107 tokens in total:
#
# - 4,096 for each possible 6-mer
# - 5 for each possible input nucleotide (A, T, C, G and N). They are used as fallback when the sequence is not divisible by 6.
# - 1 for padding (PAD)
# - 1 for the mask (MASK)
# - 1 for the entire sequence (CLS)
# - 3 additional tokens, which I will mostly ignore
#
# All summed up, 4,107 tokens, each represented by an integer.

# %%
from transformers import AutoTokenizer

# %% [markdown]
# Let's see how the tokenizer works by studying a few examples. Let's first load the models:

# %%
tokenizer = AutoTokenizer.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", trust_remote_code=True
)

# %% [markdown]
# Let's tokenize the empty sequence. This should only contain the CLS token:

# %%
tokenizer([""])["input_ids"]

# %% [markdown]
# Neat. We see that NT uses integer 3 for CLS. Let's see now how the individual nucleotides are embedded by feeding the tokenizer five 1-nucleotide sequences. Since they are only 1 nucleotide long, it won't use any token representing 6-mer:

# %%
tokenizer(["A", "T", "C", "G", "N"])["input_ids"]

# %% [markdown]
# We see now that A = 4102, T = 4103, C = 4104, G = 4105 and N = 4106. Let's play a bit with the sequence length:

# %%
tokenizer(["A", "AA", "AAA", "AAAA", "AAAAA", "AAAAA", "AAAAAA"])["input_ids"]

# %% [markdown]
# As soon as the length of the sequence is longer than 6, the tokenizer starts using tokens to represent an 6-mer. In this case, AAAAAA = 6. Let's see now how order matters:

# %%
tokenizer(["AAAAAAT", "TAAAAAA"])["input_ids"]

# %% [markdown]
# Interesting! This shows us that the tokenizer reads left-to-right in a greedy manner. It tries to split the sequence into 6-mers starting from the left. If there is any subsequence remaining, it will revert to individual nucleotides. Hence, it decomposes AAAAAAT as AAAAAA + T, and TAAAAAA as TAAAAA + A.
#
# Let's look into the PAD token. As we have seen, the tokenizer depending on the size of inputs and how they are tokenized, the outputs can be different. The PAD token is a special token to ensure that all output tokens have the same size. We do this by specifying a padding strategy:

# %%
tokenizer(["AAAAAAAAAAAAAAAAAAAAAAAA", "AAAAAA", "A"], padding="longest")["input_ids"]

# %% [markdown]
# The padding strategy was to find the length of the longest tokenized sequence, and add pad tokens to the shorter tokenized sequences until their length is the same. In this case, the longest sequence was `AAAAAAAAAAAAAAAAAAAAAAAA`, which is reduced to 4 tokens (4 × `AAAAAA`). The other two sequences were padded with the PAD token (1).
#
# We can access all the tokens:

# %%
i = 0
for k, v in tokenizer.get_vocab().items():
    print(f"{k}:\t{v}")
    i += 1

    if i == 15:
        break

# %%
len(tokenizer.get_vocab())

# %% [markdown]
# The tokenizer contains tokens that were not considered in the NT article. Some of them are needed to make the model robust to artifacts, like the unknown token (e.g., if our sequence contained other [IUPAC nucleotides](https://en.wikipedia.org/wiki/Nucleic_acid_notation)). The vocabulary also contains tokens to specify the beginning and the end of the sequence, let's see if the tokenizer is able to use them.

# %%
tokenizer(["A"], add_special_tokens=True)["input_ids"]

# %% [markdown]
# It seems it doesn't.
#
# Last, we should consider the maximum input size of the model. Transformer models cannot take sequences of any length. Specifically, the NT can consider at most sequences of 2,048 tokens (~12 kbp). We can make the tokenizer aware of this limit (`max_length=2048`) and instructing it to truncate longer sequences (`truncation=True`).

# %%
tokenizer(["A"], max_length=5000, truncation=True)["input_ids"]

# %% [markdown]
# Worked out example:

# %%
tokenizer(["ATGGTAGCTACATCATCTG"], max_length=5000, truncation=True)["input_ids"]
