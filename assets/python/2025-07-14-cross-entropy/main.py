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
#     display_name: 2025-07-14-cross-entropy
#     language: python
#     name: python3
# ---

# %%
import heapq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from typing import Dict, Optional

sys.path.append("../")

from utils import (
    save_fig,
)

p_ldn_weather = {
    "C": 0.5,
    "R": 0.4,
    "S": 0.1,
}
N = 10


def generate_kmer_probabilities(
    symbol_prob: Dict[str, float], k: int
) -> Dict[str, float]:
    kmer_prob = symbol_prob.copy()

    for _ in range(1, k):
        new_kmer_prob = {}
        for kmer, p_kmer in kmer_prob.items():
            for symbol, p_symbol in symbol_prob.items():
                new_kmer = kmer + symbol
                new_kmer_prob[new_kmer] = p_symbol * p_kmer
        kmer_prob = new_kmer_prob

    kmer_prob = dict(sorted(kmer_prob.items(), key=lambda item: item[1], reverse=True))
    return kmer_prob


p_ldn_kmer = generate_kmer_probabilities(p_ldn_weather, N)


# %%
def generate_huffman_codes(probabilities: Dict[str, float]) -> Dict[str, str]:
    """Generates Huffman codes for a given probability distribution."""
    # Create a priority queue (min-heap) of nodes. Each node is a tuple:
    # (probability, symbol_or_tree)
    heap = [[weight, [symbol, ""]] for symbol, weight in probabilities.items()]
    heapq.heapify(heap)

    # Combine nodes until only one root node remains
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # The heap now contains a single element which is the root of the Huffman tree
    # Return a dictionary of {symbol: code}
    return dict(sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p[0])))


enc_ldn = generate_huffman_codes(p_ldn_kmer)


def get_average_length(kmer_prob: Dict[str, float], encoding: Dict[str, str]) -> float:
    """Calculate the average code length for a given k-mer probability distribution and encoding."""
    k = len(next(iter(kmer_prob)))
    avg_length = 0
    for kmer, prob in kmer_prob.items():
        kmer_enc = encoding.get(kmer, "")
        avg_length += len(kmer_enc) * prob / k
    return avg_length


print(
    f"Average code length Spain weather in Spain: {get_average_length(p_ldn_kmer, enc_ldn):.4f}"
)


# %%
def tabulate_encoding(
    encoding: Dict[str, str], code_prob: Optional[Dict[str, float]] = None
) -> str:
    print(
        "| 10-day weather | Probability | Codeword                         | Codeword length |"
    )
    print(
        "| -------------- | ----------- | -------------------------------- | --------------- |"
    )

    for symbol, code in encoding.items():
        p = code_prob.get(symbol, 0)
        if p == 0:
            continue
        print(
            f"| {symbol}     | {p:.2e}    | {code}{' ' * (32 - len(code))} | {len(code)} {' ' * (14 - len(str(len(code))))} |"
        )


tabulate_encoding(enc_ldn, p_ldn_kmer)

# %%
p_bcn_weather = {
    "C": 0.2,
    "R": 0.1,
    "S": 0.7,
}

p_bcn_kmer = generate_kmer_probabilities(p_bcn_weather, N)
enc_bcn = generate_huffman_codes(p_bcn_kmer)

print(
    f"Average code length Spain weather in Spain: {get_average_length(p_bcn_kmer, enc_bcn):.4f}"
)
print(
    f"Average code length Spain weather in UK: {get_average_length(p_ldn_kmer, enc_bcn):.4f}"
)

# %%
tabulate_encoding(enc_bcn, p_bcn_kmer)

# %% [markdown]
# # Plots
#
# ## Entropy

# %%
avg_lengths = []

for k in range(1, 15):
    p_kmer = generate_kmer_probabilities(p_ldn_weather, k)
    enc_ldn = generate_huffman_codes(p_kmer)
    avg_length = get_average_length(p_kmer, enc_ldn)
    avg_lengths.append([k, avg_length])

df_avg_lengths = pd.DataFrame(avg_lengths, columns=["k", "avg_length"])

sns.set_theme(
    style="white",
    context="notebook",
    font_scale=1,
    palette=None,
)

fig, ax = plt.subplots(figsize=(7, 4.5))

sns.lineplot(
    data=df_avg_lengths,
    x="k",
    y="avg_length",
    marker="o",
    ax=ax,
    linewidth=2,
    color="steelblue",
)

entropy = sum(p * np.log2(1 / p) for p in p_ldn_weather.values())
ax.axhline(
    y=entropy,
    color="firebrick",
    linestyle="--",
    linewidth=2,
    label=f"Entropy ≈ {entropy:.2f}",
)

ax.set_xlabel("Batch size (k)", fontsize=13)
ax.set_ylabel("Average Bits per Day", fontsize=13)
ax.legend(frameon=False)

plt.tight_layout()
save_fig(plt.gcf(), "entropy-batch_size_vs_avg_bits_per_day")

# %% [markdown]
# ## Cross-entropy

# %%
avg_lengths = []

for k in range(1, 15):
    p_bcn_kmer = generate_kmer_probabilities(p_bcn_weather, k)
    p_ldn_kmer = generate_kmer_probabilities(p_ldn_weather, k)

    enc_bcn = generate_huffman_codes(p_bcn_kmer)
    avg_length = get_average_length(p_ldn_kmer, enc_bcn)
    avg_lengths.append([k, avg_length])

df_avg_lengths = pd.DataFrame(avg_lengths, columns=["k", "avg_length"])

sns.set_theme(
    style="white",
    context="notebook",
    font_scale=1,
    palette=None,
)

fig, ax = plt.subplots(figsize=(7, 4.5))

sns.lineplot(
    data=df_avg_lengths,
    x="k",
    y="avg_length",
    marker="o",
    ax=ax,
    linewidth=2,
    color="steelblue",
)

crossentropy = sum(
    p_ldn_weather[k] * np.log2(1 / p_bcn_weather[k]) for k in p_ldn_weather.keys()
)
ax.axhline(
    y=crossentropy,
    color="firebrick",
    linestyle="--",
    linewidth=2,
    label=f"Cross-entropy ≈ {crossentropy:.2f}",
)

ax.set_xlabel("Batch size (k)", fontsize=13)
ax.set_ylabel("Average Bits per Day", fontsize=13)
ax.legend(frameon=False)

plt.tight_layout()
save_fig(plt.gcf(), "crossentropy-batch_size_vs_avg_bits_per_day")
