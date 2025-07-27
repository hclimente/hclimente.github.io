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
from typing import Dict, Optional

p_london_weather = {
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


p_london_kmer = generate_kmer_probabilities(p_london_weather, N)


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


enc_london = generate_huffman_codes(p_london_kmer)


def get_average_length(kmer_prob: Dict[str, float], encoding: Dict[str, str]) -> float:
    """Calculate the average code length for a given k-mer probability distribution and encoding."""
    k = len(next(iter(kmer_prob)))
    avg_length = 0
    for kmer, prob in kmer_prob.items():
        kmer_enc = encoding.get(kmer, "")
        avg_length += len(kmer_enc) * prob / k
    return avg_length


print(
    f"Average code length Spain weather in Spain: {get_average_length(p_london_kmer, enc_london):.4f}"
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


tabulate_encoding(enc_london, p_london_kmer)

# %%
p_barcelona_weather = {
    "C": 0.2,
    "R": 0.1,
    "S": 0.7,
}

p_barcelona_kmer = generate_kmer_probabilities(p_barcelona_weather, N)
enc_barcelona = generate_huffman_codes(p_barcelona_kmer)

print(
    f"Average code length Spain weather in Spain: {get_average_length(p_barcelona_kmer, enc_barcelona):.4f}"
)
print(
    f"Average code length Spain weather in UK: {get_average_length(p_london_kmer, enc_barcelona):.4f}"
)

# %%
tabulate_encoding(enc_barcelona, p_barcelona_kmer)
