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
from dahuffman import HuffmanCodec

day_prob = {
    "C": 0.5,
    "R": 0.4,
    "S": 0.1,
}
symbol_prob = day_prob.copy()

N = 10

for _ in range(1, N):
    new_symbol_prob = {}
    for seq, p_seq in symbol_prob.items():
        for day, p_day in day_prob.items():
            new_seq = seq + day
            new_symbol_prob[new_seq] = p_day * p_seq
    symbol_prob = new_symbol_prob

codec = HuffmanCodec.from_frequencies(symbol_prob)

avg_length = 0
for symbol, (bits, _) in codec._table.items():
    avg_length += bits * symbol_prob.get(symbol, 0) / N
print(f"Average code length: {avg_length:.2f}")


# %%
print("| 10-day weather | Probability | Length of the binary string |")
print("|----------------|-------------|-----------------------------|")

code = codec._table.items()
# sort by probability
code = sorted(code, key=lambda x: symbol_prob.get(x[0], 0), reverse=True)

for symbol, (bits, _) in code:
    p = symbol_prob.get(symbol, 0)
    if p == 0:
        continue
    print(f"| {symbol}     | {p:.2e}    | {bits}                          |")
