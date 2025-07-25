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
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from typing import Dict, Optional

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


# %%
def generate_huffman_codes(probabilities: Dict[str, float]) -> Dict[str, str]:
    """Generates optimal prefix codes (Huffman codes) for a given probability distribution."""
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


def plot_entropy_bars(
    p_dist: Dict[str, float],
    q_dist: Optional[Dict[str, float]] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Visualizes Entropy or Cross-Entropy using variable-width bars.

    - For Entropy: Call with just p_dist. Widths and code lengths are from p_dist.
    - For Cross-Entropy H(p, q): Call with p_dist and q_dist.
      Widths are from p_dist (true probability), but code lengths are from q_dist (model).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Determine if we're calculating Entropy or Cross-Entropy
    is_cross_entropy = q_dist is not None
    code_source_dist = q_dist if is_cross_entropy else p_dist

    # Generate optimal codes based on the source distribution
    huffman_codes = generate_huffman_codes(code_source_dist)

    # Prepare data for plotting
    labels = list(p_dist.keys())
    probs = [p_dist[label] for label in labels]
    code_lengths = [len(huffman_codes[label]) for label in labels]

    # Calculate the final value (Entropy or Cross-Entropy)
    total_avg_length = sum(p * length for p, length in zip(probs, code_lengths))

    # Set up plot title and colors
    if is_cross_entropy:
        title = f"Cross-Entropy H(p, q) = {total_avg_length:.2f} bits"
        palette = sns.color_palette("Reds_d", n_colors=len(labels))
    else:
        title = f"Entropy H(p) = {total_avg_length:.2f} bits"
        palette = sns.color_palette("Blues_d", n_colors=len(labels))

    # Sort items for drawing, typically by probability for a clean look
    sorted_items = sorted(
        zip(labels, probs, code_lengths, palette), key=lambda x: x[1], reverse=True
    )

    current_x = 0
    for label, prob, length, color in sorted_items:
        # Create a rectangle patch: xy, width, height
        rect = patches.Rectangle(
            (current_x, 0),
            width=prob,
            height=length,
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            label=f"{label} (p={prob:.2f}, len={length})",
        )
        ax.add_patch(rect)

        # Add text label inside the rectangle
        text_color = "white"
        ax.text(
            current_x + prob / 2,
            length / 2,
            f"{label}\n{length} bits",
            ha="center",
            va="center",
            color=text_color,
            fontsize=12,
            fontweight="bold",
        )
        current_x += prob

    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(code_lengths) * 1.15)
    ax.set_xlabel("Cumulative Probability", fontsize=12)
    ax.set_ylabel("Code Length (bits)", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # %%
    # Set a pretty theme
    sns.set_theme(style="ticks")

    # Define the true (London) and model (Barcelona) probability distributions
    p_london_weather = {"Cloudy": 0.5, "Rainy": 0.4, "Sunny": 0.1}
    q_barcelona_weather = {"Sunny": 0.7, "Cloudy": 0.2, "Rainy": 0.1}

    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Entropy of London Weather H(p)
    # The codes are optimized for the London distribution itself.
    plot_entropy_bars(p_dist=p_london_weather, ax=ax1)

    # Plot 2: Cross-Entropy H(p, q)
    # We experience London weather (widths from p) but use codes
    # optimized for Barcelona weather (heights from q).
    plot_entropy_bars(p_dist=p_london_weather, q_dist=q_barcelona_weather, ax=ax2)

    fig.suptitle(
        "Visualizing Entropy vs. Cross-Entropy", fontsize=20, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# %%
