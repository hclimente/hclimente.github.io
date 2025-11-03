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

# %%
import sys

import duckdb
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append("../")

from utils import (
    save_fig,
)

json_file_path = (
    "/Users/hclimente/Developer/papers_please/results/validated_articles.json"
)

con = duckdb.connect(database=":memory:")

df = con.sql(f"SELECT * FROM read_json_auto('{json_file_path}');").df()


def process_priority(col):
    priority_map = {
        "high": "High",
        "medium": "Medium",
        "low": "Low",
    }
    col = col.map(priority_map)

    priority_categories = ["High", "Medium", "Low", "Failed\nScreening"]
    col = col.fillna("Failed\nScreening")
    col = pd.Categorical(col, categories=priority_categories, ordered=True)
    return col


# as categorical with specific order: low, medium, high
df["my_priority"] = process_priority(df["my_priority"])
df["priority_decision"] = process_priority(df["priority_decision"])


def process_screening(col):
    screening_map = {
        True: "Pass",
        False: "Fail",
    }
    col = col.map(screening_map)

    decision_categories = ["Pass", "Fail"]
    col = pd.Categorical(col, categories=decision_categories, ordered=True)
    return col


df["my_screening"] = process_screening(df["my_screening"])
df["screening_decision"] = process_screening(df["screening_decision"])


# %%
def barplot_priority(data, column, title, subtitle, xlabel):
    """
    Create an Economist-style bar plot with three categories (High, Medium, Low).

    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe containing the data to plot
    column : str
        The column name to count and plot
    title : str
        Main title for the plot
    subtitle : str
        Subtitle (italicized) for the plot
    xlabel : str
        Label for the x-axis
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # The Economist color palette (their signature red/blue scheme)
    economist_colors = ["#e3120b", "#6c7a89", "#3b6182"]

    # Create the count plot
    counts = (
        data[column].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    )
    bars = ax.bar(range(len(counts)), counts.values, color=economist_colors, width=0.6)

    # Styling in The Economist manner
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, fontsize=11, fontfamily="sans-serif")
    ax.set_ylabel("Count", fontsize=11, fontfamily="sans-serif")
    ax.set_xlabel(xlabel, fontsize=11, fontfamily="sans-serif")

    # The Economist typically uses subtle gridlines
    ax.yaxis.grid(True, linestyle="-", alpha=0.2, color="gray", linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    # Title in The Economist style (clear, informative)
    ax.text(
        0.0,
        1.08,
        title,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        fontfamily="sans-serif",
    )
    ax.text(
        0.0,
        1.03,
        subtitle,
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="sans-serif",
        style="italic",
        color="#555555",
    )

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontfamily="sans-serif",
            )

    plt.tight_layout()


# %%
confusion_matrix = pd.crosstab(
    df["my_screening"],
    df["screening_decision"],
    rownames=["Target decision"],
    colnames=["LLM Decision"],
)

fig, ax = plt.subplots(figsize=(8, 6))

# The Economist style heatmap with custom colormap
economist_blues = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08519c"]
cmap = LinearSegmentedColormap.from_list("economist", economist_blues)

# Create heatmap
sns.heatmap(
    confusion_matrix,
    annot=True,
    fmt="d",
    cmap=cmap,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"shrink": 0.8},
    ax=ax,
    annot_kws={"fontsize": 11, "fontfamily": "sans-serif"},
)

# Style the labels
ax.set_xlabel("LLM Decision", fontsize=11, fontfamily="sans-serif", fontweight="normal")
ax.set_ylabel(
    "Target decision", fontsize=11, fontfamily="sans-serif", fontweight="normal"
)

# Rotate labels for better readability
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=0, fontsize=10, fontfamily="sans-serif"
)
ax.set_yticklabels(
    ax.get_yticklabels(), rotation=0, fontsize=10, fontfamily="sans-serif"
)

# Title in The Economist style
ax.text(
    0.0,
    1.10,
    "Screening decision accuracy",
    transform=ax.transAxes,
    fontsize=14,
    fontweight="bold",
    fontfamily="sans-serif",
)
ax.text(
    0.0,
    1.05,
    "Confusion matrix for include/exclude decisions",
    transform=ax.transAxes,
    fontsize=10,
    fontfamily="sans-serif",
    style="italic",
    color="#555555",
)

# Remove spines for cleaner look
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()


# %%
barplot_priority(
    data=df[(df["my_screening"] == "Pass")],
    column="my_priority",
    title="Target Priority*",
    subtitle="*Only for articles that passed the screening",
    xlabel="Target Priority",
)

save_fig(plt.gcf(), "target_priority")

# %%
barplot_priority(
    data=df[(df["my_screening"] == "Fail") & (df["screening_decision"] == "Pass")],
    column="priority_decision",
    title="Priority of the false positives",
    subtitle="Articles that should be excluded, but were included by LLM",
    xlabel="LLM Priority Decision",
)


# %%
screened_df = df[(df["my_screening"] == "Pass")]

confusion_matrix = pd.crosstab(
    screened_df["my_priority"],
    screened_df["priority_decision"],
    rownames=["Target Priority"],
    colnames=["LLM Priority"],
)
confusion_matrix

# %%
fig, ax = plt.subplots(figsize=(8, 6))

# The Economist style heatmap with custom colormap
# Using a blue color scheme that's more muted and professional
economist_blues = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08519c"]
cmap = LinearSegmentedColormap.from_list("economist", economist_blues)

# Create heatmap
sns.heatmap(
    confusion_matrix,
    annot=True,
    fmt="d",
    cmap=cmap,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"shrink": 0.8},
    ax=ax,
    annot_kws={"fontsize": 11, "fontfamily": "sans-serif"},
)

# Style the labels
ax.set_xlabel("LLM Priority", fontsize=11, fontfamily="sans-serif", fontweight="normal")
ax.set_ylabel(
    "Target Priority", fontsize=11, fontfamily="sans-serif", fontweight="normal"
)

# Rotate labels for better readability
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=0, fontsize=10, fontfamily="sans-serif"
)
ax.set_yticklabels(
    ax.get_yticklabels(), rotation=0, fontsize=10, fontfamily="sans-serif"
)

# Title in The Economist style
ax.text(
    0.0,
    1.10,
    "Priority assignment accuracy",
    transform=ax.transAxes,
    fontsize=14,
    fontweight="bold",
    fontfamily="sans-serif",
)
ax.text(
    0.0,
    1.05,
    "Confusion matrix for articles that should be included",
    transform=ax.transAxes,
    fontsize=10,
    fontfamily="sans-serif",
    style="italic",
    color="#555555",
)

# Remove spines for cleaner look
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()


# %%
