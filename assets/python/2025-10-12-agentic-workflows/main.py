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
import json
import sys

import duckdb
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

sys.path.append("../")

from utils import (
    save_fig,
)

# articles.json is stored where the agent can't see it
results_json = "articles.json"


def process_screening(col):
    screening_map = {
        True: "Pass",
        False: "Fail",
    }
    col = col.map(screening_map)

    decision_categories = ["Pass", "Fail"]
    col = pd.Categorical(col, categories=decision_categories, ordered=True)
    return col


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


def cool_barplot(
    values, title, subtitle, xlabel, palette=["#DB444B", "#6c7a89", "#006BA2"], ax=None
):
    """
    Create an Economist-style bar plot with three categories (High, Medium, Low).

    Parameters:
    -----------
    values: pd.Series
        The dataframe containing the data to plot
    title : str
        Main title for the plot
    subtitle : str
        Subtitle (italicized) for the plot
    xlabel : str
        Label for the x-axis
    palette : list
        List of colors to use for the bars
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure and axes.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes object with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    counts = values.value_counts()
    counts = counts[counts > 0]

    bars = ax.bar(
        range(len(counts)), counts.values, color=palette[: len(counts)], width=0.6
    )

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, fontsize=12, fontfamily="sans-serif")
    ax.set_ylabel("Count", fontsize=12, fontfamily="sans-serif")
    ax.set_xlabel(xlabel, fontsize=12, fontfamily="sans-serif")

    ax.yaxis.grid(True, linestyle="-", alpha=0.2, color="gray", linewidth=0.5)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    ax.text(
        0.0,
        1.12,
        title,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        fontfamily="sans-serif",
    )
    ax.text(
        0.0,
        1.06,
        subtitle,
        transform=ax.transAxes,
        fontsize=11,
        fontfamily="sans-serif",
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
                fontsize=14,
                fontfamily="sans-serif",
            )

    return ax


# %%
con = duckdb.connect(database=":memory:")

df = con.sql(f"SELECT * FROM read_json_auto('{results_json}');").df()

df["date"] = df["date"].astype("str")
df["access_date"] = df["access_date"].astype("str")

# create train/test splits
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

priority_fields = ["doi", "my_screening", "my_priority"]
json_fields = ["title", "url", "journal_name", "date", "access_date", "raw_contents"]

priority2_fields = ["doi", "screening_decision", "priority_decision"]
# train_df[priority2_fields].to_csv("results/0_train_screen_priority.csv", index=False)
# test_df[priority2_fields].to_csv("results/0_test_screen_priority.csv", index=False)

train_df[priority_fields].to_csv("train_articles.csv", index=False)
json.dump(
    [row.dropna().to_dict() for _, row in train_df[json_fields].iterrows()],
    open("train_articles.json", "w"),
    indent=4,
)
test_df[priority_fields].to_csv("test_articles.csv", index=False)
json.dump(
    [row.dropna().to_dict() for _, row in test_df[json_fields].iterrows()],
    open("test_articles.json", "w"),
    indent=4,
)

train_df["my_priority"] = process_priority(train_df["my_priority"])
train_df["priority_decision"] = process_priority(train_df["priority_decision"])
train_df["my_screening"] = process_screening(train_df["my_screening"])
train_df["screening_decision"] = process_screening(train_df["screening_decision"])

# %%
perc_pass = 100 * sum(train_df["my_screening"] == "Pass") / train_df.shape[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

cool_barplot(
    values=train_df["my_screening"],
    title=f"{perc_pass:.0f}% of articles pass",
    subtitle="Based on manual review",
    xlabel="Screening decision",
    palette=["#DB444B", "#006BA2"],
    ax=ax1,
    values2=test_df["my_screening"],
)

perc_high = (
    100
    * sum((train_df["my_screening"] == "Pass") & (train_df["my_priority"] == "High"))
    / sum(train_df["my_screening"] == "Pass")
)
cool_barplot(
    values=train_df[(train_df["my_screening"] == "Pass")]["my_priority"],
    title=f"{perc_high:.0f}% of articles are high priority",
    subtitle="Based on manual review ",
    xlabel="Priority",
    ax=ax2,
    values2=test_df[(test_df["my_screening"] == "Pass")]["my_priority"],
)

plt.tight_layout()

save_fig(plt.gcf(), "target_priority")

# %% [markdown]
# # Comparison to score systems

# %%
tr_score_df = pd.read_json("results/1_train_scoring.json")
tr_score_df = tr_score_df.merge(
    train_df[["doi", "my_priority", "priority_decision"]], on="doi", how="left"
)

# %%
# boxplot my decisions vs LLM score
sns.boxplot(
    data=tr_score_df,
    x="my_priority",
    y="score",
    order=["Low", "Medium", "High"],
    palette=["#006BA2", "#FFB400", "#DB444B"],
)
plt.title("LLM Final Score vs. My Priority Decisions")
plt.xlabel("Manual Priority")
plt.ylabel("LLM Score")

# %%
tr_score_df[(tr_score_df["my_priority"] == "Medium") & (tr_score_df["score"] > 8)][
    "title"
].values

# %%
tr_score_df[(tr_score_df["my_priority"] == "High") & (tr_score_df["score"] < 4)][
    ["title", "doi", "score", "reasoning"]
].values

# %%
tr_score_df[(tr_score_df["my_priority"] == "High") & (tr_score_df["score"] < 3)][
    "title"
].values

# %%
confusion_matrix = pd.crosstab(
    train_df["my_screening"],
    train_df["screening_decision"],
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
cool_barplot(
    values=train_df[
        (train_df["my_screening"] == "Fail")
        & (train_df["screening_decision"] == "Pass")
    ]["priority_decision"],
    title="Priority of the false positives",
    subtitle="Articles that should be excluded, but were included by LLM",
    xlabel="LLM Priority Decision",
)


# %%
screened_df = train_df[(train_df["my_screening"] == "Pass")]

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
