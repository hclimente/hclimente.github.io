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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sys

sys.path.append("../")

from utils import (
    save_fig,
)


# %%
def sample_2_gaussian_vectors(mean, cov, n=100, seed=42):
    """
    Generate two correlated Gaussian vectors.
    """
    np.random.seed(seed)
    x = np.random.multivariate_normal(mean, cov, n)
    return x[:, 0], x[:, 1]


# %%
sns.set_context("talk")
sns.set_theme(
    style="white",
    context="notebook",
    font_scale=1,
    palette=None,
)

fig, axs = plt.subplots(4, 4, figsize=(8, 8.5), sharey=True, sharex=True)
variances = [0.25, 0.5, 0.75, 1]
covariances = variances

for i, v in enumerate(variances):
    for j, c in enumerate(covariances):

        if abs(c) > v:
            axs[i, j].set_visible(False)
            continue

        mean = [0, 0]
        cov = [[v, c], [c, v]]
        x1, x2 = sample_2_gaussian_vectors(mean=mean, cov=cov)

        sns.scatterplot(x=x1, y=x2, ax=axs[i, j])
        axs[i, j].tick_params(labelleft=False, labelbottom=False)
        if i == len(variances) - 1:
            axs[i, j].set_xlabel(f"$X_1$\nCov($X_1$, $X_2$) = {c}")

        if j == 0:
            axs[i, j].set_ylabel(f"Var($X_k$) = {v}\n$X_2$")

plt.tight_layout()
plt.show()

# %% [markdown]
# # Partial correlations

# %%
np.random.seed(42)

v_u = 1
v_xy = 0.5
n = 200

df = pd.DataFrame(
    {
        "X": np.random.normal(0, v_xy, n),
        "U": np.random.normal(0, v_u, n),
        "Y": np.random.normal(0, v_xy, n),
    }
)
df["X"] = df["U"] + df["X"]
df["Y"] = df["U"] + df["Y"]

residuals = {}

print("--- Linear Regression Models (for Residuals) ---")
print("Model\tCoeff\tIntercept")
for target_name in df.columns:
    for predictor_name in df.columns:
        if target_name != predictor_name:
            model = LinearRegression()
            model.fit(df[[predictor_name]], df[target_name])
            print(f"{target_name} ~ {predictor_name}", end="\t")
            print(f"{model.coef_[0]:.4f}\t{model.intercept_:.4f}")

            residuals[(target_name, predictor_name)] = df[target_name] - model.predict(
                df[[predictor_name]]
            )

print("-" * 48)

print("\n--- Correlation analysis " + "-" * 23)

fig, ax = plt.subplots(2, 3, figsize=(9, 6))
pairs = (("X", "U"), ("Y", "U"), ("X", "Y"))

print("Var 1\tVar 2\tControl\tCorr\tpCorr")
for i, (target_1, target_2) in enumerate(pairs):
    predictor = (set(df.columns) - set([target_1, target_2])).pop()
    print(f"{target_1}\t{target_2}\t{predictor}\t", end="")

    corr, p = pearsonr(df[target_1], df[target_2])
    print(f"{corr:.2f}{'*' if p < 0.01 else ''}\t", end="")

    sns.scatterplot(x=df[target_1], y=df[target_2], alpha=0.5, ax=ax[0, i])
    ax[0, i].set_xlabel(f"{target_1}")
    ax[0, i].set_ylabel(f"{target_2}")
    ax[0, i].set_title(f"$\\rho_{{{target_1}, {target_2}}}$ = {corr:.2f}, P = {p:.1e}")

    res_1 = residuals[(target_1, predictor)]
    res_2 = residuals[(target_2, predictor)]
    corr, p = pearsonr(res_1, res_2)
    print(f"{corr:.2f}{'*' if p < 0.01 else ''}")

    sns.scatterplot(x=res_1, y=res_2, alpha=0.5, ax=ax[1, i], color="orange")
    ax[1, i].set_xlabel(f"{target_1} | {predictor}")
    ax[1, i].set_ylabel(f"{target_2} | {predictor}")
    ax[1, i].set_title(
        f"$\\rho_{{{target_1}, {target_2} | {predictor}}}$ = {corr:.2f}, P = {p:.1e}"
    )

print("-" * 48)

plt.tight_layout()
save_fig(plt.gcf(), "partial_correlations")

# %%
