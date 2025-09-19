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
#     display_name: 2025-05-29-precision-matrix
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
from sklearn.linear_model import LinearRegression
import sys
import time

sys.path.append("../")

from utils import (
    save_fig,
)


# %%
def invert_cov(cov, psd=True):
    """
    Invert a covariance matrix using Cholesky decomposition.
    If `psd` is False, it uses LU decomposition.
    """
    if psd:
        n = cov.shape[0]
        eye = np.identity(n)
        return linalg.solve(cov, eye, assume_a="positive definite", overwrite_b=True)
    else:
        return np.linalg.inv(cov)


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

N = 200

df = pd.DataFrame(
    {
        "U": np.random.normal(0, 1, N),
        "X": np.random.normal(0, 0.5, N),
        "Y": np.random.normal(0, 0.5, N),
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

pairs = (("X", "U"), ("Y", "U"), ("X", "Y"))

print("Var 1\tVar 2\tControl\tCorr\tpCorr")
for i, (target_1, target_2) in enumerate(pairs):
    predictor = (set(df.columns) - set([target_1, target_2])).pop()
    print(f"{target_1}\t{target_2}\t{predictor}\t", end="")

    corr, p = pearsonr(df[target_1], df[target_2])
    print(f"{corr:.2f}{'*' if p < 0.01 else ''}\t", end="")

    res_1 = residuals[(target_1, predictor)]
    res_2 = residuals[(target_2, predictor)]
    corr, p = pearsonr(res_1, res_2)
    print(f"{corr:.2f}{'*' if p < 0.01 else ''}")

print("-" * 48)

# %%
fig, ax = plt.subplots(1, 3, figsize=(9, 3.3))
pairs = (("X", "U"), ("Y", "U"), ("X", "Y"))

for i, (target_1, target_2) in enumerate(pairs):
    predictor = (set(df.columns) - set([target_1, target_2])).pop()

    corr, p = pearsonr(df[target_1], df[target_2])

    sns.scatterplot(x=df[target_1], y=df[target_2], alpha=0.5, ax=ax[i])
    ax[i].set_xlabel(f"{target_1}")
    ax[i].set_ylabel(f"{target_2}")
    ax[i].set_title(
        f"$\\hat{{\\rho}}_{{{target_1}, {target_2}}}$ = {corr:.2f}, P = {p:.1e}"
    )

plt.tight_layout()
save_fig(plt.gcf(), "correlations")

# %%
fig, ax = plt.subplots(1, 3, figsize=(9, 3.3))
pairs = (("X", "U"), ("Y", "U"), ("X", "Y"))

for i, (target_1, target_2) in enumerate(pairs):
    predictor = (set(df.columns) - set([target_1, target_2])).pop()

    res_1 = residuals[(target_1, predictor)]
    res_2 = residuals[(target_2, predictor)]
    corr, p = pearsonr(res_1, res_2)

    sns.scatterplot(x=res_1, y=res_2, alpha=0.5, ax=ax[i], color="orange")
    ax[i].set_xlabel(f"{target_1} | {predictor}")
    ax[i].set_ylabel(f"{target_2} | {predictor}")
    ax[i].set_title(
        f"$\\hat{{\\rho}}_{{{target_1}, {target_2} | {predictor}}}$ = {corr:.2f}, P = {p:.1e}"
    )

plt.tight_layout()
save_fig(plt.gcf(), "partial_correlations")

# %%
np.random.seed(42)

N = 200

df = pd.DataFrame(
    {
        "U": np.random.normal(0, 1, N),
        "X": np.random.normal(0, 0.5, N),
        "Y": np.random.normal(0, 0.5, N),
    }
)
df["X"] = df["U"] + df["X"]
df["Y"] = df["U"] + df["Y"]

centered_df = df - df.mean()
covariance = np.dot(centered_df.T, centered_df) / (N - 1)
precision = invert_cov(covariance)

normalization_factors = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
partial_correlations = precision / normalization_factors

print(np.round(partial_correlations, 2))

# %%
model = GraphicalLassoCV().fit(df)

precision = model.precision_
norm_matrix = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
partial_correlations = precision / norm_matrix

print(np.round(partial_correlations, 2))


# %% [markdown]
# # Implementations


# %%
def pcorr_residuals(X: np.ndarray) -> np.ndarray:  # (n, p) -> (p, p)
    """
    Compute the matrix of partial correlations from the residuals of linear regression models.

    Parameters
    ----------
    X : np.ndarray
        The input data matrix.

    Returns
    -------
    np.ndarray
        The matrix of partial correlations.
    """

    n, p = X.shape

    # dictionary to store the centered residuals
    # keys are tuples of (target, excluded) and values are the residuals
    residuals = np.empty((p, p), dtype=object)

    for i in range(p):
        for j in range(i + 1, p):
            idx_covars = [k for k in range(p) if k != i and k != j]
            X_covars = X[:, idx_covars]

            for target, excluded in [(i, j), (j, i)]:
                y = X[:, target]

                beta, *_ = np.linalg.lstsq(X_covars, y, rcond=None)
                y_pred = X_covars @ beta

                r = y - y_pred

                # center the residuals
                residuals[(target, excluded)] = r - r.mean()

    pcorr = np.eye(p)

    for i in range(p):
        for j in range(i + 1, p):
            res_1 = residuals[(i, j)]
            res_2 = residuals[(j, i)]
            corr = np.dot(res_1, res_2) / (
                np.linalg.norm(res_1) * np.linalg.norm(res_2)
            )

            pcorr[i, j] = pcorr[j, i] = corr

    return pcorr


pcorr_residuals(df.values)


# %%
def pcorr_linalg(X: np.ndarray, psd: bool = True) -> np.ndarray:  # (n, p) -> (p, p)
    """
    Compute the matrix of partial correlations from the covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        The input data matrix.

    Returns
    -------
    np.ndarray
        The matrix of partial correlations.
    """

    n, p = X.shape

    def invert_matrix(cov, psd):
        """
        Invert a covariance matrix using Cholesky decomposition.
        If `psd` is False, it uses LU decomposition.
        """
        if psd:
            n = cov.shape[0]
            eye = np.identity(n)
            return linalg.solve(
                cov, eye, assume_a="positive definite", overwrite_b=True
            )
        else:
            return np.linalg.inv(cov)

    centered_X = X - X.mean(axis=0)
    covariance = np.dot(centered_X.T, centered_X) / (n - 1)
    precision = invert_matrix(covariance, psd=psd)

    normalization_factors = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    partial_correlations = precision / normalization_factors

    return partial_correlations, covariance, precision


# %%
X = np.random.randn(1000, 100)

start = time.time()
p_linalg_psd, _, _ = pcorr_linalg(X)
print("Linalg-psd method took:", time.time() - start)

start = time.time()
p_linalg_lu, _, _ = pcorr_linalg(X, psd=False)
print("Linalg-lu method took:", time.time() - start)

assert np.allclose(p_linalg_psd, p_linalg_lu)

start = time.time()
p_residuals = pcorr_residuals(X)
print("Residuals method took:", time.time() - start)

# %% [markdown]
# # High-dimensional example

# %%
# Define partial correlations
partial_corr = np.eye(20)
partial_corr[0, 9] = partial_corr[9, 0] = -0.95
partial_corr[0, 5] = partial_corr[5, 0] = -0.5
partial_corr[4, 14] = partial_corr[14, 4] = -0.1
partial_corr[10, 15] = partial_corr[15, 10] = 0.95
partial_corr[19, 18] = partial_corr[18, 19] = 0.1

# Convert to precision matrix
p = partial_corr.shape[0]
precision = np.zeros((p, p))

# Assume unit variances (precision[ii] = 1), then use:
for i in range(p):
    precision[i, i] = 1.0
    for j in range(i + 1, p):
        precision[i, j] = precision[j, i] = -partial_corr[
            i, j
        ]  # since precision[ii] = 1

# Now convert to covariance matrix
cov = np.linalg.inv(precision)

# Sample data
mean = np.zeros(p)
n_samples = 20
X = np.random.multivariate_normal(mean, cov, size=n_samples)


# %%
def plot_matrix(matrix, ax, **kwargs):
    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        square=True,
        center=0,
        cbar_kws={"shrink": 0.5},
        ax=ax,
        **kwargs,
    )
    ax.set_xticks([])
    ax.set_yticks([])


fig, ax = plt.subplots(3, 4, figsize=(15, 9))

row_labels = ["Covariance", "Precision", "Structure"]
col_labels = ["Ground Truth", "Maximum Likelihood", "Ledoit-Wolf", "Graphical Lasso"]

plot_matrix(cov, ax[0, 0])
plot_matrix(precision, ax[1, 0])
plot_matrix(precision != 0, ax[2, 0], vmin=0, vmax=1)

mle_pcorr, mle_cov, mle_precision = pcorr_linalg(X, psd=False)
mle_structure = np.abs(mle_pcorr) > 0.1
plot_matrix(mle_cov, ax[0, 1])
plot_matrix(mle_precision, ax[1, 1])
plot_matrix(mle_structure, ax[2, 1], vmin=0, vmax=1)

model = LedoitWolf().fit(X)
lw_precision = model.precision_
lw_denom = np.sqrt(np.outer(np.diag(lw_precision), np.diag(lw_precision)))
lw_structure = np.abs(lw_precision / lw_denom) > 0.1
plot_matrix(model.covariance_, ax[0, 2])
plot_matrix(model.precision_, ax[1, 2])
plot_matrix(lw_structure, ax[2, 2], vmin=0, vmax=1)
# indicate that a filter was applied
ax[2, 2].text(22, 2, "*", fontsize=40, ha="center", va="center")

model = GraphicalLassoCV().fit(X)
plot_matrix(model.covariance_, ax[0, 3])
plot_matrix(model.precision_, ax[1, 3])
plot_matrix(model.precision_ != 0, ax[2, 3], vmin=0, vmax=1)

# Add big labels to the left (rows)
for i, label in enumerate(row_labels):
    ax[i, 0].set_ylabel(
        label, rotation=90, ha="center", va="center", fontsize=19, labelpad=20
    )

# Add big labels to the top (columns)
for j, label in enumerate(col_labels):
    ax[0, j].set_title(label, ha="center", fontsize=18, pad=12)

plt.tight_layout()
save_fig(plt.gcf(), "high_dimensional_experiments")

# %%
