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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


data = fetch_openml(name="AP_Colon_Kidney", version=1, as_frame=True, parser="auto")

# %%
data.frame.shape

# %%
data.frame.shape

# %%
# do a PCA and plot the first two components
# data.frame
X = data.frame.drop(columns=["Tissue"])
y = data.frame["Tissue"]
X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y.map({"Colon": 0, "Kidney": 1}),
    cmap="viridis",
    alpha=0.7,
)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA of AP_Colon_Kidney Dataset")

# %%
