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
#     display_name: x-VW6uKvAAFJ6FbaR1Atg
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from sklearn.decomposition import PCA

iris = sklearn.datasets.load_iris()
X, y = iris.data, iris.target

# %%
pca = PCA(n_components=2)

pca.fit(X)
X_pca = pca.transform(X)

for colored in [False, True]:
    plt.figure()
    plt.gca().set_axis_off()

    if colored:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor='none', alpha=0.5, s=40)
        fig_path = f"img/pca_colored.jpg"
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='none', alpha=0.5, s=40)
        fig_path = f"img/pca_plain.jpg"

    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.show()

# %%
