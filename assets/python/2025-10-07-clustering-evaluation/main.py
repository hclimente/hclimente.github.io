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
from itertools import cycle, islice, product
import operator
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_moons,
)
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AgglomerativeClustering,
    SpectralClustering,
    MeanShift,
    DBSCAN,
    HDBSCAN,
    OPTICS,
    AffinityPropagation,
    Birch,
)
from sklearn.metrics import (
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.append("../")

from utils import (
    save_fig,
)

warnings.filterwarnings("ignore")

# %% [markdown]
# # Toy example
#
# From https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html

# %%
n_samples = 500
seed = 0

noisy_circles = make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed
)
noisy_moons = make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
blobs = make_blobs(n_samples=n_samples, random_state=seed)

# random data, no structure
rng = np.random.RandomState(seed)
no_structure = rng.rand(n_samples, 2), np.array([-1] * n_samples)

# anisotropicly distributed data
X, y = make_blobs(n_samples=n_samples, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=170)

# %%
datasets = {
    "Noisy Circles": noisy_circles,
    "Noisy Moons": noisy_moons,
    "Blobs": blobs,
    "Anisotropicly Distributed": aniso,
    "Varied": varied,
    "No Structure": no_structure,
}

clustering_algorithms = {
    "Ground\nTruth": (None, None),
    "K-Means": (KMeans, {"n_clusters": range(2, 6)}),
    "Batch\nK-Means": (MiniBatchKMeans, {"n_clusters": range(2, 6)}),
    "Agglomerative\nClustering": (AgglomerativeClustering, {"n_clusters": range(2, 6)}),
    "Ward": (AgglomerativeClustering, {"n_clusters": range(2, 6), "linkage": ["ward"]}),
    "Average\nLinkage": (
        AgglomerativeClustering,
        {"n_clusters": range(2, 6), "linkage": ["average"], "metric": ["cityblock"]},
    ),
    "Spectral\nClustering": (SpectralClustering, {"n_clusters": range(2, 6)}),
    "MeanShift": (MeanShift, {"bandwidth": [0.5, 1.0, 1.5, 2.0]}),
    "DBSCAN": (DBSCAN, {"eps": [0.1, 0.2, 0.3]}),
    "HDBSCAN": (
        HDBSCAN,
        {
            "allow_single_cluster": [True, False],
            "min_cluster_size": [5, 10, 15],
            "min_samples": [5, 10],
        },
    ),
    "OPTICS": (OPTICS, {"min_samples": [5, 10, 15], "xi": [0.01, 0.05, 0.1]}),
    "Affinity\nPropagation": (
        AffinityPropagation,
        {"damping": [0.5, 0.7, 0.9], "preference": [-50, -100, -200]},
    ),
    "Birch": (Birch, {"n_clusters": range(2, 6)}),
    "Gaussian\nMixture": (
        mixture.GaussianMixture,
        {
            "n_components": range(2, 6),
            "covariance_type": ["full", "tied", "diag", "spherical"],
        },
    ),
}

metrics = {
    # "Calinski-Harabasz Index": (calinski_harabasz_score, {}, operator.gt),
    # "Davies-Bouldin Index": (davies_bouldin_score, {}, operator.lt),
    "Silhouette Score": (silhouette_score, {"metric": "euclidean"}, operator.gt),
}

res = {}


def get_params(param_grid):
    if not param_grid:
        yield {}
        return

    keys, values = zip(*param_grid.items())
    for v in product(*values):
        params = dict(zip(keys, v))
        yield params


for i_dataset, (dataset_name, (X, y)) in enumerate(datasets.items()):
    for i_clustering, (clustering_name, (cluster, clustering_param_grid)) in enumerate(
        clustering_algorithms.items()
    ):
        for i_metric, (metric_name, (metric, metric_params, is_better)) in enumerate(
            metrics.items()
        ):
            res.setdefault(metric_name, {})
            res[metric_name].setdefault(dataset_name, {})
            res[metric_name][dataset_name][clustering_name] = {
                "score": float("-inf"),
                "params": {},
                "y_pred": np.array([-1] * len(y)),
                "fit_time": None,
            }

            if clustering_name == "Ground\nTruth":
                try:
                    score = metric(X, y, **metric_params)
                except ValueError:
                    score = None

                res[metric_name][dataset_name][clustering_name] = {
                    "score": score,
                    "params": {},
                    "y_pred": y,
                    "fit_time": None,
                }
                continue

            for params in get_params(clustering_param_grid):
                clf = cluster(**params)
                y_pred = clf.fit_predict(X)
                try:
                    y_pred = clf.labels_.astype(int)
                except AttributeError:
                    y_pred = clf.predict(X)

                fit_time = clf._fit_time if hasattr(clf, "_fit_time") else 0.0

                try:
                    score = metric(X, y_pred, **metric_params)
                except ValueError:
                    continue

                if is_better(
                    score, res[metric_name][dataset_name][clustering_name]["score"]
                ):
                    res[metric_name][dataset_name][clustering_name] = {
                        "score": score,
                        "params": params,
                        "y_pred": y_pred,
                        "fit_time": fit_time,
                    }


# %%
plotting_metric = "Silhouette Score"

plt.figure(figsize=(13.5, 5.8))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
)

plot_num = 1
for i_dataset, (dataset_name, res_dataset) in enumerate(res[plotting_metric].items()):
    # Get the correct X, y for this dataset
    X, _ = datasets[dataset_name]
    X = StandardScaler().fit_transform(X)

    for i_clustering, (clustering_name, res_dict) in enumerate(res_dataset.items()):
        y_pred = res_dict["y_pred"]

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(clustering_name, size=12)

        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=4, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())

        plot_num += 1

save_fig(plt.gcf(), "clustering_on_toy_datasets")

# %%
