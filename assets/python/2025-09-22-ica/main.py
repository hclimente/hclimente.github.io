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

# %% [markdown]
# # Examples from scikit-learn
#
# ## ICA vs. PCA

# %%
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA, FastICA

sys.path.append("../")

from utils import (
    save_fig,
)

# Generate sample data
rng = np.random.RandomState(42)
S = rng.standard_t(1.5, size=(20000, 2))
S[:, 0] *= 2.0

## Mix data
A = np.array([[1, 1], [0, 2]])  # Mixing matrix

X = np.dot(S, A.T)  # Generate observations

pca = PCA()
S_pca_ = pca.fit(X).transform(X)

ica = FastICA(random_state=rng, whiten="arbitrary-variance")
S_ica_ = ica.fit(X).transform(X)  # Estimate the sources


# Plot results
def plot_samples(S, axis_list=None):
    plt.scatter(
        S[:, 0], S[:, 1], s=2, marker="o", zorder=10, color="steelblue", alpha=0.5
    )
    if axis_list is not None:
        for axis, color, label in axis_list:
            x_axis, y_axis = axis / axis.std()
            plt.quiver(
                (0, 0),
                (0, 0),
                x_axis,
                y_axis,
                zorder=11,
                width=0.01,
                scale=6,
                color=color,
                label=label,
            )

    plt.hlines(0, -5, 5, color="black", linewidth=0.5)
    plt.vlines(0, -3, 3, color="black", linewidth=0.5)
    plt.xlim(-5, 5)
    plt.ylim(-3, 3)
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")


plt.figure(dpi=300)  # Increased figure size and DPI


# Function to add panel labels
def add_panel_label(label, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.text(-0.2, 1.1, label, transform=ax.transAxes, fontsize=14, fontweight="bold")


# Plot A
plt.subplot(2, 2, 1)
plot_samples(S / S.std())
plt.title("True Independent Sources")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
add_panel_label("A")

# Plot B
plt.subplot(2, 2, 2)
axis_list = [(pca.components_.T, "orange", "PCA"), (ica.mixing_, "red", "ICA")]
plot_samples(X / np.std(X), axis_list=axis_list)
legend = plt.legend(loc="upper left")
legend.set_zorder(100)
plt.title("Observations")
plt.xlabel("x")
plt.ylabel("y")
add_panel_label("B")

# Plot C
plt.subplot(2, 2, 3)
plot_samples(S_ica_ / np.std(S_ica_))
plt.title("ICA recovered signals")
plt.xlabel("Recovered comp. 1")
plt.ylabel("Recovered comp. 2")
add_panel_label("C")

# Plot D
plt.subplot(2, 2, 4)
plot_samples(S_pca_ / np.std(S_pca_))
plt.title("PCA recovered signals")
plt.xlabel("Recovered comp. 1")
plt.ylabel("Recovered comp. 2")
add_panel_label("D")

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
plt.tight_layout()
save_fig(plt.gcf(), "ica_pca_sklearn_example")

# %% [markdown]
# ## Audio signal

# %%
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# %%
sources = {
    "Microphone 1": "red",
    "Microphone 2": "steelblue",
    "Microphone 3": "orange",
}

plt.figure(dpi=300, figsize=(8, 2.1))
for sig, (source, color) in zip(X.T, sources.items()):
    plt.plot(sig, color=color, label=source)
plt.legend(loc="lower left")
plt.xlabel("Time (s)")

plt.tight_layout()
save_fig(plt.gcf(), "cocktail_microphone_signals")

# %%
sources = {
    "Piano": "orange",
    "Speaker 1": "red",
    "Speaker 2": "steelblue",
}

plt.figure(dpi=300, figsize=(8, 2.1))
for sig, (source, color) in zip(S.T, sources.items()):
    plt.plot(sig, color=color, label=source)
plt.legend(loc="lower left")
plt.xlabel("Time (s)")

plt.tight_layout()
save_fig(plt.gcf(), "cocktail_source_signals")

# %%
# Compute ICA
ica = FastICA(n_components=3, whiten="arbitrary-variance", random_state=0)
S_ = ica.fit_transform(X)  # Reconstruct signals

sources = {
    "Component 1": "orange",
    "Component 3": "steelblue",
    "Component 2": "red",
}

plt.figure(dpi=300, figsize=(8, 2.3))
for sig, (source, color) in zip(S_.T, sources.items()):
    plt.plot(sig, color=color, label=source)
plt.legend(loc="lower left")
plt.title("ICA Recovered Signals")
plt.xlabel("Time (s)")

plt.tight_layout()
save_fig(plt.gcf(), "cocktail_ica_signals")
