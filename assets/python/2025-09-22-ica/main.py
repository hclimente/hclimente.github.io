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
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
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

# %% [markdown]
# # ICA implementation

# %%
# Generate sample data
m = 2
n = 20000
rng = np.random.RandomState(42)
S = rng.standard_t(1.5, size=(n, m))
S[:, 0] *= 2.0

## Mix data
A = np.array([[1, 1], [0, 2]])  # Mixing matrix

X = np.dot(S, A.T)  # Generate observations

# Preprocess data: center and whiten
X_centered = X - np.mean(X, axis=0)
X_scaled = X_centered / np.std(X_centered, axis=0)

pca = PCA(whiten=True)
X_whitened = pca.fit_transform(X_centered)


# %%
def animate(
    frame,
    animation_data,
    X_alter,
    alter_title,
    previous_w_white=None,
    previous_w_raw=None,
):
    data = animation_data[frame]

    plt.suptitle(
        f"Iteration {data['iteration']}",
        fontsize=16,
        horizontalalignment="left",
        fontweight="bold",
    )

    # Clear axes
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # Plot histogram
    cropped_projection = data["projection"][
        (data["projection"] > -5) & (data["projection"] < 5)
    ]
    ax1.hist(cropped_projection, bins=10, alpha=0.7, color="steelblue", density=True)
    ax1.set_title(f"Excess kurtosis: {data['kurtosis']:.2f}")
    ax1.set_xlabel("Projected values (cropped to [-5, 5])")
    ax1.set_ylabel("Density")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 5)  # Fixed x-axis for better comparison
    ax1.set_ylim(0, 0.9)  # Fixed y-axis for better comparison

    # Plot weight vector direction on original data
    ax2.scatter(X_alter[:, 0], X_alter[:, 1], s=2, alpha=0.5, color="steelblue")

    ax2.hlines(0, -5, 5, color="black", linewidth=0.5)
    ax2.vlines(0, -5, 5, color="black", linewidth=0.5)

    if previous_w_white is not None:
        ax2.arrow(
            0,
            0,
            previous_w_white[0],
            previous_w_white[1],
            head_width=0.2,
            head_length=0.3,
            fc="black",
            ec="black",
            linewidth=2,
        )
    ax2.arrow(
        0,
        0,
        data["weight_white"][0],
        data["weight_white"][1],
        head_width=0.2,
        head_length=0.3,
        fc="red",
        ec="red",
        linewidth=2,
    )

    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(alter_title)
    ax2.grid(True, alpha=0.3)

    # Draw weight vector
    ax3.scatter(X_scaled[:, 0], X_scaled[:, 1], s=2, alpha=0.5, color="steelblue")

    ax3.hlines(0, -5, 5, color="black", linewidth=0.5)
    ax3.vlines(0, -3, 3, color="black", linewidth=0.5)

    if previous_w_raw is not None:
        ax3.arrow(
            0,
            0,
            previous_w_raw[0],
            previous_w_raw[1],
            head_width=0.2,
            head_length=0.3,
            fc="black",
            ec="black",
            linewidth=2,
        )
    ax3.arrow(
        0,
        0,
        data["weight"][0],
        data["weight"][1],
        head_width=0.2,
        head_length=0.3,
        fc="red",
        ec="red",
        linewidth=2,
    )

    ax3.set_xlim(-5, 5)
    ax3.set_ylim(-3, 3)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_title("Original space")
    ax3.grid(True, alpha=0.3)

    # Apply tight layout to each frame
    plt.tight_layout()


def kurtosis(x):
    n = x.shape[0]
    mean = np.mean(x)
    var = np.var(x)
    return np.sum((x - mean) ** 4) / (n * var**2) - 3


# %%
STEP_SIZE = 1e-3
N_ITERATIONS = 50

# Random initial weight vector
np.random.seed(0)
w1 = rng.rand(m)
w1 /= np.linalg.norm(w1) + 1e-10
w1_raw = pca.components_.T @ w1

animation_data = []

for i in range(N_ITERATIONS):
    # Project data onto weight vector
    s = np.dot(X_whitened, w1)
    k = kurtosis(s)

    # Store data for animation
    animation_data.append(
        {
            "iteration": i,
            "projection": s.copy(),
            "kurtosis": k,
            # Transform back to original space for visualization
            "weight": w1_raw,
            "weight_white": w1.copy(),
        }
    )

    # Compute the gradient
    gradient = 4 / n * np.dot(np.pow(s, 3), X_whitened)

    # Update the weight vector
    w1 += STEP_SIZE * gradient

    # Normalize the weight vector
    w1 /= np.linalg.norm(w1) + 1e-10
    w1_raw = pca.components_.T @ w1

# Create the animation
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4), dpi=300)

# Create animation
anim = animation.FuncAnimation(
    fig,
    animate,
    frames=11,
    interval=100,
    repeat=True,
    fargs=(animation_data[:11], X_whitened, "Whitened space"),
)

# Save as GIF
writer = PillowWriter(fps=5)
anim.save("img/ica_kurtosis_gd_component_1.gif", writer=writer)

plt.show()

# %%
np.random.seed(1)
w2 = rng.rand(m)
w2 /= np.linalg.norm(w2) + 1e-10
# w2 = w2 - np.dot(w2, w1) * w1  # Start orthogonal
# w2 /= np.linalg.norm(w2) + 1e-10

# Deflate the whitened data to remove first component
X_deflated = X_whitened.copy()
X_deflated = X_deflated - np.outer(np.dot(X_whitened, w1), w1)

animation_data = []

for i in range(N_ITERATIONS):
    # Project data onto weight vector
    s = np.dot(X_deflated, w2)
    k = kurtosis(s)

    # Store data for animation
    animation_data.append(
        {
            "iteration": i,
            "projection": s.copy(),
            "kurtosis": k,
            "weight": pca.components_.T @ w2,
            "weight_white": w2.copy(),
        }
    )

    # Compute the gradient
    gradient = 4 / n * np.dot(np.pow(s, 3), X_deflated)

    # Update the weight vector
    w2 += STEP_SIZE * gradient

    # Normalize the weight vector
    w2 /= np.linalg.norm(w2) + 1e-10


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4), dpi=300)
anim = animation.FuncAnimation(
    fig,
    animate,
    frames=11,
    interval=100,
    repeat=True,
    fargs=(
        animation_data[:11],
        X_deflated,
        "Whitened, deflated space",
        w1,
        w1_raw,
    ),
)

# Save as GIF
writer = PillowWriter(fps=5)
anim.save("img/ica_kurtosis_gd_component_2.gif", writer=writer)

plt.show()

# Finally, check for orthogonality
print(f"Dot product of w1 and w2 in whitened space: {np.dot(w1, w2):.4f}")
