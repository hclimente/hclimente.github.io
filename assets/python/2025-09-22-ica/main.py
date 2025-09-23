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
import numpy as np
np.random.seed(0) # for reproducibility

# %%
# Create 3 independent, non-Gaussian source signals
num_samples = 500
S1 = np.random.uniform(-1, 1, num_samples) # Uniform distribution
S2 = np.random.laplace(0, 1, num_samples) # Laplace distribution
S3 = np.random.beta(a=2, b=5, size=num_samples) - 0.5 # Beta distribution

# %%
# Create a random mixing matrix
num_sources = 3
num_genes = 3
A = np.random.rand(num_genes, num_sources)

# %%
# Stack sources into a matrix (rows are sources, columns are samples)
S_matrix = np.vstack((S1, S2, S3))

# %%
# Generate the observed data
X_matrix = A @ S_matrix

# %%
# Add some noise to make it more realistic
noise = np.random.normal(0, 0.1, X_matrix.shape)
X_matrix_noisy = X_matrix + noise

# %%
import numpy as np
from sklearn.decomposition import FastICA

# %%
# Assuming X_matrix_noisy from the last example is your data
# The data needs to have shape (n_samples, n_features) for scikit-learn
# Here, n_samples = number of genes, n_features = number of samples
X_data_ica = X_matrix_noisy.T

# %%
# Center the data
X_centered = X_data_ica - X_data_ica.mean(axis=0)

# %%
# Create and fit the ICA model
ica = FastICA(n_components=3, random_state=0)
S_recovered = ica.fit_transform(X_centered) # Recover the independent components

# %%
# Recover the mixing matrix
A_recovered = ica.mixing_ # This attribute holds the recovered mixing matrix

# %%
(S_recovered - S_matrix.T) / S_matrix.std(axis = 1)

# %%

# %%
