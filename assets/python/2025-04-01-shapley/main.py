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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.decomposition import PCA

X, y = shap.datasets.diabetes()
X = X.rename(
    columns={
        "age": "Age",
        "sex": "Sex",
        "bmi": "BMI",
        "bp": "Blood pressure",
        "s1": "Total cholesterol",
        "s2": "LDL cholesterol",
        "s3": "HDL cholesterol",
        "s4": "Total cholesterol/HDL ratio",
        "s5": "Triglycerides",
        "s6": "Blood sugar",
    }
)


# %%
def save_fig(fig, basename):
    fig.savefig(f"img/{basename}.webp", bbox_inches="tight", dpi=300)
    dpi_800 = 800 / fig.get_size_inches()[0]
    fig.savefig(f"img/{basename}-800.webp", bbox_inches="tight", dpi=dpi_800)
    dpi_1600 = 1600 / fig.get_size_inches()[0]
    fig.savefig(f"img/{basename}-1600.webp", bbox_inches="tight", dpi=dpi_1600)
    plt.show()
    plt.close(fig)


# %%
df = X.copy()
df["Target"] = y

# print df in markdown table format
print(
    df[["Age", "Sex", "BMI", "Blood pressure", "Target"]]
    .head(4)
    .to_markdown(index=False)
)

# %%
# Train a model
model = sklearn.ensemble.RandomForestRegressor()
model.fit(X, y)

# %%
# Explain the model's predictions using SHAP
explainer = shap.Explainer(model)
explanation = explainer(X, check_additivity=False)

assert explanation.values.shape == X.shape

# %%
shap.plots.beeswarm(explanation, show=False)
save_fig(plt.gcf(), "beeswarm_diabetes")

# %%
shap.plots.waterfall(explanation[0, ...], show=False)
save_fig(plt.gcf(), "waterfall_diabetes")

# %%
shap.plots.bar(explanation, show=False)
save_fig(plt.gcf(), "global_diabetes")

# %%
experiment = {
    "Unsupervised": (X, "original features"),
    "Supervised": (explanation.values, "SHAP values"),
}

for clust_type, (data, features) in experiment.items():
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)
    X_pca = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
    X_pca["Diabetes Progression"] = y

    plt.figure(figsize=(10, 6))
    plt.gca().set_axis_off()
    plt.scatter(X_pca["PC1"], X_pca["PC2"], c=y, cmap="viridis", s=50, alpha=0.5)
    plt.colorbar(label="Diabetes Progression", orientation="horizontal")
    plt.title(f"PCA on the {features}", fontsize=16)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    save_fig(plt.gcf(), f"{clust_type.lower()}_pca")

# %% [markdown]
# # SHAP limitations
#
# ## No association

# %%
np.random.seed(0)

X_sim = np.random.rand(1000, 20)
y_rand = np.random.rand(1000)

X_sim_tr, X_sim_te, y_rand_tr, y_rand_te = sklearn.model_selection.train_test_split(
    X_sim, y_rand, train_size=0.9, random_state=0
)

model = sklearn.ensemble.RandomForestRegressor()
model.fit(X_sim_tr, y_rand_tr)

explainer = shap.Explainer(model)
explanation = explainer(X_sim_tr, check_additivity=False)

shap.plots.beeswarm(explanation, show=False)
save_fig(plt.gcf(), "beeswarm_random")

# plot prediction on test set
y_pred = model.predict(X_sim_te)
plt.figure(figsize=(6, 6))
# plt.gca().set_axis_off()
plt.scatter(y_rand_te, y_pred, s=50, alpha=0.5)
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.title("Random Forest predictions on test set", fontsize=16)
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.xlim(0, 1)
plt.ylim(0, 1)
save_fig(plt.gcf(), "random_forest_test")

# %%
y_intx = np.logical_xor(X_sim[:, 0] > 0.5, X_sim[:, 1] > 0.5)


model = sklearn.ensemble.RandomForestClassifier()

model.fit(X_sim, y_intx)

explainer = shap.Explainer(model)
explanation = explainer(X_sim, check_additivity=False)

shap.plots.beeswarm(explanation[1, :], show=False)
save_fig(plt.gcf(), "beeswarm_interaction")
