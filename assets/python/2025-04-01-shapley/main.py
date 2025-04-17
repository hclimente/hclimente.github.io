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
plt.savefig("img/beeswarm_diabetes.jpg", bbox_inches="tight", dpi=300)
plt.show()

# %%
shap.plots.waterfall(explanation[0, ...], show=False)
plt.savefig("img/waterfall_diabetes.jpg", bbox_inches="tight", dpi=300)
plt.show()

# %%
shap.plots.bar(explanation, show=False)
plt.savefig("img/global_diabetes.jpg", bbox_inches="tight", dpi=300)
plt.show()

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
    plt.savefig(f"img/{clust_type.lower()}_pca.jpg", bbox_inches="tight", dpi=300)
    plt.show()
