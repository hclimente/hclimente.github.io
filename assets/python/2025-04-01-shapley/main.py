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
import shap
import sklearn

X, y = shap.datasets.diabetes()
X = X.rename(columns={
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
})


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
shap.plots.beeswarm(explanation)

# %%
shap.plots.waterfall(explanation[0, ...])
