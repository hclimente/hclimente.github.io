# -*- coding: utf-8 -*-
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
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers.pipelines.pt_utils import KeyDataset

from pipelines import DNAEmbeddingPipeline, DNAClassificationPipeline
from project_utils import (
    compute_accuracy,
    create_dataset,
    compute_trainer_metrics,
    plot_umap,
    plot_confusion_matrix,
)

sys.path.append("../")

from utils import save_fig

# configuration
MODEL = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
DATASETS = [
    ("Arabidopsis_thaliana", "A. thaliana"),
    ("Drosophila_melanogaster", "D. melanogaster"),
    ("Escherichia_coli_gca_001606525", "E. coli"),
    ("Homo_sapiens", "H. sapiens"),
    ("Mus_musculus", "M. musculus"),
    ("Saccharomyces_cerevisiae", "S. cerevisiae"),
]

# %%
# trust_remote_code is needed since the model is not native to the library
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(MODEL, trust_remote_code=True)

pipeline = DNAEmbeddingPipeline(model=model, tokenizer=tokenizer)

train_ds = create_dataset("data/train/", DATASETS)
test_ds = create_dataset("data/test/", DATASETS)
# %%
train_emb = []

for e in pipeline(KeyDataset(train_ds, "text"), max_length=30):
    train_emb.append(e)

train_emb = np.concatenate(train_emb, axis=0)

labels = [label for _, label in DATASETS for _ in range(1000)]
fig = plot_umap(train_emb, labels)
save_fig(fig, "umap_embeddings")

# %% [markdown]
# # Predict the species from the embedding

# %%
model = LogisticRegression()
model.fit(train_emb, labels)

tr_pred = model.predict(train_emb)
accuracy = compute_accuracy(labels, tr_pred)
print(f"Train Accuracy: {accuracy:.2f}")

fig = plot_confusion_matrix(labels, tr_pred)

# %%
test_emb = []

for e in pipeline(KeyDataset(test_ds, "text"), max_length=30):
    test_emb.append(e)

test_emb = np.concatenate(test_emb, axis=0)

te_pred = model.predict(test_emb)
accuracy = compute_accuracy(labels, te_pred)
print(f"Test Accuracy: {accuracy:.2f}")
fig = plot_confusion_matrix(labels, te_pred)
save_fig(fig, "confusion_matrix_test")

random_accuracy = compute_accuracy(labels, np.random.permutation(labels))
print(f"(Random Accuracy: {random_accuracy:.2f})")

# %% [markdown]
# # Fine-tune the model on the training set
#
# ## Load model

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=6, trust_remote_code=True
)


# %% [markdown]
# ## Prepare data


# %%
def preprocess(ds):

    def tokenize_batch(batch):
        batch = tokenizer(
            batch["text"],
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return batch

    tokenized_ds = ds.map(tokenize_batch, batched=True, remove_columns=["text"])

    return tokenized_ds


tokenized_train_ds = preprocess(train_ds)
split = tokenized_train_ds.train_test_split(test_size=0.1, seed=42)

# train - validation split
tr_train_ds = split["train"]
tr_val_ds = split["test"]

tokenized_test_ds = preprocess(test_ds)

# %% [markdown]
# ## Prepare trainer

# %%
# Freeze the bottom N transformer layers to speed up / regularize
for param in model.base_model.embeddings.parameters():
    param.requires_grad = False

for layer_idx, layer_module in enumerate(model.base_model.encoder.layer):
    if layer_idx < 10:
        for param in layer_module.parameters():
            param.requires_grad = False

# Set up Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    dataloader_pin_memory=False,  # not supported by mps
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tr_train_ds,
    eval_dataset=tr_val_ds,
    data_collator=data_collator,
    compute_metrics=compute_trainer_metrics,
)


# %% [markdown]
# ## Train and evaluate

# %%
perf = trainer.evaluate(tokenized_test_ds)
print(f"Test accuracy before training: {perf['eval_accuracy']:.2f}")

trainer.train()

perf = trainer.evaluate(tokenized_test_ds)
print(f"Test accuracy after training: {perf['eval_accuracy']:.2f}")


# %% [markdown]
# ## Downstream tasks

# %%
emb_pipe = DNAClassificationPipeline(model=trainer.model, tokenizer=tokenizer)

test_emb_ft = []
test_pred_ft = []

for e in emb_pipe(KeyDataset(test_ds, "text"), max_length=30):
    test_emb_ft.append(e["embedding"])
    test_pred_ft.append(e["logits"])

test_emb_ft = np.concatenate(test_emb_ft, axis=0)
test_pred_ft = np.concatenate(test_pred_ft, axis=0)

# %%
fig = plot_umap(test_emb_ft, labels)
save_fig(fig, "umap_embeddings_ft-model")

# %%
accuracy = compute_accuracy(test_ds["labels"], test_pred_ft.argmax(1))
print(f"Test Accuracy: {accuracy:.2f}")
