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
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from pipelines import DNAEmbeddingPipeline, DNAClassificationPipeline
from project_utils import (
    compute_accuracy,
    create_dataset,
    compute_trainer_metrics,
    extract_embeddings_from_pipeline,
    plot_umap,
    plot_confusion_matrix,
)

sys.path.append("../")

from utils import save_fig

# global parameters
MAX_LENGTH = 15
MODEL = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
DATASETS = [
    ("Arabidopsis_thaliana", "A. thaliana"),
    ("Drosophila_melanogaster", "D. melanogaster"),
    ("Escherichia_coli_gca_001606525", "E. coli"),
    ("Homo_sapiens", "H. sapiens"),
    ("Mus_musculus", "M. musculus"),
    ("Saccharomyces_cerevisiae", "S. cerevisiae"),
]
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# fine-tuning parameters
BATCH_SIZE = 64
N_EPOCHS = 10
N_LAYERS_TO_FREEZE_FINETUNE = 10
VALIDATION_SPLIT_SIZE = 0.1

# %%
# trust_remote_code is needed since the model is not native to the library
tokenizer_nt = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
embedder_nt = AutoModelForMaskedLM.from_pretrained(MODEL, trust_remote_code=True)

pipe = DNAEmbeddingPipeline(model=embedder_nt, tokenizer=tokenizer_nt)

train_ds = create_dataset("data/train/", DATASETS)
tr_labels = [DATASETS[x][1] for x in train_ds["labels"]]

test_ds = create_dataset("data/test/", DATASETS)
te_labels = [DATASETS[x][1] for x in test_ds["labels"]]
# %%
train_emb, _ = extract_embeddings_from_pipeline(pipe, train_ds, max_length=MAX_LENGTH)

fig = plot_umap(train_emb, tr_labels, seed=SEED)
save_fig(fig, "umap_embeddings")

# %% [markdown]
# # Predict the species from the embedding

# %%
lr = LogisticRegression(random_state=SEED)
lr.fit(train_emb, tr_labels)

tr_pred = lr.predict(train_emb)
accuracy = compute_accuracy(tr_labels, tr_pred)
print(f"Train Accuracy: {accuracy:.2f}")

fig = plot_confusion_matrix(tr_labels, tr_pred)

# %%
test_emb, _ = extract_embeddings_from_pipeline(pipe, test_ds, max_length=MAX_LENGTH)

te_pred = lr.predict(test_emb)
accuracy = compute_accuracy(te_labels, te_pred)
print(f"Test Accuracy: {accuracy:.2f}")
fig = plot_confusion_matrix(te_labels, te_pred)
save_fig(fig, "confusion_matrix_test")

random_accuracy = compute_accuracy(te_labels, np.random.permutation(te_labels))
print(f"(Random Accuracy: {random_accuracy:.2f})")

# %% [markdown]
# # Fine-tune the model on the training set
#
# ## Load model

# %%
classif_nt = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(DATASETS), trust_remote_code=True
)


# %% [markdown]
# ## Prepare data


# %%
def preprocess(ds):

    def tokenize_batch(batch):
        batch = tokenizer_nt(
            batch["text"],
            padding="longest",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        return batch

    tokenized_ds = ds.map(tokenize_batch, batched=True, remove_columns=["text"])

    return tokenized_ds


tokenized_train_ds = preprocess(train_ds)
split = tokenized_train_ds.train_test_split(test_size=VALIDATION_SPLIT_SIZE, seed=SEED)

# train - validation split
tr_train_ds = split["train"]
tr_val_ds = split["test"]

tokenized_test_ds = preprocess(test_ds)

# %% [markdown]
# ## Prepare trainer

# %%
# Freeze the bottom N transformer layers to speed up / regularize
for param in classif_nt.base_model.embeddings.parameters():
    param.requires_grad = False

for layer_idx, layer_module in enumerate(classif_nt.base_model.encoder.layer):
    if layer_idx < N_LAYERS_TO_FREEZE_FINETUNE:
        for param in layer_module.parameters():
            param.requires_grad = False

# Set up Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=N_EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    seed=SEED,
    dataloader_pin_memory=False,  # not supported by mps
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer_nt)

trainer = Trainer(
    model=classif_nt,
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
pipe = DNAClassificationPipeline(model=trainer.model, tokenizer=tokenizer_nt)

test_emb_ft, test_pred_ft = extract_embeddings_from_pipeline(
    pipe, test_ds, max_length=MAX_LENGTH
)

# %%
fig = plot_umap(test_emb_ft, te_labels, seed=SEED)
save_fig(fig, "umap_embeddings_ft-model")

# %%
accuracy = compute_accuracy(test_ds["labels"], test_pred_ft.argmax(1))
print(f"Test Accuracy: {accuracy:.2f}")
