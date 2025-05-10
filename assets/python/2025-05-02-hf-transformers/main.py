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
    plot_umap,
    plot_confusion_matrix,
    compute_accuracy,
    create_dataset,
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

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=6, trust_remote_code=True
)


# %%
def preprocess(ds):

    def tokenize_batch(batch):
        batch = tokenizer(
            batch["text"],
            truncation=True,
            padding="longest",
            max_length=30,
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

# %%
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 3. (Optional) Freeze the bottom N transformer layers to speed up / regularize
for param in model.base_model.embeddings.parameters():
    param.requires_grad = False

for layer_idx, layer_module in enumerate(model.base_model.encoder.layer):
    if layer_idx < 10:
        for param in layer_module.parameters():
            param.requires_grad = False


# 4. Set up Trainer
training_args = TrainingArguments(
    output_dir="finetune_out",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    learning_rate=5e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    dataloader_pin_memory=False,  # not supported by mps
)


def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": (preds == p.label_ids).mean()}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tr_train_ds,
    eval_dataset=tr_val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 5. Train & evaluate
perf = trainer.evaluate(tokenized_test_ds)
print(f"Test accuracy before training: {perf['eval_accuracy']:.2f}")

trainer.train()

perf = trainer.evaluate(tokenized_test_ds)
print(f"Test accuracy after training: {perf['eval_accuracy']:.2f}")


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

# %%
accuracy = compute_accuracy(test_ds["labels"], test_pred_ft.argmax(1))
print(f"Test Accuracy: {accuracy:.2f}")

# %%
# variants = pd.read_csv("41586_2018_461_MOESM3_ESM.tsv", sep="\t")

# variants = variants[["chromosome", "position (hg19)", "reference", "alt"]]
# variants = variants.rename(
#     columns={
#         "chromosome": "chrom",
#         "position (hg19)": "pos",
#         "reference": "ref",
#         "alt": "alt",
#     }
# )
# variants["chrom"] = variants["chrom"].apply(lambda x: "chr" + str(x))

# sequences = []

# for v in variants.itertuples(index=False):
#     ref_seq, alt_seq = variant.variant_to_seqs(
#         seq_len=max_length,
#         genome="hg19",
#         chrom=v.chrom,
#         pos=v.pos,
#         ref=v.ref,
#         alt=v.alt,
#     )

#     sequences.append(ref_seq)
#     sequences.append(alt_seq)

# # Compute the embeddings
# attention_mask = tokens_ids != tokenizer.pad_token_id
# torch_outs = model(
#     tokens_ids,
#     attention_mask=attention_mask,
#     encoder_attention_mask=attention_mask,
#     output_hidden_states=True,
# )

# # Compute sequences embeddings
# embeddings = torch_outs["hidden_states"][-1].detach().numpy()
# print(f"Embeddings shape: {embeddings.shape}")
# print(f"Embeddings per token: {embeddings}")

# # Add embed dimension axis
# attention_mask = torch.unsqueeze(attention_mask, dim=-1)

# # Compute mean embeddings per sequence
# mean_sequence_embeddings = torch.sum(attention_mask * embeddings, axis=-2) / torch.sum(
#     attention_mask, axis=1
# )
# print(f"Mean sequence embeddings: {mean_sequence_embeddings}")
