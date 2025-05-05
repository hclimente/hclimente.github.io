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

from datasets import load_dataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.pipelines.pt_utils import KeyDataset

from pipelines import DNAEmbeddingPipeline
from project_utils import plot_umap, plot_confusion_matrix, compute_accuracy

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

# %%
tr_embeddings = []
labels = [label for _, label in DATASETS for _ in range(1000)]

for filename, label in DATASETS:
    print(f"Processing {label}...")
    tr_data = load_dataset(
        "text", data_files=f"data/train/{filename}.txt", split="train"
    )

    for e in pipeline(KeyDataset(tr_data, "text"), max_length=30):
        tr_embeddings.append(e)

tr_embeddings = np.concatenate(tr_embeddings, axis=0)

# %%
fig = plot_umap(tr_embeddings, labels)
save_fig(fig, "umap_embeddings")

# %% [markdown]
# # Predict the species from the embedding

# %%
model = LogisticRegression()
model.fit(tr_embeddings, labels)

tr_pred = model.predict(tr_embeddings)
accuracy = compute_accuracy(labels, tr_pred)
print(f"Train Accuracy: {accuracy:.2f}")

fig = plot_confusion_matrix(labels, tr_pred)

# %%
te_embeddings = []

for filename, label in DATASETS:
    print(f"Processing {label}...")
    tr_data = load_dataset(
        "text", data_files=f"data/test/{filename}.txt", split="train"
    )

    for e in pipeline(KeyDataset(tr_data, "text"), max_length=30):
        te_embeddings.append(e)

te_embeddings = np.concatenate(te_embeddings, axis=0)

te_pred = model.predict(te_embeddings)
accuracy = compute_accuracy(labels, te_pred)
print(f"Test Accuracy: {accuracy:.2f}")
fig = plot_confusion_matrix(labels, te_pred)
save_fig(fig, "confusion_matrix_test")

random_accuracy = compute_accuracy(labels, np.random.permutation(labels))
print(f"(Random Accuracy: {random_accuracy:.2f})")

# %% [markdown]
# # Fine-tune the model on the training set

# %%

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
