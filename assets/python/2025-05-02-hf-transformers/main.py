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
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.pipelines.pt_utils import KeyDataset

from pipelines import DNAEmbeddingPipeline

import pandas as pd

# %%
tr_data = load_dataset("text", data_files="data/train/Mus_musculus.txt", split="train")

tokenizer = AutoTokenizer.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True
)
model = AutoModelForMaskedLM.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True
)

pipeline = DNAEmbeddingPipeline(model=model, tokenizer=tokenizer)

# %%
embeddings = []
for e in pipeline(KeyDataset(tr_data, "text"), max_length=10):
    embeddings.append(e)

embeddings = np.concatenate(embeddings, axis=0)

# %%
variants = pd.read_csv("41586_2018_461_MOESM3_ESM.tsv", sep="\t")

variants = variants[["chromosome", "position (hg19)", "reference", "alt"]]
variants = variants.rename(
    columns={
        "chromosome": "chrom",
        "position (hg19)": "pos",
        "reference": "ref",
        "alt": "alt",
    }
)
variants["chrom"] = variants["chrom"].apply(lambda x: "chr" + str(x))

# %%
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
