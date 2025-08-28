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
#     display_name: 2025-08-16-rags
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from sentence_transformers import SentenceTransformer
from umap import UMAP

from embedding_utils import (
    collect_markdowns,
    compute_embeddings,
    compute_similarity_matrix,
    extract_frontmatter_and_body,
    infer_post_url,
    load_documents,
    make_overlaps,
    split_text,
)

sys.path.append("../")

from utils import (
    PLOTLY_AXIS_ATTR_DICT,
    PLOTLY_LEGEND_ATTR_DICT,
    save_fig,
    save_plotly,
)

MD_PATH = Path("../../../_posts")
MAX_CHUNK_SIZE = 600
MIN_CHUNK_SIZE = 100
MODEL_NAME = "all-MiniLM-L6-v2" # Small, high-quality model that is lightweight on CPU
SPLIT_CHARS = ["\n\n", "\n", [".", "!", "?"], [";", ","]]

print(f"Loading model '{MODEL_NAME}' (this will download the model if needed)...")
model = SentenceTransformer(MODEL_NAME)

# %% [markdown]
# # Whole document embedding

# %%
print("Scanning for markdown files in:", MD_PATH)
files = collect_markdowns(MD_PATH)
print(f"Found {len(files)} markdown files")
texts, metadata = load_documents(files)

embeddings = compute_embeddings(texts, model)

# %%
# Ensure embeddings array is a numeric 2D numpy array
embeddings = np.asarray(embeddings)
if embeddings.ndim == 1:
    try:
        embeddings = np.stack(embeddings)
    except Exception:
        raise RuntimeError(
            "Embeddings appear to be a 1-D object array and could not be stacked into a 2-D array."
        )

print(f"Embeddings shape: {embeddings.shape}")

print("Computing 2D UMAP...")
umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
emb2 = umap.fit_transform(embeddings)

# Build hover texts from metadata (ensure length matches emb2)
hover_texts = []
for i, m in enumerate(metadata):
    title = m.get("title") if isinstance(m, dict) else None
    if not title:
        title = Path(m.get("path", "")).name if isinstance(m, dict) else f"doc-{i}"
    date = m.get("date", "") if isinstance(m, dict) else ""
    tags = m.get("tags", "") if isinstance(m, dict) else ""
    if isinstance(tags, list):
        tags = ", ".join(tags)
    hover = f"{title}"
    hover_texts.append(hover)

if len(hover_texts) != emb2.shape[0]:
    hover_texts = [f"doc-{i}" for i in range(emb2.shape[0])]

# Prepare category mappings (simplified)
categories = ["coding", "machine_learning", "genetics", "graphs"]
# Normalize tags -> set of lowercased tokens per document
tags_list = []
for m in metadata:
    if isinstance(m, dict):
        t = m.get("tags", [])
        if isinstance(t, str):
            # split on commas or whitespace
            if "," in t:
                toks = [s.strip() for s in t.split(",") if s.strip()]
            else:
                toks = [s.strip() for s in t.split() if s.strip()]
        elif isinstance(t, list):
            toks = [str(x).strip() for x in t]
        else:
            toks = []
        toks = [x.replace("-", "_").lower() for x in toks]
        tags_list.append(set(toks))
    else:
        tags_list.append(set())

# Determine primary category per document (first matching in categories order)
primary = []
for tags in tags_list:
    p = None
    for c in categories:
        if c in tags:
            p = c
            break
    primary.append(p or "other")

# Color mapping for categories (add 'other')
cat_colors = {
    "coding": "#636EFA",
    "machine_learning": "#00CC96",
    "genetics": "#EF553B",
    "graphs": "#FFA15A",
    "other": "#bebebe",
}

urls = [m["url"] for m in metadata]

# Build 2D traces, one per category, with customdata for click URLs
traces = []
order = categories + ["other"]
for lab in order:
    idx = [i for i, p in enumerate(primary) if p == lab]
    if idx:
        trace_urls = [urls[i] for i in idx]
        traces.append(
            go.Scatter(
                x=emb2[idx, 0],
                y=emb2[idx, 1],
                mode="markers",
                marker=dict(size=8, color=cat_colors.get(lab, "#888"), opacity=0.9),
                text=[hover_texts[i] for i in idx],
                hoverinfo="text",
                name=f"{lab}",
                customdata=trace_urls,
            )
        )
    else:
        traces.append(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(size=8),
                name=f"{lab}",
                visible=False,
                customdata=[],
            )
        )

fig = go.Figure(data=traces)

# Add JS that opens the customdata URL on click
div_id = "umap_plot"
post_script = f"""
var gd = document.getElementById('{div_id}');
if(gd) {{
  gd.on('plotly_click', function(data) {{
    var pt = data.points[0];
    var url = pt.customdata;
    if(url) {{ window.open(url, '_blank'); }}
  }});
}}
"""

xaxis_attr_dict = PLOTLY_AXIS_ATTR_DICT.copy()
xaxis_attr_dict["title"] = "UMAP 1"
yaxis_attr_dict = PLOTLY_AXIS_ATTR_DICT.copy()
yaxis_attr_dict["title"] = "UMAP 2"

save_plotly(
    fig, "posts_umap", xaxis_attr_dict, yaxis_attr_dict, PLOTLY_LEGEND_ATTR_DICT, div_id=div_id, post_script=post_script
)

# %% [markdown]
# # Chunked embedding

# %%
embeddings = []
titles = []
chunk_list = []
for txt, meta in zip(texts, metadata):
    chunks = split_text(txt, split_chars=SPLIT_CHARS, max_size=MAX_CHUNK_SIZE)
    chunks = make_overlaps(chunks)
    # Remove short chunks
    chunks = [x for x in chunks if len(x) > MIN_CHUNK_SIZE]
    chunk_list.extend(chunks)

    title = [meta["title"]] * len(chunks)
    titles.extend(title)

embeddings = compute_embeddings(chunk_list, model)
embeddings = np.array(embeddings)
print(f"Chunked embeddings shape: {embeddings.shape}")

# %%
chunks[9].split('\n\n')[1]

# %%
umap = UMAP(n_components=2, random_state=42, n_jobs=1)
emb_umap = umap.fit_transform(embeddings)

# Prepare hover text: title + chunk text (truncate for display)
hover_texts = [f"<b>{title}</b><br>{chunk.split('\n\n')[1][:200]}..." for title, chunk in zip(titles, chunk_list)]

# Factorize titles for color assignment
unique_titles, title_ids = np.unique(titles, return_inverse=True)
colors = [f"hsl({(i*360)//len(unique_titles)},70%,50%)" for i in title_ids]

fig = go.Figure(
    data=[go.Scatter(
        x=emb_umap[:, 0],
        y=emb_umap[:, 1],
        mode="markers",
        marker=dict(
            color=colors,
            size=8,
            opacity=0.7,
        ),
        text=hover_texts,
        hoverinfo="text",
    )]
)

save_plotly(
    fig,
    "paragraphs_umap",
    xaxis_attr_dict,
    yaxis_attr_dict,
    PLOTLY_LEGEND_ATTR_DICT,
    div_id=div_id,
    post_script=post_script
)

# %% [markdown]
# # Cosine similarity between the chunks

# %%
# compute cosine similarities matrix
sim = compute_similarity_matrix(embeddings)

# hierarchically cluster the embeddings
Z = linkage(embeddings, method="ward")
leaf_order = leaves_list(Z)
sim_ordered = sim[leaf_order][:, leaf_order]

# Create unique chunk labels instead of just article titles
chunk_labels = []
title_counts = {}
for i in leaf_order:
    title = titles[i]
    if title not in title_counts:
        title_counts[title] = 0
    title_counts[title] += 1
    chunk_labels.append(f"{title} (chunk {title_counts[title]})")

# round to 3 decimals to shrink JSON payload
sim_ordered = np.round(sim_ordered, 3)

n = sim_ordered.shape[0]

heatmap = go.Heatmap(
    z=sim_ordered,
    x=chunk_labels,  # unique labels per chunk
    y=chunk_labels,  # unique labels per chunk
    zmin=-1, zmax=1,
    colorscale="RdBu",
    reversescale=True,
    colorbar=dict(title="Cosine similarity"),
    hovertemplate="x: %{x}<br>y: %{y}<br>similarity: %{z:.3f}<extra></extra>",
)

fig = go.Figure(data=[heatmap])

xaxis_attr = PLOTLY_AXIS_ATTR_DICT.copy()
xaxis_attr.update(dict(title="", showticklabels=False, ticks=""))

yaxis_attr = PLOTLY_AXIS_ATTR_DICT.copy()
yaxis_attr.update(dict(title="", showticklabels=False, ticks="", autorange="reversed"))

save_plotly(
    fig,
    "paragraph_similarity_heatmap",
    xaxis_attr,
    yaxis_attr,
    PLOTLY_LEGEND_ATTR_DICT,
    div_id="sim_heatmap",
)

# %% [markdown]
# # Query

# %%
QUERY = """
Interpretable machine learning
""".strip()
query_embedding = compute_embeddings([QUERY], model)

df = pd.DataFrame({
    "title": ["query"] + titles,
    "text": [QUERY] + [f"{chunk.split('\n\n')[1][:200]}..." for chunk in chunk_list],
    "similarity": compute_similarity_matrix(np.vstack([query_embedding, embeddings]))[0, :],
})

df.sort_values("similarity", ascending=False).head(10).reset_index(drop=True)

# %%
