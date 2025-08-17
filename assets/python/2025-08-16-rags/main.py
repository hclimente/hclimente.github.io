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
from typing import List, Tuple
import yaml

import numpy as np
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from umap import UMAP

MD_PATH = Path("../../../_posts")


def extract_frontmatter_and_body(text: str) -> Tuple[dict, str]:
    """Return (metadata dict, body text).

    Frontmatter is expected to be the first section between two lines that
    contain only '---'. If no frontmatter is present, returns ({}, original).
    """
    # Quick check
    if not text.lstrip().startswith("---"):
        return {}, text

    # Split at the first two occurrences of '---'
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    # parts[1] is the YAML-ish header, parts[2] is the rest
    header = parts[1].strip()
    body = parts[2].strip()

    # Parse header using a local, optional import to keep startup light
    meta = yaml.safe_load(header) or {}

    # Normalize tags if present (possible space-separated string)
    if "tags" in meta and isinstance(meta["tags"], str):
        meta["tags"] = [t for t in meta["tags"].split() if t]

    return meta, body


def collect_markdowns(root: Path) -> List[Path]:
    """Find all markdown files under `root` (recursively)."""
    patterns = ("**/*.md", "**/*.markdown")
    files = []
    for p in patterns:
        files.extend(list(root.glob(p)))
    # Deduplicate and sort for stable ordering
    files = sorted(set(files))
    return files


def load_documents(paths: List[Path]) -> Tuple[List[str], List[dict]]:
    """Load files, extract frontmatter and return (texts, metadata_list).

    Each metadata dict will include at least: 'path' and any parsed header keys.
    The returned texts are the document bodies (without the frontmatter).
    """
    texts = []
    metadatas = []
    for p in paths:
        try:
            raw = p.read_text(encoding="utf-8")
        except Exception:
            # Skip unreadable files
            continue
        meta, body = extract_frontmatter_and_body(raw)
        # ensure JSON serializable basic metadata
        m = {k: v for k, v in (meta or {}).items()}
        m.setdefault("path", str(p))
        # Convert possible non-serializable values
        if "date" in m:
            m["date"] = str(m["date"])
        if "tags" in m and not isinstance(m["tags"], list):
            # try splitting on whitespace or commas
            if isinstance(m["tags"], str):
                if "," in m["tags"]:
                    m["tags"] = [t.strip() for t in m["tags"].split(",") if t.strip()]
                else:
                    m["tags"] = [t for t in m["tags"].split() if t]

        texts.append(body)
        metadatas.append(m)

    return texts, metadatas


def compute_embeddings(texts: List[str], metadatas: List[dict]):
    """Compute embeddings using a small sentence-transformers model and save.

    The actual import is done inside the function so the module can be
    syntactically validated without heavy dependencies present.
    """
    if not texts:
        print("No documents found. Nothing to embed.")
        return

    # Small, high-quality model that is lightweight on CPU
    model_name = "all-MiniLM-L6-v2"
    print(f"Loading model '{model_name}' (this will download the model if needed)...")
    model = SentenceTransformer(model_name)

    print(f"Computing embeddings for {len(texts)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"Embeddings computed (shape={embeddings.shape})")

    return embeddings


print("Scanning for markdown files in:", MD_PATH)
files = collect_markdowns(MD_PATH)
print(f"Found {len(files)} markdown files")
texts, metadata = load_documents(files)
embeddings = compute_embeddings(texts, metadata)

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

# Build 2D traces, one per category
traces = []
order = categories + ["other"]
for lab in order:
    idx = [i for i, p in enumerate(primary) if p == lab]
    if idx:
        traces.append(
            go.Scatter(
                x=emb2[idx, 0],
                y=emb2[idx, 1],
                mode="markers",
                marker=dict(size=8, color=cat_colors.get(lab, "#888"), opacity=0.9),
                text=[hover_texts[i] for i in idx],
                hoverinfo="text",
                name=f"{lab}",
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
            )
        )

fig = go.Figure(data=traces)

fig.update_layout(
    height=720,
    template="simple_white",
    xaxis=dict(title="UMAP 1", gridcolor="lightgrey"),
    yaxis=dict(title="UMAP 2", gridcolor="lightgrey"),
    legend=dict(
        itemsizing="constant",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
)

fig.write_html("img/posts_umap.html", include_plotlyjs="cdn")
