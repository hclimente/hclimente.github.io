from pathlib import Path
import re
from typing import List, Tuple
from urllib.parse import quote
import yaml

import numpy as np

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


def infer_post_url(path: str) -> str:
    """Infer URLs for each document based on filename."""

    base_url = "https://hclimente.eu/blog"

    stem = Path(path).stem

    # remove fixed 11-char date prefix YYYY-MM-DD-
    slug = stem[11:]
    slug = slug.strip()
    slug = quote(slug)
    url = f"{base_url}/{slug}/"

    return url


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

        m["url"] = infer_post_url(p)

        texts.append(body)
        metadatas.append(m)

    return texts, metadatas


def compute_embeddings(texts: List[str], model) -> np.ndarray:
    """Compute embeddings using a small sentence-transformers model and save.

    The actual import is done inside the function so the module can be
    syntactically validated without heavy dependencies present.
    """

    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, convert_to_numpy=True)
    print(f"Embeddings computed (shape={embeddings.shape})")

    return embeddings


def split_text(text, split_chars = ["\n\n", "\n", ". ", " "], max_size=500):
    """Recursively split text into chunks no larger than max_size."""

    if len(text) <= max_size:
        return [text]
    elif not split_chars:
        return [text[:max_size]]
    
    splitter = split_chars[0]
    splitter = splitter if isinstance(splitter, str) else "|".join(map(re.escape, splitter))
    splits = []

    for chunk in re.split(splitter, text.strip()):
        splits.extend(split_text(chunk, split_chars[1:], max_size=max_size))

    return splits

def make_overlaps(chunks, max_overlap = 0.1):
    """Add overlapping text to each chunk from its neighbors."""

    overlapping_chunks = []

    for i in range(len(chunks)):

        curr_chunk = chunks[i]
        overlap = int(max_overlap * len(curr_chunk))

        prev_chunk = "" if i == 0 else chunks[i-1]
        prev_chunk = prev_chunk[-overlap:]

        next_chunk = "" if i == (len(chunks) - 1) else chunks[i+1]
        next_chunk = next_chunk[:overlap]

        new_chunk = f"{prev_chunk}\n\n{curr_chunk}\n\n{next_chunk}"
        new_chunk = new_chunk.strip()

        overlapping_chunks.append(new_chunk)

    return overlapping_chunks


def compute_similarity_matrix(X):
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    return np.dot(X_norm, X_norm.T)