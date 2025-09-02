#!/usr/bin/env python3
"""
RAG CLI - Command Line Interface for the RAG system

Usage:
    python rag_cli.py "Your question here"
    python rag_cli.py --interactive  # Start chat mode
    python rag_cli.py --model qwen2-1.5b "Your question"  # Use specific model
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from rag_system import create_rag_system, SMALL_LLM_OPTIONS
from embedding_utils import collect_markdowns, load_documents, compute_embeddings, split_text, make_overlaps
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np


def setup_rag_system(model_choice="qwen2-1.5b", data_path="../../../_posts"):
    """Set up the complete RAG system."""
    
    # Configuration
    MD_PATH = Path(data_path)
    COLLECTION_NAME = "blog_chunks"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_CHUNK_SIZE = 600
    MIN_CHUNK_SIZE = 100
    SPLIT_CHARS = ["\n\n", "\n", [". ", "! ", "? "], "; ", ", ", " "]
    
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = TextEmbedding(EMBEDDING_MODEL_NAME)
    
    # Load and process documents
    print(f"Loading documents from: {MD_PATH}")
    files = collect_markdowns(MD_PATH)
    
    if not files:
        print(f"No markdown files found in {MD_PATH}")
        return None
        
    texts, metadata = load_documents(files)
    print(f"Loaded {len(texts)} documents")
    
    # Create chunks
    chunk_list = []
    chunk_metadata = []
    
    for txt, meta in zip(texts, metadata):
        chunks = split_text(txt, split_chars=SPLIT_CHARS, max_size=MAX_CHUNK_SIZE)
        chunks = make_overlaps(chunks)
        chunks = [x for x in chunks if len(x) > MIN_CHUNK_SIZE]
        
        chunk_list.extend(chunks)
        chunk_metadata.extend([meta] * len(chunks))
    
    print(f"Created {len(chunk_list)} chunks")
    
    # Compute embeddings
    embeddings = compute_embeddings(chunk_list, embedding_model)
    embeddings = np.array(embeddings)
    
    # Set up Qdrant
    client = QdrantClient(":memory:")
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embeddings.shape[1],
            distance=Distance.COSINE,
        ),
    )
    
    # Insert points
    points = []
    for i, (embedding, chunk, meta) in enumerate(zip(embeddings, chunk_list, chunk_metadata)):
        chunk_preview = chunk[:300] if len(chunk) > 300 else chunk
        
        point = PointStruct(
            id=i,
            vector=embedding,
            payload={
                "title": meta.get("title", f"Document {i}"),
                "chunk_text": chunk,
                "chunk_preview": chunk_preview,
                "chunk_index": i,
            },
        )
        points.append(point)
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Vector database ready with {len(points)} chunks")
    
    # Create RAG system
    rag = create_rag_system(
        qdrant_client=client,
        collection_name=COLLECTION_NAME,
        embedding_model=embedding_model,
        model_choice=model_choice
    )
    
    return rag


def main():
    parser = argparse.ArgumentParser(description="RAG CLI - Ask questions about your blog posts")
    parser.add_argument("query", nargs="?", help="Question to ask the RAG system")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--model", "-m", choices=list(SMALL_LLM_OPTIONS.keys()), 
                       default="qwen2-1.5b", help="Choose the LLM model")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="Number of context chunks to retrieve")
    parser.add_argument("--max-length", "-l", type=int, default=300, help="Max length of generated response")
    parser.add_argument("--data-path", "-d", default="../../../_posts", help="Path to markdown files")
    
    args = parser.parse_args()
    
    if not args.query and not args.interactive:
        parser.print_help()
        return
    
    print(f"Setting up RAG system with {args.model} model...")
    rag = setup_rag_system(model_choice=args.model, data_path=args.data_path)
    
    if rag is None:
        print("Failed to set up RAG system")
        return
    
    if args.interactive:
        rag.chat()
    else:
        result = rag.ask(args.query, top_k=args.top_k, max_length=args.max_length)
        
        print("\n" + "="*60)
        print(f"Query: {result['query']}")
        print("-" * 60)
        print(f"Answer: {result['answer']}")
        print("-" * 60)
        print(f"Sources: {', '.join(result['sources'])}")
        print("="*60)


if __name__ == "__main__":
    main()
