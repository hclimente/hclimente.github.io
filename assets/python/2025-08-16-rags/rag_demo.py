#!/usr/bin/env python3
"""
RAG Demo Script

A simple demonstration of the RAG system that can be run independently.
This script shows how to use the RAG system for question answering.
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from rag_system import create_rag_system, SMALL_LLM_OPTIONS
from embedding_utils import collect_markdowns, load_documents, compute_embeddings
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np


def setup_rag_demo():
    """Set up the RAG system with minimal data for demonstration."""

    # Configuration
    MD_PATH = Path("../../../_posts")
    COLLECTION_NAME = "blog_chunks_demo"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    print("Setting up RAG demo...")

    # Load embedding model
    print(f"Loading embedding model: {MODEL_NAME}")
    embedding_model = TextEmbedding(MODEL_NAME)

    # Load documents (limited for demo)
    print(f"Scanning for markdown files in: {MD_PATH}")
    files = collect_markdowns(MD_PATH)[:5]  # Limit to first 5 files for demo
    print(f"Using {len(files)} markdown files for demo")

    if not files:
        print("No markdown files found! Make sure the path is correct.")
        return None

    texts, metadata = load_documents(files)

    # Create simple chunks (no overlap for demo)
    chunks = []
    chunk_metadata = []

    for text, meta in zip(texts, metadata):
        # Simple splitting by paragraphs
        paragraphs = [
            p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 100
        ]

        for para in paragraphs[:3]:  # Max 3 chunks per document for demo
            chunks.append(para)
            chunk_metadata.append(meta)

    print(f"Created {len(chunks)} chunks from {len(files)} documents")

    # Compute embeddings
    embeddings = compute_embeddings(chunks, embedding_model)
    embeddings = np.array(embeddings)

    # Set up Qdrant
    client = QdrantClient(":memory:")

    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embeddings.shape[1],
            distance=Distance.COSINE,
        ),
    )

    # Insert data
    points = []
    for i, (embedding, chunk, meta) in enumerate(
        zip(embeddings, chunks, chunk_metadata)
    ):
        point = PointStruct(
            id=i,
            vector=embedding,
            payload={
                "title": meta.get("title", f"Document {i}"),
                "chunk_text": chunk,
                "chunk_preview": chunk[:200],
                "chunk_index": i,
            },
        )
        points.append(point)

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Inserted {len(points)} chunks into Qdrant")

    # Create RAG system
    rag = create_rag_system(
        qdrant_client=client,
        collection_name=COLLECTION_NAME,
        embedding_model=embedding_model,
        model_choice="qwen2-1.5b",  # Use smallest model for demo
    )

    return rag


def demo_rag():
    """Run a simple RAG demonstration."""

    # Set up the system
    rag = setup_rag_demo()
    if rag is None:
        return

    # Demo queries
    demo_queries = [
        "What topics are covered in these blog posts?",
        "How can I get started with machine learning?",
        "What are some coding tips mentioned?",
    ]

    print("\n" + "=" * 80)
    print("RAG SYSTEM DEMONSTRATION")
    print("=" * 80)

    for i, query in enumerate(demo_queries, 1):
        print(f"\nDemo Query {i}: {query}")
        print("-" * 60)

        try:
            result = rag.ask(query, top_k=3, max_length=200)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {', '.join(result['sources'])}")
        except Exception as e:
            print(f"Error processing query: {e}")

        print("-" * 60)

    print("\nDemo completed! You can now use rag.chat() for interactive mode.")
    return rag


if __name__ == "__main__":
    print("Available small LLM models:")
    for key, value in SMALL_LLM_OPTIONS.items():
        print(f"  {key}: {value}")
    print()

    rag_system = demo_rag()

    # Uncomment the line below for interactive chat
    # if rag_system:
    #     rag_system.chat()
