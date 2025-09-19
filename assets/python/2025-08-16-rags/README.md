# RAG System with Small LLM

This directory contains a Retrieval-Augmented Generation (RAG) system that leverages:

- **Qdrant** for vector storage and similarity search
- **FastEmbed** for embeddings (using `sentence-transformers/all-MiniLM-L6-v2`)
- **Small LLMs** optimized for CPU usage (Phi-3.5-mini, Qwen2-1.5B, etc.)

## Features

- 🔍 **Semantic Search**: Find relevant document chunks using vector similarity
- 🤖 **Local LLM**: Run small language models entirely on CPU
- 💬 **Interactive Chat**: Chat interface for asking questions about your documents
- 📊 **Visualization**: UMAP plots and similarity heatmaps
- 🎯 **Filtering**: Search within specific articles or by metadata

## Quick Start

### 1. Run the Main Analysis

```python
# This will process your markdown files, create embeddings, and set up the RAG system
python main.py
```

### 2. Interactive Demo

```python
# Run a simple demo with a smaller dataset
python rag_demo.py
```

### 3. Use the RAG System Programmatically

```python
from rag_system import create_rag_system
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

# Set up your components (see main.py for full setup)
client = QdrantClient(":memory:")
embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

# Create RAG system
rag = create_rag_system(
    qdrant_client=client,
    collection_name="your_collection",
    embedding_model=embedding_model,
    model_choice="qwen2-1.5b"  # Fastest option
)

# Ask questions
result = rag.ask("What is machine learning?")
print(result['answer'])

# Start interactive chat
rag.chat()
```

## Available Small LLM Models

| Model          | Size | Speed  | Quality  | Best For                          |
| -------------- | ---- | ------ | -------- | --------------------------------- |
| `qwen2-1.5b`   | 1.5B | ⭐⭐⭐ | ⭐⭐     | Quick responses, limited hardware |
| `gemma2-2b`    | 2B   | ⭐⭐   | ⭐⭐⭐   | Balanced performance              |
| `phi-3-mini`   | 3.8B | ⭐     | ⭐⭐⭐   | Better reasoning                  |
| `phi-3.5-mini` | 3.8B | ⭐     | ⭐⭐⭐⭐ | Best overall (default)            |

## Configuration Options

You can customize the RAG system by:

1. **Changing the LLM model**:

   ```python
   rag = create_rag_system(..., model_choice="qwen2-1.5b")
   ```

2. **Adjusting retrieval parameters**:

   ```python
   result = rag.ask("question", top_k=10, max_length=500)
   ```

3. **Using different embedding models** (in `main.py`):
   ```python
   MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Current
   # MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Alternative
   ```

## Performance Notes

- **First run**: Will download models (~1-4GB depending on LLM choice)
- **CPU usage**: Optimized for CPU inference with float32 precision
- **Memory**: Requires 4-8GB RAM depending on model size
- **Speed**: Generation takes 10-30 seconds per response on modern CPUs

## File Structure

- `main.py`: Complete pipeline from markdown processing to RAG system
- `rag_system.py`: Core RAG implementation
- `rag_demo.py`: Standalone demo script
- `embedding_utils.py`: Utilities for text processing and embeddings

## Dependencies

Core dependencies are managed via `pyproject.toml`:

- `transformers`: For LLM models
- `torch`: PyTorch backend
- `qdrant-client`: Vector database
- `fastembed`: Fast embedding computation
- `sentence-transformers`: Embedding models

## Tips

1. **For fastest responses**: Use `qwen2-1.5b` model
2. **For best quality**: Use `phi-3.5-mini` model (default)
3. **For memory constraints**: Reduce `top_k` parameter and `max_length`
4. **For better context**: Increase chunk overlap in document processing
