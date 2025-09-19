"""
Simple RAG (Retrieval-Augmented Generation) System

This module provides a lightweight RAG implementation that uses:
- Qdrant for vector storage and similarity search
- FastEmbed for embeddings
- Small LLMs (like qwen3-1.7b) for CPU-based text generation
"""

from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from qdrant_client import QdrantClient
from embedding_utils import compute_embeddings


class SimpleRAG:
    """A lightweight RAG system using Qdrant and a small LLM."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        embedding_model,
        llm_model_name: str = "google/gemma-3-1b-it",
    ):
        """
        Initialize the RAG system.

        Args:
            qdrant_client: QdrantClient instance
            collection_name: Name of the Qdrant collection
            embedding_model: FastEmbed model for computing embeddings
            llm_model_name: Small LLM model name (default: Qwen3-1.7B for CPU)
        """
        self.client = qdrant_client
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        print(
            f"\t⏳ Loading small LLM: {llm_model_name}. This may take a few minutes on first run..."
        )

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name, trust_remote_code=True
        )

        # Load model with CPU optimization
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✅ RAG system initialized successfully!")

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from the vector database."""
        query_embedding = compute_embeddings([query], self.embedding_model)[0]

        search_results = self.client.query_points(
            collection_name=self.collection_name, query=query_embedding, limit=top_k
        ).points

        contexts = []
        for result in search_results:
            contexts.append(
                {
                    "title": result.payload["title"],
                    "text": result.payload["chunk_text"],
                    "similarity": result.score,
                    "preview": result.payload["chunk_preview"],
                }
            )

        return contexts

    def generate_answer(
        self, query: str, contexts: List[Dict[str, Any]], max_length: int = 512
    ) -> str:
        """Generate an answer using the LLM and retrieved contexts."""

        # Prepare context text (limit to avoid token overflow)
        context_text = "\n\n".join(
            [
                f"Document: {ctx['title']}\nContent: {ctx['text'][:800]}..."
                for ctx in contexts[:3]  # Use top 3 contexts to avoid token limit
            ]
        )

        # Create prompt optimized for Phi-3.5
        prompt = f"""<|system|>
You are a helpful assistant that answers questions based on provided context from blog posts. Provide accurate, concise answers based on the given information.<|end|>
<|user|>
Context:
{context_text}

Question: {query}<|end|>
<|assistant|>
"""

        # Tokenize and generate
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048, padding=True
        )

        # Generate response
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response (skip the input prompt)
        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        ).strip()

        return response

    def ask(self, query: str, top_k: int = 5, max_length: int = 512) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve context and generate answer."""
        print(f"\t⏳ Processing query: {query}")

        # Retrieve relevant contexts
        print("\t⏳ Retrieving relevant contexts...")
        contexts = self.retrieve_context(query, top_k)

        print(f"\t✅ Found {len(contexts)} relevant chunks")
        for i, ctx in enumerate(contexts, 1):
            print(f"\t\t{i}. {ctx['title']} (similarity: {ctx['similarity']:.3f})")

        # Generate answer
        print("\t💭 Generating answer...")
        answer = self.generate_answer(query, contexts, max_length)

        return {
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "sources": [ctx["title"] for ctx in contexts],
        }

    def chat(self):
        """Interactive chat interface."""
        print("\n" + "=" * 60)
        print("\t🤖 RAG Chat Interface - Type 'quit' to exit")
        print("=" * 60)

        while True:
            try:
                query = input("\n❓ Your question: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!")
                    break

                if not query:
                    continue

                result = self.ask(query)
                print(f"\n\t💡 Answer: {result['answer']}")
                print(f"\t📚 Sources: {', '.join(result['sources'])}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


# Alternative lightweight models for CPU usage
SMALL_LLM_OPTIONS = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "gemma3-1b": "google/gemma-3-1b-it",
}


def create_rag_system(
    qdrant_client: QdrantClient,
    collection_name: str,
    embedding_model,
    model_choice: str = "qwen3-1.7b",
) -> SimpleRAG:
    """
    Factory function to create a RAG system with different model options.

    Args:
        qdrant_client: QdrantClient instance
        collection_name: Name of the Qdrant collection
        embedding_model: FastEmbed model for computing embeddings
        model_choice: Choice of small LLM ('gemma3-1b', 'qwen3-1.7b')
    """
    if model_choice not in SMALL_LLM_OPTIONS:
        print(f"\t⚠️  Unknown model choice: {model_choice}")
        print(f"\t📝 Available options: {list(SMALL_LLM_OPTIONS.keys())}")
        model_choice = "qwen3-1.7b"

    llm_model_name = SMALL_LLM_OPTIONS[model_choice]
    print(f"\n⏳ Creating RAG system with {model_choice} ({llm_model_name})")

    return SimpleRAG(qdrant_client, collection_name, embedding_model, llm_model_name)
