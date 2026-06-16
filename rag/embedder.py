# rag/embedder.py
"""
Phase 2 — Embeddings with sentence-transformers
Wraps HuggingFace's all-MiniLM-L6-v2 as a LangChain-compatible embedding model.

Model facts:
  - 384-dimensional output vectors
  - ~90 MB download (cached after first run in ~/.cache/huggingface/)
  - CPU-friendly; switch model_kwargs device to "cuda" if you have a GPU
  - Normalized vectors → cosine similarity works correctly out of the box
"""

from langchain_huggingface import HuggingFaceEmbeddings

# Default model — small, fast, and surprisingly good for study-note retrieval
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Singleton cache so we don't reload weights on every call within a session
_embeddings_instance = None


def get_embeddings(
    model_name: str = _MODEL_NAME,
    device: str = "cpu",
    normalize: bool = True,
) -> HuggingFaceEmbeddings:
    """
    Return a LangChain-compatible HuggingFaceEmbeddings instance.

    On the first call the model weights are downloaded (~90 MB) and cached.
    Subsequent calls within the same Python process reuse the cached instance.

    Args:
        model_name: HuggingFace model identifier.
        device:     "cpu" (default) or "cuda" for GPU acceleration.
        normalize:  Normalize output vectors for cosine similarity (recommended).

    Returns:
        HuggingFaceEmbeddings instance ready for embed_documents / embed_query.
    """
    global _embeddings_instance

    if _embeddings_instance is None:
        print(f"[Embedder] Loading model '{model_name}' on {device} …")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": normalize},
        )
        print("[Embedder] Model ready.")

    return _embeddings_instance
