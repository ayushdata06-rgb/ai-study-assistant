# rag/vectorstore.py
"""
Phase 3 — Vector Store with ChromaDB
Handles persisting, loading, and querying document embeddings.

Collection name: "study_notes"
Persistence dir: ./chroma_db  (gitignored — stays local)
"""

import os
from langchain_chroma import Chroma
from rag.embedder import get_embeddings

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "study_notes"


# ─────────────────────────────────────────────
# Create / populate
# ─────────────────────────────────────────────

def create_vectorstore(chunks: list) -> Chroma:
    """
    Embed all chunks and store them in a persistent ChromaDB collection.

    Call this when the user uploads a new document.  If the collection
    already exists, ChromaDB will ADD the new chunks to it (no wipe).

    Args:
        chunks: List of LangChain Document objects (output of loader.py).

    Returns:
        A live Chroma instance you can immediately query.
    """
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    print(
        f"[VectorStore] Stored {len(chunks)} chunks in ChromaDB "
        f"-> '{CHROMA_DIR}' (collection: {COLLECTION_NAME})"
    )
    return vectorstore


# ─────────────────────────────────────────────
# Load existing store
# ─────────────────────────────────────────────

def load_vectorstore() -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.

    Call this at app startup when chroma_db/ already exists.

    Returns:
        A live Chroma instance backed by the persisted collection.

    Raises:
        FileNotFoundError if the persist directory does not exist yet.
    """
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"ChromaDB directory not found at '{CHROMA_DIR}'. "
            "Upload a document first to create the vector store."
        )

    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    print(f"[VectorStore] Loaded existing ChromaDB from '{CHROMA_DIR}'")
    return vectorstore


# ─────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────

def get_retriever(vectorstore: Chroma, k: int = 4):
    """
    Return a LangChain Retriever that fetches the top-k most relevant chunks.

    k=4 is a good default:
      - Enough context for multi-step reasoning
      - Small enough to keep the LLM prompt tight

    Args:
        vectorstore: A live Chroma instance.
        k:           Number of chunks to retrieve per query.

    Returns:
        LangChain BaseRetriever compatible with LCEL chains.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def vectorstore_exists() -> bool:
    """Return True if a persisted ChromaDB collection is already on disk."""
    return os.path.exists(CHROMA_DIR) and bool(os.listdir(CHROMA_DIR))
