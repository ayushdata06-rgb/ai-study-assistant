# rag/loader.py
"""
Phase 1 — Document Loading & Text Splitting
Supports PDF and TXT/MD files. Splits them into overlapping chunks
suitable for embedding and vector-store retrieval.
"""

import sys
import io
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ensure stdout is UTF-8 on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────

def load_documents(file_path: str) -> list:
    """
    Load a PDF or TXT/MD file and return a list of LangChain Document objects.

    Each Document carries:
      - page_content : raw text of the page / file
      - metadata     : {"source": "<path>", "page": <int>}  (page only for PDFs)

    Raises:
      ValueError  if the file extension is not supported.
      FileNotFoundError  if the path does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Supported types: .pdf, .txt, .md"
        )

    documents = loader.load()
    print(f"[Loader] Loaded {len(documents)} page(s) from '{path.name}'")
    return documents


# ─────────────────────────────────────────────
# Splitter
# ─────────────────────────────────────────────

def split_documents(
    documents: list,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list:
    """
    Split LangChain Documents into smaller, overlapping chunks.

    Args:
        documents:      List of LangChain Document objects (output of load_documents).
        chunk_size:     Max characters per chunk (1000 works well for study notes).
        chunk_overlap:  Characters shared between consecutive chunks to preserve
                        context across boundaries.

    Returns:
        List of Document chunks, each retaining the original metadata.

    Separator priority (tries largest logical unit first):
        paragraph → newline → sentence-end → word → character
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(
        f"[Loader] Split {len(documents)} page(s) into "
        f"{len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


# ─────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────

def load_and_split(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list:
    """
    Convenience wrapper: load a file and split it in a single call.

    Args:
        file_path:      Path to the PDF or TXT/MD file.
        chunk_size:     Passed through to split_documents.
        chunk_overlap:  Passed through to split_documents.

    Returns:
        List of Document chunks ready for embedding.
    """
    docs = load_documents(file_path)
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunks
