"""Phase 4 end-to-end test: loader -> vectorstore -> chain -> answer."""
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from rag.loader import load_and_split
from rag.vectorstore import create_vectorstore, get_retriever
from rag.chain import build_rag_chain, ask_question

print("Step 1: Loading and indexing...")
chunks = load_and_split("data/uploaded_docs/sample_physics.txt")
vs = create_vectorstore(chunks)
retriever = get_retriever(vs)

print("Step 2: Building RAG chain...")
chain = build_rag_chain(retriever)

print("Step 3: Asking question...")
result = ask_question(chain, "What is Newton's second law and what is the formula?")

print()
print("=== ANSWER ===")
print(result["answer"])
print()
print("=== SOURCES ===")
for i, s in enumerate(result["sources"], 1):
    print(f"Source {i} | Page: {s['page']} | File: {s['source']}")
    print(f"  {s['content'][:120]}...")
