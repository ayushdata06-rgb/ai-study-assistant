# rag/chain.py
"""
Phase 4 — RAG Chain with Groq LLM
Connects: user question -> retriever (ChromaDB) -> LLM (Groq/Llama) -> cited answer.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# ─────────────────────────────────────────────
# Prompt — tuned for competitive exam prep
# ─────────────────────────────────────────────

STUDY_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a knowledgeable study assistant helping students prepare for competitive exams like JEE and GATE.

Use ONLY the context below to answer the question. If the answer is not in the context, say:
"I couldn't find this in your uploaded notes. Please check your study material or ask a more specific question."

Do not make up information. Be precise and use examples from the context when possible.

Context from your notes:
-----------------------
{context}
-----------------------

Question: {question}

Answer (be clear and structured — use bullet points or steps if helpful):""",
)


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

def get_llm() -> ChatGroq:
    """
    Return the Groq-hosted Llama 3.1 8B Instant model.

    Groq provides extremely fast inference on a generous free tier.
    temperature=0.2 keeps answers factual and reproducible.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Copy .env.example to .env and add your key from console.groq.com"
        )

    return ChatGroq(
        api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=1024,
    )


# ─────────────────────────────────────────────
# Chain
# ─────────────────────────────────────────────

def build_rag_chain(retriever) -> RetrievalQA:
    """
    Build a RetrievalQA chain:
      user question -> retriever fetches top-k chunks -> LLM generates grounded answer.

    chain_type="stuff": all retrieved chunks are stuffed into a single prompt.
    Works well when k is small (default k=4) and chunks are ~1000 chars.

    Args:
        retriever: LangChain BaseRetriever (from vectorstore.get_retriever).

    Returns:
        A RetrievalQA chain ready to be called with {"query": "..."}.
    """
    llm = get_llm()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": STUDY_PROMPT},
    )

    return chain


# ─────────────────────────────────────────────
# Ask
# ─────────────────────────────────────────────

def ask_question(chain: RetrievalQA, question: str) -> dict:
    """
    Ask a question and receive an answer with source citations.

    Args:
        chain:    Built RetrievalQA chain.
        question: Natural-language question string.

    Returns:
        {
            "answer":  str   — LLM-generated answer,
            "sources": list  — [{"content", "page", "source"}, ...]
        }
    """
    result = chain.invoke({"query": question})

    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "content": doc.page_content[:300],
            "page": doc.metadata.get("page", "N/A"),
            "source": doc.metadata.get("source", "Unknown"),
        })

    return {
        "answer": result["result"],
        "sources": sources,
    }
