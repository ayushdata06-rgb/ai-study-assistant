# rag/chain.py
"""
Phase 4 + Phase 7A -- RAG Chain with Groq LLM + Conversation Memory
Connects: user question -> retriever (ChromaDB) -> LLM (Groq/Llama) -> cited answer.
Phase 7A adds ConversationalRetrievalChain so the bot remembers previous Q&A turns.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

load_dotenv()

# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

# Used by the single-turn RetrievalQA chain (kept for compatibility)
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

Answer (be clear and structured -- use bullet points or steps if helpful):""",
)

# Used by the conversational chain for the final answer step
CONV_ANSWER_PROMPT = PromptTemplate(
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

Answer (be clear and structured -- use bullet points or steps if helpful):""",
)


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

def get_llm() -> ChatGroq:
    """
    Return the Groq-hosted Llama 3.1 8B Instant model.
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
# Phase 4 -- Single-turn chain (kept for reference)
# ─────────────────────────────────────────────

def build_rag_chain(retriever) -> RetrievalQA:
    """Single-turn RetrievalQA -- no memory of previous questions."""
    llm = get_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": STUDY_PROMPT},
    )


def ask_question(chain: RetrievalQA, question: str) -> dict:
    """Ask a single question (no history). Returns answer + sources."""
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": _extract_sources(result.get("source_documents", [])),
    }


# ─────────────────────────────────────────────
# Phase 7A -- Conversational chain (multi-turn)
# ─────────────────────────────────────────────

def build_conversational_chain(retriever) -> ConversationalRetrievalChain:
    """
    Build a ConversationalRetrievalChain -- Phase 7A upgrade.

    How it works:
      1. The LLM first *condenses* the current question + chat history into a
         standalone question (so the retriever understands follow-ups like
         "explain it more" or "give an example").
      2. The standalone question is sent to ChromaDB to fetch relevant chunks.
      3. The LLM generates a grounded answer from those chunks.

    This means follow-up questions like:
      "What about its formula?"  ->  understood as a continuation of the topic.

    Args:
        retriever: LangChain BaseRetriever from vectorstore.get_retriever().

    Returns:
        ConversationalRetrievalChain ready for {"question": ..., "chat_history": [...]}
    """
    llm = get_llm()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": CONV_ANSWER_PROMPT},
        verbose=False,
    )

    return chain


def ask_with_memory(
    chain: ConversationalRetrievalChain,
    question: str,
    chat_history: list,
) -> dict:
    """
    Ask a question with full conversation context.

    Args:
        chain:        Built ConversationalRetrievalChain.
        question:     The current user question.
        chat_history: List of past messages from st.session_state.chat_history
                      in format [{"role": "user/assistant", "content": "..."}, ...]
                      Only messages BEFORE the current question should be passed.

    Returns:
        {"answer": str, "sources": list[dict]}
    """
    # Convert Streamlit chat history format -> LangChain (question, answer) tuples
    history_tuples = _build_history_tuples(chat_history)

    result = chain.invoke({
        "question": question,
        "chat_history": history_tuples,
    })

    return {
        "answer": result["answer"],
        "sources": _extract_sources(result.get("source_documents", [])),
    }


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _extract_sources(source_documents: list) -> list:
    """Convert LangChain Documents to serialisable source dicts."""
    sources = []
    for doc in source_documents:
        sources.append({
            "content": doc.page_content[:300],
            "page": doc.metadata.get("page", "N/A"),
            "source": doc.metadata.get("source", "Unknown"),
        })
    return sources


def _build_history_tuples(chat_history: list) -> list:
    """
    Convert Streamlit chat_history list to LangChain (user, assistant) tuples.

    Streamlit format:
        [{"role": "user", "content": "q1"},
         {"role": "assistant", "content": "a1", "sources": [...]},
         ...]

    LangChain format:
        [("q1", "a1"), ("q2", "a2"), ...]
    """
    tuples = []
    messages = [m for m in chat_history if m["role"] in ("user", "assistant")]
    i = 0
    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            tuples.append((messages[i]["content"], messages[i + 1]["content"]))
            i += 2
        else:
            i += 1
    return tuples
