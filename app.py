# app.py
"""
Phase 5 -- Streamlit Chat UI
Upload PDFs/TXT -> index into ChromaDB -> chat with your notes.
"""

import os

# Silence ChromaDB telemetry before any chromadb import
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from rag.loader import load_and_split
from rag.vectorstore import (
    create_vectorstore,
    load_vectorstore,
    get_retriever,
    vectorstore_exists,
)
from rag.chain import build_conversational_chain, ask_with_memory

load_dotenv()

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Glassmorphism sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Header gradient text */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: rgba(255,255,255,0.55);
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        backdrop-filter: blur(8px);
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(167,139,250,0.4) !important;
        border-radius: 12px;
        color: white !important;
        font-family: 'Inter', sans-serif;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: opacity 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover { opacity: 0.88; }

    /* Secondary button */
    .stButton > button {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: white !important;
        transition: background 0.2s ease;
    }
    .stButton > button:hover {
        background: rgba(255,255,255,0.13) !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.1) !important; }

    /* Info/success/error boxes */
    [data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.06) !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.04);
        border: 2px dashed rgba(167,139,250,0.35);
        border-radius: 12px;
        padding: 0.5rem;
    }

    /* Source citation text */
    .source-badge {
        display: inline-block;
        background: rgba(167,139,250,0.15);
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.8rem;
        color: #a78bfa;
        font-weight: 500;
    }

    /* Spinner text */
    [data-testid="stSpinner"] p { color: #a78bfa !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

UPLOAD_DIR = Path("data/uploaded_docs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─── Session state ────────────────────────────────────────────────────────────
for key, default in [
    ("chain", None),
    ("chat_history", []),
    ("indexed_files", []),
    ("total_chunks", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Study Material")
    st.markdown("<hr>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Supports PDF, TXT, and Markdown files",
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"*{len(uploaded_files)} file(s) selected*")
        if st.button("⚡ Index Documents", type="primary", use_container_width=True):
            all_chunks = []
            progress = st.progress(0, text="Starting...")

            for i, file in enumerate(uploaded_files):
                save_path = UPLOAD_DIR / file.name
                save_path.write_bytes(file.getbuffer())

                progress.progress(
                    int((i + 0.5) / len(uploaded_files) * 80),
                    text=f"Reading {file.name}...",
                )

                try:
                    chunks = load_and_split(str(save_path))
                    all_chunks.extend(chunks)
                    if file.name not in st.session_state.indexed_files:
                        st.session_state.indexed_files.append(file.name)
                except Exception as e:
                    st.error(f"Error processing **{file.name}**: {e}")

            if all_chunks:
                progress.progress(85, text="Building vector index...")
                vs = create_vectorstore(all_chunks)
                retriever = get_retriever(vs)
                progress.progress(95, text="Connecting LLM...")
                st.session_state.chain = build_conversational_chain(retriever)
                st.session_state.total_chunks += len(all_chunks)
                progress.progress(100, text="Done!")
                st.success(
                    f"Indexed **{len(all_chunks)} chunks** from "
                    f"**{len(uploaded_files)} file(s)**"
                )

    # Auto-load existing DB
    if vectorstore_exists() and st.session_state.chain is None:
        st.markdown("---")
        st.markdown("*Existing index found on disk*")
        if st.button("📂 Load Existing Index", use_container_width=True):
            with st.spinner("Loading vector store..."):
                vs = load_vectorstore()
                retriever = get_retriever(vs)
                st.session_state.chain = build_conversational_chain(retriever)
            st.success("Loaded existing index!")

    st.markdown("---")

    # Stats
    if st.session_state.indexed_files:
        st.markdown("**Indexed files**")
        for f in st.session_state.indexed_files:
            st.markdown(f"- `{f}`")
        if st.session_state.total_chunks:
            st.metric("Total chunks indexed", st.session_state.total_chunks)
        st.markdown("---")

    # Actions
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col_b:
        if st.button("♻️ Reset All", use_container_width=True):
            for k in ["chain", "chat_history", "indexed_files", "total_chunks"]:
                st.session_state[k] = None if k == "chain" else ([] if k != "total_chunks" else 0)
            st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size:0.78rem; color:rgba(255,255,255,0.4); line-height:1.7'>
        🧠 <b>Model:</b> Llama 3.1 8B (Groq)<br>
        🔢 <b>Embeddings:</b> all-MiniLM-L6-v2<br>
        🗄️ <b>Vector DB:</b> ChromaDB<br>
        ⚡ <b>Framework:</b> LangChain
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Main area ───────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📚 AI Study Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Ask questions from your uploaded notes — answers grounded in your material, never hallucinated.</div>',
    unsafe_allow_html=True,
)

if st.session_state.chain is None:
    # Welcome / empty state
    st.info("👆 Upload your study material in the sidebar and click **Index Documents** to get started.")

    st.markdown("### 💡 Example questions you can ask")
    c1, c2, c3 = st.columns(3)
    examples = [
        ("⚛️ Physics", [
            "Explain Newton's second law with examples",
            "What are kinematic equations?",
            "Derive the formula for kinetic energy",
        ]),
        ("📐 Maths", [
            "What is the chain rule in differentiation?",
            "Explain integration by parts",
            "List all trigonometric identities",
        ]),
        ("⚡ Engineering", [
            "Difference between NPN and PNP transistors",
            "Explain Bernoulli's equation",
            "What is Thevenin's theorem?",
        ]),
    ]
    for col, (header, qs) in zip([c1, c2, c3], examples):
        with col:
            st.markdown(f"**{header}**")
            for q in qs:
                st.markdown(f"- *{q}*")

else:
    # ── Chat history ──
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📖 {len(msg['sources'])} source(s) used", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        fname = Path(src["source"]).name
                        page = src["page"]
                        page_str = f"p.{page}" if page != "N/A" else "txt"
                        st.markdown(
                            f'<span class="source-badge">Source {i} &mdash; {fname} [{page_str}]</span>',
                            unsafe_allow_html=True,
                        )
                        st.code(src["content"], language=None)
                        if i < len(msg["sources"]):
                            st.markdown("---")

    # ── Chat input ──
    if question := st.chat_input("Ask anything from your notes..."):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your notes and generating answer..."):
                try:
                    # Pass history BEFORE the current question for context
                    result = ask_with_memory(
                        st.session_state.chain,
                        question,
                        st.session_state.chat_history[:-1],  # exclude the just-appended user msg
                    )
                    answer = result["answer"]
                    sources = result["sources"]
                except Exception as e:
                    answer = f"Something went wrong: {e}"
                    sources = []

            st.markdown(answer)

            if sources:
                with st.expander(f"📖 {len(sources)} source(s) used", expanded=False):
                    for i, src in enumerate(sources, 1):
                        fname = Path(src["source"]).name
                        page = src["page"]
                        page_str = f"p.{page}" if page != "N/A" else "txt"
                        st.markdown(
                            f'<span class="source-badge">Source {i} &mdash; {fname} [{page_str}]</span>',
                            unsafe_allow_html=True,
                        )
                        st.code(src["content"], language=None)
                        if i < len(sources):
                            st.markdown("---")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
