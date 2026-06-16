# 📚 AI Study Assistant

> Ask questions from your study notes and get precise, cited answers grounded in your material — no hallucinations.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## ✨ Features

- **Upload PDFs or text notes** and index them instantly
- **Semantic search** over your material (not keyword matching)
- **Grounded answers** — the LLM only uses your uploaded content
- **Page-level source citations** shown alongside every answer
- **Persistent vector store** — re-open the app and your index survives
- **100% free stack** — no paid APIs, no cloud fees beyond the free Groq tier

---

## 🛠️ Tech Stack

| Layer | Tool | Why |
|---|---|---|
| LLM | Groq (Llama 3.1 8B Instant) | Free tier, ~500ms inference |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | 384-dim, runs on CPU, ~90 MB |
| Vector DB | ChromaDB | Local persistence, zero config |
| Orchestration | LangChain | Loader → splitter → retriever → chain |
| UI | Streamlit | One-file chat interface |

---

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone https://github.com/ayushdata06-rgb/ai-study-assistant.git
cd ai-study-assistant

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / Mac

pip install -r requirements.txt
```

### 2. Get your free Groq API key

1. Visit **https://console.groq.com**
2. Sign up → **API Keys** → **Create Key**
3. Copy the key (starts with `gsk_...`)

### 3. Configure environment

```bash
copy .env.example .env      # Windows
# cp .env.example .env      # Linux / Mac
```

Open `.env` and paste your key:

```env
GROQ_API_KEY=gsk_your_key_here
```

### 4. Run

```bash
streamlit run app.py
```

Open **http://localhost:8501**, upload your notes, and start asking!

---

## 🏗️ Project Structure

```
ai-study-assistant/
├── app.py                    # Streamlit UI (Phase 5)
├── rag/
│   ├── __init__.py
│   ├── loader.py             # PDF/TXT loader + recursive text splitter (Phase 1)
│   ├── embedder.py           # sentence-transformers embeddings (Phase 2)
│   ├── vectorstore.py        # ChromaDB vector store manager (Phase 3)
│   └── chain.py              # LangChain RAG chain + Groq LLM (Phase 4)
├── data/
│   └── uploaded_docs/        # uploaded files land here (gitignored)
├── chroma_db/                # persisted vector DB (gitignored)
├── .env                      # API keys (gitignored)
├── .env.example              # template for contributors
├── requirements.txt
└── README.md
```

---

## 🔄 How It Works

```
User uploads PDF
      │
      ▼
loader.py ──► PyPDFLoader / TextLoader
      │        RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
      ▼
embedder.py ──► all-MiniLM-L6-v2 → 384-dim vectors
      │
      ▼
vectorstore.py ──► ChromaDB (persisted to ./chroma_db)
      │
      ▼  (on user question)
retriever ──► top-4 most similar chunks (cosine similarity)
      │
      ▼
chain.py ──► Groq Llama 3.1 + custom study prompt
      │
      ▼
Answer + source citations shown in Streamlit UI
```

---

## 🗺️ Roadmap

- [x] PDF & TXT document loading
- [x] Semantic embedding with sentence-transformers
- [x] ChromaDB persistent vector store
- [x] RAG chain with Groq LLM
- [x] Streamlit chat UI with source citations
- [ ] Multi-turn conversation memory
- [ ] Per-document filtering (ask from a specific file only)
- [ ] Reranking for better retrieval accuracy
- [ ] Web search fallback when notes don't have the answer
- [ ] Export Q&A sessions as PDF
- [ ] Streamlit Cloud deployment

---

## 🚀 Deployment (Streamlit Cloud — free)

1. Push your repo to GitHub
2. Go to **https://share.streamlit.io** and sign in with GitHub
3. Select repo → `app.py` → **Deploy**
4. Add your secret in **Settings → Secrets**:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

You get a public URL: `https://your-app.streamlit.app`

---

## 🤝 Contributing

PRs and issues are welcome! Please open an issue first to discuss larger changes.

---

## 📄 License

MIT — see [LICENSE](LICENSE)
