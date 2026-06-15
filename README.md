# 🎓 AI Study Assistant

An intelligent study assistant powered by LangChain, Groq LLMs, and ChromaDB — built with Streamlit.

## Features

- 📄 Upload and parse PDF documents
- 🔍 Semantic search over your study materials
- 🤖 Chat with your documents using Groq LLMs
- 💾 Persistent vector storage via ChromaDB

## Setup

### Prerequisites

- Python 3.9+
- Git

### Installation

```bash
# Clone the repo
git clone https://github.com/ayushdata06-rgb/ai-study-assistant.git
cd ai-study-assistant

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Running

```bash
streamlit run app.py
```

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key from console.groq.com |

> ⚠️ Never commit your `.env` file. It is listed in `.gitignore`.

## Tech Stack

| Library | Purpose |
|---|---|
| Streamlit | Frontend UI |
| LangChain | LLM orchestration |
| langchain-groq | Groq LLM integration |
| ChromaDB | Vector database |
| sentence-transformers | Embeddings |
| PyPDF | PDF parsing |
