# Second Brain - Groq RAG

RAG personal knowledge assistant using Groq, LangChain, LangGraph, ChromaDB, Hugging Face embeddings, FastAPI and Streamlit.

## Setup

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install:
```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Groq API key.

Backend:
```bash
uvicorn backend.main:app --reload
```

Frontend:
```bash
streamlit run frontend/app.py
```

## Important

If the previous Gemini version was used, delete `backend/data/chroma/` and ingest documents again because this version uses Hugging Face embeddings.

## RAG flow

Documents -> Loaders -> Recursive Chunking -> Hugging Face Embeddings -> ChromaDB -> Groq Query Rewrite -> Retrieval -> Groq Relevance Grading -> Groq Grounded Answer -> Source Citation
