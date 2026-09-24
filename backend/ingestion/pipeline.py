from backend.ingestion.loaders import load_file, load_web_page
from backend.ingestion.chunker import split_documents
from backend.rag.vectorstore import add_documents

def ingest_file(file_path: str):
    docs = load_file(file_path)
    chunks = split_documents(docs)
    return {"documents": len(docs), "chunks": add_documents(chunks)}

def ingest_url(url: str):
    docs = load_web_page(url)
    chunks = split_documents(docs)
    return {"documents": len(docs), "chunks": add_documents(chunks)}
