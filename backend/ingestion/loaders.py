from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader, Docx2txtLoader

def load_file(file_path: str):
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
    elif ext in [".txt", ".md"]:
        loader = TextLoader(str(path), encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    docs = loader.load()
    for doc in docs:
        doc.metadata.update({"source": str(path), "file_name": path.name, "file_type": ext})
    return docs

def load_web_page(url: str):
    docs = WebBaseLoader(url).load()
    for doc in docs:
        doc.metadata.update({"source": url, "file_name": url, "file_type": "web"})
    return docs
