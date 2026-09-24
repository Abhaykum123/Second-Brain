from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from backend.config import EMBEDDING_MODEL, CHROMA_DIR, COLLECTION_NAME

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

def add_documents(documents):
    if not documents:
        return 0
    ids = [
        f"{d.metadata.get('source', 'unknown')}-{d.metadata.get('chunk_id', i)}"
        for i, d in enumerate(documents)
    ]
    vectorstore.add_documents(documents=documents, ids=ids)
    return len(documents)

def similarity_search(query: str, k: int = 5):
    return vectorstore.similarity_search(query, k=k)
