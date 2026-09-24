import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from backend.schemas import QuestionRequest, QuestionResponse, URLRequest
from backend.ingestion.pipeline import ingest_file, ingest_url
from backend.rag.graph import ask_question

app = FastAPI(title="Second Brain API", version="1.0.0")
UPLOAD_DIR = Path("backend/data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Second Brain API is running"}

@app.post("/ingest/file")
async def upload_file(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".md", ".docx"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    path = UPLOAD_DIR / file.filename
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        return {"message": "File ingested successfully", "filename": file.filename, **ingest_file(str(path))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {e}")

@app.post("/ingest/url")
def ingest_web_url(request: URLRequest):
    try:
        return {"message": "URL ingested successfully", **ingest_url(request.url)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL ingestion error: {e}")

@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        return QuestionResponse(answer=ask_question(request.question))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")
