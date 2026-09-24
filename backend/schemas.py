from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str

class URLRequest(BaseModel):
    url: str

class QuestionResponse(BaseModel):
    answer: str
