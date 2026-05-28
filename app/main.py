import unsloth

from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import answer_legal_question
from app.model import load_model

from app.rag import answer_legal_question, vectordb

app = FastAPI(
    title="Nepal Law Assistant API",
    description="Ask questions about Nepali laws in plain English",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
def root():
    return {"message": "Nepal Law Assistant API is running"}

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    answer = answer_legal_question(request.question)
    return AnswerResponse(
        question=request.question,
        answer=answer
    )

@app.post("/search")
def search(request: QuestionRequest):
    from app.rag import vectordb
    docs = vectordb.similarity_search(request.question, k=3)
    return {
        "question": request.question,
        "chunks": [
            {
                "chunk_id": doc.metadata["chunk_id"],
                "content": doc.page_content[:300]
            }
            for doc in docs
        ]
    }