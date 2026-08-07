from fastapi import FastAPI
from pydantic import BaseModel
from app.hybrid_retrieval import HybridRetriever
from langchain_chroma import Chroma
from app.rag import answer_legal_question, vectordb, embeddings
from app.router import classify_query

retriever = HybridRetriever(vectordb)

raptor_db = Chroma(
    persist_directory="./data/nepal_legal_raptor_tree",
    embedding_function=embeddings  # reuse the same embeddings object already defined
)

app = FastAPI(
    title="Nepal Law Assistant API",
    description="Ask questions about Nepali laws in plain English",
    version="1.0.0"
)

BROAD_QUERY_SIGNALS = [
    "overall", "in general", "generally", "summarize", "summary",
    "what does the constitution say about", "broadly", "as a whole",
]

def is_broad_query(question: str) -> bool:
    q_lower = question.lower()
    return any(signal in q_lower for signal in BROAD_QUERY_SIGNALS)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

@app.on_event("startup")
async def startup_event():
    pass # load_model()

@app.get("/")
def root():
    return {"message": "Nepal Law Assistant API is running"}

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    route = classify_query(request.question)

    if route == "broad_summary":
        results = raptor_db.similarity_search_with_score(request.question, k=3)
        context = "\n\n".join(
            f"[Cluster {doc.metadata.get('cluster_id')} — Articles {doc.metadata.get('child_articles')}]\n{doc.page_content}"
            for doc, score in results
        )
        answer = answer_legal_question(request.question, context=context)
    elif route == "no_retrieval_needed":
        answer = "I can only answer questions about the Constitution of Nepal. Please ask a specific question about its provisions."
    else:
        answer = answer_legal_question(request.question, retriever=retriever)

    return AnswerResponse(question=request.question, answer=answer)

@app.post("/search")
def search(request: QuestionRequest):
    route = classify_query(request.question)

    if route == "broad_summary":
        results = raptor_db.similarity_search_with_score(request.question, k=3)
        return {
            "question": request.question,
            "route": route,
            "chunks": [
                {
                    "cluster_id": doc.metadata.get("cluster_id"),
                    "child_articles": doc.metadata.get("child_articles"),
                    "content": doc.page_content[:300],
                }
                for doc, score in results
            ],
        }
    elif route == "no_retrieval_needed":
        return {
            "question": request.question,
            "route": route,
            "chunks": [],
        }
    else:
        results = retriever.hybrid_search(request.question, k=3, bm25_weight=0.4, dense_weight=0.6)
        return {
            "question": request.question,
            "route": route,
            "chunks": [
                {
                    "chunk_id": r["chunk_id"],
                    "article": r["metadata"]["article"],
                    "part": r["metadata"]["part"],
                    "content": r["content"][:300],
                }
                for r in results
            ],
        }