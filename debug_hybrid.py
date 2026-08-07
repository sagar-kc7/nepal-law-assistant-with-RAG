from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.hybrid_retrieval import HybridRetriever

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(
    persist_directory="./data/nepal_legal_db_v3_contextual",
    embedding_function=embeddings
)
retriever = HybridRetriever(vectordb)

query = "What happens to a law that conflicts with the Constitution?"

print("=== BM25 top 10 ===")
for chunk_id, meta, content, score in retriever.bm25_search(query, k=10):
    print(chunk_id, meta.get("article"), round(score, 4), content[:60].replace("\n", " "))

print("\n=== Dense top 10 ===")
for chunk_id, meta, content, score in retriever.dense_search(query, k=10):
    print(chunk_id, meta.get("article"), round(score, 4), content[:60].replace("\n", " "))

print("\n=== Hybrid + Reranked top 3 ===")
for r in retriever.hybrid_search(query, k=3):
    print(r["chunk_id"], r["metadata"].get("article"), round(r["score"], 4), r["content"][:60].replace("\n", " "))