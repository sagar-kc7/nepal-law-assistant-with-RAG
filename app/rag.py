import os
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(
    persist_directory="./data/nepal_legal_db_v3_contextual",
    embedding_function=embeddings
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def answer_legal_question(question: str, context: str = None, retriever=None) -> str:
    if context is None:
        if retriever:
            results = retriever.hybrid_search(question, k=3)
            context = "\n\n".join(
                f"[Article {r['metadata']['article']}]\n{r['content']}" for r in results
            )
        else:
            docs = vectordb.similarity_search(question, k=3)
            context = "\n\n".join(
                f"[Article {doc.metadata['article']}]\n{doc.page_content}" for doc in docs
            )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a Nepali legal assistant. Answer using ONLY the provided context. Cite article numbers where possible. Never add extra explanation.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.1,
        max_tokens=250,  # slightly higher — RAPTOR summaries need more room to synthesize
    )
    return response.choices[0].message.content.strip()