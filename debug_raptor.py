from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
raptor_db = Chroma(persist_directory="./data/nepal_legal_raptor_tree", embedding_function=embeddings)

query = "What does the Constitution say overall about fundamental rights?"

results = raptor_db.similarity_search_with_score(query, k=3)
for doc, score in results:
    print(f"[{doc.metadata.get('cluster_id')}] score={score:.4f}")
    print(f"  child_articles: {doc.metadata.get('child_articles')[:100]}")
    print(f"  summary: {doc.page_content[:150]}")
    print()