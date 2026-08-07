import os
import numpy as np
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Loading existing contextual chunks...")
vectordb = Chroma(persist_directory="./data/nepal_legal_db_v3_contextual", embedding_function=embeddings_model)
all_docs = vectordb.get()
documents = all_docs["documents"]
metadatas = all_docs["metadatas"]

print(f"Loaded {len(documents)} leaf chunks")

def embed_texts(texts):
    return np.array(embeddings_model.embed_documents(texts))

def summarize_cluster(texts, level, cluster_id):
    combined = "\n\n---\n\n".join(texts[:15])  # cap input size for very large clusters
    prompt = (
        f"Below are excerpts from the Constitution of Nepal, grouped because they are "
        f"topically related.\n\n{combined}\n\n"
        f"Write a concise summary (3-5 sentences) capturing what these provisions "
        f"collectively establish. Mention specific Article numbers where relevant. "
        f"Output ONLY the summary."
    )
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=250,
    )
    return (response.choices[0].message.content or "").strip()

def cluster_level(texts, metadatas, level, max_clusters=15):
    vectors = embed_texts(texts)
    n = len(texts)
    if n <= 3:
        # too few to cluster meaningfully — treat as one cluster
        labels = [0] * n
        k = 1
    else:
        best_k, best_score = 2, -1
        max_k = min(max_clusters, n - 1)
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(vectors)
            if len(set(km.labels_)) < 2:
                continue
            score = silhouette_score(vectors, km.labels_)
            if score > best_score:
                best_score, best_k = score, k
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(vectors)
        labels = km.labels_
        k = best_k

    print(f"  Level {level}: clustered {n} nodes into {k} clusters")

    new_documents = []
    for cluster_id in range(k):
        cluster_texts = [texts[i] for i in range(n) if labels[i] == cluster_id]
        cluster_articles = [metadatas[i].get("article", "?") for i in range(n) if labels[i] == cluster_id]

        summary = summarize_cluster(cluster_texts, level, cluster_id)
        print(f"    Cluster {cluster_id} ({len(cluster_texts)} nodes, articles {cluster_articles[:5]}...): {summary[:80]}")

        new_documents.append(Document(
            page_content=summary,
            metadata={
                "source": "Constitution of Nepal 2015",
                "level": level,
                "cluster_id": f"L{level}-{cluster_id}",
                "child_articles": ",".join(str(a) for a in cluster_articles),
                "chunk_id": f"raptor-L{level}-{cluster_id}",
            }
        ))
    return new_documents

# Build tree level by level until we're down to a small number of top-level summaries
current_texts = documents
current_metadatas = metadatas
all_raptor_nodes = []
level = 1

while len(current_texts) > 8:
    new_docs = cluster_level(current_texts, current_metadatas, level)
    all_raptor_nodes.extend(new_docs)
    current_texts = [d.page_content for d in new_docs]
    current_metadatas = [d.metadata for d in new_docs]
    level += 1
    if level > 4:  # safety cap
        break

print(f"\nTotal RAPTOR summary nodes created: {len(all_raptor_nodes)}")

print("Adding RAPTOR nodes to a new collection...")
raptor_vectordb = Chroma.from_documents(
    documents=all_raptor_nodes,
    embedding=embeddings_model,
    persist_directory="./data/nepal_legal_raptor_tree"
)
print(f"Done. RAPTOR tree DB has {raptor_vectordb._collection.count()} summary nodes.")