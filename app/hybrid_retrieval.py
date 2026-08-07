from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import re

def tokenize(text):
    return re.findall(r"\w+", text.lower())

class HybridRetriever:
    def __init__(self, vectordb):
        self.vectordb = vectordb
        all_docs = vectordb.get()
        self.documents = all_docs["documents"]
        self.metadatas = all_docs["metadatas"]

        tokenized_corpus = [tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # cross-encoder for reranking — small, fast, good quality
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def bm25_search(self, query, k=10):
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            (self.metadatas[i]["chunk_id"], self.metadatas[i], self.documents[i], float(scores[i]))
            for i in ranked_indices
        ]

    def dense_search(self, query, k=10):
        results = self.vectordb.similarity_search_with_score(query, k=k)
        return [
            (doc.metadata.get("chunk_id"), doc.metadata, doc.page_content, float(dist))
            for doc, dist in results
        ]

    def _normalize(self, values, higher_is_better=True):
        if not values:
            return {}
        vmin, vmax = min(values.values()), max(values.values())
        if vmax == vmin:
            return {k: 1.0 for k in values}
        if higher_is_better:
            return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}
        else:
            return {k: (vmax - v) / (vmax - vmin) for k, v in values.items()}

    def hybrid_search(self, query, k=3, bm25_k=10, dense_k=10, bm25_weight=0.4, dense_weight=0.6, rerank_pool=10):
        bm25_results = self.bm25_search(query, k=bm25_k)
        dense_results = self.dense_search(query, k=dense_k)

        bm25_raw = {r[0]: r[3] for r in bm25_results}
        dense_raw = {r[0]: r[3] for r in dense_results}

        bm25_norm = self._normalize(bm25_raw, higher_is_better=True)
        dense_norm = self._normalize(dense_raw, higher_is_better=False)

        content_map = {}
        for chunk_id, metadata, content, _ in bm25_results:
            content_map[chunk_id] = (metadata, content)
        for chunk_id, metadata, content, _ in dense_results:
            content_map[chunk_id] = (metadata, content)

        combined = {}
        for chunk_id in set(bm25_norm) | set(dense_norm):
            combined[chunk_id] = (
                bm25_weight * bm25_norm.get(chunk_id, 0.0)
                + dense_weight * dense_norm.get(chunk_id, 0.0)
            )

        pool = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:rerank_pool]

        pairs = [(query, content_map[chunk_id][1]) for chunk_id, _ in pool]
        rerank_scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip([chunk_id for chunk_id, _ in pool], rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        return [
            {
                "chunk_id": chunk_id,
                "metadata": content_map[chunk_id][0],
                "content": content_map[chunk_id][1],
                "score": float(score),
            }
            for chunk_id, score in reranked
        ]