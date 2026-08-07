# Nepal Law Assistant — Advanced RAG Pipeline

Ask questions about the Constitution of Nepal in plain English and get answers grounded in the actual text, with sources shown.

This started as a simple dense-retrieval RAG chatbot (v1). This is v2 — a rebuild focused on one question: **how do I know my RAG is actually good, and how do I make it better with evidence instead of guesswork?**

Every technique below was added, tested against a fixed evaluation set, and kept only if it measurably helped. Two of them didn't help on the first try — I kept those failures in this README because they taught me more than the parts that worked cleanly.

## What it does

- Answers direct questions about specific Articles ("Who holds sovereignty in Nepal?")
- Answers broad questions that span many Articles ("What does the Constitution say overall about fundamental rights?")
- Routes each question automatically to the right retrieval strategy
- Shows its sources for every answer — you can see exactly which Article(s) or topic cluster the answer came from

## Architecture

```
Question
   │
   ▼
LLM Router (Groq, llama-3.1-8b-instant)
   │
   ├── specific_lookup ──► Hybrid Retrieval (BM25 + dense, weighted fusion)
   │                          │
   │                          ▼
   │                       Cross-encoder reranker (top 10 → top 3)
   │
   ├── broad_summary ────► RAPTOR tree (hierarchical cluster summaries)
   │
   └── no_retrieval_needed ► Direct fallback response
                          │
                          ▼
                 Groq (llama-3.3-70b-versatile) generates the final answer
```

**Chunking:** structure-aware, one chunk per Article (parsed from the Constitution's actual numbering), not a blind fixed-size splitter. Long Articles get sub-chunked but keep their Article/Part metadata.

**Contextual Retrieval:** each chunk gets a short LLM-generated blurb prepended before embedding, describing where it sits in the Constitution and what it covers — closes the gap when a question's wording doesn't match the source text's wording.

**Hybrid Retrieval:** BM25 (keyword) and dense embeddings are each normalized to 0-1 and combined with a weighted sum (40% BM25 / 60% dense here) — not plain rank fusion, for reasons explained below.

**Reranking:** a cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores the top 10 hybrid candidates by reading the actual query and chunk text together, then keeps the top 3.

**RAPTOR:** the 483 leaf chunks are clustered by embedding similarity and summarized into 13 topic clusters, then those are clustered again into a small top level. Broad questions retrieve a whole cluster summary instead of trying to piece one together from a handful of individual Articles.

**Routing:** a small LLM call classifies each question as a specific lookup, a broad summary request, or not something the corpus can answer — and sends it down the right path.

## Evaluation

I built a small ground-truth question set (question → which Article(s) should be retrieved) and measured context precision/recall at every stage, rather than eyeballing whether answers "looked right."

| Pipeline stage | Avg Precision | Avg Recall | What happened |
|---|---|---|---|
| Structure-aware chunking, dense-only | 0.25 | 0.75 | Baseline. One question failed outright. |
| + Hybrid (BM25 + dense, plain rank fusion) | 0.17 | 0.50 | **Regressed.** Rank-only fusion tied two candidates and lost the right one on a coin-flip. |
| + Hybrid (normalized weighted-score fusion) | 0.25 | 0.75 | Fixed the regression by fusing on actual score magnitude, not just rank position. |
| + Contextual Retrieval (first-pass blurbs) | 0.17 | 0.50 | **Regressed again.** The LLM paraphrased away the exact terms (e.g. "republic") that made a question answerable. |
| + Contextual Retrieval (term-preserving blurbs) | 0.25 | 0.75 | Fixed by forcing the prompt to keep key terms from the source text instead of abstracting them. |
| + Reranking | 0.33 | **1.00** | Closed the one remaining gap — a case neither BM25 nor dense retrieval ranked highly enough on its own. |

Precision caps out at 0.33 here because retrieval returns k=3 candidates while most questions only have one correct Article — that's an artifact of the metric at this k, not a quality ceiling.

*(Note: this eval set currently has 4 hand-built questions across direct/paraphrased/multi-hop types. It's small by design — enough to validate the pipeline logic and catch real regressions, as shown above — and is actively being expanded to 15-20 questions across more sections of the Constitution.)*

## What actually broke, and what that taught me

- **Rank-based fusion (RRF) isn't always the right call.** It threw away *how confident* each retriever was and only kept *where it ranked*, which meant a strong dense match could lose a tie to a weak BM25 match just because they landed on the same rank position. Switching to normalized score fusion fixed it.
- **Contextual Retrieval can hurt as easily as help.** If the LLM-generated context blurb paraphrases away the specific terms a question needs, you've actively made retrieval worse, not better. The fix wasn't more context — it was more *precise* context, forcing the prompt to preserve exact terminology from the source.
- **Some retrieval gaps genuinely need reranking.** One case survived every fusion and prompt tweak because neither BM25 nor dense retrieval, on their own, ranked it in the top 3 — only a cross-encoder comparing the actual query against the actual chunk text closed the gap.

## Stack

- **Retrieval:** ChromaDB, `sentence-transformers/all-MiniLM-L6-v2`, `rank_bm25`, `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Generation & routing:** Groq API (`llama-3.3-70b-versatile` for answers, `llama-3.1-8b-instant` for routing/context generation)
- **Serving:** FastAPI (`/ask`, `/search`)
- **Frontend:** Streamlit — shows the answer, which route handled it, and the actual retrieved sources

## Running it locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"

# build the indexes (one-time)
python3 scripts/build_vectordb.py
python3 scripts/build_vectordb_contextual.py
python3 scripts/build_raptor_tree.py

# run the API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# in a second terminal, run the frontend
streamlit run streamlit_app.py
```

## What's next

- Return sources directly from `/ask` instead of a second call to `/search`
- Redeploy to a hosted environment now that generation runs through Groq instead of a local GPU model