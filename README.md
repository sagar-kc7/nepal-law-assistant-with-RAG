# Nepal Law Assistant 🇳🇵

A GenAI-powered legal assistant that answers questions about Nepali laws 
in simple plain English.

## Live API
- Ask endpoint: https://sagar-kc7--nepal-law-assistant-nepallawapi-ask.modal.run
- Health check: https://sagar-kc7--nepal-law-assistant-nepallawapi-health.modal.run

## Tech Stack
- LLM: Llama 3.2 3B (QLoRA fine-tuned)
- RAG: LangChain + ChromaDB
- Backend: FastAPI
- Deployment: Modal (T4 GPU)

## Pipeline
1. PDF cleaning (PyMuPDF)
2. Text chunking (597 chunks)
3. Vector embeddings (sentence-transformers)
4. QLoRA fine-tuning (Unsloth + TRL)
5. RAG pipeline (ChromaDB retrieval)
6. FastAPI backend
7. Modal deployment

## Example
curl -X POST "https://sagar-kc7--nepal-law-assistant-nepallawapi-ask.modal.run" \
     -H "Content-Type: application/json" \
     -d '{"question": "Does Nepal allow death penalty?"}'

## Dataset
- Constitution of Nepal 2015 (202 pages)
- More laws coming soon