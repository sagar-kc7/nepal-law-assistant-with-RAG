import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

ROUTE_LABELS = ["broad_summary", "specific_lookup", "no_retrieval_needed"]

def classify_query(question: str) -> str:
    prompt = (
        f"Classify this question about the Constitution of Nepal into exactly one category.\n\n"
        f"broad_summary — the question explicitly asks for an overview, summary, or general "
        f"treatment spanning MANY provisions or an entire topic area. Signal words: 'overall', "
        f"'in general', 'summarize', 'broadly', 'as a whole'.\n"
        f"Example: 'What does the constitution say overall about fundamental rights?'\n\n"
        f"specific_lookup — the question asks about ONE specific fact, rule, right, role, or "
        f"provision, even if the underlying topic sounds broad or conceptual. Most questions "
        f"are this category by default.\n"
        f"Example: 'Who holds sovereignty in Nepal?' (asks for ONE specific fact)\n"
        f"Example: 'What rights do citizens have to free speech?' (asks about ONE specific right)\n\n"
        f"no_retrieval_needed — a meta question not actually about Constitution content.\n"
        f"Example: 'What can you help me with?'\n\n"
        f"Question: {question}\n\n"
        f"Output ONLY one label: broad_summary, specific_lookup, or no_retrieval_needed. Nothing else."
    )
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=20,
    )
    label = (response.choices[0].message.content or "").strip()
    return label if label in ROUTE_LABELS else "specific_lookup"# safe fallback