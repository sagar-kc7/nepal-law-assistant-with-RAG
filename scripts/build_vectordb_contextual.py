import os
import re
import time
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("Loading cleaned text...")
with open("./data/consitution_cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read()

part_pattern = re.compile(r"Part-(\d+)")
article_pattern = re.compile(r"\n(\d+)\.\s+([^\n:]+):")

part_matches = [(m.start(), m.group(1)) for m in part_pattern.finditer(text)]

def get_part_for_position(pos):
    current_part = "Preamble"
    for start, part_num in part_matches:
        if start <= pos:
            current_part = part_num
        else:
            break
    return current_part

article_matches = list(article_pattern.finditer(text))
print(f"Found {len(article_matches)} articles")

def generate_context(article_text, article_num, article_title, part_num):
    """Ask the LLM for a short context blurb to prepend before embedding."""
    prompt = (
        f"This is Article {article_num} ('{article_title}') from Part {part_num} "
        f"of the Constitution of Nepal.\n\n"
        f"Article text:\n{article_text[:600]}\n\n"
        f"Write ONE short sentence (max 25 words) that situates this Article within the "
        f"Constitution's structure. You MUST include the 2-3 most important specific terms, "
        f"names, or classifications named in the Article text itself — do not paraphrase them away. "
        f"For example, if the Article names a specific status, right, role, or system (e.g. "
        f"'republic', 'federal', a named body or right), keep that exact word in your sentence. "
        f"Output ONLY the sentence, nothing else."
    )
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=60,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  Context generation failed for Article {article_num}: {e}")
        return ""

documents = []
for i, match in enumerate(article_matches):
    article_num = match.group(1)
    article_title = match.group(2).strip()
    start = match.start()
    end = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(text)
    article_text = text[start:end].strip()
    part_num = get_part_for_position(start)

    max_len = 1200
    if len(article_text) <= max_len:
        sub_chunks = [article_text]
    else:
        sub_chunks = [article_text[j:j+max_len] for j in range(0, len(article_text), max_len - 150)]

    context_blurb = generate_context(article_text, article_num, article_title, part_num)
    print(f"[{i+1}/{len(article_matches)}] Article {article_num}: {context_blurb[:80]}")

    for sub_idx, sub_text in enumerate(sub_chunks):
        # prepend context ONLY to the text that gets embedded — store original separately
        embedded_text = f"{context_blurb}\n\n{sub_text}" if context_blurb else sub_text

        documents.append(Document(
            page_content=sub_text,  # original text, shown to the LLM at answer time
            metadata={
                "source": "Constitution of Nepal 2015",
                "part": part_num,
                "article": article_num,
                "article_title": article_title,
                "chunk_id": f"{article_num}-{sub_idx}",
                "context_blurb": context_blurb,
                "embedded_text": embedded_text,  # what actually got embedded
            }
        ))

    time.sleep(0.05)  # light rate-limit courtesy

print(f"Total chunks: {len(documents)}")

print("Building contextual vector DB...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Embed the CONTEXTUALIZED text, but store original page_content
contextual_docs = [
    Document(page_content=doc.metadata["embedded_text"], metadata=doc.metadata)
    for doc in documents
]

vectordb = Chroma.from_documents(
    documents=contextual_docs,
    embedding=embeddings,
    persist_directory="./data/nepal_legal_db_v3_contextual"
)

print(f"Done. Total chunks in DB: {vectordb._collection.count()}")