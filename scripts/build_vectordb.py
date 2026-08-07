import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

print("Loading cleaned text...")
with open("./data/consitution_cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Track current Part as we walk through the text
part_pattern = re.compile(r"Part-(\d+)")
article_pattern = re.compile(r"\n(\d+)\.\s+([^\n:]+):")

# Split the text into (position, part_number) markers
part_matches = [(m.start(), m.group(1)) for m in part_pattern.finditer(text)]

def get_part_for_position(pos):
    current_part = "Preamble"
    for start, part_num in part_matches:
        if start <= pos:
            current_part = part_num
        else:
            break
    return current_part

# Find all Article boundaries
article_matches = list(article_pattern.finditer(text))
print(f"Found {len(article_matches)} articles")

documents = []
for i, match in enumerate(article_matches):
    article_num = match.group(1)
    article_title = match.group(2).strip()
    start = match.start()
    end = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(text)
    article_text = text[start:end].strip()
    part_num = get_part_for_position(start)

    # If an article is unusually long, sub-chunk it but keep the same metadata
    max_len = 1200
    if len(article_text) <= max_len:
        sub_chunks = [article_text]
    else:
        sub_chunks = [article_text[j:j+max_len] for j in range(0, len(article_text), max_len - 150)]

    for sub_idx, sub_text in enumerate(sub_chunks):
        documents.append(Document(
            page_content=sub_text,
            metadata={
                "source": "Constitution of Nepal 2015",
                "part": part_num,
                "article": article_num,
                "article_title": article_title,
                "chunk_id": f"{article_num}-{sub_idx}",
            }
        ))

print(f"Total chunks: {len(documents)}")

print("Building vector DB...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./data/nepal_legal_db_v2"  # new folder — don't overwrite v1 DB yet
)

print(f"Done. Total chunks in DB: {vectordb._collection.count()}")