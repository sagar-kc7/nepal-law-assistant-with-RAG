from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load cleaned text directly
print("Loading cleaned text...")
with open("./data/consitution_cleaned.txt", "r", encoding="utf-8") as f:
    cleaned = f.read()

print(f"Total characters: {len(cleaned)}")

# chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\nPart-", "\n\nSchedule-", "\n\nArticle ", "\n\n", "\n"]
)
chunks = splitter.split_text(cleaned)
print(f"Total chunks: {len(chunks)}")

# convert to documents
documents = []
for i, chunk in enumerate(chunks):
    documents.append(Document(
        page_content=chunk,
        metadata={"source": "Constitution of Nepal 2015", "chunk_id": i}
    ))

# build vector DB
print("Building vector DB...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./data/nepal_legal_db"
)

print(f"Done. Total chunks in DB: {vectordb._collection.count()}")