from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="./data/nepal_legal_db",
    embedding_function=embeddings
)

def answer_legal_question(question: str) -> str:
    # import here so model is already loaded by startup
    from app.model import model, tokenizer

    docs = vectordb.similarity_search(question, k=3)
    context = "\n\n".join([
        f"[Chunk {doc.metadata['chunk_id']}]\n{doc.page_content}"
        for doc in docs
    ])

    messages = [
        {
            "role": "system",
            "content": "You are a Nepali legal assistant. Answer in 2-3 sentences using ONLY the provided context. Always cite the article number. Never add extra explanation."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    attention_mask = (inputs != tokenizer.pad_token_id).long()

    outputs = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        max_new_tokens=150,
        temperature=0.1,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    input_length = inputs.shape[1]
    new_tokens = outputs[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

