import modal

app = modal.App("nepal-law-assistant")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "unsloth",
        "fastapi",
        "langchain-huggingface",
        "langchain-chroma",
        "langchain-text-splitters",
        "sentence-transformers",
        "chromadb",
        "pymupdf",
        "uvicorn",
        "pydantic",
        "torch",
        "bitsandbytes",
        "accelerate",
        "transformers",
        "peft",
    ])
    .add_local_dir(
        "/home/sagar_kc7/nepal_law/data/nepal_legal_db",
        remote_path="/root/nepal_legal_db"
    )
)

@app.cls(
    image=image,
    gpu="T4",
    timeout=300,
    scaledown_window=120,
)
class NepalLawAPI:

    @modal.enter()
    def load(self):
        from unsloth import FastLanguageModel
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectordb = Chroma(
            persist_directory="/root/nepal_legal_db",
            embedding_function=embeddings
        )

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Llama-3.2-3B-Instruct",
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        print("Model and vector DB loaded.")

    def answer(self, question: str) -> str:
        docs = self.vectordb.similarity_search(question, k=3)
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

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        attention_mask = (inputs != self.tokenizer.pad_token_id).long()

        outputs = self.model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        input_length = inputs.shape[1]
        new_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @modal.fastapi_endpoint(method="POST")
    def ask(self, request: dict) -> dict:
        question = request.get("question", "")
        answer = self.answer(question)
        return {"question": question, "answer": answer}

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        return {"status": "Nepal Law Assistant API is running"}