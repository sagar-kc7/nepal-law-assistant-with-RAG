import unsloth
from unsloth import FastLanguageModel

model, tokenizer = None, None

def load_model():
    global model, tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",  # 3B fits in 4GB
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded.")