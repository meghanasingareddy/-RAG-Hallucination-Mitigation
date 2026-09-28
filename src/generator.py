import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Generator:
    def __init__(self, model_name: str = 'google/flan-t5-small'):
        print(f"Loading generator model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        print("Generator model loaded.")

    def generate_answer(self, query: str, context_documents: list) -> str:
        # Combine context documents
        context = " ".join(context_documents)
        
        # Strict prompt to mitigate hallucination at generation stage
        prompt = (
            f"Answer the question using ONLY the provided context. "
            f"If the answer is not in the context, say 'I don't know based on the context'.\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=150)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

if __name__ == "__main__":
    # Test
    gen = Generator()
    ctx = ["The Solar System formed 4.6 billion years ago."]
    ans = gen.generate_answer("How old is the Solar system?", ctx)
    print(f"Ans: {ans}")
