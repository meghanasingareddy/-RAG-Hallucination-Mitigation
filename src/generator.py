import torch
from transformers import pipeline

class Generator:
    def __init__(self, model_name: str = 'google/flan-t5-small'):
        print(f"Loading generator model: {model_name}...")
        self.pipeline = pipeline(
            "text2text-generation", 
            model=model_name, 
            device=0 if torch.cuda.is_available() else -1
        )
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
        
        result = self.pipeline(prompt, max_length=150, num_return_sequences=1)
        answer = result[0]['generated_text']
        return answer.strip()

if __name__ == "__main__":
    # Test
    gen = Generator()
    ctx = ["The Solar System formed 4.6 billion years ago."]
    ans = gen.generate_answer("How old is the Solar system?", ctx)
    print(f"Ans: {ans}")
