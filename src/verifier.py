import torch
from transformers import pipeline

class Verifier:
    def __init__(self, model_name: str = 'cross-encoder/nli-deberta-v3-small'):
        print(f"Loading verifier model: {model_name}...")
        # We use zero-shot-classification or explicit NLI
        self.pipeline = pipeline(
            "text-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        print("Verifier model loaded.")

    def is_hallucination(self, context_documents: list, generated_answer: str) -> bool:
        """
        Returns True if the generated answer is a hallucination (not entailed by the context).
        """
        # If the model safely admitted it doesn't know, it's not a hallucination.
        if "i don't know" in generated_answer.lower():
            return False

        context = " ".join(context_documents)
        
        # NLI typically takes a pair: (premise, hypothesis)
        # We want to check if context (premise) entails the answer (hypothesis)
        result = self.pipeline({"text": context, "text_pair": generated_answer})
        
        # 'cross-encoder/nli-deberta-v3-small' labels: 'contradiction', 'entailment', 'neutral'
        # Often mapped to LABEL_0, LABEL_1, LABEL_2. We'll find the highest score.
        label = result['label'].lower()
        
        # Some models use LABEL_0 (contradiction), LABEL_1 (entailment), LABEL_2 (neutral)
        # Let's map explicitly based on standard deberta NLI
        is_entailed = False
        if label == 'entailment' or label == 'label_1': 
            is_entailed = True
        elif label == 'neutral' or label == 'label_2':
            # Strict mode: if it's neutral (context doesn't confirm or deny), it's a hallucination
            is_entailed = False 
            
        return not is_entailed

if __name__ == "__main__":
    # Test
    verifier = Verifier()
    ctx = ["The Solar System formed 4.6 billion years ago."]
    ans_good = "The Solar System is 4.6 billion years old."
    ans_bad = "The Solar System is 10 billion years old."
    
    print(f"Good answer hallucination check: {verifier.is_hallucination(ctx, ans_good)}") # Should be False
    print(f"Bad answer hallucination check: {verifier.is_hallucination(ctx, ans_bad)}")   # Should be True
