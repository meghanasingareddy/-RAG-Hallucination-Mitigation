from datasets import load_dataset
import os

def build_kb():
    print("Loading SQuAD dataset (fast & small)...")
    # SQuAD is tiny (~30MB), loads instantly, has rich context passages
    dataset = load_dataset("rajpurkar/squad", split="train")

    os.makedirs("data", exist_ok=True)
    kb_path = "data/sample_kb.txt"

    seen = set()
    with open(kb_path, "w", encoding="utf-8", errors="ignore") as f:
        for item in dataset:
            ctx = item["context"].strip()
            # encode/decode to strip any bad characters
            ctx = ctx.encode("ascii", errors="ignore").decode("ascii")
            if ctx and ctx not in seen:
                seen.add(ctx)
                f.write(ctx + "\n")

    print(f"Knowledge base built: {len(seen)} unique passages -> {kb_path}")

if __name__ == "__main__":
    build_kb()
