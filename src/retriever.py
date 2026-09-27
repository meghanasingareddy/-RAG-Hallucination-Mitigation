import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, kb_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.kb_path = kb_path
        self.embedder = SentenceTransformer(model_name)
        self.documents = []
        self.index = None
        self._build_index()

    def _build_index(self):
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge base file not found at {self.kb_path}")
        
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # simple chunking by line/sentence
        self.documents = [line.strip() for line in lines if line.strip()]
        
        if not self.documents:
            raise ValueError("Knowledge base is empty.")
            
        # create embeddings
        print(f"Embedding {len(self.documents)} documents...")
        embeddings = self.embedder.encode(self.documents, convert_to_numpy=True)
        
        # init FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        print("FAISS index built successfully.")

    def retrieve(self, query: str, top_k: int = 3):
        if self.index is None:
            raise ValueError("Index is not built.")
            
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append(self.documents[idx])
        return results

if __name__ == "__main__":
    # Test
    retriever = Retriever("../data/sample_kb.txt")
    print(retriever.retrieve("What is the Sun made of?"))
