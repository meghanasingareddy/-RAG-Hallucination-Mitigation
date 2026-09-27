# Mitigating Hallucinations in LLM-based Knowledge Bases: A Retrieval-Augmented Transformer Framework

This repository demonstrates a clean, deep learning project that tackles one of the most critical issues with Large Language Models: **Hallucinations**. It implements a Retrieval-Augmented Generation (RAG) framework equipped with an explicit hallucination-checking mechanism using Natural Language Inference (NLI).

## 🚀 Features

- **Knowledge Base Retrieval:** Uses `sentence-transformers` and `FAISS` to embed and retrieve the most relevant context from a knowledge base.
- **Strict Generation Prompting:** Uses a local transformer model (`google/flan-t5-small`) to generate answers based strictly on the retrieved context.
- **NLI Hallucination Verification:** Uses a Cross-Encoder NLI model (`cross-encoder/nli-deberta-v3-small`) to verify if the generated answer is entailed by the retrieved context. If it isn't, the hallucination is caught and flagged!
- **Interactive Web UI:** Built with `Streamlit` to easily interact with the knowledge base, ask questions, and see the retrieval and verification processes in real-time.

## 📁 Project Structure

```
RAGX/
│
├── data/
│   └── sample_kb.txt       # Sample knowledge base about the Solar System
├── src/
│   ├── retriever.py        # Handles chunking, embedding (SentenceTransformers) and vector search (FAISS)
│   ├── generator.py        # Handles the LLM text generation with strict system prompts
│   └── verifier.py         # Handles hallucination detection using Cross-Encoder NLI
├── app.py                  # Streamlit application integrating all components
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🛠️ Tech Stack

- **PyTorch:** Deep Learning framework
- **Hugging Face Transformers:** For the LLM Generator and NLI Verifier models
- **Sentence Transformers:** For dense vector embeddings
- **FAISS:** For efficient similarity search
- **Streamlit:** For the interactive web interface

## 🏃‍♂️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/meghanasingareddy/RAGX.git
cd RAGX
```

### 2. Create a virtual environment (Optional but recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
python -m streamlit run app.py
```

### 5. Access the App
Open your browser and navigate to `http://localhost:8501`.

## 🧪 Testing the System

Once the app is running, try asking questions to see the hallucination mitigation in action:
- **Good Question (In Knowledge Base):** *"What is the Sun made of?"*
  - The system will retrieve the context, generate the answer, and verify it successfully (`✅ VERIFIED ✅`).
- **Out of Context / Deceptive Question:** *"Who is the president of Mars?"*
  - The model will either safely admit it doesn't know, or if it tries to hallucinate an answer, the Verifier will catch the contradiction and flag it (`⚠️ HALLUCINATION DETECTED ⚠️`).

## 📚 Customizing the Knowledge Base

You can easily use your own data by updating the `data/sample_kb.txt` file. Just add your text content line by line or in chunks. The `retriever.py` will automatically re-embed the new knowledge base the next time you start the app.
