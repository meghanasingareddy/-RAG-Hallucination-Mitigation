import streamlit as st
import os
from src.retriever import Retriever
from src.generator import Generator
from src.verifier import Verifier

st.set_page_config(page_title="RAG Anti-Hallucination", layout="centered")

@st.cache_resource
def load_models():
    kb_path = "data/sample_kb.txt"
    if not os.path.exists(kb_path):
        st.error("Knowledge base not found!")
        return None, None, None
        
    retriever = Retriever(kb_path)
    generator = Generator()
    verifier = Verifier()
    return retriever, generator, verifier

st.title("Mitigating Hallucinations in RAG")
st.markdown("This simple clean deep learning project demonstrates a RAG framework with an explicit hallucination-checking mechanism using NLI.")

retriever, generator, verifier = load_models()

if retriever and generator and verifier:
    with st.expander("View Knowledge Base"):
        with open("data/sample_kb.txt", "r", encoding="utf-8") as f:
            st.text(f.read())

    query = st.text_input("Ask a question based on the Knowledge Base:")

    if st.button("Submit") and query:
        with st.spinner("Retrieving context..."):
            contexts = retriever.retrieve(query, top_k=3)
            
        st.subheader("Retrieved Context")
        for i, ctx in enumerate(contexts):
            st.write(f"- {ctx}")
            
        with st.spinner("Generating answer..."):
            initial_answer = generator.generate_answer(query, contexts)
            
        st.subheader("Generated Answer")
        st.write(initial_answer)
        
        with st.spinner("Checking for hallucination..."):
            is_hallucination = verifier.is_hallucination(contexts, initial_answer)
            
        st.subheader("Hallucination Verification")
        if is_hallucination:
            st.error("⚠️ HALLUCINATION DETECTED ⚠️\nThe generated answer is not fully supported by the retrieved context.")
            st.warning("Final Answer: I don't have enough information to answer that based on the context.")
        else:
            st.success("✅ VERIFIED ✅\nThe answer is entailed by the context.")
            st.success(f"Final Answer: {initial_answer}")
