import os
import streamlit as st

st.set_page_config(page_title="RAG Anti-Hallucination", page_icon="🧠", layout="centered")
st.title("🧠 RAG Hallucination Mitigation")
st.caption("Retrieval-Augmented Transformer with NLI Verification")

# ── Step 1: Download models with live progress bars ──────────────────────────
@st.cache_resource(show_spinner=False)
def download_and_load():
    from model_downloader import download_models_with_progress
    download_models_with_progress()

    st.markdown("### 🔧 Initialising Pipeline")

    prog = st.progress(0, text="📂 Building FAISS index from knowledge base...")
    from src.retriever import Retriever
    retriever = Retriever("data/sample_kb.txt")
    prog.progress(0.5, text="🤖 Loading generator (flan-t5-small)...")

    from src.generator import Generator
    generator = Generator()
    prog.progress(0.85, text="🔍 Loading hallucination verifier (NLI)...")

    from src.verifier import Verifier
    verifier = Verifier()
    prog.progress(1.0, text="✅ All components ready!")

    return retriever, generator, verifier


retriever, generator, verifier = download_and_load()

st.success("✅ System Ready — ask anything below!")
st.divider()

# ── Step 2: Knowledge base preview ───────────────────────────────────────────
with st.expander("📖 Sample Knowledge Base Entries"):
    try:
        with open("data/sample_kb.txt", "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                st.write(f"- {line.strip()[:250]}")
    except Exception as e:
        st.warning(f"Could not read KB: {e}")

# ── Step 3: Query UI ──────────────────────────────────────────────────────────
query = st.text_input(
    "💬 Ask a question:",
    placeholder="e.g. Who was Beyonce's husband?  /  What is the boiling point of water?"
)

if st.button("🔎 Submit", type="primary") and query.strip():

    # Retrieval
    with st.spinner("🔎 Retrieving relevant passages..."):
        contexts = retriever.retrieve(query, top_k=3)

    st.subheader("📚 Retrieved Context")
    for i, ctx in enumerate(contexts):
        st.info(f"**Passage {i+1}:** {ctx[:350]}")

    # Generation
    with st.spinner("🤖 Generating answer with FLAN-T5..."):
        initial_answer = generator.generate_answer(query, contexts)

    st.subheader("📝 Raw Generated Answer")
    st.write(f"> {initial_answer}")

    # Verification
    with st.spinner("🔍 Checking for hallucination..."):
        is_hallucination = verifier.is_hallucination(contexts, initial_answer)

    st.divider()
    st.subheader("🔍 Final Verdict")
    if is_hallucination:
        st.error("⚠️ HALLUCINATION DETECTED — Not supported by context.")
        st.warning("**Safe Final Answer:** I don't have enough information in the knowledge base to answer that.")
    else:
        st.success("✅ VERIFIED — Answer is entailed by the retrieved context.")
        st.success(f"**Final Answer:** {initial_answer}")
