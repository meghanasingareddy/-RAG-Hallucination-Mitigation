import os
import streamlit as st
from huggingface_hub import list_repo_files, hf_hub_download


MODELS = {
    "Generator (flan-t5-small)": "google/flan-t5-small",
    "Verifier (nli-deberta-v3-small)": "cross-encoder/nli-deberta-v3-small",
    "Embedder (all-MiniLM-L6-v2)": "sentence-transformers/all-MiniLM-L6-v2",
}

# File extensions to skip (large/unnecessary files)
SKIP_EXTS = (".h5", ".ot", ".msgpack", ".arrow", ".parquet")


def download_models_with_progress():
    """Pre-download all models showing live progress bars in Streamlit."""
    st.markdown("### 📦 Downloading Models")
    all_done = True

    for label, repo_id in MODELS.items():
        try:
            files = [
                f for f in list_repo_files(repo_id)
                if not f.endswith(SKIP_EXTS) and not f.startswith(".")
            ]
        except Exception as e:
            st.warning(f"Could not list files for {label}: {e}")
            continue

        total = len(files)
        bar = st.progress(0, text=f"⬇️ {label} — starting...")

        for i, filename in enumerate(files):
            try:
                hf_hub_download(repo_id=repo_id, filename=filename)
            except Exception:
                pass  # already cached or optional file
            pct = (i + 1) / total
            bar.progress(pct, text=f"⬇️ **{label}** — `{filename}` ({i+1}/{total})")

        bar.progress(1.0, text=f"✅ **{label}** — fully downloaded!")

    return all_done
