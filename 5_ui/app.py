"""Streamlit chat app for the Financial Filings RAG assistant.

Run from the project folder:
    streamlit run 5_ui/app.py
"""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3_rag"))
from rag_pipeline import CHROMA_DIR, COMPANIES, METHODS, Retriever, ask, load_llm  # noqa: E402

st.set_page_config(page_title="Financial Filings Assistant", page_icon="📊", layout="centered")

EXAMPLES = [
    "What were Apple's total net sales in fiscal 2026?",
    "What was NVIDIA's total revenue in fiscal year 2026?",
    "What was Amazon's net income in 2025?",
    "What are Tesla's main risk factors?",
]
METHOD_LABELS = {
    "hybrid_rerank": "Hybrid search + reranker (best)",
    "vector_filter": "Embedding search, filtered by company",
    "vector": "Embedding search only (baseline)",
}


@st.cache_resource(show_spinner="Loading the search index and models…")
def load_components():
    """Load the retriever and the LLM once; Streamlit reuses them for every question."""
    return Retriever(), load_llm()


def show_sources(sources):
    with st.expander(f"Sources ({len(sources)} chunks)"):
        for chunk in sources:
            st.markdown(f"**{chunk['chunk_id']}** · {COMPANIES[chunk['ticker']]} · {chunk['section']}")
            preview = chunk["text"][:600] + ("…" if len(chunk["text"]) > 600 else "")
            st.text(preview)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("📄 Reports covered")
    filings_file = ROOT / "data" / "raw" / "filings.json"
    if filings_file.exists():
        filings = json.loads(filings_file.read_text())
        for ticker, info in filings.items():
            year_end = info["url"].rsplit("-", 1)[-1].replace(".htm", "")
            st.markdown(f"**{COMPANIES[ticker]}** ({ticker}): fiscal year ended "
                        f"{year_end[:4]}-{year_end[4:6]}-{year_end[6:]}")

    st.header("⚙️ Settings")
    method = st.selectbox("Search method", METHODS[::-1], format_func=METHOD_LABELS.get)
    k = st.slider("Chunks sent to the AI", min_value=3, max_value=10, value=5)

    st.header("💡 Try an example")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending_question = example

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []

# ------------------------------------------------------------------ main page
st.title("📊 Financial Filings Assistant")
st.caption("Ask questions about the latest annual reports (10-K) of Apple, Microsoft, NVIDIA, Tesla and Amazon. "
           "Answers come only from the reports, with citations.")

if not CHROMA_DIR.exists():
    st.error("Search index not found. Run `1_data_retrieval/data_retrieval.ipynb` and "
             "`2_embedding/embedding.ipynb` first.")
    st.stop()

retriever, llm = load_components()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

question = st.chat_input("Ask about revenue, profit, risks, employees…")
question = question or st.session_state.pop("pending_question", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the reports…"):
                result = ask(retriever, llm, question, k=k, method=method)
            st.markdown(result["answer"])
            show_sources(result["sources"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": result["sources"]})
        except Exception as error:
            message = ("The free Groq API limit was reached. Wait a minute and try again."
                       if "rate" in str(error).lower() or "429" in str(error) else f"Something went wrong: {error}")
            st.warning(message)
