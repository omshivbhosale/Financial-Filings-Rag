"""RAG pipeline: find the most relevant 10-K chunks and answer questions with a Groq LLM.

Used by rag_demo.ipynb, the evaluation notebook and the Streamlit app, so every part
of the project answers questions in exactly the same way.

Three search methods (compared in 4_evaluation/evaluation.ipynb):
    "vector"         embedding search over all chunks
    "vector_filter"  embedding search, limited to the companies named in the question
    "hybrid_rerank"  embedding + keyword (BM25) candidates for the named companies,
                     re-scored by a cross-encoder reranker
"""
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CHUNKS_FILE = ROOT / "data" / "chunks.csv"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "filings"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
NOT_FOUND = "I couldn't find that in the reports."
METHODS = ["vector", "vector_filter", "hybrid_rerank"]

COMPANIES = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "TSLA": "Tesla", "AMZN": "Amazon"}

# Words that identify each company in a question
ALIASES = {
    "apple": "AAPL", "aapl": "AAPL", "iphone": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT", "azure": "MSFT",
    "nvidia": "NVDA", "nvda": "NVDA",
    "tesla": "TSLA", "tsla": "TSLA",
    "amazon": "AMZN", "amzn": "AMZN", "aws": "AMZN",
}

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about company annual reports (10-K filings) using ONLY the context below.\n"
     f"- If the answer is not in the context, say exactly: \"{NOT_FOUND}\"\n"
     "- Cite the chunk IDs you used in square brackets, e.g. [AAPL_139].\n"
     "- Numbers in tables are in millions of US dollars unless stated otherwise.\n"
     "- Each company's fiscal year ends at a different time; use the years exactly as written in the context.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def tokenize(text):
    """Lowercase words and numbers, used by the keyword (BM25) search."""
    return re.findall(r"[a-z0-9]+", text.lower())


# Everyday words -> the wording 10-K reports use. Reports never say "profit", they say "net income",
# so a casual question like "Apple profit" would otherwise miss the income statement.
SYNONYMS = {
    "profit": "net income", "profits": "net income", "earnings": "net income",
    "sales": "net sales revenue",
    "staff": "employees", "workers": "employees", "headcount": "employees",
    "debt": "term debt borrowings",
}


def expand_query(question):
    """Add report wording for everyday finance words, e.g. "Apple profit" -> "Apple profit (net income)".
    Skipped when the question already uses the report's wording (e.g. "net sales")."""
    q = question.lower()
    extra = [terms for word, terms in SYNONYMS.items()
             if re.search(rf"\b{word}\b", q) and " ".join(terms.split()[:2]) not in q]
    return question + (" (" + "; ".join(dict.fromkeys(extra)) + ")" if extra else "")


def detect_companies(question):
    """Return the tickers of companies named in the question, e.g. ["NVDA"]."""
    return sorted({ALIASES[w] for w in tokenize(question) if w in ALIASES})


def build_index(embeddings, chunks):
    """Create the Chroma index from chunks.csv, the same way as 2_embedding/embedding.ipynb.
    Used when the index folder is missing, e.g. on a fresh server (chroma_db/ is not in the repo)."""
    documents = [Document(page_content=row.text,
                          metadata={"ticker": row.ticker, "section": row.section, "chunk_id": row.chunk_id})
                 for row in chunks.itertuples()]
    return Chroma.from_documents(documents, embeddings, ids=chunks["chunk_id"].tolist(),
                                 persist_directory=str(CHROMA_DIR), collection_name=COLLECTION)


class Retriever:
    """Holds everything needed for search: the Chroma index, keyword indexes and the reranker."""

    def __init__(self):
        # Both models are small, so they run on the CPU; this avoids running out of GPU memory on Macs
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
        self.chunks = pd.read_csv(CHUNKS_FILE).set_index("chunk_id", drop=False)
        if CHROMA_DIR.exists():
            self.vector_db = Chroma(persist_directory=str(CHROMA_DIR), collection_name=COLLECTION,
                                    embedding_function=embeddings)
        else:
            self.vector_db = build_index(embeddings, self.chunks.reset_index(drop=True))

        # One keyword index for all chunks, plus one per company
        self.keyword_index = {None: self._build_keyword_index(self.chunks)}
        for ticker, group in self.chunks.groupby("ticker"):
            self.keyword_index[ticker] = self._build_keyword_index(group)

        self.reranker = CrossEncoder(RERANKER_MODEL, device="cpu")

    @staticmethod
    def _build_keyword_index(chunks):
        return BM25Okapi([tokenize(t) for t in chunks["text"]]), chunks["chunk_id"].tolist()

    def _vector_ids(self, question, n, ticker=None):
        docs = self.vector_db.similarity_search(question, k=n, filter={"ticker": ticker} if ticker else None)
        return [d.metadata["chunk_id"] for d in docs]

    def _keyword_ids(self, question, n, ticker=None):
        index, ids = self.keyword_index[ticker]
        scores = index.get_scores(tokenize(question))
        return [ids[i] for i in scores.argsort()[::-1][:n]]

    def search(self, question, k=5, method="hybrid_rerank", n_candidates=40):
        """Return the chunk_ids of the k most relevant chunks (k per company if several are named)."""
        tickers = detect_companies(question) if method != "vector" else []
        if method == "hybrid_rerank":
            question = expand_query(question)
        groups = tickers or [None]

        if method in ("vector", "vector_filter"):
            ids = []
            for ticker in groups:
                ids += self._vector_ids(question, k, ticker)
            return ids

        # hybrid_rerank: broad candidate list from both searches, then the reranker picks the best
        ids = []
        for ticker in groups:
            candidates = self._vector_ids(question, n_candidates, ticker) + self._keyword_ids(question, n_candidates, ticker)
            candidates = list(dict.fromkeys(candidates))                     # remove duplicates, keep order
            scores = self.reranker.predict([(question, self.chunks.at[c, "text"]) for c in candidates])
            ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)]
            ids += ranked[:k]
        return ids

    def get_chunks(self, chunk_ids):
        return self.chunks.loc[chunk_ids].to_dict("records")


def load_llm(model=LLM_MODEL):
    """Connect to the Groq LLM. max_retries waits and retries when the free-tier rate limit is hit."""
    return ChatGroq(model=model, temperature=0, reasoning_effort="low", max_retries=8)


def format_context(chunks):
    """Join the chunks into one block of text, each with a header showing its ID, company and section."""
    return "\n\n".join(f"[{c['chunk_id']}] ({c['ticker']}, {c['section']})\n{c['text']}" for c in chunks)


def generate_answer(llm, question, chunks):
    """Ask the LLM to answer from the retrieved chunks only."""
    reply = (PROMPT | llm).invoke({"context": format_context(chunks), "question": question})
    return reply.content.replace("【", "[").replace("】", "]")   # unify citation brackets


def ask(retriever, llm, question, k=10, method="hybrid_rerank"):
    """Full RAG: retrieve chunks, then generate an answer. Returns the answer and the chunks used."""
    chunks = retriever.get_chunks(retriever.search(question, k=k, method=method))
    return {"answer": generate_answer(llm, question, chunks), "sources": chunks}
