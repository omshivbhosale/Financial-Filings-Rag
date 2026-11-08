# Financial Filings RAG

A chatbot that answers questions about the latest annual reports (10-K filings) of **Apple, Microsoft, NVIDIA, Tesla and Amazon**, citing the exact part of the report each answer comes from, and an evaluation that measures how often it is right.

**🔗 Live demo:** _coming soon_

| | Baseline (embedding search, 5 chunks) | Hybrid search + reranker, 5 chunks | **Hybrid search + reranker, 10 chunks** |
|---|---|---|---|
| Retrieval: right chunk retrieved | 55.6% (top 5) | 77.8% (top 5) | **88.9%** (top 10) |
| Answers correct (27 in-scope questions) | 55.6% | 77.8% | **85.2%** |
| Out-of-scope questions refused (6) | 100% | 100% | **100%** |
| **Overall accuracy (33 questions)** | 63.6% | 81.8% | **87.9%** |

Each change was measured on the same test set, one change at a time.

**Stack:** Python · LangChain · ChromaDB · sentence-transformers (embeddings + cross-encoder reranker) · BM25 · Groq (`gpt-oss-120b`) · Streamlit

---

## How it works

```
SEC website ─► download 10-Ks ─► clean HTML ─► split into sections ─► chunks (LangChain)
                                                                          │
                                                       embeddings ─► Chroma vector database
                                                                          │
question ─► embedding search + keyword search (BM25) ─► reranker picks the best 10 chunks
                                                                          │
                                         Groq LLM answers ONLY from those chunks, with citations
```

| Folder | What it does |
|---|---|
| [`1_data_retrieval/`](1_data_retrieval/data_retrieval.ipynb) | Downloads the 5 reports from the SEC, removes hidden XBRL data, keeps table rows on one line, splits each report into its standard 10-K sections, and cuts them into 1,633 labeled chunks |
| [`2_embedding/`](2_embedding/embedding.ipynb) | Turns chunks into embeddings (`all-MiniLM-L6-v2`, runs locally) and builds the Chroma search index |
| [`3_rag/`](3_rag/) | `rag_pipeline.py`: retrieval (three methods) and answer generation, shared by every other part. `rag_demo.ipynb`: example questions |
| [`4_evaluation/`](4_evaluation/evaluation.ipynb) | Test set, retrieval metrics (hit rate, MRR), answer grading with an LLM judge, failure analysis |
| [`5_ui/`](5_ui/app.py) | Streamlit chat app with sources and a search-method switch |

## Key decisions

- **Answers only from the reports.** The prompt forbids outside knowledge, requires chunk citations (e.g. `[AAPL_139]`), and gives the model a safe way out: *"I couldn't find that in the reports."*
- **Section-aware chunks.** Every chunk is labeled with its company and 10-K section (e.g. *Item 8. Financial Statements*), so answers can be traced back. Headings differ by company (tables, capitals, split lines), so sections are detected by item number and given standard names; the table of contents is skipped by position.
- **Tables kept intact.** Each table row becomes one line (`Total net sales | 478,834 | 15 | 416,161`), so labels stay next to their numbers.
- **Hybrid search + reranker.** Embedding search is weak with number-heavy tables: NVIDIA's revenue table ranked around #30. Combining embedding and keyword (BM25) candidates, then rescoring them with a cross-encoder, raised retrieval hit rate from 56% to 78%.
- **10 chunks instead of 5.** Retrieval already found the answer in the top 10 for 89% of questions but only 78% in the top 5, so answers were being cut off before reaching the LLM. Sending 10 chunks fixed 2 questions (including Microsoft's total revenue), broke none, and kept every out-of-scope refusal, at the cost of about twice the input tokens per answer.
- **Deployable as is.** The search index isn't stored in the repo: on a fresh server the app rebuilds it from `data/chunks.csv` in about 20 seconds, with the same retrieval scores. On Linux, `requirements.txt` installs the CPU-only PyTorch build (about 210 MB instead of 2.5 GB of GPU libraries).

## Evaluation

The test set (`4_evaluation/eval_set.csv`) has 33 questions:
- **16 generated** by an LLM from random chunks, then reviewed by hand (trivia and unanswerable-to-grade questions removed),
- **11 hand-written** headline questions (each company's revenue and net income, employees), with answers checked in the reports, including a trap: Amazon's tables list years oldest first,
- **6 out-of-scope** questions the reports can't answer (e.g. *"Who is the CEO of Google?"*).

**Findings**
- Hybrid search + reranker improved answer accuracy by **22 points** over plain embedding search.
- Sending 10 chunks instead of 5 added another **6 points** (81.8% → 87.9%) by reducing wrong refusals from 18.5% to 11.1%.
- Filtering by the company named in the question changed nothing: embedding search already finds the right company; the hard part is the right chunk.
- The chatbot refused **all** out-of-scope questions, and most remaining failures are safe refusals rather than wrong answers (1 wrong number in 27 in-scope answers).

**Limitations:** a small test set, mostly LLM-generated questions, and an LLM judge. Remaining failures: Microsoft's net income and Amazon's employee count (refused), and one NVIDIA tax question (a different tax line was reported).

**Next steps:** a larger, mostly hand-written test set, and table-aware chunking that keeps each table's header with its rows (for the remaining income-statement failures).

## How to run

Python 3.14:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project folder:
```
GROQ_API_KEY=your_groq_key                      # free at https://console.groq.com
SEC_USER_AGENT=Your Name your_email@example.com  # the SEC requires a name and email for downloads
```

Start the app (the processed chunks are included, so the notebooks are optional; the first start builds the search index):
```bash
streamlit run 5_ui/app.py
```
The app opens at http://localhost:8501. To rebuild everything from the SEC, run the notebooks in order (`1_data_retrieval` → `2_embedding`; `3_rag` and `4_evaluation` are optional).

**Deploying on Streamlit Community Cloud:** main file `5_ui/app.py`, Python 3.14, and add `GROQ_API_KEY` under the app's secrets.

**Groq free tier:** about 8,000 tokens per minute (roughly 2 answers per minute with 10 chunks). The evaluation saves its results to CSV files, so re-running it doesn't call the API again.

## Data

Annual reports from the [SEC EDGAR](https://www.sec.gov/edgar) database (public domain), downloaded within the SEC's [fair access](https://www.sec.gov/os/accessing-edgar-data) limits:

| Company | Fiscal year ended |
|---|---|
| Apple | September 26, 2026 |
| Microsoft | June 30, 2026 |
| NVIDIA | January 25, 2026 |
| Tesla | December 31, 2025 |
| Amazon | December 31, 2025 |
