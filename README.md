# Financial Filings RAG

A chatbot that answers questions about the latest annual reports (10-K filings) of **Apple, Microsoft, NVIDIA, Tesla and Amazon**, citing the exact part of the report each answer comes from, and an evaluation that measures how often it is right.

| | Baseline (embedding search) | **Hybrid search + reranker** |
|---|---|---|
| Retrieval: right chunk in top 5 | 55.6% | **77.8%** |
| Answers correct (27 in-scope questions) | 55.6% | **77.8%** |
| Out-of-scope questions refused (6) | 100% | **100%** |
| **Overall accuracy (33 questions)** | 63.6% | **81.8%** |

**Stack:** Python · LangChain · ChromaDB · sentence-transformers (embeddings + cross-encoder reranker) · BM25 · Groq (`gpt-oss-120b`) · Streamlit

---

## How it works

```
SEC website ─► download 10-Ks ─► clean HTML ─► split into sections ─► chunks (LangChain)
                                                                          │
                                                       embeddings ─► Chroma vector database
                                                                          │
question ─► embedding search + keyword search (BM25) ─► reranker picks the best 5 chunks
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

## Evaluation

The test set (`4_evaluation/eval_set.csv`) has 33 questions:
- **16 generated** by an LLM from random chunks, then reviewed by hand (trivia and unanswerable-to-grade questions removed),
- **11 hand-written** headline questions (each company's revenue and net income, employees), with answers checked in the reports, including a trap: Amazon's tables list years oldest first,
- **6 out-of-scope** questions the reports can't answer (e.g. *"Who is the CEO of Google?"*).

**Findings**
- Hybrid search + reranker improved answer accuracy by **22 points** over plain embedding search.
- Filtering by the company named in the question changed nothing: embedding search already finds the right company; the hard part is the right chunk.
- The chatbot refused **all** out-of-scope questions, and most remaining failures are safe refusals rather than wrong answers (1 wrong number in 27 in-scope answers).

**Limitations:** a small test set, mostly LLM-generated questions, and an LLM judge. Remaining failures include Microsoft's headline figures, where retrieval misses the income-statement chunk.

**Next steps:** a larger test set, sending more chunks to the LLM (hit rate@10 is 89%), table-aware chunking that keeps each table's header with its rows, and deployment.

## How to run

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project folder:
```
GROQ_API_KEY=your_groq_key                      # free at https://console.groq.com
SEC_USER_AGENT=Your Name your_email@example.com  # the SEC requires a name and email for downloads
```

Run the notebooks in order (`1_data_retrieval` → `2_embedding`; `3_rag` and `4_evaluation` are optional), then start the app:
```bash
streamlit run 5_ui/app.py
```
The app opens at http://localhost:8501.

**Groq free tier:** about 8,000 tokens per minute (roughly 3 answers per minute). The evaluation saves its results to CSV files, so re-running it doesn't call the API again.

## Data

Annual reports from the [SEC EDGAR](https://www.sec.gov/edgar) database (public domain), downloaded within the SEC's [fair access](https://www.sec.gov/os/accessing-edgar-data) limits:

| Company | Fiscal year ended |
|---|---|
| Apple | September 26, 2026 |
| Microsoft | June 30, 2026 |
| NVIDIA | January 25, 2026 |
| Tesla | December 31, 2025 |
| Amazon | December 31, 2025 |
