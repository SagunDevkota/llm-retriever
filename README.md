# Hierarchical RAG

An experimental retrieval pipeline that turns a documentation site into a tree of
summarized sections and answers questions by letting an LLM navigate that tree level by
level. There are no embeddings or vector search: every retrieval decision is an LLM
judgment over each node's summary text.

The pipeline runs in five stages: scrape → parse → summarize → store (Postgres) → retrieve.
Code lives in [app/](app/).

## Evaluation

`llm_retriever` is compared against naive top-*k* RAG and RAPTOR on 200 questions
(100 Playwright + 100 Django REST framework). [data_evaluation/](data_evaluation/) holds the
**primary configuration** (DeepSeek routing, Gemini 3.1 Flash Lite for everything else).
Its per-corpus files are identical to the ones in
[docs/data_evaluation-deepseek-routing-3.1flash-lite-all/](docs/data_evaluation-deepseek-routing-3.1flash-lite-all/).

### Significance testing

| File | What it is |
|---|---|
| [ragas_significance_pooled.csv](data_evaluation/ragas_significance_pooled.csv) | **Primary result.** Paired tests over all 200 questions: Wilcoxon signed-rank (exact McNemar for `retrieval_failed`), rank-biserial effect, Hodges–Lehmann CI, Holm-corrected across 14 tests. |
| [ragas_significance_pooled.py](data_evaluation/ragas_significance_pooled.py) | Writes the CSV. |

The script reads `data_evaluation/playwright/` and `data_evaluation/django-rest-framework/`:

```bash
uv run python data_evaluation/ragas_significance_pooled.py
```

Per-corpus inputs: `questions*.json` (questions plus each pipeline's answers and contexts),
`ragas_results_*.xlsx` (RAGAS scores per pipeline) and `evaluation_report_*.xlsx`.
[data_evaluation/generate_evaluation_report.py](data_evaluation/generate_evaluation_report.py)
builds the reports.

### Cost calculation

Per-query token and dollar cost for each pipeline, primary configuration:

| Corpus | Script | Output |
|---|---|---|
| Playwright | [cost_calculation.py](docs/data_evaluation-deepseek-routing-3.1flash-lite-all/playwright/cost_calculation.py) | [token_cost_summary.json](docs/data_evaluation-deepseek-routing-3.1flash-lite-all/playwright/token_cost_summary.json) |
| Django REST framework | [cost_calculation.py](docs/data_evaluation-deepseek-routing-3.1flash-lite-all/django-rest-framework/cost_calculation.py) | [token_cost_summary.json](docs/data_evaluation-deepseek-routing-3.1flash-lite-all/django-rest-framework/token_cost_summary.json) |

Average cost per query, from those summaries:

| Corpus | llm_retriever | RAPTOR | Naive RAG | llm_retriever / mean(RAPTOR, naive) |
|---|---|---|---|---|
| Playwright | $0.002483 | $0.000815 | $0.000803 | 3.07× |
| Django REST framework | $0.001834 | $0.000863 | $0.000817 | 2.18× |

Per-million-token prices for routing and generation are constants at the top of each script.
The script writes `token_cost_summary.json` to the current directory, so run it from inside
its own folder.

### Ablation studies

Each ablation is a full evaluation run on the same questions, kept under [docs/](docs/):

| Ablation | Folder | Varies |
|---|---|---|
| Routing model: Gemini 3.1 Flash Lite | [data_evaluation-3.1flash-lite-all/](docs/data_evaluation-3.1flash-lite-all/) | Gemini 3.1 Flash Lite for all roles, routing included |
| Routing model: Gemini 3.5 Flash | [data_evaluation-3.5flash-routing-3.1flash-lite-all/](docs/data_evaluation-3.5flash-routing-3.1flash-lite-all/) | Gemini 3.5 Flash for routing |
| Routing model: DeepSeek (primary) | [data_evaluation-deepseek-routing-3.1flash-lite-all/](docs/data_evaluation-deepseek-routing-3.1flash-lite-all/) | DeepSeek for routing |
| Chunk size 600 | [data_evaluation-3.5flash-routing-3.1flash-lite-all-600-chunk-size/](docs/data_evaluation-3.5flash-routing-3.1flash-lite-all-600-chunk-size/) | llm_retriever only, 3.5 Flash routing, chunk size 600 |
| Naive RAG dynamic chunking | [data_evaluation-naive-rag-dynamic-chunking/](docs/data_evaluation-naive-rag-dynamic-chunking/) | Naive RAG with fixed chunking vs. dynamic chunking |

To rerun the
significance tests for an ablation, copy its `playwright/` and `django-rest-framework/`
folders into `data_evaluation/`, or change `CORPORA` in the scripts to point at them.
