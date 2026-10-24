# LLM Retriever

An experimental retrieval pipeline that turns a documentation site into a tree of
summarized sections and answers questions by letting an LLM navigate that tree level by
level. There are no embeddings or vector search: every retrieval decision is an LLM
judgment over each node's summary text.

The pipeline runs in five stages: scrape → parse → summarize → store (Postgres) → retrieve.
Code lives in [app/](app/).

## Evaluation

`llm_retriever` is compared against five baselines on 200 questions (100 Playwright + 100
Django REST framework): naive top-*k* RAG, RAPTOR, hybrid retrieval with reranking, KETRAG
and RankGPT. Every pipeline answers with the same model and the same
[answer prompt](docs/PROMPTS.md#c-answer-generation-prompt), so differences come from
retrieval.

**All results for the primary configuration** (DeepSeek routing, Gemini 3.1 Flash Lite for
everything else) **are in
[docs/data_evaluation-deepseek-routing-3.1flash-lite-all/](docs/data_evaluation-deepseek-routing-3.1flash-lite-all/).**
[data_evaluation/](data_evaluation/) holds the analysis scripts and a working copy of the
per-corpus data they read.

The prompts used for summarization, traversal and answer generation are in
[docs/PROMPTS.md](docs/PROMPTS.md).

### Result files

In [docs/data_evaluation-deepseek-routing-3.1flash-lite-all/](docs/data_evaluation-deepseek-routing-3.1flash-lite-all/):

| File | What it is |
|---|---|
| [ragas_significance_pooled.csv](docs/data_evaluation-deepseek-routing-3.1flash-lite-all/ragas_significance_pooled.csv) | **Primary result.** `llm_retriever` vs each baseline over all 200 questions: Wilcoxon signed-rank (exact McNemar for `retrieval_failed`), rank-biserial effect size (`effect` column), Hodges–Lehmann CI and Holm-corrected p-values across all 35 tests (7 outcomes × 5 baselines). |
| [bootstrap_ci_summary.csv](docs/data_evaluation-deepseek-routing-3.1flash-lite-all/bootstrap_ci_summary.csv) | 95% bootstrap CI on the mean paired difference per metric and baseline (10,000 resamples, uncorrected). |
| `<corpus>/evaluation_report_*.xlsx` | Average scores, cost and tokens per question type and pipeline, with charts. |
| `<corpus>/token_cost_summary.json` | Average tokens and dollar cost per query, ranked against the cheapest pipeline. |
| `<corpus>/questions_*.json` | Questions plus each pipeline's answer, retrieved contexts and token usage. |
| `<corpus>/ragas_results_*.xlsx` | RAGAS scores per question for each pipeline. |

`<corpus>` is `playwright/` or `django-rest-framework/`. The `*_ketrag_full*` files are a
KETRAG variant with the full graph context (about 56 chunks per question instead of 7); the
comparisons above don't use them.

### Scripts

| Script | Writes |
|---|---|
| [ragas_significance_pooled.py](data_evaluation/ragas_significance_pooled.py) | Pooled paired tests, plus a per-corpus robustness check |
| [ragas_significance_by_category.py](data_evaluation/ragas_significance_by_category.py) | The same tests per question type |
| [report_bootstrap_ci.py](data_evaluation/report_bootstrap_ci.py) | Bootstrap CIs |
| [generate_evaluation_report.py](data_evaluation/generate_evaluation_report.py) | `evaluation_report_*.xlsx` with charts |

Each takes all baselines from one list near its top and pairs questions on corpus and question
text. The significance scripts read `data_evaluation/playwright/` and
`data_evaluation/django-rest-framework/`:

```bash
uv run python data_evaluation/ragas_significance_pooled.py
uv run python data_evaluation/report_bootstrap_ci.py docs/data_evaluation-deepseek-routing-3.1flash-lite-all
uv run python data_evaluation/generate_evaluation_report.py docs/data_evaluation-deepseek-routing-3.1flash-lite-all/playwright docs/data_evaluation-deepseek-routing-3.1flash-lite-all/django-rest-framework
```

### Cost calculation

Each corpus folder has a `cost_calculation.py` that writes `token_cost_summary.json`. Cost is
priced from token counts at fixed per-million-token rates (constants at the top of the
script), not from provider billing, so every pipeline is priced the same way. The script
writes to the current directory, so run it from inside its own folder.

Average cost per query, as a multiple of the cheapest pipeline:

| Pipeline | Playwright | Django REST framework |
|---|---|---|
| KETRAG | $0.000791 (1.00×) | $0.000821 (1.01×) |
| Naive RAG | $0.000803 (1.02×) | $0.000817 (1.00×) |
| RAPTOR | $0.000815 (1.03×) | $0.000863 (1.06×) |
| Hybrid + rerank | $0.000834 (1.05×) | $0.000840 (1.03×) |
| RankGPT | $0.002218 (2.81×) | $0.002177 (2.67×) |
| llm_retriever | $0.002483 (3.14×) | $0.001834 (2.25×) |

The summaries also rank pipelines by average tokens per query. `llm_retriever` and RankGPT use
11–18× the tokens of the leanest pipeline, but most of those are routing or reranking input,
priced lower than generation input.

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
