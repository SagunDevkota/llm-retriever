# Baseline configurations

The exact configuration of every pipeline in the evaluation: the LLM Retriever and its five
baselines. For each published method, it also lists what was taken unchanged from the paper or
reference implementation and what was changed to keep the comparison fair. Results are in
[data_evaluation-deepseek-routing-3.1flash-lite-all/](data_evaluation-deepseek-routing-3.1flash-lite-all/).

| Pipeline | Retrieval | LLM calls per question |
|---|---|---|
| [LLM Retriever](#llm-retriever) | LLM walks the summary tree level by level | 3–4 routing (average) + 1 generation |
| [Naive RAG](#naive-rag) | Dense top-7 | 1 generation |
| [RAPTOR](#raptor) | Dense top-7 over a clustered summary tree (collapsed) | 1 generation |
| [Hybrid + rerank](#hybrid-retrieval--cross-encoder-reranker) | BM25 + dense, RRF, cross-encoder | 1 generation |
| [RankGPT](#rankgpt) | Hybrid pool reranked by an LLM in sliding windows | 7–9 ranking + 1 generation |
| [KET-RAG](#ket-rag) | Graph channel + keyword channel | 1 generation |

## Shared by every pipeline

| Setting | Value |
|---|---|
| Corpora | Django REST framework docs (48 pages) and Playwright docs (127 pages) |
| Questions | 100 per corpus, 200 in total, the same for every pipeline |
| Passages to the generator (k) | 7 (the LLM Retriever returns at most 7) |
| Generator | `google/gemini-3.1-flash-lite` on OpenRouter, temperature 0 |
| Answer prompt | One shared system prompt and template with numbered excerpts; see [PROMPTS.md](PROMPTS.md#c-answer-generation-prompt) |
| Query-time LLM, where a method needs one | `DeepSeek-V4-Flash-0731`, reasoning off, temperature 0 |
| Index-building LLM for the baselines, where needed | `google/gemma-4-31b-it:free` on OpenRouter, temperature 0 |
| Embeddings, for every baseline that uses them | `BAAI/bge-base-en-v1.5`, 768-dim, normalised; queries get the BGE instruction "Represent this sentence for searching relevant passages: " |
| Passages (Naive RAG, Hybrid, RankGPT, KET-RAG) | LangChain `RecursiveCharacterTextSplitter`, 1,000 characters, 200 overlap: 1,050 chunks for DRF, 3,178 for Playwright |
| Recorded per question | `llm_response`, `llm_references` (the texts given to the generator), `llm_cost` (totals plus `routing` and `generation` buckets) |

Only retrieval differs between pipelines. The generator, its prompt and k are the same, so a
difference in answer quality comes from what was retrieved.

## LLM Retriever

The proposed system. Each documentation page becomes a tree of heading sections. Every node gets
two summaries, built bottom-up, and an LLM walks the tree level by level. There are no
embeddings.

| Parameter | Value |
|---|---|
| Tree | One node per `h1`–`h6` section; page-level nodes are the roots |
| Oversized sections | Split at 800 words into sibling chunks with 100 words of overlap |
| Summarizer | `google/gemini-3.1-flash-lite` on OpenRouter, JSON output; one call per node, children before parents |
| Node texts | An evidence text (from the node's own content) and a routing text (from its children's summaries); each is title path + summary + keywords |
| Router | `DeepSeek-V4-Flash-0731` on Hetzner, reasoning off, temperature 0, `max_tokens` 500 |
| Nodes explored per level | At most 7 |
| Maximum depth | 6 levels |
| Evidence kept | 7, ranked by the router's evidence score |
| Routing calls per question | 4.43 on average for DRF, 3.02 for Playwright |
| Context given to the generator | The selected nodes' raw content, with title path and anchor |
| Generator | Shared; `max_tokens` 1,500 |

The summarization and traversal prompts are in [PROMPTS.md](PROMPTS.md).

## Naive RAG

Flat dense retrieval over the shared passages.

| Parameter | Value |
|---|---|
| Passages | The shared 1,000-character chunks |
| Embeddings | BGE, stored in a SQLite vector table (`sqlite-vec`, exact L2 search on normalised vectors, equivalent to cosine) |
| Retrieval | Top 7 by similarity |
| LLM calls per question | 1 (generation) |

## RAPTOR

Sarthi et al., *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*, ICLR 2024,
via LlamaIndex's RAPTOR pack (`llama-index-packs-raptor`).

| Parameter | Value |
|---|---|
| Leaf chunks | LlamaIndex `SentenceSplitter`, 256 tokens, 50 overlap |
| Tree | One global tree over the whole corpus, built level by level until clustering converges to a single root (cap of 6 levels; the DRF tree has 4) |
| Clustering | The pack's defaults: global then local UMAP to 10 dimensions, GMM with BIC model selection, soft assignment at threshold 0.1, 10k-token cluster cap |
| Summaries | `google/gemma-4-31b-it:free`, temperature 0, prompt "Summarize the provided text in at most 150 words, keeping the most important key details.", 4 concurrent |
| Store | Chroma (cosine HNSW) |
| Retrieval | Collapsed tree: top 7 over all leaves and summaries together; 56 candidates (8 × 7) are fetched and the best 7 kept |
| LLM calls per question | 1 (generation) |

**Taken from the paper / pack unchanged:** recursive clustering and summarization, every
clustering hyperparameter, and collapsed-tree retrieval, which the paper found better than tree
traversal.

**Different**
- **Leaf chunk size:** 256 tokens in place of the paper's 100, so leaves are about the size of the 1,000-character naive chunks. Changing it would mix the effect of chunking with the effect of the tree.
- **One global tree:** the whole corpus is clustered in one pass, as the paper describes. Inserting documents in batches would build an independent tree per batch.
- **Summary length:** capped at 150 words, so summaries stay comparable to the leaves and fit BGE's 512-token window. The pack's default prompt sets no limit.
- **Over-fetch:** Chroma's HNSW search used too small a candidate list at k = 7 (`search_ef` defaults to 10), so its top 7 disagreed with a wider search on half the questions. Asking for 8 × 7 neighbours and keeping the best 7 fixes that on an already-built tree.
- **Answer generation:** one call with the shared prompt, not LlamaIndex's query engine, whose own QA and refine templates would replace that prompt.

## Hybrid retrieval + cross-encoder reranker

Not one paper's configuration: a standard hybrid pipeline (BM25 plus dense, fused by reciprocal
rank fusion, then a cross-encoder) built from LangChain components.

| Parameter | Value |
|---|---|
| Lexical retriever | BM25 (`rank_bm25` via LangChain `BM25Retriever`), top 25 |
| Dense retriever | BGE over the shared chunks, top 25 |
| Fusion | LangChain `EnsembleRetriever`, weighted RRF, weights 0.5 / 0.5, c = 60 |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder (LangChain `CrossEncoderReranker`), keeps top 7 |
| LLM calls per question | 1 (generation) |
| Pinned versions | `langchain-classic==1.0.8`, `langchain-community==0.4.2`, `rank-bm25>=0.2.2` |

## RankGPT

Sun et al., *Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking
Agents*, EMNLP 2023; reference code github.com/sunnweiwei/RankGPT.

| Parameter | Value |
|---|---|
| Candidate pool | Hybrid BM25 + dense with RRF (as above), 50 per retriever, de-duplicated to at most 100 |
| Ranking method | Permutation generation with a sliding window |
| Window / step / depth | 20 / 10 / top 100, back to front |
| Passage truncation | 300 words |
| Ranking LLM | `deepseek/deepseek-v4-flash-0731` on OpenRouter, reasoning off, temperature 0, `max_tokens` 300 |
| Kept | Top 7 |
| LLM calls per question | 7–9 ranking windows (8.5 on average for DRF) + 1 generation |

**Taken from the paper / reference code unchanged**
- The ranking prompt, verbatim from the reference code (the paper's Appendix A.5).
- Window 20, step 10, rerank the top 100, back to front.
- 300-word passage truncation.
- Parsing the returned permutation; passages it leaves out keep their original order behind the ranked ones.

**Different**
- **Ranking LLM:** DeepSeek-V4-Flash with reasoning off, in place of gpt-3.5-turbo / gpt-4, so it matches the LLM Retriever's router and token costs are comparable.
- **First-stage retrieval:** the same hybrid BM25 + dense pool as the reranker baseline, in place of BM25 top 100.
- **Last window:** clamped to start at rank 0. The reference loop stops when the next window would start below 0, which leaves the top few positions unranked whenever the depth is not a multiple of the step.
- **Failures:** a reply cut off by the token limit, or one with no ranking in it, raises an error and the question is retried later. The reference code keeps the original order without reporting anything.

## KET-RAG

Huang et al., *KET-RAG: A Cost-Efficient Multi-Granular Indexing Framework for Graph-RAG*,
KDD 2025; official code github.com/waetr/KET-RAG at commit `c632ff4` (a fork of Microsoft
GraphRAG 0.4.1), used unchanged for indexing and context building.

| Parameter | Value |
|---|---|
| Skeleton method | PageRank over the KNN chunk graph (KET-RAG-P) |
| K (KNN edges) | 2 |
| β (skeleton budget) | 0.8 |
| Split size | 150 tokens |
| GraphRAG chunking | 1,200 tokens, overlap 100, one document per page |
| Entity extraction | Prompt-tuned with `--discover-entity-types`; `max_gleanings` 1 |
| Communities | Leiden, `max_cluster_size` 10; reports are built but not used at query time |
| θ (graph channel's share) | 0.4 |
| Context budget | 12,000 tokens (cl100k) |
| Graph channel | GraphRAG local search: `text_unit_prop` 0.5, `community_prop` 0, `top_k_mapped_entities` 10, `top_k_relationships` 10, 300-token text pieces |
| Keyword channel | The official keyword-based chunk selection |
| Index LLM | `google/gemma-4-31b-it:free`, temperature 0, `max_tokens` 4000, 4 concurrent |
| Embeddings | BGE, served locally over an OpenAI-style API |
| Query-time LLM calls | None (only the local query embedding) |

Index sizes: DRF has 132 skeleton units, 1,298 entities, 1,926 relationships and 254
communities. Playwright has 395 units, 2,370 entities, 5,024 relationships and 400 communities.

**Taken from the official code unchanged**
- Indexing: GraphRAG prompt tuning and indexing from the fork, including the KNN chunk graph, PageRank skeleton selection, chunk merging and the keyword index.
- K = 2, β = 0.8, PageRank skeleton, split 150, θ = 0.4, 12k context: the paper's defaults.
- Graph channel: GraphRAG's mixed local-search context with the official parameters, text units split into 300-token pieces.
- Keyword channel: the official selection, including its 2× budget cap.
- Context assembly for the full-context variant: graph context followed by keyword text, with the graph budget equal to 12,000 minus the keyword text's tokens.

**Different**
- **Index LLM:** Gemma 4 31B in place of gpt-4o-mini.
- **Embeddings:** BGE in place of text-embedding-3-small.
- **Input:** one document per page, from the same Markdown files the naive chunks come from. (An earlier build used one document per chunk; GraphRAG's hash ordering and chunk merging then joined unrelated pages into every skeleton unit, so it was rebuilt.)
- **Output, main configuration (7 passages):** the official code passes a ~12k-token context string, not a ranked list. Here 7 passages are selected:
  - round(θ · 7) = 3 from the graph channel, then the keyword channel, without duplicates; either channel fills in if the other runs short;
  - the graph channel's 300-token pieces are ranked by BGE similarity to the question before the 3 are taken (its own order is entity rank, then page order within a unit);
  - each piece is mapped to the naive chunk sharing the most 8-word shingles, with a substring fallback for whitespace-stripped code, so the passages are the same texts every other pipeline returns.
- **Generation:** Gemini with the shared prompt.
- **Rate limits:** GraphRAG 0.4.1 counts every request twice against `requests_per_minute`, so the configured value is double the intended rate.
- **Failures:** a build whose extraction calls failed raises an error and writes no completion marker; a rerun replays the cache and retries only the failures.

### Full-context variant

The `*_ketrag_full*` result files. The generator sees KET-RAG's official ~12k-token context
(about 14k as counted by Gemini), entity and relationship descriptions included, through the
shared prompt. `llm_references` is that context as a list: the entity table, the relationship
table, then every text unit, about 56 items in total.

Its retrieval metrics aren't comparable with a 7-passage budget, so only its answer metrics
(answer accuracy, factual correctness, helpfulness, ROUGE) are scored. It is not part of the
significance tests.

## Notes on reading the results

- **KET-RAG's main row** uses the same 7-passage budget as every other pipeline, so the entity and relationship descriptions it normally puts in its context are left out. The full-context variant shows the method with its own context.
- **Rank order matters:** KET-RAG puts the graph channel first, following the official concatenation order. Context precision depends on rank, so read it together with context recall.
- **Token counts:** KET-RAG budgets use cl100k tokens, as in its code. Gemini reports 12–15% more tokens for the same text.
- **Cost:** the cost comparison prices every pipeline from its token counts at the same per-million rates, because provider billing differs (the LLM Retriever's router runs on Hetzner, RankGPT's on OpenRouter).
- **HippoRAG 2** was configured but never built or run, so it is not part of the evaluation.
