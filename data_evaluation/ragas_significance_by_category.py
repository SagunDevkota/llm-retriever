"""
Per-category significance testing: does hierarchical retrieval pay off more on
some question types than others?
=============================================================================

Motivation: the pooled analysis (ragas_significance.py) finds a large, robust
context_precision win for llm_retriever but no measurable answer-quality gain.
One explanation is dilution -- if the advantage is concentrated in question types
where a naive top-k retrieval is likely to miss (multi_hop, debugging), then
averaging over all 200 questions, most of which are simple lookups, will wash it
out. This script tests that directly.

Design: the 200 pooled questions are evenly split across 4 categories
(50 each: 25 from playwright + 25 from DRF), so each category is a balanced
paired subsample. Pairing is preserved within category -- all three pipelines
answer the same 50 questions.

Every RAGAS metric in the result files is tested per category, EXCEPT the two
rouge_score columns. ROUGE is excluded deliberately: it is n-gram overlap against
a reference answer, a weak proxy for answer quality, and it is mechanically
biased here -- llm_retriever returns more context, which depresses rouge
precision regardless of answer quality. State this exclusion in the paper's
methods rather than dropping it silently.

Two derived outcomes are added:
  - answer_quality    -- composite of the 3 answer metrics on a common 0-1 scale,
                         included because the individual answer metrics are too
                         tied to have resolution (88% ties on helpfulness)
  - retrieval_failed  -- binary (context_precision == 0), tested with an exact
                         McNemar; the pooled analysis showed the effect lives here

Holm correction is applied WITHIN each outcome family (4 categories x 2
comparisons = 8 tests). Correcting within outcome is the standard choice when
outcomes answer distinct questions; state it in the paper. Because 7 outcomes x
8 tests = 56 tests is substantial multiplicity, a `p_holm_global` column applies
Holm across ALL tests as a conservative sensitivity check -- report both, since
some findings are sensitive to which one you use.

!! INTERPRETIVE WARNING !!
This is a subgroup analysis with n=50 per cell. Subgroup analyses are the
classic route to false positives: with 24 tests over 4 slices, something will
look significant by chance. Two rules for reporting it honestly:
  - Treat the category split as PRE-SPECIFIED (it is -- the categories were
    assigned at question-generation time, not chosen after seeing results).
  - Report the direction/effect size across ALL categories, not just the
    winners. A real effect shows a consistent gradient across question types;
    a false positive shows one category spiking with the rest flat.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon

BASE_DIR = Path(__file__).resolve().parent
CORPORA = {"playwright": BASE_DIR / "playwright", "drf": BASE_DIR / "django-rest-framework"}
RETRIEVERS = ["naive_rag", "llm_retriever", "raptor"]
COMPARISONS = [("llm_retriever", "naive_rag"), ("llm_retriever", "raptor")]
# Ordered easiest -> hardest, which is the gradient the hypothesis predicts.
CATEGORIES = ["straightforward", "intent_driven", "multi_hop", "debugging"]

# All RAGAS metrics in the files except rouge_score(precision|recall) -- see the
# module docstring for why ROUGE is excluded.
FILE_METRICS = [
    "answer_accuracy",
    "factual_correctness",
    "helpfulness",
    "context_precision",
    "context_recall",
]
DERIVED = ["answer_quality", "retrieval_failed"]
OUTCOMES = FILE_METRICS + DERIVED
ALPHA = 0.05
RESULTS_CSV = BASE_DIR / "ragas_significance_by_category.csv"


def load(retriever):
    """Merge both corpora for one pipeline, attaching each question's category."""
    parts = []
    for corpus, folder in CORPORA.items():
        df = pd.read_excel(folder / f"ragas_results_{retriever}.xlsx")
        cats = {q["user_input"]: q["category"] for q in json.loads((folder / f"questions_{retriever}.json").read_text())}
        df.insert(0, "corpus", corpus)
        df.insert(1, "category", df["user_input"].map(cats))
        parts.append(df)
    merged = pd.concat(parts, ignore_index=True)
    if merged["category"].isna().any():
        raise ValueError(f"{retriever}: some questions have no category (join by user_input failed)")
    # Composite answer quality: 3 metrics mapped onto a common 0-1 scale.
    # Equal weighting is a deliberate pre-specified choice, not tuned.
    merged["answer_quality"] = (
        merged["answer_accuracy"] + merged["factual_correctness"] + merged["helpfulness"] / 5
    ) / 3
    merged["retrieval_failed"] = (merged["context_precision"] == 0).astype(int)
    return merged.sort_values(["corpus", "user_input"], kind="stable").reset_index(drop=True)


dfs = {r: load(r) for r in RETRIEVERS}

reference = dfs["llm_retriever"][["corpus", "user_input", "category"]]
for name, df in dfs.items():
    pd.testing.assert_frame_equal(
        df[["corpus", "user_input", "category"]].reset_index(drop=True),
        reference.reset_index(drop=True),
        obj=f"{name} question alignment",
    )


def rank_biserial(diffs):
    """Signed matched-pairs rank-biserial: (W+ - W-) / (W+ + W-). See
    ragas_significance.py for why this rather than Z/sqrt(N)."""
    nz = diffs[diffs != 0]
    if nz.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nz))
    return (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    n, order, adjusted, run = len(pvalues), np.argsort(pvalues), np.empty(len(pvalues)), 0.0
    for rank, idx in enumerate(order):
        run = max(run, (n - rank) * pvalues[idx])
        adjusted[idx] = min(run, 1.0)
    return adjusted


rows = []
for outcome in OUTCOMES:
    for cat in CATEGORIES:
        for a, b in COMPARISONS:
            mask = dfs[a]["category"] == cat
            x = dfs[a].loc[mask, outcome].to_numpy(float)
            y = dfs[b].loc[mask, outcome].to_numpy(float)
            ok = ~(np.isnan(x) | np.isnan(y))
            x, y = x[ok], y[ok]
            diffs = x - y

            row = dict(outcome=outcome, category=cat, comparison=f"{a} vs {b}",
                       n=len(x), mean_a=x.mean(), mean_b=y.mean(), diff=x.mean() - y.mean())

            if outcome == "retrieval_failed":
                # Paired binary -> exact McNemar on the discordant pairs.
                a_only, b_only = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
                n_disc = a_only + b_only
                row.update(n_informative=n_disc,
                           p_value=binomtest(a_only, n_disc, 0.5).pvalue if n_disc else 1.0,
                           # Signed so + always favours llm_retriever, matching the
                           # other outcomes (fewer failures = better).
                           effect=(b_only - a_only) / n_disc if n_disc else 0.0)
            else:
                row.update(n_informative=int((diffs != 0).sum()), effect=rank_biserial(diffs),
                           p_value=1.0 if np.all(diffs == 0)
                           else wilcoxon(x, y, zero_method="wilcox", correction=False, method="auto")[1])
            rows.append(row)

results = pd.DataFrame(rows)
# Primary: Holm within each outcome family (8 tests each).
results["p_holm"] = results.groupby("outcome")["p_value"].transform(lambda s: holm(s.values))
results[f"sig_{ALPHA}"] = results["p_holm"] < ALPHA
# Sensitivity: Holm across every test run here, the conservative reading.
results["p_holm_global"] = holm(results["p_value"].values)
results[f"sig_global_{ALPHA}"] = results["p_holm_global"] < ALPHA

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
LABELS = {"context_precision": "context_precision (primary — the pooled analysis's robust win)",
          "answer_quality": "answer_quality (composite 0-1, derived)",
          "retrieval_failed": "retrieval failure rate (binary, exact McNemar)"}
for outcome in OUTCOMES:
    sub = results[results["outcome"] == outcome].copy()
    sub["category"] = pd.Categorical(sub["category"], CATEGORIES, ordered=True)
    print(f"\n=== {LABELS.get(outcome, outcome)} — Holm within {len(sub)} tests ===")
    print(sub.sort_values(["comparison", "category"])[
        ["category", "comparison", "n", "n_informative", "mean_a", "mean_b", "diff",
         "p_value", "p_holm", f"sig_{ALPHA}", "p_holm_global", f"sig_global_{ALPHA}",
         "effect"]].round(4).to_string(index=False))

hits = results[results[f"sig_{ALPHA}"]].sort_values("p_value")
print(f"\n=== SUMMARY: {len(hits)} of {len(results)} tests significant within-family; "
      f"{int(results[f'sig_global_{ALPHA}'].sum())} survive global Holm ===")
print(hits[["outcome", "category", "comparison", "diff", "p_value", "p_holm",
            "p_holm_global", f"sig_global_{ALPHA}", "effect"]].round(4).to_string(index=False))

results.to_csv(RESULTS_CSV, index=False)
print(f"\nSaved -> {RESULTS_CSV}")
